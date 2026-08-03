"""
置信度引导路由模块 (Confidence-aware Routing)

创新点：第一层作物预测不确定时，同时激活多个作物分支
- 传统硬路由：只选概率最高的作物
- 置信度路由：概率接近时同时走多个分支，最后加权融合

优势：对容易混淆的作物（如玉米拔节期 vs 小麦拔节期）更稳定
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CropRouter(nn.Module):
    """
    作物路由器

    CLIP特征 → 3种作物的概率分布
    用于决定激活哪些作物分支
    """

    def __init__(self, feat_dim, hidden_dim=256, num_crops=3):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_crops)
        )

    def forward(self, features):
        """
        Args:
            features: [B, D] CLIP 图像特征

        Returns:
            crop_logits: [B, num_crops] 作物分类 logits
        """
        return self.router(features)


class StageBranch(nn.Module):
    """
    单个作物的阶段分类器

    每种作物（Corn/Wheat/Cotton）有独立的阶段分类器，
    只负责该作物的5个生长阶段
    """

    def __init__(self, feat_dim, num_stages=5, hidden_dim=256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_stages)
        )

    def forward(self, features):
        """
        Args:
            features: [B, D] CLIP 图像特征

        Returns:
            stage_logits: [B, num_stages] 阶段分类 logits
        """
        return self.classifier(features)


class ConfidenceRouterClassifier(nn.Module):
    """
    置信度引导路由分类器

    核心逻辑：
    1. 第一层：CropRouter 预测作物概率
    2. 判断置信度：
       - 高置信度（max_prob > threshold）：只走 top-1 分支
       - 低置信度（max_prob <= threshold）：同时走多个分支，加权融合
    3. 输出：15类全局 logits

    创新对比：
    - 传统：硬路由，只选一个作物
    - 本方案：软路由，不确定时多分支融合

    农业意义：
    - 玉米拔节期和小麦拔节期形态相似
    - 单一分支可能误判，多分支融合更稳定
    """

    def __init__(self, feat_dim, num_classes=15, hidden_dim=256,
                 num_crops=3, stages_per_crop=5):
        super().__init__()
        self.num_classes = num_classes
        self.num_crops = num_crops
        self.stages_per_crop = stages_per_crop

        # 作物路由器
        self.crop_router = CropRouter(feat_dim, hidden_dim, num_crops)

        # 每种作物独立的阶段分类器
        self.stage_branches = nn.ModuleList([
            StageBranch(feat_dim, stages_per_crop, hidden_dim)
            for _ in range(num_crops)
        ])

        # 可选：特征增强（SE注意力）
        self.feature_enhance = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid()
        )

    def forward(self, features, threshold=0.7, return_routing_info=False):
        """
        Args:
            features: [B, D] CLIP 图像特征
            threshold: 置信度阈值
                - 0.7: 硬路由（只走 top-1）
                - 0.5: 软路由（top-2 融合）
                - 0.0: 全融合（所有分支加权）
            return_routing_info: 是否返回路由信息（用于分析）

        Returns:
            logits: [B, 15] 全局分类 logits
            crop_logits: [B, 3] 作物分类 logits
            routing_info: dict（可选）路由决策详情
        """
        B = features.size(0)
        device = features.device

        # 0. 特征增强
        gate = self.feature_enhance(features)
        features = features * gate

        # 1. 作物路由
        crop_logits = self.crop_router(features)  # [B, 3]
        crop_probs = F.softmax(crop_logits, dim=-1)  # [B, 3]

        # 2. 各作物分支的阶段 logits
        stage_logits_list = [
            branch(features) for branch in self.stage_branches
        ]  # 每个 [B, 5]

        # 3. 拼接为全局阶段 logits
        all_stage_logits = torch.cat(stage_logits_list, dim=-1)  # [B, 15]

        # 4. 向量化置信度路由融合 (纯 Tensor 操作，全面兼容 GPU 显卡)
        # 将 [B, 15] 重构为 [B, 3, 5]（3 种作物，每种 5 个阶段）
        stage_logits_3d = all_stage_logits.view(B, self.num_crops, self.stages_per_crop)

        max_prob, top_crop = crop_probs.max(dim=-1)  # [B], [B]
        hard_mask = (max_prob > threshold).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]

        # 硬路由：仅保留 top-1 作物分支的 logits
        top_crop_one_hot = F.one_hot(top_crop, num_classes=self.num_crops).float()  # [B, 3]
        hard_logits_3d = stage_logits_3d * top_crop_one_hot.unsqueeze(-1)  # [B, 3, 5]

        # 软路由：按 crop_probs 概率加权融合
        soft_logits_3d = stage_logits_3d * crop_probs.unsqueeze(-1)  # [B, 3, 5]

        # 按置信度掩码选择硬/软路由
        fused_logits_3d = torch.where(hard_mask, hard_logits_3d, soft_logits_3d)
        fused_logits = fused_logits_3d.view(B, self.num_classes)  # [B, 15]

        if return_routing_info:
            hard_count = (max_prob > threshold).sum().item()
            soft_count = B - hard_count
            routing_info = {
                'hard_count': hard_count,
                'soft_count': soft_count,
                'hard_ratio': hard_count / B,
                'crop_probs': crop_probs.detach().cpu(),
                'max_prob': max_prob.detach().cpu(),
                'threshold': threshold,
            }
            return fused_logits, crop_logits, routing_info

        return fused_logits, crop_logits


class ConfidenceRouterLoss(nn.Module):
    """
    置信度路由损失函数

    L = L_stage + λ * L_crop + γ * L_entropy

    其中：
    - L_stage: 阶段分类损失（15类）
    - L_crop: 作物分类损失（3类）
    - L_entropy: 熵正则化（鼓励路由决策更果断）

    熵正则化的意义：
    - 如果路由概率太均匀（熵高），说明模型不确定
    - 惩罚高熵，鼓励模型做出更明确的路由决策
    """

    def __init__(self, lambda_crop=0.3, gamma_entropy=0.05,
                 label_smoothing=0.1):
        super().__init__()
        self.lambda_crop = lambda_crop
        self.gamma_entropy = gamma_entropy
        self.stage_loss_fn = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        )
        self.crop_loss_fn = nn.CrossEntropyLoss()

    def forward(self, stage_logits, crop_logits, stage_labels, crop_labels):
        """
        Args:
            stage_logits: [B, 15] 阶段分类 logits
            crop_logits: [B, 3] 作物分类 logits
            stage_labels: [B] 阶段标签（0-14）
            crop_labels: [B] 作物标签（0-2）

        Returns:
            loss: 标量损失值
            loss_dict: 各项损失的详细信息
        """
        # 阶段损失
        loss_stage = self.stage_loss_fn(stage_logits, stage_labels)

        # 作物损失
        loss_crop = self.crop_loss_fn(crop_logits, crop_labels)

        # 熵正则化（鼓励路由决策更果断）
        crop_probs = F.softmax(crop_logits, dim=-1)
        entropy = -torch.sum(crop_probs * torch.log(crop_probs + 1e-8),
                             dim=-1)
        loss_entropy = entropy.mean()

        # 总损失
        total_loss = (loss_stage
                      + self.lambda_crop * loss_crop
                      + self.gamma_entropy * loss_entropy)

        loss_dict = {
            'loss_stage': loss_stage.item(),
            'loss_crop': loss_crop.item(),
            'loss_entropy': loss_entropy.item(),
            'loss_total': total_loss.item(),
        }

        return total_loss, loss_dict
