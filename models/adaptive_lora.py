"""
Adaptive LoRA Rank 模块

创新点：不同作物使用不同 rank 的 LoRA
- 玉米（视觉变化明显）：rank=4（少参数）
- 小麦（与玉米有相似阶段）：rank=8（中等参数）
- 棉花（蕾/花/铃形态接近）：rank=16（多参数）

优势：
1. 按复杂度分配参数，比均匀 rank 更合理
2. 棉花获得最多参数，符合其最难分类的事实
3. 总参数量可控，不会显著增加模型大小
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AdaptiveLoRAExpert(nn.Module):
    """
    单个 LoRA 专家（特定 rank）

    LoRA 分解：W + α/r * A @ B
    - A: [in, rank] 随机初始化
    - B: [rank, out] 零初始化（确保初始 LoRA 输出为 0）
    """

    def __init__(self, in_features, out_features, rank=8, alpha=16,
                 dropout=0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        # 初始化
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        """
        Args:
            x: [B, in_features]

        Returns:
            out: [B, out_features] LoRA 增量
        """
        return self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class AdaptiveLoRALinear(nn.Module):
    """
    自适应 LoRA 线性层

    根据作物 ID 选择对应 rank 的 LoRA 专家：
    - crop_id=0 (Corn): rank=4
    - crop_id=1 (Wheat): rank=8
    - crop_id=2 (Cotton): rank=16

    原始 Linear 层保持冻结，只训练 LoRA 参数

    创新对比：
    - 标准 LoRA：所有输入用相同 rank
    - MoE-LoRA：多个相同 rank 的专家，路由器选择
    - 本方案：不同作物用不同 rank，按复杂度分配
    """

    def __init__(self, original_linear, crop_ranks, alpha_multiplier=2.0,
                 dropout=0.0):
        """
        Args:
            original_linear: 原始冻结的 nn.Linear 层
            crop_ranks: dict {crop_id: rank}
                例如 {0: 4, 1: 8, 2: 16}
            alpha_multiplier: alpha = rank * multiplier
            dropout: LoRA dropout
        """
        super().__init__()
        self.original = original_linear
        self.crop_ranks = crop_ranks

        # 冻结原始参数
        for param in self.original.parameters():
            param.requires_grad = False

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # 为每种作物创建不同 rank 的 LoRA 专家
        self.experts = nn.ModuleDict()
        for crop_id, rank in crop_ranks.items():
            alpha = rank * alpha_multiplier
            self.experts[str(crop_id)] = AdaptiveLoRAExpert(
                in_features, out_features,
                rank=rank, alpha=alpha, dropout=dropout
            )

    def forward(self, x, crop_ids=None):
        """
        Args:
            x: [B, in_features] 或 [B, seq_len, in_features]
            crop_ids: [B] 作物 ID（0=Corn, 1=Wheat, 2=Cotton）
                如果为 None，则使用所有专家的平均输出

        Returns:
            output: [B, out_features]
        """
        # 原始线性变换
        base_output = self.original(x)

        # 处理 3D 输入（ViT 的序列维度）
        orig_shape = base_output.shape
        if x.dim() == 3:
            x_flat = x.reshape(-1, x.size(-1))
            base_flat = base_output.reshape(-1, base_output.size(-1))
        else:
            x_flat = x
            base_flat = base_output

        if crop_ids is not None:
            # 按作物 ID 分组，走对应 rank 的 LoRA
            lora_output = torch.zeros_like(base_flat)

            for crop_id_str, expert in self.experts.items():
                crop_id = int(crop_id_str)
                mask = (crop_ids == crop_id)

                if mask.dim() > 1:
                    mask = mask.any(dim=-1)  # 处理序列维度

                if mask.any():
                    x_crop = x_flat[mask]
                    lora_output[mask] = expert(x_crop)

            # 恢复形状
            if x.dim() == 3:
                lora_output = lora_output.reshape(orig_shape)

            return base_output + lora_output
        else:
            # 无 crop_ids 时：所有专家平均
            lora_sum = torch.zeros_like(base_flat)
            for expert in self.experts.values():
                lora_sum = lora_sum + expert(x_flat)
            lora_avg = lora_sum / len(self.experts)

            if x.dim() == 3:
                lora_avg = lora_avg.reshape(orig_shape)

            return base_output + lora_avg

    def get_param_stats(self):
        """获取各作物 LoRA 的参数统计"""
        stats = {}
        total_lora_params = 0
        for crop_id_str, expert in self.experts.items():
            crop_id = int(crop_id_str)
            rank = expert.rank
            params = sum(p.numel() for p in expert.parameters())
            stats[crop_id] = {
                'rank': rank,
                'params': params,
                'alpha': expert.alpha,
            }
            total_lora_params += params
        stats['total_lora_params'] = total_lora_params
        stats['original_params'] = sum(
            p.numel() for p in self.original.parameters()
        )
        return stats


def apply_adaptive_lora(model, crop_ranks=None, alpha_multiplier=2.0,
                        target_modules=None, dropout=0.0):
    """
    将模型中的 Linear 层替换为 AdaptiveLoRALinear

    Args:
        model: CLIP 模型
        crop_ranks: dict {crop_id: rank}
            默认：{0: 4, 1: 8, 2: 16}（Corn, Wheat, Cotton）
        alpha_multiplier: alpha = rank * multiplier
        target_modules: 要替换的模块名称列表
            默认：["q_proj"]（只替换 q_proj，减少计算量）
        dropout: LoRA dropout

    Returns:
        model: 替换后的模型
        stats: 参数统计信息
    """
    if crop_ranks is None:
        crop_ranks = {0: 4, 1: 8, 2: 16}

    if target_modules is None:
        target_modules = ["q_proj"]  # 只替换 q_proj，与 MoE-LoRA 一致

    replaced_count = 0
    total_lora_params = 0

    for name, module in model.named_modules():
        for target_name in target_modules:
            if hasattr(module, target_name):
                original_linear = getattr(module, target_name)

                if isinstance(original_linear, nn.Linear):
                    # 替换为 AdaptiveLoRALinear
                    adaptive_lora = AdaptiveLoRALinear(
                        original_linear, crop_ranks,
                        alpha_multiplier=alpha_multiplier,
                        dropout=dropout
                    )
                    setattr(module, target_name, adaptive_lora)

                    stats = adaptive_lora.get_param_stats()
                    total_lora_params += stats['total_lora_params']
                    replaced_count += 1

    print(f"[AdaptiveLoRA] 替换了 {replaced_count} 个 Linear 层")
    print(f"[AdaptiveLoRA] LoRA 参数量: {total_lora_params:,}")
    print(f"[AdaptiveLoRA] Rank 配置: {crop_ranks}")

    return model, {
        'replaced_count': replaced_count,
        'total_lora_params': total_lora_params,
        'crop_ranks': crop_ranks,
    }


class AdaptiveLoRAModel(nn.Module):
    """
    包装模型：集成 Adaptive LoRA 的完整分类器

    用于统一训练脚本，将 AdaptiveLoRA 与分类头组合
    """

    def __init__(self, clip_model, feat_dim, num_classes=15,
                 crop_ranks=None, hidden_dim=256, alpha_multiplier=2.0):
        super().__init__()
        self.clip_model = clip_model

        if crop_ranks is None:
            crop_ranks = {0: 4, 1: 8, 2: 16}

        # 应用 Adaptive LoRA
        self.clip_model, self.lora_stats = apply_adaptive_lora(
            clip_model, crop_ranks, alpha_multiplier
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, pixel_values, crop_ids=None):
        """
        Args:
            pixel_values: [B, 3, H, W] 图像输入
            crop_ids: [B] 作物 ID（用于选择 LoRA rank）

        Returns:
            logits: [B, num_classes]
        """
        # CLIP 视觉编码
        vision_outputs = self.clip_model.vision_model(pixel_values)
        features = vision_outputs.pooler_output  # [B, feat_dim]

        # 分类
        logits = self.classifier(features)

        return logits
