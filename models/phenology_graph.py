"""
生育期关系建模模块 (Phenology-aware Relation Modeling)

创新点：将生长阶段视为连续变化序列，而非独立类别
- 高斯邻接矩阵：相邻阶段关系强，远距离关系弱
- GraphConv 阶段分类头：用图卷积增强阶段特征
- PhenologyAwareLoss：结合图拉普拉斯正则化的损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def build_gaussian_adjacency(num_stages, sigma=1.5):
    """
    构建高斯邻接矩阵

    相邻阶段关系强，远距离关系弱：
    adjacency[i][j] = exp(-|i-j|^2 / (2*sigma^2))

    Args:
        num_stages: 阶段数量（如5）
        sigma: 高斯核宽度，越大越平滑

    Returns:
        adj: [num_stages, num_stages] 归一化邻接矩阵
    """
    positions = torch.arange(num_stages, dtype=torch.float32)
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # [S, S]
    adj = torch.exp(-diff ** 2 / (2 * sigma ** 2))

    # 添加自连接（对角线已经是1.0）
    # 归一化：D^{-1/2} A D^{-1/2}
    degree = adj.sum(dim=1)  # [S]
    deg_inv_sqrt = torch.pow(degree, -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    D = torch.diag(deg_inv_sqrt)
    adj_normalized = D @ adj @ D

    return adj_normalized


def build_ordinal_adjacency(num_stages):
    """
    构建顺序邻接矩阵（只连接相邻阶段）

    适合显式建模阶段的顺序关系：
    - 阶段 i 只与 i-1 和 i+1 相连
    - 边权 = 1.0（相邻）或 0.0（不相邻）

    Args:
        num_stages: 阶段数量

    Returns:
        adj: [num_stages, num_stages] 归一化邻接矩阵
    """
    adj = torch.zeros(num_stages, num_stages)
    for i in range(num_stages):
        adj[i, i] = 1.0  # 自连接
        if i > 0:
            adj[i, i - 1] = 1.0
        if i < num_stages - 1:
            adj[i, i + 1] = 1.0

    # 归一化
    degree = adj.sum(dim=1)
    deg_inv_sqrt = torch.pow(degree, -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    D = torch.diag(deg_inv_sqrt)
    adj_normalized = D @ adj @ D

    return adj_normalized


class GraphConvLayer(nn.Module):
    """
    图卷积层：H' = σ(A · H · W + b)

    用于在生长阶段之间传递信息：
    - 每个阶段的特征会聚合其相邻阶段的信息
    - 使模型理解阶段之间的连续关系
    """

    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        """
        Args:
            x: [B, S, D] 或 [S, D] 阶段特征
            adj: [S, S] 邻接矩阵

        Returns:
            output: [B, S, out_dim] 图卷积后的特征
        """
        # adj @ x: 聚合邻居信息
        # result @ W: 线性变换
        support = torch.matmul(x, self.weight)  # [B, S, out_dim]
        output = torch.matmul(adj, support)      # [B, S, out_dim]

        if self.bias is not None:
            output = output + self.bias

        return F.relu(output)


class GraphConvStageHead(nn.Module):
    """
    图卷积增强的阶段分类头

    结构：
    1. 特征投影：CLIP features → 低维阶段特征
    2. 图卷积：在阶段之间传递信息（2层）
    3. 分类：每个阶段输出一个 logit

    创新点：传统分类头把每个阶段看作独立类别，
    这里通过图卷积让模型知道"相邻阶段更相似"
    """

    def __init__(self, feat_dim, hidden_dim=256, num_stages=5,
                 adjacency=None, num_graph_layers=2):
        super().__init__()
        self.num_stages = num_stages

        # 特征投影
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # 图卷积层（共享邻接矩阵）
        self.graph_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim)
            for _ in range(num_graph_layers)
        ])

        # 分类头
        self.classifier = nn.Linear(hidden_dim, 1)

        # 邻接矩阵（注册为 buffer，不参与梯度更新）
        if adjacency is None:
            adjacency = build_gaussian_adjacency(num_stages, sigma=1.5)
        self.register_buffer('adjacency', adjacency)

    def forward(self, features):
        """
        Args:
            features: [B, D] CLIP 图像特征

        Returns:
            logits: [B, num_stages] 阶段分类 logits
        """
        B = features.size(0)

        # 1. 特征投影 → [B, hidden_dim]
        h = self.projection(features)

        # 2. 扩展为 [B, S, hidden_dim]（每个阶段共享同一特征）
        h = h.unsqueeze(1).expand(-1, self.num_stages, -1)

        # 3. 图卷积：在阶段之间传递信息
        for graph_layer in self.graph_layers:
            h = graph_layer(h, self.adjacency)  # [B, S, hidden_dim]

        # 4. 分类：每个阶段一个 logit
        logits = self.classifier(h).squeeze(-1)  # [B, S]

        return logits


class PhenologyAwareLoss(nn.Module):
    """
    生育期感知损失函数

    组合三个损失：
    1. L_CE: 标准交叉熵损失
    2. L_EMD: Earth Movers Distance（已有实现，这里简化版）
    3. L_graph: 图拉普拉斯正则化（鼓励相邻阶段有相似概率）

    L = L_CE + α * L_EMD + β * L_graph

    创新点：通过图正则化，让模型知道：
    - 预测为 Jointing（相邻）比预测为 Maturity（远距离）的惩罚更小
    """

    def __init__(self, num_stages=5, sigma=1.5, alpha=0.5, beta=0.1):
        super().__init__()
        self.num_stages = num_stages
        self.alpha = alpha
        self.beta = beta

        # 构建邻接矩阵用于图正则化
        adj = build_gaussian_adjacency(num_stages, sigma)
        # 拉普拉斯矩阵 L = I - D^{-1/2} A D^{-1/2}
        self.register_buffer('laplacian',
                             torch.eye(num_stages) - adj)

    def forward(self, logits, labels):
        """
        Args:
            logits: [B, S] 阶段分类 logits
            labels: [B] 真实阶段标签（0-based）

        Returns:
            loss: 标量损失值
        """
        probs = F.softmax(logits, dim=-1)  # [B, S]

        # 1. 交叉熵损失
        loss_ce = F.cross_entropy(logits, labels)

        # 2. EMD 损失（简化实现）
        # 构建目标累积分布
        target_onehot = F.one_hot(labels, self.num_stages).float()
        target_cdf = torch.cumsum(target_onehot, dim=-1)
        pred_cdf = torch.cumsum(probs, dim=-1)
        loss_emd = torch.mean(torch.abs(pred_cdf - target_cdf))

        # 3. 图拉普拉斯正则化
        # L_graph = sum_b (p_b^T · L · p_b)
        # 鼓励相邻阶段有相似的概率分布
        Lp = torch.matmul(probs, self.laplacian)  # [B, S]
        loss_graph = torch.mean(torch.sum(Lp * probs, dim=-1))

        # 组合损失
        total_loss = loss_ce + self.alpha * loss_emd + self.beta * loss_graph

        return total_loss


class PhenologyAwareClassifier(nn.Module):
    """
    完整的生育期感知分类器

    与 ConfidenceRouter 结合使用：
    - 每个作物分支使用 GraphConvStageHead
    - 损失函数使用 PhenologyAwareLoss
    """

    def __init__(self, feat_dim, num_classes=15, hidden_dim=256,
                 sigma=1.5, num_graph_layers=2):
        super().__init__()
        self.num_classes = num_classes

        # 每种作物独立的图卷积阶段分类头
        self.corn_head = GraphConvStageHead(
            feat_dim, hidden_dim, num_stages=5,
            adjacency=build_gaussian_adjacency(5, sigma),
            num_graph_layers=num_graph_layers
        )
        self.wheat_head = GraphConvStageHead(
            feat_dim, hidden_dim, num_stages=5,
            adjacency=build_gaussian_adjacency(5, sigma),
            num_graph_layers=num_graph_layers
        )
        self.cotton_head = GraphConvStageHead(
            feat_dim, hidden_dim, num_stages=5,
            adjacency=build_gaussian_adjacency(5, sigma),
            num_graph_layers=num_graph_layers
        )

        # 作物路由器
        self.crop_router = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3)  # 3种作物
        )

        # 注册作物到阶段的映射
        # corn: 0-4, wheat: 5-9, cotton: 10-14
        self.register_buffer('crop_stage_offsets',
                             torch.tensor([0, 5, 10]))

    def forward(self, features, threshold=0.7):
        """
        Args:
            features: [B, D] CLIP 图像特征
            threshold: 置信度阈值

        Returns:
            logits: [B, 15] 全局分类 logits
            crop_logits: [B, 3] 作物分类 logits
        """
        B = features.size(0)
        device = features.device

        # 1. 作物路由
        crop_logits = self.crop_router(features)  # [B, 3]
        crop_probs = F.softmax(crop_logits, dim=-1)  # [B, 3]

        # 2. 各作物分支的阶段 logits
        corn_stage_logits = self.corn_head(features)    # [B, 5]
        wheat_stage_logits = self.wheat_head(features)  # [B, 5]
        cotton_stage_logits = self.cotton_head(features)  # [B, 5]

        # 3. 拼接为全局 logits
        all_stage_logits = torch.cat([
            corn_stage_logits,
            wheat_stage_logits,
            cotton_stage_logits
        ], dim=-1)  # [B, 15]

        # 4. 向量化置信度路由融合 (纯 Tensor 操作，全面兼容 GPU 显卡)
        stage_logits_3d = all_stage_logits.view(B, 3, 5)

        max_prob, top_crop = crop_probs.max(dim=-1)  # [B], [B]
        hard_mask = (max_prob > threshold).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]

        top_crop_one_hot = F.one_hot(top_crop, num_classes=3).float()  # [B, 3]
        hard_logits_3d = stage_logits_3d * top_crop_one_hot.unsqueeze(-1)  # [B, 3, 5]
        soft_logits_3d = stage_logits_3d * crop_probs.unsqueeze(-1)  # [B, 3, 5]

        fused_logits_3d = torch.where(hard_mask, hard_logits_3d, soft_logits_3d)
        fused_logits = fused_logits_3d.view(B, self.num_classes)  # [B, 15]

        return fused_logits, crop_logits
