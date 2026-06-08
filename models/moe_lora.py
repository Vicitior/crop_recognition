# -*- coding: utf-8 -*-
"""
Mixture-of-LoRA-Experts (MoE-LoRA) for CLIP Fine-tuning
创新点 2：面向多作物扩展的"混合专家 LoRA"

核心思想：
  - 每个作物单独训练一个 LoRA → 无法共享植物生长的通用视觉特征
  - 所有作物混在一起训一个 LoRA → 不同作物相同阶段的形态差异造成特征混淆
  - MoE-LoRA：并联多个轻量 LoRA 专家 + 门控网络动态分配权重

架构：
  CLIP ViT Block (冻结)
      ↓
  Attention (q_proj / v_proj / k_proj / out_proj)
      ↓
  ┌─────────┬─────────┬─────────┬─────────┐
  │ LoRA_0  │ LoRA_1  │ LoRA_2  │ LoRA_3  │   ← N 个并行 LoRA 专家
  │(shared) │(shared) │(cotton) │(corn)   │
  └────┬────┴────┬────┴────┬────┴────┬────┘
       │         │         │         │
       └──── Router(x) → softmax weights ──→ weighted sum
      ↓
  Output = Original_Linear(x) + weighted_sum(LoRA_i(x)) * scaling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRAExpert(nn.Module):
    """单个 LoRA 专家适配器"""

    def __init__(self, in_dim, out_dim, rank=8, alpha=16, dropout=0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Linear(in_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 初始化：A 用 Kaiming，B 用零 → 初始时 LoRA 输出为零
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class Router(nn.Module):
    """
    门控网络：根据输入特征决定各专家的权重。

    支持两种路由模式：
    1. 全局路由：所有输入共享同一套权重
    2. 作物感知路由：接收作物 ID 作为额外输入，为不同作物学习不同路由策略
    """

    def __init__(self, input_dim, num_experts, num_crops=1, top_k=None):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k or num_experts

        # 作物感知嵌入
        self.crop_embedding = nn.Embedding(num_crops, num_experts) if num_crops > 1 else None

        # 路由 MLP
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.SiLU(),
            nn.Linear(input_dim // 4, num_experts)
        )

    def forward(self, x, crop_id=None):
        """
        Args:
            x: (B, D) 输入特征
            crop_id: (B,) 作物类别 ID（可选）

        Returns:
            weights: (B, num_experts) softmax 权重
            load_balance_loss: 辅助损失，防止专家坍缩
        """
        logits = self.gate(x)  # (B, E)

        # 作物先验偏置
        if self.crop_embedding is not None and crop_id is not None:
            crop_bias = self.crop_embedding(crop_id)  # (B, E)
            logits = logits + crop_bias

        # Top-K 稀疏路由（可选）
        if self.top_k < self.num_experts:
            topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)
            mask = torch.full_like(logits, float('-inf'))
            mask.scatter_(1, topk_idx, topk_vals)
            logits = mask

        weights = F.softmax(logits, dim=-1)

        # 负载均衡损失：鼓励专家被均匀使用
        # f_i = 平均路由概率, P_i = 平均门控权重
        # loss = N * sum(f_i * P_i)
        f = (weights > 0).float().mean(dim=0)  # (E,)
        P = weights.mean(dim=0)                 # (E,)
        load_balance_loss = self.num_experts * (f * P).sum()

        return weights, load_balance_loss


class MoELoRALinear(nn.Module):
    """
    MoE-LoRA 线性层：替代单个 LoRA，用 N 个并行专家 + 门控路由。

    原始线性层的输出保持不变，LoRA 部分由 MoE 加权组合。

    Args:
        original_linear: 被替换的 nn.Linear 层
        num_experts: 专家总数
        num_shared: 共享专家数（剩余为作物特异性专家）
        rank: 每个专家的 LoRA rank
        alpha: LoRA scaling
        num_crops: 作物种类数（用于作物感知路由）
        top_k: Top-K 稀疏路由（None = 使用全部专家）
    """

    def __init__(self, original_linear, num_experts=4, num_shared=2,
                 rank=8, alpha=16, num_crops=1, top_k=None, dropout=0.0):
        super().__init__()
        self.original = original_linear
        in_dim = original_linear.in_features
        out_dim = original_linear.out_features

        # 冻结原始权重
        for p in self.original.parameters():
            p.requires_grad = False

        # 创建 N 个并行 LoRA 专家
        self.experts = nn.ModuleList([
            LoRAExpert(in_dim, out_dim, rank, alpha, dropout)
            for _ in range(num_experts)
        ])

        # 门控路由器
        self.router = Router(in_dim, num_experts, num_crops, top_k)

        # 记录专家角色
        self.num_shared = num_shared
        self.num_experts = num_experts

        # 辅助损失权重（训练时用）
        self.aux_loss_weight = 0.01

    def forward(self, x, crop_id=None):
        """
        支持 2D (B, D) 和 3D (B, seq_len, D) 输入。
        CLIP 注意力层传入的是 3D，需要 reshape 处理。

        辅助损失存储在 self.last_aux_loss 中，通过 get_aux_loss() 获取。
        """
        original_shape = x.shape
        # 统一 reshape 为 2D 处理
        if x.dim() == 3:
            B, S, D = x.shape
            x_flat = x.reshape(B * S, D)
        else:
            x_flat = x
            B, S = x.shape[0], None

        # 原始输出（冻结）
        original_out = self.original(x_flat)

        # 各专家的 LoRA 输出 → (B*S, E, out_dim)
        expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=1)

        # 路由权重 → (B*S, E)
        # 路由用每个 token 的特征独立决定（或用 CLS token 的特征）
        weights, aux_loss = self.router(x_flat, crop_id)

        # 加权组合 → (B*S, out_dim)
        moe_out = (weights.unsqueeze(-1) * expert_outputs).sum(dim=1)

        # 存储辅助损失
        self.last_aux_loss = aux_loss * self.aux_loss_weight

        result = original_out + moe_out

        # 恢复原始形状
        if S is not None:
            result = result.reshape(B, S, -1)

        return result

    def get_aux_loss(self):
        """获取最近一次 forward 计算的辅助损失"""
        return getattr(self, 'last_aux_loss', torch.tensor(0.0))


def apply_moe_lora(clip_model, num_experts=4, num_shared=2, rank=8,
                   alpha=16, num_crops=1, top_k=None, dropout=0.0):
    """
    给 CLIP 模型的所有注意力投影层添加 MoE-LoRA 适配器。

    Args:
        clip_model: HuggingFace CLIPModel
        num_experts: 每层的专家数
        num_shared: 共享专家数
        rank: LoRA rank
        alpha: LoRA alpha
        num_crops: 作物种类数
        top_k: Top-K 路由（None = 全部）
        dropout: LoRA dropout

    Returns:
        clip_model: 添加了 MoE-LoRA 的模型
        moe_layers: 所有 MoE-LoRA 层的列表（用于提取辅助损失）
    """
    target_modules = ["q_proj"]  # 只对q_proj加MoE-LoRA，减少一半计算量

    moe_layers = []
    count = 0

    for name, module in clip_model.named_modules():
        if isinstance(module, nn.Linear):
            if any(target in name for target in target_modules):
                # 找到父模块
                parts = name.split(".")
                parent = clip_model
                for p in parts[:-1]:
                    parent = getattr(parent, p)

                moe_layer = MoELoRALinear(
                    module,
                    num_experts=num_experts,
                    num_shared=num_shared,
                    rank=rank,
                    alpha=alpha,
                    num_crops=num_crops,
                    top_k=top_k,
                    dropout=dropout
                )
                setattr(parent, parts[-1], moe_layer)
                moe_layers.append(moe_layer)
                count += 1

    print(f"Applied MoE-LoRA to {count} layers "
          f"({num_experts} experts, {num_shared} shared, rank={rank})")
    return clip_model, moe_layers


def get_moe_aux_loss(model):
    """遍历模型，汇总所有 MoELoRALinear 层的辅助负载均衡损失"""
    total = torch.tensor(0.0)
    for module in model.modules():
        if isinstance(module, MoELoRALinear):
            aux = module.get_aux_loss()
            if aux.device != total.device:
                total = total.to(aux.device)
            total = total + aux
    return total
