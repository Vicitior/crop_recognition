# -*- coding: utf-8 -*-
"""
创新点 1：知识引导 Prompt 微调 (Knowledge-guided Prompt Tuning, KGPT)

核心思想：
  将农学物理机制（积温、生长周期斜率、叶面积指数）编码为可学习向量，
  与 CLIP 文本编码器深度融合，突破纯数据驱动的局限。

物理先验知识：
  1. 有效积温 (Growing Degree Days, GDD)
     - 作物发育由 ≥10°C 的累积温度驱动
     - 不同生育期对应不同的积温区间
  2. 生长周期斜率 (Growth Rate)
     - 相邻阶段间的发育速率变化
     - 反映作物生长的动态特征
  3. 叶面积指数 (LAI) 变化曲线
     - LAI 在不同生育期呈规律性变化
     - 出苗期→峰值→成熟期下降
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 农学物理先验知识库
# ============================================================

# 有效积温 (GDD, base=10°C) 区间 — 单位：°C·day
# 来源：中国主要农作物生育期积温指标
GDD_RANGES = {
    "corn": {
        "seedling":   (0, 200),
        "jointing":   (200, 600),
        "tasseling":  (600, 1000),
        "filling":    (1000, 1600),
        "maturity":   (1600, 2200),
    },
    "wheat": {
        "seedling":   (0, 300),
        "tillering":  (300, 800),
        "jointing":   (800, 1200),
        "heading":    (1200, 1600),
        "maturity":   (1600, 2200),
    },
    "cotton": {
        "seedling":   (0, 400),
        "squaring":   (400, 900),
        "flowering":  (900, 1500),
        "boll_setting": (1500, 2200),
        "boll_opening": (2200, 3000),
    },
}

# 生长周期斜率 — 相邻阶段间的发育速率（归一化）
# 表征从一个阶段到下一个阶段的"过渡急缓程度"
GROWTH_SLOPES = {
    "corn": {
        "seedling":   0.2,   # 慢 — 刚出苗
        "jointing":   0.8,   # 快 — 拔节期生长迅速
        "tasseling":  0.6,   # 中 — 抽穗开花
        "filling":    0.5,   # 中 — 灌浆
        "maturity":   0.3,   # 慢 — 趋于成熟
    },
    "wheat": {
        "seedling":   0.2,
        "tillering":  0.5,
        "jointing":   0.8,
        "heading":    0.6,
        "maturity":   0.3,
    },
    "cotton": {
        "seedling":   0.2,
        "squaring":   0.5,
        "flowering":  0.7,
        "boll_setting": 0.6,
        "boll_opening": 0.3,
    },
}

# 叶面积指数 (LAI) — 无量纲
# 表征作物冠层叶面积与地面面积之比
LAI_VALUES = {
    "corn": {
        "seedling":   0.3,
        "jointing":   2.5,
        "tasseling":  5.0,   # LAI 峰值期
        "filling":    4.0,
        "maturity":   1.5,   # 叶片枯萎
    },
    "wheat": {
        "seedling":   0.2,
        "tillering":  2.0,
        "jointing":   4.5,
        "heading":    5.0,   # LAI 峰值
        "maturity":   1.0,
    },
    "cotton": {
        "seedling":   0.2,
        "squaring":   1.5,
        "flowering":  3.5,
        "boll_setting": 4.0,  # LAI 峰值
        "boll_opening": 2.0,
    },
}


def build_knowledge_vectors(class_names):
    """
    为每个类别构建物理先验向量 [GDD_norm, Slope, LAI_norm, GDD_range_norm]

    Args:
        class_names: 类别名称列表，如 ["corn_seedling", "corn_jointing", ...]

    Returns:
        Tensor [num_classes, 4] — 归一化的物理先验向量
    """
    vectors = []
    for cls_name in class_names:
        parts = cls_name.split("_")
        crop = parts[0]
        stage = "_".join(parts[1:])

        # GDD: 取区间中点，归一化到 [0, 1]
        gdd_low, gdd_high = GDD_RANGES.get(crop, {}).get(stage, (0, 1000))
        gdd_mid = (gdd_low + gdd_high) / 2
        gdd_max = max(v[1] for v in GDD_RANGES.get(crop, {}).values())
        gdd_norm = gdd_mid / gdd_max if gdd_max > 0 else 0

        # GDD 区间宽度（归一化）
        gdd_range_norm = (gdd_high - gdd_low) / gdd_max if gdd_max > 0 else 0

        # 生长斜率: 已在 [0, 1]
        slope = GROWTH_SLOPES.get(crop, {}).get(stage, 0.5)

        # LAI: 归一化到 [0, 1]
        lai = LAI_VALUES.get(crop, {}).get(stage, 1.0)
        lai_max = max(v for v in LAI_VALUES.get(crop, {}).values()) if LAI_VALUES.get(crop) else 1.0
        lai_norm = lai / lai_max if lai_max > 0 else 0

        vectors.append([gdd_norm, slope, lai_norm, gdd_range_norm])

    return torch.tensor(vectors, dtype=torch.float32)


# ============================================================
# 物理知识编码器
# ============================================================

class PhysicsKnowledgeEncoder(nn.Module):
    """
    将物理先验向量编码为与 CLIP 文本嵌入同维度的向量。

    输入: [B, 4] (GDD, Slope, LAI, GDD_range)
    输出: [B, D] (与 CLIP 文本嵌入同维度)
    """

    def __init__(self, input_dim=4, hidden_dim=256, output_dim=768):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        # 初始化：使输出接近零，不干扰原始 CLIP 特征
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)

    def forward(self, physics_vectors):
        """
        Args:
            physics_vectors: [B, 4] 物理先验向量
        Returns:
            [B, D] 知识嵌入
        """
        return self.encoder(physics_vectors)


# ============================================================
# 知识引导融合模块
# ============================================================

class KnowledgeGuidedFusion(nn.Module):
    """
    将物理知识嵌入与 CLIP 文本嵌入深度融合。

    三种融合策略：
    1. additive: 加法融合（最简单）
    2. gating: 门控融合（自适应权重）
    3. cross_attention: 交叉注意力融合（最灵活）
    """

    def __init__(self, embed_dim, fusion_type="gating"):
        super().__init__()
        self.fusion_type = fusion_type
        self.embed_dim = embed_dim

        if fusion_type == "gating":
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid()
            )
            # 初始化 gate 偏向保留原始文本特征
            nn.init.constant_(self.gate[0].bias, -2.0)

        elif fusion_type == "cross_attention":
            self.query_proj = nn.Linear(embed_dim, embed_dim)
            self.key_proj = nn.Linear(embed_dim, embed_dim)
            self.value_proj = nn.Linear(embed_dim, embed_dim)
            self.scale = math.sqrt(embed_dim)
            self.out_proj = nn.Linear(embed_dim, embed_dim)
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, text_embeds, knowledge_embeds):
        """
        Args:
            text_embeds: [B, D] CLIP 文本嵌入
            knowledge_embeds: [B, D] 物理知识嵌入
        Returns:
            [B, D] 融合后的嵌入
        """
        if self.fusion_type == "additive":
            return text_embeds + knowledge_embeds

        elif self.fusion_type == "gating":
            combined = torch.cat([text_embeds, knowledge_embeds], dim=-1)
            gate_weight = self.gate(combined)
            return gate_weight * text_embeds + (1 - gate_weight) * knowledge_embeds

        elif self.fusion_type == "cross_attention":
            Q = self.query_proj(text_embeds)       # [B, D]
            K = self.key_proj(knowledge_embeds)    # [B, D]
            V = self.value_proj(knowledge_embeds)  # [B, D]

            # 自注意力: 文本查询知识
            attn = (Q @ K.T) / self.scale          # [B, B]
            attn = F.softmax(attn, dim=-1)
            attended = attn @ V                     # [B, D]

            return text_embeds + self.out_proj(attended)

        else:
            return text_embeds


# ============================================================
# 知识引导 Prompt 学习器
# ============================================================

class KnowledgeGuidedPromptLearner(nn.Module):
    """
    完整的知识引导 Prompt 微调模块。

    流程：
      1. 物理先验向量 → PhysicsKnowledgeEncoder → 知识嵌入
      2. CLIP 文本编码器 → 文本嵌入
      3. KnowledgeGuidedFusion → 融合嵌入
      4. 用融合嵌入替代原始文本嵌入进行分类

    特点：
      - 只训练 PhysicsKnowledgeEncoder 和 Fusion 模块
      - CLIP 模型完全冻结，保留零样本能力
      - 物理先验作为硬约束，防止过拟合
    """

    def __init__(self, clip_model, class_names, fusion_type="gating", hidden_dim=256, processor=None):
        super().__init__()
        self.clip_model = clip_model
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.processor = processor

        # 获取 CLIP 嵌入维度
        config = clip_model.config
        if hasattr(config, 'projection_dim'):
            self.embed_dim = config.projection_dim
        elif hasattr(config, 'vision_config'):
            self.embed_dim = config.vision_config.hidden_size
        else:
            self.embed_dim = 768

        # 物理知识编码器
        self.knowledge_encoder = PhysicsKnowledgeEncoder(
            input_dim=4, hidden_dim=hidden_dim, output_dim=self.embed_dim
        )

        # 融合模块
        self.fusion = KnowledgeGuidedFusion(self.embed_dim, fusion_type=fusion_type)

        # 构建物理先验向量 [num_classes, 4]
        physics_vectors = build_knowledge_vectors(class_names)
        self.register_buffer("physics_vectors", physics_vectors)

        # 冻结 CLIP
        for param in clip_model.parameters():
            param.requires_grad = False

        # 温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def encode_text_prompts(self, device):
        """编码所有类别的文本提示"""
        prompts = [f"a photo of {name.replace('_', ' ')}" for name in self.class_names]
        if self.processor is not None:
            inputs = self.processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
        else:
            from transformers import CLIPTokenizer
            tokenizer = CLIPTokenizer.from_pretrained(self.clip_model.config._name_or_path)
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
            if not isinstance(text_features, torch.Tensor):
                text_features = text_features.pooler_output if hasattr(text_features, 'pooler_output') else text_features.last_hidden_state[:, 0, :]
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features  # [num_classes, D]

    def forward(self, images):
        """
        Args:
            images: [B, 3, H, W] 输入图像
        Returns:
            logits: [B, num_classes] 分类 logits
        """
        # 图像特征
        img_features = self.clip_model.get_image_features(pixel_values=images)
        if not isinstance(img_features, torch.Tensor):
            img_features = img_features.pooler_output if hasattr(img_features, 'pooler_output') else img_features.last_hidden_state[:, 0, :]
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)

        # 文本特征（从 buffer 获取，无需重新编码）
        text_features = self.encode_text_prompts(img_features.device)  # [C, D]

        # 物理知识编码
        knowledge_embeds = self.knowledge_encoder(self.physics_vectors)  # [C, D]
        knowledge_embeds = knowledge_embeds / (knowledge_embeds.norm(dim=-1, keepdim=True) + 1e-8)

        # 融合
        fused_features = self.fusion(text_features, knowledge_embeds)  # [C, D]
        fused_features = fused_features / (fused_features.norm(dim=-1, keepdim=True) + 1e-8)

        # 相似度
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * img_features @ fused_features.T

        return logits
