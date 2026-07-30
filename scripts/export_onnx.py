"""
ONNX 模型融合与导出脚本 (Crop Growth Stage Recognition ONNX Exporter)

功能：
1. 加载 PyTorch 创新模型 (CLIP ViT-L/14@336 + Adaptive LoRA + Confidence Router)
2. 融合 Adaptive LoRA 权重回 CLIP ViT 主干网络 (消除运行时 LoRA 计算开销)
3. 向量化重构 ConfidenceRouterClassifier (消除 Python 条件分支与循环，确保 ONNX 导出兼容)
4. 导出为端到端 ONNX 模型 (输入: [1, 3, 336, 336] 图片 Tensor, 输出: [1, 15] 概率分布)
"""

import os
import sys
import argparse
from pathlib import Path

# 配置控制台输出 UTF-8 编码 (解决 Windows 终端 GBK 打印 Emoji 异常)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch
import torch.nn as nn
import torch.nn.functional as F

# 添加项目根目录到 sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from transformers import CLIPModel
from models.confidence_router import ConfidenceRouterClassifier
from models.adaptive_lora import apply_adaptive_lora
from models.growth_stages import CLASS_MAP


def fuse_adaptive_lora_and_restore(model, target_modules=None):
    """
    将 AdaptiveLoRA 的专家权重计算融合回原始 nn.Linear 中，
    并将模块还原为原生 nn.Linear，提升移动端推理性能。
    """
    if target_modules is None:
        target_modules = ["q_proj"]

    fused_count = 0
    for name, module in model.named_modules():
        for target_name in target_modules:
            if hasattr(module, target_name):
                layer = getattr(module, target_name)
                if hasattr(layer, "experts") and hasattr(layer, "original"):
                    orig_linear = layer.original
                    num_experts = len(layer.experts)
                    delta_w = torch.zeros_like(orig_linear.weight.data)
                    for expert in layer.experts.values():
                        a_w = expert.lora_A.weight.data  # [rank, in_feat]
                        b_w = expert.lora_B.weight.data  # [out_feat, rank]
                        scaling = expert.scaling
                        delta_w += scaling * (b_w @ a_w)
                    delta_w = delta_w / num_experts
                    orig_linear.weight.data += delta_w
                    setattr(module, target_name, orig_linear)
                    fused_count += 1

    print(f"✅ [LoRA 融合] 成功融合并还原了 {fused_count} 个 Linear 层。")
    return model


class VectorizedEndToEndCropModel(nn.Module):
    """
    端到端向量化 ONNX 兼容包装类
    输入: pixel_values [B, 3, 336, 336]
    输出: probs [B, 15] (15 个生育阶段的 Softmax 概率分布)
    """

    def __init__(self, clip_model, router, threshold=0.7):
        super().__init__()
        self.clip_model = clip_model
        self.router = router
        self.threshold = threshold

    def forward(self, pixel_values):
        # 1. CLIP 视觉编码 -> 提取 768 维投影特征
        vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values)
        pooled = vision_outputs.pooler_output
        features = self.clip_model.visual_projection(pooled)  # [B, 768]

        # 2. 特征门控增强
        gate = self.router.feature_enhance(features)
        enhanced_features = features * gate

        # 3. 作物路由分布预测
        crop_logits = self.router.crop_router(enhanced_features)  # [B, 3]
        crop_probs = F.softmax(crop_logits, dim=-1)  # [B, 3]

        # 4. 各作物 5 阶段分支预测
        s0 = self.router.stage_branches[0](enhanced_features)  # [B, 5]
        s1 = self.router.stage_branches[1](enhanced_features)  # [B, 5]
        s2 = self.router.stage_branches[2](enhanced_features)  # [B, 5]

        # 5. 置信度判断 (向量化掩码)
        max_prob, top_crop = crop_probs.max(dim=-1, keepdim=True)  # [B, 1]

        # 5.1 软路由 (Soft Routing): 多分支加权
        soft_s0 = s0 * crop_probs[:, 0:1]
        soft_s1 = s1 * crop_probs[:, 1:2]
        soft_s2 = s2 * crop_probs[:, 2:3]
        soft_logits = torch.cat([soft_s0, soft_s1, soft_s2], dim=-1)  # [B, 15]

        # 5.2 硬路由 (Hard Routing): 只保留 Top-1 分支
        m0 = (top_crop == 0).float()
        m1 = (top_crop == 1).float()
        m2 = (top_crop == 2).float()
        hard_logits = torch.cat([s0 * m0, s1 * m1, s2 * m2], dim=-1)  # [B, 15]

        # 5.3 根据置信度阈值选择分支
        use_hard = (max_prob > self.threshold).float()  # [B, 1]
        fused_logits = use_hard * hard_logits + (1.0 - use_hard) * soft_logits

        # 6. 全局 Softmax
        probs = F.softmax(fused_logits, dim=-1)
        return probs


def export_onnx(model_path, output_path, device="cpu", dynamic_batch=False):
    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"未找到训练模型文件: {model_path}")

    print(f"📦 正在从 {model_path} 加载 PyTorch 权重...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # 1. 初始化 CLIP 架构
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
    crop_ranks = {0: 4, 1: 8, 2: 16}
    clip_model, _ = apply_adaptive_lora(
        clip_model, crop_ranks=crop_ranks,
        target_modules=["q_proj"], dropout=0.0
    )

    if "lora_state" in checkpoint:
        clip_model.load_state_dict(checkpoint["lora_state"], strict=False)

    # 2. 融合 LoRA 权重
    clip_model = fuse_adaptive_lora_and_restore(clip_model, target_modules=["q_proj"])
    clip_model.eval()

    # 3. 初始化并加载 ConfidenceRouterClassifier
    feat_dim = clip_model.config.projection_dim  # 768
    router = ConfidenceRouterClassifier(
        feat_dim=feat_dim, num_classes=15, hidden_dim=256
    )

    if "classifier_state" in checkpoint and "confidence_router" in checkpoint["classifier_state"]:
        router.load_state_dict(checkpoint["classifier_state"]["confidence_router"])

    router.eval()

    # 4. 构建向量化端到端包装模型
    end_to_end_model = VectorizedEndToEndCropModel(clip_model, router, threshold=0.7)
    end_to_end_model.eval()
    end_to_end_model.to(device)

    # 5. 准备 Dummy Input [1, 3, 336, 336]
    dummy_input = torch.randn(1, 3, 336, 336, device=device)

    print(f"🚀 开始导出 ONNX 模型至 {output_path}...")
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"pixel_values": {0: "batch_size"}, "output_probs": {0: "batch_size"}}

    torch.onnx.export(
        end_to_end_model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["output_probs"],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"🎉 ONNX 导出成功！")
    print(f"📁 路径: {output_path}")
    print(f"📊 模型体积: {size_mb:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Crop Growth Stage Recognition Model to ONNX")
    parser.add_argument("--model-path", type=str, default="saved_models/innovations/all_innovations/best.pth",
                        help="Path to best.pth model file")
    parser.add_argument("--output-path", type=str, default="saved_models/onnx/crop_model_fp32.onnx",
                        help="Output ONNX file path")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic batch size")
    args = parser.parse_args()

    export_onnx(args.model_path, args.output_path, dynamic_batch=args.dynamic)
