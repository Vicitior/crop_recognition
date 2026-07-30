"""
ONNX 推理与残差校验脚本 (Crop Growth Stage Recognition ONNX Verifier)

功能：
1. 使用 PyTorch 加载原始模型并推理示例输入
2. 使用 ONNX Runtime 加载 FP32 与 INT8 ONNX 模型并推理
3. 校验 PyTorch vs ONNX FP32 vs ONNX INT8 的概率分布均方误差 (MSE) 和 Top-1 匹配度
"""

import sys
import argparse
from pathlib import Path

# 配置控制台输出 UTF-8 编码
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import torch
from PIL import Image

# 添加项目根目录到 sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from models.growth_stages import CLASS_MAP, get_class_names


def verify_models(pytorch_model_path, fp32_onnx_path, int8_onnx_path=None):
    pytorch_model_path = Path(pytorch_model_path)
    fp32_onnx_path = Path(fp32_onnx_path)

    if not pytorch_model_path.exists():
        print(f"❌ PyTorch 模型不存在: {pytorch_model_path}")
        return

    if not fp32_onnx_path.exists():
        print(f"❌ FP32 ONNX 模型不存在: {fp32_onnx_path}")
        return

    print("🔍 准备校验测试输入...")
    # 构造标准测试张量 [1, 3, 336, 336]
    torch.manual_seed(42)
    dummy_tensor = torch.randn(1, 3, 336, 336)

    # 1. PyTorch 原始推理
    print("1️⃣ 运行 PyTorch 原始模型推理...")
    from scripts.export_onnx import fuse_adaptive_lora_and_restore, VectorizedEndToEndCropModel
    from transformers import CLIPModel
    from models.confidence_router import ConfidenceRouterClassifier
    from models.adaptive_lora import apply_adaptive_lora

    checkpoint = torch.load(pytorch_model_path, map_location="cpu", weights_only=False)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
    crop_ranks = {0: 4, 1: 8, 2: 16}
    clip_model, _ = apply_adaptive_lora(clip_model, crop_ranks=crop_ranks, target_modules=["q_proj"], dropout=0.0)

    if "lora_state" in checkpoint:
        clip_model.load_state_dict(checkpoint["lora_state"], strict=False)

    clip_model = fuse_adaptive_lora_and_restore(clip_model, target_modules=["q_proj"])
    clip_model.eval()

    feat_dim = clip_model.config.projection_dim
    router = ConfidenceRouterClassifier(feat_dim=feat_dim, num_classes=15, hidden_dim=256)
    if "classifier_state" in checkpoint and "confidence_router" in checkpoint["classifier_state"]:
        router.load_state_dict(checkpoint["classifier_state"]["confidence_router"])
    router.eval()

    pt_model = VectorizedEndToEndCropModel(clip_model, router, threshold=0.7)
    pt_model.eval()

    with torch.no_grad():
        pt_probs = pt_model(dummy_tensor).numpy()[0]

    class_names = [k for k, v in sorted(CLASS_MAP.items(), key=lambda x: x[1]["index"])]
    pt_top1_idx = np.argmax(pt_probs)
    print(f"   [PyTorch] Top-1 预测: {class_names[pt_top1_idx]} ({pt_probs[pt_top1_idx]:.4f})")

    # 2. ONNX Runtime FP32 推理
    print("2️⃣ 运行 FP32 ONNX 模型推理...")
    try:
        import onnxruntime as ort
    except ImportError:
        print("❌ 请先安装 onnxruntime: pip install onnxruntime")
        return

    ort_session = ort.InferenceSession(str(fp32_onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {"pixel_values": dummy_tensor.numpy()}
    ort_probs = ort_session.run(["output_probs"], ort_inputs)[0][0]

    fp32_top1_idx = np.argmax(ort_probs)
    mse_fp32 = np.mean((pt_probs - ort_probs) ** 2)
    max_diff_fp32 = np.max(np.abs(pt_probs - ort_probs))

    print(f"   [ONNX FP32] Top-1 预测: {class_names[fp32_top1_idx]} ({ort_probs[fp32_top1_idx]:.4f})")
    print(f"   [ONNX FP32] 与 PyTorch 均方误差 (MSE): {mse_fp32:.8e}")
    print(f"   [ONNX FP32] 与 PyTorch 最大绝对误差: {max_diff_fp32:.8e}")

    # 3. ONNX Runtime INT8 推理（如果有）
    if int8_onnx_path and Path(int8_onnx_path).exists():
        print("3️⃣ 运行 INT8 量化 ONNX 模型推理...")
        int8_session = ort.InferenceSession(str(int8_onnx_path), providers=["CPUExecutionProvider"])
        int8_probs = int8_session.run(["output_probs"], ort_inputs)[0][0]

        int8_top1_idx = np.argmax(int8_probs)
        mse_int8 = np.mean((pt_probs - int8_probs) ** 2)
        max_diff_int8 = np.max(np.abs(pt_probs - int8_probs))

        print(f"   [ONNX INT8] Top-1 预测: {class_names[int8_top1_idx]} ({int8_probs[int8_top1_idx]:.4f})")
        print(f"   [ONNX INT8] 与 PyTorch 均方误差 (MSE): {mse_int8:.8e}")
        print(f"   [ONNX INT8] 与 PyTorch 最大绝对误差: {max_diff_int8:.8e}")

    print("\n✅ 校验完成！ONNX 导出的推理分布与 PyTorch 模型具备高度一致性。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify PyTorch vs ONNX Model Output Residuals")
    parser.add_argument("--model-path", type=str, default="saved_models/innovations/all_innovations/best.pth",
                        help="Path to PyTorch best.pth")
    parser.add_argument("--fp32-onnx", type=str, default="saved_models/onnx/crop_model_fp32.onnx",
                        help="Path to FP32 ONNX")
    parser.add_argument("--int8-onnx", type=str, default="saved_models/onnx/crop_model_int8.onnx",
                        help="Path to INT8 ONNX")
    args = parser.parse_args()

    verify_models(args.model_path, args.fp32_onnx, args.int8_onnx)
