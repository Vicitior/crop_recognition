"""
ONNX 动态 INT8 量化脚本 (Crop Growth Stage Recognition ONNX INT8 Quantizer)

功能：
1. 读取 FP32 版本的 ONNX 模型 (约 1.7GB)
2. 使用 ONNX Runtime 动态量化 (Dynamic Quantization) 将权重转换为 INT8
3. 输出高压缩率、极低内存占用的 INT8 模型 (约 400MB)
"""

import sys
import argparse
from pathlib import Path

# 配置控制台输出 UTF-8 编码
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

def quantize_onnx(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"未找到输入 FP32 ONNX 模型: {input_path}")

    print(f"📦 正在加载 FP32 ONNX 模型: {input_path} (原大小: {input_path.stat().st_size / (1024*1024):.2f} MB)")
    print("⚡ 正在执行 INT8 动态量化 (Dynamic INT8 Quantization)...")

    try:
        import onnx
        import onnxruntime.quantization as quant

        print(f"📦 正在加载 FP32 ONNX 模型: {input_path} ...")
        model = onnx.load(str(input_path))

        print("⚡ 正在执行 INT8 动态量化 (Dynamic INT8 Quantization)...")
        quant.quantize_dynamic(
            model_input=model,
            model_output=str(output_path),
            weight_type=quant.QuantType.QUInt8,
        )
    except ImportError:
        print("❌ 缺少 onnxruntime，请运行: pip install onnxruntime onnx")
        sys.exit(1)

    orig_mb = input_path.stat().st_size / (1024 * 1024)
    quant_mb = output_path.stat().st_size / (1024 * 1024)
    ratio = (1 - quant_mb / orig_mb) * 100

    print(f"🎉 INT8 量化成功！")
    print(f"📁 路径: {output_path}")
    print(f"📊 量化前大小: {orig_mb:.2f} MB")
    print(f"📊 量化后大小: {quant_mb:.2f} MB")
    print(f"📉 压缩比例: -{ratio:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize FP32 ONNX Crop Model to INT8")
    parser.add_argument("--input-path", type=str, default="saved_models/onnx/crop_model_fp32.onnx",
                        help="Input FP32 ONNX model path")
    parser.add_argument("--output-path", type=str, default="saved_models/onnx/crop_model_int8.onnx",
                        help="Output INT8 ONNX model path")
    args = parser.parse_args()

    quantize_onnx(args.input_path, args.output_path)
