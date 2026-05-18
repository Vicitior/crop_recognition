"""
CLIP微调模型预测脚本
用法:
    python scripts/predict_clip.py --image test.jpg --model-path saved_models/clip/best.pth
"""

import sys
import argparse
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_finetuned_clip(model_path, device="cpu"):
    """加载微调后的CLIP模型"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    method = checkpoint.get("method", "lora")
    model_name = checkpoint.get("model_name", "openai/clip-vit-base-patch32")
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    # 加载CLIP模型
    from transformers import CLIPModel, AutoModel

    if "siglip" in model_name.lower():
        clip_model = AutoModel.from_pretrained(model_name)
    else:
        clip_model = CLIPModel.from_pretrained(model_name)

    # 如果是LoRA，先应用LoRA
    if method == "lora":
        from scripts.train_clip import apply_lora
        clip_model = apply_lora(clip_model)

    # 构建完整模型
    from scripts.train_clip import CLIPWithClassifier
    model = CLIPWithClassifier(clip_model, num_classes)

    # 加载权重
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # 加载processor
    from transformers import CLIPProcessor, AutoProcessor
    if "siglip" in model_name.lower():
        processor = AutoProcessor.from_pretrained(model_name)
    else:
        processor = CLIPProcessor.from_pretrained(model_name)

    return model, processor, class_names


@torch.no_grad()
def predict(model, processor, image_path, class_names, device="cpu"):
    """预测单张图片"""
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    outputs = model(image_tensor)
    probs = torch.softmax(outputs, dim=1)

    # 获取top-k结果
    top_probs, top_indices = probs[0].topk(min(5, len(class_names)))

    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append({
            "class": class_names[idx.item()],
            "confidence": prob.item()
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="CLIP微调模型预测")
    parser.add_argument("--image", required=True, help="图片路径")
    parser.add_argument("--model-path", default="saved_models/clip/best.pth",
                        help="模型路径")
    parser.add_argument("--device", default="auto", help="设备")
    args = parser.parse_args()

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"加载模型: {args.model_path}")
    model, processor, class_names = load_finetuned_clip(args.model_path, device)
    print(f"类别: {class_names}")

    print(f"\n预测: {args.image}")
    results = predict(model, processor, args.image, class_names, device)

    print("\n预测结果:")
    for i, r in enumerate(results):
        print(f"  {i+1}. {r['class']}: {r['confidence']:.2%}")


if __name__ == "__main__":
    main()
