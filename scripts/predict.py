"""
单张图片预测脚本
识别农作物种类和生长阶段

使用方法（CLIP零样本，无需训练）：
    python scripts/predict.py --image path/to/image.jpg
    python scripts/predict.py --image path/to/image.jpg --top-k 3

使用方法（微调模型）：
    python scripts/predict.py --image path/to/image.jpg --mode finetuned --model-path saved_models/best.pth
"""
import os
import sys
import argparse
from pathlib import Path

import torch
from PIL import Image

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.growth_stages import CROP_INFO


def print_results(results, image_path):
    print(f"\n图片: {image_path}")
    print("=" * 50)
    for i, r in enumerate(results):
        info = r.get("info")
        print(f"\n[Top {i+1}] 置信度: {r['confidence']:.2%}")
        if info:
            print(f"  作物: {info['crop_name']}")
            print(f"  生长阶段: {info['stage_name']}")
            print(f"  阶段时间: {info['stage_days']}")
            print(f"  全生育期: {info['total_days']}")
            print(f"  特征描述: {info['description']}")
        else:
            print(f"  类别: {r['class_name']}")


def predict_clip(image_path, top_k=3, model_name="siglip2-so400m"):
    from models.clip_classifier import get_clip_classifier
    classifier = get_clip_classifier(model_name=model_name)
    image = Image.open(image_path).convert("RGB")
    return classifier.predict(image, top_k=top_k)


def predict_finetuned(image_path, model_path, top_k=3):
    from models.classifier import build_model
    from models.growth_stages import NUM_CLASSES, get_crop_info, get_class_names
    from utils.augmentation import get_predict_transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint.get("class_names", get_class_names())

    model = build_model(num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    transform = get_predict_transforms()
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        top_probs, top_indices = probs.topk(top_k, dim=1)

    results = []
    for i in range(top_k):
        idx = top_indices[0][i].item()
        prob = top_probs[0][i].item()
        class_name = class_names[idx]
        info = get_crop_info(class_name)
        results.append({
            "class_name": class_name,
            "confidence": prob,
            "info": info
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="农作物图片预测")
    parser.add_argument("--image", type=str, required=True, help="图片路径")
    parser.add_argument("--mode", type=str, default="clip", choices=["clip", "finetuned"],
                        help="预测模式: clip(零样本) 或 finetuned(微调模型)")
    parser.add_argument("--clip-model", type=str, default="siglip2-so400m",
                        help="零样本模型，可选: siglip2-so400m(默认/最强), siglip-so400m, clip-large-336 等")
    parser.add_argument("--model-path", type=str, default="saved_models/best.pth", help="微调模型路径")
    parser.add_argument("--top-k", type=int, default=3, help="显示前K个预测结果")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"错误: 图片不存在 - {args.image}")
        return

    if args.mode == "clip":
        results = predict_clip(args.image, top_k=args.top_k, model_name=args.clip_model)
    else:
        if not os.path.isfile(args.model_path):
            print(f"错误: 模型文件不存在 - {args.model_path}")
            return
        results = predict_finetuned(args.image, args.model_path, top_k=args.top_k)

    print_results(results, args.image)


if __name__ == "__main__":
    main()
