"""
TTA效果评估脚本
比较不同TTA策略的精度提升

用法:
    python scripts/evaluate_tta.py --model-path saved_models/clip/best.pth
    python scripts/evaluate_tta.py --model-path saved_models/clip/best.pth --tta-level strong
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_finetuned_clip(model_path, device="cpu"):
    """加载微调后的CLIP模型"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    method = checkpoint.get("method", "lora")
    model_name = checkpoint.get("model_name", "openai/clip-vit-base-patch32")
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    # 确定图片大小
    img_size = 336 if "336" in model_name else 224

    # 从checkpoint推断LoRA rank
    state_dict = checkpoint["model_state_dict"]
    lora_rank = 16  # 默认值
    for key in state_dict.keys():
        if "lora_A" in key:
            lora_rank = state_dict[key].shape[0]
            break

    print(f"检测到LoRA rank: {lora_rank}")

    # 加载CLIP模型
    from transformers import CLIPModel, AutoModel

    if "siglip" in model_name.lower():
        clip_model = AutoModel.from_pretrained(model_name)
    else:
        clip_model = CLIPModel.from_pretrained(model_name)

    # 如果是LoRA，先应用LoRA
    if method == "lora":
        from scripts.train_clip_v2 import apply_lora
        clip_model = apply_lora(clip_model, rank=lora_rank, alpha=lora_rank*2)

    # 构建完整模型（使用V2版本的更深分类头）
    from scripts.train_clip_v2 import CLIPWithClassifier
    model = CLIPWithClassifier(clip_model, num_classes, img_size=img_size)

    # 加载权重
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, class_names, img_size


def get_test_dataset(data_dir):
    """加载测试数据集"""
    from torchvision import transforms

    test_dir = Path(data_dir) / "test"
    if not test_dir.exists():
        raise ValueError(f"测试目录不存在: {test_dir}")

    samples = []
    class_to_idx = {}

    classes = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
    for idx, cls_name in enumerate(classes):
        class_to_idx[cls_name] = idx
        cls_dir = test_dir / cls_name
        for img_path in cls_dir.glob("*"):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                samples.append((str(img_path), idx))

    return samples, class_to_idx


@torch.no_grad()
def evaluate_without_tta(model, test_samples, device, img_size=None):
    """不使用TTA评估"""
    from torchvision import transforms

    # 自动检测图片大小
    if img_size is None:
        # 从模型路径推断
        img_size = 336  # 默认使用336

    transform = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    correct = 0
    total = len(test_samples)
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for img_path, label in tqdm(test_samples, desc="评估（无TTA）"):
        image = Image.open(img_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        outputs = model(image_tensor)
        _, predicted = outputs.max(1)

        if predicted.item() == label:
            correct += 1
            class_correct[label] += 1
        class_total[label] += 1

    accuracy = 100.0 * correct / total
    return accuracy, class_correct, class_total


@torch.no_grad()
def evaluate_with_tta(model, test_samples, device, tta_level="medium", batch_size=8):
    """使用TTA评估"""
    from scripts.predict_tta import get_tta_transforms

    correct = 0
    total = len(test_samples)
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for img_path, label in tqdm(test_samples, desc=f"评估（TTA-{tta_level}）"):
        image = Image.open(img_path).convert("RGB")

        # 获取TTA变换
        tta_transforms_list = get_tta_transforms(tta_level)

        # 应用变换
        augmented_images = []
        for t in tta_transforms_list:
            augmented_images.append(t(image))

        # 批量预测
        all_probs = []
        num_batches = (len(augmented_images) + batch_size - 1) // batch_size

        for i in range(num_batches):
            batch = augmented_images[i * batch_size: (i + 1) * batch_size]
            batch_tensor = torch.stack(batch).to(device)

            outputs = model(batch_tensor)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu())

        # 合并预测
        all_probs = torch.cat(all_probs, dim=0)
        avg_probs = all_probs.mean(dim=0)
        predicted = avg_probs.argmax().item()

        if predicted == label:
            correct += 1
            class_correct[label] += 1
        class_total[label] += 1

    accuracy = 100.0 * correct / total
    return accuracy, class_correct, class_total


def main():
    parser = argparse.ArgumentParser(description="TTA效果评估")
    parser.add_argument("--data-dir", default="dataset", help="数据集目录")
    parser.add_argument("--model-path", default="saved_models/clip/best.pth",
                        help="模型路径")
    parser.add_argument("--tta-level", choices=["basic", "medium", "strong"],
                        default="medium", help="TTA级别")
    parser.add_argument("--batch-size", type=int, default=8, help="批处理大小")
    parser.add_argument("--device", default="auto", help="设备")
    args = parser.parse_args()

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"设备: {device}")

    # 加载模型
    print(f"加载模型: {args.model_path}")
    model, class_names, img_size = load_finetuned_clip(args.model_path, device)
    print(f"图片大小: {img_size}x{img_size}")

    # 加载测试数据
    print(f"加载测试数据: {args.data_dir}")
    test_samples, class_to_idx = get_test_dataset(args.data_dir)
    print(f"测试样本数: {len(test_samples)}")

    # 评估（无TTA）
    print("\n" + "="*60)
    print("评估（无TTA）")
    print("="*60)
    acc_no_tta, class_correct_no_tta, class_total = evaluate_without_tta(
        model, test_samples, device, img_size
    )

    # 评估（有TTA）
    print("\n" + "="*60)
    print(f"评估（TTA-{args.tta_level}）")
    print("="*60)
    acc_tta, class_correct_tta, _ = evaluate_with_tta(
        model, test_samples, device, args.tta_level, args.batch_size
    )

    # 输出结果
    print("\n" + "="*60)
    print("评估结果对比")
    print("="*60)
    print(f"无TTA准确率: {acc_no_tta:.2f}%")
    print(f"有TTA准确率: {acc_tta:.2f}%")
    print(f"提升: {acc_tta - acc_no_tta:+.2f}%")

    # 逐类对比
    print("\n逐类准确率对比:")
    print("-" * 60)
    print(f"{'类别':<25} {'无TTA':<12} {'有TTA':<12} {'提升':<12}")
    print("-" * 60)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    for idx in sorted(class_total.keys()):
        cls_name = idx_to_class[idx]
        total = class_total[idx]
        acc_no = 100.0 * class_correct_no_tta[idx] / total
        acc_yes = 100.0 * class_correct_tta[idx] / total
        improvement = acc_yes - acc_no
        print(f"{cls_name:<25} {acc_no:<12.2f} {acc_yes:<12.2f} {improvement:<+12.2f}")

    # 保存结果
    results = {
        "model_path": args.model_path,
        "tta_level": args.tta_level,
        "accuracy_without_tta": acc_no_tta,
        "accuracy_with_tta": acc_tta,
        "improvement": acc_tta - acc_no_tta,
        "class_details": {}
    }

    for idx in sorted(class_total.keys()):
        cls_name = idx_to_class[idx]
        total = class_total[idx]
        results["class_details"][cls_name] = {
            "total": total,
            "correct_no_tta": class_correct_no_tta[idx],
            "correct_tta": class_correct_tta[idx],
            "accuracy_no_tta": 100.0 * class_correct_no_tta[idx] / total,
            "accuracy_tta": 100.0 * class_correct_tta[idx] / total,
        }

    import json
    output_file = Path(args.model_path).parent / f"tta_results_{args.tta_level}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_file}")


if __name__ == "__main__":
    main()
