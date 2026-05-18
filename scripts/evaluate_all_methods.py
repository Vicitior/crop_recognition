"""
综合评估脚本
评估所有精度提升方法的效果

用法:
    python scripts/evaluate_all_methods.py
    python scripts/evaluate_all_methods.py --compare-tta
"""

import sys
import argparse
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_model(model_path, device):
    """加载微调后的模型"""
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

    # 根据方法构建模型
    if "attention" in method:
        # 注意力增强模型
        from scripts.train_with_attention import CLIPWithAttentionClassifier, apply_lora
        attention_type = method.split("_")[1]
        clip_model = apply_lora(clip_model)
        model = CLIPWithAttentionClassifier(clip_model, num_classes, attention_type=attention_type)
    elif "prompt_learning" in method:
        # 提示学习模型
        from scripts.train_prompt_learning import SimplePromptLearner
        model = SimplePromptLearner(clip_model, class_names)
    elif "mmd" in method:
        # MMD-LoRA模型
        from scripts.train_mmd_lora import MMDLoRAClassifier, apply_mmd_lora
        clip_model = apply_mmd_lora(clip_model)
        model = MMDLoRAClassifier(clip_model, num_classes)
    else:
        # 标准LoRA模型
        from scripts.train_clip import CLIPWithClassifier, apply_lora
        clip_model = apply_lora(clip_model)
        model = CLIPWithClassifier(clip_model, num_classes)

    # 加载权重
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, class_names, checkpoint


def get_test_samples(data_dir):
    """获取测试样本"""
    test_dir = Path(data_dir) / "test"
    samples = []
    class_to_idx = {}

    classes = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
    for idx, cls_name in enumerate(classes):
        class_to_idx[cls_name] = idx
        cls_dir = test_dir / cls_name
        for img_path in cls_dir.glob("*"):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                samples.append((str(img_path), idx, cls_name))

    return samples, class_to_idx


@torch.no_grad()
def evaluate_model(model, test_samples, device, img_size=336, tta=False):
    """评估模型"""
    from torchvision import transforms

    # 基础变换
    base_transform = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    correct = 0
    total = len(test_samples)
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    all_probs = []
    all_labels = []

    for img_path, label, cls_name in tqdm(test_samples, desc="评估"):
        image = Image.open(img_path).convert("RGB")

        if tta:
            # 测试时增强
            from scripts.predict_tta import get_tta_transforms
            tta_transforms = get_tta_transforms("medium")
            augmented_images = [t(image) for t in tta_transforms]
            batch = torch.stack(augmented_images).to(device)
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1).mean(dim=0)
        else:
            image_tensor = base_transform(image).unsqueeze(0).to(device)
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)[0]

        predicted = probs.argmax().item()
        all_probs.append(probs.cpu())
        all_labels.append(label)

        if predicted == label:
            correct += 1
            class_correct[label] += 1
        class_total[label] += 1

    accuracy = 100.0 * correct / total

    # 计算逐类准确率
    class_accuracies = {}
    for idx in class_total:
        acc = 100.0 * class_correct[idx] / class_total[idx]
        class_accuracies[idx] = acc

    return accuracy, class_accuracies, class_total


def main():
    parser = argparse.ArgumentParser(description="综合评估所有方法")
    parser.add_argument("--data-dir", default="dataset", help="数据集目录")
    parser.add_argument("--compare-tta", action="store_true", help="比较TTA效果")
    parser.add_argument("--device", default="auto", help="设备")
    args = parser.parse_args()

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"设备: {device}")

    # 获取测试样本
    print(f"加载测试数据: {args.data_dir}")
    test_samples, class_to_idx = get_test_samples(args.data_dir)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    print(f"测试样本数: {len(test_samples)}")

    # 基准结果（当前最佳模型）
    baseline_acc = 95.65  # 4模型集成精度

    # 扫描所有模型
    saved_models_dir = Path("saved_models/clip")
    model_dirs = [d for d in saved_models_dir.iterdir() if d.is_dir()]

    results = {}

    for model_dir in model_dirs:
        best_path = model_dir / "best.pth"
        if not best_path.exists():
            continue

        print(f"\n{'='*60}")
        print(f"评估模型: {model_dir.name}")
        print('='*60)

        try:
            model, class_names, checkpoint = load_model(str(best_path), device)
            method = checkpoint.get("method", "unknown")

            # 评估（无TTA）
            accuracy, class_accs, class_total = evaluate_model(
                model, test_samples, device, tta=False
            )

            # 评估（有TTA）
            if args.compare_tta:
                accuracy_tta, class_accs_tta, _ = evaluate_model(
                    model, test_samples, device, tta=True
                )
            else:
                accuracy_tta = None

            # 记录结果
            results[model_dir.name] = {
                "method": method,
                "accuracy": accuracy,
                "accuracy_tta": accuracy_tta,
                "improvement": accuracy - baseline_acc,
                "class_accuracies": {idx_to_class[idx]: acc for idx, acc in class_accs.items()},
                "config": checkpoint.get("config", {})
            }

            # 打印结果
            print(f"方法: {method}")
            print(f"准确率: {accuracy:.2f}%")
            if accuracy_tta:
                print(f"TTA准确率: {accuracy_tta:.2f}%")
            print(f"相比基准: {accuracy - baseline_acc:+.2f}%")

            # 打印逐类准确率
            print("\n逐类准确率:")
            for idx, acc in sorted(class_accs.items()):
                cls_name = idx_to_class[idx]
                print(f"  {cls_name}: {acc:.2f}%")

        except Exception as e:
            print(f"加载模型失败: {e}")
            continue

    # 汇总结果
    print("\n" + "="*80)
    print("汇总结果")
    print("="*80)
    print(f"{'模型':<40} {'准确率':<12} {'TTA准确率':<12} {'提升':<12}")
    print("-"*80)

    # 按准确率排序
    sorted_results = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)

    for model_name, result in sorted_results:
        acc = result["accuracy"]
        acc_tta = result.get("accuracy_tta", "-")
        improvement = result["improvement"]

        acc_tta_str = f"{acc_tta:.2f}%" if isinstance(acc_tta, float) else acc_tta
        print(f"{model_name:<40} {acc:<12.2f} {acc_tta_str:<12} {improvement:<+12.2f}")

    # 保存结果
    output_file = saved_models_dir / "evaluation_summary.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "baseline_accuracy": baseline_acc,
            "evaluation_date": datetime.now().isoformat(),
            "results": results
        }, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_file}")

    # 推荐最佳方法
    if sorted_results:
        best_model, best_result = sorted_results[0]
        print(f"\n推荐最佳模型: {best_model}")
        print(f"  准确率: {best_result['accuracy']:.2f}%")
        print(f"  提升: {best_result['improvement']:+.2f}%")


if __name__ == "__main__":
    main()
