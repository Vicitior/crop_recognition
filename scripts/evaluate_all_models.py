"""
评估所有CLIP模型的精度，并进行微调
用法:
    # 只评估排名
    python scripts/evaluate_all_models.py --eval-only

    # 评估并微调排名前N的模型
    python scripts/evaluate_all_models.py --top-n 3 --epochs 15
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.clip_classifier import AVAILABLE_MODELS, CLIPCropClassifier
from models.growth_stages import CLASS_MAP


# ============================================================
# 数据集
# ============================================================

class CropStageDataset(Dataset):
    """作物生长阶段数据集"""

    def __init__(self, data_dir, transform=None, split="test"):
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        if not self.data_dir.exists():
            raise ValueError(f"数据目录不存在: {self.data_dir}")

        classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        for idx, cls_name in enumerate(classes):
            self.class_to_idx[cls_name] = idx
            cls_dir = self.data_dir / cls_name
            for img_path in cls_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    self.samples.append((str(img_path), idx))

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes = len(classes)
        print(f"[{split}] {self.num_classes} 类, {len(self.samples)} 张图片")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ============================================================
# 评估函数
# ============================================================

def evaluate_model(model_name, dataset, device="cuda"):
    """评估单个模型的精度"""
    print(f"\n{'='*60}")
    print(f"评估模型: {model_name}")
    print(f"{'='*60}")

    try:
        # 加载模型
        classifier = CLIPCropClassifier(model_name=model_name, device=device)

        correct = 0
        total = 0
        class_correct = {}
        class_total = {}

        # 初始化统计
        for cls_name in dataset.class_to_idx:
            class_correct[cls_name] = 0
            class_total[cls_name] = 0

        # 评估
        start_time = time.time()
        for img_path, label_idx in dataset.samples:
            # 获取真实类别名
            true_class = dataset.idx_to_class[label_idx]

            # 加载图片
            image = Image.open(img_path).convert("RGB")

            # 预测
            results = classifier.predict(image, top_k=1)
            pred_class = results[0]["class_name"]

            # 调试：打印前几个预测结果
            if total < 3:
                print(f"  真实: {true_class}, 预测: {pred_class}")

            # 统计
            if pred_class == true_class:
                correct += 1
                class_correct[true_class] += 1
            total += 1
            class_total[true_class] += 1

            # 打印进度
            if total % 50 == 0:
                print(f"  进度: {total}/{len(dataset)}, 当前精度: {100.0*correct/total:.1f}%")

        elapsed = time.time() - start_time
        accuracy = 100.0 * correct / total

        # 计算每个类别的精度
        class_accuracies = {}
        for cls_name in class_total:
            if class_total[cls_name] > 0:
                class_accuracies[cls_name] = 100.0 * class_correct[cls_name] / class_total[cls_name]
            else:
                class_accuracies[cls_name] = 0.0

        # 清理显存
        del classifier
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        result = {
            "model": model_name,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "time": elapsed,
            "class_accuracies": class_accuracies,
            "status": "success"
        }

        print(f"\n结果: {accuracy:.2f}% ({correct}/{total})")
        print(f"耗时: {elapsed:.1f}s")
        print(f"各类别精度:")
        for cls_name, acc in class_accuracies.items():
            print(f"  {cls_name}: {acc:.1f}%")

        return result

    except Exception as e:
        import traceback
        print(f"评估失败: {e}")
        traceback.print_exc()
        return {
            "model": model_name,
            "accuracy": 0.0,
            "status": f"error: {str(e)}"
        }


def main():
    parser = argparse.ArgumentParser(description="评估所有CLIP模型并微调")
    parser.add_argument("--data-dir", default="dataset", help="数据集目录")
    parser.add_argument("--eval-only", action="store_true", help="只评估，不微调")
    parser.add_argument("--top-n", type=int, default=3, help="微调排名前N的模型")
    parser.add_argument("--epochs", type=int, default=15, help="微调轮数")
    parser.add_argument("--batch-size", type=int, default=16, help="批大小")
    parser.add_argument("--method", choices=["linear", "lora", "full"], default="lora",
                        help="微调方法")
    parser.add_argument("--device", default="auto", help="设备")
    parser.add_argument("--output-dir", default="saved_models/clip", help="输出目录")
    args = parser.parse_args()

    # 设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"使用设备: {device}")

    # 加载测试数据集
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = CropStageDataset(args.data_dir, transform, "test")

    # 评估所有模型
    results = []
    models_to_evaluate = list(AVAILABLE_MODELS.keys())

    print(f"\n开始评估 {len(models_to_evaluate)} 个模型...")
    print(f"数据集: {args.data_dir}")
    print(f"测试样本数: {len(dataset)}")

    for model_name in models_to_evaluate:
        result = evaluate_model(model_name, dataset, device)
        results.append(result)

        # 保存中间结果
        with open(Path(args.output_dir) / "eval_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # 按精度排名
    successful_results = [r for r in results if r["status"] == "success"]
    successful_results.sort(key=lambda x: x["accuracy"], reverse=True)

    # 打印排名
    print("\n" + "=" * 70)
    print("模型精度排名")
    print("=" * 70)
    print(f"{'排名':<4} {'模型':<25} {'精度':<10} {'耗时':<10}")
    print("-" * 70)

    for i, r in enumerate(successful_results):
        print(f"{i+1:<4} {r['model']:<25} {r['accuracy']:.2f}%{'':<4} {r['time']:.1f}s")

    # 保存排名结果
    ranking_file = Path(args.output_dir) / "model_ranking.json"
    with open(ranking_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "dataset": args.data_dir,
            "test_samples": len(dataset),
            "ranking": successful_results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n排名结果已保存: {ranking_file}")

    # 微调排名前N的模型
    if not args.eval_only and successful_results:
        top_models = successful_results[:args.top_n]
        print(f"\n{'='*70}")
        print(f"开始微调排名前 {len(top_models)} 的模型")
        print(f"{'='*70}")

        for i, r in enumerate(top_models):
            model_name = r["model"]
            print(f"\n[{i+1}/{len(top_models)}] 微调模型: {model_name}")
            print(f"当前精度: {r['accuracy']:.2f}%")

            # 构建训练命令
            output_dir = Path(args.output_dir) / model_name.replace("/", "_")
            cmd = (
                f"python scripts/train_clip.py "
                f"--model {model_name} "
                f"--method {args.method} "
                f"--data-dir {args.data_dir} "
                f"--epochs {args.epochs} "
                f"--batch-size {args.batch_size} "
                f"--output-dir {output_dir}"
            )

            print(f"执行命令: {cmd}")
            os.system(cmd)

        print("\n所有微调完成！")
        print(f"微调模型保存在: {args.output_dir}/")


if __name__ == "__main__":
    main()
