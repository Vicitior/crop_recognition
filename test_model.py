# -*- coding: utf-8 -*-
"""
测试 EfficientNet 模型在 test 集上的精度
用法: python test_model.py --model saved_models/best.pth --data-dir dataset
"""
import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.classifier import build_model
from utils.augmentation import get_val_transforms
from models.growth_stages import CLASS_MAP


def evaluate(model, loader, class_names, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 总体准确率
    total_correct = (all_preds == all_labels).sum()
    total_count = len(all_labels)
    overall_acc = total_correct / total_count

    # 每个类别的准确率、精确率、召回率、F1
    print("\n" + "=" * 80)
    print(f"{'类别':<25} {'样本数':>6} {'正确':>6} {'准确率':>8} {'精确率':>8} {'召回率':>8} {'F1':>8}")
    print("=" * 80)

    num_classes = len(class_names)
    for i in range(num_classes):
        mask = all_labels == i
        count = mask.sum()
        if count == 0:
            print(f"{class_names[i]:<25} {'0':>6} {'-':>6} {'-':>8} {'-':>8} {'-':>8} {'-':>8}")
            continue

        correct = (all_preds[mask] == i).sum()
        acc = correct / count

        # 精确率: 预测为该类的样本中，真正属于该类的比例
        pred_mask = all_preds == i
        precision = (all_preds[pred_mask] == all_labels[pred_mask]).sum() / pred_mask.sum() if pred_mask.sum() > 0 else 0

        # 召回率 = acc (真正属于该类的样本中，被正确预测的比例)
        recall = acc

        # F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # 类别信息
        info = CLASS_MAP.get(class_names[i], {})
        crop_cn = info.get("crop_cn", "")
        stage_cn = info.get("stage_cn", "")
        label = f"{crop_cn} {stage_cn}" if crop_cn else class_names[i]

        print(f"{label:<25} {count:>6} {correct:>6} {acc:>8.2%} {precision:>8.2%} {recall:>8.2%} {f1:>8.2%}")

    print("=" * 80)
    print(f"{'总体':<25} {total_count:>6} {total_correct:>6} {overall_acc:>8.2%}")

    # 混淆矩阵摘要
    print("\n--- 混淆矩阵 (行=真实, 列=预测) ---")
    print(f"{'':>25}", end="")
    short_names = [n.split("_")[1][:6] for n in class_names]
    for sn in short_names:
        print(f"{sn:>8}", end="")
    print()

    for i in range(num_classes):
        info = CLASS_MAP.get(class_names[i], {})
        label = f"{info.get('crop_cn', '')}_{info.get('stage_cn', '')}"
        print(f"{label:>25}", end="")
        mask = all_labels == i
        for j in range(num_classes):
            count = ((all_preds[mask]) == j).sum()
            if count > 0:
                print(f"{count:>8}", end="")
            else:
                print(f"{'·':>8}", end="")
        print()

    return overall_acc


def main():
    parser = argparse.ArgumentParser(description="测试 EfficientNet 模型精度")
    parser.add_argument("--model", type=str, default="saved_models/best.pth", help="模型路径")
    parser.add_argument("--data-dir", type=str, default="dataset", help="数据集目录")
    parser.add_argument("--batch-size", type=int, default=32, help="批大小")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="评估哪个集")
    args = parser.parse_args()

    # 设备
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"设备: {device}")

    # 加载模型
    print(f"加载模型: {args.model}")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    class_names = checkpoint.get("class_names", [])
    num_classes = len(class_names)
    print(f"类别数: {num_classes}")
    print(f"类别: {class_names}")

    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # 加载数据
    data_dir = os.path.join(args.data_dir, args.split)
    print(f"数据目录: {data_dir}")

    transform = get_val_transforms()
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"测试样本数: {len(dataset)}")
    print(f"数据集类别: {dataset.classes}")

    # 评估
    acc = evaluate(model, loader, class_names, device)
    print(f"\n总体准确率: {acc:.2%}")


if __name__ == "__main__":
    main()
