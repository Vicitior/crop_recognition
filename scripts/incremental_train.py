# -*- coding: utf-8 -*-
"""
增量训练脚本
在现有模型基础上，用新数据继续训练

用法:
    # 使用默认设置（推荐）
    python scripts/incremental_train.py

    # 指定模型和参数
    python scripts/incremental_train.py --model-path saved_models/clip/clip-vit-large-patch14-336-v2/best.pth --epochs 10

    # 只用新数据训练
    python scripts/incremental_train.py --new-data-only
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_clip_v2 import CLIPWithClassifier, apply_lora, FocalLoss


class IncrementalDataset(Dataset):
    """增量数据集 - 支持只加载新数据"""

    def __init__(self, data_dir, transform=None, split="train", new_data_only=False):
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        # 加载原始训练数据
        train_dir = Path(data_dir) / split
        if train_dir.exists():
            classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
            for idx, cls_name in enumerate(classes):
                self.class_to_idx[cls_name] = idx
                cls_dir = train_dir / cls_name
                for img_path in cls_dir.glob("*"):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        if not new_data_only:
                            self.samples.append((str(img_path), idx, 'original'))

        # 加载用户反馈数据（新数据）
        feedback_dir = Path(data_dir) / "user_feedback"
        if feedback_dir.exists():
            new_count = 0
            for cls_dir in feedback_dir.iterdir():
                if cls_dir.is_dir():
                    cls_name = cls_dir.name
                    if cls_name not in self.class_to_idx:
                        continue
                    idx = self.class_to_idx[cls_name]
                    for img_path in cls_dir.glob("*"):
                        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                            if not img_path.name.endswith('.json'):
                                self.samples.append((str(img_path), idx, 'new'))
                                new_count += 1

            print(f"新数据: {new_count} 张图片")

        # 加载增强数据（如果存在）
        aug_dir = Path(data_dir) / "train_augmented"
        if aug_dir.exists() and not new_data_only:
            aug_count = 0
            for cls_dir in aug_dir.iterdir():
                if cls_dir.is_dir():
                    cls_name = cls_dir.name
                    if cls_name not in self.class_to_idx:
                        continue
                    idx = self.class_to_idx[cls_name]
                    for img_path in cls_dir.glob("*"):
                        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                            self.samples.append((str(img_path), idx, 'augmented'))
                            aug_count += 1

            print(f"增强数据: {aug_count} 张图片")

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)

        # 统计
        source_counts = defaultdict(int)
        for _, _, source in self.samples:
            source_counts[source] += 1

        print(f"\n数据集统计:")
        print(f"  类别数: {self.num_classes}")
        print(f"  总图片: {len(self.samples)}")
        for source, count in source_counts.items():
            print(f"  {source}: {count} 张")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, _ = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = image.transform(image)
        return image, label


def check_new_data(data_dir):
    """检查是否有新数据"""
    feedback_dir = Path(data_dir) / "user_feedback"
    if not feedback_dir.exists():
        return 0

    count = 0
    for cls_dir in feedback_dir.iterdir():
        if cls_dir.is_dir():
            for img_path in cls_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    if not img_path.name.endswith('.json'):
                        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="增量训练脚本")
    parser.add_argument("--data-dir", default="dataset", help="数据集目录")
    parser.add_argument("--model-path", default="saved_models/clip/clip-vit-large-patch14-336-v2/best.pth",
                        help="基础模型路径")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数（增量训练建议5-15）")
    parser.add_argument("--batch-size", type=int, default=4, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率（增量训练建议1e-5到5e-5）")
    parser.add_argument("--new-data-only", action="store_true", help="只用新数据训练")
    parser.add_argument("--use-focal-loss", action="store_true", help="使用Focal Loss")
    parser.add_argument("--output-dir", default=None, help="输出目录")
    parser.add_argument("--device", default="auto", help="设备")
    args = parser.parse_args()

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")

    # 检查新数据
    new_count = check_new_data(args.data_dir)
    print(f"\n检查新数据: {new_count} 张")
    if new_count == 0 and not args.new_data_only:
        print("没有发现新数据，使用原始训练集")

    # 检查模型
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"错误: 模型不存在 - {model_path}")
        print("请先训练基础模型或指定正确的路径")
        return

    # 输出目录
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"saved_models/clip/incremental_{timestamp}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型配置
    config_path = model_path.parent / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {}

    model_name = config.get("model", "openai/clip-vit-large-patch14-336")
    lora_rank = config.get("lora_rank", 16)
    lora_alpha = config.get("lora_alpha", 32)
    num_classes = config.get("num_classes", 15)

    print(f"\n模型配置:")
    print(f"  基础模型: {model_name}")
    print(f"  LoRA rank: {lora_rank}")
    print(f"  类别数: {num_classes}")

    # 加载CLIP模型
    from transformers import CLIPModel, CLIPProcessor, AutoModel, AutoProcessor

    if "siglip" in model_name.lower():
        clip_model = AutoModel.from_pretrained(model_name)
        model_type = "siglip"
    else:
        clip_model = CLIPModel.from_pretrained(model_name)
        model_type = "clip"

    # 应用LoRA
    clip_model = apply_lora(clip_model, rank=lora_rank, alpha=lora_alpha)

    # 创建模型
    img_size = 336 if "336" in model_name else 224
    model = CLIPWithClassifier(clip_model, num_classes, model_type, img_size=img_size)

    # 加载权重
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print(f"模型加载成功: {model_path}")

    # 数据增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    print("\n加载数据集...")
    train_dataset = IncrementalDataset(args.data_dir, train_transform, "train", args.new_data_only)

    # 计算类别权重
    class_sample_counts = [0] * train_dataset.num_classes
    for _, label, _ in train_dataset.samples:
        class_sample_counts[label] += 1

    total_samples = sum(class_sample_counts)
    class_weights = []
    for c in class_sample_counts:
        if c > 0:
            class_weights.append(total_samples / (train_dataset.num_classes * c))
        else:
            class_weights.append(0.0)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    # 采样器
    sample_weights = [class_weights[label] for _, label, _ in train_dataset.samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=sampler, num_workers=0, pin_memory=True)

    # 损失函数
    if args.use_focal_loss:
        criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)
        print("使用 Focal Loss")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
        print("使用 CrossEntropyLoss")

    # 优化器（使用更小的学习率）
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n or "classifier" in n]
    optimizer = optim.AdamW(lora_params, lr=args.lr, weight_decay=1e-4)

    # 训练
    print(f"\n开始增量训练")
    print(f"  Epochs: {args.epochs}")
    print(f"  学习率: {args.lr}")
    print(f"  批大小: {args.batch_size}")
    print("=" * 60)

    best_acc = 0
    history = {"train_loss": [], "train_acc": []}

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {train_loss:.4f} Acc: {train_acc:.2f}%")

        # 保存最佳模型
        if train_acc > best_acc:
            best_acc = train_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "train_acc": train_acc,
            }, output_dir / "best.pth")
            print(f"  >> 保存最佳模型 (Acc: {train_acc:.2f}%)")

    # 保存最终模型
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs - 1,
        "train_acc": train_acc,
    }, output_dir / "last.pth")

    # 保存配置
    config.update({
        "incremental_epochs": args.epochs,
        "incremental_lr": args.lr,
        "base_model": str(model_path),
        "new_data_count": new_count,
        "best_acc": best_acc,
        "timestamp": datetime.now().isoformat(),
    })

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("增量训练完成!")
    print(f"  最佳准确率: {best_acc:.2f}%")
    print(f"  模型保存: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
