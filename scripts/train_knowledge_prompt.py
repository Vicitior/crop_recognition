# -*- coding: utf-8 -*-
"""
创新点 1 训练脚本：知识引导 Prompt 微调 (KGPT)
将农学物理机制（积温、生长速率、LAI）编码为可学习向量，与 CLIP 文本嵌入融合

用法:
    python scripts/train_knowledge_prompt.py --model openai/clip-vit-large-patch14-336
    python scripts/train_knowledge_prompt.py --model openai/clip-vit-large-patch14-336 --fusion gating
    python scripts/train_knowledge_prompt.py --model openai/clip-vit-large-patch14-336 --fusion cross_attention
"""

import os
import sys
import argparse
import json
import time
import random
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.knowledge_encoder import KnowledgeGuidedPromptLearner, build_knowledge_vectors


# ============================================================
# 数据集
# ============================================================

class CropStageDataset(Dataset):
    def __init__(self, data_dir, transform=None, split="train"):
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        if not self.data_dir.exists():
            raise ValueError(f"数据目录不存在: {self.data_dir}")

        classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        for idx, cls_name in enumerate(classes):
            self.class_to_idx[cls_name] = idx
            self.idx_to_class[idx] = cls_name
            cls_dir = self.data_dir / cls_name
            for img_path in cls_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append((str(img_path), idx))

        self.num_classes = len(classes)
        self.class_names = list(self.class_to_idx.keys())
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
# Mixup
# ============================================================

def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    index = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam


# ============================================================
# 学习率调度器
# ============================================================

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            factor = (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1 + np.cos(np.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = max(self.min_lr, base_lr * factor)


# ============================================================
# 训练 / 评估
# ============================================================

def train_epoch(model, dataloader, criterion, optimizer, device, mixup_alpha=0.4):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        if mixup_alpha > 0 and random.random() < 0.5:
            mixed_images, y_a, y_b, lam = mixup_data(images, labels, mixup_alpha)
            outputs = model(mixed_images)
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(dataloader), 100. * correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        for pred, label in zip(predicted, labels):
            class_total[label.item()] += 1
            if pred.item() == label.item():
                class_correct[label.item()] += 1

    class_accuracies = {
        idx: 100.0 * class_correct[idx] / class_total[idx]
        for idx in class_total
    }
    return total_loss / len(dataloader), 100. * correct / total, class_accuracies


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="知识引导 Prompt 微调 (KGPT)")
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--model", default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--fusion", choices=["additive", "gating", "cross_attention"],
                        default="gating", help="融合策略")
    parser.add_argument("--hidden-dim", type=int, default=256, help="知识编码器隐藏维度")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--early-stop", type=int, default=10)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" \
        else torch.device(args.device)
    print(f"设备: {device}")

    # 输出目录
    if args.output_dir is None:
        model_short = args.model.split("/")[-1]
        args.output_dir = f"saved_models/clip/{model_short}-kgpt-{args.fusion}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载 CLIP
    print(f"加载 CLIP: {args.model}")
    from transformers import CLIPModel, CLIPProcessor

    clip_model = CLIPModel.from_pretrained(args.model)
    processor = CLIPProcessor.from_pretrained(args.model)
    img_size = 336 if "336" in args.model else 224
    print(f"图片大小: {img_size}x{img_size}")

    # 数据增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # 数据集
    train_dataset = CropStageDataset(args.data_dir, train_transform, "train")
    val_dataset = CropStageDataset(args.data_dir, val_transform, "val")
    test_dataset = CropStageDataset(args.data_dir, val_transform, "test")

    num_classes = train_dataset.num_classes
    class_names = train_dataset.class_names
    idx_to_class = train_dataset.idx_to_class
    print(f"类别: {num_classes} — {class_names}")

    # 类别权重
    class_sample_counts = [0] * num_classes
    for _, label in train_dataset.samples:
        class_sample_counts[label] += 1
    total_samples = sum(class_sample_counts)
    class_weights = []
    for c in class_sample_counts:
        if c > 0:
            class_weights.append(total_samples / (num_classes * c))
        else:
            class_weights.append(0.0)

    sample_weights = [class_weights[label] for _, label in train_dataset.samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=sampler, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)

    # 构建模型
    print(f"构建 KGPT 模型 (fusion={args.fusion}, hidden_dim={args.hidden_dim})")
    model = KnowledgeGuidedPromptLearner(
        clip_model, class_names,
        fusion_type=args.fusion, hidden_dim=args.hidden_dim,
        processor=processor
    )
    model = model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")

    # 损失函数
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=args.label_smoothing)

    # 优化器：只训练知识编码器和融合模块
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params_list, lr=args.lr, weight_decay=1e-4)
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs, args.epochs)

    # 训练
    best_val_acc = 0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_class_acc": [], "lr": []}

    print(f"\n{'='*70}")
    print(f" 开始训练: KGPT (Knowledge-guided Prompt Tuning)")
    print(f" Fusion={args.fusion}, LR={args.lr}, Epochs={args.epochs}")
    print(f"{'='*70}\n")

    for epoch in range(args.epochs):
        t0 = time.time()
        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, args.mixup_alpha)
        val_loss, val_acc, class_accs = evaluate(model, val_loader, criterion, device)

        class_acc_names = {idx_to_class[idx]: acc for idx, acc in class_accs.items()}
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_class_acc"].append(class_acc_names)
        history["lr"].append(current_lr)

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.2e} | {elapsed:.1f}s")

        minority_classes = [c for c, n in zip(class_names, class_sample_counts) if n < 30]
        if minority_classes:
            acc_strs = [f"{c}:{class_acc_names.get(c, 0):.1f}%" for c in minority_classes]
            print(f"  少数类: {', '.join(acc_strs)}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "best_val_acc": best_val_acc,
                "class_names": class_names,
                "args": vars(args),
                "history": history,
            }, output_dir / "best.pth")
            print(f"  >> 保存最佳模型 (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop:
                print(f"\n早停! 连续 {args.early_stop} 轮未提升")
                break

    # 测试
    print("\n加载最佳模型测试...")
    best_ckpt = torch.load(output_dir / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_loss, test_acc, test_class_accs = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    print("\n逐类准确率:")
    for idx, acc in sorted(test_class_accs.items()):
        print(f"  {idx_to_class[idx]}: {acc:.2f}%")

    # 保存配置
    config = {
        "method": "kgpt",
        "fusion": args.fusion,
        "hidden_dim": args.hidden_dim,
        "model": args.model,
        "epochs": args.epochs,
        "actual_epochs": epoch + 1,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_class_accs": {idx_to_class[idx]: acc for idx, acc in test_class_accs.items()},
        "class_sample_counts": dict(zip(class_names, class_sample_counts)),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"\n训练完成! 最佳验证准确率: {best_val_acc:.2f}%")
    print(f"测试集准确率: {test_acc:.2f}%")
    print(f"模型保存: {output_dir}")


if __name__ == "__main__":
    main()
