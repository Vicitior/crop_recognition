"""
CLIP微调脚本 V2 - 优化版
相比V1的改进：
  1. Mixup数据增强
  2. RandomErasing
  3. 标签平滑 (Label Smoothing)
  4. 学习率预热 (Warmup) + Cosine Annealing
  5. Early Stopping
  6. 增大LoRA rank (默认16)
  7. 更深的分类头 (BatchNorm + 两层隐藏层)
  8. 更多训练轮数 (默认50)
  9. 支持从已有checkpoint继续训练

用法:
    # 推荐：LoRA + ViT-L/14@336
    python scripts/train_clip_v2.py --model openai/clip-vit-large-patch14-336

    # 使用更大的LoRA rank
    python scripts/train_clip_v2.py --model openai/clip-vit-large-patch14-336 --lora-rank 32

    # 从之前的checkpoint继续训练
    python scripts/train_clip_v2.py --model openai/clip-vit-large-patch14-336 --resume saved_models/clip/clip-large-336/best.pth
"""

import os
import sys
import argparse
import json
import time
import random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# Mixup / CutMix
# ============================================================

def mixup_data(x, y, alpha=0.4):
    """Mixup数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup损失计算"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
# Focal Loss
# ============================================================

class FocalLoss(nn.Module):
    """Focal Loss - 处理类别不平衡问题

    通过降低易分类样本的权重，让模型专注于难分类样本
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # 类别权重
        self.gamma = gamma  # 聚焦参数，越大越关注难样本
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ============================================================
# 数据集
# ============================================================

class CropStageDataset(Dataset):
    """作物生长阶段数据集"""

    def __init__(self, data_dir, transform=None, split="train"):
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
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
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
# CLIP模型包装
# ============================================================

class CLIPWithClassifier(nn.Module):
    """CLIP模型 + 增强分类头"""

    def __init__(self, clip_model, num_classes, model_type="clip", freeze_clip=True, img_size=224):
        super().__init__()
        self.clip_model = clip_model
        self.model_type = model_type
        self.freeze_clip = freeze_clip

        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            feat = clip_model.get_image_features(pixel_values=dummy)
            if isinstance(feat, torch.Tensor):
                self.feat_dim = feat.shape[-1]
            elif hasattr(feat, 'pooler_output'):
                self.feat_dim = feat.pooler_output.shape[-1]
            else:
                self.feat_dim = feat.last_hidden_state[:, 0, :].shape[-1]

        # 更深的分类头：两层隐藏层 + BatchNorm
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

        if freeze_clip:
            for param in clip_model.parameters():
                param.requires_grad = False

    def freeze_backbone(self):
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.clip_model.parameters():
            param.requires_grad = True

    def forward(self, pixel_values):
        features = self.clip_model.get_image_features(pixel_values=pixel_values)
        if isinstance(features, torch.Tensor):
            pass
        elif hasattr(features, 'pooler_output'):
            features = features.pooler_output
        else:
            features = features.last_hidden_state[:, 0, :]
        return self.classifier(features)


# ============================================================
# LoRA实现
# ============================================================

class LoRALinear(nn.Module):
    """LoRA线性层"""

    def __init__(self, original_linear, rank=16, alpha=32):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.alpha = alpha

        in_dim = original_linear.in_features
        out_dim = original_linear.out_features

        self.lora_A = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        self.scaling = alpha / rank

        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x):
        result = self.original(x)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return result + lora_out


def apply_lora(model, rank=16, alpha=32):
    """给模型添加LoRA适配器"""
    target_modules = ["q_proj", "v_proj", "k_proj", "out_proj",
                      "fc1", "fc2", "query", "value", "key"]

    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(target in name for target in target_modules):
                parts = name.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)

                lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
                setattr(parent, parts[-1], lora_layer)
                count += 1

    print(f"已添加 {count} 个LoRA适配器 (rank={rank}, alpha={alpha})")
    return model


# ============================================================
# 学习率调度器（带Warmup）
# ============================================================

class WarmupCosineScheduler:
    """带预热的余弦退火调度器"""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # 线性预热
            factor = (epoch + 1) / self.warmup_epochs
        else:
            # 余弦退火
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1 + np.cos(np.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = max(self.min_lr, base_lr * factor)


# ============================================================
# 训练函数
# ============================================================

def train_epoch(model, dataloader, criterion, optimizer, device, mixup_alpha=0.4):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # Mixup
        if mixup_alpha > 0 and random.random() < 0.5:
            mixed_images, y_a, y_b, lam = mixup_data(images, labels, mixup_alpha)
            outputs = model(mixed_images)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
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
        images = images.to(device)
        labels = labels.to(device)

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


def main():
    parser = argparse.ArgumentParser(description="CLIP微调训练 V2 (优化版)")
    parser.add_argument("--data-dir", default="dataset", help="数据集目录")
    parser.add_argument("--model", default="openai/clip-vit-large-patch14-336",
                        help="CLIP模型名称")
    parser.add_argument("--method", choices=["linear", "lora", "full"],
                        default="lora", help="微调方法")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=16, help="批大小")
    parser.add_argument("--lr", type=float, default=5e-4, help="学习率")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA秩")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="预热轮数")
    parser.add_argument("--early-stop", type=int, default=10, help="早停耐心值(轮数)")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="标签平滑系数")
    parser.add_argument("--mixup-alpha", type=float, default=0.4, help="Mixup alpha (0=禁用)")
    parser.add_argument("--random-erasing", type=float, default=0.2, help="RandomErasing概率 (0=禁用)")
    parser.add_argument("--use-focal-loss", action="store_true", help="使用Focal Loss替代CrossEntropyLoss")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal Loss的gamma参数")
    parser.add_argument("--output-dir", default=None, help="输出目录 (默认自动命名)")
    parser.add_argument("--resume", default=None, help="从checkpoint恢复训练")
    parser.add_argument("--device", default="auto", help="设备")
    args = parser.parse_args()

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")

    # 输出目录
    if args.output_dir is None:
        model_short = args.model.split("/")[-1]
        args.output_dir = f"saved_models/clip/{model_short}-v2"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载CLIP模型
    print(f"加载CLIP模型: {args.model}")
    from transformers import CLIPModel, CLIPProcessor, AutoModel, AutoProcessor

    model_type = "clip"
    if "siglip" in args.model.lower():
        model_type = "siglip"
        clip_model = AutoModel.from_pretrained(args.model)
        processor = AutoProcessor.from_pretrained(args.model)
    else:
        clip_model = CLIPModel.from_pretrained(args.model)
        processor = CLIPProcessor.from_pretrained(args.model)

    # 确定图片大小
    img_size = 336 if "336" in args.model else 224
    print(f"使用图片大小: {img_size}x{img_size}")

    # 数据增强（增强版）
    train_transform_list = [
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    if args.random_erasing > 0:
        train_transform_list.append(transforms.RandomErasing(p=args.random_erasing, scale=(0.02, 0.2)))

    train_transform = transforms.Compose(train_transform_list)

    val_transform = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    train_dataset = CropStageDataset(args.data_dir, train_transform, "train")
    val_dataset = CropStageDataset(args.data_dir, val_transform, "val")
    test_dataset = CropStageDataset(args.data_dir, val_transform, "test")

    num_classes = train_dataset.num_classes
    class_names = list(train_dataset.class_to_idx.keys())
    print(f"类别数: {num_classes}, 类别: {class_names}")

    # 统计类别分布，计算逆频率权重
    class_sample_counts = [0] * num_classes
    for _, label in train_dataset.samples:
        class_sample_counts[label] += 1
    print(f"各类别样本数: {dict(zip(class_names, class_sample_counts))}")

    # 逆频率权重：样本越少权重越大
    total_samples = sum(class_sample_counts)
    class_weights = []
    for c in class_sample_counts:
        if c > 0:
            class_weights.append(total_samples / (num_classes * c))
        else:
            class_weights.append(0.0)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    print(f"类别权重: {dict(zip(class_names, [f'{w:.2f}' for w in class_weights]))}")

    # WeightedRandomSampler：让少数类在每个epoch中被采样更多次
    sample_weights = [class_weights[label] for _, label in train_dataset.samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=sampler, num_workers=0, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)

    # 构建模型
    if args.method == "lora":
        print(f"应用LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})")
        clip_model = apply_lora(clip_model, rank=args.lora_rank, alpha=args.lora_alpha)

    model = CLIPWithClassifier(clip_model, num_classes, model_type, img_size=img_size)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}, 可训练: {trainable_params:,}")

    # 恢复训练
    start_epoch = 0
    best_val_acc = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_class_acc": [], "lr": []}

    if args.resume:
        print(f"从checkpoint恢复: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_val_acc = checkpoint.get("best_val_acc", 0)
        start_epoch = checkpoint.get("epoch", 0)
        if "history" in checkpoint:
            history = checkpoint["history"]
        print(f"从第 {start_epoch} 轮继续, 当前最佳验证准确率: {best_val_acc:.2f}%")
        # optimizer状态会在optimizer创建后恢复
        resume_optimizer_state = checkpoint.get("optimizer_state_dict", None)
    else:
        resume_optimizer_state = None

    # 损失函数
    if args.use_focal_loss:
        criterion = FocalLoss(alpha=class_weights_tensor, gamma=args.focal_gamma)
        print(f"使用 Focal Loss (gamma={args.focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=args.label_smoothing)
        print(f"使用 CrossEntropyLoss (label_smoothing={args.label_smoothing})")

    # 优化器
    if args.method == "linear":
        optimizer = optim.AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.method == "lora":
        lora_params = [p for n, p in model.named_parameters()
                       if "lora_" in n or "classifier" in n]
        optimizer = optim.AdamW(lora_params, lr=args.lr, weight_decay=1e-4)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 学习率调度器
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs, args.epochs)

    # 恢复optimizer状态（断点续训）
    if resume_optimizer_state is not None:
        try:
            optimizer.load_state_dict(resume_optimizer_state)
            print(f"  >> 已恢复optimizer状态")
        except Exception as e:
            print(f"  >> 恢复optimizer状态失败: {e}，使用新optimizer")

    # 训练
    patience_counter = 0

    print(f"\n开始训练 ({args.method}) - V2优化版")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}, LoRA rank: {args.lora_rank}")
    print(f"  Mixup: {args.mixup_alpha}, Label Smoothing: {args.label_smoothing}")
    print(f"  Early Stop: {args.early_stop} epochs patience")
    print("=" * 70)

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()

        # 更新学习率
        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']

        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                            optimizer, device, args.mixup_alpha)
        val_loss, val_acc, class_accs = evaluate(model, val_loader, criterion, device)
        idx_to_class = train_dataset.idx_to_class
        class_acc_names = {idx_to_class[idx]: acc for idx, acc in class_accs.items()}

        elapsed = time.time() - start_time

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_class_acc"].append(class_acc_names)
        history["lr"].append(current_lr)

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s")

        # 打印逐类准确率（关注少数类）
        minority_classes = [c for c, n in zip(class_names, class_sample_counts) if n < 30]
        if minority_classes:
            acc_strs = [f"{c}:{class_acc_names.get(c, 0):.1f}%" for c in minority_classes]
            print(f"  少数类准确率: {', '.join(acc_strs)}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "class_names": class_names,
                "method": args.method,
                "model_name": args.model,
                "history": history
            }
            torch.save(checkpoint, output_dir / "best.pth")
            print(f"  >> 保存最佳模型 (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop:
                print(f"\n早停! 验证准确率已连续 {args.early_stop} 轮未提升")
                break

        # 每个epoch都保存checkpoint（断点续训）
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "class_names": class_names,
            "method": args.method,
            "model_name": args.model,
            "history": history
        }
        torch.save(checkpoint, output_dir / "last.pth")

    # 保存最后一轮（兜底）
    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
        "class_names": class_names,
        "method": args.method,
        "model_name": args.model,
        "history": history
    }
    torch.save(checkpoint, output_dir / "last.pth")

    # 加载最佳模型进行测试
    print("\n加载最佳模型进行测试...")
    best_checkpoint = torch.load(output_dir / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    test_loss, test_acc, test_class_accs = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    # 打印测试集逐类准确率
    print("\n测试集逐类准确率:")
    for idx, acc in sorted(test_class_accs.items()):
        cls_name = idx_to_class[idx]
        print(f"  {cls_name}: {acc:.2f}%")

    # 保存训练曲线
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        ax1.plot(history["train_loss"], label="Train")
        ax1.plot(history["val_loss"], label="Val")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss Curve")
        ax1.legend()

        ax2.plot(history["train_acc"], label="Train")
        ax2.plot(history["val_acc"], label="Val")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Accuracy Curve")
        ax2.legend()

        ax3.plot(history["lr"], label="LR")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Learning Rate")
        ax3.set_title("Learning Rate Schedule")
        ax3.legend()

        plt.tight_layout()
        plt.savefig(output_dir / "training_curves.png", dpi=150)
        print(f"训练曲线已保存: {output_dir / 'training_curves.png'}")
    except ImportError:
        pass

    # 保存配置
    config = {
        "method": args.method,
        "model": args.model,
        "epochs": args.epochs,
        "actual_epochs": epoch + 1,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lora_rank": args.lora_rank if args.method == "lora" else None,
        "lora_alpha": args.lora_alpha if args.method == "lora" else None,
        "warmup_epochs": args.warmup_epochs,
        "early_stop_patience": args.early_stop,
        "label_smoothing": args.label_smoothing,
        "mixup_alpha": args.mixup_alpha,
        "random_erasing": args.random_erasing,
        "num_classes": num_classes,
        "class_names": class_names,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_class_accs": {idx_to_class[idx]: acc for idx, acc in test_class_accs.items()},
        "class_sample_counts": dict(zip(class_names, class_sample_counts)),
        "class_weights": dict(zip(class_names, [round(w, 4) for w in class_weights])),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "timestamp": datetime.now().isoformat(),
        "version": "v2"
    }

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"\n训练完成！最佳验证准确率: {best_val_acc:.2f}%")
    print(f"测试集准确率: {test_acc:.2f}%")
    print(f"模型保存在: {output_dir}")


if __name__ == "__main__":
    main()
