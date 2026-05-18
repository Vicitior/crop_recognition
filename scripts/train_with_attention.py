"""
使用注意力机制增强的CLIP微调脚本
集成SE、CBAM等注意力模块提升特征区分能力

用法:
    python scripts/train_with_attention.py --model openai/clip-vit-large-patch14-336 --attention se
    python scripts/train_with_attention.py --model openai/clip-vit-large-patch14-336 --attention cbam
    python scripts/train_with_attention.py --model openai/clip-vit-large-patch14-336 --attention multihead
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


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
# CLIP模型包装（带注意力）
# ============================================================

class CLIPWithAttentionClassifier(nn.Module):
    """CLIP模型 + 注意力增强分类头"""

    def __init__(self, clip_model, num_classes, attention_type="se", img_size=224):
        super().__init__()
        self.clip_model = clip_model

        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            feat = clip_model.get_image_features(pixel_values=dummy)
            if isinstance(feat, torch.Tensor):
                self.feat_dim = feat.shape[-1]
            else:
                self.feat_dim = feat.pooler_output.shape[-1] if hasattr(feat, 'pooler_output') else 768

        # 使用注意力增强分类头
        from models.attention import EnhancedClassifier
        self.classifier = EnhancedClassifier(
            self.feat_dim, num_classes,
            attention_type=attention_type,
            reduction=16
        )

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
# 数据集
# ============================================================

class CropStageDataset(Dataset):
    """作物生长阶段数据集"""

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
# 学习率调度器
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
            factor = (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1 + np.cos(np.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = max(self.min_lr, base_lr * factor)


# ============================================================
# 训练函数
# ============================================================

def train_epoch(model, dataloader, criterion, optimizer, device, mixup_alpha=0.4):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

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
    """评估模型"""
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
    parser = argparse.ArgumentParser(description="注意力增强CLIP训练")
    parser.add_argument("--data-dir", default="dataset", help="数据集目录")
    parser.add_argument("--model", default="openai/clip-vit-large-patch14-336",
                        help="CLIP模型名称")
    parser.add_argument("--attention", choices=["se", "cbam", "multihead", "none"],
                        default="se", help="注意力类型")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=16, help="批大小")
    parser.add_argument("--lr", type=float, default=5e-4, help="学习率")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA秩")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="预热轮数")
    parser.add_argument("--early-stop", type=int, default=10, help="早停耐心值")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="标签平滑")
    parser.add_argument("--output-dir", default=None, help="输出目录")
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
        args.output_dir = f"saved_models/clip/{model_short}-attention-{args.attention}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载CLIP模型
    print(f"加载CLIP模型: {args.model}")
    from transformers import CLIPModel, CLIPProcessor

    clip_model = CLIPModel.from_pretrained(args.model)
    processor = CLIPProcessor.from_pretrained(args.model)

    # 确定图片大小
    img_size = 336 if "336" in args.model else 224
    print(f"使用图片大小: {img_size}x{img_size}")

    # 数据增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)

    num_classes = train_dataset.num_classes
    class_names = train_dataset.class_names
    print(f"类别数: {num_classes}, 类别: {class_names}")

    # 应用LoRA
    print(f"应用LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})")
    clip_model = apply_lora(clip_model, rank=args.lora_rank, alpha=args.lora_alpha)

    # 构建模型
    print(f"使用注意力类型: {args.attention}")
    model = CLIPWithAttentionClassifier(
        clip_model, num_classes,
        attention_type=args.attention,
        img_size=img_size
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}, 可训练: {trainable_params:,}")

    # 损失函数
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # 优化器
    lora_params = [p for n, p in model.named_parameters()
                   if "lora_" in n or "classifier" in n]
    optimizer = optim.AdamW(lora_params, lr=args.lr, weight_decay=1e-4)

    # 学习率调度器
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs, args.epochs)

    # 训练
    best_val_acc = 0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "val_class_acc": [], "lr": []}

    print(f"\n开始训练 - 注意力增强 ({args.attention})")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}, LoRA rank: {args.lora_rank}")
    print("=" * 70)

    for epoch in range(args.epochs):
        start_time = time.time()

        # 更新学习率
        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']

        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 评估
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
                "method": f"attention_{args.attention}",
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

    # 保存配置
    config = {
        "method": f"attention_{args.attention}",
        "model": args.model,
        "attention_type": args.attention,
        "epochs": args.epochs,
        "actual_epochs": epoch + 1,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "num_classes": num_classes,
        "class_names": class_names,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_class_accs": {idx_to_class[idx]: acc for idx, acc in test_class_accs.items()},
        "trainable_params": trainable_params,
        "total_params": total_params,
        "timestamp": datetime.now().isoformat()
    }

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"\n训练完成！最佳验证准确率: {best_val_acc:.2f}%")
    print(f"测试集准确率: {test_acc:.2f}%")
    print(f"模型保存在: {output_dir}")


if __name__ == "__main__":
    main()
