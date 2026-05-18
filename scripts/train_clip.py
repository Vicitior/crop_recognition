"""
CLIP微调脚本 - 支持Linear Probe、LoRA、Full Fine-tuning
用法:
    # Linear Probe（推荐数据少时使用）
    python scripts/train_clip.py --method linear --data-dir dataset

    # LoRA（推荐，平衡效果和效率）
    python scripts/train_clip.py --method lora --data-dir dataset

    # Full Fine-tuning（数据充足时）
    python scripts/train_clip.py --method full --data-dir dataset
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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


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

        # 扫描目录
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
    """CLIP模型 + 分类头"""

    def __init__(self, clip_model, num_classes, model_type="clip", freeze_clip=True, img_size=224):
        super().__init__()
        self.clip_model = clip_model
        self.model_type = model_type
        self.freeze_clip = freeze_clip

        # 获取特征维度
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            feat = clip_model.get_image_features(pixel_values=dummy)
            # 处理不同类型的输出
            if isinstance(feat, torch.Tensor):
                self.feat_dim = feat.shape[-1]
            elif hasattr(feat, 'pooler_output'):
                self.feat_dim = feat.pooler_output.shape[-1]
            else:
                self.feat_dim = feat.last_hidden_state[:, 0, :].shape[-1]

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
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
        # 处理不同类型的输出
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

    def __init__(self, original_linear, rank=8, alpha=16):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.alpha = alpha

        in_dim = original_linear.in_features
        out_dim = original_linear.out_features

        self.lora_A = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        self.scaling = alpha / rank

        # 冻结原始权重
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x):
        result = self.original(x)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return result + lora_out


def apply_lora(model, rank=8, alpha=16):
    """给模型添加LoRA适配器"""
    import re
    target_modules = ["q_proj", "v_proj", "k_proj", "out_proj",
                      "fc1", "fc2", "query", "value", "key"]

    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 检查是否是目标模块
            if any(target in name for target in target_modules):
                # 获取父模块
                parts = name.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)

                # 替换为LoRA层
                lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
                setattr(parent, parts[-1], lora_layer)
                count += 1

    print(f"已添加 {count} 个LoRA适配器 (rank={rank}, alpha={alpha})")
    return model


# ============================================================
# 训练函数
# ============================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
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

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(dataloader), 100. * correct / total


def main():
    parser = argparse.ArgumentParser(description="CLIP微调训练")
    parser.add_argument("--data-dir", default="dataset", help="数据集目录")
    parser.add_argument("--model", default="openai/clip-vit-base-patch32",
                        help="CLIP模型名称")
    parser.add_argument("--method", choices=["linear", "lora", "full"],
                        default="lora", help="微调方法")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=16, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA秩")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--freeze-epochs", type=int, default=5,
                        help="冻结骨干网络的轮数（linear/full模式）")
    parser.add_argument("--output-dir", default="saved_models/clip",
                        help="输出目录")
    parser.add_argument("--device", default="auto", help="设备")
    args = parser.parse_args()

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")

    # 输出目录
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

    # 数据增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    train_dataset = CropStageDataset(args.data_dir, train_transform, "train")
    val_dataset = CropStageDataset(args.data_dir, val_transform, "val")
    test_dataset = CropStageDataset(args.data_dir, val_transform, "test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)

    num_classes = train_dataset.num_classes
    class_names = list(train_dataset.class_to_idx.keys())
    print(f"类别数: {num_classes}, 类别: {class_names}")

    # 构建模型
    if args.method == "lora":
        print(f"应用LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})")
        clip_model = apply_lora(clip_model, rank=args.lora_rank, alpha=args.lora_alpha)

    model = CLIPWithClassifier(clip_model, num_classes, model_type, img_size=img_size)
    model = model.to(device)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}, 可训练: {trainable_params:,}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()

    if args.method == "linear":
        # 只训练分类头
        optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    elif args.method == "lora":
        # 只训练LoRA参数和分类头
        lora_params = [p for n, p in model.named_parameters()
                       if "lora_" in n or "classifier" in n]
        optimizer = optim.Adam(lora_params, lr=args.lr)
    else:
        # 全部参数
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 训练
    best_val_acc = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n开始训练 ({args.method})...")
    print("=" * 60)

    for epoch in range(args.epochs):
        start_time = time.time()

        # 两阶段训练（仅linear和full模式）
        if args.method in ["linear", "full"] and epoch == args.freeze_epochs:
            print(f"\n>> 解冻骨干网络，开始全模型微调")
            model.unfreeze_backbone()
            # 降低学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                            optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start_time

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"Time: {elapsed:.1f}s")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
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

    # 保存最后一轮
    checkpoint = {
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
        "class_names": class_names,
        "method": args.method,
        "model_name": args.model,
        "history": history
    }
    torch.save(checkpoint, output_dir / "last.pth")

    # 测试集评估
    print("\n" + "=" * 60)
    print("测试集评估:")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    # 保存训练曲线
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

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
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lora_rank": args.lora_rank if args.method == "lora" else None,
        "lora_alpha": args.lora_alpha if args.method == "lora" else None,
        "num_classes": num_classes,
        "class_names": class_names,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "timestamp": datetime.now().isoformat()
    }

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"\n训练完成！最佳验证准确率: {best_val_acc:.2f}%")
    print(f"模型保存在: {output_dir}")


if __name__ == "__main__":
    main()
