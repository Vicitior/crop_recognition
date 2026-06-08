# -*- coding: utf-8 -*-
"""
创新点 3 训练脚本：边缘蒸馏 + 双频特征对齐
将 CLIP 大模型的知识蒸馏到轻量级学生模型（MobileNetV3 / EfficientNet-B0），
结合双频特征对齐滤除背景噪声。

用法:
    # 使用 EfficientNet-B0 作为学生模型
    python scripts/train_distillation.py --student efficientnet_b0

    # 使用 MobileNetV3-Large
    python scripts/train_distillation.py --student mobilenet_v3_large

    # 使用训练好的 KGPT 模型作为教师
    python scripts/train_distillation.py --teacher saved_models/clip/clip-large-336-kgpt-gating/best.pth
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
from torchvision import transforms, models
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.frequency_filter import DualFrequencyAlignment, DualFrequencyFilter


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
# 教师模型（CLIP + 分类头）
# ============================================================

class CLIPTeacher(nn.Module):
    """CLIP 教师模型，提取特征和 logits"""

    def __init__(self, clip_model, num_classes, checkpoint_path=None):
        super().__init__()
        self.clip_model = clip_model

        # 获取特征维度
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat = clip_model.get_image_features(pixel_values=dummy)
            if isinstance(feat, torch.Tensor):
                self.feat_dim = feat.shape[-1]
            else:
                self.feat_dim = feat.pooler_output.shape[-1] if hasattr(feat, 'pooler_output') else 768

        # 分类头
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

        # 加载预训练权重
        if checkpoint_path and os.path.isfile(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.load_state_dict(ckpt["model_state_dict"], strict=False)
            print(f"教师模型加载自: {checkpoint_path}")

        # 冻结教师模型
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, images, return_features=False):
        features = self.clip_model.get_image_features(pixel_values=images)
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output if hasattr(features, 'pooler_output') else features.last_hidden_state[:, 0, :]

        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


# ============================================================
# 学生模型
# ============================================================

class LightweightStudent(nn.Module):
    """轻量级学生模型"""

    def __init__(self, backbone_name, num_classes, pretrained=True):
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
            self.feat_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
            self.backbone = backbone

        elif backbone_name == "mobilenet_v3_large":
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            backbone = models.mobilenet_v3_large(weights=weights)
            self.feat_dim = backbone.classifier[-1].in_features
            backbone.classifier = nn.Identity()
            self.backbone = backbone

        elif backbone_name == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            backbone = models.mobilenet_v3_small(weights=weights)
            self.feat_dim = backbone.classifier[-1].in_features
            backbone.classifier = nn.Identity()
            self.backbone = backbone

        else:
            raise ValueError(f"不支持的骨干网络: {backbone_name}")

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, return_features=False):
        features = self.backbone(images)
        if features.dim() > 2:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)

        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


# ============================================================
# 蒸馏损失
# ============================================================

class DistillationLoss(nn.Module):
    """
    知识蒸馏损失 = KL散度 + 特征对齐 + 双频对齐
    """

    def __init__(self, temperature=4.0, alpha=0.5, beta=0.3, gamma=0.2):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # KL 散度权重
        self.beta = beta    # 特征对齐权重
        self.gamma = gamma  # 双频对齐权重

    def forward(self, student_logits, teacher_logits, student_feat, teacher_feat,
                labels, criterion_ce, freq_alignment=None):
        # 1. 标准交叉熵损失
        ce_loss = criterion_ce(student_logits, labels)

        # 2. KL 散度蒸馏损失
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        kl_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)

        # 3. 特征对齐损失（MSE）
        # 维度对齐
        if student_feat.shape[-1] != teacher_feat.shape[-1]:
            if not hasattr(self, '_feat_proj') or self._feat_proj is None:
                self._feat_proj = nn.Linear(student_feat.shape[-1], teacher_feat.shape[-1]).to(student_feat.device)
            student_feat_aligned = self._feat_proj(student_feat)
        else:
            student_feat_aligned = student_feat

        if student_feat_aligned.shape != teacher_feat.shape:
            student_feat_aligned = F.adaptive_avg_pool1d(
                student_feat_aligned.unsqueeze(1), teacher_feat.shape[1]
            ).squeeze(1) if student_feat_aligned.dim() == 2 else student_feat_aligned

        feat_loss = F.mse_loss(student_feat_aligned, teacher_feat.detach())

        # 4. 双频对齐损失
        freq_loss = torch.tensor(0.0, device=student_logits.device)
        if freq_alignment is not None:
            freq_loss, _, _ = freq_alignment.compute_alignment_loss(student_feat, teacher_feat.detach())

        # 总损失
        total = ce_loss + self.alpha * kl_loss + self.beta * feat_loss + self.gamma * freq_loss

        return total, {
            "ce": ce_loss.item(),
            "kl": kl_loss.item(),
            "feat": feat_loss.item(),
            "freq": freq_loss.item() if isinstance(freq_loss, torch.Tensor) else freq_loss,
        }


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

def train_epoch(student, teacher, dataloader, distill_criterion, criterion_ce,
                optimizer, device, freq_alignment=None):
    student.train()
    total_loss = 0
    total_losses = defaultdict(float)
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # 教师推理
        with torch.no_grad():
            teacher_logits, teacher_feat = teacher(images, return_features=True)

        # 学生推理
        student_logits, student_feat = student(images, return_features=True)

        # 蒸馏损失
        loss, loss_dict = distill_criterion(
            student_logits, teacher_logits, student_feat, teacher_feat,
            labels, criterion_ce, freq_alignment
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        for k, v in loss_dict.items():
            total_losses[k] += v

        _, predicted = student_logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    n = len(dataloader)
    avg_losses = {k: v / n for k, v in total_losses.items()}
    return total_loss / n, 100. * correct / total, avg_losses


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
    parser = argparse.ArgumentParser(description="知识蒸馏 + 双频特征对齐")
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--teacher-model", default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--teacher-ckpt", default=None, help="教师模型 checkpoint 路径")
    parser.add_argument("--student", choices=["efficientnet_b0", "mobilenet_v3_large", "mobilenet_v3_small"],
                        default="efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--early-stop", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=4.0, help="蒸馏温度")
    parser.add_argument("--alpha", type=float, default=0.5, help="KL 散度权重")
    parser.add_argument("--beta", type=float, default=0.3, help="特征对齐权重")
    parser.add_argument("--gamma", type=float, default=0.2, help="双频对齐权重")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto" else "cpu")
    print(f"设备: {device}")

    # 输出目录
    if args.output_dir is None:
        args.output_dir = f"saved_models/clip/distill-{args.student}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_size = 224  # 轻量级模型用 224

    # 数据增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
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

    # 教师模型
    print(f"加载教师模型: {args.teacher_model}")
    from transformers import CLIPModel
    clip_model = CLIPModel.from_pretrained(args.teacher_model)
    teacher = CLIPTeacher(clip_model, num_classes, args.teacher_ckpt).to(device)
    teacher.eval()

    # 学生模型
    print(f"创建学生模型: {args.student}")
    student = LightweightStudent(args.student, num_classes, pretrained=True).to(device)

    total_params = sum(p.numel() for p in student.parameters())
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"学生模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")

    # 双频对齐
    freq_alignment = DualFrequencyAlignment(
        feat_dim=max(teacher.feat_dim, student.feat_dim)
    ).to(device)
    freq_alignment.build_projection(student.feat_dim, teacher.feat_dim)

    # 损失函数
    criterion_ce = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(class_weights).to(device),
        label_smoothing=args.label_smoothing
    )
    distill_criterion = DistillationLoss(
        temperature=args.temperature, alpha=args.alpha,
        beta=args.beta, gamma=args.gamma
    )

    # 优化器
    optimizer = optim.AdamW(
        list(student.parameters()) + list(freq_alignment.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs, args.epochs)

    # 训练
    best_val_acc = 0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_class_acc": [], "lr": []}

    print(f"\n{'='*70}")
    print(f" 知识蒸馏: {args.teacher_model} -> {args.student}")
    print(f" T={args.temperature}, alpha={args.alpha}, beta={args.beta}, gamma={args.gamma}")
    print(f"{'='*70}\n")

    for epoch in range(args.epochs):
        t0 = time.time()
        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']

        train_loss, train_acc, loss_dict = train_epoch(
            student, teacher, train_loader, distill_criterion, criterion_ce,
            optimizer, device, freq_alignment
        )
        val_loss, val_acc, class_accs = evaluate(student, val_loader, criterion_ce, device)

        elapsed = time.time() - t0
        class_acc_names = {idx_to_class[idx]: acc for idx, acc in class_accs.items()}

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_class_acc"].append(class_acc_names)
        history["lr"].append(current_lr)

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Loss: {train_loss:.4f} (ce:{loss_dict['ce']:.3f} kl:{loss_dict['kl']:.3f} "
              f"feat:{loss_dict['feat']:.3f} freq:{loss_dict['freq']:.3f}) | "
              f"Acc: {train_acc:.2f}% | Val: {val_acc:.2f}% | "
              f"LR: {current_lr:.2e} | {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": student.state_dict(),
                "best_val_acc": best_val_acc,
                "class_names": class_names,
                "args": vars(args),
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
    student.load_state_dict(best_ckpt["model_state_dict"])

    test_loss, test_acc, test_class_accs = evaluate(student, test_loader, criterion_ce, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    print("\n逐类准确率:")
    for idx, acc in sorted(test_class_accs.items()):
        print(f"  {idx_to_class[idx]}: {acc:.2f}%")

    # 模型大小
    model_size_mb = os.path.getsize(output_dir / "best.pth") / (1024 * 1024)
    print(f"\n模型大小: {model_size_mb:.2f} MB")

    config = {
        "method": "distillation",
        "teacher": args.teacher_model,
        "student": args.student,
        "temperature": args.temperature,
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_class_accs": {idx_to_class[idx]: acc for idx, acc in test_class_accs.items()},
        "model_size_mb": round(model_size_mb, 2),
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
