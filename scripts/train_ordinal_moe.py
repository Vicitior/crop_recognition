# -*- coding: utf-8 -*-
"""
Combined Training Script: Ordinal Loss + MoE-LoRA
创新点 1 + 创新点 2 的组合实验脚本

基于 train_clip_v2.py 改造，新增：
  1. 序数约束损失（高斯软标签 + EMD）替代标准 CrossEntropy
  2. MoE-LoRA 替代单个 LoRA 适配器
  3. 支持消融实验：可单独开启/关闭每个创新点

用法:
    # 同时启用 Ordinal Loss + MoE-LoRA（完整创新）
    python scripts/train_ordinal_moe.py --model openai/clip-vit-large-patch14-336

    # 消融实验：仅 Ordinal Loss（无 MoE）
    python scripts/train_ordinal_moe.py --model openai/clip-vit-large-patch14-336 --no-moe

    # 消融实验：仅 MoE-LoRA（无序数损失，用标准 CE）
    python scripts/train_ordinal_moe.py --model openai/clip-vit-large-patch14-336 --loss-type ce

    # 消融实验：仅高斯软标签（无 EMD）
    python scripts/train_ordinal_moe.py --model openai/clip-vit-large-patch14-336 --loss-type gaussian

    # 调参：高斯带宽 sigma
    python scripts/train_ordinal_moe.py --model openai/clip-vit-large-patch14-336 --sigma 1.2
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
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.growth_stages import CLASS_MAP, CROP_STAGE_ORDINAL, CROP_NUM_STAGES, CROP_INFO
from models.ordinal_loss import CombinedOrdinalLoss, OrdinalGaussianLoss, EarthMoversDistanceLoss


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
# 数据集
# ============================================================

class CropStageDataset(Dataset):
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

        # 为每个样本预计算作物 ID（用于 MoE 路由）
        self.crop_ids = []
        for _, label in self.samples:
            cls_name = self.idx_to_class[label]
            # 从类别名推断作物：cotton_xxx -> 0, corn_xxx -> 1, wheat_xxx -> 2
            crop = cls_name.split("_")[0]
            crop_map = {"cotton": 0, "corn": 1, "wheat": 2}
            self.crop_ids.append(crop_map.get(crop, 0))

        print(f"[{split}] {self.num_classes} 类, {len(self.samples)} 张图片")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        crop_id = self.crop_ids[idx]
        return image, label, crop_id


# ============================================================
# CLIP + MoE-LoRA 模型
# ============================================================

class CLIPWithMoELoRA(nn.Module):
    """CLIP 模型 + MoE-LoRA 适配器 + 分类头"""

    def __init__(self, clip_model, num_classes, num_experts=4, num_shared=2,
                 lora_rank=8, lora_alpha=16, num_crops=1, img_size=224):
        super().__init__()
        self.clip_model = clip_model

        # 从 config 获取特征维度（避免 dummy forward，MoE 层太慢）
        config = clip_model.config
        if hasattr(config, 'projection_dim'):
            self.feat_dim = config.projection_dim
        elif hasattr(config, 'vision_config'):
            self.feat_dim = config.vision_config.hidden_size
        else:
            self.feat_dim = 768

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

        self.moe_layers = []
        self.num_experts = num_experts

    def forward(self, pixel_values, crop_id=None):
        """
        前向传播。MoE-LoRA 已经嵌入到 clip_model 的注意力层中。
        crop_id 用于门控路由（可选）。
        """
        features = self.clip_model.get_image_features(pixel_values=pixel_values)
        if isinstance(features, torch.Tensor):
            pass
        elif hasattr(features, 'pooler_output'):
            features = features.pooler_output
        else:
            features = features.last_hidden_state[:, 0, :]
        return self.classifier(features)

    def get_moe_aux_loss(self):
        """汇总所有 MoE 层的辅助损失（通过 get_aux_loss() 获取）"""
        from models.moe_lora import get_moe_aux_loss
        return get_moe_aux_loss(self.clip_model)


# ============================================================
# 标准 LoRA（用于 --no-moe 消融实验）
# ============================================================

class LoRALinear(nn.Module):
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
        return self.original(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


def apply_standard_lora(model, rank=16, alpha=32):
    """与 train_clip_v2.py 完全一致的 LoRA 实现"""
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
    print(f"Applied standard LoRA to {count} layers (rank={rank}, alpha={alpha})")
    return model


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

def train_epoch(model, dataloader, criterion, optimizer, device,
                mixup_alpha=0.4, aux_loss_weight=0.01, use_amp=False, scaler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    num_batches = len(dataloader)
    t_start = time.time()
    print(f"  train_epoch: {num_batches} batches, 开始迭代...", flush=True)

    for batch_idx, (images, labels, crop_ids) in enumerate(dataloader):
        if batch_idx == 0:
            print(f"  第一个batch加载完成, shape={images.shape}", flush=True)
        images, labels, crop_ids = images.to(device), labels.to(device), crop_ids.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=use_amp):
            # Mixup
            if mixup_alpha > 0 and random.random() < 0.5:
                mixed_images, y_a, y_b, lam = mixup_data(images, labels, mixup_alpha)
                outputs = model(mixed_images)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            # MoE 辅助损失（仅在使用MoE时计算）
            if hasattr(model, 'moe_layers') and model.moe_layers:
                moe_aux = model.get_moe_aux_loss()
                loss = loss + aux_loss_weight * moe_aux

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 每10个batch打印进度
        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            elapsed = time.time() - t_start
            avg_per_batch = elapsed / (batch_idx + 1)
            remaining = avg_per_batch * (num_batches - batch_idx - 1)
            cur_acc = 100. * correct / total
            cur_loss = total_loss / (batch_idx + 1)
            print(f"  [{batch_idx+1}/{num_batches}] loss={cur_loss:.4f} acc={cur_acc:.1f}% "
                  f"| {avg_per_batch:.1f}s/batch, ETA {remaining/60:.1f}min", flush=True)

    return total_loss / len(dataloader), 100. * correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels, crop_ids in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(dataloader), 100. * correct / total


@torch.no_grad()
def evaluate_per_class(model, dataloader, device, class_names):
    """逐类准确率评估"""
    model.eval()
    class_correct = {}
    class_total = {}

    for images, labels, crop_ids in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)

        for i in range(labels.size(0)):
            cls = class_names[labels[i].item()]
            class_total[cls] = class_total.get(cls, 0) + 1
            if predicted[i] == labels[i]:
                class_correct[cls] = class_correct.get(cls, 0) + 1

    return {cls: 100. * class_correct.get(cls, 0) / class_total[cls]
            for cls in class_total}


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ordinal Loss + MoE-LoRA 联合训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
消融实验示例:
  # 完整创新 (Ordinal + MoE)
  python train_ordinal_moe.py --model openai/clip-vit-large-patch14-336

  # 仅序数损失 (无 MoE，用标准 LoRA)
  python train_ordinal_moe.py --model openai/clip-vit-large-patch14-336 --no-moe

  # 仅 MoE (标准 CE 损失)
  python train_ordinal_moe.py --model openai/clip-vit-large-patch14-336 --loss-type ce

  # 仅高斯软标签 (无 EMD)
  python train_ordinal_moe.py --model openai/clip-vit-large-patch14-336 --loss-type gaussian

  # 仅 EMD (无高斯软标签)
  python train_ordinal_moe.py --model openai/clip-vit-large-patch14-336 --loss-type emd

  # 基线 (标准 LoRA + CE，复现 V2)
  python train_ordinal_moe.py --model openai/clip-vit-large-patch14-336 --no-moe --loss-type ce
        """)

    # 模型参数
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--model", default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--early-stop", type=int, default=10)
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    parser.add_argument("--random-erasing", type=float, default=0.2)
    parser.add_argument("--amp", action="store_true", help="启用混合精度训练 (节省显存)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None)

    # 创新点参数
    parser.add_argument("--loss-type", choices=["combined", "gaussian", "emd", "ce"],
                        default="combined", help="损失函数类型")
    parser.add_argument("--sigma", type=float, default=1.0, help="高斯软标签带宽")
    parser.add_argument("--alpha", type=float, default=0.5, help="CE 在高斯损失中的权重")
    parser.add_argument("--beta", type=float, default=1.0, help="EMD 在组合损失中的权重")
    parser.add_argument("--no-moe", action="store_true", help="禁用 MoE-LoRA，使用标准 LoRA")
    parser.add_argument("--num-experts", type=int, default=4, help="MoE 专家总数")
    parser.add_argument("--num-shared", type=int, default=2, help="共享专家数")
    parser.add_argument("--aux-loss-weight", type=float, default=0.01, help="MoE 辅助损失权重")

    args = parser.parse_args()

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" \
        else torch.device(args.device)
    print(f"设备: {device}")

    # 输出目录
    experiment_name = "ordinal_moe"
    if args.no_moe:
        experiment_name = "ordinal_only"
    if args.loss_type == "ce":
        experiment_name = experiment_name.replace("ordinal", "moe_only") if not args.no_moe else "baseline_lora"

    if args.output_dir is None:
        model_short = args.model.split("/")[-1]
        args.output_dir = f"saved_models/clip/{model_short}-{experiment_name}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 数据增强
    img_size = 336 if "336" in args.model else 224
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=args.random_erasing, scale=(0.02, 0.2))
    ])
    val_transform = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 数据集
    train_dataset = CropStageDataset(args.data_dir, train_transform, "train")
    val_dataset = CropStageDataset(args.data_dir, val_transform, "val")
    test_dataset = CropStageDataset(args.data_dir, val_transform, "test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)

    num_classes = train_dataset.num_classes
    class_names = list(train_dataset.class_to_idx.keys())
    print(f"类别: {num_classes} 个 — {class_names}")

    # 判断作物数
    crop_types = set(cls.split("_")[0] for cls in class_names)
    num_crops = len(crop_types)
    print(f"作物: {num_crops} 种 — {crop_types}")

    # ============================================================
    # 构建模型
    # ============================================================
    from transformers import CLIPModel, CLIPProcessor

    print(f"加载 CLIP: {args.model}")
    clip_model = CLIPModel.from_pretrained(args.model)
    print("模型加载完成")

    if args.no_moe:
        clip_model = apply_standard_lora(clip_model, rank=args.lora_rank, alpha=args.lora_alpha)
        print("LoRA 应用完成，创建分类模型...")
        model = CLIPWithMoELoRA(clip_model, num_classes, img_size=img_size)
        print("分类模型创建完成")
    else:
        from models.moe_lora import apply_moe_lora
        clip_model, moe_layers = apply_moe_lora(
            clip_model,
            num_experts=args.num_experts,
            num_shared=args.num_shared,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            num_crops=num_crops
        )
        print("MoE-LoRA 应用完成，创建分类模型...")
        model = CLIPWithMoELoRA(clip_model, num_classes, num_experts=args.num_experts,
                                 num_shared=args.num_shared, img_size=img_size)
        model.moe_layers = moe_layers
        print("分类模型创建完成")

    print("移动模型到 GPU...")
    model = model.to(device)
    print("模型已就位")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")

    # ============================================================
    # 损失函数
    # ============================================================
    if args.loss_type == "ce":
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        print("损失: CrossEntropy (label_smoothing=0.1)")
    elif args.loss_type == "gaussian":
        criterion = OrdinalGaussianLoss(num_classes, sigma=args.sigma, alpha=args.alpha)
        print(f"损失: OrdinalGaussianLoss (sigma={args.sigma}, alpha={args.alpha})")
    elif args.loss_type == "emd":
        criterion = EarthMoversDistanceLoss(num_classes)
        print("损失: EarthMoversDistanceLoss")
    else:  # combined
        criterion = CombinedOrdinalLoss(num_classes, sigma=args.sigma,
                                         alpha=args.alpha, beta=args.beta)
        print(f"损失: CombinedOrdinalLoss (sigma={args.sigma}, alpha={args.alpha}, beta={args.beta})")

    # ============================================================
    # 优化器
    # ============================================================
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params_list, lr=args.lr, weight_decay=1e-4)
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs, args.epochs)

    # AMP 混合精度
    use_amp = args.amp
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("已启用混合精度训练 (AMP)")

    # ============================================================
    # 训练
    # ============================================================
    start_epoch = 0
    best_val_acc = 0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        best_val_acc = ckpt.get("best_val_acc", 0)
        start_epoch = ckpt.get("epoch", 0)
        print(f"恢复自 epoch {start_epoch}, best_val_acc={best_val_acc:.2f}%")

    print(f"\n{'='*70}", flush=True)
    print(f" 开始训练: {experiment_name}", flush=True)
    print(f" Epochs={args.epochs}, LR={args.lr}, LoRA rank={args.lora_rank}", flush=True)
    print(f" MoE={not args.no_moe}, Loss={args.loss_type}, AMP={use_amp}", flush=True)
    print(f"{'='*70}\n", flush=True)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch [{epoch+1}/{args.epochs}] 开始训练...", flush=True)

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            args.mixup_alpha, args.aux_loss_weight, use_amp, scaler
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.2e} | {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "best_val_acc": best_val_acc,
                "class_names": class_names,
                "history": history,
                "args": vars(args)
            }, output_dir / "best.pth")
            print(f"  >> 保存最佳模型 (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop:
                print(f"\n早停! 连续 {args.early_stop} 轮未提升")
                break

    # ============================================================
    # 测试
    # ============================================================
    print("\n加载最佳模型进行测试...")
    best_ckpt = torch.load(output_dir / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"测试集: Loss={test_loss:.4f}, Acc={test_acc:.2f}%")

    # 逐类准确率
    per_class = evaluate_per_class(model, test_loader, device, class_names)
    print("\n逐类准确率:")
    for cls, acc in per_class.items():
        print(f"  {cls}: {acc:.2f}%")

    # ============================================================
    # 保存结果
    # ============================================================
    config = {
        "experiment": experiment_name,
        "model": args.model,
        "loss_type": args.loss_type,
        "sigma": args.sigma,
        "alpha": args.alpha,
        "beta": args.beta,
        "use_moe": not args.no_moe,
        "num_experts": args.num_experts,
        "num_shared": args.num_shared,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "actual_epochs": epoch + 1,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "per_class_acc": per_class,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "timestamp": datetime.now().isoformat()
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # 训练曲线
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(history["train_loss"], label="Train")
        ax1.plot(history["val_loss"], label="Val")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title(f"Loss ({experiment_name})")
        ax1.legend()

        ax2.plot(history["train_acc"], label="Train")
        ax2.plot(history["val_acc"], label="Val")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title(f"Accuracy ({experiment_name})")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(output_dir / "training_curves.png", dpi=150)
    except ImportError:
        pass

    print(f"\n训练完成! 最佳验证准确率: {best_val_acc:.2f}%")
    print(f"测试集准确率: {test_acc:.2f}%")
    print(f"模型保存: {output_dir}")


if __name__ == "__main__":
    main()
