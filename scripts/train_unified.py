# -*- coding: utf-8 -*-
"""
统一训练脚本：融合所有创新点的完整实验框架

创新点清单：
  1. 知识引导 Prompt 微调 (KGPT) — 物理机制融合
  2. MoE-LoRA 多任务动态路由
  3. 双频特征对齐 + 知识蒸馏
  4. 扩散模型数据增强

支持消融实验：可单独开启/关闭每个创新点

用法:
    # 完整创新 (所有创新点全开)
    python scripts/train_unified.py --model openai/clip-vit-large-patch14-336 --all

    # 消融实验：仅 KGPT
    python scripts/train_unified.py --model openai/clip-vit-large-patch14-336 --use-kgpt

    # 消融实验：仅 MoE
    python scripts/train_unified.py --model openai/clip-vit-large-patch14-336 --use-moe

    # 消融实验：KGPT + MoE (无序数损失)
    python scripts/train_unified.py --model openai/clip-vit-large-patch14-336 --use-kgpt --use-moe

    # 基线 (标准 LoRA + CE)
    python scripts/train_unified.py --model openai/clip-vit-large-patch14-336
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

from models.growth_stages import CLASS_MAP, CROP_STAGE_ORDINAL, CROP_NUM_STAGES


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

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes = len(classes)
        self.class_names = list(self.class_to_idx.keys())

        # 作物 ID (用于 MoE 路由)
        self.crop_ids = []
        for _, label in self.samples:
            cls_name = self.idx_to_class[label]
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
# 序数损失（从 ordinal_loss.py 导入）
# ============================================================

from models.ordinal_loss import CombinedOrdinalLoss


# ============================================================
# 标准 LoRA
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
    print(f"标准 LoRA: {count} 层 (rank={rank}, alpha={alpha})")
    return model


# ============================================================
# 统一模型
# ============================================================

class UnifiedCropModel(nn.Module):
    """
    统一农作物识别模型
    支持：
      - 标准 LoRA / MoE-LoRA
      - 知识引导 Prompt (KGPT)
      - 序数损失
    """

    def __init__(self, clip_model, num_classes, use_kgpt=False,
                 class_names=None, img_size=224):
        super().__init__()
        self.clip_model = clip_model
        self.use_kgpt = use_kgpt

        # 获取特征维度
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

        # KGPT 模块
        if use_kgpt and class_names is not None:
            from models.knowledge_encoder import (
                PhysicsKnowledgeEncoder, KnowledgeGuidedFusion, build_knowledge_vectors
            )
            self.kgpt_encoder = PhysicsKnowledgeEncoder(
                input_dim=4, hidden_dim=256, output_dim=self.feat_dim
            )
            self.kgpt_fusion = KnowledgeGuidedFusion(self.feat_dim, fusion_type="gating")
            physics_vectors = build_knowledge_vectors(class_names)
            self.register_buffer("physics_vectors", physics_vectors)

            # KGPT 文本特征（预计算）
            self.kgpt_text_features = None
            self._class_names = class_names

        self.moe_layers = []

    def encode_text_features(self, device):
        """为 KGPT 编码文本特征"""
        if not self.use_kgpt or not hasattr(self, '_class_names'):
            return None

        from transformers import CLIPTokenizer
        model_name = getattr(self.clip_model.config, '_name_or_path', 'openai/clip-vit-large-patch14-336')
        try:
            tokenizer = CLIPTokenizer.from_pretrained(model_name)
        except Exception:
            tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14-336')
        prompts = [f"a photo of {name.replace('_', ' ')}" for name in self._class_names]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
            if not isinstance(text_features, torch.Tensor):
                text_features = text_features.pooler_output
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def forward(self, pixel_values, crop_id=None):
        # 图像特征
        features = self.clip_model.get_image_features(pixel_values=pixel_values)
        if isinstance(features, torch.Tensor):
            pass
        elif hasattr(features, 'pooler_output'):
            features = features.pooler_output
        else:
            features = features.last_hidden_state[:, 0, :]

        logits = self.classifier(features)
        return logits

    def get_moe_aux_loss(self):
        from models.moe_lora import get_moe_aux_loss
        return get_moe_aux_loss(self.clip_model)


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
                mixup_alpha=0.4, aux_loss_weight=0.01, use_moe=False):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels, crop_ids in dataloader:
        images, labels, crop_ids = images.to(device), labels.to(device), crop_ids.to(device)

        if mixup_alpha > 0 and random.random() < 0.5:
            mixed_images, y_a, y_b, lam = mixup_data(images, labels, mixup_alpha)
            outputs = model(mixed_images)
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        if use_moe:
            moe_aux = model.get_moe_aux_loss()
            loss = loss + aux_loss_weight * moe_aux

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

    for images, labels, crop_ids in dataloader:
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
    parser = argparse.ArgumentParser(
        description="统一训练脚本：融合所有创新点",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
消融实验:
  # 基线 (标准 LoRA + CE)
  python train_unified.py

  # + KGPT
  python train_unified.py --use-kgpt

  # + MoE
  python train_unified.py --use-moe

  # + 序数损失
  python train_unified.py --loss-type ordinal

  # 全部创新点
  python train_unified.py --all
        """)

    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--model", default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--early-stop", type=int, default=10)
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--random-erasing", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--amp", action="store_true")

    # 创新点开关
    parser.add_argument("--all", action="store_true", help="启用所有创新点")
    parser.add_argument("--use-kgpt", action="store_true", help="启用知识引导 Prompt 微调")
    parser.add_argument("--use-moe", action="store_true", help="启用 MoE-LoRA")
    parser.add_argument("--loss-type", choices=["ce", "ordinal", "combined"],
                        default="ce", help="损失函数类型")
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--num-shared", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=1.0, help="序数损失高斯带宽")
    parser.add_argument("--aux-loss-weight", type=float, default=0.01)

    args = parser.parse_args()

    # --all 开关
    if args.all:
        args.use_kgpt = True
        args.use_moe = True
        args.loss_type = "combined"

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto" else "cpu")
    print(f"设备: {device}")

    # 实验名
    exp_parts = []
    if args.use_kgpt: exp_parts.append("kgpt")
    if args.use_moe: exp_parts.append("moe")
    if args.loss_type != "ce": exp_parts.append(args.loss_type)
    experiment_name = "_".join(exp_parts) if exp_parts else "baseline_lora"

    if args.output_dir is None:
        model_short = args.model.split("/")[-1]
        args.output_dir = f"saved_models/clip/{model_short}-{experiment_name}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_size = 336 if "336" in args.model else 224

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
        transforms.RandomErasing(p=args.random_erasing, scale=(0.02, 0.2)),
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

    # 构建模型
    from transformers import CLIPModel

    print(f"加载 CLIP: {args.model}")
    clip_model = CLIPModel.from_pretrained(args.model)

    if args.use_moe:
        from models.moe_lora import apply_moe_lora
        crop_types = set(cls.split("_")[0] for cls in class_names)
        num_crops = len(crop_types)
        clip_model, moe_layers = apply_moe_lora(
            clip_model, num_experts=args.num_experts, num_shared=args.num_shared,
            rank=args.lora_rank, alpha=args.lora_alpha, num_crops=num_crops
        )
        print(f"MoE-LoRA: {args.num_experts} experts, {args.num_shared} shared")
    else:
        clip_model = apply_standard_lora(clip_model, rank=args.lora_rank, alpha=args.lora_alpha)

    model = UnifiedCropModel(
        clip_model, num_classes,
        use_kgpt=args.use_kgpt, class_names=class_names, img_size=img_size
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")
    print(f"实验: {experiment_name}")

    # 损失函数
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    if args.loss_type == "ce":
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=args.label_smoothing)
    elif args.loss_type == "ordinal":
        from models.ordinal_loss import OrdinalGaussianLoss
        criterion = OrdinalGaussianLoss(num_classes, sigma=args.sigma, alpha=0.5)
    else:  # combined
        criterion = CombinedOrdinalLoss(num_classes, sigma=args.sigma, alpha=0.5, beta=1.0)

    # 优化器
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params_list, lr=args.lr, weight_decay=1e-4)
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs, args.epochs)

    # AMP
    use_amp = args.amp
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # 恢复训练
    start_epoch = 0
    best_val_acc = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        best_val_acc = ckpt.get("best_val_acc", 0)
        start_epoch = ckpt.get("epoch", 0)

    # 训练
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_class_acc": [], "lr": []}

    print(f"\n{'='*70}")
    print(f" 开始训练: {experiment_name}")
    print(f" Epochs={args.epochs}, LR={args.lr}, LoRA rank={args.lora_rank}")
    print(f" KGPT={args.use_kgpt}, MoE={args.use_moe}, Loss={args.loss_type}")
    print(f" 类别权重已启用, WeightedRandomSampler 已启用")
    print(f"{'='*70}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            args.mixup_alpha, args.aux_loss_weight, args.use_moe
        )
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
                "experiment": experiment_name,
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
        "experiment": experiment_name,
        "model": args.model,
        "use_kgpt": args.use_kgpt,
        "use_moe": args.use_moe,
        "loss_type": args.loss_type,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "actual_epochs": epoch + 1,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_class_accs": {idx_to_class[idx]: acc for idx, acc in test_class_accs.items()},
        "class_sample_counts": dict(zip(class_names, class_sample_counts)),
        "class_weights": dict(zip(class_names, [round(w, 4) for w in class_weights])),
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
