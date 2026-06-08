# -*- coding: utf-8 -*-
"""
95类多作物训练脚本 - 整合创新点
创新点 1：序数约束损失（Ordinal Loss）- 生长阶段类使用
创新点 2：MoE-LoRA - 多作物混合专家

用法:
    # 完整创新（Ordinal + MoE）
    python scripts/train_95class_ordinal_moe.py

    # 消融实验：仅 MoE（标准 CE）
    python scripts/train_95class_ordinal_moe.py --loss-type ce

    # 消融实验：仅 Ordinal（标准 LoRA）
    python scripts/train_95class_ordinal_moe.py --no-moe

    # 基线（标准 LoRA + CE）
    python scripts/train_95class_ordinal_moe.py --no-moe --loss-type ce
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# 95类数据集的序数关系定义
# ============================================================

# 每种作物的生长阶段（按时间顺序排列）
CROP_GROWTH_STAGES = {
    "corn": [
        "seedling", "trefoil", "seven_leaf", "jointing", "booting", "heading",
        "silking", "tasseling", "grain_filling", "dough", "maturity", "leaf"
    ],
    "cotton": [
        "seedling", "true_leaf", "five_leaf", "squaring",
        "flowering", "full_flowering", "boll_cracking", "boll_opening",
        "full_opening", "defoliation"
    ],
    "rapeseed": [
        "seedling", "five_leaf", "bolting", "squaring", "flowering",
        "full_flowering", "green_maturity", "maturity", "transplant_survival"
    ],
    "rice": [
        "seedling", "trefoil", "transplanting", "tillering", "jointing",
        "booting", "heading", "grain_filling", "maturity", "greening"
    ],
    "soybean": [
        "seedling", "true_leaf", "branching", "flowering",
        "pod_setting", "pod_filling", "maturity"
    ],
    "wheat": [
        "seedling", "trefoil", "overwintering", "rising", "greening",
        "tillering", "jointing", "booting", "heading", "flowering",
        "grain_filling", "dough", "maturity"
    ],
}

# 病害和健康类（无序数关系，使用标准 CE）
DISEASE_AND_HEALTHY_CROPS = [
    "apple", "blueberry", "cherry", "grape", "orange", "peach",
    "pepper", "potato", "raspberry", "squash", "strawberry", "tomato"
]


def build_class_metadata(class_names):
    """
    为每个类别构建元数据：
    - 是否为生长阶段类
    - 所属作物
    - 作物内阶段序号（仅生长阶段类）
    - 该作物的总阶段数
    """
    metadata = {}
    for idx, cls_name in enumerate(class_names):
        parts = cls_name.split("_", 1)
        crop = parts[0]
        stage_or_disease = parts[1] if len(parts) > 1 else ""

        if crop in CROP_GROWTH_STAGES and stage_or_disease in CROP_GROWTH_STAGES[crop]:
            # 生长阶段类
            stage_list = CROP_GROWTH_STAGES[crop]
            metadata[idx] = {
                "is_growth_stage": True,
                "crop": crop,
                "stage": stage_or_disease,
                "ordinal_idx": stage_list.index(stage_or_disease),
                "num_stages": len(stage_list),
            }
        else:
            # 病害/健康类
            metadata[idx] = {
                "is_growth_stage": False,
                "crop": crop,
                "stage": stage_or_disease,
                "ordinal_idx": None,
                "num_stages": None,
            }
    return metadata


def build_ordinal_soft_labels(labels, class_metadata, sigma=1.0):
    """
    为生长阶段类生成高斯软标签，病害类使用 one-hot。

    Args:
        labels: (B,) 整数标签
        class_metadata: 类别元数据字典
        sigma: 高斯带宽

    Returns:
        soft_labels: (B, num_classes) 概率分布
    """
    device = labels.device
    batch_size = labels.size(0)
    num_classes = max(class_metadata.keys()) + 1

    soft_labels = torch.zeros(batch_size, num_classes, device=device)

    for i in range(batch_size):
        label = labels[i].item()
        meta = class_metadata[label]

        if meta["is_growth_stage"]:
            # 生长阶段类：在同作物的阶段上生成高斯分布
            crop = meta["crop"]
            ordinal_idx = meta["ordinal_idx"]
            num_stages = meta["num_stages"]

            # 找到同作物的所有阶段类
            same_crop_classes = [
                idx for idx, m in class_metadata.items()
                if m["crop"] == crop and m["is_growth_stage"]
            ]

            # 按阶段顺序排列
            same_crop_classes.sort(key=lambda x: class_metadata[x]["ordinal_idx"])

            # 计算高斯权重
            for j, cls_idx in enumerate(same_crop_classes):
                dist = abs(j - ordinal_idx)
                weight = np.exp(-0.5 * (dist / sigma) ** 2)
                soft_labels[i, cls_idx] = weight

            # 归一化
            total = soft_labels[i].sum()
            if total > 0:
                soft_labels[i] /= total
        else:
            # 病害/健康类：使用 one-hot
            soft_labels[i, label] = 1.0

    return soft_labels


class OrdinalAwareLoss(nn.Module):
    """
    序数感知损失：生长阶段类使用高斯软标签 + EMD，病害类使用标准 CE。
    """

    def __init__(self, class_metadata, sigma=1.0, alpha=0.5, beta=1.0):
        super().__init__()
        self.class_metadata = class_metadata
        self.sigma = sigma
        self.alpha = alpha  # CE 权重
        self.beta = beta    # EMD 权重

        # 统计生长阶段类和病害类
        self.growth_stage_classes = [
            idx for idx, m in class_metadata.items() if m["is_growth_stage"]
        ]
        self.disease_classes = [
            idx for idx, m in class_metadata.items() if not m["is_growth_stage"]
        ]
        print(f"损失函数: {len(self.growth_stage_classes)} 个生长阶段类 (序数约束), "
              f"{len(self.disease_classes)} 个病害/健康类 (标准 CE)")

    def forward(self, logits, labels):
        """
        Args:
            logits: (B, C) 模型输出
            labels: (B,) 整数标签
        """
        device = logits.device
        batch_size = logits.size(0)

        # 生成软标签
        soft_labels = build_ordinal_soft_labels(labels, self.class_metadata, self.sigma)

        # 1. KL 散度损失（所有类）
        log_probs = F.log_softmax(logits, dim=-1)
        kl_loss = F.kl_div(log_probs, soft_labels, reduction='batchmean')

        # 2. EMD 损失（仅生长阶段类）
        emd_loss = torch.tensor(0.0, device=device)
        growth_mask = torch.tensor(
            [self.class_metadata[l.item()]["is_growth_stage"] for l in labels],
            device=device, dtype=torch.bool
        )

        if growth_mask.any():
            growth_logits = logits[growth_mask]
            growth_labels = labels[growth_mask]
            growth_soft = soft_labels[growth_mask]

            pred_probs = F.softmax(growth_logits, dim=-1)
            cdf_pred = torch.cumsum(pred_probs, dim=-1)
            cdf_target = torch.cumsum(growth_soft, dim=-1)
            emd_loss = (cdf_pred - cdf_target).abs().sum(dim=-1).mean()

        # 3. 标准 CE 损失
        ce_loss = F.cross_entropy(logits, labels, label_smoothing=0.1)

        # 组合损失
        total_loss = (1 - self.alpha) * kl_loss + self.alpha * ce_loss + self.beta * emd_loss

        return total_loss


# ============================================================
# 数据集
# ============================================================

class MultiCropDataset(Dataset):
    """多作物数据集"""

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
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    self.samples.append((str(img_path), idx))

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes = len(classes)

        # 作物 ID
        crop_set = sorted(set(cls.split("_")[0] for cls in classes))
        self.crop_to_id = {crop: i for i, crop in enumerate(crop_set)}
        self.num_crops = len(crop_set)

        self.crop_ids = []
        for _, label in self.samples:
            cls_name = self.idx_to_class[label]
            crop = cls_name.split("_")[0]
            self.crop_ids.append(self.crop_to_id.get(crop, 0))

        print(f"[{split}] {self.num_classes} 类, {len(self.samples)} 张图片, {self.num_crops} 种作物")

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
# MoE-LoRA 模型
# ============================================================

class CLIPWithMoELoRA(nn.Module):
    """CLIP + MoE-LoRA + 分类头"""

    def __init__(self, clip_model, num_classes, num_experts=4, num_shared=2,
                 lora_rank=8, lora_alpha=16, num_crops=1, img_size=224):
        super().__init__()
        self.clip_model = clip_model

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

        self.moe_layers = []

    def forward(self, pixel_values, crop_id=None):
        features = self.clip_model.get_image_features(pixel_values=pixel_values)
        if isinstance(features, torch.Tensor):
            pass
        elif hasattr(features, 'pooler_output'):
            features = features.pooler_output
        else:
            features = features.last_hidden_state[:, 0, :]
        return self.classifier(features)

    def get_moe_aux_loss(self):
        """汇总 MoE 辅助损失"""
        from models.moe_lora import get_moe_aux_loss
        return get_moe_aux_loss(self.clip_model)


# ============================================================
# 标准 LoRA（消融实验用）
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
# 训练 / 评估
# ============================================================

def train_epoch(model, dataloader, criterion, optimizer, device,
                mixup_alpha=0.4, aux_loss_weight=0.01,
                grad_accum_steps=1, scaler=None,
                epoch=0, save_callback=None, save_interval=600):
    """
    训练一个 epoch，支持定期保存断点。

    Args:
        save_callback: 保存回调函数，接受 (epoch, step, loss, acc) 参数
        save_interval: 保存间隔（秒），默认 600 秒（10 分钟）
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    last_save_time = time.time()

    optimizer.zero_grad()

    for step, (images, labels, crop_ids) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        # 混合精度训练
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                # Mixup
                if mixup_alpha > 0 and random.random() < 0.5:
                    mixed_images, y_a, y_b, lam = mixup_data(images, labels, mixup_alpha)
                    outputs = model(mixed_images)
                    loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                # MoE 辅助损失
                if hasattr(model, 'get_moe_aux_loss'):
                    moe_aux = model.get_moe_aux_loss()
                    loss = loss + aux_loss_weight * moe_aux

                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()
            if (step + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            # Mixup
            if mixup_alpha > 0 and random.random() < 0.5:
                mixed_images, y_a, y_b, lam = mixup_data(images, labels, mixup_alpha)
                outputs = model(mixed_images)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            # MoE 辅助损失
            if hasattr(model, 'get_moe_aux_loss'):
                moe_aux = model.get_moe_aux_loss()
                loss = loss + aux_loss_weight * moe_aux

            loss = loss / grad_accum_steps
            loss.backward()
            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 定期保存断点
        current_time = time.time()
        if save_callback and (current_time - last_save_time) >= save_interval:
            avg_loss = total_loss / (step + 1)
            avg_acc = 100. * correct / total
            save_callback(epoch, step, avg_loss, avg_acc)
            last_save_time = current_time
            print(f"  [断点保存] Epoch {epoch+1}, Step {step+1}/{len(dataloader)}, "
                  f"Loss: {avg_loss:.4f}, Acc: {avg_acc:.2f}%", flush=True)

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
    parser = argparse.ArgumentParser(description="95类多作物训练 - Ordinal + MoE")

    # 基础参数
    parser.add_argument("--data-dir", default="D:/crop_datasets/unified")
    parser.add_argument("--model", default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--early-stop", type=int, default=10)
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    parser.add_argument("--random-erasing", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--grad-accum", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--mixed-precision", action="store_true", help="使用混合精度训练")

    # 创新点参数
    parser.add_argument("--loss-type", choices=["ordinal", "ce"], default="ordinal",
                        help="损失类型: ordinal=序数感知损失, ce=标准交叉熵")
    parser.add_argument("--sigma", type=float, default=1.0, help="高斯软标签带宽")
    parser.add_argument("--alpha", type=float, default=0.5, help="CE 在组合损失中的权重")
    parser.add_argument("--beta", type=float, default=1.0, help="EMD 损失权重")
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
    experiment_name = "ordinal_moe" if not args.no_moe else "ordinal_only"
    if args.loss_type == "ce":
        experiment_name = "moe_only" if not args.no_moe else "baseline"

    if args.output_dir is None:
        args.output_dir = f"saved_models/clip/95class-{experiment_name}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    last_checkpoint = output_dir / "last.pth"

    # 数据增强
    img_size = 336 if "336" in args.model else 224
    print(f"图片大小: {img_size}x{img_size}")

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
    train_dataset = MultiCropDataset(args.data_dir, train_transform, "train")
    val_dataset = MultiCropDataset(args.data_dir, val_transform, "val")
    test_dataset = MultiCropDataset(args.data_dir, val_transform, "test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)

    num_classes = train_dataset.num_classes
    class_names = list(train_dataset.class_to_idx.keys())
    num_crops = train_dataset.num_crops
    print(f"类别: {num_classes} 个, 作物: {num_crops} 种")

    # 构建类别元数据
    class_metadata = build_class_metadata(class_names)
    growth_stage_count = sum(1 for m in class_metadata.values() if m["is_growth_stage"])
    print(f"生长阶段类: {growth_stage_count} 个, 病害/健康类: {num_classes - growth_stage_count} 个")

    # ============================================================
    # 构建模型
    # ============================================================
    from transformers import CLIPModel

    print(f"加载 CLIP: {args.model}")
    clip_model = CLIPModel.from_pretrained(args.model)
    print("模型加载完成")

    if args.no_moe:
        clip_model = apply_standard_lora(clip_model, rank=args.lora_rank, alpha=args.lora_alpha)
        print("标准 LoRA 应用完成")
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
        print(f"MoE-LoRA 应用完成 ({args.num_experts} 专家, {args.num_shared} 共享)")

    model = CLIPWithMoELoRA(clip_model, num_classes, num_experts=args.num_experts,
                             num_shared=args.num_shared, img_size=img_size)
    if not args.no_moe:
        model.moe_layers = moe_layers

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")

    # ============================================================
    # 损失函数
    # ============================================================
    if args.loss_type == "ce":
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        print("损失: CrossEntropy (label_smoothing=0.1)")
    else:
        criterion = OrdinalAwareLoss(
            class_metadata,
            sigma=args.sigma,
            alpha=args.alpha,
            beta=args.beta
        )
        print(f"损失: OrdinalAwareLoss (sigma={args.sigma}, alpha={args.alpha}, beta={args.beta})")

    # ============================================================
    # 优化器
    # ============================================================
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params_list, lr=args.lr, weight_decay=1e-4)
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs, args.epochs)

    # ============================================================
    # 恢复训练
    # ============================================================
    start_epoch = 0
    best_val_acc = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    resume_path = args.resume or (str(last_checkpoint) if last_checkpoint.exists() else None)
    if resume_path and Path(resume_path).exists():
        print(f"从断点恢复: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        best_val_acc = checkpoint.get("best_val_acc", 0)
        start_epoch = checkpoint.get("epoch", 0)
        if "history" in checkpoint:
            history = checkpoint["history"]
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print("  >> 已恢复 optimizer 状态")
            except Exception as e:
                print(f"  >> 恢复 optimizer 失败: {e}")
        print(f"从第 {start_epoch} 轮继续, 当前最佳验证准确率: {best_val_acc:.2f}%")

    # ============================================================
    # 训练循环
    # ============================================================
    patience_counter = 0

    print(f"\n开始训练 - {experiment_name}")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}, LoRA rank: {args.lora_rank}")
    print(f"  Early Stop: {args.early_stop} epochs patience")
    print(f"  Grad Accum: {args.grad_accum} steps")
    print(f"  Mixed Precision: {args.mixed_precision}")
    print("=" * 70)

    # 混合精度训练
    scaler = torch.amp.GradScaler('cuda') if args.mixed_precision else None

    # 定期保存断点的回调函数
    def save_checkpoint_callback(epoch, step, loss, acc):
        """定期保存断点"""
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "class_names": class_names,
            "experiment": experiment_name,
            "history": history,
            "train_loss": loss,
            "train_acc": acc
        }
        torch.save(checkpoint, last_checkpoint)

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()

        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            args.mixup_alpha, args.aux_loss_weight,
            args.grad_accum, scaler,
            epoch=epoch, save_callback=save_checkpoint_callback, save_interval=600
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s", flush=True)

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
                "experiment": experiment_name,
                "history": history
            }
            torch.save(checkpoint, output_dir / "best.pth")
            print(f"  >> 新最佳模型! Val Acc: {val_acc:.2f}%", flush=True)
        else:
            patience_counter += 1

        # 保存最新模型
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "class_names": class_names,
            "experiment": experiment_name,
            "history": history
        }
        torch.save(checkpoint, last_checkpoint)

        # Early Stopping
        if patience_counter >= args.early_stop:
            print(f"\n早停! 验证准确率 {args.early_stop} 轮未提升")
            break

    # ============================================================
    # 测试评估
    # ============================================================
    print("\n加载最佳模型进行测试...")
    best_ckpt = torch.load(output_dir / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    # 逐类准确率
    per_class = evaluate_per_class(model, test_loader, device, class_names)
    print("\n逐类准确率:")
    for cls, acc in sorted(per_class.items()):
        print(f"  {cls}: {acc:.1f}%")

    # 保存配置
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
        "grad_accum": args.grad_accum,
        "mixed_precision": args.mixed_precision,
        "lr": args.lr,
        "num_classes": num_classes,
        "num_crops": num_crops,
        "growth_stage_classes": growth_stage_count,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "per_class_acc": per_class,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "timestamp": datetime.now().isoformat(),
        "version": "v2"
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\n训练完成! 最佳验证准确率: {best_val_acc:.2f}%")
    print(f"测试准确率: {test_acc:.2f}%")
    print(f"模型保存在: {output_dir}")


if __name__ == "__main__":
    main()
