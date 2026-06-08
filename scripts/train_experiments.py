# -*- coding: utf-8 -*-
"""
综合实验脚本 - 支持多种创新点的消融实验
基于 train_clip_v2.py，新增：
  A. 层级分类（先作物后阶段）
  B. 时间对比学习（相邻阶段特征拉近）
  C. 课程学习（先易后难）
  D. 简化多模态融合（物理向量拼接）

用法:
    # 基线（复现 train_clip_v2.py）
    python scripts/train_experiments.py --exp baseline

    # 实验A: 层级分类
    python scripts/train_experiments.py --exp hierarchical

    # 实验B: 时间对比学习
    python scripts/train_experiments.py --exp temporal_contrast

    # 实验C: 课程学习
    python scripts/train_experiments.py --exp curriculum

    # 实验D: 简化多模态融合
    python scripts/train_experiments.py --exp multimodal

    # 实验A+D: 层级 + 多模态
    python scripts/train_experiments.py --exp hierarchical --use-multimodal
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
# 数据集（支持返回 crop_id 和 ordinal）
# ============================================================

class CropStageDataset(Dataset):
    def __init__(self, data_dir, transform=None, split="train", include_user_feedback=True):
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

        # 加载用户反馈图片（仅训练集）
        if include_user_feedback and split == "train":
            user_feedback_dir = Path(data_dir) / "user_feedback"
            if user_feedback_dir.exists():
                user_count = 0
                for cls_dir in user_feedback_dir.iterdir():
                    if cls_dir.is_dir():
                        cls_name = cls_dir.name
                        if cls_name in self.class_to_idx:
                            idx = self.class_to_idx[cls_name]
                            for img_path in cls_dir.glob("*"):
                                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                                    # 排除 JSON 元数据文件
                                    if not img_path.name.endswith('.json'):
                                        self.samples.append((str(img_path), idx))
                                        user_count += 1
                if user_count > 0:
                    print(f"  加载了 {user_count} 张用户反馈图片")

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes = len(classes)

        # 预计算 crop_id 和 ordinal
        self.crop_ids = []
        self.ordinal_positions = []
        crop_stage_order = {
            "corn": ["seedling", "jointing", "tasseling", "filling", "maturity"],
            "wheat": ["seedling", "tillering", "jointing", "heading", "maturity"],
            "cotton": ["seedling", "squaring", "flowering", "boll_setting", "boll_opening"],
        }
        crop_map = {"cotton": 0, "corn": 1, "wheat": 2}

        for _, label in self.samples:
            cls_name = self.idx_to_class[label]
            parts = cls_name.split("_")
            crop = parts[0]
            stage = "_".join(parts[1:])
            self.crop_ids.append(crop_map.get(crop, 0))
            order = crop_stage_order.get(crop, [])
            self.ordinal_positions.append(order.index(stage) if stage in order else 0)

        print(f"[{split}] {self.num_classes} 类, {len(self.samples)} 张图片")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, self.crop_ids[idx], self.ordinal_positions[idx]


# ============================================================
# LoRA
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


def apply_lora(model, rank=16, alpha=32):
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
    print(f"Applied {count} LoRA adapters (rank={rank}, alpha={alpha})")
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
# 模型定义
# ============================================================

class BaselineModel(nn.Module):
    """基线模型：CLIP + 分类头"""
    def __init__(self, clip_model, num_classes, feat_dim, img_size=336):
        super().__init__()
        self.clip_model = clip_model
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, pixel_values, crop_id=None, ordinal=None):
        features = self.clip_model.get_image_features(pixel_values=pixel_values)
        if isinstance(features, torch.Tensor):
            pass
        elif hasattr(features, 'pooler_output'):
            features = features.pooler_output
        else:
            features = features.last_hidden_state[:, 0, :]
        return self.classifier(features)


class HierarchicalModel(nn.Module):
    """实验A: 层级分类 - 先识别作物，再识别阶段"""
    def __init__(self, clip_model, num_classes, feat_dim, num_crops=3, img_size=336):
        super().__init__()
        self.clip_model = clip_model
        self.num_crops = num_crops

        # 作物分类头（第一层）
        self.crop_classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_crops)
        )

        # 阶段分类头（第二层）- 每个作物独立的阶段分类器
        self.stage_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim + num_crops, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 5)  # 每个作物5个阶段
            )
            for _ in range(num_crops)
        ])

        # 全局分类头（用于联合训练）
        self.global_classifier = nn.Sequential(
            nn.Linear(feat_dim + num_crops, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, pixel_values, crop_id=None, ordinal=None):
        features = self.clip_model.get_image_features(pixel_values=pixel_values)
        if isinstance(features, torch.Tensor):
            pass
        elif hasattr(features, 'pooler_output'):
            features = features.pooler_output
        else:
            features = features.last_hidden_state[:, 0, :]

        # 第一层：作物分类
        crop_logits = self.crop_classifier(features)

        # 用真实 crop_id 或预测的 crop_id
        if crop_id is not None:
            crop_onehot = F.one_hot(crop_id, self.num_crops).float()
        else:
            crop_onehot = F.softmax(crop_logits, dim=-1)

        # 第二层：全局分类（拼接作物信息）
        combined = torch.cat([features, crop_onehot], dim=-1)
        global_logits = self.global_classifier(combined)

        return global_logits, crop_logits


class TemporalContrastiveModel(nn.Module):
    """实验B: 时间对比学习 - 相邻阶段特征拉近，远离阶段推远"""
    def __init__(self, clip_model, num_classes, feat_dim, img_size=336, temperature=0.07):
        super().__init__()
        self.clip_model = clip_model
        self.temperature = temperature
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        # 特征投影头（用于对比学习）
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, pixel_values, crop_id=None, ordinal=None):
        features = self.clip_model.get_image_features(pixel_values=pixel_values)
        if isinstance(features, torch.Tensor):
            pass
        elif hasattr(features, 'pooler_output'):
            features = features.pooler_output
        else:
            features = features.last_hidden_state[:, 0, :]
        return self.classifier(features), features

    def temporal_contrastive_loss(self, features, crop_ids, ordinals):
        """时间对比损失：同作物内，相邻阶段特征拉近，远离阶段推远"""
        proj = self.projector(features)
        proj = F.normalize(proj, dim=-1)

        batch_size = features.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=features.device)

        # 计算特征相似度矩阵
        sim_matrix = torch.matmul(proj, proj.T) / self.temperature

        # 构建权重矩阵：同作物内，阶段距离越近权重越大
        crop_ids_exp = crop_ids.unsqueeze(1)  # [B, 1]
        ordinals_exp = ordinals.unsqueeze(1)  # [B, 1]

        same_crop = (crop_ids_exp == crop_ids_exp.T).float()  # [B, B]
        ordinal_dist = torch.abs(ordinals_exp - ordinals_exp.T).float()  # [B, B]

        # 相邻阶段（距离=1）权重高，远离阶段权重低
        # 使用高斯核：exp(-dist^2 / 2*sigma^2)
        sigma = 1.5
        temporal_weight = torch.exp(-ordinal_dist ** 2 / (2 * sigma ** 2))

        # 只在同作物内计算
        mask = same_crop * (1 - torch.eye(batch_size, device=features.device))

        # 正样本对：同作物且阶段相邻（距离<=1）
        positive_mask = mask * (ordinal_dist <= 1).float()

        # 负样本对：同作物但阶段远离（距离>=3）
        negative_mask = mask * (ordinal_dist >= 3).float()

        # InfoNCE 风格的损失
        exp_sim = torch.exp(sim_matrix)

        # 对每个样本，计算正样本对的概率
        pos_sum = (exp_sim * positive_mask).sum(dim=-1)
        neg_sum = (exp_sim * negative_mask).sum(dim=-1)

        # 避免除零
        valid = (positive_mask.sum(dim=-1) > 0) & (negative_mask.sum(dim=-1) > 0)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=features.device)

        loss = -torch.log(pos_sum[valid] / (pos_sum[valid] + neg_sum[valid] + 1e-8) + 1e-8).mean()
        return loss


class MultimodalModel(nn.Module):
    """实验D: 简化多模态融合 - 将作物ID和序数位置编码后拼接到CLIP特征"""
    def __init__(self, clip_model, num_classes, feat_dim, num_crops=3, num_ordinals=5, img_size=336):
        super().__init__()
        self.clip_model = clip_model

        # 作物嵌入
        self.crop_embedding = nn.Embedding(num_crops, 32)
        # 序数嵌入
        self.ordinal_embedding = nn.Embedding(num_ordinals, 32)

        # 融合后的分类头
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim + 64, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, pixel_values, crop_id=None, ordinal=None):
        features = self.clip_model.get_image_features(pixel_values=pixel_values)
        if isinstance(features, torch.Tensor):
            pass
        elif hasattr(features, 'pooler_output'):
            features = features.pooler_output
        else:
            features = features.last_hidden_state[:, 0, :]

        batch_size = features.shape[0]

        # 获取作物和序数嵌入
        if crop_id is not None:
            crop_emb = self.crop_embedding(crop_id)
        else:
            crop_emb = torch.zeros(batch_size, 32, device=features.device)

        if ordinal is not None:
            ord_emb = self.ordinal_embedding(ordinal)
        else:
            ord_emb = torch.zeros(batch_size, 32, device=features.device)

        # 拼接
        combined = torch.cat([features, crop_emb, ord_emb], dim=-1)
        return self.classifier(combined)


class CurriculumModel(nn.Module):
    """实验C: 课程学习 - 使用与基线相同的模型结构，训练策略不同"""
    def __init__(self, clip_model, num_classes, feat_dim, img_size=336):
        super().__init__()
        self.clip_model = clip_model
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, pixel_values, crop_id=None, ordinal=None):
        features = self.clip_model.get_image_features(pixel_values=pixel_values)
        if isinstance(features, torch.Tensor):
            pass
        elif hasattr(features, 'pooler_output'):
            features = features.pooler_output
        else:
            features = features.last_hidden_state[:, 0, :]
        return self.classifier(features)


# ============================================================
# 训练/评估函数
# ============================================================

def train_epoch(model, dataloader, criterion, optimizer, device,
                exp_type="baseline", mixup_alpha=0.4, contrast_weight=0.1,
                temporal_model=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels, crop_ids, ordinals in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        crop_ids = crop_ids.to(device)
        ordinals = ordinals.to(device)

        optimizer.zero_grad()

        # Mixup
        feat_for_contrast = None
        if mixup_alpha > 0 and random.random() < 0.5:
            mixed_images, y_a, y_b, lam = mixup_data(images, labels, mixup_alpha)
            model_out = model(mixed_images, crop_ids, ordinals)
            if exp_type == "hierarchical":
                outputs, crop_logits = model_out
            elif exp_type == "temporal_contrast":
                outputs, feat_for_contrast = model_out
                crop_logits = None
            else:
                outputs = model_out
                crop_logits = None
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
        else:
            model_out = model(images, crop_ids, ordinals)
            if exp_type == "hierarchical":
                outputs, crop_logits = model_out
            elif exp_type == "temporal_contrast":
                outputs, feat_for_contrast = model_out
                crop_logits = None
            else:
                outputs = model_out
                crop_logits = None
            loss = criterion(outputs, labels)

        # 层级分类额外损失（使用forward中已计算的crop_logits）
        if exp_type == "hierarchical" and crop_logits is not None:
            crop_loss = F.cross_entropy(crop_logits, crop_ids)
            loss = loss + 0.3 * crop_loss

        # 时间对比损失（使用forward中已计算的features）
        if exp_type == "temporal_contrast" and feat_for_contrast is not None:
            contrast_loss = model.temporal_contrastive_loss(feat_for_contrast, crop_ids, ordinals)
            loss = loss + contrast_weight * contrast_loss

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

    for images, labels, crop_ids, ordinals in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        crop_ids = crop_ids.to(device)
        ordinals = ordinals.to(device)

        model_out = model(images, crop_ids, ordinals)
        if isinstance(model_out, tuple):
            outputs = model_out[0]
        else:
            outputs = model_out
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
# 课程学习采样器
# ============================================================

class CurriculumSampler:
    """课程学习采样器：先采样容易的样本（种子期vs成熟期），逐步加入难样本"""
    def __init__(self, dataset, difficulty_schedule):
        self.dataset = dataset
        self.difficulty_schedule = difficulty_schedule  # list of (epoch, max_ordinal_dist)
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def get_sample_indices(self):
        """根据当前epoch返回可用样本的索引"""
        max_dist = 5  # 默认全部
        for ep, dist in self.difficulty_schedule:
            if self.current_epoch >= ep:
                max_dist = dist

        indices = []
        for idx, (_, label) in enumerate(self.dataset.samples):
            cls_name = self.dataset.idx_to_class[label]
            parts = cls_name.split("_")
            crop = parts[0]
            stage = "_".join(parts[1:])

            # 计算该样本的难度（基于阶段位置）
            crop_stage_order = {
                "corn": ["seedling", "jointing", "tasseling", "filling", "maturity"],
                "wheat": ["seedling", "tillering", "jointing", "heading", "maturity"],
                "cotton": ["seedling", "squaring", "flowering", "boll_setting", "boll_opening"],
            }
            order = crop_stage_order.get(crop, [])
            ordinal = order.index(stage) if stage in order else 2

            # 难度定义：中间阶段（2-3）最难区分，两端（0,4）最容易
            difficulty = min(ordinal, 4 - ordinal)  # 0=最易, 2=最难

            if difficulty <= max_dist:
                indices.append(idx)

        return indices


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="综合实验脚本")
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--model", default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--exp", choices=["baseline", "hierarchical", "temporal_contrast",
                                           "multimodal", "curriculum"],
                        default="baseline", help="实验类型")
    parser.add_argument("--use-multimodal", action="store_true", help="在其他实验基础上叠加多模态融合")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--early-stop", type=int, default=10)
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    parser.add_argument("--contrast-weight", type=float, default=0.1,
                        help="对比损失权重")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" \
        else torch.device(args.device)
    print(f"设备: {device}")

    # 输出目录
    exp_name = args.exp
    if args.use_multimodal:
        exp_name += "+multimodal"
    if args.output_dir is None:
        model_short = args.model.split("/")[-1]
        args.output_dir = f"saved_models/clip/{model_short}-{exp_name}"
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
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))
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

    num_classes = train_dataset.num_classes
    class_names = list(train_dataset.class_to_idx.keys())
    print(f"类别: {num_classes} — {class_names}")

    # 类别权重
    class_sample_counts = [0] * num_classes
    for _, label in train_dataset.samples:
        class_sample_counts[label] += 1
    total_samples = sum(class_sample_counts)
    class_weights = [total_samples / (num_classes * c) if c > 0 else 0 for c in class_sample_counts]
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    # 采样器
    sample_weights = [class_weights[label] for _, label in train_dataset.samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=sampler, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)

    # 加载CLIP
    from transformers import CLIPModel
    print(f"加载 CLIP: {args.model}")
    clip_model = CLIPModel.from_pretrained(args.model)
    clip_model = apply_lora(clip_model, rank=args.lora_rank, alpha=args.lora_alpha)

    # 获取特征维度
    config = clip_model.config
    feat_dim = config.projection_dim if hasattr(config, 'projection_dim') else \
        config.vision_config.hidden_size if hasattr(config, 'vision_config') else 768

    # 构建模型
    exp_type = args.exp
    if exp_type == "baseline":
        model = BaselineModel(clip_model, num_classes, feat_dim, img_size)
    elif exp_type == "hierarchical":
        model = HierarchicalModel(clip_model, num_classes, feat_dim, img_size=img_size)
    elif exp_type == "temporal_contrast":
        model = TemporalContrastiveModel(clip_model, num_classes, feat_dim, img_size)
    elif exp_type == "multimodal":
        model = MultimodalModel(clip_model, num_classes, feat_dim, img_size=img_size)
    elif exp_type == "curriculum":
        model = CurriculumModel(clip_model, num_classes, feat_dim, img_size)

    if args.use_multimodal and exp_type != "multimodal":
        print("警告: --use-multimodal 仅在 --exp multimodal 时生效")

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")

    # 损失函数
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)

    # 优化器
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params_list, lr=args.lr, weight_decay=1e-4)
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs, args.epochs)

    # 课程学习调度（优化版：更快的ramp-up）
    curriculum_schedule = [
        (0, 2),   # 从开始就使用中等难度样本
        (3, 3),   # 3轮后加入更多样本
        (6, 5),   # 6轮后使用全部样本
    ]

    # 训练
    best_val_acc = 0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    print(f"\n{'='*70}")
    print(f" 实验: {exp_type}, 模型: {args.model.split('/')[-1]}")
    print(f" Epochs={args.epochs}, LR={args.lr}, LoRA rank={args.lora_rank}")
    print(f"{'='*70}\n")

    for epoch in range(args.epochs):
        t0 = time.time()
        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']

        # 课程学习：动态调整可用样本
        if exp_type == "curriculum":
            max_diff = 5
            for ep, diff in curriculum_schedule:
                if epoch >= ep:
                    max_diff = diff
            # 重建采样器
            available_indices = []
            for idx, (_, label) in enumerate(train_dataset.samples):
                cls_name = train_dataset.idx_to_class[label]
                parts = cls_name.split("_")
                crop = parts[0]
                stage = "_".join(parts[1:])
                crop_stage_order = {
                    "corn": ["seedling", "jointing", "tasseling", "filling", "maturity"],
                    "wheat": ["seedling", "tillering", "jointing", "heading", "maturity"],
                    "cotton": ["seedling", "squaring", "flowering", "boll_setting", "boll_opening"],
                }
                order = crop_stage_order.get(crop, [])
                ordinal = order.index(stage) if stage in order else 2
                difficulty = min(ordinal, 4 - ordinal)
                if difficulty <= max_diff:
                    available_indices.append(idx)

            if len(available_indices) < args.batch_size:
                available_indices = list(range(len(train_dataset)))

            curriculum_sampler = torch.utils.data.SubsetRandomSampler(available_indices)
            train_loader_curriculum = DataLoader(train_dataset, batch_size=args.batch_size,
                                                 sampler=curriculum_sampler, num_workers=0,
                                                 pin_memory=True, drop_last=True)
            print(f"  课程学习 Epoch {epoch+1}: 使用 {len(available_indices)}/{len(train_dataset)} 样本 (max_diff={max_diff})")
        else:
            train_loader_curriculum = train_loader

        train_loss, train_acc = train_epoch(
            model, train_loader_curriculum, criterion, optimizer, device,
            exp_type=exp_type, mixup_alpha=args.mixup_alpha,
            contrast_weight=args.contrast_weight
        )
        val_loss, val_acc, class_accs = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        idx_to_class = train_dataset.idx_to_class
        class_acc_names = {idx_to_class[idx]: acc for idx, acc in class_accs.items()}

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.2e} | {elapsed:.1f}s")

        # 打印少数类准确率
        minority = [c for c, n in zip(class_names, class_sample_counts) if n < 60]
        if minority:
            acc_strs = [f"{c}:{class_acc_names.get(c, 0):.1f}%" for c in minority]
            print(f"  少数类: {', '.join(acc_strs)}")

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
            print(f"  >> 保存最佳模型 ({val_acc:.2f}%)")
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

    print(f"\n测试集: Loss={test_loss:.4f}, Acc={test_acc:.2f}%")
    print("\n逐类准确率:")
    for idx, acc in sorted(test_class_accs.items()):
        print(f"  {idx_to_class[idx]}: {acc:.2f}%")

    # 保存结果
    config = {
        "experiment": exp_name,
        "model": args.model,
        "epochs": args.epochs,
        "actual_epochs": epoch + 1,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lora_rank": args.lora_rank,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_class_accs": {idx_to_class[idx]: acc for idx, acc in test_class_accs.items()},
        "trainable_params": trainable_params,
        "total_params": total_params,
        "timestamp": datetime.now().isoformat(),
        "method": "lora",
        "num_classes": num_classes,
        "class_names": class_names,
        "version": "v2"
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
        ax1.set_title(f"Loss ({exp_name})")
        ax1.legend()
        ax2.plot(history["train_acc"], label="Train")
        ax2.plot(history["val_acc"], label="Val")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title(f"Accuracy ({exp_name})")
        ax2.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "training_curves.png", dpi=150)
    except ImportError:
        pass

    print(f"\n训练完成! 最佳验证: {best_val_acc:.2f}%, 测试: {test_acc:.2f}%")
    print(f"模型: {output_dir}")


if __name__ == "__main__":
    main()
