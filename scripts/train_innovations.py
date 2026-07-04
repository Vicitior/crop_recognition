"""
统一创新训练脚本

集成三大创新点：
1. 置信度引导路由（Confidence-aware Routing）
2. 生育期关系建模（Phenology-aware Relation）
3. Adaptive LoRA Rank

支持消融实验：可单独或组合启用各创新点
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.growth_stages import CLASS_MAP


# ============================================================
# 数据集
# ============================================================

class CropGrowthDataset(Dataset):
    """
    农作物生长阶段数据集

    返回：(image, stage_label, crop_label)
    - stage_label: 0-14（全局阶段标签）
    - crop_label: 0-2（作物标签）
    """

    def __init__(self, data_dir, transform=None, stage_to_idx=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []

        # 构建标签映射
        if stage_to_idx is None:
            self.stage_to_idx = {}
            idx = 0
            for crop in ['corn', 'wheat', 'cotton']:
                for stage in self._get_stages(crop):
                    self.stage_to_idx[f"{crop}_{stage}"] = idx
                    idx += 1
        else:
            self.stage_to_idx = stage_to_idx

        self.crop_to_idx = {'corn': 0, 'wheat': 1, 'cotton': 2}

        # 扫描数据集
        self._scan_dataset()

    def _get_stages(self, crop):
        stages = {
            'corn': ['seedling', 'jointing', 'tasseling', 'filling', 'maturity'],
            'wheat': ['seedling', 'tillering', 'jointing', 'heading', 'maturity'],
            'cotton': ['seedling', 'squaring', 'flowering', 'boll_setting', 'boll_opening'],
        }
        return stages[crop]

    def _scan_dataset(self):
        for class_dir in self.data_dir.iterdir():
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            if class_name not in self.stage_to_idx:
                continue

            stage_label = self.stage_to_idx[class_name]
            crop_name = class_name.split('_')[0]
            crop_label = self.crop_to_idx.get(crop_name, -1)

            if crop_label == -1:
                continue

            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append({
                        'path': str(img_path),
                        'stage_label': stage_label,
                        'crop_label': crop_label,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, sample['stage_label'], sample['crop_label']


# ============================================================
# 模型构建
# ============================================================

def build_model(args, device):
    """
    根据参数构建模型

    支持的组合：
    - baseline: 标准分类头
    - confidence_router: 置信度路由
    - phenology_graph: 生育期图建模
    - adaptive_lora: 自适应 LoRA
    - all: 全部启用

    如果指定 --base-model-path，则从已有微调模型继续训练
    """
    from transformers import CLIPModel, CLIPProcessor
    from scripts.train_clip import CLIPWithClassifier, apply_lora

    # 加载 CLIP 模型
    model_name = "openai/clip-vit-large-patch14-336"
    print(f"[Model] 加载 CLIP: {model_name}")

    clip_model = CLIPModel.from_pretrained(model_name)
    feat_dim = clip_model.config.projection_dim  # 768

    # 如果有基线模型，加载已有的 LoRA 权重
    if args.base_model_path and os.path.isfile(args.base_model_path):
        print(f"[Model] 从基线模型继续训练: {args.base_model_path}")
        checkpoint = torch.load(args.base_model_path, map_location='cpu', weights_only=False)
        base_sd = checkpoint['model_state_dict']
        base_acc = checkpoint.get('best_val_acc', 'N/A')
        print(f"[Model] 基线准确率: {base_acc}%")

        # 从 checkpoint 推断 LoRA rank
        base_lora_rank = 8  # 默认值
        for key, val in base_sd.items():
            if 'lora_A' in key:
                base_lora_rank = val.shape[0]
                break
        print(f"[Model] 基线 LoRA rank: {base_lora_rank}")

        # 先对 CLIP 应用 LoRA（与基线模型相同的结构）
        base_model = CLIPWithClassifier(clip_model, num_classes=15, img_size=336)
        base_model = apply_lora(base_model, rank=base_lora_rank)

        # 只加载 LoRA 权重（跳过分类器）
        lora_sd = {}
        for key, val in base_sd.items():
            if 'clip_model.' in key:
                lora_sd[key] = val

        missing, unexpected = base_model.load_state_dict(lora_sd, strict=False)
        print(f"[Model] 加载基线 LoRA 权重: {len(lora_sd)} keys")

        # 提取 CLIP 部分（带 LoRA）
        clip_model = base_model.clip_model

        # 解冻 LoRA 参数
        for param in clip_model.parameters():
            param.requires_grad = False
        for name, param in clip_model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True

        print(f"[Model] 已加载基线 LoRA 权重")
    else:
        print(f"[Model] 从零开始训练（无基线模型）")
        # 冻结 CLIP 参数
        for param in clip_model.parameters():
            param.requires_grad = False

    # 移到设备
    clip_model = clip_model.to(device)

    models_dict = {}
    total_params = 0

    # 1. 置信度路由
    if args.use_confidence_router or args.all:
        from models.confidence_router import ConfidenceRouterClassifier
        router = ConfidenceRouterClassifier(
            feat_dim=feat_dim,
            num_classes=15,
            hidden_dim=256,
        ).to(device)
        models_dict['confidence_router'] = router
        params = sum(p.numel() for p in router.parameters() if p.requires_grad)
        total_params += params
        print(f"[Model] 置信度路由: {params:,} 参数")

    # 2. 生育期图建模
    if args.use_phenology_graph or args.all:
        from models.phenology_graph import PhenologyAwareClassifier
        phenology = PhenologyAwareClassifier(
            feat_dim=feat_dim,
            num_classes=15,
            hidden_dim=256,
            sigma=1.5,
        ).to(device)
        models_dict['phenology_graph'] = phenology
        params = sum(p.numel() for p in phenology.parameters() if p.requires_grad)
        total_params += params
        print(f"[Model] 生育期图建模: {params:,} 参数")

    # 3. Adaptive LoRA
    if args.use_adaptive_lora or args.all:
        from models.adaptive_lora import apply_adaptive_lora
        crop_ranks = {
            0: args.corn_rank,
            1: args.wheat_rank,
            2: args.cotton_rank,
        }
        clip_model, lora_stats = apply_adaptive_lora(
            clip_model, crop_ranks=crop_ranks,
            target_modules=["q_proj"],
            dropout=args.lora_dropout,
        )
        # 必须在 apply 之后再移一次设备（新创建的 LoRA 模块在 CPU 上）
        clip_model = clip_model.to(device)
        # 解冻 LoRA 参数
        for name, param in clip_model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True

        lora_params = lora_stats['total_lora_params']
        total_params += lora_params
        print(f"[Model] Adaptive LoRA: {lora_params:,} 参数")
        print(f"[Model]   Corn rank={args.corn_rank}, "
              f"Wheat rank={args.wheat_rank}, "
              f"Cotton rank={args.cotton_rank}")

    # 4. 标准分类头（如果不用置信度路由和生育期图）
    if not args.use_confidence_router and not args.use_phenology_graph \
            and not args.all:
        classifier = nn.Sequential(
            nn.Linear(feat_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 15)
        ).to(device)
        models_dict['classifier'] = classifier
        params = sum(p.numel() for p in classifier.parameters())
        total_params += params
        print(f"[Model] 标准分类头: {params:,} 参数")

    print(f"[Model] 总可训练参数: {total_params:,}")

    return clip_model, models_dict, feat_dim


# ============================================================
# 损失函数
# ============================================================

def build_loss_fn(args, device):
    """构建损失函数"""
    losses = {}

    if args.use_confidence_router or args.all:
        from models.confidence_router import ConfidenceRouterLoss
        losses['confidence_router'] = ConfidenceRouterLoss(
            lambda_crop=0.3,
            gamma_entropy=0.05,
        ).to(device)

    if args.use_phenology_graph or args.all:
        from models.phenology_graph import PhenologyAwareLoss
        losses['phenology_graph'] = PhenologyAwareLoss(
            num_stages=5,
            sigma=1.5,
            alpha=0.5,
            beta=0.1,
        ).to(device)

    # 标准损失（baseline 或其他情况）
    if not losses:
        losses['standard'] = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    return losses


# ============================================================
# 训练循环
# ============================================================

def train_one_epoch(clip_model, models_dict, loader, optimizer,
                    loss_fns, device, epoch, args, scaler=None):
    """训练一个 epoch（支持混合精度）"""
    clip_model.train()
    for m in models_dict.values():
        if isinstance(m, nn.Module):
            m.train()

    total_loss = 0
    correct = 0
    total = 0
    loss_details = {}

    use_amp = scaler is not None

    for batch_idx, (images, stage_labels, crop_labels) in enumerate(loader):
        images = images.to(device)
        stage_labels = stage_labels.to(device)
        crop_labels = crop_labels.to(device)

        optimizer.zero_grad()

        # 混合精度前向 + 损失
        with torch.amp.autocast('cuda', enabled=use_amp):
            # CLIP 视觉编码 → 投影到 768 维
            vision_outputs = clip_model.vision_model(pixel_values=images)
            pooled = vision_outputs.pooler_output  # [B, 1024]
            features = clip_model.visual_projection(pooled)  # [B, 768]

            loss = torch.tensor(0.0, device=device, requires_grad=True)

            # --all 模式：置信度路由 + 生育期损失联合
            if args.all and 'confidence_router' in models_dict:
                router = models_dict['confidence_router']
                stage_logits, crop_logits = router(features, threshold=args.threshold)
                l_router, details = loss_fns['confidence_router'](
                    stage_logits, crop_logits, stage_labels, crop_labels
                )
                loss = loss + l_router
                for k, v in details.items():
                    loss_details[k] = loss_details.get(k, 0) + v

                # 叠加生育期图损失
                if 'phenology_graph' in loss_fns:
                    corn_logits = stage_logits[:, 0:5]
                    wheat_logits = stage_logits[:, 5:10]
                    cotton_logits = stage_logits[:, 10:15]
                    crop_ids = crop_labels
                    stage_in_crop = stage_labels - crop_ids * 5
                    l_phen = torch.tensor(0.0, device=device)
                    for cid in range(3):
                        mask = (crop_ids == cid)
                        if mask.any():
                            logits_c = [corn_logits, wheat_logits, cotton_logits][cid][mask]
                            labels_c = stage_in_crop[mask]
                            l_phen = l_phen + loss_fns['phenology_graph'](logits_c, labels_c)
                    loss = loss + 0.3 * l_phen
                    loss_details['loss_phenology'] = loss_details.get(
                        'loss_phenology', 0) + l_phen.item()

                preds = stage_logits.argmax(dim=-1)
                correct += (preds == stage_labels).sum().item()

            # 置信度路由（单独）
            elif 'confidence_router' in models_dict:
                router = models_dict['confidence_router']
                stage_logits, crop_logits = router(features, threshold=args.threshold)
                l, details = loss_fns['confidence_router'](
                    stage_logits, crop_logits, stage_labels, crop_labels
                )
                loss = loss + l
                for k, v in details.items():
                    loss_details[k] = loss_details.get(k, 0) + v
                preds = stage_logits.argmax(dim=-1)
                correct += (preds == stage_labels).sum().item()

            # 生育期图建模（单独）
            elif 'phenology_graph' in models_dict:
                phenology = models_dict['phenology_graph']
                stage_logits, crop_logits = phenology(features)
                l_stage = loss_fns['phenology_graph'](stage_logits, stage_labels)
                l_crop = nn.functional.cross_entropy(crop_logits, crop_labels)
                loss = loss + l_stage + 0.3 * l_crop
                loss_details['loss_stage'] = loss_details.get(
                    'loss_stage', 0) + l_stage.item()
                loss_details['loss_crop'] = loss_details.get(
                    'loss_crop', 0) + l_crop.item()
                preds = stage_logits.argmax(dim=-1)
                correct += (preds == stage_labels).sum().item()

            # Adaptive LoRA 单独 / 标准分类头
            elif 'classifier' in models_dict:
                classifier = models_dict['classifier']
                logits = classifier(features)
                l = loss_fns['standard'](logits, stage_labels)
                loss = loss + l
                preds = logits.argmax(dim=-1)
                correct += (preds == stage_labels).sum().item()
            else:
                if 'classifier' not in models_dict:
                    models_dict['classifier'] = nn.Sequential(
                        nn.Linear(features.size(1), 1024),
                        nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.3),
                        nn.Linear(1024, 512),
                        nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(512, 15)
                    ).to(device)
                    loss_fns['standard'] = nn.CrossEntropyLoss(label_smoothing=0.1)
                logits = models_dict['classifier'](features)
                l = loss_fns['standard'](logits, stage_labels)
                loss = loss + l
                preds = logits.argmax(dim=-1)
                correct += (preds == stage_labels).sum().item()

        # 混合精度反向传播
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total += stage_labels.size(0)

        if (batch_idx + 1) % 20 == 0:
            acc = 100. * correct / total
            print(f"  Batch {batch_idx+1}/{len(loader)} | "
                  f"Loss: {total_loss/(batch_idx+1):.4f} | "
                  f"Acc: {acc:.2f}%")

    avg_loss = total_loss / len(loader)
    acc = 100. * correct / total

    return avg_loss, acc, loss_details


def evaluate(clip_model, models_dict, loader, device, args):
    """评估模型"""
    clip_model.eval()
    for m in models_dict.values():
        if isinstance(m, nn.Module):
            m.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, stage_labels, crop_labels in loader:
            images = images.to(device)
            stage_labels = stage_labels.to(device)

            # CLIP 视觉编码 → 投影到 768 维
            vision_outputs = clip_model.vision_model(pixel_values=images)
            pooled = vision_outputs.pooler_output  # [B, 1024]
            features = clip_model.visual_projection(pooled)  # [B, 768]

            # 获取 logits
            if 'confidence_router' in models_dict:
                stage_logits, _ = models_dict['confidence_router'](
                    features, threshold=args.threshold)
            elif 'phenology_graph' in models_dict:
                stage_logits, _ = models_dict['phenology_graph'](features)
            elif 'classifier' in models_dict:
                stage_logits = models_dict['classifier'](features)
            else:
                continue

            preds = stage_logits.argmax(dim=-1)
            correct += (preds == stage_labels).sum().item()
            total += stage_labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(stage_labels.cpu().numpy())

    acc = 100. * correct / total if total > 0 else 0

    # 计算每类准确率
    class_correct = {}
    class_total = {}
    for pred, label in zip(all_preds, all_labels):
        class_total[label] = class_total.get(label, 0) + 1
        if pred == label:
            class_correct[label] = class_correct.get(label, 0) + 1

    class_acc = {}
    # CLASS_MAP 结构: {class_name: {"index": idx, ...}}
    idx_to_class = {v["index"]: k for k, v in CLASS_MAP.items()}
    for label in sorted(class_total.keys()):
        cls_name = idx_to_class.get(label, f"class_{label}")
        cls_acc = 100. * class_correct.get(label, 0) / class_total[label]
        class_acc[cls_name] = cls_acc

    return acc, class_acc


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='农作物识别创新训练脚本'
    )

    # 创新开关
    parser.add_argument('--use-confidence-router', action='store_true',
                        help='启用置信度路由')
    parser.add_argument('--use-phenology-graph', action='store_true',
                        help='启用生育期图建模')
    parser.add_argument('--use-adaptive-lora', action='store_true',
                        help='启用 Adaptive LoRA')
    parser.add_argument('--all', action='store_true',
                        help='启用全部创新')

    # Adaptive LoRA 参数
    parser.add_argument('--corn-rank', type=int, default=4,
                        help='Corn LoRA rank')
    parser.add_argument('--wheat-rank', type=int, default=8,
                        help='Wheat LoRA rank')
    parser.add_argument('--cotton-rank', type=int, default=16,
                        help='Cotton LoRA rank')
    parser.add_argument('--lora-dropout', type=float, default=0.1,
                        help='LoRA dropout')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='置信度路由阈值')
    parser.add_argument('--base-model-path', type=str,
                        default='saved_models/clip/clip-vit-large-patch14-336-v2/best.pth',
                        help='基线微调模型路径（从该模型继续训练）')

    # 数据路径
    parser.add_argument('--train-dir', type=str,
                        default='dataset/train_augmented')
    parser.add_argument('--val-dir', type=str, default='dataset/val')
    parser.add_argument('--test-dir', type=str, default='dataset/test')

    # 保存
    parser.add_argument('--save-dir', type=str,
                        default='saved_models/innovations')
    parser.add_argument('--exp-name', type=str, default=None)

    args = parser.parse_args()

    # 自动生成实验名
    if args.exp_name is None:
        parts = []
        if args.use_confidence_router or args.all:
            parts.append('router')
        if args.use_phenology_graph or args.all:
            parts.append('phenology')
        if args.use_adaptive_lora or args.all:
            parts.append('adalora')
        if not parts:
            parts.append('baseline')
        args.exp_name = '_'.join(parts)

    save_dir = Path(args.save_dir) / args.exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 60)
    print(f"实验: {args.exp_name}")
    print(f"置信度路由: {args.use_confidence_router or args.all}")
    print(f"生育期图建模: {args.use_phenology_graph or args.all}")
    print(f"Adaptive LoRA: {args.use_adaptive_lora or args.all}")
    print("=" * 60)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 数据增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(336, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
        transforms.RandomErasing(p=0.2),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(336),
        transforms.CenterCrop(336),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    # 数据集
    print(f"\n加载训练集: {args.train_dir}")
    train_dataset = CropGrowthDataset(args.train_dir, train_transform)
    print(f"训练样本: {len(train_dataset)}")

    print(f"加载验证集: {args.val_dir}")
    val_dataset = CropGrowthDataset(
        args.val_dir, val_transform,
        stage_to_idx=train_dataset.stage_to_idx
    )
    print(f"验证样本: {len(val_dataset)}")

    # 数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=True
    )

    # 模型
    print("\n构建模型...")
    clip_model, models_dict, feat_dim = build_model(args, device)

    # 损失函数
    loss_fns = build_loss_fn(args, device)

    # 优化器（只优化可训练参数）
    trainable_params = []
    for name, param in clip_model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
    for m in models_dict.values():
        if isinstance(m, nn.Module):
            trainable_params.extend(
                [p for p in m.parameters() if p.requires_grad]
            )

    optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # 训练
    print("\n开始训练...")
    best_acc = 0
    history = []

    # 混合精度训练
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("[训练] 启用混合精度 (AMP)")

    for epoch in range(args.epochs):
        t0 = time.time()

        # 训练
        train_loss, train_acc, loss_details = train_one_epoch(
            clip_model, models_dict, train_loader, optimizer,
            loss_fns, device, epoch, args, scaler=scaler
        )

        # 验证
        val_acc, class_acc = evaluate(
            clip_model, models_dict, val_loader, device, args
        )

        scheduler.step()

        elapsed = time.time() - t0

        print(f"\nEpoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}% | "
              f"Time: {elapsed:.1f}s")

        # 记录历史
        epoch_info = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'class_acc': class_acc,
            'loss_details': loss_details,
            'lr': scheduler.get_last_lr()[0],
        }
        history.append(epoch_info)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            # 保存模型
            save_dict = {
                'epoch': epoch + 1,
                'val_acc': val_acc,
                'args': vars(args),
            }
            # 保存 CLIP LoRA 参数
            lora_state = {}
            for k, v in clip_model.state_dict().items():
                if 'lora' in k:
                    lora_state[k] = v
            save_dict['lora_state'] = lora_state

            # 保存分类器参数
            classifier_state = {}
            for name, m in models_dict.items():
                if isinstance(m, nn.Module):
                    classifier_state[name] = m.state_dict()
            save_dict['classifier_state'] = classifier_state

            torch.save(save_dict, save_dir / 'best.pth')
            print(f"  ✓ 保存最佳模型 (Val Acc: {val_acc:.2f}%)")

    # 保存训练历史
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # 最终报告
    print("\n" + "=" * 60)
    print(f"训练完成!")
    print(f"最佳验证准确率: {best_acc:.2f}%")
    print(f"模型保存至: {save_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
