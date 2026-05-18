"""
MMD-LoRA训练脚本
结合多模态对比学习和领域对齐的LoRA微调方法

用法:
    python scripts/train_mmd_lora.py --model openai/clip-vit-large-patch14-336
    python scripts/train_mmd_lora.py --model openai/clip-vit-large-patch14-336 --lora-rank 32

参考论文: arXiv:2412.20162
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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# 对比学习损失
# ============================================================

class NTXentLoss(nn.Module):
    """
    NT-Xent对比损失
    用于视觉-文本特征对齐
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        Args:
            z_i: 视觉特征 [batch_size, feat_dim]
            z_j: 文本特征 [batch_size, feat_dim]
        """
        batch_size = z_i.shape[0]

        # 归一化
        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)

        # 计算相似度矩阵
        sim_matrix = torch.mm(z_i, z_j.t()) / self.temperature

        # 正样本对角线
        labels = torch.arange(batch_size, device=z_i.device)

        # 对称损失
        loss_i2t = F.cross_entropy(sim_matrix, labels)
        loss_t2i = F.cross_entropy(sim_matrix.t(), labels)

        return (loss_i2t + loss_t2i) / 2


class DomainAlignmentLoss(nn.Module):
    """
    领域对齐损失
    使用MMD（最大均值差异）对齐源域和目标域特征分布
    """

    def __init__(self, kernel_type="rbf", kernel_mul=2.0, kernel_num=5):
        super().__init__()
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    def _gaussian_kernel(self, source, target):
        """计算高斯核矩阵"""
        n_samples = source.shape[0] + target.shape[0]
        total = torch.cat([source, target], dim=0)

        # 计算L2距离
        total0 = total.unsqueeze(0).expand(total.shape[0], total.shape[0], total.shape[1])
        total1 = total.unsqueeze(1).expand(total.shape[0], total.shape[0], total.shape[1])
        L2_distance = ((total0 - total1) ** 2).sum(2)

        # 带宽
        bandwidth = torch.sum(L2_distance) / (n_samples ** 2 - n_samples)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]

        # 高斯核
        kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        """计算MMD损失"""
        batch_size = source.shape[0]
        kernels = self._gaussian_kernel(source, target)

        # 分割核矩阵
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]

        loss = torch.mean(XX + YY - XY - YX)
        return loss


# ============================================================
# MMD-LoRA层
# ============================================================

class MMDLoRALinear(nn.Module):
    """
    MMD-LoRA线性层
    结合LoRA和多模态对比学习
    """

    def __init__(self, original_linear, rank=16, alpha=32):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.alpha = alpha

        in_dim = original_linear.in_features
        out_dim = original_linear.out_features

        # LoRA参数
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


def apply_mmd_lora(model, rank=16, alpha=32):
    """给模型添加MMD-LoRA适配器"""
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

                lora_layer = MMDLoRALinear(module, rank=rank, alpha=alpha)
                setattr(parent, parts[-1], lora_layer)
                count += 1

    print(f"已添加 {count} 个MMD-LoRA适配器 (rank={rank}, alpha={alpha})")
    return model


# ============================================================
# MMD-LoRA分类器
# ============================================================

class MMDLoRAClassifier(nn.Module):
    """MMD-LoRA分类器"""

    def __init__(self, clip_model, num_classes, img_size=224):
        super().__init__()
        self.clip_model = clip_model

        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
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

        # 视觉-文本对齐层
        self.alignment_layer = nn.Sequential(
            nn.Linear(self.feat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.feat_dim)
        )

    def forward(self, pixel_values, return_features=False):
        # 获取视觉特征
        visual_features = self.clip_model.get_image_features(pixel_values=pixel_values)
        if not isinstance(visual_features, torch.Tensor):
            visual_features = visual_features.pooler_output if hasattr(visual_features, 'pooler_output') else visual_features.last_hidden_state[:, 0, :]

        # 对齐特征
        aligned_features = self.alignment_layer(visual_features)

        # 分类
        logits = self.classifier(visual_features)

        if return_features:
            return logits, visual_features, aligned_features
        return logits


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

def train_epoch_with_mmd(model, dataloader, criterion, optimizer, device,
                         contrastive_loss_fn, alignment_loss_fn,
                         class_names, clip_model,
                         contrastive_weight=0.1, alignment_weight=0.05):
    """带MMD损失的训练epoch"""
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_contrastive_loss = 0
    total_alignment_loss = 0
    correct = 0
    total = 0

    # 预计算类别文本特征
    text_features_list = []
    for cls_name in class_names:
        prompt = f"a photo of {cls_name.replace('_', ' ')}"
        inputs = clip_model.tokenizer([prompt], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            text_feat = clip_model.get_text_features(**inputs)
            if not isinstance(text_feat, torch.Tensor):
                text_feat = text_feat.pooler_output if hasattr(text_feat, 'pooler_output') else text_feat.last_hidden_state[:, 0, :]
        text_features_list.append(text_feat)
    text_features = torch.cat(text_features_list, dim=0)  # [num_classes, feat_dim]

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        logits, visual_features, aligned_features = model(images, return_features=True)

        # 分类损失
        cls_loss = criterion(logits, labels)

        # 对比学习损失（视觉-文本对齐）
        # 为每个样本选择对应的文本特征
        batch_text_features = text_features[labels]  # [batch_size, feat_dim]
        contrastive_loss = contrastive_loss_fn(visual_features, batch_text_features)

        # 领域对齐损失（对齐视觉特征和文本特征的分布）
        alignment_loss = alignment_loss_fn(visual_features, batch_text_features)

        # 总损失
        loss = cls_loss + contrastive_weight * contrastive_loss + alignment_weight * alignment_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_contrastive_loss += contrastive_loss.item()
        total_alignment_loss += alignment_loss.item()

        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    n_batches = len(dataloader)
    return (
        total_loss / n_batches,
        total_cls_loss / n_batches,
        total_contrastive_loss / n_batches,
        total_alignment_loss / n_batches,
        100. * correct / total
    )


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
    parser = argparse.ArgumentParser(description="MMD-LoRA训练")
    parser.add_argument("--data-dir", default="dataset", help="数据集目录")
    parser.add_argument("--model", default="openai/clip-vit-large-patch14-336",
                        help="CLIP模型名称")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=16, help="批大小")
    parser.add_argument("--lr", type=float, default=5e-4, help="学习率")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA秩")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="预热轮数")
    parser.add_argument("--early-stop", type=int, default=10, help="早停耐心值")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="标签平滑")
    parser.add_argument("--contrastive-weight", type=float, default=0.1,
                        help="对比损失权重")
    parser.add_argument("--alignment-weight", type=float, default=0.05,
                        help="对齐损失权重")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="对比损失温度参数")
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
        args.output_dir = f"saved_models/clip/{model_short}-mmd-lora"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载CLIP模型
    print(f"加载CLIP模型: {args.model}")
    from transformers import CLIPModel, CLIPProcessor

    clip_model_full = CLIPModel.from_pretrained(args.model)
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

    # 应用MMD-LoRA
    print(f"应用MMD-LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})")
    clip_model_full = apply_mmd_lora(clip_model_full, rank=args.lora_rank, alpha=args.lora_alpha)

    # 构建模型
    model = MMDLoRAClassifier(clip_model_full, num_classes, img_size=img_size)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}, 可训练: {trainable_params:,}")

    # 损失函数
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    contrastive_loss_fn = NTXentLoss(temperature=args.temperature)
    alignment_loss_fn = DomainAlignmentLoss()

    # 优化器
    lora_params = [p for n, p in model.named_parameters()
                   if "lora_" in n or "classifier" in n or "alignment" in n]
    optimizer = optim.AdamW(lora_params, lr=args.lr, weight_decay=1e-4)

    # 学习率调度器
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs, args.epochs)

    # 训练
    best_val_acc = 0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "cls_loss": [], "contrastive_loss": [], "alignment_loss": [], "lr": []}

    print(f"\n开始训练 - MMD-LoRA")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}, LoRA rank: {args.lora_rank}")
    print(f"  Contrastive Weight: {args.contrastive_weight}")
    print(f"  Alignment Weight: {args.alignment_weight}")
    print("=" * 70)

    for epoch in range(args.epochs):
        start_time = time.time()

        # 更新学习率
        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']

        # 训练
        (train_loss, cls_loss, contrastive_loss, alignment_loss,
         train_acc) = train_epoch_with_mmd(
            model, train_loader, criterion, optimizer, device,
            contrastive_loss_fn, alignment_loss_fn,
            class_names, clip_model_full,
            args.contrastive_weight, args.alignment_weight
        )

        # 评估
        val_loss, val_acc, class_accs = evaluate(model, val_loader, criterion, device)

        idx_to_class = train_dataset.idx_to_class
        elapsed = time.time() - start_time

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["cls_loss"].append(cls_loss)
        history["contrastive_loss"].append(contrastive_loss)
        history["alignment_loss"].append(alignment_loss)
        history["lr"].append(current_lr)

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} (cls:{cls_loss:.4f} ct:{contrastive_loss:.4f} al:{alignment_loss:.4f}) | "
              f"Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
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
                "method": "mmd_lora",
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
        "method": "mmd_lora",
        "model": args.model,
        "epochs": args.epochs,
        "actual_epochs": epoch + 1,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "contrastive_weight": args.contrastive_weight,
        "alignment_weight": args.alignment_weight,
        "temperature": args.temperature,
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
