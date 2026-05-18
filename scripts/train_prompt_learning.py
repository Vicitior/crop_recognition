"""
提示学习训练脚本
实现CoOp和MaPLe方法，通过可学习的提示向量提升CLIP微调效果

用法:
    # CoOp方法
    python scripts/train_prompt_learning.py --method coop --model openai/clip-vit-large-patch14-336

    # MaPLe方法（多模态提示学习）
    python scripts/train_prompt_learning.py --method maple --model openai/clip-vit-large-patch14-336

参考论文:
    - CoOp: arXiv:2109.01134
    - MaPLe: arXiv:2210.03117
    - CasPL: arXiv:2409.17805
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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# 可学习提示模块
# ============================================================

class CoOpPromptLearner(nn.Module):
    """
    CoOp: Context Optimization
    学习连续的上下文提示向量，替代手工设计的文本提示

    参考论文: arXiv:2109.01134
    """

    def __init__(self, clip_model, class_names, n_ctx=16, ctx_init="a photo of",
                 class_token_position="middle"):
        super().__init__()
        self.clip_model = clip_model
        self.class_names = class_names
        self.n_ctx = n_ctx
        self.class_token_position = class_token_position

        # 获取文本编码器的嵌入维度
        embed_dim = clip_model.text_projection.shape[1]

        # 初始化上下文向量
        if ctx_init:
            # 使用预定义文本初始化
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip_model.tokenize([ctx_init])
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            self.n_ctx = n_ctx
        else:
            # 随机初始化
            ctx_vectors = torch.empty(n_ctx, embed_dim)
            nn.init.normal_(ctx_vectors, std=0.02)

        self.ctx = nn.Parameter(ctx_vectors)  # 可学习的上下文提示

        # 类别名称嵌入
        prompts = [f"a photo of a {name.replace('_', ' ')}" for name in class_names]
        tokenized = clip_model.tokenize(prompts)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized)
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        # 温度参数
        self.logit_scale = clip_model.logit_scale

    def forward(self):
        """生成所有类别的提示嵌入"""
        ctx = self.ctx  # [n_ctx, embed_dim]
        prefix = self.token_prefix  # [num_classes, 1, embed_dim]
        suffix = self.token_suffix  # [num_classes, *, embed_dim]

        # 构建提示
        if self.class_token_position == "middle":
            # 将类别token放在中间
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(len(self.class_names)):
                name = self.class_names[i]
                # 类别token的嵌入
                name_token = self.clip_model.tokenize([name])
                with torch.no_grad():
                    name_embed = self.clip_model.token_embedding(name_token)
                name_embed = name_embed[:, 1:2, :]  # 取类别token

                # 构建: prefix + ctx1 + name + ctx2 + suffix
                ctx_i = ctx.unsqueeze(0)  # [1, n_ctx, embed_dim]
                prompt = torch.cat([
                    prefix[i:i+1],
                    ctx_i[:, :half_n_ctx],
                    name_embed,
                    ctx_i[:, half_n_ctx:],
                    suffix[i:i+1]
                ], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            # 类别token放在最后
            prompts = []
            for i in range(len(self.class_names)):
                name = self.class_names[i]
                name_token = self.clip_model.tokenize([name])
                with torch.no_grad():
                    name_embed = self.clip_model.token_embedding(name_token)
                name_embed = name_embed[:, 1:2, :]

                ctx_i = ctx.unsqueeze(0)
                prompt = torch.cat([
                    prefix[i:i+1],
                    ctx_i,
                    name_embed,
                    suffix[i:i+1]
                ], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        return prompts


class MaPLePromptLearner(nn.Module):
    """
    MaPLe: Multi-modal Prompt Learning
    同时学习图像和文本编码器的提示

    参考论文: arXiv:2210.03117
    """

    def __init__(self, clip_model, class_names, n_ctx=16, ctx_init="a photo of",
                 prompt_depth=1):
        super().__init__()
        self.clip_model = clip_model
        self.class_names = class_names
        self.n_ctx = n_ctx
        self.prompt_depth = prompt_depth

        # 文本编码器维度
        embed_dim = clip_model.text_projection.shape[1]

        # 视觉编码器维度
        visual_width = clip_model.visual_projection.shape[1] if hasattr(clip_model, 'visual_projection') else embed_dim

        # 文本提示
        ctx_vectors = torch.empty(n_ctx, embed_dim)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        # 视觉提示（与文本提示深度耦合）
        self.visual_ctx = nn.ParameterList([
            nn.Parameter(torch.empty(n_ctx, visual_width))
            for _ in range(prompt_depth)
        ])
        for p in self.visual_ctx:
            nn.init.normal_(p, std=0.02)

        # 深度耦合层
        self.depth_coupling = nn.ModuleList([
            nn.Linear(embed_dim, visual_width)
            for _ in range(prompt_depth)
        ])

        # 类别名称嵌入
        prompts = [f"a photo of a {name.replace('_', ' ')}" for name in class_names]
        tokenized = clip_model.tokenize(prompts)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized)
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.logit_scale = clip_model.logit_scale

    def forward(self):
        """生成多模态提示"""
        ctx = self.ctx
        prefix = self.token_prefix
        suffix = self.token_suffix

        # 文本提示
        prompts = torch.cat([
            prefix,
            ctx.unsqueeze(0).expand(len(self.class_names), -1, -1),
            suffix
        ], dim=1)

        # 视觉提示（通过深度耦合）
        visual_prompts = []
        for i in range(self.prompt_depth):
            # 文本提示 -> 视觉提示
            coupled = self.depth_coupling[i](ctx)
            visual_prompts.append(self.visual_ctx[i] + coupled)

        return prompts, visual_prompts


# ============================================================
# 提示学习分类器
# ============================================================

class PromptLearningClassifier(nn.Module):
    """基于提示学习的分类器"""

    def __init__(self, clip_model, prompt_learner, num_classes, method="coop"):
        super().__init__()
        self.clip_model = clip_model
        self.prompt_learner = prompt_learner
        self.num_classes = num_classes
        self.method = method

        # 冻结CLIP模型
        for param in clip_model.parameters():
            param.requires_grad = False

    def forward(self, images):
        # 获取图像特征
        image_features = self.clip_model.get_image_features(pixel_values=images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        if self.method == "maple":
            # MaPLe: 使用多模态提示
            text_prompts, visual_prompts = self.prompt_learner()

            # 应用视觉提示
            # 这里简化处理，实际需要修改CLIP的forward
            # 暂时只使用文本提示
        else:
            # CoOp: 使用文本提示
            text_prompts = self.prompt_learner()

        # 编码文本提示
        text_features = []
        for i in range(len(text_prompts)):
            prompt = text_prompts[i:i+1]
            # 使用CLIP的文本编码器
            text_feat = self.clip_model.get_text_features(
                input_ids=prompt.argmax(dim=-1),
                attention_mask=torch.ones_like(prompt.argmax(dim=-1))
            )
            text_features.append(text_feat)

        text_features = torch.cat(text_features, dim=0)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 计算相似度
        logit_scale = self.prompt_learner.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


# ============================================================
# 简化版提示学习（无需修改CLIP内部）
# ============================================================

class SimplePromptLearner(nn.Module):
    """
    简化版提示学习
    直接优化分类头的权重，模拟提示学习的效果
    """

    def __init__(self, clip_model, class_names, n_ctx=16):
        super().__init__()
        self.clip_model = clip_model
        self.class_names = class_names
        self.n_ctx = n_ctx

        # 获取特征维度
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat = clip_model.get_image_features(pixel_values=dummy)
            if isinstance(feat, torch.Tensor):
                feat_dim = feat.shape[-1]
            else:
                feat_dim = feat.pooler_output.shape[-1] if hasattr(feat, 'pooler_output') else 768

        # 可学习的类别原型
        self.class_prototypes = nn.Parameter(
            torch.randn(len(class_names), feat_dim) * 0.02
        )

        # 可学习的上下文向量（用于增强类别原型）
        self.context_vectors = nn.Parameter(
            torch.randn(n_ctx, feat_dim) * 0.02
        )

        # 上下文融合层
        self.context_fusion = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

        # 温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def forward(self, images):
        # 获取图像特征
        image_features = self.clip_model.get_image_features(pixel_values=images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # 增强类别原型（使用上下文）
        context = self.context_vectors.mean(dim=0, keepdim=True)
        context = context.expand(len(self.class_names), -1)

        # 融合类别原型和上下文
        enhanced_prototypes = self.context_fusion(
            torch.cat([self.class_prototypes, context], dim=-1)
        )
        enhanced_prototypes = enhanced_prototypes / enhanced_prototypes.norm(dim=-1, keepdim=True)

        # 计算相似度
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ enhanced_prototypes.t()

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

def train_epoch(model, dataloader, criterion, optimizer, device):
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
    parser = argparse.ArgumentParser(description="提示学习训练")
    parser.add_argument("--data-dir", default="dataset", help="数据集目录")
    parser.add_argument("--model", default="openai/clip-vit-large-patch14-336",
                        help="CLIP模型名称")
    parser.add_argument("--method", choices=["coop", "maple", "simple"],
                        default="simple", help="提示学习方法")
    parser.add_argument("--n-ctx", type=int, default=16, help="上下文长度")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=16, help="批大小")
    parser.add_argument("--lr", type=float, default=2e-3, help="学习率")
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
        args.output_dir = f"saved_models/clip/{model_short}-{args.method}"
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

    # 构建模型
    if args.method == "coop":
        print(f"使用CoOp方法 (n_ctx={args.n_ctx})")
        prompt_learner = CoOpPromptLearner(clip_model, class_names, n_ctx=args.n_ctx)
        model = PromptLearningClassifier(clip_model, prompt_learner, num_classes, "coop")
    elif args.method == "maple":
        print(f"使用MaPLe方法 (n_ctx={args.n_ctx})")
        prompt_learner = MaPLePromptLearner(clip_model, class_names, n_ctx=args.n_ctx)
        model = PromptLearningClassifier(clip_model, prompt_learner, num_classes, "maple")
    else:
        print(f"使用简化提示学习方法 (n_ctx={args.n_ctx})")
        model = SimplePromptLearner(clip_model, class_names, n_ctx=args.n_ctx)

    model = model.to(device)

    # 统计参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数: {total_params:,}, 可训练: {trainable_params:,}")

    # 损失函数
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 学习率调度器
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs, args.epochs)

    # 训练
    best_val_acc = 0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "val_class_acc": [], "lr": []}

    print(f"\n开始训练 ({args.method})")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}, n_ctx: {args.n_ctx}")
    print("=" * 70)

    for epoch in range(args.epochs):
        start_time = time.time()

        # 更新学习率
        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']

        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

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
                "method": f"prompt_learning_{args.method}",
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
        "method": f"prompt_learning_{args.method}",
        "model": args.model,
        "n_ctx": args.n_ctx,
        "epochs": args.epochs,
        "actual_epochs": epoch + 1,
        "batch_size": args.batch_size,
        "lr": args.lr,
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
