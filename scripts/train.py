"""
农作物分类模型训练脚本

使用方法：
    python scripts/train.py --data-dir dataset --epochs 30 --batch-size 16

支持功能：
- 断点续训（--resume）
- 冻结/解冻骨干网络
- 学习率调度
- 训练过程可视化
- 自动保存最佳模型
"""
import os
import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.classifier import build_model
from utils.dataset import get_dataloaders


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="训练")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.2f}%")

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def plot_training_curves(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curve")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history["train_acc"], label="Train Acc")
    ax2.plot(history["val_acc"], label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curve")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"训练曲线已保存至 {save_path}")


def main():
    parser = argparse.ArgumentParser(description="农作物分类模型训练")
    parser.add_argument("--data-dir", type=str, default="dataset", help="数据集目录")
    parser.add_argument("--output-dir", type=str, default="saved_models", help="模型保存目录")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=16, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step", "none"])
    parser.add_argument("--freeze-epochs", type=int, default=5, help="冻结骨干网络训练的轮数")
    parser.add_argument("--resume", type=str, default=None, help="断点续训的检查点路径")
    parser.add_argument("--num-workers", type=int, default=4, help="数据加载线程数")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 检测CUDA是否真正可用
    use_cuda = False
    if torch.cuda.is_available():
        try:
            t = torch.zeros(1, device="cuda")
            _ = t + 1
            use_cuda = True
        except RuntimeError:
            pass
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"使用设备: {device}")

    # 加载数据
    train_loader, val_loader, class_names = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )
    num_classes = len(class_names)
    print(f"类别数: {num_classes}")
    print(f"类别: {class_names}")
    print(f"训练样本: {len(train_loader.dataset)}")
    print(f"验证样本: {len(val_loader.dataset)}")

    # 构建模型（自动适配数据集类别数）
    model = build_model(num_classes=num_classes, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    start_epoch = 0
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # 断点续训
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        history = checkpoint.get("history", history)
        print(f"从 epoch {start_epoch} 恢复训练，最佳验证准确率: {best_val_acc:.4f}")

    # 训练策略：先冻结骨干网络训练分类头，再解冻全部微调
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # 冻结/解冻策略
        if epoch < args.freeze_epochs:
            model.freeze_backbone()
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr
            )
            print("  [冻结骨干网络] 仅训练分类头")
        else:
            if epoch == args.freeze_epochs:
                model.unfreeze_backbone()
                print("  [解冻骨干网络] 全模型微调")
            optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.1)

        # 学习率调度
        if args.scheduler == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - epoch)
        elif args.scheduler == "step":
            scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        else:
            scheduler = None

        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step()

        # 记录历史
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # 保存检查点
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "best_val_acc": best_val_acc,
            "history": history
        }
        torch.save(checkpoint, os.path.join(args.output_dir, "last.pth"))

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint["best_val_acc"] = best_val_acc
            torch.save(checkpoint, os.path.join(args.output_dir, "best.pth"))
            print(f"  -> 保存最佳模型 (val_acc: {best_val_acc:.4f})")

    # 保存训练曲线
    plot_training_curves(history, os.path.join(args.output_dir, "training_curves.png"))

    print(f"\n训练完成！最佳验证准确率: {best_val_acc:.4f}")
    print(f"模型保存在: {args.output_dir}/")


if __name__ == "__main__":
    main()
