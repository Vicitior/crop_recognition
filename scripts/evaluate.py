"""
模型评估脚本
在测试集上评估模型性能，输出分类报告和混淆矩阵
"""
import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.classifier import build_model
from models.growth_stages import NUM_CLASSES
from utils.dataset import get_test_dataloader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="评估"):
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.colorbar(im)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=7)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=5)

    ax.set_ylabel('True', fontsize=12)
    ax.set_xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"混淆矩阵已保存至 {save_path}")


def main():
    parser = argparse.ArgumentParser(description="模型评估")
    parser.add_argument("--data-dir", type=str, default="dataset", help="数据集目录")
    parser.add_argument("--model-path", type=str, default="saved_models/best.pth", help="模型路径")
    parser.add_argument("--batch-size", type=int, default=16, help="批大小")
    parser.add_argument("--num-workers", type=int, default=4, help="数据加载线程数")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    checkpoint = torch.load(args.model_path, map_location=device)
    class_names = checkpoint.get("class_names", None)

    model = build_model(num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print(f"模型已加载: {args.model_path}")
    print(f"训练时最佳验证准确率: {checkpoint.get('best_val_acc', 'N/A')}")

    # 加载测试集
    test_loader, test_classes = get_test_dataloader(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )
    print(f"测试样本: {len(test_loader.dataset)}")

    if class_names is None:
        class_names = test_classes

    # 评估
    preds, labels = evaluate(model, test_loader, device)

    # 分类报告
    report = classification_report(labels, preds, target_names=class_names, digits=4)
    print("\n分类报告：")
    print(report)

    # 保存报告
    report_path = os.path.join(os.path.dirname(args.model_path), "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"分类报告已保存至 {report_path}")

    # 混淆矩阵
    cm_path = os.path.join(os.path.dirname(args.model_path), "confusion_matrix.png")
    plot_confusion_matrix(labels, preds, class_names, cm_path)


if __name__ == "__main__":
    main()
