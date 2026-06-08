"""
模型集成评估脚本
将多个fine-tuned CLIP模型的预测结果进行集成，提升最终准确率

用法:
    # 评估所有已训练模型的集成效果
    python scripts/ensemble_predict.py --data-dir dataset

    # 指定模型目录
    python scripts/ensemble_predict.py --data-dir dataset --model-dirs saved_models/clip/clip-large-336 saved_models/clip/clip-large
"""

import os
import sys
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_clip_v2 import CLIPWithClassifier as CLIPWithClassifierV2, apply_lora
from scripts.train_clip import CLIPWithClassifier as CLIPWithClassifierV1


class CropStageDataset(Dataset):
    """作物生长阶段数据集"""

    def __init__(self, data_dir, transform=None, split="test"):
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def load_model(model_dir, device):
    """加载单个模型"""
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    checkpoint_path = model_dir / "best.pth"

    if not checkpoint_path.exists():
        print(f"  [跳过] {model_dir} - best.pth 不存在")
        return None, None

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model_name = config.get("model", "openai/clip-vit-base-patch32")
    method = config.get("method", "lora")
    num_classes = config.get("num_classes", 5)
    class_names = config.get("class_names", [])
    lora_rank = config.get("lora_rank", 16) or 16
    lora_alpha = config.get("lora_alpha", 32) or 32

    # 从checkpoint检测实际的lora_rank（config可能不准确）
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    for key, val in state_dict.items():
        if "lora_A" in key:
            actual_rank = val.shape[0]
            if actual_rank != lora_rank:
                print(f"  [修正] lora_rank: {lora_rank} -> {actual_rank} (从checkpoint检测)")
                lora_rank = actual_rank
            break

    # 加载CLIP模型
    from transformers import CLIPModel, AutoModel

    if "siglip" in model_name.lower():
        clip_model = AutoModel.from_pretrained(model_name)
        model_type = "siglip"
    else:
        clip_model = CLIPModel.from_pretrained(model_name)
        model_type = "clip"

    # 应用LoRA
    if method == "lora":
        clip_model = apply_lora(clip_model, rank=lora_rank, alpha=lora_alpha)

    img_size = 336 if "336" in model_name else 224
    version = config.get("version", "v1")
    if version == "v2":
        model = CLIPWithClassifierV2(clip_model, num_classes, model_type, img_size=img_size)
    else:
        model = CLIPWithClassifierV1(clip_model, num_classes, model_type, img_size=img_size)

    # 加载权重（checkpoint已在前面加载用于检测lora_rank）
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    val_acc = config.get("best_val_acc", 0)
    test_acc = config.get("test_acc", 0)
    print(f"  已加载: {model_dir.name} (Val: {val_acc:.2f}%, Test: {test_acc:.2f}%)")

    return model, config


@torch.no_grad()
def get_model_predictions(model, dataloader, device, tta=False):
    """获取模型的预测概率"""
    all_probs = []

    for images, _ in tqdm(dataloader, desc="预测", leave=False):
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)

        if tta:
            # 水平翻转TTA
            flipped = torch.flip(images, dims=[3])
            outputs_flip = model(flipped)
            probs_flip = F.softmax(outputs_flip, dim=1)
            probs = (probs + probs_flip) / 2

        all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)


def evaluate_ensemble(all_probs_list, labels, class_names, weights=None):
    """评估集成效果"""
    # 加权平均
    if weights is None:
        weights = [1.0] * len(all_probs_list)

    ensemble_probs = np.zeros_like(all_probs_list[0])
    for probs, w in zip(all_probs_list, weights):
        ensemble_probs += w * probs
    ensemble_probs /= sum(weights)

    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    correct = np.sum(ensemble_preds == labels)
    total = len(labels)
    acc = 100.0 * correct / total

    # 每类准确率
    class_correct = {}
    class_total = {}
    for pred, label in zip(ensemble_preds, labels):
        cls_name = class_names[label]
        class_total[cls_name] = class_total.get(cls_name, 0) + 1
        if pred == label:
            class_correct[cls_name] = class_correct.get(cls_name, 0) + 1

    class_acc = {}
    for cls in class_names:
        if cls in class_total and class_total[cls] > 0:
            class_acc[cls] = 100.0 * class_correct.get(cls, 0) / class_total[cls]
        else:
            class_acc[cls] = 0.0

    return acc, ensemble_preds, class_acc


def main():
    parser = argparse.ArgumentParser(description="模型集成评估")
    parser.add_argument("--data-dir", default="dataset", help="数据集目录")
    parser.add_argument("--model-dirs", nargs="+", default=None,
                        help="模型目录列表 (默认搜索saved_models/clip/)")
    parser.add_argument("--batch-size", type=int, default=16, help="批大小")
    parser.add_argument("--tta", action="store_true", default=True, help="使用TTA")
    parser.add_argument("--no-tta", action="store_true", help="禁用TTA")
    parser.add_argument("--device", default="auto", help="设备")
    parser.add_argument("--min-acc", type=float, default=0.0, help="最低验证准确率阈值")
    parser.add_argument("--num-classes", type=int, default=15, help="只集成指定类别数的模型")
    args = parser.parse_args()

    if args.no_tta:
        args.tta = False

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")

    # 搜索模型目录
    if args.model_dirs is None:
        clip_dir = Path("saved_models/clip")
        args.model_dirs = []
        for d in sorted(clip_dir.iterdir()):
            if d.is_dir() and (d / "best.pth").exists() and (d / "config.json").exists():
                # 读取config检查num_classes
                try:
                    with open(d / "config.json", "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    # 过滤类别数
                    if args.num_classes and cfg.get("num_classes", 0) != args.num_classes:
                        continue
                    # 过滤最低准确率
                    if cfg.get("best_val_acc", 0) < args.min_acc:
                        continue
                    # 排除特殊实验模型（架构不同）
                    experiment = cfg.get("experiment", "")
                    if experiment in ["hierarchical", "temporal_contrast", "multimodal"]:
                        continue
                    args.model_dirs.append(str(d))
                except:
                    continue

    if not args.model_dirs:
        print("未找到任何符合条件的模型!")
        return

    print(f"\n找到 {len(args.model_dirs)} 个模型:")
    for d in args.model_dirs:
        print(f"  {d}")

    # 逐个模型评估（避免同时加载多个模型导致显存不足）
    all_probs = []
    configs = []
    model_names = []
    individual_accs = []

    # 先获取labels
    first_model_name_cfg = json.load(open(Path(args.model_dirs[0]) / "config.json", "r", encoding="utf-8")).get("model", "")
    first_img_size = 336 if "336" in first_model_name_cfg else 224
    first_transform = transforms.Compose([
        transforms.Resize(first_img_size + 32),
        transforms.CenterCrop(first_img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ref_dataset = CropStageDataset(args.data_dir, first_transform, "test")
    labels = np.array([s[1] for s in ref_dataset.samples])
    class_names = list(ref_dataset.class_to_idx.keys())
    print(f"\n测试集: {len(labels)} 张图片, {len(class_names)} 类")

    print(f"\n{'='*60}")
    print("各模型单独表现:")
    print(f"{'='*60}")

    for model_dir in args.model_dirs:
        model_dir_path = Path(model_dir)
        config_path = model_dir_path / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        model, _ = load_model(model_dir, device)
        if model is None:
            continue

        model_name_cfg = config.get("model", "")
        img_size = 336 if "336" in model_name_cfg else 224
        val_transform = transforms.Compose([
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_ds = CropStageDataset(args.data_dir, val_transform, "test")
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        probs = get_model_predictions(model, test_loader, device, tta=args.tta)
        all_probs.append(probs)
        configs.append(config)
        model_names.append(model_dir_path.name)

        preds = np.argmax(probs, axis=1)
        acc = 100.0 * np.sum(preds == labels) / len(labels)
        individual_accs.append(acc)
        print(f"  {model_dir_path.name}: {acc:.2f}%")

        # 释放显存
        del model
        torch.cuda.empty_cache()

    # 简单平均集成
    print(f"\n{'='*60}")
    print("集成结果:")
    print(f"{'='*60}")

    acc_avg, preds_avg, class_acc_avg = evaluate_ensemble(all_probs, labels, class_names)
    print(f"\n[简单平均] 集成准确率: {acc_avg:.2f}%")
    print("  每类准确率:")
    for cls, a in sorted(class_acc_avg.items()):
        print(f"    {cls}: {a:.2f}%")

    # 加权平均集成（按验证准确率加权）
    val_accs = [c.get("best_val_acc", 50) for c in configs]
    weights = [a / sum(val_accs) for a in val_accs]
    acc_weighted, preds_weighted, class_acc_weighted = evaluate_ensemble(
        all_probs, labels, class_names, weights
    )
    print(f"\n[加权平均] 集成准确率: {acc_weighted:.2f}%")
    print("  权重:", {n: f"{w:.3f}" for n, w in zip(model_names, weights)})
    print("  每类准确率:")
    for cls, a in sorted(class_acc_weighted.items()):
        print(f"    {cls}: {a:.2f}%")

    # 最佳单模型
    best_single_idx = np.argmax(individual_accs)
    print(f"\n[最佳单模型] {model_names[best_single_idx]}: {individual_accs[best_single_idx]:.2f}%")

    # 改善幅度
    best_single = max(individual_accs)
    best_ensemble = max(acc_avg, acc_weighted)
    improvement = best_ensemble - best_single
    print(f"\n{'='*60}")
    print(f"最佳单模型: {best_single:.2f}%")
    print(f"最佳集成:   {best_ensemble:.2f}%")
    print(f"提升:       {improvement:+.2f}%")
    print(f"{'='*60}")

    # 保存结果
    results = {
        "individual_accs": {n: a for n, a in zip(model_names, individual_accs)},
        "ensemble_avg_acc": acc_avg,
        "ensemble_weighted_acc": acc_weighted,
        "best_single_acc": best_single,
        "improvement": improvement,
        "class_accuracies_avg": class_acc_avg,
        "class_accuracies_weighted": class_acc_weighted,
        "tta": args.tta
    }

    results_path = Path("saved_models/clip/ensemble_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {results_path}")


if __name__ == "__main__":
    main()
