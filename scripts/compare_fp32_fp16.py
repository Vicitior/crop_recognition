"""
FP32 vs FP16 模型精度对比
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import json
import time
from pathlib import Path


class CropStageDataset(Dataset):
    def __init__(self, data_dir, transform=None, split="test"):
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        for idx, cls_name in enumerate(classes):
            self.class_to_idx[cls_name] = idx
            cls_dir = self.data_dir / cls_name
            for img_path in cls_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append((str(img_path), idx))

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes = len(classes)
        print(f"[{split}] {self.num_classes} 类, {len(self.samples)} 张图片")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=8, alpha=16):
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
        result = self.original(x)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return result + lora_out


def apply_lora(model, rank=8, alpha=16):
    target_modules = ["q_proj", "v_proj", "k_proj", "out_proj",
                      "fc1", "fc2", "query", "value", "key"]

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(target in name for target in target_modules):
                parts = name.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
                setattr(parent, parts[-1], lora_layer)
    return model


class CLIPWithClassifier(nn.Module):
    def __init__(self, clip_model, num_classes, model_type="clip", img_size=224):
        super().__init__()
        self.clip_model = clip_model
        self.model_type = model_type

        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            feat = clip_model.get_image_features(pixel_values=dummy)
            if isinstance(feat, torch.Tensor):
                self.feat_dim = feat.shape[-1]
            elif hasattr(feat, 'pooler_output'):
                self.feat_dim = feat.pooler_output.shape[-1]
            else:
                self.feat_dim = feat.last_hidden_state[:, 0, :].shape[-1]

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

        for param in clip_model.parameters():
            param.requires_grad = False

    def forward(self, pixel_values):
        features = self.clip_model.get_image_features(pixel_values=pixel_values)
        if isinstance(features, torch.Tensor):
            pass
        elif hasattr(features, 'pooler_output'):
            features = features.pooler_output
        else:
            features = features.last_hidden_state[:, 0, :]
        return self.classifier(features)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    class_correct = {}
    class_total = {}

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = outputs.max(1)

        for label, pred in zip(labels, predicted):
            total += 1
            if label == pred:
                correct += 1

    return 100.0 * correct / total


@torch.no_grad()
def evaluate_per_class(model, dataloader, device, class_names):
    model.eval()
    class_correct = {c: 0 for c in class_names}
    class_total = {c: 0 for c in class_names}

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = outputs.max(1)

        for label, pred in zip(labels, predicted):
            cls_name = class_names[label]
            class_total[cls_name] += 1
            if label == pred:
                class_correct[cls_name] += 1

    class_accs = {}
    for cls in class_names:
        if class_total[cls] > 0:
            class_accs[cls] = 100.0 * class_correct[cls] / class_total[cls]
        else:
            class_accs[cls] = 0.0

    return class_accs


def load_model(model_path, model_name, num_classes, lora_rank, lora_alpha, img_size, device):
    """加载模型"""
    from transformers import CLIPModel, AutoModel

    print(f"  加载CLIP模型: {model_name}")
    if "siglip" in model_name.lower():
        clip_model = AutoModel.from_pretrained(model_name)
    else:
        clip_model = CLIPModel.from_pretrained(model_name)

    clip_model = apply_lora(clip_model, rank=lora_rank, alpha=lora_alpha)
    model = CLIPWithClassifier(clip_model, num_classes, "clip", img_size=img_size)

    print(f"  加载权重: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FP32 vs FP16 模型精度对比")
    parser.add_argument("--data-dir", default="dataset", help="数据集目录")
    parser.add_argument("--fp32-path", default="saved_models/clip/clip-vit-large-patch14-336-v2/best.pth",
                        help="FP32模型路径")
    parser.add_argument("--fp16-path", default="nserver-master/models/clip/clip-vit-large-patch14-336-v2/best.pth",
                        help="FP16模型路径")
    parser.add_argument("--batch-size", type=int, default=16, help="批大小")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载配置
    config_path = Path(args.fp32_path).parent / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    class_names = config["class_names"]
    num_classes = config["num_classes"]
    model_name = config["model"]
    img_size = 336 if "336" in model_name else 224

    # 从checkpoint推断LoRA rank
    checkpoint = torch.load(args.fp32_path, map_location="cpu", weights_only=False)
    for k, v in checkpoint["model_state_dict"].items():
        if "lora_A" in k:
            lora_rank = v.shape[0]
            break
    else:
        lora_rank = 8
    lora_alpha = lora_rank * 2

    print(f"模型: {model_name}")
    print(f"LoRA rank: {lora_rank}, alpha: {lora_alpha}")
    print(f"类别数: {num_classes}")

    # 数据变换
    val_transform = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 加载测试数据集
    test_dataset = CropStageDataset(args.data_dir, val_transform, "test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)

    # ============================================================
    # 1. 评估 FP32 模型
    # ============================================================
    print("\n" + "=" * 60)
    print("1. 评估 FP32 模型")
    print("=" * 60)

    fp32_size = Path(args.fp32_path).stat().st_size / (1024 * 1024)
    print(f"模型大小: {fp32_size:.2f} MB")

    fp32_model = load_model(args.fp32_path, model_name, num_classes, lora_rank, lora_alpha, img_size, device)

    start = time.time()
    fp32_acc = evaluate(fp32_model, test_loader, device)
    fp32_time = time.time() - start
    print(f"测试精度: {fp32_acc:.2f}%")
    print(f"评估耗时: {fp32_time:.2f}s")

    fp32_class_accs = evaluate_per_class(fp32_model, test_loader, device, class_names)
    print("\n各类别精度:")
    for cls, acc in fp32_class_accs.items():
        print(f"  {cls}: {acc:.2f}%")

    # 释放FP32模型
    del fp32_model
    torch.cuda.empty_cache()

    # ============================================================
    # 2. 评估 FP16 模型
    # ============================================================
    print("\n" + "=" * 60)
    print("2. 评估 FP16 模型")
    print("=" * 60)

    fp16_size = Path(args.fp16_path).stat().st_size / (1024 * 1024)
    print(f"模型大小: {fp16_size:.2f} MB")

    fp16_model = load_model(args.fp16_path, model_name, num_classes, lora_rank, lora_alpha, img_size, device)

    start = time.time()
    fp16_acc = evaluate(fp16_model, test_loader, device)
    fp16_time = time.time() - start
    print(f"测试精度: {fp16_acc:.2f}%")
    print(f"评估耗时: {fp16_time:.2f}s")

    fp16_class_accs = evaluate_per_class(fp16_model, test_loader, device, class_names)
    print("\n各类别精度:")
    for cls, acc in fp16_class_accs.items():
        print(f"  {cls}: {acc:.2f}%")

    # ============================================================
    # 3. 对比结果
    # ============================================================
    print("\n" + "=" * 60)
    print("3. FP32 vs FP16 对比")
    print("=" * 60)

    print(f"\n{'指标':<20} {'FP32':<15} {'FP16':<15} {'差异':<15}")
    print("-" * 65)
    print(f"{'整体精度':<20} {fp32_acc:.2f}%{'':<9} {fp16_acc:.2f}%{'':<9} {fp16_acc - fp32_acc:+.2f}%")
    print(f"{'模型大小 (MB)':<20} {fp32_size:.2f}{'':<10} {fp16_size:.2f}{'':<10} {(fp16_size - fp32_size):.2f}")
    print(f"{'压缩比':<20} {'1x':<15} {fp32_size / fp16_size:.2f}x{'':<10}")

    print(f"\n各类别精度对比:")
    print(f"{'类别':<25} {'FP32':<12} {'FP16':<12} {'差异':<12}")
    print("-" * 61)
    for cls in class_names:
        fp32_cls = fp32_class_accs[cls]
        fp16_cls = fp16_class_accs[cls]
        diff = fp16_cls - fp32_cls
        marker = "⚠️" if abs(diff) > 5 else ""
        print(f"{cls:<25} {fp32_cls:.2f}%{'':<6} {fp16_cls:.2f}%{'':<6} {diff:+.2f}% {marker}")

    # 保存结果
    results = {
        "fp32": {
            "accuracy": fp32_acc,
            "model_size_mb": fp32_size,
            "eval_time_s": fp32_time,
            "class_accuracies": fp32_class_accs
        },
        "fp16": {
            "accuracy": fp16_acc,
            "model_size_mb": fp16_size,
            "eval_time_s": fp16_time,
            "class_accuracies": fp16_class_accs
        },
        "diff": {
            "accuracy_diff": fp16_acc - fp32_acc,
            "compression_ratio": fp32_size / fp16_size
        }
    }

    output_path = Path(args.fp32_path).parent / "fp32_vs_fp16_compare.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_path}")


if __name__ == "__main__":
    main()
