"""
量化前后精度对比脚本
对比 CLIP ViT-Large-Patch14-336 + LoRA 模型在 INT8 动态量化前后的精度差异
"""

import os
import sys
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.quantization as quantization
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# 数据集
# ============================================================

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
# CLIP模型包装（与训练脚本一致）
# ============================================================

class CLIPWithClassifier(nn.Module):
    """CLIP模型 + 分类头（与训练时架构一致）"""

    def __init__(self, clip_model, num_classes, model_type="clip", freeze_clip=True, img_size=224):
        super().__init__()
        self.clip_model = clip_model
        self.model_type = model_type
        self.freeze_clip = freeze_clip

        # 获取特征维度
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            feat = clip_model.get_image_features(pixel_values=dummy)
            if isinstance(feat, torch.Tensor):
                self.feat_dim = feat.shape[-1]
            elif hasattr(feat, 'pooler_output'):
                self.feat_dim = feat.pooler_output.shape[-1]
            else:
                self.feat_dim = feat.last_hidden_state[:, 0, :].shape[-1]

        # 分类头（与训练时一致：Linear -> BN -> ReLU -> Dropout -> Linear -> BN -> ReLU -> Dropout -> Linear）
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, 1024),      # 0
            nn.BatchNorm1d(1024),                  # 1
            nn.ReLU(),                             # 2
            nn.Dropout(0.3),                       # 3
            nn.Linear(1024, 512),                  # 4
            nn.BatchNorm1d(512),                   # 5
            nn.ReLU(),                             # 6
            nn.Dropout(0.2),                       # 7
            nn.Linear(512, num_classes)             # 8
        )

        if freeze_clip:
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


# ============================================================
# LoRA实现（与训练脚本一致）
# ============================================================

class LoRALinear(nn.Module):
    """LoRA线性层"""

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

        # 冻结原始权重
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x):
        result = self.original(x)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return result + lora_out


def apply_lora(model, rank=8, alpha=16):
    """给模型添加LoRA适配器"""
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

    print(f"已添加 {count} 个LoRA适配器 (rank={rank}, alpha={alpha})")
    return model


# ============================================================
# 评估函数
# ============================================================

@torch.no_grad()
def evaluate(model, dataloader, device, desc="评估中"):
    """评估模型精度和推理速度"""
    model.eval()
    correct = 0
    total = 0
    total_time = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        start = time.time()
        outputs = model(images)
        total_time += time.time() - start

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    avg_time = total_time / len(dataloader)
    return acc, avg_time


@torch.no_grad()
def evaluate_per_class(model, dataloader, device, class_names):
    """评估每个类别的精度"""
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
            class_accs[cls] = 100. * class_correct[cls] / class_total[cls]
        else:
            class_accs[cls] = 0.0

    return class_accs


def get_model_size(model):
    """计算模型大小（MB）"""
    import tempfile
    tmp_path = os.path.join(tempfile.gettempdir(), "temp_model.pth")
    torch.save(model.state_dict(), tmp_path)
    size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    os.remove(tmp_path)
    return size_mb


# ============================================================
# 主函数
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="量化前后精度对比")
    parser.add_argument("--data-dir", default="dataset", help="数据集目录")
    parser.add_argument("--model-path", default="saved_models/clip/clip-vit-large-patch14-336-v2/best.pth",
                        help="模型路径")
    parser.add_argument("--batch-size", type=int, default=16, help="批大小")
    parser.add_argument("--device", default="auto", help="设备")
    args = parser.parse_args()

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")

    # 加载配置
    config_path = Path(args.model_path).parent / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    class_names = config["class_names"]
    num_classes = config["num_classes"]
    model_name = config["model"]
    img_size = 336 if "336" in model_name else 224

    # 加载checkpoint
    print(f"加载权重: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)

    # 从checkpoint推断LoRA rank（config可能不准确）
    for k, v in checkpoint["model_state_dict"].items():
        if "lora_A" in k:
            lora_rank = v.shape[0]
            break
    else:
        lora_rank = config.get("lora_rank", 8)
    lora_alpha = config.get("lora_alpha", lora_rank * 2)
    print(f"LoRA rank: {lora_rank}, alpha: {lora_alpha}")

    print(f"模型: {model_name}")
    print(f"类别数: {num_classes}")
    print(f"图片大小: {img_size}x{img_size}")

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

    # 加载CLIP模型
    print(f"加载CLIP模型: {model_name}")
    from transformers import CLIPModel, CLIPProcessor, AutoModel, AutoProcessor

    model_type = "clip"
    if "siglip" in model_name.lower():
        model_type = "siglip"
        clip_model = AutoModel.from_pretrained(model_name)
    else:
        clip_model = CLIPModel.from_pretrained(model_name)

    # 应用LoRA
    print(f"应用LoRA (rank={lora_rank}, alpha={lora_alpha})")
    clip_model = apply_lora(clip_model, rank=lora_rank, alpha=lora_alpha)

    # 构建完整模型
    model = CLIPWithClassifier(clip_model, num_classes, model_type, img_size=img_size)

    # 加载训练好的权重
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # ============================================================
    # 1. 评估原始模型
    # ============================================================
    print("\n" + "=" * 60)
    print("1. 评估原始模型 (FP32)")
    print("=" * 60)

    fp32_size = get_model_size(model)
    print(f"模型大小: {fp32_size:.2f} MB")

    fp32_acc, fp32_time = evaluate(model, test_loader, device, "FP32评估")
    print(f"测试精度: {fp32_acc:.2f}%")
    print(f"平均推理时间: {fp32_time*1000:.2f} ms/batch")

    fp32_class_accs = evaluate_per_class(model, test_loader, device, class_names)
    print("\n各类别精度:")
    for cls, acc in fp32_class_accs.items():
        print(f"  {cls}: {acc:.2f}%")

    # ============================================================
    # 2. 动态量化
    # ============================================================
    print("\n" + "=" * 60)
    print("2. 执行动态量化 (INT8)")
    print("=" * 60)

    # 动态量化：量化 Linear 层
    model_cpu = model.cpu()
    model_cpu.eval()

    quantized_model = quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear},  # 量化 Linear 层
        dtype=torch.qint8
    )

    quantized_size = get_model_size(quantized_model)
    print(f"量化后模型大小: {quantized_size:.2f} MB")
    print(f"压缩比: {fp32_size / quantized_size:.2f}x")

    # ============================================================
    # 3. 评估量化模型（量化模型只能在CPU上运行）
    # ============================================================
    print("\n" + "=" * 60)
    print("3. 评估量化模型 (INT8, CPU)")
    print("=" * 60)

    cpu_device = torch.device("cpu")
    # 创建CPU数据加载器
    test_loader_cpu = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0)
    int8_acc, int8_time = evaluate(quantized_model, test_loader_cpu, cpu_device, "INT8评估")
    print(f"测试精度: {int8_acc:.2f}%")
    print(f"平均推理时间: {int8_time*1000:.2f} ms/batch")

    int8_class_accs = evaluate_per_class(quantized_model, test_loader_cpu, cpu_device, class_names)
    print("\n各类别精度:")
    for cls, acc in int8_class_accs.items():
        print(f"  {cls}: {acc:.2f}%")

    # ============================================================
    # 4. 对比结果
    # ============================================================
    print("\n" + "=" * 60)
    print("4. 量化前后对比")
    print("=" * 60)

    print(f"\n{'指标':<20} {'FP32':<15} {'INT8':<15} {'差异':<15}")
    print("-" * 65)
    print(f"{'整体精度':<20} {fp32_acc:.2f}%{'':<9} {int8_acc:.2f}%{'':<9} {int8_acc - fp32_acc:+.2f}%")
    print(f"{'模型大小 (MB)':<20} {fp32_size:.2f}{'':<10} {quantized_size:.2f}{'':<10} {(quantized_size - fp32_size):.2f}")
    print(f"{'压缩比':<20} {'1x':<15} {fp32_size / quantized_size:.2f}x{'':<10}")

    print(f"\n各类别精度对比:")
    print(f"{'类别':<25} {'FP32':<12} {'INT8':<12} {'差异':<12}")
    print("-" * 61)
    for cls in class_names:
        fp32_cls = fp32_class_accs[cls]
        int8_cls = int8_class_accs[cls]
        diff = int8_cls - fp32_cls
        print(f"{cls:<25} {fp32_cls:.2f}%{'':<6} {int8_cls:.2f}%{'':<6} {diff:+.2f}%")

    # 保存结果
    results = {
        "fp32": {
            "accuracy": fp32_acc,
            "model_size_mb": fp32_size,
            "inference_time_ms": fp32_time * 1000,
            "class_accuracies": fp32_class_accs
        },
        "int8": {
            "accuracy": int8_acc,
            "model_size_mb": quantized_size,
            "inference_time_ms": int8_time * 1000,
            "class_accuracies": int8_class_accs
        },
        "diff": {
            "accuracy_diff": int8_acc - fp32_acc,
            "compression_ratio": fp32_size / quantized_size
        }
    }

    output_path = Path(args.model_path).parent / "quantization_compare.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_path}")

    # 保存量化模型
    quantized_model_path = Path(args.model_path).parent / "model_int8.pth"
    torch.save(quantized_model.state_dict(), quantized_model_path)
    print(f"量化模型已保存: {quantized_model_path}")


if __name__ == "__main__":
    main()
