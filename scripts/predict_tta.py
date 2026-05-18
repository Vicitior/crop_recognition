"""
测试时增强（TTA）预测脚本
通过多种图像变换提升预测精度

用法:
    python scripts/predict_tta.py --image test.jpg --model-path saved_models/clip/best.pth
    python scripts/predict_tta.py --image test.jpg --tta-level medium  # 基础/中等/强
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# TTA变换策略
# ============================================================

class TTACompose:
    """TTA变换组合"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        return [t(img) for t in self.transforms]

    def __len__(self):
        return len(self.transforms)


def get_tta_transforms(level="medium"):
    """
    获取TTA变换策略

    Args:
        level: 基础/中等/强
            - basic: 原图 + 水平翻转
            - medium: 多尺度 + 翻转
            - strong: 多尺度 + 翻转 + 旋转 + 色彩
    """
    img_size = 336  # CLIP ViT-L/14@336 的输入尺寸

    if level == "basic":
        # 基础：原图 + 水平翻转
        return [
            lambda img: _resize_center_crop(img, img_size),
            lambda img: _resize_center_crop(ImageOps.mirror(img), img_size),
        ]

    elif level == "medium":
        # 中等：多尺度 + 翻转
        scales = [0.9, 1.0, 1.1]
        transforms = []
        for scale in scales:
            size = int(img_size * scale)
            transforms.append(lambda img, s=size: _resize_center_crop(img, s))
            transforms.append(lambda img, s=size: _resize_center_crop(ImageOps.mirror(img), s))
        return transforms

    elif level == "strong":
        # 强：多尺度 + 翻转 + 旋转 + 色彩
        scales = [0.85, 0.9, 1.0, 1.1, 1.15]
        rotations = [0, 90, 180, 270]
        color_factors = [0.9, 1.0, 1.1]

        transforms = []
        for scale in scales:
            size = int(img_size * scale)
            for rot in rotations:
                for color in color_factors:
                    # 原图
                    transforms.append(
                        lambda img, s=size, r=rot, c=color:
                            _apply_augmentations(img, s, r, c, mirror=False)
                    )
                    # 水平翻转
                    transforms.append(
                        lambda img, s=size, r=rot, c=color:
                            _apply_augmentations(img, s, r, c, mirror=True)
                    )
        return transforms

    else:
        raise ValueError(f"未知的TTA级别: {level}")


def _resize_center_crop(img, size, target_size=336):
    """调整大小并中心裁剪，最终resize到目标大小"""
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(size + 32),
        transforms.CenterCrop(size),
        transforms.Resize(target_size),  # 确保最终大小一致
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img)


def _apply_augmentations(img, size, rotation, color_factor, mirror=False, target_size=336):
    """应用多种增强"""
    from torchvision import transforms

    # 水平翻转
    if mirror:
        img = ImageOps.mirror(img)

    # 旋转
    if rotation > 0:
        img = img.rotate(rotation, expand=False, fillcolor=(128, 128, 128))

    # 色彩调整
    if color_factor != 1.0:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(color_factor)

    # 调整大小并裁剪，最终resize到目标大小
    transform = transforms.Compose([
        transforms.Resize(size + 32),
        transforms.CenterCrop(size),
        transforms.Resize(target_size),  # 确保最终大小一致
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img)


# ============================================================
# TTA预测
# ============================================================

@torch.no_grad()
def predict_with_tta(model, image_path, class_names, device, tta_level="medium",
                     batch_size=8):
    """
    使用TTA进行预测

    Args:
        model: 微调后的CLIP模型
        image_path: 图片路径
        class_names: 类别名称列表
        device: 设备
        tta_level: TTA级别 (basic/medium/strong)
        batch_size: 批处理大小

    Returns:
        预测结果列表
    """
    image = Image.open(image_path).convert("RGB")

    # 获取TTA变换
    tta_transforms = get_tta_transforms(tta_level)
    print(f"TTA级别: {tta_level}, 变换数量: {len(tta_transforms)}")

    # 应用变换并收集张量
    augmented_images = []
    for t in tta_transforms:
        augmented_images.append(t(image))

    # 批量预测
    all_probs = []
    num_batches = (len(augmented_images) + batch_size - 1) // batch_size

    for i in range(num_batches):
        batch = augmented_images[i * batch_size: (i + 1) * batch_size]
        batch_tensor = torch.stack(batch).to(device)

        outputs = model(batch_tensor)
        probs = torch.softmax(outputs, dim=1)
        all_probs.append(probs.cpu())

    # 合并所有预测
    all_probs = torch.cat(all_probs, dim=0)  # [N, num_classes]

    # 不同的聚合策略
    results = {}

    # 1. 平均概率
    avg_probs = all_probs.mean(dim=0)
    results["avg"] = _probs_to_results(avg_probs, class_names)

    # 2. 最大概率（对每个类别取最大）
    max_probs = all_probs.max(dim=0)[0]
    results["max"] = _probs_to_results(max_probs, class_names)

    # 3. 几何平均（对log概率取平均）
    log_probs = torch.log(all_probs + 1e-8)
    geo_mean_probs = torch.exp(log_probs.mean(dim=0))
    results["geometric_mean"] = _probs_to_results(geo_mean_probs, class_names)

    # 4. 中位数（更鲁棒）
    median_probs = all_probs.median(dim=0)[0]
    results["median"] = _probs_to_results(median_probs, class_names)

    return results


def _probs_to_results(probs, class_names):
    """将概率转换为结果列表"""
    top_probs, top_indices = probs.topk(min(5, len(class_names)))
    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append({
            "class": class_names[idx.item()],
            "confidence": prob.item()
        })
    return results


# ============================================================
# 加载模型
# ============================================================

def load_finetuned_clip(model_path, device="cpu"):
    """加载微调后的CLIP模型"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    method = checkpoint.get("method", "lora")
    model_name = checkpoint.get("model_name", "openai/clip-vit-base-patch32")
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    # 加载CLIP模型
    from transformers import CLIPModel, AutoModel

    if "siglip" in model_name.lower():
        clip_model = AutoModel.from_pretrained(model_name)
    else:
        clip_model = CLIPModel.from_pretrained(model_name)

    # 如果是LoRA，先应用LoRA
    if method == "lora":
        from scripts.train_clip import apply_lora
        clip_model = apply_lora(clip_model)

    # 构建完整模型
    from scripts.train_clip import CLIPWithClassifier
    model = CLIPWithClassifier(clip_model, num_classes)

    # 加载权重
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, class_names


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="TTA预测脚本")
    parser.add_argument("--image", required=True, help="图片路径")
    parser.add_argument("--model-path", default="saved_models/clip/best.pth",
                        help="模型路径")
    parser.add_argument("--tta-level", choices=["basic", "medium", "strong"],
                        default="medium", help="TTA级别")
    parser.add_argument("--batch-size", type=int, default=8, help="批处理大小")
    parser.add_argument("--device", default="auto", help="设备")
    parser.add_argument("--strategy", choices=["avg", "max", "geometric_mean", "median", "all"],
                        default="all", help="聚合策略")
    args = parser.parse_args()

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"加载模型: {args.model_path}")
    model, class_names = load_finetuned_clip(args.model_path, device)
    print(f"类别: {class_names}")

    print(f"\n预测: {args.image}")
    results = predict_with_tta(
        model, args.image, class_names, device,
        tta_level=args.tta_level,
        batch_size=args.batch_size
    )

    # 输出结果
    if args.strategy == "all":
        for strategy, preds in results.items():
            print(f"\n{'='*50}")
            print(f"聚合策略: {strategy}")
            print('='*50)
            for i, r in enumerate(preds):
                print(f"  {i+1}. {r['class']}: {r['confidence']:.2%}")
    else:
        print(f"\n聚合策略: {args.strategy}")
        for i, r in enumerate(results[args.strategy]):
            print(f"  {i+1}. {r['class']}: {r['confidence']:.2%}")


if __name__ == "__main__":
    main()
