# -*- coding: utf-8 -*-
"""
创新点 4：基于扩散模型的数据增强
使用 Stable Diffusion 生成少数类农作物图像，解决数据不均衡问题

用法:
    # 为玉米灌浆期生成图像（训练集中为0张）
    python scripts/generate_diffusion_aug.py --target-class corn_filling --num-images 50

    # 为所有少数类自动生成
    python scripts/generate_diffusion_aug.py --auto --min-count 30

    # 使用本地模型
    python scripts/generate_diffusion_aug.py --auto --model-path path/to/sd-model
"""

import os
import sys
import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================
# 农作物生长阶段的详细 Prompt 模板
# ============================================================

CROP_STAGE_PROMPTS = {
    "corn": {
        "seedling": [
            "a photo of young corn seedlings emerging from soil, tiny green leaves, early growth stage, agricultural field",
            "corn sprouts in a farm field, very short plants with first leaves, close-up view",
            "baby corn plants growing in rows, seedling stage, green tender leaves, soil visible",
        ],
        "jointing": [
            "corn plants at jointing stage, thick green stems with visible nodes, waist height, agricultural field",
            "tall corn stalks showing stem joints and nodes, vegetative growth, green leaves spreading",
            "mid-growth corn field, plants with thick stems and visible internodes, no ears yet",
        ],
        "tasseling": [
            "corn plants with tassels emerging from the top, flowering stage, silk visible on ear shoots",
            "corn field at tasseling stage, brown tassels on top of each stalk, silk threads visible",
            "flowering corn plants, male tassels at apex, female silk emerging from ear positions",
        ],
        "filling": [
            "corn ears with plump yellow kernels, green husks partially wrapping the cob, grain filling stage",
            "maturing corn field, full-sized ears with visible yellow kernels, stalks still green",
            "corn plants with developing grain, ears with rows of yellow kernels, husks starting to loosen",
        ],
        "maturity": [
            "fully mature corn field, brown dried plants, open husks exposing dry yellow cobs",
            "harvest-ready corn, brown stalks with dried leaves, mature cobs hanging down",
            "senescent corn plants, completely brown foliage, dried husks peeling back, hard kernels",
        ],
    },
    "wheat": {
        "seedling": [
            "wheat seedlings emerging from soil, thin grass-like leaves, very short plants, agricultural field",
            "young wheat shoots in rows, seedling stage, fine green blades, bare soil visible",
            "newly sprouted wheat field, small individual green plants, cereal seedling appearance",
        ],
        "tillering": [
            "wheat plants at tillering stage, multiple side shoots from base, bushy green clumps",
            "tillering wheat field, dense green tufts, each plant a cluster of leaves from the base",
            "wheat producing multiple tillers, forming bushy rosettes, low-growing canopy",
        ],
        "jointing": [
            "wheat stems elongating at jointing stage, visible nodes, tall green stalks, no grain heads",
            "jointing wheat field, plants with swollen stem nodes, stems thickening, pre-heading growth",
            "wheat at stem elongation, upright green stalks with visible joints, growing taller",
        ],
        "heading": [
            "wheat heads emerging from flag leaf sheath, compact green grain spikes at top of stems",
            "heading wheat field, grain ears fully emerged, compact spikes with visible anthers",
            "wheat at heading stage, each stem topped with an erect grain head, flowering stage",
        ],
        "maturity": [
            "golden amber wheat field, all plants uniformly yellow-gold, grain heads heavy and drooping",
            "mature wheat ready for harvest, dry golden stalks, grain heads bent down",
            "ripe wheat field, amber colored, dried stalks with full grain heads hanging down",
        ],
    },
    "cotton": {
        "seedling": [
            "cotton seedlings with round cotyledon leaves close to the ground, thin green stems",
            "young cotton plants just emerged, round seed leaves, delicate stems, agricultural field",
            "cotton at seedling stage, small plants with round leaves, very short, no flower buds",
        ],
        "squaring": [
            "cotton plants with small square-shaped green flower buds, bushy branching growth",
            "cotton at squaring stage, triangular green buds among lush foliage, no open flowers",
            "pre-flowering cotton, multiple branches bearing small square buds, deep green leaves",
        ],
        "flowering": [
            "cotton plants with open flowers, creamy white petals with dark red spots, yellow stamens",
            "flowering cotton field, large open blooms with white petals turning pink",
            "cotton in full bloom, distinctive flowers with white cream petals, yellow stamen column",
        ],
        "boll_setting": [
            "cotton plants with dark green round bolls developing on branches, no open flowers",
            "cotton at boll-setting stage, firm green spherical fruit capsules on branches",
            "cotton field with developing bolls, green rounded capsules of various sizes",
        ],
        "boll_opening": [
            "cotton bolls splitting open revealing fluffy white cotton fiber, brown dried shells",
            "cotton field with open bolls, white fluffy cotton exposed from cracked capsules",
            "mature cotton with bolls cracked wide open, white cotton fiber bursting out",
        ],
    },
}


def get_prompt_for_class(class_name, num_prompts=3):
    """获取指定类别的生成 prompt"""
    parts = class_name.split("_")
    crop = parts[0]
    stage = "_".join(parts[1:])

    prompts = CROP_STAGE_PROMPTS.get(crop, {}).get(stage, [])
    if not prompts:
        prompts = [f"a photo of {crop} at {stage} growth stage in agricultural field"]

    # 随机选择并添加变体
    selected = random.sample(prompts, min(num_prompts, len(prompts)))
    return selected


def get_negative_prompt():
    """通用负面提示"""
    return "blurry, low quality, distorted, deformed, cartoon, anime, painting, drawing, sketch, watermark, text, logo"


# ============================================================
# 扩散模型图像生成器
# ============================================================

class DiffusionAugmenter:
    """基于 Stable Diffusion 的图像生成器"""

    def __init__(self, model_name="stabilityai/stable-diffusion-2-1", device=None, dtype=torch.float16):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = dtype if device == "cuda" else torch.float32

        print(f"加载扩散模型: {model_name}")
        try:
            from diffusers import StableDiffusionPipeline
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                safety_checker=None,
                requires_safety_checker=False,
            ).to(self.device)
            # 内存优化
            if hasattr(self.pipe, 'enable_attention_slicing'):
                self.pipe.enable_attention_slicing()
            if device == "cuda" and hasattr(self.pipe, 'enable_xformers_memory_efficient_attention'):
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass
            print("扩散模型加载完成")
        except ImportError:
            print("错误: 需要安装 diffusers 库: pip install diffusers")
            sys.exit(1)
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("使用备用方案: 基于现有图像的增强")
            self.pipe = None

    def generate(self, prompt, negative_prompt=None, num_images=1,
                 height=512, width=512, num_inference_steps=30, guidance_scale=7.5):
        """生成图像"""
        if self.pipe is None:
            return []

        if negative_prompt is None:
            negative_prompt = get_negative_prompt()

        images = []
        for i in range(num_images):
            seed = random.randint(0, 2**32 - 1)
            generator = torch.Generator(self.device).manual_seed(seed)

            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            images.append(result.images[0])

        return images

    def generate_for_class(self, class_name, num_images=30, img_size=512):
        """为指定类别生成图像"""
        prompts = get_prompt_for_class(class_name)
        images = []

        for i in range(num_images):
            prompt = prompts[i % len(prompts)]
            # 添加随机变体
            variations = [
                ", professional agricultural photography, natural lighting",
                ", close-up view, shallow depth of field",
                ", wide angle field view, clear sky",
                ", detailed crop texture, high resolution",
                ", morning light, dew drops on leaves",
            ]
            prompt += random.choice(variations)

            generated = self.generate(prompt, height=img_size, width=img_size, num_images=1)
            images.extend(generated)

        return images


# ============================================================
# 基于现有图像的增强（备用方案，无需 GPU）
# ============================================================

class OfflineAugmenter:
    """基于现有图像的离线增强（当无法使用扩散模型时）"""

    def __init__(self):
        from torchvision import transforms
        self.transforms = transforms.Compose([
            transforms.RandomResizedCrop(336, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
            transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.8, 1.2), shear=10),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ])

    def augment_from_existing(self, source_dir, num_images=30):
        """从现有图像生成增强版本"""
        source_path = Path(source_dir)
        if not source_path.exists():
            return []

        source_images = []
        for img_path in source_path.glob("*"):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                source_images.append(img_path)

        if not source_images:
            return []

        augmented = []
        for i in range(num_images):
            src = random.choice(source_images)
            img = Image.open(src).convert("RGB")
            aug_img = self.transforms(img)
            augmented.append(aug_img)

        return augmented


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="扩散模型数据增强")
    parser.add_argument("--data-dir", default="dataset", help="数据集目录")
    parser.add_argument("--target-class", type=str, help="目标类别名（如 corn_filling）")
    parser.add_argument("--num-images", type=int, default=50, help="每类生成图像数")
    parser.add_argument("--auto", action="store_true", help="自动为所有少数类生成")
    parser.add_argument("--min-count", type=int, default=30, help="少于此数量的类别会被增强")
    parser.add_argument("--model", default="stabilityai/stable-diffusion-2-1", help="扩散模型名称")
    parser.add_argument("--offline", action="store_true", help="使用离线增强（无需GPU）")
    parser.add_argument("--output-dir", default=None, help="输出目录")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    # 确定要增强的类别
    train_dir = Path(args.data_dir) / "train"
    if not train_dir.exists():
        print(f"错误: 训练目录不存在 {train_dir}")
        return

    # 统计各类别数量
    class_counts = {}
    for cls_dir in train_dir.iterdir():
        if cls_dir.is_dir():
            count = len([f for f in cls_dir.glob("*")
                        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
            class_counts[cls_dir.name] = count

    print("各类别训练集数量:")
    for cls, count in sorted(class_counts.items()):
        marker = " <-- 需增强" if count < args.min_count else ""
        print(f"  {cls}: {count}{marker}")

    # 确定目标类别
    if args.target_class:
        target_classes = [args.target_class]
    elif args.auto:
        target_classes = [cls for cls, count in class_counts.items() if count < args.min_count]
    else:
        print("请指定 --target-class 或使用 --auto")
        return

    if not target_classes:
        print("没有需要增强的类别")
        return

    print(f"\n需要增强的类别: {target_classes}")

    # 输出目录
    if args.output_dir is None:
        args.output_dir = str(Path(args.data_dir) / "augmented")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 选择增强方式
    if args.offline:
        print("\n使用离线增强模式（基于现有图像变换）")
        augmenter = OfflineAugmenter()

        for cls_name in target_classes:
            print(f"\n处理 {cls_name}...")
            source_dir = train_dir / cls_name
            cls_output = output_dir / cls_name
            cls_output.mkdir(exist_ok=True)

            num_to_gen = max(0, args.num_images - class_counts.get(cls_name, 0))
            if num_to_gen <= 0:
                print(f"  跳过（已有足够图像）")
                continue

            images = augmenter.augment_from_existing(source_dir, num_to_gen)
            for i, img in enumerate(images):
                save_path = cls_output / f"aug_offline_{i:04d}.png"
                img.save(save_path)

            print(f"  生成 {len(images)} 张图像 -> {cls_output}")
    else:
        print(f"\n使用扩散模型: {args.model}")
        device = "cuda" if torch.cuda.is_available() and args.device == "auto" else "cpu"
        augmenter = DiffusionAugmenter(model_name=args.model, device=device)

        for cls_name in target_classes:
            print(f"\n为 {cls_name} 生成图像...")
            cls_output = output_dir / cls_name
            cls_output.mkdir(exist_ok=True)

            num_to_gen = max(0, args.num_images - class_counts.get(cls_name, 0))
            if num_to_gen <= 0:
                print(f"  跳过（已有足够图像）")
                continue

            images = augmenter.generate_for_class(cls_name, num_to_gen)
            for i, img in enumerate(images):
                save_path = cls_output / f"aug_diffusion_{i:04d}.png"
                img.save(save_path)

            print(f"  生成 {len(images)} 张图像 -> {cls_output}")

    # 保存增强配置
    config = {
        "target_classes": target_classes,
        "num_images_per_class": args.num_images,
        "min_count_threshold": args.min_count,
        "class_counts_before": class_counts,
        "method": "offline" if args.offline else "diffusion",
        "model": args.model if not args.offline else "offline_augmentation",
    }
    with open(output_dir / "aug_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"\n增强完成! 图像保存在: {output_dir}")
    print("将增强图像复制到 dataset/train/ 对应类别目录后即可用于训练")


if __name__ == "__main__":
    main()
