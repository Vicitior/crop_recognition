# -*- coding: utf-8 -*-
"""
数据增强：为每类生成更多图片
使用多种增强策略：旋转、翻转、颜色抖动、裁剪、模糊等
"""
import os
import sys
import random
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

sys.stdout.reconfigure(encoding='utf-8')

SRC_DIR = Path("dataset/train")
DST_DIR = Path("dataset/train_augmented")
TARGET_PER_CLASS = 120  # 每类目标图片数（平衡棉花和玉米/小麦）

def augment_image(img, idx):
    """对一张图片应用随机增强"""
    augmented = img.copy()

    # 随机选择增强策略组合
    strategies = random.sample([
        'rotate', 'flip_h', 'flip_v', 'brightness', 'contrast',
        'saturation', 'blur', 'sharpen', 'crop', 'equalize',
        'posterize', 'solarize', 'color_jitter'
    ], k=random.randint(2, 4))

    for strategy in strategies:
        if strategy == 'rotate':
            angle = random.choice([90, 180, 270, -15, -30, 15, 30])
            augmented = augmented.rotate(angle, expand=True, fillcolor=(128, 128, 128))
        elif strategy == 'flip_h':
            augmented = ImageOps.mirror(augmented)
        elif strategy == 'flip_v':
            augmented = ImageOps.flip(augmented)
        elif strategy == 'brightness':
            factor = random.uniform(0.6, 1.4)
            augmented = ImageEnhance.Brightness(augmented).enhance(factor)
        elif strategy == 'contrast':
            factor = random.uniform(0.6, 1.4)
            augmented = ImageEnhance.Contrast(augmented).enhance(factor)
        elif strategy == 'saturation':
            factor = random.uniform(0.6, 1.4)
            augmented = ImageEnhance.Color(augmented).enhance(factor)
        elif strategy == 'blur':
            augmented = augmented.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))
        elif strategy == 'sharpen':
            augmented = augmented.filter(ImageFilter.SHARPEN)
        elif strategy == 'crop':
            w, h = augmented.size
            crop_ratio = random.uniform(0.7, 0.9)
            new_w, new_h = int(w * crop_ratio), int(h * crop_ratio)
            left = random.randint(0, w - new_w)
            top = random.randint(0, h - new_h)
            augmented = augmented.crop((left, top, left + new_w, top + new_h))
            augmented = augmented.resize((w, h), Image.LANCZOS)
        elif strategy == 'equalize':
            augmented = ImageOps.equalize(augmented)
        elif strategy == 'posterize':
            bits = random.randint(2, 6)
            augmented = ImageOps.posterize(augmented, bits)
        elif strategy == 'solarize':
            threshold = random.randint(64, 192)
            augmented = ImageOps.solarize(augmented, threshold)
        elif strategy == 'color_jitter':
            r_factor = random.uniform(0.8, 1.2)
            g_factor = random.uniform(0.8, 1.2)
            b_factor = random.uniform(0.8, 1.2)
            r, g, b = augmented.split()
            r = ImageEnhance.Brightness(r).enhance(r_factor)
            g = ImageEnhance.Brightness(g).enhance(g_factor)
            b = ImageEnhance.Brightness(b).enhance(b_factor)
            augmented = Image.merge('RGB', (r, g, b))

    # 确保尺寸一致
    augmented = augmented.resize(img.size, Image.LANCZOS)
    return augmented

def main():
    total_generated = 0

    for cls_dir in sorted(SRC_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue

        cls_name = cls_dir.name
        dst_cls = DST_DIR / cls_name
        dst_cls.mkdir(parents=True, exist_ok=True)

        # 复制原始图片
        original_images = sorted([
            f for f in cls_dir.iterdir()
            if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        ])

        for img_path in original_images:
            dst_path = dst_cls / img_path.name
            if not dst_path.exists():
                img = Image.open(img_path).convert("RGB")
                img.save(dst_path)

        current_count = len(original_images)

        if current_count >= TARGET_PER_CLASS:
            print(f"  [跳过] {cls_name}: 已有 {current_count} 张")
            continue

        needed = TARGET_PER_CLASS - current_count
        print(f"  [增强] {cls_name}: {current_count} -> {TARGET_PER_CLASS} (生成 {needed} 张)")

        generated = 0
        for i in range(needed):
            # 随机选择一张原始图片作为基础
            src_img_path = random.choice(original_images)
            try:
                img = Image.open(src_img_path).convert("RGB")
                # 统一尺寸
                img = img.resize((336, 336), Image.LANCZOS)
                aug_img = augment_image(img, i)
                aug_path = dst_cls / f"aug_{i:04d}_{src_img_path.stem}.jpg"
                aug_img.save(aug_path, quality=95)
                generated += 1
            except Exception as e:
                print(f"    [错误] {src_img_path.name}: {e}")

        total_generated += generated
        final_count = len(list(dst_cls.glob("*")))
        print(f"    完成: {final_count} 张")

    print(f"\n总计生成 {total_generated} 张增强图片")
    print(f"增强数据集保存在: {DST_DIR}")

    # 统计
    print("\n=== 增强后的数据集统计 ===")
    total = 0
    for cls_dir in sorted(DST_DIR.iterdir()):
        if cls_dir.is_dir():
            count = len(list(cls_dir.glob("*")))
            total += count
    print(f"  总类别: {len(list(DST_DIR.iterdir()))}")
    print(f"  总图片: {total}")

if __name__ == "__main__":
    main()
