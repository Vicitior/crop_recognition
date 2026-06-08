# -*- coding: utf-8 -*-
"""
合并爬取的图片和增强的数据集，创建最终训练数据集
"""
import os
import sys
import shutil
from pathlib import Path
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

CRAWLED_DIR = Path("D:/crop_datasets/crawled")
AUGMENTED_DIR = Path("D:/crop_datasets/unified_growth_augmented/train")
ORIGINAL_DIR = Path("D:/crop_datasets/unified_growth")
OUTPUT_DIR = Path("D:/crop_datasets/unified_growth_final")

def is_valid_image(path):
    """检查图片是否有效"""
    try:
        img = Image.open(path)
        img.verify()
        return True
    except:
        return False

def copy_images(src_dir, dst_dir, prefix=""):
    """复制图片到目标目录"""
    count = 0
    for img_path in src_dir.iterdir():
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
            continue
        if not is_valid_image(img_path):
            continue
        dst_path = dst_dir / f"{prefix}{img_path.name}"
        if not dst_path.exists():
            shutil.copy2(img_path, dst_path)
            count += 1
    return count

def main():
    total_copied = 0

    # 获取所有类别
    all_classes = set()

    # 从原始数据获取类别
    for split in ["train", "val", "test"]:
        split_dir = ORIGINAL_DIR / split
        if split_dir.exists():
            for cls_dir in split_dir.iterdir():
                if cls_dir.is_dir():
                    all_classes.add(cls_dir.name)

    # 从增强数据获取类别
    if AUGMENTED_DIR.exists():
        for cls_dir in AUGMENTED_DIR.iterdir():
            if cls_dir.is_dir():
                all_classes.add(cls_dir.name)

    # 从爬取数据获取类别
    if CRAWLED_DIR.exists():
        for cls_dir in CRAWLED_DIR.iterdir():
            if cls_dir.is_dir():
                all_classes.add(cls_dir.name)

    print(f"总类别数: {len(all_classes)}")

    # 创建输出目录结构
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)

    for cls_name in sorted(all_classes):
        # 复制原始数据
        for split in ["train", "val", "test"]:
            src_cls = ORIGINAL_DIR / split / cls_name
            dst_cls = OUTPUT_DIR / split / cls_name
            dst_cls.mkdir(parents=True, exist_ok=True)

            if src_cls.exists():
                count = copy_images(src_cls, dst_cls, prefix="orig_")
                total_copied += count

        # 复制增强数据（只到 train）
        aug_cls = AUGMENTED_DIR / cls_name
        dst_train = OUTPUT_DIR / "train" / cls_name
        if aug_cls.exists():
            count = copy_images(aug_cls, dst_train, prefix="aug_")
            total_copied += count

        # 复制爬取数据（按 70/15/15 分配到 train/val/test）
        crawled_cls = CRAWLED_DIR / cls_name
        if crawled_cls.exists():
            images = sorted([
                f for f in crawled_cls.iterdir()
                if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
            ])

            n = len(images)
            splits = {
                "train": images[:int(n * 0.7)],
                "val": images[int(n * 0.7):int(n * 0.85)],
                "test": images[int(n * 0.85):],
            }

            for split, split_images in splits.items():
                dst_cls = OUTPUT_DIR / split / cls_name
                dst_cls.mkdir(parents=True, exist_ok=True)
                for img_path in split_images:
                    if not is_valid_image(img_path):
                        continue
                    dst_path = dst_cls / f"crawled_{img_path.name}"
                    if not dst_path.exists():
                        shutil.copy2(img_path, dst_path)
                        total_copied += 1

    print(f"总计复制 {total_copied} 张图片")

    # 统计最终数据集
    print("\n=== 最终数据集统计 ===")
    for split in ["train", "val", "test"]:
        split_dir = OUTPUT_DIR / split
        if split_dir.exists():
            total = 0
            classes = 0
            for cls_dir in sorted(split_dir.iterdir()):
                if cls_dir.is_dir():
                    count = len(list(cls_dir.glob("*")))
                    total += count
                    classes += 1
                    if count < 10:
                        print(f"  [警告] {split}/{cls_dir.name}: 仅 {count} 张")
            print(f"  {split}: {classes} 类, {total} 张图片")

    # 保存类别列表
    classes_list = sorted(all_classes)
    with open(OUTPUT_DIR / "classes.txt", "w", encoding="utf-8") as f:
        for cls in classes_list:
            f.write(f"{cls}\n")
    print(f"\n类别列表保存到: {OUTPUT_DIR / 'classes.txt'}")

if __name__ == "__main__":
    main()
