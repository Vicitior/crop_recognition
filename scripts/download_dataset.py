"""
数据集构建脚本
帮助用户构建农作物图像数据集

使用方法：
1. 手动收集图片（推荐）
2. 运行此脚本进行目录整理和数据集划分

数据集来源推荐：
- PlantVillage: https://github.com/spMohanty/PlantVillage-Dataset
- Kaggle: 搜索 "crop" 或 "plant" 数据集
- 自行拍摄：手机拍摄不同生长阶段的农作物照片
- 网络搜索：通过搜索引擎收集各作物各阶段的参考图片

目录结构说明：
将收集到的图片放入 dataset/raw/ 目录下，按以下结构组织：
dataset/raw/
├── corn_seedling/          # 玉米-出苗期
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── corn_jointing/          # 玉米-拔节期
├── wheat_seedling/         # 小麦-出苗期
└── ...

然后运行此脚本，自动划分 train/val/test
"""
import os
import sys
import shutil
import random
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.growth_stages import CLASS_MAP, CROP_INFO


def print_dataset_guide():
    """打印数据集收集指南"""
    print("=" * 60)
    print("农作物图像数据集收集指南")
    print("=" * 60)
    print()
    print("请为以下每个类别收集至少 30 张图片（推荐 50-100 张）：")
    print()

    for class_name, info in sorted(CLASS_MAP.items()):
        print(f"  {class_name:30s} -> {info['crop_cn']} - {info['stage_cn']}")

    print()
    print("图片来源推荐：")
    print("  1. PlantVillage 数据集（GitHub免费下载）")
    print("  2. Kaggle 搜索 'crop identification' 数据集")
    print("  3. 使用手机在田间拍摄真实照片（最佳质量）")
    print("  4. 搜索引擎图片搜索（注意版权）")
    print()
    print("图片要求：")
    print("  - 每张图片尽量只包含一种作物的一个生长阶段")
    print("  - 图片清晰，光线充足")
    print("  - 不同角度、不同光照条件下拍摄")
    print("  - 推荐图片尺寸：至少 224x224 像素")
    print()


def create_class_dirs(base_dir):
    """创建所有类别目录"""
    for class_name in CLASS_MAP:
        dir_path = os.path.join(base_dir, class_name)
        os.makedirs(dir_path, exist_ok=True)
    print(f"已在 {base_dir} 下创建 {len(CLASS_MAP)} 个类别目录")


def split_dataset(raw_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    将原始数据集划分为 train/val/test

    Args:
        raw_dir: 原始图片目录
        output_dir: 输出目录（包含 train/val/test 子目录）
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    total_images = 0
    class_stats = {}

    for class_name in sorted(CLASS_MAP.keys()):
        raw_class_dir = os.path.join(raw_dir, class_name)
        if not os.path.isdir(raw_class_dir):
            print(f"  [跳过] {class_name} - 目录不存在")
            continue

        images = [f for f in os.listdir(raw_class_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]

        if len(images) == 0:
            print(f"  [跳过] {class_name} - 无图片")
            continue

        random.shuffle(images)
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }

        for split, split_images in splits.items():
            split_class_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(split_class_dir, exist_ok=True)
            for img_name in split_images:
                src = os.path.join(raw_class_dir, img_name)
                dst = os.path.join(split_class_dir, img_name)
                shutil.copy2(src, dst)

        total_images += n
        class_stats[class_name] = {
            "total": n,
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"])
        }
        print(f"  [完成] {class_name}: {n}张 -> train:{len(splits['train'])} val:{len(splits['val'])} test:{len(splits['test'])}")

    print()
    print(f"数据集划分完成！共处理 {total_images} 张图片")
    print(f"输出目录：{output_dir}")

    # 打印统计
    print()
    print("类别统计：")
    print(f"  {'类别':30s} {'总数':>6s} {'训练':>6s} {'验证':>6s} {'测试':>6s}")
    print("  " + "-" * 60)
    for class_name, stats in sorted(class_stats.items()):
        print(f"  {class_name:30s} {stats['total']:>6d} {stats['train']:>6d} {stats['val']:>6d} {stats['test']:>6d}")

    return class_stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="农作物图像数据集构建工具")
    parser.add_argument("--guide", action="store_true", help="显示数据集收集指南")
    parser.add_argument("--create-dirs", type=str, help="在指定目录下创建所有类别目录")
    parser.add_argument("--split", action="store_true", help="划分数据集")
    parser.add_argument("--raw-dir", type=str, default="dataset/raw", help="原始图片目录")
    parser.add_argument("--output-dir", type=str, default="dataset", help="输出目录")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    args = parser.parse_args()

    if args.guide:
        print_dataset_guide()
    elif args.create_dirs:
        create_class_dirs(args.create_dirs)
    elif args.split:
        split_dataset(
            args.raw_dir,
            args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=1 - args.train_ratio - args.val_ratio
        )
    else:
        print("请指定操作：")
        print("  --guide        显示数据集收集指南")
        print("  --create-dirs  创建类别目录")
        print("  --split        划分数据集")
        print()
        print("典型流程：")
        print("  1. python download_dataset.py --guide")
        print("  2. python download_dataset.py --create-dirs dataset/raw")
        print("  3. 将收集的图片放入 dataset/raw/ 对应目录")
        print("  4. python download_dataset.py --split")


if __name__ == "__main__":
    main()
