# -*- coding: utf-8 -*-
"""
整合玉米和小麦数据到 crop_recognition/dataset/
- 玉米: china-phenology 9阶段 → 5阶段
- 小麦: china-phenology 12阶段 + 生长周期 2阶段 → 5阶段
- HEIC 格式自动转换为 JPG
"""
import os
import sys
import shutil
import random
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# 路径配置
# ============================================================
PHENOLOGY_BASE = Path(r"D:\crop_datasets\raw\china-phenology\Six Crops")
WHEAT_GROWTH_BASE = Path(r"D:\crop_datasets\raw\生长周期\生长周期")
OUTPUT_BASE = Path(r"C:\Users\Vicitior\Desktop\新建文件夹 (4)\crop_recognition\dataset")

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.heic', '.HEIC'}

# 随机种子，保证可复现
random.seed(42)

# ============================================================
# 阶段映射：原始阶段 → 合并后的5阶段
# ============================================================
CORN_STAGE_MAP = {
    "01出苗期": "seedling",
    "02三叶期": "seedling",
    "03七叶期": "seedling",
    "04拔节期": "jointing",
    "05抽雄期": "tasseling",
    "06开花期": "tasseling",
    "07吐丝期": "tasseling",
    "08乳熟期": "filling",
    "09成熟期": "maturity",
}

WHEAT_STAGE_MAP = {
    "01出苗期": "seedling",
    "02三叶期": "seedling",
    "03分蘖": "seedling",
    "04越冬开始": "seedling",
    "05返青期": "tillering",
    "06起身期": "tillering",
    "07拔节期": "tillering",
    "08孕穗期": "jointing",
    "09抽穗期": "jointing",
    "10开花期": "heading",
    "11乳熟期": "heading",
    "12成熟期": "maturity",
}

# 生长周期数据集的映射（只有2个阶段）
WHEAT_GROWTH_MAP = {
    "开花期": "heading",
    "成熟期": "maturity",
}


def try_import_heif():
    """尝试导入 HEIC 支持"""
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
        print("  HEIC 支持已启用 (pillow-heif)")
        return True
    except ImportError:
        print("  [警告] pillow-heif 未安装，HEIC 文件将跳过")
        print("  安装命令: pip install pillow-heif")
        return False


def get_image_files(directory):
    """获取目录下所有图片文件"""
    files = []
    if not directory.exists():
        return files
    for f in directory.iterdir():
        if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.heic'}:
            files.append(f)
    return files


def convert_heic_to_jpg(src_path, dst_path):
    """将 HEIC 转换为 JPG"""
    try:
        from PIL import Image
        img = Image.open(src_path)
        img = img.convert("RGB")
        img.save(dst_path, "JPEG", quality=95)
        return True
    except Exception as e:
        print(f"    [转换失败] {src_path.name}: {e}")
        return False


def copy_or_convert(src_path, dst_dir, prefix=""):
    """复制图片，如果是 HEIC 则转换为 JPG"""
    ext = src_path.suffix.lower()
    if ext == '.heic':
        new_name = f"{prefix}{src_path.stem}.jpg"
        dst_path = dst_dir / new_name
        if dst_path.exists():
            return False
        return convert_heic_to_jpg(src_path, dst_path)
    else:
        new_name = f"{prefix}{src_path.name}"
        dst_path = dst_dir / new_name
        if dst_path.exists():
            return False
        shutil.copy2(src_path, dst_path)
        return True


def collect_images(base_dir, stage_map):
    """按合并后的阶段收集图片路径"""
    merged = {}  # stage_name -> [file_paths]
    for orig_stage, merged_stage in stage_map.items():
        stage_dir = base_dir / orig_stage
        if not stage_dir.exists():
            continue
        files = get_image_files(stage_dir)
        if merged_stage not in merged:
            merged[merged_stage] = []
        merged[merged_stage].extend(files)
    return merged


def split_and_copy(images, crop_prefix, output_base, train_ratio=0.7, val_ratio=0.15):
    """将图片按比例分割并复制到 train/val/test"""
    random.shuffle(images)
    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    counts = {}
    for split_name, split_files in splits.items():
        dst_dir = output_base / split_name / crop_prefix
        dst_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for src_path in split_files:
            if copy_or_convert(src_path, dst_dir, prefix=f"{crop_prefix}_"):
                count += 1
        counts[split_name] = count
    return counts


def main():
    print("=" * 60)
    print("整合玉米和小麦数据到 crop_recognition/dataset/")
    print("=" * 60)

    has_heif = try_import_heif()

    # 确保输出目录存在
    for split in ["train", "val", "test"]:
        (OUTPUT_BASE / split).mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 1. 玉米 (china-phenology)
    # ============================================================
    print("\n--- 玉米 (china-phenology) ---")
    corn_dir = PHENOLOGY_BASE / "03玉米"
    corn_images = collect_images(corn_dir, CORN_STAGE_MAP)

    total_corn = 0
    for stage in ["seedling", "jointing", "tasseling", "filling", "maturity"]:
        files = corn_images.get(stage, [])
        class_name = f"corn_{stage}"
        print(f"  {class_name}: {len(files)} 张原始图片")
        if files:
            counts = split_and_copy(files, class_name, OUTPUT_BASE)
            total_corn += sum(counts.values())
            print(f"    → train:{counts['train']} val:{counts['val']} test:{counts['test']}")

    print(f"  玉米总计: {total_corn} 张")

    # ============================================================
    # 2. 小麦 (china-phenology)
    # ============================================================
    print("\n--- 小麦 (china-phenology) ---")
    wheat_dir = PHENOLOGY_BASE / "02小麦"
    wheat_phenology = collect_images(wheat_dir, WHEAT_STAGE_MAP)

    for stage in ["seedling", "tillering", "jointing", "heading", "maturity"]:
        files = wheat_phenology.get(stage, [])
        print(f"  wheat_{stage}: {len(files)} 张 (china-phenology)")

    # ============================================================
    # 3. 小麦 (生长周期)
    # ============================================================
    print("\n--- 小麦 (生长周期) ---")
    wheat_growth = collect_images(WHEAT_GROWTH_BASE, WHEAT_GROWTH_MAP)

    for stage in ["seedling", "tillering", "jointing", "heading", "maturity"]:
        files = wheat_growth.get(stage, [])
        print(f"  wheat_{stage}: {len(files)} 张 (生长周期)")

    # ============================================================
    # 4. 合并小麦数据
    # ============================================================
    print("\n--- 合并小麦数据 ---")
    wheat_merged = {}
    for stage in ["seedling", "tillering", "jointing", "heading", "maturity"]:
        wheat_merged[stage] = []
        wheat_merged[stage].extend(wheat_phenology.get(stage, []))
        wheat_merged[stage].extend(wheat_growth.get(stage, []))

    total_wheat = 0
    for stage in ["seedling", "tillering", "jointing", "heading", "maturity"]:
        files = wheat_merged[stage]
        class_name = f"wheat_{stage}"
        print(f"  {class_name}: {len(files)} 张原始图片")
        if files:
            counts = split_and_copy(files, class_name, OUTPUT_BASE)
            total_wheat += sum(counts.values())
            print(f"    → train:{counts['train']} val:{counts['val']} test:{counts['test']}")

    print(f"  小麦总计: {total_wheat} 张")

    # ============================================================
    # 5. 最终统计
    # ============================================================
    print("\n" + "=" * 60)
    print("最终数据集统计")
    print("=" * 60)
    for split in ["train", "val", "test"]:
        split_dir = OUTPUT_BASE / split
        print(f"\n[{split}]")
        total = 0
        for cls_dir in sorted(split_dir.iterdir()):
            if cls_dir.is_dir():
                count = len(list(cls_dir.glob("*")))
                total += count
                status = "✓" if count >= 20 else ("△" if count >= 5 else "✗")
                print(f"  {status} {cls_dir.name}: {count} 张")
        print(f"  合计: {total} 张")

    # 检查是否有已存在的棉花数据
    cotton_count = 0
    for split in ["train", "val", "test"]:
        for cls_dir in (OUTPUT_BASE / split).glob("cotton_*"):
            if cls_dir.is_dir():
                cotton_count += len(list(cls_dir.glob("*")))
    if cotton_count > 0:
        print(f"\n棉花数据已存在: {cotton_count} 张（未修改）")

    print("\n完成！")


if __name__ == "__main__":
    main()
