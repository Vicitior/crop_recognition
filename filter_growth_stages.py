# -*- coding: utf-8 -*-
"""过滤数据集，只保留作物生长阶段类（去掉病虫害和健康类）"""
import os
import shutil
from pathlib import Path

# 生长阶段类定义（与训练脚本一致）
CROP_GROWTH_STAGES = {
    "corn": ["seedling", "trefoil", "jointing", "booting", "heading",
             "silking", "grain_filling", "dough", "maturity"],
    "cotton": ["seedling", "true_leaf", "third_leaf", "budding", "squaring",
               "flowering", "boll_setting", "boll_opening", "full_opening", "defoliation"],
    "rapeseed": ["seedling", "true_leaf", "bolting", "budding", "flowering",
                 "pod_setting", "pod_filling", "ripening", "maturity"],
    "rice": ["seedling", "tillering", "jointing", "booting", "heading",
             "flowering", "grain_filling", "dough", "ripening", "maturity"],
    "soybean": ["seedling", "true_leaf", "branching", "budding", "flowering",
                "pod_setting", "grain_filling"],
    "wheat": ["seedling", "overwintering", "greening", "tillering", "jointing",
              "booting", "heading", "flowering", "grain_filling", "dough",
              "ripening", "maturity"],
}

# 病虫害关键词（排除这些）
DISEASE_KEYWORDS = [
    "healthy", "rust", "blight", "spot", "mold", "virus", "mildew",
    "scab", "rot", "measles", "esca", "leaf_spot", "leaf_blight",
    "leafroll", "haunglongbing", "greening", "spider", "mosaic",
    "yellow_leaf", "powdery", "bacterial", "cercospora", "northern",
    "gray_leaf", "common_rust", "target", "septoria", "scorch"
]

DATA_DIR = Path("D:/crop_datasets/unified")
OUTPUT_DIR = Path("D:/crop_datasets/unified_growth")

def is_growth_stage_class(dir_name):
    """判断是否为生长阶段类"""
    parts = dir_name.split("_", 1)
    if len(parts) < 2:
        return False
    crop, stage = parts[0], parts[1]

    # 检查是否为已定义的生长阶段
    if crop in CROP_GROWTH_STAGES:
        # 检查 stage 是否在定义的阶段列表中
        if stage in CROP_GROWTH_STAGES[crop]:
            return True

    # 排除明显是病虫害的类
    dir_lower = dir_name.lower()
    for kw in DISEASE_KEYWORDS:
        if kw in dir_lower:
            return False

    # 对于以作物名开头但不在定义中的，检查是否看起来像生长阶段
    if crop in CROP_GROWTH_STAGES:
        # 可能是新增的生长阶段名称，保留
        return True

    return False

def main():
    growth_classes = []
    removed_classes = []

    # 扫描所有类别
    all_classes = set()
    for split in ["train", "val", "test"]:
        split_dir = DATA_DIR / split
        if split_dir.exists():
            for d in split_dir.iterdir():
                if d.is_dir():
                    all_classes.add(d.name)

    for cls_name in sorted(all_classes):
        if is_growth_stage_class(cls_name):
            growth_classes.append(cls_name)
        else:
            removed_classes.append(cls_name)

    print(f"保留 {len(growth_classes)} 个生长阶段类，移除 {len(removed_classes)} 个病虫害/健康类")
    print(f"\n保留的类别:")
    for cls in growth_classes:
        print(f"  {cls}")
    print(f"\n移除的类别:")
    for cls in removed_classes:
        print(f"  {cls}")

    # 创建过滤后的数据集（使用符号链接节省空间）
    for split in ["train", "val", "test"]:
        src_split = DATA_DIR / split
        dst_split = OUTPUT_DIR / split
        if not src_split.exists():
            continue

        dst_split.mkdir(parents=True, exist_ok=True)

        for cls_name in growth_classes:
            src_cls = src_split / cls_name
            dst_cls = dst_split / cls_name
            if src_cls.exists() and not dst_cls.exists():
                # 使用 junction（Windows 符号链接）或复制
                try:
                    # 尝试创建 junction（不需要管理员权限）
                    os.system(f'mklink /J "{dst_cls}" "{src_cls}"')
                    print(f"  [链接] {split}/{cls_name}")
                except:
                    # 如果失败，复制目录
                    shutil.copytree(src_cls, dst_cls)
                    print(f"  [复制] {split}/{cls_name}")

    print(f"\n过滤后的数据集保存在: {OUTPUT_DIR}")

    # 统计
    for split in ["train", "val", "test"]:
        split_dir = OUTPUT_DIR / split
        if split_dir.exists():
            total = 0
            for cls_dir in split_dir.iterdir():
                if cls_dir.is_dir():
                    count = len(list(cls_dir.glob("*")))
                    total += count
            print(f"  {split}: {len(list(split_dir.iterdir()))} 类, {total} 张图片")

if __name__ == "__main__":
    main()
