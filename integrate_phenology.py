# -*- coding: utf-8 -*-
"""将 china-phenology 数据集整合到 unified_growth 数据集中"""
import os
import shutil
from pathlib import Path

# china-phenology 中文阶段名 -> 英文阶段名映射
PHENOLOGY_STAGE_MAP = {
    # 水稻 (Rice)
    "rice": {
        "01出苗期": "seedling",
        "02三叶期": "trefoil",
        "03移栽期": "transplanting",
        "04返青期": "greening",
        "05分蘖期": "tillering",
        "06拔节期": "jointing",
        "07孕穗期": "booting",
        "08抽穗期": "heading",
        "09乳熟期": "grain_filling",
        "10成熟期": "maturity",
    },
    # 小麦 (Wheat)
    "wheat": {
        "01出苗期": "seedling",
        "02三叶期": "trefoil",
        "03分蘖": "tillering",
        "04越冬开始": "overwintering",
        "05返青期": "greening",
        "06起身期": "rising",
        "07拔节期": "jointing",
        "08孕穗期": "booting",
        "09抽穗期": "heading",
        "10开花期": "flowering",
        "11乳熟期": "grain_filling",
        "12成熟期": "maturity",
    },
    # 玉米 (Corn)
    "corn": {
        "01出苗期": "seedling",
        "02三叶期": "trefoil",
        "03七叶期": "seven_leaf",
        "04拔节期": "jointing",
        "05抽雄期": "tasseling",
        "06开花期": "flowering",
        "07吐丝期": "silking",
        "08乳熟期": "grain_filling",
        "09成熟期": "maturity",
    },
    # 棉花 (Cotton)
    "cotton": {
        "01出苗期": "seedling",
        "02三真叶期": "true_leaf",
        "03五真叶期": "five_leaf",
        "04现蕾期": "squaring",
        "05开花期": "flowering",
        "06开花盛期": "full_flowering",
        "07裂铃期": "boll_cracking",
        "08吐絮期": "boll_opening",
        "09吐絮盛期": "full_opening",
        "10停止生长期": "defoliation",
    },
    # 油菜 (Rapeseed)
    "rapeseed": {
        "01出苗期": "seedling",
        "02五真叶期": "five_leaf",
        "03移栽成活期": "transplant_survival",
        "04现蕾期": "squaring",
        "05抽薹期": "bolting",
        "06开花期": "flowering",
        "07开花盛期": "full_flowering",
        "08绿熟期": "green_maturity",
        "09成熟期": "maturity",
    },
    # 大豆 (Soybean)
    "soybean": {
        "01出苗期": "seedling",
        "02三真叶期": "true_leaf",
        "03分枝期": "branching",
        "04开花期": "flowering",
        "05结荚期": "pod_setting",
        "06鼓粒期": "pod_filling",
        "07成熟期": "maturity",
    },
}

SRC_DIR = Path("D:/crop_datasets/raw/china-phenology/Six Crops")
DST_DIR = Path("D:/crop_datasets/unified_growth")

CROP_NAME_MAP = {
    "01水稻": "rice",
    "02小麦": "wheat",
    "03玉米": "corn",
    "04棉花": "cotton",
    "05油菜": "rapeseed",
    "06大豆": "soybean",
}

def main():
    total_added = 0

    for crop_cn, crop_en in CROP_NAME_MAP.items():
        crop_src = SRC_DIR / crop_cn
        if not crop_src.exists():
            print(f"[跳过] {crop_cn} 不存在")
            continue

        stage_map = PHENOLOGY_STAGE_MAP[crop_en]

        for stage_cn, stage_en in stage_map.items():
            stage_src = crop_src / stage_cn
            if not stage_src.exists():
                print(f"[跳过] {crop_cn}/{stage_cn} 不存在")
                continue

            target_class = f"{crop_en}_{stage_en}"

            for split in ["train", "val", "test"]:
                dst_cls = DST_DIR / split / target_class
                if not dst_cls.exists():
                    dst_cls.mkdir(parents=True, exist_ok=True)

                # 复制图片（分配到 train: 70%, val: 15%, test: 15%）
                images = sorted([
                    f for f in stage_src.iterdir()
                    if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
                ])

                if not images:
                    continue

                n = len(images)
                if split == "train":
                    split_images = images[:int(n * 0.7)]
                elif split == "val":
                    split_images = images[int(n * 0.7):int(n * 0.85)]
                else:
                    split_images = images[int(n * 0.85):]

                for img in split_images:
                    dst_path = dst_cls / img.name
                    if not dst_path.exists():
                        shutil.copy2(img, dst_path)
                        total_added += 1

            print(f"[添加] {crop_cn}/{stage_cn} -> {target_class}")

    print(f"\n总计添加 {total_added} 张图片")

    # 统计更新后的数据集
    print("\n=== 更新后的数据集统计 ===")
    for split in ["train", "val", "test"]:
        split_dir = DST_DIR / split
        if split_dir.exists():
            total = 0
            classes = 0
            for cls_dir in sorted(split_dir.iterdir()):
                if cls_dir.is_dir():
                    count = len(list(cls_dir.glob('*')))
                    total += count
                    classes += 1
            print(f"  {split}: {classes} 类, {total} 张图片")

if __name__ == "__main__":
    main()
