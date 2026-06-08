# -*- coding: utf-8 -*-
"""
爬取作物生长阶段图片，扩充训练数据集
使用 Bing 图片搜索，中英文关键词混合搜索
"""
import os
import sys
import time
import shutil
import hashlib
from pathlib import Path
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

from icrawler.builtin import BingImageCrawler

# 搜索关键词映射：class_name -> [中文关键词, 英文关键词]
SEARCH_QUERIES = {
    # 玉米
    "corn_seedling": ["玉米幼苗", "corn seedling plant"],
    "corn_trefoil": ["玉米三叶期", "corn three leaf stage"],
    "corn_seven_leaf": ["玉米七叶期", "corn seven leaf"],
    "corn_jointing": ["玉米拔节期", "corn jointing stage"],
    "corn_tasseling": ["玉米抽雄期", "corn tasseling"],
    "corn_flowering": ["玉米开花", "corn flowering"],
    "corn_silking": ["玉米吐丝期", "corn silking"],
    "corn_grain_filling": ["玉米灌浆期", "corn grain filling"],
    "corn_maturity": ["玉米成熟", "corn maturity harvest"],
    "corn_leaf": ["玉米叶片", "corn leaf"],
    # 棉花
    "cotton_seedling": ["棉花幼苗", "cotton seedling"],
    "cotton_true_leaf": ["棉花真叶", "cotton true leaf"],
    "cotton_five_leaf": ["棉花五叶期", "cotton five leaf"],
    "cotton_squaring": ["棉花现蕾期", "cotton squaring bud"],
    "cotton_flowering": ["棉花开花", "cotton flowering"],
    "cotton_full_flowering": ["棉花盛花期", "cotton full flowering"],
    "cotton_boll_cracking": ["棉花裂铃", "cotton boll cracking"],
    "cotton_boll_opening": ["棉花吐絮", "cotton boll opening"],
    "cotton_full_opening": ["棉花吐絮盛期", "cotton full boll opening"],
    "cotton_defoliation": ["棉花落叶成熟", "cotton defoliation maturity"],
    # 油菜
    "rapeseed_seedling": ["油菜幼苗", "rapeseed canola seedling"],
    "rapeseed_five_leaf": ["油菜五叶期", "rapeseed five leaf"],
    "rapeseed_transplant_survival": ["油菜移栽", "rapeseed transplant"],
    "rapeseed_squaring": ["油菜现蕾", "rapeseed squaring bud"],
    "rapeseed_bolting": ["油菜抽薹", "rapeseed bolting"],
    "rapeseed_flowering": ["油菜开花", "rapeseed canola flowering"],
    "rapeseed_full_flowering": ["油菜盛花期", "rapeseed full flowering"],
    "rapeseed_green_maturity": ["油菜绿熟期", "rapeseed green maturity"],
    "rapeseed_maturity": ["油菜成熟", "rapeseed maturity harvest"],
    # 水稻
    "rice_seedling": ["水稻秧苗", "rice seedling"],
    "rice_trefoil": ["水稻三叶期", "rice three leaf"],
    "rice_transplanting": ["水稻移栽", "rice transplanting"],
    "rice_greening": ["水稻返青期", "rice greening reviving"],
    "rice_tillering": ["水稻分蘖期", "rice tillering"],
    "rice_jointing": ["水稻拔节期", "rice jointing"],
    "rice_booting": ["水稻孕穗期", "rice booting"],
    "rice_heading": ["水稻抽穗", "rice heading panicle"],
    "rice_grain_filling": ["水稻灌浆", "rice grain filling"],
    "rice_maturity": ["水稻成熟", "rice maturity harvest"],
    # 大豆
    "soybean_seedling": ["大豆幼苗", "soybean seedling"],
    "soybean_true_leaf": ["大豆真叶", "soybean true leaf"],
    "soybean_branching": ["大豆分枝期", "soybean branching"],
    "soybean_flowering": ["大豆开花", "soybean flowering"],
    "soybean_pod_setting": ["大豆结荚", "soybean pod setting"],
    "soybean_pod_filling": ["大豆鼓粒", "soybean pod filling grain"],
    "soybean_maturity": ["大豆成熟", "soybean maturity harvest"],
    # 小麦
    "wheat_seedling": ["小麦幼苗", "wheat seedling"],
    "wheat_trefoil": ["小麦三叶期", "wheat three leaf"],
    "wheat_overwintering": ["小麦越冬期", "wheat overwintering"],
    "wheat_rising": ["小麦起身期", "wheat rising"],
    "wheat_greening": ["小麦返青期", "wheat greening reviving"],
    "wheat_tillering": ["小麦分蘖期", "wheat tillering"],
    "wheat_jointing": ["小麦拔节期", "wheat jointing"],
    "wheat_booting": ["小麦孕穗期", "wheat booting"],
    "wheat_heading": ["小麦抽穗", "wheat heading"],
    "wheat_flowering": ["小麦开花", "wheat flowering"],
    "wheat_grain_filling": ["小麦灌浆", "wheat grain filling"],
    "wheat_dough": ["小麦面团期", "wheat dough stage"],
    "wheat_maturity": ["小麦成熟", "wheat maturity harvest"],
    "wheat_trefoil": ["小麦三叶期", "wheat trefoil three leaf"],
}

OUTPUT_DIR = Path("D:/crop_datasets/crawled")
TARGET_PER_CLASS = 50  # 每类目标爬取数量

def is_valid_image(path):
    """检查图片是否有效"""
    try:
        img = Image.open(path)
        img.verify()
        return True
    except:
        return False

def crawl_class(class_name, queries, max_num=30):
    """为一个类别爬取图片"""
    save_dir = OUTPUT_DIR / class_name
    save_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(save_dir.glob("*")))
    if existing >= TARGET_PER_CLASS:
        print(f"  [跳过] {class_name}: 已有 {existing} 张")
        return 0

    needed = TARGET_PER_CLASS - existing
    total_added = 0

    for query in queries:
        if total_added >= needed:
            break

        crawler = BingImageCrawler(
            storage={'root_dir': str(save_dir)},
            log_level=40  # ERROR only
        )
        try:
            crawler.crawl(
                keyword=query,
                max_num=min(needed - total_added + 5, 20),  # 多爬几张，后面去重
                min_size=(100, 100),
                file_idx_offset='auto'
            )
            time.sleep(2)  # 避免请求太快
        except Exception as e:
            print(f"    [错误] 搜索 '{query}' 失败: {e}")

    # 验证和去重
    valid_count = 0
    seen_hashes = set()
    for img_path in sorted(save_dir.glob("*")):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
            img_path.unlink()
            continue

        # 检查哈希去重
        h = hashlib.md5(img_path.read_bytes()).hexdigest()
        if h in seen_hashes:
            img_path.unlink()
            continue
        seen_hashes.add(h)

        # 验证图片
        if not is_valid_image(img_path):
            img_path.unlink()
            continue

        valid_count += 1

    added = valid_count - existing
    return max(0, added)

def main():
    total_added = 0

    # 只爬取数据量少的类别
    unified_train = Path("D:/crop_datasets/unified_growth/train")
    classes_to_crawl = []

    for cls_dir in sorted(unified_train.iterdir()):
        if cls_dir.is_dir():
            count = len(list(cls_dir.glob("*")))
            if count < 30:  # 少于30张的类别需要爬取
                classes_to_crawl.append((cls_dir.name, count))

    print(f"需要爬取 {len(classes_to_crawl)} 个类别（图片数 < 30）")
    for cls, count in classes_to_crawl:
        print(f"  {cls}: {count} 张")

    for class_name, current_count in classes_to_crawl:
        if class_name not in SEARCH_QUERIES:
            print(f"  [跳过] {class_name}: 无搜索关键词")
            continue

        print(f"\n[爬取] {class_name} (当前 {current_count} 张)...")
        queries = SEARCH_QUERIES[class_name]
        added = crawl_class(class_name, queries)
        total_added += added
        print(f"  新增 {added} 张")

    print(f"\n总计新增 {total_added} 张图片")
    print(f"爬取的图片保存在: {OUTPUT_DIR}")

    # 统计
    print("\n=== 爬取后的类别统计 ===")
    for cls_dir in sorted(OUTPUT_DIR.iterdir()):
        if cls_dir.is_dir():
            count = len(list(cls_dir.glob("*")))
            if count > 0:
                print(f"  {cls_dir.name}: {count} 张")

if __name__ == "__main__":
    main()
