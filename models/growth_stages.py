"""
农作物生长阶段知识库
定义3种常见农作物的生长阶段及其特征描述
"""

CROP_INFO = {
    "corn": {
        "name_cn": "玉米",
        "total_days": "100-120天",
        "stages": {
            "seedling": {
                "name_cn": "出苗期",
                "days": "0-15天",
                "description": "种子发芽出土，长出2-3片真叶，植株矮小，叶片嫩绿"
            },
            "jointing": {
                "name_cn": "拔节期",
                "days": "15-40天",
                "description": "茎节开始伸长，叶片增多，植株快速长高，可见明显的节间"
            },
            "tasseling": {
                "name_cn": "抽穗期",
                "days": "40-65天",
                "description": "雄穗抽出，雌穗吐丝，是玉米开花授粉的关键时期"
            },
            "filling": {
                "name_cn": "灌浆期",
                "days": "65-95天",
                "description": "籽粒开始灌浆充实，由乳熟期过渡到蜡熟期，籽粒逐渐饱满"
            },
            "maturity": {
                "name_cn": "成熟期",
                "days": "95-120天",
                "description": "籽粒完全成熟，苞叶变黄，籽粒变硬，黑层形成，可收获"
            }
        }
    },
    "wheat": {
        "name_cn": "小麦",
        "total_days": "180-230天（冬小麦）",
        "stages": {
            "seedling": {
                "name_cn": "出苗期",
                "days": "0-15天",
                "description": "种子萌发出土，长出第一片真叶，幼苗嫩绿细小"
            },
            "tillering": {
                "name_cn": "分蘖期",
                "days": "15-90天",
                "description": "从基部节上长出分蘖，植株丛生，叶片增多，形成分蘖丛"
            },
            "jointing": {
                "name_cn": "拔节期",
                "days": "90-130天",
                "description": "茎节伸长，植株快速长高，茎秆明显可见节间，叶片挺立"
            },
            "heading": {
                "name_cn": "抽穗期",
                "days": "130-150天",
                "description": "麦穗从旗叶鞘中抽出，穗部可见小花，开始开花授粉"
            },
            "maturity": {
                "name_cn": "成熟期",
                "days": "150-230天",
                "description": "籽粒饱满变硬，穗部变黄下垂，茎秆枯黄，蜡熟末期可收获"
            }
        }
    },
    "cotton": {
        "name_cn": "棉花",
        "total_days": "150-180天",
        "stages": {
            "seedling": {
                "name_cn": "苗期",
                "days": "0-20天",
                "description": "子叶展开，长出真叶，植株矮小，茎秆细弱"
            },
            "squaring": {
                "name_cn": "蕾期",
                "days": "20-50天",
                "description": "开始出现花蕾，叶片增多增大，植株分枝增多"
            },
            "flowering": {
                "name_cn": "开花期",
                "days": "50-80天",
                "description": "花朵开放，花冠乳白色或淡黄色，逐渐变红凋萎"
            },
            "boll_setting": {
                "name_cn": "结铃期",
                "days": "80-130天",
                "description": "棉铃发育膨大，植株上可见大小不一的棉铃"
            },
            "boll_opening": {
                "name_cn": "吐絮期",
                "days": "130-180天",
                "description": "棉铃开裂吐出白色棉絮，可分批采收棉花"
            }
        }
    }
}

# 类别映射：文件夹名 -> (作物英文名, 阶段英文名)
CLASS_MAP = {}
idx = 0
for crop_en, crop_data in CROP_INFO.items():
    for stage_en in crop_data["stages"]:
        class_name = f"{crop_en}_{stage_en}"
        CLASS_MAP[class_name] = {
            "index": idx,
            "crop_en": crop_en,
            "crop_cn": crop_data["name_cn"],
            "stage_en": stage_en,
            "stage_cn": crop_data["stages"][stage_en]["name_cn"],
            "description": crop_data["stages"][stage_en]["description"],
            "days": crop_data["stages"][stage_en]["days"],
            "total_days": crop_data["total_days"]
        }
        idx += 1

NUM_CLASSES = len(CLASS_MAP)

# ============================================================
# 序数索引映射：全局类别索引 → 作物内阶段序号
# 用于 Ordinal Loss 计算高斯软标签
# ============================================================
CROP_STAGE_ORDINAL = {}
for crop_en, crop_data in CROP_INFO.items():
    for stage_idx, stage_en in enumerate(crop_data["stages"]):
        class_name = f"{crop_en}_{stage_en}"
        global_idx = CLASS_MAP[class_name]["index"]
        CROP_STAGE_ORDINAL[global_idx] = stage_idx

# 每种作物的阶段数
CROP_NUM_STAGES = {crop: len(data["stages"]) for crop, data in CROP_INFO.items()}

# ============================================================
# 生育期关系矩阵（用于 Phenology-aware 建模）
# 高斯邻接矩阵：相邻阶段关系强，远距离关系弱
# adjacency[i][j] = exp(-|i-j|^2 / (2*sigma^2))
# ============================================================
import math

def _build_gaussian_adjacency(num_stages, sigma=1.5):
    """构建高斯邻接矩阵"""
    adj = []
    for i in range(num_stages):
        row = []
        for j in range(num_stages):
            diff = abs(i - j)
            val = math.exp(-diff ** 2 / (2 * sigma ** 2))
            row.append(round(val, 4))
        adj.append(row)
    return adj

# 各作物的生育期邻接矩阵（sigma=1.5）
# 阶段顺序：seedling(0) → jointing/tillering/squaring(1) → tasseling/heading/flowering(2) → filling/boll_setting(3) → maturity/boll_opening(4)
CROP_STAGE_ADJACENCY = {
    'corn': _build_gaussian_adjacency(5, sigma=1.5),
    'wheat': _build_gaussian_adjacency(5, sigma=1.5),
    'cotton': _build_gaussian_adjacency(5, sigma=1.5),
}

# 生育期阶段的农业知识（用于知识增强）
# GDD: 积温需求 (Growing Degree Days)
# LAI: 叶面积指数 (Leaf Area Index)
# growth_rate: 相对生长速率 (0-1)
PHENOLOGY_KNOWLEDGE = {
    'corn': {
        'seedling':  {'gdd': (0, 200),    'lai': 0.3, 'growth_rate': 0.2},
        'jointing':  {'gdd': (200, 600),  'lai': 2.5, 'growth_rate': 0.8},
        'tasseling': {'gdd': (600, 1000), 'lai': 5.0, 'growth_rate': 0.6},
        'filling':   {'gdd': (1000, 1600),'lai': 4.0, 'growth_rate': 0.4},
        'maturity':  {'gdd': (1600, 2200),'lai': 1.5, 'growth_rate': 0.1},
    },
    'wheat': {
        'seedling':  {'gdd': (0, 150),    'lai': 0.2, 'growth_rate': 0.15},
        'tillering': {'gdd': (150, 500),  'lai': 1.5, 'growth_rate': 0.5},
        'jointing':  {'gdd': (500, 900),  'lai': 4.0, 'growth_rate': 0.9},
        'heading':   {'gdd': (900, 1200), 'lai': 5.5, 'growth_rate': 0.7},
        'maturity':  {'gdd': (1200, 1800),'lai': 2.0, 'growth_rate': 0.1},
    },
    'cotton': {
        'seedling':    {'gdd': (0, 300),    'lai': 0.2, 'growth_rate': 0.15},
        'squaring':    {'gdd': (300, 800),  'lai': 1.8, 'growth_rate': 0.6},
        'flowering':   {'gdd': (800, 1300), 'lai': 3.5, 'growth_rate': 0.8},
        'boll_setting':{'gdd': (1300, 2000),'lai': 4.0, 'growth_rate': 0.5},
        'boll_opening':{'gdd': (2000, 2800),'lai': 2.0, 'growth_rate': 0.1},
    },
}

# 作物视觉复杂度（用于 Adaptive LoRA rank 选择）
# 复杂度越高，需要的 LoRA rank 越大
CROP_VISUAL_COMPLEXITY = {
    'corn':  {'complexity': 'low',    'suggested_rank': 4,
              'reason': '形态变化明显，各阶段易区分'},
    'wheat': {'complexity': 'medium', 'suggested_rank': 8,
              'reason': '与玉米有相似阶段（如拔节期），需中等参数'},
    'cotton': {'complexity': 'high',  'suggested_rank': 16,
               'reason': '蕾/花/铃形态接近，需更多参数区分'},
}

def get_ordinal_label(global_idx):
    """全局类别索引 → 该作物内的阶段序号 (0-based)"""
    return CROP_STAGE_ORDINAL.get(global_idx, global_idx)

def get_num_stages_for_crop(crop_en):
    """获取某种作物的生长阶段数"""
    return CROP_NUM_STAGES.get(crop_en, 5)

def get_class_names():
    """返回所有类别名称列表，按索引排序"""
    return [k for k, v in sorted(CLASS_MAP.items(), key=lambda x: x[1]["index"])]

def get_crop_info(class_name):
    """根据类别名获取完整信息"""
    if class_name in CLASS_MAP:
        info = CLASS_MAP[class_name]
        crop = CROP_INFO[info["crop_en"]]
        stage = crop["stages"][info["stage_en"]]
        return {
            "crop_name": info["crop_cn"],
            "stage_name": stage["name_cn"],
            "stage_days": stage["days"],
            "total_days": crop["total_days"],
            "description": stage["description"]
        }
    return None
