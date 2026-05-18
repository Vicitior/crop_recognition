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
