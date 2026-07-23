import os
import io
import sys
import zipfile
import argparse
import json
from datetime import datetime
from pathlib import Path

# 配置控制台输出 UTF-8 编码
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch
import gradio as gr
from PIL import Image

# 注册 HEIC 格式支持
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

from models.growth_stages import CROP_INFO, CLASS_MAP


# 全局变量
classifier = None
mode = "clip"
current_model_key = None  # 当前已加载的模型key

# 作物类型和生长阶段映射
CROP_STAGE_MAP = {
    "corn_seedling": {"crop_cn": "玉米", "stage_cn": "出苗期", "crop_en": "corn", "stage_en": "seedling"},
    "corn_jointing": {"crop_cn": "玉米", "stage_cn": "拔节期", "crop_en": "corn", "stage_en": "jointing"},
    "corn_tasseling": {"crop_cn": "玉米", "stage_cn": "抽穗期", "crop_en": "corn", "stage_en": "tasseling"},
    "corn_filling": {"crop_cn": "玉米", "stage_cn": "灌浆期", "crop_en": "corn", "stage_en": "filling"},
    "corn_maturity": {"crop_cn": "玉米", "stage_cn": "成熟期", "crop_en": "corn", "stage_en": "maturity"},
    "wheat_seedling": {"crop_cn": "小麦", "stage_cn": "出苗期", "crop_en": "wheat", "stage_en": "seedling"},
    "wheat_tillering": {"crop_cn": "小麦", "stage_cn": "分蘖期", "crop_en": "wheat", "stage_en": "tillering"},
    "wheat_jointing": {"crop_cn": "小麦", "stage_cn": "拔节期", "crop_en": "wheat", "stage_en": "jointing"},
    "wheat_heading": {"crop_cn": "小麦", "stage_cn": "抽穗期", "crop_en": "wheat", "stage_en": "heading"},
    "wheat_maturity": {"crop_cn": "小麦", "stage_cn": "成熟期", "crop_en": "wheat", "stage_en": "maturity"},
    "cotton_seedling": {"crop_cn": "棉花", "stage_cn": "苗期", "crop_en": "cotton", "stage_en": "seedling"},
    "cotton_squaring": {"crop_cn": "棉花", "stage_cn": "蕾期", "crop_en": "cotton", "stage_en": "squaring"},
    "cotton_flowering": {"crop_cn": "棉花", "stage_cn": "开花期", "crop_en": "cotton", "stage_en": "flowering"},
    "cotton_boll_setting": {"crop_cn": "棉花", "stage_cn": "结铃期", "crop_en": "cotton", "stage_en": "boll_setting"},
    "cotton_boll_opening": {"crop_cn": "棉花", "stage_cn": "吐絮期", "crop_en": "cotton", "stage_en": "boll_opening"},
}

# 作物类型选项
CROP_TYPES = [
    "玉米 (corn)",
    "小麦 (wheat)",
    "棉花 (cotton)"
]

# 生长阶段选项（按作物类型）
GROWTH_STAGES = {
    "玉米 (corn)": ["出苗期 (seedling)", "拔节期 (jointing)", "抽穗期 (tasseling)", "灌浆期 (filling)", "成熟期 (maturity)"],
    "小麦 (wheat)": ["出苗期 (seedling)", "分蘖期 (tillering)", "拔节期 (jointing)", "抽穗期 (heading)", "成熟期 (maturity)"],
    "棉花 (cotton)": ["苗期 (seedling)", "蕾期 (squaring)", "开花期 (flowering)", "结铃期 (boll_setting)", "吐絮期 (boll_opening)"]
}


def save_uploaded_image(image, crop_type, growth_stage, user_note=""):
    """保存用户上传的图片到训练集"""
    if image is None:
        return "❌ 请先上传图片"

    try:
        # 解析作物类型和生长阶段
        crop_en = crop_type.split("(")[1].rstrip(")")
        stage_en = growth_stage.split("(")[1].rstrip(")")

        # 生成保存路径
        class_name = f"{crop_en}_{stage_en}"
        save_dir = os.path.join("dataset", "user_feedback", class_name)
        os.makedirs(save_dir, exist_ok=True)

        # 生成唯一文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"user_{timestamp}.jpg"
        save_path = os.path.join(save_dir, filename)

        # 保存图片
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(save_path, "JPEG", quality=95)

        # 记录元数据
        metadata = {
            "filename": filename,
            "crop_type": crop_type,
            "growth_stage": growth_stage,
            "class_name": class_name,
            "user_note": user_note,
            "timestamp": timestamp,
            "source": "web_upload"
        }

        # 保存元数据到 JSON 文件
        metadata_path = os.path.join(save_dir, f"{filename.rsplit('.', 1)[0]}.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return f"✅ 图片已保存!\n📁 路径: {save_path}\n🏷️ 类别: {class_name}\n📝 备注: {user_note if user_note else '无'}"

    except Exception as e:
        return f"❌ 保存失败: {str(e)}"


def quick_save_image(image, recommended_crop, recommended_stage, user_note=""):
    """一键保存图片（使用识别结果）"""
    if image is None:
        return "❌ 请先上传图片"

    if not recommended_crop or not recommended_stage:
        return "❌ 请先进行识别，或手动选择作物类型和生长阶段"

    # 直接使用推荐的作物类型和生长阶段
    return save_uploaded_image(image, recommended_crop, recommended_stage, user_note)


def count_feedback_images():
    """统计用户反馈图片数量"""
    feedback_dir = Path("dataset/user_feedback")
    if not feedback_dir.exists():
        return 0, {}
    total = 0
    by_class = {}
    for cls_dir in feedback_dir.iterdir():
        if cls_dir.is_dir():
            count = 0
            for img_path in cls_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    if not img_path.name.endswith('.json'):
                        count += 1
            if count > 0:
                by_class[cls_dir.name] = count
                total += count
    return total, by_class


def export_feedback_images():
    """打包导出用户反馈图片，返回 zip 文件路径"""
    feedback_dir = Path("dataset/user_feedback")
    total, by_class = count_feedback_images()

    if total == 0:
        return None, "❌ 没有可导出的图片"

    # 创建 zip 文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = f"dataset/feedback_export_{timestamp}.zip"

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for cls_dir in feedback_dir.iterdir():
            if cls_dir.is_dir():
                for img_path in cls_dir.glob("*"):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        if not img_path.name.endswith('.json'):
                            # 保留目录结构: class_name/filename.jpg
                            arcname = f"{cls_dir.name}/{img_path.name}"
                            zf.write(img_path, arcname)
                # 也打包元数据 json
                for json_path in cls_dir.glob("*.json"):
                    arcname = f"{cls_dir.name}/{json_path.name}"
                    zf.write(json_path, arcname)

    class_summary = "、".join([f"{k}({v}张)" for k, v in by_class.items()])
    return zip_path, f"✅ 导出完成！共 {total} 张图片：{class_summary}"


def hot_reload_model(model_file):
    """热加载上传的新模型文件"""
    global classifier, current_model_key

    if model_file is None:
        return "❌ 请先上传模型文件（.pth）"

    try:
        model_path = model_file.name if hasattr(model_file, 'name') else str(model_file)

        # 验证是否是有效的模型文件
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        if "model_state_dict" not in checkpoint:
            return "❌ 无效的模型文件，缺少 model_state_dict"

        class_names = checkpoint.get("class_names")
        if not class_names:
            return "❌ 模型文件缺少 class_names，请使用本系统训练生成的模型"

        # 复制到标准路径
        target_dir = Path("saved_models/clip/hot_reload")
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / "best.pth"

        import shutil
        shutil.copy2(model_path, target_path)

        # 保存配置
        config = {
            "model": checkpoint.get("model_name", "openai/clip-vit-large-patch14-336"),
            "lora_rank": 8,
            "num_classes": len(class_names),
            "class_names": class_names,
            "method": checkpoint.get("method", "lora"),
            "hot_reload_time": datetime.now().isoformat(),
        }
        with open(target_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        # 清理旧模型，加载新模型
        if classifier is not None:
            del classifier
            classifier = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        current_model_key = None  # 重置，强制重新加载

        return f"✅ 模型已更新！类别数: {len(class_names)}，下次识别将自动加载新模型"

    except Exception as e:
        return f"❌ 模型加载失败: {str(e)}"

# 模型显示名称（只保留最佳的零样本模型）
MODEL_LABELS = {
    "siglip2-so400m": "SigLIP2-so400m (零样本-最强)",
    "clip-large-336": "CLIP ViT-L/14@336 (零样本)",
}
LABEL_TO_KEY = {v: k for k, v in MODEL_LABELS.items()}

# 微调模型选项（推荐使用）
FINETUNED_CHOICES = [
    "创新模型-置信度路由 (最佳 93.5%)",
    "CLIP微调模型 (推荐)",
    "SigLIP2-so400m (零样本-最强)",
    "CLIP ViT-L/14@336 (零样本)",
]


def load_model(model_key):
    """加载或切换零样本模型"""
    global classifier, current_model_key

    if current_model_key == model_key and classifier is not None:
        return f"模型已加载: {model_key}"

    # 清理旧模型释放显存
    if classifier is not None:
        del classifier
        classifier = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    from models.clip_classifier import CLIPCropClassifier
    classifier = CLIPCropClassifier(model_name=model_key)
    current_model_key = model_key
    return f"模型加载成功: {model_key} (设备: {classifier.device})"


def load_finetuned(model_path):
    """加载微调模型（EfficientNet）"""
    global classifier, current_model_key

    from models.classifier import build_model
    from models.growth_stages import NUM_CLASSES, get_class_names

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    class_names = checkpoint.get("class_names", get_class_names())

    model = build_model(num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    classifier = {"model": model, "class_names": class_names, "device": device, "type": "efficientnet"}
    current_model_key = "finetuned"
    return f"微调模型加载成功 (设备: {device})"


def load_finetuned_clip(model_path):
    """加载微调后的CLIP模型"""
    global classifier, current_model_key

    from scripts.train_clip import CLIPWithClassifier, apply_lora
    from transformers import CLIPModel, AutoModel, CLIPProcessor, AutoProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    method = checkpoint.get("method", "lora")
    model_name = checkpoint.get("model_name", "openai/clip-vit-base-patch32")
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)
    state_dict = checkpoint["model_state_dict"]

    # 从state_dict推断LoRA rank
    lora_rank = 8  # 默认值
    for key, val in state_dict.items():
        if "lora_A" in key:
            lora_rank = val.shape[0]
            break

    # 加载CLIP模型
    if "siglip" in model_name.lower():
        clip_model = AutoModel.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)
    else:
        clip_model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)

    # 如果是LoRA，先应用LoRA（使用正确的rank）
    if method == "lora":
        clip_model = apply_lora(clip_model, rank=lora_rank)

    # 确定图像大小
    img_size = 336 if "336" in model_name else 224

    # 构建完整模型 - 使用正确的分类器结构
    import torch.nn as nn

    # 从state_dict推断分类器结构
    # checkpoint中的结构: Linear(768,1024) -> BN(1024) -> ReLU -> Dropout -> Linear(1024,512) -> BN(512) -> ReLU -> Dropout -> Linear(512,5)
    classifier = nn.Sequential(
        nn.Linear(768, 1024),
        nn.BatchNorm1d(1024), 
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, num_classes)
    )

    # 使用自定义分类器构建模型
    model = CLIPWithClassifier(clip_model, num_classes, img_size=img_size)
    model.classifier = classifier

    # 加载权重
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    classifier = {
        "model": model,
        "processor": processor,
        "class_names": class_names,
        "device": device,
        "type": "clip_finetuned",
        "model_name": model_name
    }
    current_model_key = "clip_finetuned"
    return f"CLIP微调模型加载成功 (方法: {method}, LoRA rank: {lora_rank}, 设备: {device}, 图像大小: {img_size})"


def load_innovation_model(model_path=None):
    """加载创新模型（置信度路由 + 生育期图建模 + Adaptive LoRA）"""
    global classifier, current_model_key

    if model_path is None:
        model_path = "saved_models/innovations/all_innovations/best.pth"

    if not os.path.isfile(model_path):
        return f"❌ 创新模型不存在: {model_path}"

    from transformers import CLIPModel
    from models.confidence_router import ConfidenceRouterClassifier
    from models.adaptive_lora import apply_adaptive_lora

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # 加载 CLIP 模型
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
    clip_model = clip_model.to(device)

    # 应用 Adaptive LoRA
    crop_ranks = {0: 4, 1: 8, 2: 16}
    clip_model, _ = apply_adaptive_lora(
        clip_model, crop_ranks=crop_ranks,
        target_modules=["q_proj"], dropout=0.0
    )
    clip_model = clip_model.to(device)

    # 加载 LoRA 权重
    if 'lora_state' in checkpoint:
        clip_model.load_state_dict(checkpoint['lora_state'], strict=False)

    clip_model.eval()

    # 创建置信度路由分类器
    feat_dim = clip_model.config.projection_dim  # 768
    router = ConfidenceRouterClassifier(
        feat_dim=feat_dim, num_classes=15, hidden_dim=256
    ).to(device)

    # 加载分类器权重
    if 'classifier_state' in checkpoint and 'confidence_router' in checkpoint['classifier_state']:
        router.load_state_dict(checkpoint['classifier_state']['confidence_router'])

    router.eval()

    classifier = {
        "type": "innovation",
        "clip_model": clip_model,
        "router": router,
        "device": device,
    }
    current_model_key = "innovation"

    val_acc = checkpoint.get('val_acc', 'N/A')
    return f"✅ 创新模型加载成功 (Val Acc: {val_acc}%, 设备: {device})"


def get_agronomic_advice(crop_en, stage_en):
    """根据作物和生育期生成农艺养护与预警建议"""
    advice_db = {
        ("corn", "seedling"): {
            "water": "控水促根，幼苗期忌大水漫灌以防蹲苗。",
            "fertilizer": "轻施提苗肥，以稀薄速效氮肥为主。",
            "pest": "重点防范地老虎、蝼蛄及苗期蚜虫。"
        },
        ("corn", "jointing"): {
            "water": "水分临界期前夕，保持土壤湿度 60-70%。",
            "fertilizer": "重施拔节壮秆肥，追施尿素 15-20kg/亩。",
            "pest": "重点防治玉米螟、粘虫及大斑病。"
        },
        ("corn", "tasseling"): {
            "water": "水分极敏感期！确保花粉活力，切忌干旱。",
            "fertilizer": "补施攻粒肥，防早衰与脱肥。",
            "pest": "人工辅助授粉或防控穗腐病。"
        },
        ("corn", "filling"): {
            "water": "保持干干湿湿，利于养分向籽粒转运。",
            "fertilizer": "叶面喷施 0.2% 磷酸二氢钾。",
            "pest": "注意防范玉米锈病、粉芽害。"
        },
        ("corn", "maturity"): {
            "water": "断水落干，便于机械化收割。",
            "fertilizer": "停止施肥，促进成熟。",
            "pest": "及时抢晴收割，防范霉变。"
        },
        ("wheat", "seedling"): {
            "water": "浇好越冬水，确保分蘖幼苗安全过冬。",
            "fertilizer": "基肥充足时少施氮肥，防旺长。",
            "pest": "防治种蝇及根腐病。"
        },
        ("wheat", "tillering"): {
            "water": "适度蹲苗，促进有效分蘖成穗。",
            "fertilizer": "追施分蘖肥，壮苗增蘖。",
            "pest": "防控麦黄吸浆虫、红蜘蛛。"
        },
        ("wheat", "jointing"): {
            "water": "水肥齐攻，促进大穗多粒。",
            "fertilizer": "重施拔节孕穗肥（氮钾结合）。",
            "pest": "防治纹枯病、白粉病。"
        },
        ("wheat", "heading"): {
            "water": "开花期忌大水灌溉，防倒伏与病害。",
            "fertilizer": "喷施微量元素与磷酸二氢钾。",
            "pest": "重点防控赤霉病（见花打药）。"
        },
        ("wheat", "maturity"): {
            "water": "收获前 7-10 天停水。",
            "fertilizer": "停止施肥。",
            "pest": "防范干热风与雨后烂麦穗。"
        },
        ("cotton", "seedling"): {
            "water": "中耕保墒，提高地温促早发。",
            "fertilizer": "苗期少量齐苗肥。",
            "pest": "防治立枯病、棉蚜。"
        },
        ("cotton", "squaring"): {
            "water": "稳水控氮，防止盲目旺长。",
            "fertilizer": "适量施配方肥，控氮增磷钾。",
            "pest": "防治棉铃虫、盲蝽蟌。"
        },
        ("cotton", "flowering"): {
            "water": "盛花期水肥关键期，不能缺水。",
            "fertilizer": "重施花铃肥，防止落花落铃。",
            "pest": "防治三代棉铃虫及红蜘蛛。"
        },
        ("cotton", "boll_setting"): {
            "water": "见干见湿，防早衰防脱落。",
            "fertilizer": "补施盖顶肥或叶面肥。",
            "pest": "防烂铃与红叶茎枯病。"
        },
        ("cotton", "boll_opening"): {
            "water": "推株并行，通风透光促吐絮。",
            "fertilizer": "停止施肥。",
            "pest": "分批采收，防泥污絮。"
        }
    }
    return advice_db.get((crop_en, stage_en), {
        "water": "保持适宜土壤湿度，因地制宜灌溉。",
        "fertilizer": "根据作物长势合理补充养分。",
        "pest": "定期巡田，早发现早防控。"
    })


def recognize_crop(image, model_label):
    if image is None:
        return "请上传一张农作物图片", "", ""

    # 根据模型类型调用不同的预测方法
    if isinstance(classifier, dict) and classifier.get("type") == "innovation":
        # 创新模型（置信度路由）
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize(336),
            transforms.CenterCrop(336),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])
        ])

        image_tensor = transform(image).unsqueeze(0).to(classifier["device"])

        with torch.no_grad():
            # CLIP 视觉编码 → 投影到 768 维
            vision_outputs = classifier["clip_model"].vision_model(pixel_values=image_tensor)
            pooled = vision_outputs.pooler_output
            features = classifier["clip_model"].visual_projection(pooled)

            # 置信度路由预测
            stage_logits, crop_logits, routing_info = classifier["router"](
                features, threshold=0.7, return_routing_info=True
            )
            probs = torch.softmax(stage_logits, dim=1)

        # 类别名列表（按 index 排序）
        class_names = [k for k, v in sorted(CLASS_MAP.items(), key=lambda x: x[1]["index"])]
        top_probs, top_indices = probs[0].topk(min(3, len(class_names)))

        results = []
        for prob, idx in zip(top_probs, top_indices):
            class_name = class_names[idx.item()]
            info = None
            for k, v in CLASS_MAP.items():
                if k == class_name:
                    info = {
                        "crop_name": v["crop_cn"],
                        "stage_name": v["stage_cn"],
                        "stage_days": v["days"],
                        "total_days": v["total_days"],
                        "description": v["description"],
                        "crop_en": v["crop_en"],
                        "stage_en": v["stage_en"],
                    }
                    break
            results.append({
                "class_name": class_name,
                "confidence": prob.item(),
                "info": info
            })

    elif isinstance(classifier, dict) and classifier.get("type") == "clip_finetuned":
        # CLIP微调模型 - 根据模型名称确定图像大小
        from torchvision import transforms
        model_name = classifier.get("model_name", "")
        img_size = 336 if "336" in model_name else 224

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        image_tensor = transform(image).unsqueeze(0).to(classifier["device"])

        with torch.no_grad():
            outputs = classifier["model"](image_tensor)
            probs = torch.softmax(outputs, dim=1)

        class_names = classifier["class_names"]
        top_probs, top_indices = probs[0].topk(min(3, len(class_names)))

        results = []
        for prob, idx in zip(top_probs, top_indices):
            class_name = class_names[idx.item()]
            # 查找类别信息
            info = None
            for k, v in CLASS_MAP.items():
                if k == class_name:
                    info = {
                        "crop_name": v["crop_cn"],
                        "stage_name": v["stage_cn"],
                        "stage_days": v["days"],
                        "total_days": v["total_days"],
                        "description": v["description"],
                        "crop_en": v["crop_en"],
                        "stage_en": v["stage_en"],
                    }
                    break
            results.append({
                "class_name": class_name,
                "confidence": prob.item(),
                "info": info
            })
    elif isinstance(classifier, dict) and classifier.get("type") == "efficientnet":
        # EfficientNet微调模型
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0).to(classifier["device"])
        with torch.no_grad():
            outputs = classifier["model"](image_tensor)
            probs = torch.softmax(outputs, dim=1)
        class_names = classifier["class_names"]
        top_probs, top_indices = probs[0].topk(min(3, len(class_names)))
        results = []
        for prob, idx in zip(top_probs, top_indices):
            class_name = class_names[idx.item()]
            info = None
            for k, v in CLASS_MAP.items():
                if k == class_name:
                    info = {
                        "crop_name": v["crop_cn"],
                        "stage_name": v["stage_cn"],
                        "stage_days": v["days"],
                        "total_days": v["total_days"],
                        "description": v["description"],
                        "crop_en": v["crop_en"],
                        "stage_en": v["stage_en"],
                    }
                    break
            results.append({
                "class_name": class_name,
                "confidence": prob.item(),
                "info": info
            })
    else:
        results = classifier.predict(image, top_k=3)

    # Top 1 结果
    top = results[0]
    info = top["info"]
    conf = top["confidence"]

    # 置信度颜色、发光特效与指示标签
    if conf >= 0.7:
        conf_color = "#059669"
        conf_bg = "rgba(16, 185, 129, 0.08)"
        conf_border = "rgba(16, 185, 129, 0.25)"
        conf_shadow = "rgba(16, 185, 129, 0.2)"
        conf_label = "高置信度"
    elif conf >= 0.4:
        conf_color = "#d97706"
        conf_bg = "rgba(245, 158, 11, 0.08)"
        conf_border = "rgba(245, 158, 11, 0.25)"
        conf_shadow = "rgba(245, 158, 11, 0.2)"
        conf_label = "中置信度"
    else:
        conf_color = "#dc2626"
        conf_bg = "rgba(220, 38, 38, 0.08)"
        conf_border = "rgba(220, 38, 38, 0.25)"
        conf_shadow = "rgba(220, 38, 38, 0.2)"
        conf_label = "较低置信"

    if info:
        stage_en = info.get("stage_en", "")
        crop_en = info.get("crop_en", "")
        advice = get_agronomic_advice(crop_en, stage_en)
        
        result_text = f"""
<div class="result-scorecard fade-in" style="background: rgba(255, 255, 255, 0.96); border-radius: 20px; padding: 24px; border: 1px solid rgba(16, 185, 129, 0.2); box-shadow: 0 12px 32px -8px rgba(16, 185, 129, 0.12); backdrop-filter: blur(12px);">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 20px; flex-wrap: wrap; gap: 12px;">
        <div style="display: flex; align-items: center; gap: 16px;">
            <div style="width: 56px; height: 56px; border-radius: 16px; background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); display: flex; align-items: center; justify-content: center; font-size: 28px; box-shadow: inset 0 2px 4px rgba(255,255,255,0.8); border: 1px solid rgba(16, 185, 129, 0.2);">
                🌾
            </div>
            <div>
                <h2 style="margin: 0; color: #064e3b; font-size: 1.65em; font-weight: 700; letter-spacing: -0.02em;">{info['crop_name']}</h2>
                <div style="display: flex; align-items: center; gap: 8px; margin-top: 4px;">
                    <span style="color: #059669; font-weight: 600; font-size: 1.1em;">{info['stage_name']}</span>
                    {f'<span style="background:#ecfdf5; color:#047857; font-size:0.75em; padding:2px 8px; border-radius:12px; font-weight:500;">{stage_en}</span>' if stage_en else ''}
                </div>
            </div>
        </div>
        <div style="background: {conf_bg}; border: 1px solid {conf_border}; padding: 6px 14px; border-radius: 30px; display: flex; align-items: center; gap: 8px; box-shadow: 0 2px 8px {conf_shadow};">
            <span style="width: 10px; height: 10px; border-radius: 50%; background: {conf_color}; display: inline-block; box-shadow: 0 0 10px {conf_color};"></span>
            <span style="color: {conf_color}; font-weight: 600; font-size: 0.9em;">{conf_label} ({conf:.1%})</span>
        </div>
    </div>

    <!-- 核心指标网格 -->
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 20px;">
        <div style="background: #f8fafc; border: 1px solid #f1f5f9; padding: 14px 12px; border-radius: 14px; text-align: center;">
            <div style="color: #64748b; font-size: 0.8em; font-weight: 500; margin-bottom: 4px;">置信度</div>
            <div style="color: {conf_color}; font-size: 1.6em; font-weight: 700;">{conf:.1%}</div>
            <div style="background: #e2e8f0; height: 4px; border-radius: 4px; overflow: hidden; margin-top: 8px;">
                <div style="background: {conf_color}; width: {conf*100}%; height: 100%; border-radius: 4px;"></div>
            </div>
        </div>
        <div style="background: #f8fafc; border: 1px solid #f1f5f9; padding: 14px 12px; border-radius: 14px; text-align: center;">
            <div style="color: #64748b; font-size: 0.8em; font-weight: 500; margin-bottom: 4px;">阶段持续</div>
            <div style="color: #0f172a; font-size: 1.25em; font-weight: 700; margin-top: 4px;">{info['stage_days']}</div>
            <div style="color: #94a3b8; font-size: 0.75em; margin-top: 4px;">估算周期</div>
        </div>
        <div style="background: #f8fafc; border: 1px solid #f1f5f9; padding: 14px 12px; border-radius: 14px; text-align: center;">
            <div style="color: #64748b; font-size: 0.8em; font-weight: 500; margin-bottom: 4px;">全生育期</div>
            <div style="color: #0f172a; font-size: 1.25em; font-weight: 700; margin-top: 4px;">{info['total_days']}</div>
            <div style="color: #94a3b8; font-size: 0.75em; margin-top: 4px;">完整生命周期</div>
        </div>
    </div>

    <!-- 阶段特征描述 -->
    <div style="background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%); border: 1px solid rgba(16, 185, 129, 0.15); border-radius: 14px; padding: 16px; margin-bottom: 14px;">
        <div style="color: #047857; font-size: 0.82em; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;">
            <span>💡</span> 阶段农艺特征描述
        </div>
        <div style="color: #1e293b; line-height: 1.6; font-size: 0.93em;">{info['description']}</div>
    </div>
</div>
"""
    else:
        result_text = f"""
<div style="background: rgba(255, 255, 255, 0.95); border-radius: 20px; padding: 24px; text-align: center; border: 1px solid rgba(16, 185, 129, 0.2); box-shadow: 0 10px 30px rgba(0,0,0,0.05);">
    <h2 style="color: #064e3b; margin: 0; font-size: 1.6em;">{top['class_name']}</h2>
    <div style="color: {conf_color}; font-size: 2.2em; font-weight: 700; margin: 15px 0;">{conf:.1%}</div>
    <span style="background: {conf_bg}; color: {conf_color}; padding: 6px 16px; border-radius: 20px; font-weight: 600; border: 1px solid {conf_border};">
        置信度: {conf_label}
    </span>
</div>
"""

    # Top 3 详情
    details = """
<div style="background: rgba(255, 255, 255, 0.96); border-radius: 20px; padding: 20px; margin-top: 16px; border: 1px solid rgba(226, 232, 240, 0.8); box-shadow: 0 4px 20px rgba(0,0,0,0.03);">
    <h3 style="color: #064e3b; margin: 0 0 16px 0; font-size: 1.05em; font-weight: 700; display: flex; align-items: center; gap: 8px;">
        <span>📈</span> Top-3 候选匹配分析
    </h3>
"""
    for i, r in enumerate(results):
        ci_info = r.get("info")
        rank_badge = ["🥇", "🥈", "🥉"][i] if i < 3 else f"#{i+1}"

        if ci_info:
            bar_width = int(r['confidence'] * 100)
            bar_color = "#059669" if r['confidence'] >= 0.7 else "#d97706" if r['confidence'] >= 0.4 else "#dc2626"

            details += f"""
    <div style="background: #f8fafc; border: 1px solid #f1f5f9; padding: 12px 16px; border-radius: 12px; margin: 10px 0; display: flex; align-items: center; gap: 14px; transition: transform 0.2s ease;">
        <span style="font-size: 1.3em; width: 32px; height: 32px; border-radius: 50%; background: white; display: flex; align-items: center; justify-content: center; box-shadow: 0 2px 6px rgba(0,0,0,0.06); flex-shrink: 0;">{rank_badge}</span>
        <div style="flex: 1; min-width: 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-weight: 600; color: #1e293b; font-size: 0.95em;">{ci_info['crop_name']} · {ci_info['stage_name']}</span>
                <span style="color: {bar_color}; font-weight: 700; flex-shrink: 0; margin-left: 12px; font-size: 0.95em;">{r['confidence']:.1%}</span>
            </div>
            <div style="background: #e2e8f0; border-radius: 8px; height: 8px; overflow: hidden;">
                <div style="background: linear-gradient(90deg, {bar_color} 0%, {bar_color}dd 100%); width: {bar_width}%; height: 100%; border-radius: 8px; transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);"></div>
            </div>
        </div>
    </div>
"""
        else:
            details += f"""
    <div style="background: #f8fafc; border: 1px solid #f1f5f9; padding: 12px 16px; border-radius: 12px; margin: 10px 0; display: flex; align-items: center; gap: 14px;">
        <span style="font-size: 1.3em; width: 32px; height: 32px; border-radius: 50%; background: white; display: flex; align-items: center; justify-content: center; box-shadow: 0 2px 6px rgba(0,0,0,0.06); flex-shrink: 0;">{rank_badge}</span>
        <div style="flex: 1; min-width: 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-weight: 600; color: #1e293b; font-size: 0.95em;">{r['class_name']}</span>
                <span style="color: #64748b; font-weight: 700; flex-shrink: 0; margin-left: 12px; font-size: 0.95em;">{r['confidence']:.1%}</span>
            </div>
            <div style="background: #e2e8f0; border-radius: 8px; height: 8px; overflow: hidden;">
                <div style="background: #64748b; width: {int(r['confidence'] * 100)}%; height: 100%; border-radius: 8px;"></div>
            </div>
        </div>
    </div>
"""

    details += "</div>"

    # 作物生长周期时间轴（带当前阶段高亮）
    cycle_text = ""
    if info:
        crop_en = info.get("crop_en")
        if not crop_en:
            for k, v in CLASS_MAP.items():
                if v["crop_cn"] == info["crop_name"]:
                    crop_en = v["crop_en"]
                    break
        if crop_en and crop_en in CROP_INFO:
            crop_data = CROP_INFO[crop_en]
            cycle_text = f"""
<div style="background: rgba(255, 255, 255, 0.96); border-radius: 20px; padding: 22px; margin-top: 16px; border: 1px solid rgba(16, 185, 129, 0.18); box-shadow: 0 4px 20px rgba(0,0,0,0.03);">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 18px;">
        <h3 style="color: #064e3b; margin: 0; font-size: 1.05em; font-weight: 700; display: flex; align-items: center; gap: 8px;">
            <span>🌱</span> {info['crop_name']} 生育期里程碑时间轴
        </h3>
        <span style="background: #ecfdf5; border: 1px solid rgba(16, 185, 129, 0.2); color: #047857; padding: 4px 12px; border-radius: 20px; font-size: 0.82em; font-weight: 600;">
            全生育期 {crop_data['total_days']}
        </span>
    </div>
    <div style="position: relative; padding-left: 24px; border-left: 2px dashed #cbd5e1; margin-left: 10px;">
"""
            for stage_en, stage_data in crop_data["stages"].items():
                is_current = (stage_en == info.get("stage_en"))
                if is_current:
                    cycle_text += f"""
        <div style="position: relative; margin-bottom: 16px;">
            <!-- 时间轴当前高亮点 -->
            <div style="position: absolute; left: -31px; top: 12px; width: 14px; height: 14px; border-radius: 50%; background: #059669; border: 3px solid #ecfdf5; box-shadow: 0 0 12px #059669; animation: pulse-ring 2s infinite;"></div>
            
            <div style="background: linear-gradient(135deg, #059669 0%, #047857 100%); color: white; padding: 14px 18px; border-radius: 14px; box-shadow: 0 8px 24px -4px rgba(5, 150, 105, 0.35);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="flex: 1; min-width: 0;">
                        <div style="font-weight: 700; font-size: 1.05em; display: flex; align-items: center; gap: 8px;">
                            <span>{stage_data['name_cn']}</span>
                            <span style="background: rgba(255,255,255,0.25); font-size: 0.72em; padding: 2px 8px; border-radius: 10px; font-weight: 600;">📍 当前阶段</span>
                        </div>
                        <div style="opacity: 0.92; margin-top: 4px; font-size: 0.88em; line-height: 1.5;">{stage_data['description']}</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.2); padding: 5px 12px; border-radius: 20px; font-weight: 600; flex-shrink: 0; margin-left: 12px; font-size: 0.85em;">
                        {stage_data['days']}
                    </div>
                </div>
            </div>
        </div>
"""
                else:
                    cycle_text += f"""
        <div style="position: relative; margin-bottom: 12px;">
            <!-- 普通时间轴节点 -->
            <div style="position: absolute; left: -29px; top: 14px; width: 10px; height: 10px; border-radius: 50%; background: #94a3b8; border: 2px solid white;"></div>
            
            <div style="background: #f8fafc; border: 1px solid #f1f5f9; padding: 12px 16px; border-radius: 12px; transition: background 0.2s ease;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="flex: 1; min-width: 0;">
                        <div style="font-weight: 600; color: #334155; font-size: 0.95em;">{stage_data['name_cn']}</div>
                        <div style="color: #64748b; font-size: 0.85em; margin-top: 2px;">{stage_data['description']}</div>
                    </div>
                    <div style="color: #64748b; font-weight: 500; flex-shrink: 0; margin-left: 12px; font-size: 0.85em; background: #e2e8f0; padding: 3px 10px; border-radius: 12px;">{stage_data['days']}</div>
                </div>
            </div>
        </div>
"""

            cycle_text += """
    </div>
</div>
"""

    # 提取推荐的作物类型和生长阶段
    recommended_crop = ""
    recommended_stage = ""
    if info:
        crop_en = info.get("crop_en", "")
        stage_en = info.get("stage_en", "")
        if crop_en and stage_en:
            # 查找对应的中文名称
            for key, value in CROP_STAGE_MAP.items():
                if value["crop_en"] == crop_en and value["stage_en"] == stage_en:
                    recommended_crop = f"{value['crop_cn']} ({crop_en})"
                    recommended_stage = f"{value['stage_cn']} ({stage_en})"
                    break

    return result_text, details, cycle_text, recommended_crop, recommended_stage


def build_ui():
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* 核心动画定义 */
    @keyframes pulse-ring {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(5, 150, 105, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(5, 150, 105, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(5, 150, 105, 0); }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(18px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .fade-in {
        animation: fadeInUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }

    /* 全局背景与布局 */
    .gradio-container {
        font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        background: linear-gradient(145deg, #f0fdf4 0%, #ecfdf5 35%, #f4fbf7 70%, #e6f4ea 100%) !important;
        min-height: 100vh;
        padding: 30px 20px !important;
    }

    /* 主视窗容器 */
    .main-container {
        max-width: 1380px;
        margin: 0 auto;
        background: rgba(255, 255, 255, 0.88);
        border-radius: 28px;
        padding: 36px 40px;
        box-shadow: 0 24px 64px -12px rgba(5, 150, 105, 0.12), 0 4px 16px rgba(0, 0, 0, 0.02);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.8);
    }

    /* 顶部 Banner 区域 */
    .header-area {
        text-align: center;
        padding: 36px 20px;
        background: linear-gradient(135deg, rgba(236, 253, 245, 0.9) 0%, rgba(209, 250, 229, 0.5) 100%);
        border-radius: 24px;
        margin-bottom: 32px;
        border: 1px solid rgba(16, 185, 129, 0.2);
        box-shadow: inset 0 1px 2px rgba(255, 255, 255, 0.8), 0 10px 25px -5px rgba(5, 150, 105, 0.05);
        position: relative;
        overflow: hidden;
    }
    .header-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(5, 150, 105, 0.1);
        border: 1px solid rgba(5, 150, 105, 0.25);
        color: #047857;
        font-size: 0.85em;
        font-weight: 600;
        padding: 5px 16px;
        border-radius: 30px;
        margin-bottom: 14px;
        letter-spacing: 0.02em;
    }
    .main-title {
        font-size: 2.6em;
        font-weight: 800;
        background: linear-gradient(135deg, #064e3b 0%, #059669 60%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.03em;
        line-height: 1.2;
    }
    .subtitle {
        color: #475569;
        font-size: 1.1em;
        margin-top: 10px;
        font-weight: 500;
    }
    .feature-tag {
        display: inline-block;
        background: rgba(255, 255, 255, 0.9);
        color: #047857;
        border: 1px solid rgba(16, 185, 129, 0.25);
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 600;
        margin: 4px;
        box-shadow: 0 2px 6px rgba(5, 150, 105, 0.08);
        transition: all 0.25s ease;
    }
    .feature-tag:hover {
        transform: translateY(-2px);
        background: #059669;
        color: white;
        box-shadow: 0 6px 14px rgba(5, 150, 105, 0.25);
    }

    /* 模块卡片容器 */
    .card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03), 0 1px 3px rgba(0, 0, 0, 0.02);
        margin-bottom: 20px;
        border: 1px solid rgba(226, 232, 240, 0.9);
        transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.3s ease;
    }
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px -8px rgba(0, 0, 0, 0.08);
    }
    .card-header-title {
        color: #064e3b;
        margin-top: 0;
        font-size: 1.2em;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* 主识别按钮 */
    .recognize-btn {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        border: none !important;
        border-radius: 16px !important;
        color: white !important;
        font-size: 1.15em !important;
        font-weight: 700 !important;
        padding: 16px 0 !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 8px 24px -4px rgba(5, 150, 105, 0.35) !important;
        letter-spacing: 0.02em;
        margin-top: 16px !important;
    }
    .recognize-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 32px -4px rgba(5, 150, 105, 0.45) !important;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    }
    .recognize-btn:active {
        transform: translateY(0) !important;
    }

    /* 保存与导出按钮 */
    .quick-save-btn {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        border: none !important;
        border-radius: 14px !important;
        color: white !important;
        font-size: 1.05em !important;
        font-weight: 600 !important;
        padding: 14px 0 !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.3) !important;
        transition: all 0.25s ease !important;
    }
    .quick-save-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.4) !important;
    }
    .save-btn {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important;
        border: none !important;
        border-radius: 14px !important;
        color: white !important;
        font-size: 1.05em !important;
        font-weight: 600 !important;
        padding: 14px 0 !important;
        box-shadow: 0 6px 20px rgba(245, 158, 11, 0.3) !important;
        transition: all 0.25s ease !important;
    }
    .save-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 25px rgba(245, 158, 11, 0.4) !important;
    }

    /* 浮空透视数据 Pill */
    .stat-pill {
        background: rgba(255, 255, 255, 0.7);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 16px;
        padding: 12px 10px;
        text-align: center;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 12px rgba(5, 150, 105, 0.06);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .stat-pill:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(5, 150, 105, 0.15);
        background: rgba(255, 255, 255, 0.9);
    }

    /* 下拉选择框与文本框 */
    .gradio-dropdown {
        border-radius: 14px !important;
        border: 1px solid #cbd5e1 !important;
        transition: all 0.25s ease !important;
    }
    .gradio-dropdown:focus-within {
        border-color: #059669 !important;
        box-shadow: 0 0 0 4px rgba(5, 150, 105, 0.15) !important;
    }

    /* 上传图片控件高科技定位边框 */
    .gradio-image {
        border-radius: 20px !important;
        overflow: hidden;
        border: 2px dashed #a7f3d0 !important;
        background: rgba(240, 253, 244, 0.5) !important;
        transition: all 0.3s ease !important;
        position: relative;
    }
    .gradio-image:hover {
        border-color: #059669 !important;
        background: rgba(236, 253, 245, 0.8) !important;
        box-shadow: 0 0 20px rgba(5, 150, 105, 0.15) !important;
    }

    /* 底部 Accordion 手风琴 */
    .crop-accordion {
        margin-top: 28px;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        overflow: hidden;
        border: 1px solid rgba(226, 232, 240, 0.9);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.03);
    }

    /* 页脚 */
    .footer-info {
        text-align: center;
        padding: 24px 0 10px 0;
        color: #64748b;
        font-size: 0.9em;
        border-top: 1px solid rgba(226, 232, 240, 0.8);
        margin-top: 32px;
    }

    @media (max-width: 768px) {
        .main-container {
            padding: 20px 16px;
            border-radius: 20px;
        }
        .main-title {
            font-size: 2em;
        }
        .header-area {
            padding: 24px 16px;
        }
    }
    """

    with gr.Blocks(title="农作物识别系统", theme=gr.themes.Soft()) as demo:
        gr.HTML(f'''
        <style>{custom_css}</style>
        <div class="main-container fade-in">
            <!-- 顶部 Header 区域 -->
            <div class="header-area">
                <div class="header-badge">
                    <span>✨</span> 视觉-语言大模型驱动 · 智慧农业 AI 诊断平台
                </div>
                <h1 class="main-title">农作物生长阶段智能识别系统</h1>
                <p class="subtitle">基于高精度视觉模型精准诊断玉米、小麦、棉花全生命周期阶段</p>
                
                <div style="margin-top: 18px; display: flex; justify-content: center; gap: 8px; flex-wrap: wrap;">
                    <span class="feature-tag">🧠 视觉-语言大模型</span>
                    <span class="feature-tag">🎯 置信度自适应路由</span>
                    <span class="feature-tag">🌿 生育期图建模</span>
                    <span class="feature-tag">⚡ 93.5% 诊断准确率</span>
                </div>

                <!-- 浮空透视数据卡 -->
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 24px; max-width: 900px; margin-left: auto; margin-right: auto;">
                    <div class="stat-pill">
                        <div style="font-size: 1.4em; font-weight: 800; color: #059669;">93.5%</div>
                        <div style="font-size: 0.78em; color: #047857; font-weight: 600;">路由融合精度</div>
                    </div>
                    <div class="stat-pill">
                        <div style="font-size: 1.4em; font-weight: 800; color: #059669;">15 个</div>
                        <div style="font-size: 0.78em; color: #047857; font-weight: 600;">精准生育期</div>
                    </div>
                    <div class="stat-pill">
                        <div style="font-size: 1.4em; font-weight: 800; color: #059669;">3 大</div>
                        <div style="font-size: 0.78em; color: #047857; font-weight: 600;">涵盖主粮作物</div>
                    </div>
                    <div class="stat-pill">
                        <div style="font-size: 1.4em; font-weight: 800; color: #059669;">&lt; 0.5s</div>
                        <div style="font-size: 0.78em; color: #047857; font-weight: 600;">实时推理诊断</div>
                    </div>
                </div>
            </div>
        </div>
        ''')

        with gr.Row(equal_height=True):
            # 左侧：输入区
            with gr.Column(scale=1):
                gr.HTML('''
                <div class="card fade-in" style="border-top: 4px solid #059669;">
                    <div class="card-header-title">📷 上传农作物图片</div>
                    <p style="color: #64748b; margin: 6px 0 0 0; font-size: 0.92em;">
                        支持常见的 JPG、PNG 格式照片，建议提供清晰的植株或田间近照
                    </p>
                </div>
                ''')
                with gr.Group():
                    image_input = gr.Image(
                        type="pil",
                        label=None,
                        height=380,
                        show_label=False,
                        elem_classes=["gradio-image"]
                    )

                gr.HTML('''
                <div style="font-weight: 700; color: #064e3b; margin: 18px 0 8px 6px; font-size: 1em; display: flex; align-items: center; gap: 6px;">
                    <span>🤖</span> 识别模型选择
                </div>
                ''')
                model_dropdown = gr.Dropdown(
                    choices=FINETUNED_CHOICES,
                    value="创新模型-置信度路由 (最佳 93.5%)",
                    show_label=False,
                    info="推荐使用创新模型（精度最高 93.5%）",
                    allow_custom_value=True,
                    elem_classes=["gradio-dropdown"]
                )
                submit_btn = gr.Button(
                    "🔍 开始智能识别",
                    variant="primary",
                    size="lg",
                    elem_classes=["recognize-btn"]
                )

                # 快捷测试示例图片库
                gr.Examples(
                    examples=[
                        ["dataset/test/corn_jointing/corn_jointing_玉米拔节1（内蒙科右前旗）.JPG"],
                        ["dataset/test/wheat_heading/wheat_heading_IMG_5850.jpg"],
                        ["dataset/test/cotton_flowering/aug_10_2633.png"],
                    ],
                    inputs=[image_input],
                    label="⚡ 点击一键体验示例图片"
                )

                gr.HTML('''
                <div style="text-align: center; padding: 14px 0; color: #64748b; font-size: 0.88em;">
                    <div style="display: flex; justify-content: center; gap: 16px; flex-wrap: wrap;">
                        <span>🌽 玉米</span>
                        <span>🌾 小麦</span>
                        <span>🌱 棉花</span>
                        <span>📊 15个关键生育期</span>
                    </div>
                </div>
                ''')

            # 右侧：结果区
            with gr.Column(scale=1):
                gr.HTML('''
                <div class="card fade-in" style="border-top: 4px solid #10b981;">
                    <div class="card-header-title">📊 诊断识别结果</div>
                </div>
                ''')
                with gr.Group():
                    result_text = gr.HTML()
                    details_text = gr.HTML()
                    cycle_text = gr.HTML()

                    # 一键保存按钮（识别后直接保存）
                    gr.HTML('''
                    <div style="margin-top: 16px; padding: 16px; background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); border-radius: 14px; border: 1px solid rgba(16, 185, 129, 0.2);">
                        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                            <span style="font-size: 1.2em;">⚡</span>
                            <span style="font-weight: 700; color: #064e3b;">快速沉淀训练数据</span>
                        </div>
                        <p style="color: #475569; margin: 0; font-size: 0.88em;">
                            识别正确？点击按钮即可自动入库样本，助力模型持续微调进化。
                        </p>
                    </div>
                    ''')
                    quick_save_btn = gr.Button(
                        "⚡ 一键入库（使用当前识别标签）",
                        variant="primary",
                        size="lg",
                        elem_classes=["quick-save-btn"]
                    )
                    quick_save_status = gr.HTML("")

                # 保存图片区域（手动选择）
                gr.HTML('''
                <div class="card fade-in" style="margin-top: 20px; border-top: 4px solid #f59e0b;">
                    <div class="card-header-title" style="color: #b45309;">
                        💾 手动修正与样本入库
                    </div>
                    <p style="color: #64748b; margin: 6px 0 0 0; font-size: 0.92em;">
                        若识别有偏差，可手动修正作物与生育期标签后保存
                    </p>
                </div>
                ''')
                with gr.Group():
                    with gr.Row():
                        crop_type_dropdown = gr.Dropdown(
                            choices=CROP_TYPES,
                            label="作物类型",
                            info="确认作物种类",
                            allow_custom_value=True,
                            elem_classes=["gradio-dropdown"]
                        )
                        growth_stage_dropdown = gr.Dropdown(
                            choices=[],
                            label="生长阶段",
                            info="确认生长阶段",
                            allow_custom_value=True,
                            elem_classes=["gradio-dropdown"]
                        )
                    user_note_input = gr.Textbox(
                        label="备注信息（可选）",
                        placeholder="记录拍摄环境、品种、拍摄日期或地点...",
                        lines=2
                    )
                    save_btn = gr.Button(
                        "💾 保存数据样本",
                        variant="secondary",
                        size="lg",
                        elem_classes=["save-btn"]
                    )
                    save_status = gr.HTML("")

        # ==================== 数据导出 ====================
        gr.HTML('''
        <div class="card fade-in" style="border-top: 4px solid #3b82f6; margin-top: 16px;">
            <div class="card-header-title" style="color: #1d4ed8;">📦 样本数据集导出</div>
            <p style="color: #64748b; margin: 6px 0 0 0; font-size: 0.9em;">
                导出累积的反馈标注数据集，以便在本地复现训练或迭代微调模型
            </p>
        </div>
        ''')
        feedback_count_display = gr.HTML("")
        export_btn = gr.Button("📦 一键打包导出数据集", variant="secondary", elem_classes=["save-btn"])
        export_status = gr.HTML("")
        export_file = gr.File(visible=False, label="导出文件")

        # 保存后刷新统计
        def save_and_refresh(image, recommended_crop, recommended_stage):
            """保存图片并刷新统计"""
            result = quick_save_image(image, recommended_crop, recommended_stage, "一键保存（识别结果）")
            total, _ = count_feedback_images()
            if total > 0:
                count_html = f'<div style="padding:12px 16px; background:#ecfdf5; border-radius:12px; border:1px solid rgba(16, 185, 129, 0.25); color:#047857; font-size:0.92em; font-weight:600;">📦 已沉淀 <b>{total}</b> 张高质量真实反馈标注样本</div>'
            else:
                count_html = ""
            return result, count_html

        quick_save_btn.click(
            fn=save_and_refresh,
            inputs=[image_input, crop_type_dropdown, growth_stage_dropdown],
            outputs=[quick_save_status, feedback_count_display]
        )

        def do_export():
            """导出图片"""
            zip_path, msg = export_feedback_images()
            if zip_path:
                return msg, zip_path
            return msg, None

        export_btn.click(fn=do_export, outputs=[export_status, export_file])

        # 页面加载时显示反馈图片数量
        def refresh_feedback_count():
            total, _ = count_feedback_images()
            if total > 0:
                return f'<div style="padding:12px 16px; background:#ecfdf5; border-radius:12px; border:1px solid rgba(16, 185, 129, 0.25); color:#047857; font-size:0.92em; font-weight:600;">📦 已沉淀 <b>{total}</b> 张高质量真实反馈标注样本</div>'
            return ""

        demo.load(fn=refresh_feedback_count, outputs=[feedback_count_display])

        # 底部：支持的作物信息
        with gr.Accordion("🌾 系统支持识别的作物与生长阶段图谱", open=False, elem_classes=["crop-accordion"]):
            gr.HTML('<div style="padding: 12px 6px;">')
            for crop_en, crop_data in CROP_INFO.items():
                stages_list = []
                for s in crop_data["stages"].values():
                    stages_list.append(f"<code style='background:#ecfdf5; color:#047857; padding:2px 8px; border-radius:6px; font-weight:600;'>{s['name_cn']}</code>")
                stages_str = " ".join(stages_list)
                gr.Markdown(f"### 🌾 **{crop_data['name_cn']}** (全生育期 {crop_data['total_days']})\n阶段划分：{stages_str}")
            gr.HTML('</div>')

        # 底部信息
        gr.HTML('''
        <div class="footer-info">
            <p>基于 CLIP / SigLIP / 创新混合路由算法 · 智慧农业决策支持系统</p>
            <p style="margin-top: 6px; font-size: 0.85em; opacity: 0.8;">
                农作物生长阶段智能识别系统 © · 赋能数字农业与精准种植
            </p>
        </div>
        ''')

        def recognize_with_finetuned(image, model_label):
            global classifier, current_model_key

            if image is None:
                empty_dropdown = gr.Dropdown(choices=[], value=None)
                return '<div style="text-align:center; padding:40px; color:#78909c; font-size:1.1em;">📷 请先上传一张农作物图片</div>', "", "", "", empty_dropdown

            try:
                if model_label.startswith("创新模型"):
                    # 创新模型（置信度路由 + 生育期图建模 + Adaptive LoRA）
                    innovation_path = "saved_models/innovations/all_innovations/best.pth"
                    if not os.path.isfile(innovation_path):
                        empty_dropdown = gr.Dropdown(choices=[], value=None)
                        return '<div style="text-align:center; padding:30px; color:#e53935;">❌ 创新模型不存在，请先运行训练</div>', "", "", "", empty_dropdown
                    if current_model_key != "innovation":
                        load_innovation_model(innovation_path)
                elif model_label == "CLIP微调模型 (推荐)":
                    # 尝试多个可能的路径（热加载优先）
                    model_paths = [
                        "saved_models/clip/hot_reload/best.pth",
                        "saved_models/clip/clip-vit-large-patch14-336-v2/best.pth",
                        "saved_models/clip/clip-large-336/best.pth",
                        "saved_models/clip/clip-large/best.pth",
                        "saved_models/clip/best.pth",
                    ]
                    model_path = None
                    for p in model_paths:
                        if os.path.isfile(p):
                            model_path = p
                            break
                    if model_path is None:
                        empty_dropdown = gr.Dropdown(choices=[], value=None)
                        return '<div style="text-align:center; padding:30px; color:#e53935;">❌ CLIP微调模型不存在，请先运行训练</div>', "", "", "", empty_dropdown
                    if current_model_key != "clip_finetuned":
                        load_finetuned_clip(model_path)
                else:
                    model_key = LABEL_TO_KEY.get(model_label, "siglip2-so400m")
                    if current_model_key != model_key:
                        load_model(model_key)

                result_text, details_text, cycle_text, recommended_crop, recommended_stage = recognize_crop(image, model_label)

                # 根据推荐的作物类型，更新生长阶段的选项列表
                if recommended_crop and recommended_crop in GROWTH_STAGES:
                    stage_dropdown = gr.Dropdown(
                        choices=GROWTH_STAGES[recommended_crop],
                        value=recommended_stage if recommended_stage else None
                    )
                else:
                    stage_dropdown = gr.Dropdown(choices=[], value=None)

                return result_text, details_text, cycle_text, recommended_crop, stage_dropdown
            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                print(f"识别错误: {error_msg}")
                empty_dropdown = gr.Dropdown(choices=[], value=None)
                return f'<div style="text-align:center; padding:30px; color:#e53935;">❌ 识别出错: {str(e)}</div>', "", "", "", empty_dropdown

        def update_growth_stages(crop_type, current_stage=None):
            """根据作物类型更新生长阶段选项"""
            if crop_type and crop_type in GROWTH_STAGES:
                choices = GROWTH_STAGES[crop_type]
                # 如果当前值在新的选项中，保留它
                if current_stage and current_stage in choices:
                    return gr.Dropdown(choices=choices, value=current_stage)
                return gr.Dropdown(choices=choices, value=None)
            return gr.Dropdown(choices=[], value=None)

        def save_image_with_feedback(image, crop_type, growth_stage, user_note):
            """保存图片并返回状态"""
            if not crop_type or not growth_stage:
                return "❌ 请先选择作物类型和生长阶段"
            return save_uploaded_image(image, crop_type, growth_stage, user_note)

        submit_btn.click(
            fn=recognize_with_finetuned,
            inputs=[image_input, model_dropdown],
            outputs=[result_text, details_text, cycle_text, crop_type_dropdown, growth_stage_dropdown]
        )

        # 作物类型变化时更新生长阶段选项
        crop_type_dropdown.change(
            fn=update_growth_stages,
            inputs=[crop_type_dropdown, growth_stage_dropdown],
            outputs=[growth_stage_dropdown]
        )

        # 保存按钮事件（手动选择）
        save_btn.click(
            fn=save_image_with_feedback,
            inputs=[image_input, crop_type_dropdown, growth_stage_dropdown, user_note_input],
            outputs=[save_status]
        )

    return demo


def main():
    global mode, current_model_key

    parser = argparse.ArgumentParser(description="农作物识别 Web 界面")
    parser.add_argument("--mode", type=str, default="innovation",
                        choices=["innovation", "clip", "finetuned", "clip-finetuned"],
                        help="运行模式: innovation(创新模型), clip(零样本), finetuned(EfficientNet微调), clip-finetuned(CLIP微调)")
    parser.add_argument("--clip-model", type=str, default="siglip2-so400m",
                        help="默认零样本模型")
    parser.add_argument("--model-path", type=str, default="saved_models/best.pth",
                        help="微调模型路径")
    parser.add_argument("--clip-model-path", type=str, default="saved_models/clip/clip-vit-large-patch14-336-v2/best.pth",
                        help="CLIP微调模型路径")
    parser.add_argument("--innovation-model-path", type=str,
                        default="saved_models/innovations/all_innovations/best.pth",
                        help="创新模型路径")
    parser.add_argument("--port", type=int, default=7860, help="服务端口")
    parser.add_argument("--share", action="store_true", help="创建公网链接")
    args = parser.parse_args()

    mode = args.mode

    if mode == "innovation":
        if not os.path.isfile(args.innovation_model_path):
            print(f"警告: 创新模型不存在 - {args.innovation_model_path}")
            print("回退到CLIP微调模型...")
            mode = "clip-finetuned"
        else:
            print(f"加载创新模型: {args.innovation_model_path}")
            print(load_innovation_model(args.innovation_model_path))

    if mode == "clip-finetuned":
        if not os.path.isfile(args.clip_model_path):
            print(f"警告: CLIP微调模型不存在 - {args.clip_model_path}")
            print("回退到零样本模式...")
            mode = "clip"
        else:
            print(f"加载CLIP微调模型: {args.clip_model_path}")
            print(load_finetuned_clip(args.clip_model_path))

    # 如果微调模型加载失败，回退到零样本模式
    if mode == "clip":
        print(f"正在加载零样本模型: {args.clip_model}")
        print(load_model(args.clip_model))

    demo = build_ui()
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
