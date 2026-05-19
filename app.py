import os
import argparse

import torch
import gradio as gr
from PIL import Image

from models.growth_stages import CROP_INFO, CLASS_MAP


# 全局变量
classifier = None
mode = "clip"
current_model_key = None  # 当前已加载的模型key

# 模型显示名称
MODEL_LABELS = {
    "siglip2-so400m": "SigLIP2-so400m (零样本)",
    "siglip2-base": "SigLIP2-base (零样本)",
    "siglip-so400m": "SigLIP-so400m (零样本)",
    "siglip-large": "SigLIP-large (零样本)",
    "clip-large-336": "CLIP ViT-L/14@336 (零样本)",
    "clip-large": "CLIP ViT-L/14 (零样本)",
    "clip-base": "CLIP ViT-B/32 (零样本)",
}
LABEL_TO_KEY = {v: k for k, v in MODEL_LABELS.items()}

# 微调模型选项（不在MODEL_LABELS中，单独处理）
FINETUNED_CHOICES = [
    "CLIP微调模型",
    "EfficientNet微调模型",
    "SigLIP2-so400m (零样本)",
    "SigLIP2-base (零样本)",
    "SigLIP-so400m (零样本)",
    "SigLIP-large (零样本)",
    "CLIP ViT-L/14@336 (零样本)",
    "CLIP ViT-L/14 (零样本)",
    "CLIP ViT-B/32 (零样本)",
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


def recognize_crop(image, model_label):
    if image is None:
        return "请上传一张农作物图片", "", ""

    # 根据模型类型调用不同的预测方法
    if isinstance(classifier, dict) and classifier.get("type") == "clip_finetuned":
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
    else:
        results = classifier.predict(image, top_k=3)

    # Top 1 结果
    top = results[0]
    info = top["info"]
    conf = top["confidence"]

    # 置信度颜色和样式
    if conf >= 0.7:
        conf_color = "#28a745"
        conf_label = "高"
        conf_emoji = "🎯"
    elif conf >= 0.4:
        conf_color = "#ffc107"
        conf_label = "中"
        conf_emoji = "⚡"
    else:
        conf_color = "#dc3545"
        conf_label = "低"
        conf_emoji = "❓"

    if info:
        result_text = f"""
<div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 16px; padding: 24px; border-left: 5px solid {conf_color};">
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
        <span style="font-size: 2.5em;">🌾</span>
        <div>
            <h2 style="margin: 0; color: #343a40; font-size: 1.8em;">{info['crop_name']}</h2>
            <div style="color: {conf_color}; font-weight: 600; font-size: 1.2em; margin-top: 4px;">
                {info['stage_name']} {conf_emoji}
            </div>
        </div>
    </div>

    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 20px 0;">
        <div style="background: white; padding: 16px; border-radius: 12px; text-align: center;">
            <div style="color: #6c757d; font-size: 0.85em; margin-bottom: 8px;">置信度</div>
            <div style="color: {conf_color}; font-size: 1.8em; font-weight: 700;">{conf:.1%}</div>
            <div style="color: {conf_color}; font-weight: 500;">{conf_label}</div>
        </div>
        <div style="background: white; padding: 16px; border-radius: 12px; text-align: center;">
            <div style="color: #6c757d; font-size: 0.85em; margin-bottom: 8px;">阶段时间</div>
            <div style="color: #343a40; font-size: 1.2em; font-weight: 600;">{info['stage_days']}</div>
        </div>
        <div style="background: white; padding: 16px; border-radius: 12px; text-align: center;">
            <div style="color: #6c757d; font-size: 0.85em; margin-bottom: 8px;">全生育期</div>
            <div style="color: #343a40; font-size: 1.2em; font-weight: 600;">{info['total_days']}</div>
        </div>
    </div>

    <div style="background: white; padding: 16px; border-radius: 12px; margin-top: 16px;">
        <div style="color: #6c757d; font-size: 0.85em; margin-bottom: 8px;">阶段描述</div>
        <div style="color: #495057; line-height: 1.6;">{info['description']}</div>
    </div>
</div>
"""
    else:
        result_text = f"""
<div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 16px; padding: 24px; text-align: center;">
    <h2 style="color: #343a40; margin: 0;">{top['class_name']}</h2>
    <div style="color: {conf_color}; font-size: 2em; font-weight: 700; margin: 15px 0;">{conf:.1%}</div>
    <div style="color: {conf_color}; font-weight: 500;">置信度{conf_label}</div>
</div>
"""

    # Top 3 详情
    details = """
<div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 16px; padding: 20px; margin-top: 15px;">
    <h3 style="color: #495057; margin: 0 0 15px 0; font-size: 1.1em;">📈 候选结果分析</h3>
"""
    for i, r in enumerate(results):
        ci_info = r.get("info")
        rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else f"#{i+1}"

        if ci_info:
            bar_width = int(r['confidence'] * 100)
            bar_color = "#28a745" if r['confidence'] >= 0.7 else "#ffc107" if r['confidence'] >= 0.4 else "#dc3545"

            details += f"""
    <div style="background: white; padding: 14px; border-radius: 10px; margin: 8px 0; display: flex; align-items: center; gap: 15px;">
        <span style="font-size: 1.5em; width: 40px;">{rank_emoji}</span>
        <div style="flex: 1;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span style="font-weight: 600; color: #343a40;">{ci_info['crop_name']} · {ci_info['stage_name']}</span>
                <span style="color: {bar_color}; font-weight: 600;">{r['confidence']:.1%}</span>
            </div>
            <div style="background: #e9ecef; border-radius: 8px; height: 8px; overflow: hidden;">
                <div style="background: {bar_color}; width: {bar_width}%; height: 100%; border-radius: 8px; transition: width 0.5s ease;"></div>
            </div>
        </div>
    </div>
"""
        else:
            details += f"""
    <div style="background: white; padding: 14px; border-radius: 10px; margin: 8px 0; display: flex; align-items: center; gap: 15px;">
        <span style="font-size: 1.5em; width: 40px;">{rank_emoji}</span>
        <div style="flex: 1;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span style="font-weight: 600; color: #343a40;">{r['class_name']}</span>
                <span style="color: #6c757d; font-weight: 600;">{r['confidence']:.1%}</span>
            </div>
            <div style="background: #e9ecef; border-radius: 8px; height: 8px; overflow: hidden;">
                <div style="background: #6c757d; width: {int(r['confidence'] * 100)}%; height: 100%; border-radius: 8px;"></div>
            </div>
        </div>
    </div>
"""

    details += "</div>"

    # 作物生长周期总览（带当前阶段高亮）
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
<div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 16px; padding: 20px; margin-top: 15px;">
    <h3 style="color: #495057; margin: 0 0 15px 0; font-size: 1.1em;">🌱 {info['crop_name']}生长周期</h3>
    <div style="background: white; padding: 16px; border-radius: 12px; margin-bottom: 15px;">
        <div style="color: #6c757d; font-size: 0.85em; margin-bottom: 8px;">全生育期</div>
        <div style="color: #343a40; font-size: 1.3em; font-weight: 600;">{crop_data['total_days']}</div>
    </div>
"""
            for stage_en, stage_data in crop_data["stages"].items():
                is_current = (stage_en == info.get("stage_en"))
                if is_current:
                    cycle_text += f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 14px; border-radius: 10px; margin: 8px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-weight: 600; font-size: 1.1em;">{stage_data['name_cn']}</div>
                <div style="opacity: 0.9; margin-top: 4px;">{stage_data['description']}</div>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 6px 12px; border-radius: 20px; font-weight: 500;">
                {stage_data['days']}
            </div>
        </div>
        <div style="margin-top: 10px; font-weight: 500;">📍 当前阶段</div>
    </div>
"""
                else:
                    cycle_text += f"""
    <div style="background: white; padding: 14px; border-radius: 10px; margin: 8px 0; border-left: 4px solid #e9ecef;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-weight: 500; color: #495057;">{stage_data['name_cn']}</div>
                <div style="color: #6c757d; font-size: 0.9em; margin-top: 4px;">{stage_data['description']}</div>
            </div>
            <div style="color: #6c757d; font-weight: 500;">{stage_data['days']}</div>
        </div>
    </div>
"""

            cycle_text += "</div>"

    return result_text, details, cycle_text


def build_ui():
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* 全局样式 */
    .gradio-container {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        padding: 20px;
    }

    /* 主容器 */
    .main-container {
        max-width: 1400px;
        margin: 0 auto;
        background: rgba(255,255,255,0.95);
        border-radius: 24px;
        padding: 40px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.2);
        backdrop-filter: blur(10px);
    }

    /* 标题区域 */
    .header-area {
        text-align: center;
        padding: 30px 0 40px 0;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 20px;
        margin-bottom: 30px;
        border: 1px solid rgba(0,0,0,0.05);
    }
    .main-title {
        font-size: 2.8em;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        padding: 0;
        letter-spacing: -0.02em;
    }
    .subtitle {
        color: #6c757d;
        font-size: 1.15em;
        margin-top: 12px;
        font-weight: 400;
    }
    .header-icon {
        font-size: 3em;
        margin-bottom: 15px;
        display: block;
    }

    /* 卡片容器 */
    .card {
        background: white;
        border-radius: 20px;
        padding: 28px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.06);
        margin: 10px;
        border: 1px solid rgba(0,0,0,0.04);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 50px rgba(0,0,0,0.1);
    }

    /* 输入区 */
    .input-card {
        border-top: 5px solid #667eea;
    }
    .input-card h3 {
        color: #667eea;
        margin-top: 0;
        font-size: 1.3em;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* 结果区 */
    .result-card {
        border-top: 5px solid #764ba2;
    }
    .result-card h3 {
        color: #764ba2;
        margin-top: 0;
        font-size: 1.3em;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* 按钮 */
    .recognize-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 14px !important;
        font-size: 1.2em !important;
        font-weight: 600 !important;
        padding: 18px 0 !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .recognize-btn:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.5) !important;
    }
    .recognize-btn:active {
        transform: translateY(0) !important;
    }

    /* 模型选择标签 */
    .model-label {
        font-weight: 600;
        color: #667eea;
        margin-bottom: 12px;
        font-size: 1em;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* 下拉菜单样式 */
    .gradio-dropdown {
        border-radius: 12px !important;
        border: 2px solid #e9ecef !important;
        transition: all 0.3s ease !important;
    }
    .gradio-dropdown:focus-within {
        border-color: #667eea !important;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1) !important;
    }

    /* 图片上传区域 */
    .gradio-image {
        border-radius: 16px !important;
        overflow: hidden;
        border: 3px dashed #dee2e6 !important;
        transition: all 0.3s ease !important;
    }
    .gradio-image:hover {
        border-color: #667eea !important;
        background: rgba(102, 126, 234, 0.02) !important;
    }

    /* 结果文字样式 */
    .result-text {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #667eea;
    }

    /* 底部作物信息 */
    .crop-accordion {
        margin-top: 25px;
        background: white;
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 5px 20px rgba(0,0,0,0.05);
    }
    .crop-accordion .gradio-accordion {
        border-radius: 16px !important;
        overflow: hidden;
    }

    /* 置信度指示器 */
    .confidence-high {
        color: #28a745;
        font-weight: 600;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: 600;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: 600;
    }

    /* 动画效果 */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    .fade-in {
        animation: fadeInUp 0.6s ease-out;
    }

    /* 特性标签 */
    .feature-tag {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 500;
        margin: 4px;
        box-shadow: 0 3px 10px rgba(102, 126, 234, 0.3);
    }

    /* 底部信息 */
    .footer-info {
        text-align: center;
        padding: 25px 0;
        color: #6c757d;
        font-size: 0.95em;
        border-top: 1px solid #e9ecef;
        margin-top: 30px;
    }
    .footer-info a {
        color: #667eea;
        text-decoration: none;
        font-weight: 500;
    }
    .footer-info a:hover {
        text-decoration: underline;
    }

    /* 响应式设计 */
    @media (max-width: 768px) {
        .main-container {
            padding: 20px;
            margin: 10px;
        }
        .main-title {
            font-size: 2em;
        }
        .header-area {
            padding: 20px 0 25px 0;
        }
    }
    """

    with gr.Blocks(title="农作物识别系统", theme=gr.themes.Soft(), css=custom_css) as demo:
        gr.HTML('''
        <div class="main-container fade-in">
            <!-- 标题区域 -->
            <div class="header-area">
                <span class="header-icon">🌾</span>
                <h1 class="main-title">农作物生长阶段识别系统</h1>
                <p class="subtitle">基于视觉-语言模型的智能农作物识别 · 支持玉米 / 小麦 / 棉花</p>
                <div style="margin-top: 20px;">
                    <span class="feature-tag">深度学习</span>
                    <span class="feature-tag">CLIP模型</span>
                    <span class="feature-tag">LoRA微调</span>
                    <span class="feature-tag">零样本识别</span>
                </div>
            </div>
        </div>
        ''')

        with gr.Row(equal_height=True):
            # 左侧：输入区
            with gr.Column(scale=1):
                gr.HTML('''
                <div class="card input-card fade-in">
                    <h3>📷 上传农作物图片</h3>
                    <p style="color: #6c757d; margin: 0; font-size: 0.95em;">
                        支持 JPG、PNG 等常见图片格式，建议使用清晰的农作物照片
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
                <div class="model-label" style="padding-left:12px; margin-top:20px;">
                    🤖 选择识别模型
                </div>
                ''')
                model_dropdown = gr.Dropdown(
                    choices=FINETUNED_CHOICES,
                    value="CLIP微调模型",
                    show_label=False,
                    info="推荐使用 CLIP微调模型（精度最高）",
                    elem_classes=["gradio-dropdown"]
                )
                submit_btn = gr.Button(
                    "🔍 开始识别",
                    variant="primary",
                    size="lg",
                    elem_classes=["recognize-btn"]
                )

                gr.HTML('''
                <div style="text-align: center; padding: 15px 0; color: #6c757d; font-size: 0.9em;">
                    <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
                        <span>🎯 支持3种农作物</span>
                        <span>🌱 识别15个生长阶段</span>
                        <span>⚡ 毫秒级响应</span>
                    </div>
                </div>
                ''')

            # 右侧：结果区
            with gr.Column(scale=1):
                gr.HTML('''
                <div class="card result-card fade-in">
                    <h3>📊 识别结果</h3>
                </div>
                ''')
                with gr.Group():
                    result_text = gr.Markdown()
                    details_text = gr.Markdown()
                    cycle_text = gr.Markdown()

        # 底部：支持的作物信息
        with gr.Accordion("🌾 支持识别的作物与生长阶段", open=False, elem_classes=["crop-accordion"]):
            gr.HTML('<div style="padding: 10px 0;">')
            for crop_en, crop_data in CROP_INFO.items():
                stages_list = []
                for s in crop_data["stages"].values():
                    stages_list.append(f"<code>{s['name_cn']}</code>")
                stages_str = " ".join(stages_list)
                gr.Markdown(f"**{crop_data['name_cn']}** — 全生育期 {crop_data['total_days']} | 阶段：{stages_str}")
            gr.HTML('</div>')

        # 底部信息
        gr.HTML('''
        <div class="footer-info">
            <p>基于 CLIP / SigLIP 视觉-语言模型 · 支持零样本识别与 LoRA 微调</p>
            <p style="margin-top: 8px; font-size: 0.85em;">
                农作物生长阶段识别系统 © 2024 · 为精准农业提供智能支持
            </p>
        </div>
        ''')

        def recognize_with_finetuned(image, model_label):
            global classifier, current_model_key

            if image is None:
                return "> 请先上传一张农作物图片", "", ""

            try:
                if model_label == "EfficientNet微调模型":
                    model_path = "saved_models/best.pth"
                    if not os.path.isfile(model_path):
                        return "> 错误: EfficientNet微调模型不存在", "", ""
                    if current_model_key != "efficientnet_finetuned":
                        load_finetuned(model_path)
                elif model_label == "CLIP微调模型":
                    # 尝试多个可能的路径
                    model_paths = [
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
                        return "> 错误: CLIP微调模型不存在，请先运行训练", "", ""
                    if current_model_key != "clip_finetuned":
                        load_finetuned_clip(model_path)
                else:
                    model_key = LABEL_TO_KEY.get(model_label, "siglip2-so400m")
                    if current_model_key != model_key:
                        load_model(model_key)

                return recognize_crop(image, model_label)
            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                print(f"识别错误: {error_msg}")
                return f"> 识别过程出错: {str(e)}", "", ""

        submit_btn.click(
            fn=recognize_with_finetuned,
            inputs=[image_input, model_dropdown],
            outputs=[result_text, details_text, cycle_text]
        )

    return demo


def main():
    global mode, current_model_key

    parser = argparse.ArgumentParser(description="农作物识别 Web 界面")
    parser.add_argument("--mode", type=str, default="clip",
                        choices=["clip", "finetuned", "clip-finetuned"],
                        help="运行模式: clip(零样本), finetuned(EfficientNet微调), clip-finetuned(CLIP微调)")
    parser.add_argument("--clip-model", type=str, default="siglip2-so400m",
                        help="默认零样本模型")
    parser.add_argument("--model-path", type=str, default="saved_models/best.pth",
                        help="微调模型路径")
    parser.add_argument("--clip-model-path", type=str, default="saved_models/clip/clip-vit-large-patch14-336-v2/best.pth",
                        help="CLIP微调模型路径")
    parser.add_argument("--port", type=int, default=7860, help="服务端口")
    parser.add_argument("--share", action="store_true", help="创建公网链接")
    args = parser.parse_args()

    mode = args.mode

    if mode == "clip":
        print(f"正在加载默认模型: {args.clip_model}")
        print(load_model(args.clip_model))
    elif mode == "clip-finetuned":
        if not os.path.isfile(args.clip_model_path):
            print(f"错误: CLIP微调模型不存在 - {args.clip_model_path}")
            print("请先运行: python scripts/train_clip.py --method lora --data-dir dataset")
            return
        print(f"加载CLIP微调模型: {args.clip_model_path}")
        print(load_finetuned_clip(args.clip_model_path))
    else:
        if not os.path.isfile(args.model_path):
            print(f"错误: 模型文件不存在 - {args.model_path}")
            return
        print(f"加载微调模型: {args.model_path}")
        print(load_finetuned(args.model_path))

    demo = build_ui()
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
