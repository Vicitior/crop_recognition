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
    "siglip2-so400m": "SigLIP2-so400m (最强)",
    "siglip2-base": "SigLIP2-base (轻量)",
    "siglip-so400m": "SigLIP-so400m",
    "siglip-large": "SigLIP-large",
    "clip-large-336": "CLIP ViT-L/14@336",
    "clip-large": "CLIP ViT-L/14",
    "clip-base": "CLIP ViT-B/32 (最快)",
}
LABEL_TO_KEY = {v: k for k, v in MODEL_LABELS.items()}


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

    # 加载CLIP模型
    if "siglip" in model_name.lower():
        clip_model = AutoModel.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)
    else:
        clip_model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)

    # 如果是LoRA，先应用LoRA
    if method == "lora":
        clip_model = apply_lora(clip_model)

    # 构建完整模型
    model = CLIPWithClassifier(clip_model, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    classifier = {
        "model": model,
        "processor": processor,
        "class_names": class_names,
        "device": device,
        "type": "clip_finetuned"
    }
    current_model_key = "clip_finetuned"
    return f"CLIP微调模型加载成功 (方法: {method}, 设备: {device})"


def recognize_crop(image, model_label):
    if image is None:
        return "请上传一张农作物图片", "", ""

    # 将显示名转回key
    model_key = LABEL_TO_KEY.get(model_label, "siglip2-so400m")

    # 确保模型已加载（避免重复加载）
    if current_model_key != model_key or classifier is None:
        load_model(model_key)

    # 根据模型类型调用不同的预测方法
    if isinstance(classifier, dict) and classifier.get("type") == "clip_finetuned":
        # CLIP微调模型
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

    # 置信度颜色
    if conf >= 0.7:
        conf_color = "#4caf50"  # 绿色
        conf_label = "高"
    elif conf >= 0.4:
        conf_color = "#ff9800"  # 橙色
        conf_label = "中"
    else:
        conf_color = "#f44336"  # 红色
        conf_label = "低"

    if info:
        result_text = (
            f"### {info['crop_name']} — {info['stage_name']}\n\n"
            f"置信度：**{conf:.1%}**（{conf_label}）\n\n"
            f"阶段时间：{info['stage_days']}  |  全生育期：{info['total_days']}\n\n"
            f"> {info['description']}"
        )
    else:
        result_text = f"### {top['class_name']}\n\n置信度：{conf:.1%}"

    # Top 3 详情
    details = "### 候选结果\n\n| 排名 | 作物 | 生长阶段 | 置信度 |\n|:---:|:---:|:---:|:---:|\n"
    for i, r in enumerate(results):
        ci_info = r.get("info")
        if ci_info:
            details += f"| {i+1} | {ci_info['crop_name']} | {ci_info['stage_name']} | {r['confidence']:.1%} |\n"
        else:
            details += f"| {i+1} | {r['class_name']} | - | {r['confidence']:.1%} |\n"

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
            cycle_text = f"### {info['crop_name']}生长周期\n\n"
            cycle_text += f"全生育期：**{crop_data['total_days']}**\n\n"
            for stage_en, stage_data in crop_data["stages"].items():
                is_current = (stage_en == info.get("stage_en"))
                marker = " **<-- 当前**" if is_current else ""
                cycle_text += f"- {stage_data['name_cn']}（{stage_data['days']}）：{stage_data['description']}{marker}\n"

    return result_text, details, cycle_text


def build_ui():
    model_choices = list(MODEL_LABELS.values())
    model_choices.append("EfficientNet微调模型")
    model_choices.append("CLIP微调模型")
    default_label = MODEL_LABELS.get("siglip2-so400m", model_choices[0])

    custom_css = """
    .main-title { text-align: center; margin-bottom: 0.2em; }
    .subtitle { text-align: center; color: #666; margin-bottom: 1.5em; }
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 20px; border-radius: 12px;
    }
    .result-card h3 { color: white; margin-top: 0; }
    .confidence-bar {
        background: #e0e0e0; border-radius: 8px; height: 12px; overflow: hidden;
    }
    .confidence-fill {
        background: linear-gradient(90deg, #4caf50, #8bc34a);
        height: 100%; border-radius: 8px; transition: width 0.5s;
    }
    .stage-tag {
        display: inline-block; padding: 4px 12px; border-radius: 16px;
        font-size: 0.85em; margin: 2px;
    }
    .crop-info-box {
        background: #f8f9fa; border-left: 4px solid #4caf50;
        padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 8px 0;
    }
    """

    with gr.Blocks(title="农作物识别系统", theme=gr.themes.Soft(), css=custom_css) as demo:
        gr.HTML('<h1 class="main-title">农作物生长阶段识别系统</h1>')
        gr.HTML('<p class="subtitle">基于视觉-语言模型的零样本农作物识别 | 支持玉米、小麦、棉花</p>')

        with gr.Row(equal_height=True):
            # 左侧：输入区
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 上传图片")
                    image_input = gr.Image(type="pil", label=None, height=320)
                    model_dropdown = gr.Dropdown(
                        choices=model_choices,
                        value=default_label,
                        label="选择模型",
                        info="SigLIP2-so400m 精度最高，CLIP-B/32 速度最快"
                    )
                    submit_btn = gr.Button("开始识别", variant="primary", size="lg")

            # 右侧：结果区
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 识别结果")
                    result_text = gr.Markdown()
                    details_text = gr.Markdown()
                    cycle_text = gr.Markdown()

        # 底部：支持的作物信息
        with gr.Accordion("支持识别的作物与生长阶段", open=False):
            for crop_en, crop_data in CROP_INFO.items():
                stages_list = []
                for s in crop_data["stages"].values():
                    stages_list.append(f"`{s['name_cn']}`")
                stages_str = " ".join(stages_list)
                gr.Markdown(f"**{crop_data['name_cn']}** — 全生育期 {crop_data['total_days']} | 阶段：{stages_str}")

        def recognize_with_finetuned(image, model_label):
            global classifier, current_model_key

            if image is None:
                return "> 请先上传一张农作物图片", "", ""

            if model_label == "EfficientNet微调模型":
                model_path = "saved_models/best.pth"
                if not os.path.isfile(model_path):
                    return "> 错误: EfficientNet微调模型不存在", "", ""
                if current_model_key != "efficientnet_finetuned":
                    load_finetuned(model_path)
            elif model_label == "CLIP微调模型":
                model_path = "saved_models/clip/best.pth"
                if not os.path.isfile(model_path):
                    return "> 错误: CLIP微调模型不存在，请先运行训练", "", ""
                if current_model_key != "clip_finetuned":
                    load_finetuned_clip(model_path)
            else:
                model_key = LABEL_TO_KEY.get(model_label, "siglip2-so400m")
                if current_model_key != model_key:
                    load_model(model_key)

            return recognize_crop(image, model_label)

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
    parser.add_argument("--clip-model-path", type=str, default="saved_models/clip/best.pth",
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
