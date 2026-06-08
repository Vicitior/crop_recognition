"""
CLIP零样本农作物分类器
支持多种视觉-语言模型：OpenAI CLIP, Google SigLIP, Google SigLIP2
无需训练数据，通过文字描述直接识别农作物种类和生长阶段
"""
import torch
import numpy as np
from PIL import Image, ImageOps
from models.growth_stages import CROP_INFO, CLASS_MAP


# 可选模型列表（从大到小）
AVAILABLE_MODELS = {
    # SigLIP2 - 最新最强 (Google 2025)
    "siglip2-so400m": "google/siglip2-so400m-patch16-384",
    "siglip2-base": "google/siglip2-base-patch16-256",
    # SigLIP - Google 2024
    "siglip-so400m": "google/siglip-so400m-patch14-384",
    "siglip-large": "google/siglip-large-patch16-256",
    # CLIP - OpenAI (经典)
    "clip-large-336": "openai/clip-vit-large-patch14-336",
    "clip-large": "openai/clip-vit-large-patch14",
    "clip-base": "openai/clip-vit-base-patch32",
}


# 英文视觉描述 - 强调相邻阶段之间的关键视觉区分点
# 每个阶段描述聚焦于该阶段独有的、区别于其他阶段的视觉特征
VISUAL_DESCRIPTIONS = {
    "corn": {
        "seedling": [
            # 核心特征：极矮小，单叶，无茎节，土壤裸露
            "very short corn seedlings under 30cm, single thin upright leaves, no visible stem, bare soil between plants",
            "tiny corn sprouts with first leaves just unfurling, grass-like appearance, height less than knee level",
            "baby corn plants emerging in rows, narrow leaves, no stalk visible, very early growth stage",
            "corn field with seedlings only a few inches tall, thin green blades, wide spacing between young plants",
            "newly germinated corn, short and small, leaf tips pointing upward, no branching, no ears",
            "corn at earliest growth stage, plants shorter than 20cm, single leaves, green and tender",
        ],
        "jointing": [
            # 核心特征：粗壮绿茎，可见节间（关节），齐腰高，无穗无棒
            # 优化：增加精确特征、负样本描述、场景描述
            "corn plants at vegetative stage with thick green stems showing 3-5 visible swollen nodes, height 60-120cm, NO ear development, NO tassels",
            "jointing corn stalks with elongated internodes 15-20cm apart, leaves alternating at each node, stems 2-3cm diameter, purely vegetative growth",
            "corn field showing rapid vertical growth, plants reaching waist to chest height, thick green stalks with distinct joint segments, no reproductive structures",
            "close-up of corn stem node showing leaf emergence, swollen joint area, green healthy vegetative growth, NOT the short seedling stage",
            "vegetative corn with strong upright stems, visible knuckle-like nodes along the stalk, broad leaves spreading outward, NO ear shoots visible",
            "mid-growth corn at jointing phase, stems thickening with distinct joint segments, plant height reaching adult waist, NOT the tall flowering tasseling stage",
            # 负样本描述：强调不是什么
            "corn plants with thick green stems showing prominent joints, NOT the short stubby seedlings under 30cm, NOT the tall plants with tassels at the top",
            "vegetative corn stalks with swollen stem nodes, distinctly different from filling stage which shows visible ear cobs with developing kernels",
            # 场景描述
            "summer cornfield with plants at jointing stage, rapid vegetative growth, thick green stems with visible nodes, no signs of flowering or ear formation",
        ],
        "tasseling": [
            # 核心特征：顶部有穗状花序（tassel），穗部吐出丝状花丝，但籽粒尚未灌浆
            "corn plants with feathery tassels emerging from the top of each stalk, silk threads visible at developing ear shoots",
            "flowering corn field, brown or yellow tassels hanging from plant tops, green husks with silk protruding",
            "corn at reproductive stage, male flower tassels at the apex, female silk strands emerging from ear positions on the stalk",
            "corn plants showing pollen-shedding tassels at the top, thin silky threads at mid-stalk ear shoots",
            "pollinating corn, tassel branches spreading at the top of the plant, silk emerging from young green ear husks",
            "corn field with visible tassels on every plant, no filled ears yet, silk visible on developing ear shoots",
        ],
        "filling": [
            # 核心特征：棒子上有饱满黄色籽粒，苞叶包裹，茎秆仍绿
            # 优化：增加精确特征、负样本描述、场景描述
            "corn plants WITH visible ear cobs at middle of stalk, rows of plump yellow kernels visible through partially open husks, stalks still green",
            "maturing corn field showing ear development, full-sized ears bending stalks slightly, yellow kernels filling out, silk turning brown",
            "close-up of corn ear with yellow kernels in dough stage, husks starting to loosen, heavy ear bending stalk, NOT the vegetative jointing stage with no ears",
            "corn plants with bulging ear cobs containing developing grain, kernels transitioning from white to yellow, ear position at middle height of plant",
            "grain-filling corn showing visible kernel rows through green husks, ears at full size, stalks green and healthy, post-flowering reproductive stage",
            "corn field with developing ears on most plants, yellow kernels visible, silk browning, NOT the mature brown dried stage, NOT the vegetative stage with no ears",
            # 负样本描述：强调不是什么
            "corn plants WITH visible ear cobs showing developing kernels, unlike jointing stage which has NO ear development, unlike maturity stage which has brown dried plants",
            "corn with bulging husks containing grain, stalks still green but ears prominent, distinctly different from tasseling stage which only has silk without filled ears",
            # 场景描述
            "late summer cornfield with plants at grain-filling stage, visible ear development on most plants, yellow kernels forming, heavy ears bending stalks",
        ],
        "maturity": [
            # 核心特征：全株枯黄/褐色，苞叶张开干枯，籽粒坚硬，黑层形成
            "fully mature corn field, all plants brown and dried, husks wide open exposing hard dry yellow cobs",
            "harvest-ready corn, brown stalks with dried leaves curling, open husks revealing mature cobs hanging down",
            "senescent corn plants, completely yellow-brown foliage, dried husks peeling back, hard kernels visible",
            "corn at full maturity, dried brown plant material, cobs with hard glossy kernels, husks papery and open",
            "corn field ready for harvest, uniformly brown and dry, mature cobs with exposed kernels, no green remaining",
            "post-maturity corn, stalks leaning, dried brown husks, dark hard kernels, overall desiccated appearance",
        ],
    },
    "wheat": {
        "seedling": [
            # 核心特征：极矮小，单根细叶，草状外观，无分蘖
            "very short wheat seedlings, single thin grass-like leaves, height under 15cm, bare soil visible between rows",
            "tiny wheat shoots emerging from the ground, fine green blades, no tillers, grass-like appearance",
            "newly sprouted wheat field, small individual green plants with one or two narrow leaves each",
            "wheat at earliest stage, seedlings shorter than 10cm, each plant a single thin shoot, no branching",
            "young wheat emerging in rows, very short and sparse, fine leaves, cereal seedling appearance",
            "wheat seedling stage, plants barely above the soil, individual thin green blades, no bushy growth",
        ],
        "tillering": [
            # 核心特征：基部多蘖丛生，矮丛状，无拔高茎秆
            "wheat plants producing multiple side shoots from the base, forming bushy green clumps, no elongated stems yet",
            "tillering wheat field, dense green tufts, each plant a cluster of many leaves and shoots from the base",
            "wheat at tillering stage, plants spreading outward with multiple stems from ground level, low and bushy",
            "wheat field with thick vegetative growth, plants forming dense rosettes, no vertical stem elongation",
            "tillering wheat, multiple green shoots per plant base, overall low-growing carpet-like canopy",
            "wheat plants with many tillers emerging, forming clumps of leaves, height still under 30cm, no visible stems",
            # 对比描述：强调与出苗期和拔节期的区别
            "wheat plants with multiple bushy tillers from the base, distinctly different from seedling stage which has single thin shoots",
            "tillering wheat forming dense low clumps, unlike jointing stage which shows tall elongated stems with visible nodes",
        ],
        "jointing": [
            # 核心特征：茎秆拔高伸长，可见节间，无穗
            # 优化：增加精确特征、负样本描述、场景描述
            "wheat stems elongating and growing upright, 2-3 visible nodes on hollow round stems, height 30-60cm, NO grain heads emerging from flag leaf",
            "jointing wheat with tall green stalks showing swollen nodes every 10-15cm, stems thickening, pre-heading vegetative growth, leaves still upright",
            "wheat at stem elongation phase, plants noticeably taller than tillering stage, stems with visible joints, NO ear emergence from sheath",
            "close-up of wheat stem showing elongated internodes and node swelling, hollow round stalk, no grain heads visible at top",
            "rapidly growing wheat field with plants reaching knee to waist height, distinct nodes visible on stems, purely vegetative growth",
            "wheat at jointing stage with upright green stalks, leaf sheaths still wrapped around stems, NO visible grain spikes emerging",
            # 负样本描述：强调不是什么
            "wheat stems with visible elongated internodes and swollen nodes, NOT the low bushy tillering stage with multiple shoots from base, NOT the heading stage with visible grain heads",
            "jointing wheat with tall upright stems showing clear node swelling, distinctly different from seedling stage which is very short, different from heading stage which has grain heads at top",
            # 场景描述
            "spring wheat field with plants at jointing stage, rapid vertical growth, stems elongating with visible nodes, no signs of heading or flowering",
        ],
        "heading": [
            # 核心特征：穗从旗叶鞘中抽出，穗部可见，花药悬挂
            "wheat heads emerging from the flag leaf sheath, compact green grain spikes visible at the top of each stem",
            "heading wheat field, grain ears fully emerged, compact spikes with visible anthers dangling from florets",
            "wheat at heading stage, each stem topped with an erect grain head, green spikes with yellow anthers",
            "wheat spikes emerging and beginning to flower, compact ear heads at stem tops, anthers hanging",
            "heading wheat, grain heads fully visible above the flag leaf, green compact spikes before grain fill",
            "wheat field with visible ear heads on every stem, flowering stage with dangling yellow anther filaments",
        ],
        "maturity": [
            # 核心特征：全株金黄色，穗部下垂弯曲，茎秆干燥
            "golden amber wheat field, all plants uniformly yellow-gold, grain heads heavy and drooping downward",
            "mature wheat ready for harvest, dry golden stalks, grain heads bent down, no green color remaining",
            "ripe wheat field, amber colored from top to bottom, dried stalks with full grain heads hanging down",
            "harvest-ready wheat, completely golden brown, stems dry and stiff, grain heads heavy and curved downward",
            "wheat at full maturity, dry golden plants, compact grain heads hanging from curved stems, uniform golden field",
            "mature wheat field, dried yellow stalks, grain heads full and drooping, ready for combine harvest",
        ],
    },
    "cotton": {
        "seedling": [
            # 核心特征：矮小，圆形子叶，贴近地面，无分枝
            "tiny cotton seedlings with two round cotyledon leaves close to the ground, thin green stems, bare soil visible",
            "young cotton plants just emerged, round seed leaves, delicate stems under 10cm, no branching",
            "cotton at seedling stage, small dicotyledon plants with round leaves, very short, no flower buds",
            "cotton seedlings in rows, each plant with two round flat leaves, stems thin and green, height under 15cm",
            "newly sprouted cotton, round cotyledons still visible, no true leaves yet, close to soil surface",
            "cotton field at seedling stage, sparse small plants with round leaves, wide spacing, bare ground between plants",
        ],
        "squaring": [
            # 核心特征：分枝增多，出现小方形花蕾（三角形芽），但无开放花朵
            "cotton plants with small square-shaped green flower buds appearing at branch nodes, bushy branching growth",
            "cotton at squaring stage, triangular green buds visible among lush foliage, no open flowers yet",
            "pre-flowering cotton, plants with multiple branches bearing small square buds, deep green leaves",
            "cotton field with bushy plants showing first flower buds, small green square-shaped buds at nodes",
            "squaring cotton, branching stems with small tight green buds, buds are square or triangular in shape",
            "cotton plants developing flower buds, small square green buds visible, plant structure becoming bushy with branches",
        ],
        "flowering": [
            # 核心特征：花朵开放，白色/乳白/粉红色花瓣，黄色花蕊
            # 优化：增加精确特征、负样本描述、场景描述
            "cotton plants with OPEN flowers showing creamy white petals (5 petals) with dark red spots at base, prominent yellow staminal column in center",
            "flowering cotton field with large conspicuous blooms, petals white turning pink with age, yellow pollen-bearing stamens visible, NO green closed buds",
            "cotton at anthesis stage, distinctive open flowers among green foliage, petals 3-5cm wide, dark nectary spots at petal base, yellow stamens",
            "close-up of cotton flower fully open, white petals with characteristic dark markings at base, yellow stamen column, NOT the closed green buds of squaring stage",
            "blooming cotton with scattered open flowers on plants, petals white to pink, yellow stamens visible, NO green round bolls visible",
            "cotton field during flowering period, open blooms visible on multiple plants, creamy white petals aging to pink, reproductive stage",
            # 负样本描述：强调不是什么
            "cotton plants with OPEN flowers showing white or pink petals and yellow stamens, NOT the squaring stage with only closed green buds, NOT the boll-setting stage with green round capsules",
            "flowering cotton with large conspicuous blooms, distinctly different from boll-setting stage which has green round fruit capsules instead of open flowers",
            # 场景描述
            "summer cotton field with plants in full bloom, open white flowers visible among green foliage, distinctive cotton flowers with dark petal base markings",
        ],
        "boll_setting": [
            # 核心特征：绿色圆球形棉铃（boll），无开放花朵，叶片仍绿
            "cotton plants with dark green round bolls or capsules developing on branches, no open flowers visible",
            "cotton at boll-setting stage, firm green spherical fruit capsules hanging from branches, leaves still green",
            "cotton field with developing bolls, green rounded capsules of various sizes on the plants",
            "cotton plants bearing young green bolls, round hard capsules, some dried flower remnants at boll tips",
            "boll-setting cotton, green spherical bolls growing on branch nodes, plant canopy still green and healthy",
            "cotton with developing fruit, green round bolls visible among the foliage, no white cotton fiber exposed",
            # 对比描述：强调与开花期和吐絮期的区别
            "cotton plants with green round bolls but NO open flowers, unlike flowering stage which has white or pink petals",
            "boll-setting cotton with firm green spherical capsules, distinctly different from boll-opening stage where bolls split open revealing white fiber",
        ],
        "boll_opening": [
            # 核心特征：棉铃开裂，白色棉絮外露，部分叶片枯黄
            "cotton bolls splitting open revealing fluffy white cotton fiber bursting out, brown dried boll shells",
            "cotton field with open bolls, white fluffy cotton exposed from cracked capsules, ready for picking",
            "mature cotton with bolls cracked wide open, white cotton fiber piling out, dried brown bracts around bolls",
            "cotton at harvest stage, open white bolls with fiber bulging out, some dried yellowing leaves",
            "cotton bolls fully opened, white cotton masses visible, dried boll shells curling back, harvest-ready field",
            "cotton field with bolls bursting open, white fiber balls on the plants, brown dried capsule walls",
        ],
    },
}


def build_prompts():
    """为每个作物x生长阶段构造CLIP文字提示"""
    prompts = {}
    for class_name, info in CLASS_MAP.items():
        crop_en = info["crop_en"]
        stage_en = info["stage_en"]

        desc_list = VISUAL_DESCRIPTIONS.get(crop_en, {}).get(stage_en, [])
        if not desc_list:
            desc_list = [f"a photo of {crop_en.replace('_', ' ')} at {stage_en.replace('_', ' ')} stage"]

        template_prompts = [
            f"a photograph of {crop_en.replace('_', ' ')} at {stage_en.replace('_', ' ')} stage in an agricultural field",
            f"{crop_en.replace('_', ' ')} crop at {stage_en.replace('_', ' ')} stage, close-up view",
            f"an image showing {crop_en.replace('_', ' ')} during its {stage_en.replace('_', ' ')} growth phase",
        ]

        prompts[class_name] = {
            "prompts": desc_list + template_prompts,
            "info": info
        }
    return prompts


def _resolve_model_name(model_name):
    """解析模型名称：支持简写和完整HF路径"""
    if model_name in AVAILABLE_MODELS:
        return AVAILABLE_MODELS[model_name]
    return model_name


def _detect_model_type(model_name):
    """检测模型类型"""
    name_lower = model_name.lower()
    if "siglip2" in name_lower:
        return "siglip2"
    elif "siglip" in name_lower:
        return "siglip"
    elif "clip" in name_lower:
        return "clip"
    return "clip"


class CLIPCropClassifier:
    def __init__(self, model_name="siglip2-so400m", device=None):
        model_name = _resolve_model_name(model_name)
        self.model_type = _detect_model_type(model_name)

        if device:
            self.device = device
        else:
            use_cuda = False
            if torch.cuda.is_available():
                try:
                    t = torch.zeros(1, device="cuda")
                    _ = t + 1
                    use_cuda = True
                except RuntimeError:
                    pass
            self.device = "cuda" if use_cuda else "cpu"

        print(f"正在加载模型: {model_name} (类型: {self.model_type}, 设备: {self.device}) ...")

        if self.model_type == "siglip":
            from transformers import AutoModel, AutoProcessor
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_name)
        elif self.model_type == "siglip2":
            from transformers import AutoModel, AutoProcessor
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_name)
        else:
            from transformers import CLIPModel, CLIPProcessor
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)

        self.model.eval()
        print(f"模型加载完成 (设备: {self.device})")

        self.prompts_data = build_prompts()
        self.class_names = list(self.prompts_data.keys())

        # 预编码所有文字
        self._encode_all_prompts()

    def _encode_all_prompts(self):
        """预编码所有文字提示"""
        all_texts = []
        self.prompt_to_class = []

        for class_name, data in self.prompts_data.items():
            for prompt in data["prompts"]:
                all_texts.append(prompt)
                self.prompt_to_class.append(class_name)

        batch_size = 32
        all_features = []
        with torch.no_grad():
            for i in range(0, len(all_texts), batch_size):
                batch = all_texts[i:i + batch_size]
                inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                features = self.model.get_text_features(**inputs)
                if isinstance(features, torch.Tensor):
                    pass
                elif hasattr(features, 'pooler_output'):
                    features = features.pooler_output
                else:
                    features = features.last_hidden_state[:, 0, :]
                features = features / features.norm(dim=-1, keepdim=True)
                all_features.append(features.cpu())

        self.text_features = torch.cat(all_features, dim=0)
        print(f"已编码 {len(all_texts)} 条文字提示，覆盖 {len(self.class_names)} 个类别")

    def _encode_image(self, image):
        """编码单张图片为特征向量"""
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        features = self.model.get_image_features(**inputs)
        if isinstance(features, torch.Tensor):
            pass
        elif hasattr(features, 'pooler_output'):
            features = features.pooler_output
        else:
            features = features.last_hidden_state[:, 0, :]
        return features / features.norm(dim=-1, keepdim=True)

    def _get_logit_scale(self):
        if hasattr(self.model, 'logit_scale'):
            return self.model.logit_scale.exp().item()
        return 100.0  # SigLIP默认温度

    def _scores_to_class_scores(self, similarities):
        """将prompt级别的相似度聚合为类别级别的分数"""
        class_score_sums = {}
        class_score_counts = {}
        for i, class_name in enumerate(self.prompt_to_class):
            score = similarities[i].item()
            class_score_sums[class_name] = class_score_sums.get(class_name, 0.0) + score
            class_score_counts[class_name] = class_score_counts.get(class_name, 0) + 1
        return {k: class_score_sums[k] / class_score_counts[k] for k in class_score_sums}

    @torch.no_grad()
    def predict(self, image, top_k=5):
        # 测试时增强：原图 + 水平翻转，取平均
        augments = [image, ImageOps.mirror(image)]
        logit_scale = self._get_logit_scale()

        all_class_scores = []
        for aug_img in augments:
            img_feat = self._encode_image(aug_img)
            sims = (logit_scale * img_feat.cpu() @ self.text_features.T).squeeze(0)
            all_class_scores.append(self._scores_to_class_scores(sims))

        # 多增强结果取平均
        class_scores = {}
        for k in all_class_scores[0]:
            class_scores[k] = np.mean([s[k] for s in all_class_scores])

        # 两阶段策略：先识别作物，再在作物内识别阶段
        crop_scores = {}
        for class_name, score in class_scores.items():
            crop = self.prompts_data[class_name]["info"]["crop_en"]
            if crop not in crop_scores or score > crop_scores[crop]:
                crop_scores[crop] = score

        # 找到最可能的作物
        best_crop = max(crop_scores, key=crop_scores.get)

        # 只在该作物的阶段内排序（减少跨作物误判）
        crop_classes = {k: v for k, v in class_scores.items()
                        if self.prompts_data[k]["info"]["crop_en"] == best_crop}
        sorted_classes = sorted(crop_classes.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # softmax概率
        scores = np.array([s for _, s in sorted_classes])
        exp_scores = np.exp(scores - scores.max())
        probs = exp_scores / exp_scores.sum()

        results = []
        for (class_name, _), prob in zip(sorted_classes, probs):
            info = self.prompts_data[class_name]["info"]
            results.append({
                "class_name": class_name,
                "confidence": float(prob),
                "info": {
                    "crop_name": info["crop_cn"],
                    "stage_name": info["stage_cn"],
                    "stage_days": info["days"],
                    "total_days": info["total_days"],
                    "description": info["description"],
                    "crop_en": info["crop_en"],
                    "stage_en": info["stage_en"],
                }
            })
        return results


_classifier = None

def get_clip_classifier(model_name="siglip2-so400m", device=None):
    global _classifier
    if _classifier is None:
        _classifier = CLIPCropClassifier(model_name=model_name, device=device)
    return _classifier
