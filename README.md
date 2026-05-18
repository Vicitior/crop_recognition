# 农作物生长阶段识别系统 (Crop Growth Stage Recognition)

基于 CLIP + LoRA 的农作物生长阶段智能识别系统，支持棉花、玉米、小麦三种作物的生长阶段分类。

## 项目简介

本系统利用 CLIP (Contrastive Language-Image Pre-Training) 视觉语言模型，结合 LoRA (Low-Rank Adaptation) 微调技术，实现对农作物生长阶段的高精度识别。通过 Gradio 提供友好的 Web 界面，支持图片上传和实时预测。

## 模型精度

### 最终测试结果

| 模型 | 测试集准确率 | 验证集准确率 |
|------|-------------|-------------|
| CLIP ViT-L/14@336 + LoRA (原版) | 91.30% | 87.36% |
| CLIP ViT-L/14@336 + LoRA V2 (优化版) | 92.75% | 92.72% |
| **4 模型集成** | **95.65%** | - |

### 逐类准确率（4 模型集成）

| 类别 | 准确率 |
|------|--------|
| cotton_boll_opening (吐絮期) | 100.00% |
| cotton_seedling (苗期) | 100.00% |
| cotton_squaring (蕾期) | 96.43% |
| cotton_flowering (开花期) | 93.33% |
| cotton_boll_setting (结铃期) | 89.66% |

### 模型对比

| 模型 | 参数量 | 可训练参数 | 特点 |
|------|--------|-----------|------|
| CLIP ViT-B/32 + LoRA | 153M | 265K | 轻量级，推理速度快 |
| CLIP ViT-L/14 + LoRA | 432M | 396K | 平衡性能和速度 |
| CLIP ViT-L/14@336 + LoRA | 433M | 396K | 高分辨率，精度最高 |
| CLIP ViT-L/14@336 + LoRA V2 | 438M | 1.3M | 优化版，更强的正则化 |

## 支持的作物和生长阶段

### 棉花 (Cotton) - 5 个阶段
- **苗期 (Seedling)**: 出苗至现蕾前
- **蕾期 (Squaring)**: 现蕾至开花
- **开花期 (Flowering)**: 开花至结铃
- **结铃期 (Boll Setting)**: 结铃至吐絮
- **吐絮期 (Boll Opening)**: 吐絮至收获

### 玉米 (Corn) - 5 个阶段 [待数据收集]
- 出苗期、拔节期、抽穗期、灌浆期、成熟期

### 小麦 (Wheat) - 5 个阶段 [待数据收集]
- 出苗期、分蘖期、拔节期、抽穗期、成熟期

## 项目结构

```
crop_recognition/
├── app.py                          # Gradio Web 应用
├── requirements.txt                # Python 依赖
├── README.md                       # 项目说明
├── .gitignore                      # Git 忽略规则
├── dataset/                        # 数据集 (需单独下载)
│   ├── train/                      # 训练集 (70%)
│   ├── val/                        # 验证集 (20%)
│   └── test/                       # 测试集 (10%)
├── models/                         # 模型定义
│   ├── classifier.py               # EfficientNet-B0 分类器
│   ├── clip_classifier.py          # CLIP 零样本分类器
│   └── growth_stages.py            # 作物生长阶段知识库
├── scripts/                        # 训练和评估脚本
│   ├── train.py                    # EfficientNet 训练脚本
│   ├── train_clip.py               # CLIP LoRA 训练脚本 (V1)
│   ├── train_clip_v2.py            # CLIP LoRA 训练脚本 (V2, 优化版)
│   ├── evaluate.py                 # 模型评估脚本
│   ├── evaluate_all_models.py      # 全模型基准测试
│   ├── ensemble_predict.py         # 模型集成评估
│   ├── predict.py                  # 单图预测
│   ├── predict_clip.py             # CLIP 单图预测
│   └── download_dataset.py         # 数据集构建工具
├── utils/                          # 工具模块
│   ├── augmentation.py             # 数据增强
│   └── dataset.py                  # 数据加载器
└── saved_models/                   # 保存的模型 (需单独下载)
    ├── clip/
    │   ├── clip-base/              # CLIP ViT-B/32 LoRA
    │   ├── clip-large/             # CLIP ViT-L/14 LoRA
    │   ├── clip-large-336/         # CLIP ViT-L/14@336 LoRA
    │   ├── clip-vit-large-patch14-336-v2/  # CLIP V2 优化版
    │   ├── eval_results.json       # 零样本评估结果
    │   ├── model_ranking.json      # 模型排名
    │   └── ensemble_results.json   # 集成评估结果
    └── training_curves.png         # 训练曲线
```

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (推荐使用 GPU)
- 8GB+ GPU 显存

### 安装

```bash
# 克隆项目
git clone https://github.com/your-username/crop_recognition.git
cd crop_recognition

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 下载预训练模型

由于模型文件较大（约 11GB），需要单独下载：

```bash
# 创建模型目录
mkdir -p saved_models/clip

# 下载模型文件 (请从 Releases 页面下载)
# 或使用以下命令训练自己的模型
```

### 准备数据集

```bash
# 查看数据集收集指南
python scripts/download_dataset.py --guide

# 创建类别目录
python scripts/download_dataset.py --create-dirs dataset/raw

# 将收集的图片放入 dataset/raw/ 对应目录
# 然后自动划分 train/val/test
python scripts/download_dataset.py --split
```

## 使用方法

### Web 界面

```bash
python app.py
```

启动后访问 `http://localhost:7860` 即可使用 Web 界面上传图片进行预测。

### 命令行预测

```bash
# 使用 CLIP 零样本预测
python scripts/predict.py --image path/to/image.jpg

# 使用 fine-tuned CLIP 预测
python scripts/predict_clip.py --image path/to/image.jpg --model-dir saved_models/clip/clip-large-336
```

### 训练模型

```bash
# 训练 CLIP V1 (原版)
python scripts/train_clip.py --model openai/clip-vit-large-patch14-336 --method lora --epochs 20

# 训练 CLIP V2 (优化版, 推荐)
python scripts/train_clip_v2.py --model openai/clip-vit-large-patch14-336

# 评估模型
python scripts/evaluate.py --model-path saved_models/clip/clip-large-336/best.pth

# 集成评估
python scripts/ensemble_predict.py
```

## 技术细节

### V2 优化改进

相比 V1 版本，V2 训练脚本包含以下改进：

| 改进项 | V1 | V2 |
|--------|----|----|
| LoRA rank | 8 | 16 |
| 训练轮数 | 15 | 50 |
| 分类头 | 单层 | 两层 + BatchNorm |
| 数据增强 | 基础 | Mixup + RandomErasing + Affine |
| 标签平滑 | 无 | 0.1 |
| 学习率调度 | Cosine | Warmup + Cosine |
| Early Stopping | 无 | 10 轮耐心 |
| 梯度裁剪 | 无 | max_norm=1.0 |

### 模型架构

```
CLIP ViT-L/14@336 (冻结) + LoRA 适配器
    ↓
Image Features (768-dim)
    ↓
Linear(768, 1024) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Linear(1024, 512) → BatchNorm → ReLU → Dropout(0.2)
    ↓
Linear(512, 5)
    ↓
预测类别
```

### 集成策略

系统采用 4 个模型的简单平均集成：
- CLIP ViT-B/32 + LoRA (V1)
- CLIP ViT-L/14 + LoRA (V1)
- CLIP ViT-L/14@336 + LoRA (V1)
- CLIP ViT-L/14@336 + LoRA V2

每个模型使用对应分辨率的图片进行预测，最后对 softmax 概率取平均。

## 数据集来源

- **棉花数据集**: 自行采集和标注，包含 1,323 张棉花不同生长阶段的图片
- **玉米/小麦**: 待收集 (参见 `scripts/download_dataset.py` 中的收集指南)

## 精度提升方案

本项目实现了5种最新的精度提升方法，详见 [ACCURACY_IMPROVEMENT_GUIDE.md](ACCURACY_IMPROVEMENT_GUIDE.md)。

### 测试时增强 (TTA) - 已验证有效

| 指标 | 无TTA | 有TTA | 提升 |
|------|-------|-------|------|
| **总体准确率** | 92.75% | 94.93% | **+2.17%** |
| cotton_boll_setting | 79.31% | 89.66% | **+10.34%** |

```bash
# 使用TTA预测
python scripts/predict_tta.py --image test.jpg --tta-level medium
```

### 其他方案

| 方案 | 脚本 | 预期提升 |
|------|------|---------|
| 类别级数据增强 | `train_with_class_augmentation.py` | +1-3% |
| 提示学习 (CoOp/MaPLe) | `train_prompt_learning.py` | +2-4% |
| MMD-LoRA多模态融合 | `train_mmd_lora.py` | +2-5% |
| 注意力机制增强 | `train_with_attention.py` | +1-2% |

## 引用

如果您使用了本项目，请引用：

```bibtex
@misc{crop_recognition,
  title={Crop Growth Stage Recognition using CLIP and LoRA},
  year={2026},
  howpublished={\url{https://github.com/your-username/crop_recognition}}
}
```

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。
