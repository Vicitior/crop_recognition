# 农作物生长阶段识别系统 (Crop Growth Stage Recognition)

基于 CLIP + LoRA + 三大创新模块的农作物生长阶段智能识别系统，支持棉花、玉米、小麦三种作物共15个生长阶段的分类。

## 项目简介

本系统利用 CLIP (Contrastive Language-Image Pre-Training) 视觉语言模型，结合 LoRA (Low-Rank Adaptation) 微调技术，实现对农作物生长阶段的高精度识别。通过 Gradio 提供友好的 Web 界面，支持图片上传、实时预测和一键保存训练数据。

### 三大创新点

1. **置信度引导路由 (Confidence-aware Routing)**：作物预测不确定时，同时激活多个分支加权融合，替代传统硬路由
2. **生育期关系建模 (Phenology-aware Relation)**：通过高斯邻接矩阵和图卷积，让模型理解生长阶段的连续性
3. **Adaptive LoRA Rank**：根据作物视觉复杂度自适应分配LoRA参数规模（玉米rank=4，小麦rank=8，棉花rank=16）

## 🌾 支持的作物和生长阶段

### 棉花 (Cotton) - 5 个阶段
- **苗期 (Seedling)**: 出苗至现蕾前
- **蕾期 (Squaring)**: 现蕾至开花
- **开花期 (Flowering)**: 开花至结铃
- **结铃期 (Boll Setting)**: 结铃至吐絮
- **吐絮期 (Boll Opening)**: 吐絮至收获

### 玉米 (Corn) - 5 个阶段
- **出苗期 (Seedling)**: 出苗至三叶期
- **拔节期 (Jointing)**: 拔节至抽穗
- **抽穗期 (Tasseling)**: 抽穗至灌浆
- **灌浆期 (Filling)**: 灌浆至成熟
- **成熟期 (Maturity)**: 成熟至收获

### 小麦 (Wheat) - 5 个阶段
- **出苗期 (Seedling)**: 出苗至分蘖
- **分蘖期 (Tillering)**: 分蘖至拔节
- **拔节期 (Jointing)**: 拔节至抽穗
- **抽穗期 (Heading)**: 抽穗至成熟
- **成熟期 (Maturity)**: 成熟至收获

## 📊 模型精度

### 模型对比

| 模型 | 验证集准确率 | 说明 |
|------|-------------|------|
| **创新模型（三大创新组合）** | **93.53%** | 从零训练，当前最佳 |
| 创新模型（继承基线） | 92.23% | 从基线84.73%继续训练 |
| 基线 CLIP ViT-L/14@336 + LoRA | 84.73% | 原始微调模型 |

### 创新点效果

| 方案 | Val Acc | 提升 |
|------|---------|------|
| 基线 | 84.73% | - |
| +置信度路由 | ~87% | +2.3% |
| +生育期图建模 | ~89% | +4.3% |
| +Adaptive LoRA | ~86% | +1.3% |
| **三者组合** | **93.53%** | **+8.8%** |

## 🚀 快速开始

### 环境要求

**最低配置（CPU可用）**:
- Python 3.8+
- 内存 8GB+
- 硬盘 20GB+

**推荐配置（GPU加速）**:
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+
- GPU 显存 8GB+（如 RTX 3060/4060）

### 安装

```bash
# 克隆项目
git clone https://github.com/Vicitior/crop_recognition.git
cd crop_recognition

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 启动 Web 界面

```bash
# 使用创新模型（默认，93.53%准确率）
python app.py

# 指定端口
python app.py --port 7860

# 使用基线模型
python app.py --mode clip-finetuned

# 使用零样本模型
python app.py --mode clip

# 创建公网链接（可选）
python app.py --share
```

启动后访问 `http://localhost:7860` 即可使用。

## 📸 功能特性

### 1. 图片识别
- 上传农作物图片
- 自动识别作物类型和生长阶段
- 显示置信度和详细信息

### 2. 一键保存
- 识别后直接保存图片到训练集
- 自动按类别归类
- 支持添加备注信息

### 3. 增量训练
- 在现有模型基础上继续训练
- 自动合并新数据和原始数据
- 使用小学习率避免遗忘

## 🖥️ 服务器部署

### 方案1: 本地部署（推荐个人使用）

```bash
# 直接启动
python app.py --port 7860
```

### 方案2: 云服务器部署

```bash
# 1. 克隆代码
git clone https://github.com/Vicitior/crop_recognition.git
cd crop_recognition

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动服务
python app.py --port 7860 --share
```

## 📈 训练模型

### 训练创新模型（推荐）

```bash
# 从零训练创新模型（三大创新组合）
python scripts/train_innovations.py --all --epochs 20 --lr 5e-5

# 从基线模型继续训练
python scripts/train_innovations.py --all --epochs 20 --lr 5e-5 \
    --base-model-path saved_models/clip/clip-vit-large-patch14-336-v2/best.pth
```

### 训练基线模型

```bash
python scripts/train_clip_v2.py \
    --model openai/clip-vit-large-patch14-336 \
    --lora-rank 16 \
    --epochs 50 \
    --lr 5e-4
```

### 增量训练

```bash
python scripts/incremental_train.py \
    --model-path saved_models/clip/clip-vit-large-patch14-336-v2/best.pth \
    --epochs 10 \
    --lr 1e-5
```

## 💻 CPU vs GPU

| 功能 | CPU | GPU |
|------|-----|-----|
| 图片识别 | 2-5秒/张 | 0.1-0.5秒/张 |
| 增量训练 | 2-4小时 | 15-30分钟 |
| 从头训练 | 不推荐 | 2-3小时 |

**结论**: 
- **推理（识别图片）**: CPU完全可用，只是稍慢
- **训练**: 强烈推荐GPU，CPU训练太慢

## 📁 项目结构

```
crop_recognition/
├── app.py                          # Gradio Web 应用（支持创新模型）
├── requirements.txt                # Python 依赖
├── README.md                       # 项目说明
├── .gitignore                      # Git 忽略规则
├── models/                         # 模型定义
│   ├── clip_classifier.py          # CLIP 零样本分类器
│   ├── classifier.py               # EfficientNet 分类器
│   ├── growth_stages.py            # 作物生长阶段知识库
│   ├── confidence_router.py        # 创新1: 置信度路由
│   ├── phenology_graph.py          # 创新2: 生育期关系建模
│   └── adaptive_lora.py            # 创新3: Adaptive LoRA
├── scripts/                        # 训练和评估脚本
│   ├── train_clip_v2.py            # 基线训练脚本
│   ├── train_innovations.py        # 创新模型训练脚本
│   ├── incremental_train.py        # 增量训练脚本
│   └── ...
├── saved_models/                   # 保存的模型
│   ├── clip/
│   │   └── clip-vit-large-patch14-336-v2/
│   │       └── best.pth            # 基线模型 (1.7GB)
│   └── innovations/
│       └── all_innovations/
│           └── best.pth            # 创新模型 (18MB)
└── dataset/                        # 数据集（需自行准备）
    ├── train/
    ├── val/
    └── test/
```

## 🔧 常见问题

### Q: 没有GPU能用吗？
**A**: 可以！CPU完全可以运行推理（识别图片），只是速度稍慢（2-5秒/张）。训练建议用GPU。

### Q: 模型文件在哪里？
**A**: 
- 基线模型 (1.7GB): 通过 Git LFS 自动下载
- 创新模型 (18MB): 已包含在仓库中

### Q: 如何提升准确率？
**A**: 
1. 收集更多该类别的图片
2. 运行增量训练
3. 使用创新模型（已达到93.53%）

## 📝 更新日志

### 2026-07-03
- 实现三大创新模块：置信度路由、生育期关系建模、Adaptive LoRA
- 创新模型准确率达到93.53%（比基线提升8.8%）
- 集成创新模型到app.py（默认加载）

### 2026-06-08
- 新增增量训练功能
- 优化零样本提示词（添加对比描述）
- 添加 Focal Loss 支持

### 2026-05-27
- 完成15类全量任务训练
- 基线准确率达到84.73%

## 📄 许可证

本项目仅供学习和研究使用。

## 📧 联系方式

GitHub: https://github.com/Vicitior/crop_recognition

如有问题或建议，请提交 Issue 或 Pull Request。
