# 农作物生长阶段识别系统 (Crop Growth Stage Recognition)

基于 CLIP + LoRA 的农作物生长阶段智能识别系统，支持棉花、玉米、小麦三种作物共15个生长阶段的分类。

## 项目简介

本系统利用 CLIP (Contrastive Language-Image Pre-Training) 视觉语言模型，结合 LoRA (Low-Rank Adaptation) 微调技术，实现对农作物生长阶段的高精度识别。通过 Gradio 提供友好的 Web 界面，支持图片上传、实时预测和一键保存训练数据。

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

### 当前最佳模型

| 模型 | 测试集准确率 | 说明 |
|------|-------------|------|
| CLIP ViT-L/14@336 + LoRA | **84.73%** | 15类全量任务最佳 |
| 棉花5类集成 | 95.65% | 仅棉花5类 |

### 逐类准确率

| 类别 | 准确率 | 备注 |
|------|--------|------|
| corn_maturity | 100% | |
| corn_seedling | 90% | |
| corn_tasseling | 100% | |
| corn_filling | 33.33% | ⚠️ 待提升 |
| corn_jointing | 33.33% | ⚠️ 待提升 |
| wheat_maturity | 100% | |
| wheat_seedling | 100% | |
| wheat_heading | 83.33% | |
| wheat_tillering | 66.67% | |
| wheat_jointing | 50% | ⚠️ 待提升 |
| cotton_seedling | 95.45% | |
| cotton_squaring | 89.29% | |
| cotton_flowering | 66.67% | |
| cotton_boll_setting | 72.41% | |
| cotton_boll_opening | 100% | |

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

### 准备数据集和模型

由于模型和数据集文件较大，需要单独准备：

```bash
# 1. 创建目录
mkdir -p saved_models/clip/clip-vit-large-patch14-336-v2
mkdir -p dataset/{train,val,test,user_feedback}

# 2. 从本地复制模型文件
# 将 best.pth 和 config.json 复制到 saved_models/clip/clip-vit-large-patch14-336-v2/

# 3. 从本地复制数据集
# 将 train/val/test 目录复制到 dataset/
```

### 启动 Web 界面

```bash
# CPU 模式（无需GPU）
python app.py

# 指定端口
python app.py --port 7860

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

```bash
# 增量训练命令
python scripts/incremental_train.py --epochs 10 --lr 1e-5
```

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

# 2. 上传模型和数据集
scp -r user@local:/path/to/saved_models/ ./
scp -r user@local:/path/to/dataset/ ./

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动服务
python app.py --port 7860 --share
```

### 方案3: Docker 部署（可选）

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "app.py", "--port", "7860"]
```

## 📈 训练模型

### 从头训练

```bash
# 使用推荐配置
python scripts/train_clip_v2.py \
    --model openai/clip-vit-large-patch14-336 \
    --lora-rank 16 \
    --epochs 50 \
    --lr 5e-4
```

### 增量训练（推荐）

```bash
# 在现有模型基础上继续训练
python scripts/incremental_train.py \
    --model-path saved_models/clip/clip-vit-large-patch14-336-v2/best.pth \
    --epochs 10 \
    --lr 1e-5
```

### 使用 Focal Loss（处理类别不平衡）

```bash
python scripts/train_clip_v2.py \
    --model openai/clip-vit-large-patch14-336 \
    --use-focal-loss \
    --focal-gamma 2.0
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
├── app.py                          # Gradio Web 应用
├── requirements.txt                # Python 依赖
├── README.md                       # 项目说明
├── .gitignore                      # Git 忽略规则
├── dataset/                        # 数据集
│   ├── train/                      # 训练集
│   ├── val/                        # 验证集
│   ├── test/                       # 测试集
│   └── user_feedback/              # 用户反馈数据
├── models/                         # 模型定义
│   ├── clip_classifier.py          # CLIP 零样本分类器
│   ├── classifier.py               # EfficientNet 分类器
│   └── growth_stages.py            # 作物生长阶段知识库
├── scripts/                        # 训练和评估脚本
│   ├── train_clip_v2.py            # 主训练脚本
│   ├── incremental_train.py        # 增量训练脚本
│   ├── ensemble_predict.py         # 模型集成评估
│   ├── prepare_deployment.py       # 部署准备脚本
│   └── ...
└── saved_models/                   # 保存的模型
    └── clip/
        └── clip-vit-large-patch14-336-v2/
            ├── best.pth            # 最佳模型权重
            └── config.json         # 模型配置
```

## 🔧 常见问题

### Q: 没有GPU能用吗？
**A**: 可以！CPU完全可以运行推理（识别图片），只是速度稍慢（2-5秒/张）。训练建议用GPU。

### Q: 如何添加新的作物类别？
**A**: 
1. 在 `dataset/user_feedback/` 下创建新类别目录
2. 上传该类别的图片
3. 运行增量训练

### Q: 模型文件在哪里？
**A**: 模型文件太大（1.7GB），不在GitHub上。需要从本地复制到服务器的 `saved_models/` 目录。

### Q: 如何提升准确率？
**A**: 
1. 收集更多该类别的图片
2. 运行增量训练
3. 使用 Focal Loss 处理类别不平衡

## 📝 更新日志

### 2026-06-08
- 新增增量训练功能
- 优化零样本提示词（添加对比描述）
- 添加 Focal Loss 支持
- 优化课程学习调度
- 创建部署准备脚本

### 2026-05-27
- 完成15类全量任务训练
- 基线准确率达到84.73%

## 📄 许可证

本项目仅供学习和研究使用。

## 📧 联系方式

GitHub: https://github.com/Vicitior/crop_recognition

如有问题或建议，请提交 Issue 或 Pull Request。
