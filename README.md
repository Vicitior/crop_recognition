# 农作物生长阶段识别系统 (Crop Growth Stage Recognition)

[English README](README_EN.md) | [中文说明文档](README.md)

基于 CLIP + LoRA + 三大创新模块的农作物生长阶段智能识别系统，支持棉花、玉米、小麦三种作物共 15 个生长阶段的分类。包含 **Web 端 AI 诊断 UI**、**FastAPI 后端 API 服务** 及 **端侧离线/联网 Android 移动 App**。

---

## 🌾 项目简介

本系统利用 CLIP (Contrastive Language-Image Pre-Training) 视觉语言模型，结合 LoRA (Low-Rank Adaptation) 微调技术与离线 ONNX 量化推理，实现对农作物生长阶段的高精度识别。

### 🌟 核心功能亮点

1. **三大创新 AI 架构（准确率 93.53%）**：
   * **置信度引导路由 (Confidence-aware Routing)**：作物预测不确定时，同时激活多个分支加权融合，替代传统硬路由
   * **生育期关系建模 (Phenology-aware Relation)**：通过高斯邻接矩阵和图卷积，让模型理解生长阶段的连续性
   * **Adaptive LoRA Rank**：根据作物视觉复杂度自适应分配 LoRA 参数规模（玉米 rank=4，小麦 rank=8，棉花 rank=16）
2. **端侧离线 Android App (Edge AI)**：
   * **脱网超快推理**：集成 ONNX Runtime Android SDK，模型 INT8 动态量化压缩至 293MB (-74.9%)
   * **用户结果纠错与确认 (Feedback & Correction)**：识别误判时，支持用户选择正确的作物与生育期阶段并填写农艺备注
   * **本地 SQLite 数据库**：自动存储纠错记录、本地图片沙盒与标注历史 (`crop_feedback.db`)
   * **全球远程联网样本收集 (Dataset Expansion)**：一键将纠错/确认后的样本图片与标注上传回后端，自动扩充 `dataset/user_feedback/<crop>_<stage>/` 训练集样本量！
   * **动态服务器 IP 配置**：App 顶部内置 `⚙️ 服务器` 配置按钮，支持随时切换局域网 IP、云服务器公网 IP 或 cpolar/ngrok 穿透网址
3. **FastAPI 后端与数据导出**：
   * 提供 `POST /api/recognize`、`POST /api/feedback/upload` 等 RESTful 接口
   * 一键导出未标记样本为标准的 JSON / ZIP 数据集，方便持续增量训练

---

## 🌾 支持的作物与 15 个生长阶段

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

---

## 📊 模型精度

| 模型 | 验证集准确率 (Val Acc) | 说明 |
|------|-------------|------|
| **创新模型（三大创新组合）** | **93.53%** | 从零训练，当前最佳 |
| 创新模型（继承基线） | 92.23% | 从基线 84.73% 继续训练 |
| 基线 CLIP ViT-L/14@336 + LoRA | 84.73% | 原始微调模型 |

---

## 🚀 快速开始与部署指南

### 1. 环境准备与安装

```bash
# 1. 克隆项目
git clone https://github.com/Vicitior/crop_recognition.git
cd crop_recognition

# 2. 创建虚拟环境 (推荐)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. 安装依赖
pip install -r requirements.txt
```

---

### 2. 启动 Web 诊断界面 (Gradio UI)

```bash
# 使用创新模型 (默认，93.53% 准确率)
python app.py

# 指定端口与开启公网分享
python app.py --port 7860 --share
```
启动后访问 `http://localhost:7860` 即可使用高颜值 Web 诊断界面。

---

### 3. 部署 FastAPI 后端 (用于 Android App 联网与样本收集)

```bash
# 启动 API 后端服务 (监听 0.0.0.0 端口 8000)
python run_api.py --host 0.0.0.0 --port 8000
```

#### 🌐 远程/联网样本收集部署方案：
* **局域网/WiFi 收集**：如果电脑与手机在同一 WiFi 下，App 中填写 `http://电脑局域网IP:8000`（如 `http://192.168.1.100:8000`）。
* **云服务器/公网收集**：将 API 后端部署至阿里云/腾讯云/华为云 VPS，App 中填写 `http://云公网IP:8000`。
* **免费内网穿透**：运行 `cpolar http 8000` 或 `ngrok http 8000`，在 App 中粘贴生成的 HTTPS 公网域名（如 `https://xxx.cpolar.cn`）。

---

### 4. 📱 端侧 Android App 部署与使用

#### A. 导出与量化 ONNX 模型 (可选，已内置)
```bash
# 1. 导出端到端 ONNX 计算图
python scripts/export_onnx.py

# 2. 执行 INT8 动态量化 (1.16GB -> 293MB)
python scripts/quantize_onnx.py
```

#### B. 编译与安装 Android App
1. 打开 **Android Studio**，选择 `Open` 并选中 `crop_recognition/android_app` 目录。
2. 连接 Android 手机或模拟器，点击 `Run 'app'`（或点击 `Build` -> `Build APK(s)` 导出 `.apk` 安装包）。
3. 安装后打开 App：
   * 拍照或选择农作物图片即可秒级查看诊断结果。
   * 点击顶部 **`⚙️ 服务器`** 按钮可动态修改后端 API 地址。
   * 识别有偏差时，点击 **`✏️ 结果纠错`** 重新校准作物与生育期。
   * 点击 **`☁️ 上传样本库`** 即可将照片与标注联网同步至服务器 `dataset/user_feedback/` 扩充样本量！

---

## 📈 训练与微调模型

### 训练创新模型
```bash
python scripts/train_innovations.py --all --epochs 20 --lr 5e-5
```

### 导出用户收集的反馈数据集
```bash
python -c "from api.service import record_service; print(record_service.export_data())"
```

---

## 📁 项目结构

```
crop_recognition/
├── app.py                          # Gradio Web 诊断应用
├── run_api.py                      # FastAPI 后端启动服务
├── requirements.txt                # Python 依赖清单
├── README.md                       # 项目说明文档
├── api/                            # RESTful API 后端 (FastAPI + SQLite)
│   ├── main.py                     # API 路由 (/api/recognize, /api/feedback/upload 等)
│   ├── database.py                 # SQLite 数据库 DAO
│   └── service.py                  # 业务服务与数据集样本保存逻辑
├── android_app/                    # 端侧 Android 工程 (Kotlin + ONNX)
│   ├── app/src/main/java/com/example/croprecognition/
│   │   ├── MainActivity.kt         # 主界面 UI 与事件绑定
│   │   ├── CropRecognitionEngine.kt# ONNX 本地离线推理引擎
│   │   ├── CorrectionDialog.kt     # 用户纠错与标注弹窗
│   │   ├── CropDatabaseHelper.kt   # Android 本地 SQLite 数据库
│   │   └── DatasetUploader.kt      # HTTP Multipart 样本上传器
│   └── app/src/main/res/layout/
│       └── activity_main.xml       # App 界面布局
├── models/                         # 模型定义 (CLIP, 路由, 图卷积, Adaptive LoRA)
├── scripts/                        # 导出、量化、训练与评估脚本
├── saved_models/                   # 训练保存的模型权重
└── dataset/                        # 训练数据集与 user_feedback 用户反馈扩充库
```

---

## 📝 更新日志

### 2026-08-03 (v2.5 Android 纠错、SQLite 与远程样本库扩充版)
- ✏️ **Android 用户结果纠错**：新增 `CorrectionDialog` 交互弹窗，支持选择正确的作物与生育期阶段、添加农艺说明备注。
- 💾 **Android 本地 SQLite 数据库**：集成 `CropDatabaseHelper.kt`，本地离线持久化存储用户纠错与历史诊断记录 (`crop_feedback.db`)。
- ☁️ **全球远程样本库上传管道**：实现 `DatasetUploader.kt` HTTP Multipart 协议与 FastAPI `POST /api/feedback/upload` 接口，一键将纠错图片与标注发送至服务端 `dataset/user_feedback/<crop>_<stage>/` 目录扩充训练样本量。
- ⚙️ **动态服务器配置 UI**：App 顶栏增加 `⚙️ 服务器` 按钮与 API 地址状态徽章，支持随时切换局域网、云服务器公网或内网穿透网址。
- 🎨 **Header 界面重构**：优化全宽两行响应式 Header 布局，提升按钮触控体验与视觉精致度。

---

## 📄 许可证

本项目仅供学习、农艺科研与学术研究使用。

## 📧 联系方式

GitHub: https://github.com/Vicitior/crop_recognition  
如有问题或建议，欢迎提交 Issue 或 Pull Request！
