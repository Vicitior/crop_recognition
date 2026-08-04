# 🌾 农作物生长阶段识别系统 - 微信小程序使用说明文档

本项目包含完整的 **微信小程序前端工程 (`wechat_mini_program/`)**，支持在手机微信中一键选择照片、实时拍摄、识别玉米/小麦/棉花 15 个生育期阶段，并获取针对性的农艺管理建议，同时支持用户纠错反馈与远程 API 服务器配置。

---

## 📁 小程序工程目录结构

```
wechat_mini_program/
├── app.json                # 小程序全局页面路由、TabBar 导航与标题栏样式配置
├── app.js                  # 小程序全局数据、API URL 映射与历史持久化
├── app.wxss                # 高颜值农业绿色 CSS 设计系统
├── project.config.json     # 微信开发者工具项目配置文件
├── sitemap.json            # 微信索引规则
├── images/                 # TabBar 矢量图标
└── pages/
    ├── index/              # 🌾 AI 智能诊断主页 (拍照/选图、Top3概率、农艺指导、纠错反馈)
    ├── knowledge/          # 📚 15 个生育期科普图谱与管理百科
    └── history/            # 📜 历史诊断与纠错记录
```

---

## 🚀 导入与运行步骤

### 1. 安装微信开发者工具

从微信官方下载并安装 [微信开发者工具](https://developers.weixin.qq.com/miniprogram/dev/devtools/download.html)。

### 2. 导入小程序工程

1. 打开微信开发者工具，点击 **“+ 导入项目”**。
2. **目录**：选择本项目中的 `wechat_mini_program` 文件夹（绝对路径：`C:\Users\Vicitior\Desktop\新建文件夹 (4)\crop_recognition\wechat_mini_program`）。
3. **AppID**：使用你自己的微信小程序 AppID，或者选择 **“测试号 / 体验模式”**。
4. **项目名称**：填入 `农作物生育期智能诊断`。
5. 点击 **“导入”** 即可在模拟器中看到全功能界面！

---

## 🌐 接口服务器配置说明

小程序前端通过 RESTful API 与后端的 Python FastAPI 服务通信。

### 1. 启动 Python 后端 API 服务

在电脑终端运行项目根目录下的 API 启动命令：

```bash
# 启动 API 服务（默认监听 http://0.0.0.0:8000）
python run_api.py --host 0.0.0.0 --port 8000
```

### 2. 在小程序中修改服务器 URL

小程序主页 Header 右上方内置 **`⚙️ 服务器`** 按钮：
- **局域网/WiFi 调试**：在手机与电脑处于同一 WiFi 时，填入电脑的局域网 IP（例如：`http://192.168.1.100:8000`）。
- **内网穿透 / 远程测试**：填入 cpolar / ngrok 映射网址（例如：`https://xxx.cpolar.cn`）。
- **本地开发者工具模拟器**：使用默认 `http://127.0.0.1:8000` 即可。

> **提示**：在微信开发者工具设置中，建议勾选 **“不校验合法域名、web-view（域名）、TLS版本以及HTTPS证书”**，方便本地 HTTP 测试！

---

## 🌟 核心功能一览

1. **🌾 AI 智能诊断**：
   - 拍照/选图上传至 `POST /api/recognize`。
   - 实时显示 Top-1 最佳阶段大字 Badge、Top-3 概率分布进度条与置信度。
   - 自动匹配农艺指导卡片（追肥、灌溉、病虫害防治）。

2. **✏️ 结果纠错与样本收集**：
   - 当模型识别误判时，用户可点击“结果不准？点击纠错”，选择正确的作物与阶段并输入农艺备注。
   - 点击一键上传，即可调用 `POST /api/feedback/upload` 存入后端 `dataset/user_feedback/<crop>_<stage>/` 自动扩充训练集！

3. **📚 生育期图谱百科**：
   - 快速浏览棉花、玉米、小麦共 15 个阶段的形态特征与关键农艺要点。
