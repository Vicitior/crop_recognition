#!/bin/bash
# ============================================================
# 农作物识别系统 - 服务器部署脚本
# 用法: bash deploy.sh [--port 7860] [--gpu]
# ============================================================

set -e

# 默认配置
PORT=7860
USE_GPU=false
PROJECT_DIR="crop_recognition"
REPO_URL="https://github.com/Vicitior/crop_recognition.git"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --port) PORT="$2"; shift 2 ;;
        --gpu) USE_GPU=true; shift ;;
        --dir) PROJECT_DIR="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

echo "=========================================="
echo "  🌾 农作物识别系统 - 服务器部署"
echo "=========================================="
echo "  端口: $PORT"
echo "  GPU: $USE_GPU"
echo "  目录: $PROJECT_DIR"
echo "=========================================="

# 1. 克隆代码
if [ ! -d "$PROJECT_DIR" ]; then
    echo ""
    echo "📥 克隆代码..."
    git clone $REPO_URL $PROJECT_DIR
fi

cd $PROJECT_DIR

# 2. 创建虚拟环境
echo ""
echo "🐍 创建虚拟环境..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# 3. 安装依赖
echo ""
echo "📦 安装依赖..."
pip install --upgrade pip

if [ "$USE_GPU" = true ]; then
    echo "  安装 GPU 版本 PyTorch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
    echo "  安装 CPU 版本 PyTorch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

pip install -r requirements.txt

# 4. 创建必要目录
echo ""
echo "📁 创建目录结构..."
mkdir -p saved_models/clip
mkdir -p dataset/user_feedback
mkdir -p dataset/train
mkdir -p logs

# 5. 检查模型
echo ""
echo "🔍 检查模型文件..."
if [ -f "saved_models/clip/clip-vit-large-patch14-336-v2/best.pth" ]; then
    echo "  ✅ 找到 CLIP 微调模型"
elif [ -f "saved_models/clip/hot_reload/best.pth" ]; then
    echo "  ✅ 找到热加载模型"
else
    echo "  ⚠️ 未找到模型文件！"
    echo "  请上传模型到 saved_models/clip/ 目录"
    echo "  或使用以下命令更新模型:"
    echo "    python scripts/update_model.py --model /path/to/model.pth"
fi

# 6. 创建 systemd 服务（可选）
echo ""
read -p "是否创建系统服务（开机自启）？[y/N] " CREATE_SERVICE
if [ "$CREATE_SERVICE" = "y" ] || [ "$CREATE_SERVICE" = "Y" ]; then
    SERVICE_FILE="/etc/systemd/system/crop-recognition.service"
    WORK_DIR=$(pwd)
    PYTHON_PATH=$(which python)

    sudo tee $SERVICE_FILE > /dev/null <<EOF
[Unit]
Description=Crop Recognition Web Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$WORK_DIR
Environment=PATH=$WORK_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=$WORK_DIR/venv/bin/python app.py --port $PORT
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable crop-recognition
    echo "  ✅ 服务已创建"
    echo "  启动: sudo systemctl start crop-recognition"
    echo "  状态: sudo systemctl status crop-recognition"
    echo "  日志: journalctl -u crop-recognition -f"
else
    echo "  跳过服务创建"
fi

# 7. 启动服务
echo ""
echo "=========================================="
echo "  ✅ 部署完成！"
echo "=========================================="
echo ""
echo "  启动服务:"
echo "    cd $PROJECT_DIR"
echo "    source venv/bin/activate"
echo "    python app.py --port $PORT"
echo ""
echo "  或使用 systemd:"
echo "    sudo systemctl start crop-recognition"
echo ""
echo "  访问地址: http://$(hostname -I | awk '{print $1}'):$PORT"
echo ""
echo "  日常运维:"
echo "    导出图片: Web 界面点击 '打包导出图片'"
echo "    更新模型: python scripts/update_model.py --model /path/to/model.pth"
echo "    查看模型: python scripts/update_model.py --info"
echo ""
