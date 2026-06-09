#!/bin/bash
# ============================================================
# 数据同步脚本 - 在服务器和本地之间同步数据
# 用法:
#   下载反馈图片: bash sync.sh download user@server
#   上传新模型:   bash sync.sh upload user@server /path/to/model.pth
# ============================================================

set -e

ACTION=$1
SERVER=$2
REMOTE_DIR="/path/to/crop_recognition"  # 修改为服务器实际路径
LOCAL_DIR="."

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_help() {
    echo "用法:"
    echo "  bash sync.sh download <user@server>           # 下载反馈图片"
    echo "  bash sync.sh upload <user@server> <model.pth>  # 上传新模型"
    echo "  bash sync.sh status <user@server>              # 查看服务器状态"
    echo ""
    echo "示例:"
    echo "  bash sync.sh download root@192.168.1.100"
    echo "  bash sync.sh upload root@192.168.1.100 saved_models/clip/incremental_xxx/best.pth"
}

download_feedback() {
    echo -e "${GREEN}📥 下载用户反馈图片...${NC}"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOCAL_SAVE="dataset/feedback_from_server_${TIMESTAMP}"

    mkdir -p "$LOCAL_SAVE"
    scp -r "${SERVER}:${REMOTE_DIR}/dataset/user_feedback/" "$LOCAL_SAVE/"

    # 统计
    TOTAL=$(find "$LOCAL_SAVE" -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" | wc -l)
    echo -e "${GREEN}✅ 下载完成！${NC}"
    echo "  保存位置: $LOCAL_SAVE"
    echo "  图片数量: $TOTAL"
    echo ""
    echo "下一步:"
    echo "  1. 将图片复制到 dataset/train/ 对应类别目录"
    echo "  2. 运行增量训练: python scripts/incremental_train.py"
    echo "  3. 训练完成后上传新模型: bash sync.sh upload $SERVER <model.pth>"
}

upload_model() {
    MODEL_PATH=$3
    if [ -z "$MODEL_PATH" ]; then
        echo "❌ 请指定模型文件路径"
        exit 1
    fi

    if [ ! -f "$MODEL_PATH" ]; then
        echo "❌ 模型文件不存在: $MODEL_PATH"
        exit 1
    fi

    echo -e "${GREEN}📤 上传新模型...${NC}"
    echo "  模型: $MODEL_PATH"
    echo "  目标: $SERVER:$REMOTE_DIR/"

    scp "$MODEL_PATH" "${SERVER}:${REMOTE_DIR}/new_model.pth"

    echo -e "${YELLOW}🔄 在服务器上更新模型...${NC}"
    ssh "$SERVER" "cd $REMOTE_DIR && source venv/bin/activate && python scripts/update_model.py --model new_model.pth"

    echo -e "${GREEN}✅ 模型更新完成！${NC}"
}

check_status() {
    echo -e "${GREEN}📊 服务器状态:${NC}"
    ssh "$SERVER" "
        cd $REMOTE_DIR
        echo '=== 反馈图片统计 ==='
        find dataset/user_feedback -name '*.jpg' -o -name '*.png' 2>/dev/null | wc -l
        echo ''
        echo '=== 当前模型 ==='
        python scripts/update_model.py --info 2>/dev/null || echo '未找到模型'
        echo ''
        echo '=== 服务状态 ==='
        systemctl is-active crop-recognition 2>/dev/null || echo '未使用 systemd'
    "
}

# 主逻辑
case $ACTION in
    download)
        if [ -z "$SERVER" ]; then
            echo "❌ 请指定服务器地址"
            print_help
            exit 1
        fi
        download_feedback
        ;;
    upload)
        upload_model "$@"
        ;;
    status)
        if [ -z "$SERVER" ]; then
            echo "❌ 请指定服务器地址"
            print_help
            exit 1
        fi
        check_status
        ;;
    *)
        print_help
        ;;
esac
