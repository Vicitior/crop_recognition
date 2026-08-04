# 微信云托管 (WeChat Cloud Base) 容器构建文件
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量，加速 Python 依赖安装并避免生成 pyc 缓存
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai

# 安装系统依赖（使用兼容 Debian 12 的 libgl1 替代过期的 libgl1-mesa-glx）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目代码与保存的模型权重
COPY . .

# 暴露 8000 端口（微信云托管默认通信端口）
EXPOSE 8000

# 启动 FastAPI API 后端服务
CMD ["python", "run_api.py", "--host", "0.0.0.0", "--port", "8000"]
