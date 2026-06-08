# -*- coding: utf-8 -*-
"""
部署准备脚本
打包模型和数据集，准备部署到服务器

用法:
    python scripts/prepare_deployment.py
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


def create_deployment_package():
    """创建部署包"""
    print("=" * 60)
    print("准备部署包")
    print("=" * 60)

    # 创建部署目录
    deploy_dir = Path("deployment/crop_recognition")
    deploy_dir.mkdir(parents=True, exist_ok=True)

    # 1. 复制模型文件
    print("\n[1/5] 复制模型文件...")
    model_src = Path("saved_models/clip/clip-vit-large-patch14-336-v2")
    model_dst = deploy_dir / "saved_models/clip/clip-vit-large-patch14-336-v2"
    model_dst.mkdir(parents=True, exist_ok=True)

    for f in model_src.glob("*"):
        if f.is_file():
            shutil.copy2(f, model_dst / f.name)
            print(f"  复制: {f.name}")

    # 2. 复制数据集
    print("\n[2/5] 复制数据集...")
    dataset_src = Path("dataset")
    dataset_dst = deploy_dir / "dataset"

    # 只复制必要的数据集子目录
    for subdir in ["train", "val", "test", "user_feedback"]:
        src = dataset_src / subdir
        dst = dataset_dst / subdir
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            count = len(list(dst.rglob("*.jpg"))) + len(list(dst.rglob("*.png")))
            print(f"  复制: {subdir}/ ({count} 张图片)")

    # 3. 复制代码文件
    print("\n[3/5] 复制代码文件...")
    code_files = [
        "app.py",
        "models/__init__.py",
        "models/clip_classifier.py",
        "models/growth_stages.py",
        "models/classifier.py",
        "scripts/train_clip_v2.py",
        "scripts/incremental_train.py",
        "scripts/ensemble_predict.py",
    ]

    for f in code_files:
        src = Path(f)
        dst = deploy_dir / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  复制: {f}")

    # 4. 创建启动脚本
    print("\n[4/5] 创建启动脚本...")

    # Windows启动脚本
    bat_content = '''@echo off
chcp 65001 >nul
echo ========================================
echo   农作物识别系统 - 服务器部署
echo ========================================
echo.

cd /d "%~dp0"

echo 安装依赖...
pip install -r requirements.txt

echo.
echo 启动Web服务...
python app.py --port 7860 --share

pause
'''
    with open(deploy_dir / "start.bat", "w", encoding="utf-8") as f:
        f.write(bat_content)
    print("  创建: start.bat")

    # Linux启动脚本
    sh_content = '''#!/bin/bash
echo "========================================"
echo "  农作物识别系统 - 服务器部署"
echo "========================================"

cd "$(dirname "$0")"

echo "安装依赖..."
pip install -r requirements.txt

echo ""
echo "启动Web服务..."
python app.py --port 7860 --share
'''
    with open(deploy_dir / "start.sh", "w", encoding="utf-8") as f:
        f.write(sh_content)
    os.chmod(deploy_dir / "start.sh", 0o755)
    print("  创建: start.sh")

    # 5. 创建requirements.txt
    print("\n[5/5] 创建requirements.txt...")
    requirements = """torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
gradio>=3.40.0
Pillow>=10.0.0
numpy>=1.24.0
peft>=0.4.0
"""
    with open(deploy_dir / "requirements.txt", "w", encoding="utf-8") as f:
        f.write(requirements)
    print("  创建: requirements.txt")

    # 创建README
    print("\n创建部署说明...")
    readme = """# 农作物识别系统 - 部署说明

## 快速启动

### Windows
```bash
start.bat
```

### Linux
```bash
chmod +x start.sh
./start.sh
```

## 手动部署

1. 安装依赖:
```bash
pip install -r requirements.txt
```

2. 启动服务:
```bash
python app.py --port 7860 --share
```

## 增量训练

当收集到新图片后，运行增量训练:
```bash
python scripts/incremental_train.py --epochs 10 --lr 1e-5
```

## 目录结构

```
crop_recognition/
├── app.py                    # Web界面
├── models/                   # 模型代码
├── scripts/                  # 训练脚本
├── saved_models/             # 训练好的模型
├── dataset/                  # 数据集
│   ├── train/               # 训练集
│   ├── val/                 # 验证集
│   ├── test/                # 测试集
│   └── user_feedback/       # 用户反馈图片
├── requirements.txt         # Python依赖
├── start.bat               # Windows启动脚本
└── start.sh                # Linux启动脚本
```

## 注意事项

1. 首次运行会自动下载CLIP模型（约1.5GB）
2. 建议使用GPU服务器以获得更好的性能
3. 用户反馈的图片会保存到 dataset/user_feedback/
4. 定期运行增量训练以更新模型
"""

    with open(deploy_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme)
    print("  创建: README.md")

    # 统计
    print("\n" + "=" * 60)
    print("部署包准备完成!")
    print("=" * 60)
    print(f"\n部署目录: {deploy_dir}")

    # 统计文件大小
    total_size = 0
    for f in deploy_dir.rglob("*"):
        if f.is_file():
            total_size += f.stat().st_size

    print(f"总大小: {total_size / 1024 / 1024:.1f} MB")

    # 列出主要文件
    print("\n主要文件:")
    for f in sorted(deploy_dir.rglob("*")):
        if f.is_file():
            size = f.stat().st_size
            if size > 1024 * 1024:  # 大于1MB
                print(f"  {f.relative_to(deploy_dir)} ({size / 1024 / 1024:.1f} MB)")

    print("\n下一步:")
    print("  1. 将 deployment/ 目录打包")
    print("  2. 上传到服务器")
    print("  3. 运行 start.sh 启动服务")


if __name__ == "__main__":
    create_deployment_package()
