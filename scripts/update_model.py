# -*- coding: utf-8 -*-
"""
模型热更新脚本
在服务器上运行，替换当前模型，无需重启 Web 服务

用法:
    # 上传新模型后运行
    python scripts/update_model.py --model path/to/new_model.pth

    # 查看当前模型信息
    python scripts/update_model.py --info
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


def get_current_model_info():
    """获取当前模型信息"""
    candidates = [
        "saved_models/clip/hot_reload",
        "saved_models/clip/clip-vit-large-patch14-336-v2",
        "saved_models/clip/clip-large-336",
        "saved_models/clip/clip-large",
    ]
    for d in candidates:
        config_path = os.path.join(d, "config.json")
        best_path = os.path.join(d, "best.pth")
        if os.path.exists(config_path) and os.path.exists(best_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            return {
                "path": d,
                "classes": config.get("num_classes", "?"),
                "model": config.get("model", "?"),
                "timestamp": config.get("hot_reload_time", config.get("timestamp", "?")),
            }
    return None


def update_model(model_path):
    """更新模型"""
    import torch

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return False

    # 验证模型
    print(f"📦 加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    if "model_state_dict" not in checkpoint:
        print("❌ 无效的模型文件，缺少 model_state_dict")
        return False

    class_names = checkpoint.get("class_names")
    if not class_names:
        print("❌ 模型文件缺少 class_names")
        return False

    print(f"  类别数: {len(class_names)}")
    print(f"  类别: {', '.join(class_names)}")

    # 复制到 hot_reload 目录
    target_dir = Path("saved_models/clip/hot_reload")
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "best.pth"

    shutil.copy2(model_path, target_path)
    print(f"  ✅ 模型已复制到: {target_path}")

    # 保存配置
    config = {
        "model": checkpoint.get("model_name", "openai/clip-vit-large-patch14-336"),
        "lora_rank": 8,
        "num_classes": len(class_names),
        "class_names": class_names,
        "method": checkpoint.get("method", "lora"),
        "hot_reload_time": datetime.now().isoformat(),
    }
    with open(target_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"  ✅ 配置已保存")
    print(f"\n🎉 模型更新完成！Web 服务下次识别会自动加载新模型。")
    return True


def main():
    parser = argparse.ArgumentParser(description="模型热更新脚本")
    parser.add_argument("--model", type=str, help="新模型文件路径 (.pth)")
    parser.add_argument("--info", action="store_true", help="查看当前模型信息")
    args = parser.parse_args()

    if args.info:
        info = get_current_model_info()
        if info:
            print(f"当前模型: {info['path']}")
            print(f"  类别数: {info['classes']}")
            print(f"  基础模型: {info['model']}")
            print(f"  更新时间: {info['timestamp']}")
        else:
            print("未找到模型")
        return

    if args.model:
        update_model(args.model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
