"""
训练脚本 - 95类多作物+物候期+病害分类
支持断点续训：如果存在 last.pth 自动从断点恢复
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import sys
from pathlib import Path

OUTPUT_DIR = 'saved_models/clip/multi-crop-95class-v2'
last_checkpoint = Path(OUTPUT_DIR) / 'last.pth'

# 基础参数
argv = [
    'train_clip_v2.py',
    '--data-dir', 'D:/crop_datasets/unified',
    '--model', 'C:/Users/Vicitior/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1',
    '--epochs', '30',
    '--batch-size', '16',
    '--lr', '5e-4',
    '--lora-rank', '16',
    '--lora-alpha', '32',
    '--output-dir', OUTPUT_DIR
]

# 如果存在checkpoint，自动恢复
if last_checkpoint.exists():
    argv.extend(['--resume', str(last_checkpoint)])
    print(f"[断点续训] 从 {last_checkpoint} 恢复训练")
else:
    print("[新训练] 未找到checkpoint，从头开始训练")

sys.argv = argv

from scripts.train_clip_v2 import main
main()
