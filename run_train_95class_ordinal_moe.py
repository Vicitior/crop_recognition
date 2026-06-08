"""
95类多作物训练 - 整合创新点（序数约束 + MoE-LoRA）
支持断点续训：如果存在 last.pth 自动从断点恢复
支持定期保存断点：每 10 分钟自动保存一次，防止意外关机丢失进度

用法：
    python run_train_95class_ordinal_moe.py
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_CACHE'] = 'D:/huggingface/hub'
os.environ['PYTHONUNBUFFERED'] = '1'  # 确保输出立即刷新

import sys
from pathlib import Path

OUTPUT_DIR = 'saved_models/clip/growth-stage-ordinal-moe'
last_checkpoint = Path(OUTPUT_DIR) / 'last.pth'

# 基础参数
argv = [
    'train_95class_ordinal_moe.py',
    '--data-dir', 'D:/crop_datasets/unified_growth_final',
    '--model', 'D:/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1',
    '--epochs', '30',
    '--batch-size', '4',       # 增加batch size
    '--lr', '5e-4',
    '--lora-rank', '8',        # 减小rank节省显存
    '--lora-alpha', '16',
    '--output-dir', OUTPUT_DIR,
    '--grad-accum', '4',       # 梯度累积，等效batch_size=8
    '--mixed-precision',       # 混合精度训练节省显存
    # 创新点参数
    '--loss-type', 'ordinal',  # 使用序数感知损失
    '--sigma', '1.0',
    '--alpha', '0.5',
    '--beta', '1.0',
    '--num-experts', '2',      # 减少专家数节省显存
    '--num-shared', '1',
    '--aux-loss-weight', '0.01',
]

# 如果存在checkpoint，自动恢复
if last_checkpoint.exists():
    argv.extend(['--resume', str(last_checkpoint)])
    print(f"[断点续训] 从 {last_checkpoint} 恢复训练")
else:
    print("[新训练] 未找到checkpoint，从头开始训练")

sys.argv = argv

from scripts.train_95class_ordinal_moe import main
main()
