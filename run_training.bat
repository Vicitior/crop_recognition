@echo off
chcp 65001 >nul
echo ==========================================
echo 农作物识别系统 - 通宵训练
echo ==========================================
echo.

cd /d "%~dp0"

echo [%date% %time%] Phase 1: 数据增强
python augment_dataset.py
echo.

echo [%date% %time%] Phase 2: 集成模型评估
python scripts/ensemble_predict.py --data-dir dataset --tta
echo.

echo [%date% %time%] Phase 3a: 课程学习模型
python scripts/train_experiments.py --exp curriculum --epochs 50 --lr 3e-4 --lora-rank 32 --lora-alpha 64 --output-dir saved_models/clip/clip-vit-large-patch14-336-curriculum-v2
echo.

echo [%date% %time%] Phase 3b: Focal Loss 模型
python scripts/train_clip_v2.py --model openai/clip-vit-large-patch14-336 --lora-rank 32 --lora-alpha 64 --lr 3e-4 --use-focal-loss --focal-gamma 2.0 --output-dir saved_models/clip/clip-vit-large-patch14-336-focal
echo.

echo [%date% %time%] Phase 3c: LoRA Rank 32 模型
python scripts/train_clip_v2.py --model openai/clip-vit-large-patch14-336 --lora-rank 32 --lora-alpha 64 --output-dir saved_models/clip/clip-vit-large-patch14-336-lora32
echo.

echo [%date% %time%] Phase 4: 最终集成评估
python scripts/ensemble_predict.py --data-dir dataset --tta --min-acc 70
echo.

echo ==========================================
echo 训练完成！
echo ==========================================
echo 查看结果:
echo   - 集成结果: saved_models/clip/ensemble_results.json
echo   - 各模型: saved_models/clip/ 目录
echo.
pause
