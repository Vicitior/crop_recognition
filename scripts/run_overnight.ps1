# ============================================================
# 农作物识别系统 - 通宵训练脚本
# 运行所有改进方案，明早查看结果
# ============================================================

$ErrorActionPreference = "Continue"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = "logs"
$logFile = "$logDir/overnight_$timestamp.log"

# 创建日志目录
if (!(Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

function Log($msg) {
    $time = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$time | $msg" | Tee-Object -FilePath $logFile -Append
}

Log "=========================================="
Log "农作物识别系统 - 通宵训练开始"
Log "=========================================="

# ============================================================
# Phase 1: 数据增强 (5-10分钟)
# ============================================================
Log ""
Log "=== Phase 1: 数据增强 ==="
Log "为玉米和小麦类别生成更多训练图片"

try {
    python augment_dataset.py 2>&1 | Tee-Object -FilePath $logFile -Append
    Log "Phase 1 完成"
} catch {
    Log "Phase 1 出错: $_"
}

# ============================================================
# Phase 2: 运行集成评估 (10-15分钟)
# ============================================================
Log ""
Log "=== Phase 2: 集成模型评估 ==="
Log "评估所有现有模型的集成效果"

try {
    python scripts/ensemble_predict.py --data-dir dataset --tta 2>&1 | Tee-Object -FilePath $logFile -Append
    Log "Phase 2 完成"
} catch {
    Log "Phase 2 出错: $_"
}

# ============================================================
# Phase 3: 训练新模型 (每个30-60分钟)
# ============================================================

# Phase 3a: 课程学习模型
Log ""
Log "=== Phase 3a: 课程学习模型 ==="
Log "使用优化后的课程调度训练"

try {
    python scripts/train_experiments.py --exp curriculum --epochs 50 --lr 3e-4 --lora-rank 32 --lora-alpha 64 --output-dir saved_models/clip/clip-vit-large-patch14-336-curriculum-v2 2>&1 | Tee-Object -FilePath $logFile -Append
    Log "Phase 3a 完成"
} catch {
    Log "Phase 3a 出错: $_"
}

# Phase 3b: Focal Loss 模型
Log ""
Log "=== Phase 3b: Focal Loss 模型 ==="
Log "使用Focal Loss处理类别不平衡"

try {
    python scripts/train_clip_v2.py --model openai/clip-vit-large-patch14-336 --lora-rank 32 --lora-alpha 64 --lr 3e-4 --use-focal-loss --focal-gamma 2.0 --output-dir saved_models/clip/clip-vit-large-patch14-336-focal 2>&1 | Tee-Object -FilePath $logFile -Append
    Log "Phase 3b 完成"
} catch {
    Log "Phase 3b 出错: $_"
}

# Phase 3c: 更高 LoRA Rank 模型
Log ""
Log "=== Phase 3c: LoRA Rank 32 模型 ==="
Log "使用更大的LoRA容量"

try {
    python scripts/train_clip_v2.py --model openai/clip-vit-large-patch14-336 --lora-rank 32 --lora-alpha 64 --output-dir saved_models/clip/clip-vit-large-patch14-336-lora32 2>&1 | Tee-Object -FilePath $logFile -Append
    Log "Phase 3c 完成"
} catch {
    Log "Phase 3c 出错: $_"
}

# Phase 3d: 使用增强数据训练
Log ""
Log "=== Phase 3d: 增强数据模型 ==="
Log "使用数据增强后的数据集训练"

if (Test-Path "dataset/train_augmented") {
    try {
        python scripts/train_clip_v2.py --model openai/clip-vit-large-patch14-336 --lora-rank 32 --lora-alpha 64 --data-dir dataset --output-dir saved_models/clip/clip-vit-large-patch14-336-augmented 2>&1 | Tee-Object -FilePath $logFile -Append
        Log "Phase 3d 完成"
    } catch {
        Log "Phase 3d 出错: $_"
    }
} else {
    Log "Phase 3d 跳过: 增强数据目录不存在"
}

# ============================================================
# Phase 4: 最终集成评估 (10-15分钟)
# ============================================================
Log ""
Log "=== Phase 4: 最终集成评估 ==="
Log "使用所有模型进行集成"

try {
    python scripts/ensemble_predict.py --data-dir dataset --tta --min-acc 70 2>&1 | Tee-Object -FilePath $logFile -Append
    Log "Phase 4 完成"
} catch {
    Log "Phase 4 出错: $_"
}

# ============================================================
# Phase 5: 生成汇总报告
# ============================================================
Log ""
Log "=========================================="
Log "训练完成！生成汇总报告..."
Log "=========================================="

# 读取集成结果
if (Test-Path "saved_models/clip/ensemble_results.json") {
    Log ""
    Log "集成结果:"
    Get-Content "saved_models/clip/ensemble_results.json" | Tee-Object -FilePath $logFile -Append
}

# 列出所有新训练的模型
Log ""
Log "新训练的模型目录:"
$modelDirs = @(
    "saved_models/clip/clip-vit-large-patch14-336-curriculum-v2",
    "saved_models/clip/clip-vit-large-patch14-336-focal",
    "saved_models/clip/clip-vit-large-patch14-336-lora32",
    "saved_models/clip/clip-vit-large-patch14-336-augmented"
)

foreach ($dir in $modelDirs) {
    if (Test-Path "$dir/config.json") {
        $config = Get-Content "$dir/config.json" | ConvertFrom-Json
        $testAcc = $config.test_acc
        $valAcc = $config.best_val_acc
        Log "  $dir"
        Log "    Val: $valAcc%, Test: $testAcc%"
    }
}

Log ""
Log "=========================================="
Log "通宵训练完成！"
Log "日志文件: $logFile"
Log "=========================================="

# 打开日志文件
# notepad $logFile
