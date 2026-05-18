# 精度提升方案指南

本文档介绍了5种提升农作物生长阶段识别精度的最新方法。

## 当前基准

| 指标 | 值 |
|------|-----|
| 最佳单模型精度 | 92.75% |
| 4模型集成精度 | 95.65% |
| 最难类别 | cotton_boll_setting (89.66%), cotton_flowering (93.33%) |

## 方案概览

| 方案 | 预期提升 | 实现状态 | 脚本 |
|------|---------|---------|------|
| 1. 测试时增强 (TTA) | +1-2% | ✅ 完成 | `predict_tta.py`, `evaluate_tta.py` |
| 2. 类别级数据增强 | +1-3% | ✅ 完成 | `train_with_class_augmentation.py` |
| 3. 提示学习 (CoOp/MaPLe) | +2-4% | ✅ 完成 | `train_prompt_learning.py` |
| 4. MMD-LoRA多模态融合 | +2-5% | ✅ 完成 | `train_mmd_lora.py` |
| 5. 注意力机制增强 | +1-2% | ✅ 完成 | `train_with_attention.py` |

## 使用方法

### 1. 测试时增强 (TTA)

无需重新训练，直接提升预测精度：

```bash
# 评估TTA效果
python scripts/evaluate_tta.py --model-path saved_models/clip/clip-vit-large-patch14-336-v2/best.pth --tta-level medium

# 使用TTA预测
python scripts/predict_tta.py --image test.jpg --tta-level medium --strategy all
```

**TTA级别**:
- `basic`: 原图 + 水平翻转 (2x)
- `medium`: 多尺度 (0.9x, 1.0x, 1.1x) + 翻转 (6x)
- `strong`: 多尺度 + 旋转 + 色彩抖动 (大量变换)

**聚合策略**:
- `avg`: 平均概率（推荐）
- `max`: 最大概率
- `geometric_mean`: 几何平均
- `median`: 中位数（更鲁棒）

### 2. 类别级数据增强

针对难分类别进行增强：

```bash
# 启用类别特定增强
python scripts/train_with_class_augmentation.py \
    --model openai/clip-vit-large-patch14-336 \
    --class-augmentation \
    --aug-level strong \
    --epochs 50

# 使用遗传算法优化增强参数
python scripts/optimize_augmentation.py --generations 10 --population 20
```

**针对难分类别的增强策略**:
- `cotton_boll_setting`: Mixup + Cutout + 色彩抖动 + 旋转
- `cotton_flowering`: Mixup + 色彩抖动

### 3. 提示学习 (CoOp/MaPLe)

可学习的上下文提示向量：

```bash
# 简化版提示学习（推荐，稳定）
python scripts/train_prompt_learning.py \
    --method simple \
    --model openai/clip-vit-large-patch14-336 \
    --n-ctx 16 \
    --epochs 50

# CoOp方法
python scripts/train_prompt_learning.py \
    --method coop \
    --model openai/clip-vit-large-patch14-336 \
    --n-ctx 16

# MaPLe方法（多模态提示）
python scripts/train_prompt_learning.py \
    --method maple \
    --model openai/clip-vit-large-patch14-336 \
    --n-ctx 16
```

**参数说明**:
- `n_ctx`: 上下文提示长度，推荐16-32
- `method`: 
  - `simple`: 简化版，直接优化类别原型（最稳定）
  - `coop`: CoOp方法，学习文本提示
  - `maple`: MaPLe方法，多模态提示

### 4. MMD-LoRA多模态融合

结合对比学习和领域对齐：

```bash
python scripts/train_mmd_lora.py \
    --model openai/clip-vit-large-patch14-336 \
    --lora-rank 16 \
    --contrastive-weight 0.1 \
    --alignment-weight 0.05 \
    --epochs 50
```

**关键参数**:
- `contrastive-weight`: 对比损失权重（默认0.1）
- `alignment-weight`: 领域对齐损失权重（默认0.05）
- `temperature`: 对比损失温度参数（默认0.07）

### 5. 注意力机制增强

集成SE/CBAM注意力模块：

```bash
# SE注意力
python scripts/train_with_attention.py \
    --model openai/clip-vit-large-patch14-336 \
    --attention se \
    --epochs 50

# CBAM注意力
python scripts/train_with_attention.py \
    --model openai/clip-vit-large-patch14-336 \
    --attention cbam \
    --epochs 50

# 多头注意力
python scripts/train_with_attention.py \
    --model openai/clip-vit-large-patch14-336 \
    --attention multihead \
    --epochs 50
```

**注意力类型**:
- `se`: Squeeze-and-Excitation（通道注意力）
- `cbam`: Convolutional Block Attention Module（通道+空间注意力）
- `multihead`: 多头自注意力
- `none`: 不使用注意力

## 综合评估

评估所有方法的效果：

```bash
# 评估所有模型
python scripts/evaluate_all_methods.py --compare-tta

# 结果将保存到 saved_models/clip/evaluation_summary.json
```

## 推荐实施顺序

1. **立即测试**: TTA - 无需重新训练
2. **短期**: 类别级数据增强 - 针对难分类别
3. **中期**: 提示学习 - 最新论文验证有效
4. **长期**: MMD-LoRA - 多模态融合

## 集成策略

将多个方法的结果进行集成：

```bash
# 集成多个模型
python scripts/ensemble_predict.py \
    --models saved_models/clip/clip-large-336-v2/best.pth \
             saved_models/clip/clip-large-336-simple/best.pth \
             saved_models/clip/clip-large-336-attention-se/best.pth \
    --strategy average
```

## 预期效果

| 方法组合 | 预期精度 |
|---------|---------|
| 基准 + TTA | 96-97% |
| 基准 + 类别增强 | 96-98% |
| 基准 + 提示学习 | 97-98% |
| 基准 + MMD-LoRA | 97-99% |
| 全部方法集成 | 98%+ |

## 参考论文

1. **CoOp**: arXiv:2109.01134 - Learning to Prompt for Vision-Language Models
2. **MaPLe**: arXiv:2210.03117 - Multi-modal Prompt Learning
3. **MMD-LoRA**: arXiv:2412.20162 - Multi-Modality Driven LoRA
4. **CasPL**: arXiv:2409.17805 - Cascade Prompt Learning
5. **SE-Net**: arXiv:1709.01507 - Squeeze-and-Excitation Networks
6. **CBAM**: arXiv:1807.06521 - Convolutional Block Attention Module

## 故障排除

### 内存不足
- 减小batch_size: `--batch-size 8`
- 使用更小的LoRA rank: `--lora-rank 8`

### 训练不稳定
- 降低学习率: `--lr 1e-4`
- 增加warmup: `--warmup-epochs 10`
- 使用简化版提示学习: `--method simple`

### 精度未提升
- 检查数据集质量
- 尝试不同的增强级别
- 调整损失权重参数
