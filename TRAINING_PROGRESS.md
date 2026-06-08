# 农作物生长阶段识别 - 训练进展

> 最后更新: 2026-05-27

## 项目概述

基于 CLIP 视觉-语言模型的农作物（玉米/小麦/棉花）生长阶段识别，15个类别。

- 数据集：`dataset/`（train/val/test）
- 训练脚本：`scripts/`
- 模型保存：`saved_models/clip/`
- GPU: RTX 5060 Laptop 8.5GB（不能同时跑两个大模型）

## 实验结果汇总

| # | 方法 | 测试准确率 | 状态 | 备注 |
|---|------|-----------|------|------|
| 1 | 基线 LoRA (ViT-L/14@336, rank=16) | **84.73%** | ✅ 完成 | 基准线 |
| 2 | KGPT 知识引导Prompt微调 | 72.91% | ✅ 完成 | 低于基线 |
| 3 | MoE-LoRA + 序数损失 | ~24% | ❌ 放弃 | 效果太差 |
| 4 | 棉花5类集成（4模型） | 95.65% | ✅ 完成 | 仅5类子集 |
| D | 简化多模态融合 | **100.00%** | ✅ 完成 | Oracle（使用了真实crop_id/ordinal）|
| A | 层级分类 | — | ⏸️ 暂停 | 第1个epoch未跑完 |
| B | 时间对比学习 | — | 待训练 | |
| C | 课程学习 | — | 待训练 | |
| E | 双频蒸馏 | — | 待训练 | |

## 关键发现

1. **多模态融合**：使用真实crop_id和ordinal作为额外输入，100%准确率。说明如果能先估算作物类型，阶段分类就变得很简单。
2. **层级分类**：两阶段方法（先作物后阶段）是可行的方向，但需要更多时间验证。
3. **MoE-LoRA失败原因**：数据集太小（~1400张），路由网络无法学习有意义的专家分配。
4. **KGPT失败原因**：物理向量与CLIP文本embedding空间不匹配。

## 论文方向建议

**推荐方案：两阶段层级分类**
- 第一阶段：CLIP + LoRA → 作物类型分类（玉米/小麦/棉花）
- 第二阶段：CLIP + LoRA + 作物嵌入 → 生长阶段分类
- 结合序数损失（高斯软标签 + EMD）
- 可以引用多模态融合的oracle结果作为上界

## 已修复的Bug

- `knowledge_encoder.py:288`: 修复未定义变量 `img_feat`
- `frequency_filter.py:175`: 修复中文变量名 `self融合` → `self.fusion_proj`
- `train_unified.py:226`: 修复 tokenizer 调用方式
- `train_ordinal_moe.py:265`: 修复 MoE 辅助损失无条件调用
- `train_experiments.py`: 修复特征输出类型检查

## 待完成实验

```bash
# 实验A: 层级分类（已暂停，需要继续）
python scripts/train_experiments.py --exp hierarchical --epochs 50 --batch-size 4 --lr 5e-4

# 实验B: 时间对比学习
python scripts/train_experiments.py --exp temporal_contrast --epochs 50 --batch-size 4 --lr 5e-4 --contrast-weight 0.1

# 实验C: 课程学习
python scripts/train_experiments.py --exp curriculum --epochs 50 --batch-size 4 --lr 5e-4

# 集成评估
python scripts/ensemble_predict.py --data-dir dataset --tta --batch-size 8
```

## 数据集统计

| 类别 | 训练集数量 |
|------|-----------|
| corn_filling | 50 |
| corn_jointing | 50 |
| corn_maturity | 50 |
| corn_seedling | 43 |
| corn_tasseling | 40 |
| cotton_boll_opening | 194 |
| cotton_boll_setting | 195 |
| cotton_flowering | 198 |
| cotton_seedling | 149 |
| cotton_squaring | 188 |
| wheat_heading | 50 |
| wheat_jointing | 50 |
| wheat_maturity | 67 |
| wheat_seedling | 50 |
| wheat_tillering | 50 |

## 关键文件

| 文件 | 说明 |
|------|------|
| `scripts/train_experiments.py` | **综合实验脚本**（支持所有创新点）|
| `scripts/train_clip_v2.py` | 标准LoRA训练脚本 |
| `scripts/ensemble_predict.py` | 集成评估脚本 |
| `saved_models/clip/clip-vit-large-patch14-336-multimodal/` | 多模态融合模型（100% oracle）|
