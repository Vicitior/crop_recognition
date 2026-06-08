# -*- coding: utf-8 -*-
"""
Ordinal Constraint Loss for Crop Growth Stage Recognition
创新点 1：引入作物生长时序的"序数约束"

核心思想：
  标准 CrossEntropy 把生长阶段当作独立的 Categorical 标签。
  但作物生长具有严格的单向时序性：苗期→蕾期→开花期→结铃期→吐絮期。
  把"开花期"错判为"结铃期"（邻近阶段）应该比错判为"苗期"（跨度大）惩罚更小。

本模块提供三种序数损失：
  1. OrdinalGaussianLoss — 高斯软标签 + KL 散度
  2. EarthMoversDistanceLoss — 一维 Wasserstein 距离
  3. CombinedOrdinalLoss — 两者的加权组合（推荐）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def build_gaussian_soft_labels(labels, num_classes, sigma=1.0):
    """
    为每个标签生成高斯软标签分布。

    例如：5 个阶段，标签=2（开花期），sigma=1.0 时：
      [0.018, 0.213, 0.538, 0.213, 0.018]  （归一化后）

    Args:
        labels: (B,) 整数标签
        num_classes: 类别数（同一作物内的阶段数，如棉花=5）
        sigma: 高斯带宽，越大越平滑。推荐范围 0.8 ~ 1.5

    Returns:
        soft_labels: (B, num_classes) 概率分布
    """
    device = labels.device
    # 构造类别索引 [0, 1, 2, ..., C-1]
    class_indices = torch.arange(num_classes, dtype=torch.float32, device=device)
    # (B, 1) - (1, C) -> (B, C) 的距离矩阵
    dist = (labels.float().unsqueeze(1) - class_indices.unsqueeze(0)).abs()
    # 高斯核
    soft = torch.exp(-0.5 * (dist / sigma) ** 2)
    # 归一化为概率分布
    soft = soft / soft.sum(dim=1, keepdim=True)
    return soft


class OrdinalGaussianLoss(nn.Module):
    """
    高斯软标签序数损失：KL(soft_labels || pred_probs) + alpha * CE(hard_labels, pred)

    - KL 项让模型学习"邻近阶段概率相近"的序数结构
    - CE 项保持判别能力，防止模型坍缩为全部输出均等概率
    """

    def __init__(self, num_classes_per_crop, sigma=1.0, alpha=0.5):
        super().__init__()
        self.num_classes = num_classes_per_crop
        self.sigma = sigma
        self.alpha = alpha

    def forward(self, logits, labels):
        """
        Args:
            logits: (B, C) 模型原始输出
            labels: (B,) 整数标签（同一作物内的阶段索引 0~4）
        """
        soft_labels = build_gaussian_soft_labels(labels, self.num_classes, self.sigma)
        log_probs = F.log_softmax(logits, dim=-1)
        # KL 散度
        kl_loss = F.kl_div(log_probs, soft_labels, reduction='batchmean')
        # 标准 CE
        ce_loss = F.cross_entropy(logits, labels)
        return (1 - self.alpha) * kl_loss + self.alpha * ce_loss


class EarthMoversDistanceLoss(nn.Module):
    """
    一维 Wasserstein (Earth Mover's Distance) 损失。

    将预测分布和目标分布都视为一维离散分布，计算累积分布之差的 L1 范数。
    天然满足序数约束：把类别 i 预测为 i+2 的惩罚是预测为 i+1 的两倍。
    """

    def __init__(self, num_classes_per_crop):
        super().__init__()
        self.num_classes = num_classes_per_crop

    def forward(self, logits, labels):
        soft_labels = build_gaussian_soft_labels(labels, self.num_classes, sigma=0.8)
        pred_probs = F.softmax(logits, dim=-1)
        # 累积分布
        cdf_pred = torch.cumsum(pred_probs, dim=-1)
        cdf_target = torch.cumsum(soft_labels, dim=-1)
        # EMD = 累积分布差的 L1 范数
        emd = (cdf_pred - cdf_target).abs().sum(dim=-1).mean()
        return emd


class CombinedOrdinalLoss(nn.Module):
    """
    组合序数损失 = KL(高斯软标签) + beta * EMD

    推荐用于论文实验：
    - KL 项提供平滑的序数梯度信号
    - EMD 项直接优化"预测分布的累积距离"
    - 两者互补，比单独使用任一项效果更好
    """

    def __init__(self, num_classes_per_crop, sigma=1.0, alpha=0.5, beta=1.0):
        super().__init__()
        self.ordinal_ce = OrdinalGaussianLoss(num_classes_per_crop, sigma, alpha)
        self.emd = EarthMoversDistanceLoss(num_classes_per_crop)
        self.beta = beta

    def forward(self, logits, labels):
        return self.ordinal_ce(logits, labels) + self.beta * self.emd(logits, labels)
