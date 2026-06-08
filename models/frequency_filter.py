# -*- coding: utf-8 -*-
"""
创新点 3：双频特征对齐模块
将视觉特征分解为高频（纹理细节）和低频（语义结构）分量，
滤除背景噪声，保留作物核心特征。

用于知识蒸馏过程中的特征对齐，使轻量级学生模型
能够更好地学习教师模型的多尺度特征。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FrequencyDecomposer(nn.Module):
    """
    频域分解器：将特征图分解为高频和低频分量

    使用 DCT (离散余弦变换) 或高斯滤波实现：
    - 低频：语义结构、整体形状（作物轮廓、冠层结构）
    - 高频：纹理细节、边缘信息（叶片纹理、茎节特征）
    """

    def __init__(self, method="gaussian", cutoff_ratio=0.25):
        super().__init__()
        self.method = method
        self.cutoff_ratio = cutoff_ratio

    def _gaussian_lowpass(self, x, sigma):
        """高斯低通滤波"""
        B, C, H, W = x.shape
        # 创建高斯核
        kernel_size = min(H, W, 7)
        if kernel_size % 2 == 0:
            kernel_size -= 1

        channels = torch.arange(C, device=x.device, dtype=x.dtype)
        coords_h = torch.arange(H, device=x.device, dtype=x.dtype)
        coords_w = torch.arange(W, device=x.device, dtype=x.dtype)

        # 简单的均值滤波近似低通
        padding = kernel_size // 2
        lowpass = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)
        return lowpass

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 或 [B, N, D] 特征图
        Returns:
            low_freq: 低频分量
            high_freq: 高频分量
        """
        if x.dim() == 3:
            # [B, N, D] -> [B, D, H, W] 假设 N = H*W
            B, N, D = x.shape
            H = W = int(np.sqrt(N))
            if H * W != N:
                # 不能完美 reshape，用 1D 滤波
                lowpass = F.avg_pool1d(x.transpose(1, 2), kernel_size=3, stride=1, padding=1).transpose(1, 2)
                highpass = x - lowpass
                return lowpass, highpass
            x = x.transpose(1, 2).reshape(B, D, H, W)

        if self.method == "gaussian":
            low_freq = self._gaussian_lowpass(x, sigma=self.cutoff_ratio)
        else:
            # 简单均值
            low_freq = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        high_freq = x - low_freq

        return low_freq, high_freq


class DualFrequencyAlignment(nn.Module):
    """
    双频特征对齐模块

    在知识蒸馏中，分别对齐教师和学生模型的：
    1. 低频特征：语义结构（全局作物形态）
    2. 高频特征：纹理细节（局部叶片/茎节纹理）

    通过可学习的权重自适应调整两个频段的对齐强度。
    """

    def __init__(self, feat_dim, method="gaussian"):
        super().__init__()
        self.decomposer = FrequencyDecomposer(method=method)

        # 可学习的频段权重
        self.low_freq_weight = nn.Parameter(torch.ones(1))
        self.high_freq_weight = nn.Parameter(torch.ones(1))

        # 特征对齐投影层（当教师和学生特征维度不同时）
        self.align_proj = None
        self.feat_dim = feat_dim

    def build_projection(self, student_dim, teacher_dim):
        """构建维度对齐投影"""
        if student_dim != teacher_dim:
            self.align_proj = nn.Linear(student_dim, teacher_dim).to(
                next(self.parameters()).device
            )

    def compute_alignment_loss(self, student_feat, teacher_feat):
        """
        计算双频对齐损失

        Args:
            student_feat: 学生模型特征 [B, C_s, H, W] 或 [B, N, D]
            teacher_feat: 教师模型特征 [B, C_t, H, W] 或 [B, N, D]

        Returns:
            total_loss: 总对齐损失
            low_loss: 低频对齐损失
            high_loss: 高频对齐损失
        """
        # 维度对齐
        if self.align_proj is not None:
            if student_feat.dim() == 3:
                student_feat = self.align_proj(student_feat)
            else:
                B, C, H, W = student_feat.shape
                student_feat = self.align_proj(student_feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # 大小对齐
        if student_feat.shape != teacher_feat.shape:
            if student_feat.dim() == 4:
                student_feat = F.interpolate(student_feat, size=teacher_feat.shape[2:],
                                              mode='bilinear', align_corners=False)
            elif student_feat.dim() == 3:
                student_feat = student_feat[:, :teacher_feat.shape[1], :]

        # 频域分解
        s_low, s_high = self.decomposer(student_feat)
        t_low, t_high = self.decomposer(teacher_feat)

        # 低频对齐损失（MSE）
        low_loss = F.mse_loss(s_low, t_low.detach())

        # 高频对齐损失（MSE）
        high_loss = F.mse_loss(s_high, t_high.detach())

        # 加权总损失
        total_loss = (torch.abs(self.low_freq_weight) * low_loss +
                      torch.abs(self.high_freq_weight) * high_loss)

        return total_loss, low_loss.item(), high_loss.item()


class DualFrequencyFilter(nn.Module):
    """
    双频滤波器：在推理时滤除背景噪声

    输入特征图 → 频域分解 → 抑制噪声频段 → 重建

    用于提升轻量级模型在复杂背景下的识别精度
    """

    def __init__(self, feat_dim, method="gaussian"):
        super().__init__()
        self.decomposer = FrequencyDecomposer(method=method)

        # 可学习的频段抑制/增强权重
        self.freq_gate = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.ReLU(),
            nn.Linear(feat_dim // 4, 2),  # [low_weight, high_weight]
            nn.Softmax(dim=-1)
        )

        self.fusion_proj = nn.Linear(feat_dim, feat_dim)
        nn.init.eye_(self.fusion_proj.weight)
        nn.init.zeros_(self.fusion_proj.bias)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 特征图
        Returns:
            filtered: 滤波后的特征图 [B, C, H, W]
        """
        low_freq, high_freq = self.decomposer(x)

        # 全局平均池化获取特征向量
        gap = F.adaptive_avg_pool2d(x, 1).flatten(1)  # [B, C]

        # 计算频段权重
        weights = self.freq_gate(gap)  # [B, 2]

        # 加权融合
        weighted_low = weights[:, 0:1, None, None] * low_freq
        weighted_high = weights[:, 1:2, None, None] * high_freq
        fused = weighted_low + weighted_high

        return self.fusion_proj(fused.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) + x
