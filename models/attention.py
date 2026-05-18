"""
注意力机制模块
实现SE、CBAM等注意力机制，提升特征区分能力

参考论文:
- SE-Net: arXiv:1709.01507
- CBAM: arXiv:1807.06521
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation注意力模块
    通过学习通道间的相互依赖关系来重新校准通道特征响应
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.shape
        # Squeeze
        y = self.squeeze(x.unsqueeze(-1)).view(b, c)
        # Excitation
        y = self.excitation(y).view(b, c, 1)
        # Scale
        return x.unsqueeze(-1) * y


class ChannelAttention(nn.Module):
    """
    通道注意力模块
    通过全局平均池化和最大池化捕获通道间的关系
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c = x.shape

        # 平均池化路径
        avg_out = self.fc(self.avg_pool(x.unsqueeze(-1)).view(b, c))

        # 最大池化路径
        max_out = self.fc(self.max_pool(x.unsqueeze(-1)).view(b, c))

        # 合并
        out = avg_out + max_out
        attention = self.sigmoid(out).view(b, c, 1)

        return x.unsqueeze(-1) * attention


class SpatialAttention(nn.Module):
    """
    空间注意力模块
    通过通道间的关系生成空间注意力图
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch, channels, height, width]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(combined))
        return x * attention


class CBAM(nn.Module):
    """
    CBAM: Convolutional Block Attention Module
    结合通道注意力和空间注意力
    """

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # 通道注意力
        x = self.channel_attention(x)
        # 空间注意力（需要4D输入）
        if x.dim() == 3:
            x = x.unsqueeze(-1)
        x = self.spatial_attention(x)
        return x.squeeze(-1)


class LightSE(nn.Module):
    """
    轻量级SE模块
    适用于1D特征向量
    """

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [batch, channels]
        attention = self.fc(x)
        return x * attention


class EnhancedClassifier(nn.Module):
    """
    增强分类器
    集成注意力机制
    """

    def __init__(self, feat_dim, num_classes, attention_type="se", reduction=16):
        super().__init__()

        # 第一层
        self.fc1 = nn.Linear(feat_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)

        # 注意力模块
        if attention_type == "se":
            self.attention1 = LightSE(1024, reduction)
        elif attention_type == "cbam":
            self.attention1 = CBAM(1024, reduction)
        else:
            self.attention1 = nn.Identity()

        # 第二层
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        # 注意力模块
        if attention_type == "se":
            self.attention2 = LightSE(512, reduction)
        elif attention_type == "cbam":
            self.attention2 = CBAM(512, reduction)
        else:
            self.attention2 = nn.Identity()

        # 输出层
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        # 第一层
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        # 注意力
        x = self.attention1(x)

        # 第二层
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        # 注意力
        x = self.attention2(x)

        # 输出
        x = self.fc3(x)
        return x


class MultiHeadAttentionClassifier(nn.Module):
    """
    多头注意力分类器
    使用多头自注意力增强特征交互
    """

    def __init__(self, feat_dim, num_classes, num_heads=4):
        super().__init__()

        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads

        # 多头注意力
        self.q_linear = nn.Linear(feat_dim, feat_dim)
        self.k_linear = nn.Linear(feat_dim, feat_dim)
        self.v_linear = nn.Linear(feat_dim, feat_dim)
        self.out_linear = nn.Linear(feat_dim, feat_dim)

        # LayerNorm
        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim * 4, feat_dim),
            nn.Dropout(0.1)
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        b, d = x.shape

        # 自注意力
        q = self.q_linear(x).view(b, self.num_heads, self.head_dim)
        k = self.k_linear(x).view(b, self.num_heads, self.head_dim)
        v = self.v_linear(x).view(b, self.num_heads, self.head_dim)

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(scores, dim=-1)

        # 应用注意力
        out = torch.matmul(attention, v).view(b, d)
        out = self.out_linear(out)

        # 残差连接
        x = self.norm1(x + out)

        # FFN
        out = self.ffn(x)
        x = self.norm2(x + out)

        # 分类
        return self.classifier(x)
