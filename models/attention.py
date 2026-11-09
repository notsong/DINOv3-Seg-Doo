"""
models/attention.py

Attention Modules
------------------------------------
包含：

1. SEBlock
2. SpatialAttention
3. CBAM
4. AttentionFusion

用于 DINOv3 Decoder V3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================
# SE Block
# ==========================================================

class SEBlock(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()

        hidden = max(channels // reduction, 8)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):

        w = self.pool(x)
        w = self.fc(w)

        return x * w


# ==========================================================
# Spatial Attention
# ==========================================================

class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size,
            padding=padding,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg = torch.mean(x, dim=1, keepdim=True)

        mx, _ = torch.max(x, dim=1, keepdim=True)

        w = torch.cat([avg, mx], dim=1)

        w = self.conv(w)

        w = self.sigmoid(w)

        return x * w


# ==========================================================
# CBAM
# ==========================================================

class CBAM(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.channel = SEBlock(channels)

        self.spatial = SpatialAttention()

    def forward(self, x):

        x = self.channel(x)

        x = self.spatial(x)

        return x


# ==========================================================
# Learnable Weighted Fusion
# ==========================================================

class AttentionFusion(nn.Module):
    """逐层应用独立的 CBAM，再学习同分辨率特征的融合权重。"""

    def __init__(self, channels, feature_layers):

        super().__init__()

        self.cbam = nn.ModuleDict({str(layer): CBAM(channels) for layer in feature_layers})
        self.weight = nn.Parameter(torch.ones(len(feature_layers)))

        self.fuse = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1,
                bias=False
            ),

            nn.GroupNorm(8, channels),

            nn.GELU(),

            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1,
                bias=False
            ),

            nn.GroupNorm(8, channels),

            nn.GELU()
        )

    def forward(self, feats):
        weights = F.softmax(self.weight, dim=0)
        fused = sum(
            weights[index] * cbam(feats[int(layer)])
            for index, (layer, cbam) in enumerate(self.cbam.items())
        )
        return self.fuse(fused)
