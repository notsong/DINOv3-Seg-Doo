"""
models/decoder_v3.py

DINOv3 + Attention Fusion + Residual Decoder + Boundary Head

核心结构：
1. Four-layer Attention Fusion
2. Residual Decoder (optional CNN detail skips before residual refinement)
3. Boundary Head (aux supervision)
4. Segmentation Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DEFAULT_FEATURE_LAYERS
from models.attention import AttentionFusion
from models.residual_block import ResidualBlock, UpResidualBlock
from models.boundary_head import BoundaryHead


class DecoderV3(nn.Module):

    def __init__(self, num_classes, feat_dim=768, use_detail_branch=False,
                 feature_layers=DEFAULT_FEATURE_LAYERS):

        super().__init__()
        self.use_detail_branch = use_detail_branch

        # =========================
        # 1. Fusion
        # =========================
        self.fusion = AttentionFusion(feat_dim, feature_layers)

        # =========================
        # 2. Channel Projection
        # =========================
        self.proj = nn.Sequential(
            nn.Conv2d(feat_dim, 512, 1, bias=False),
            nn.GroupNorm(8, 512),
            nn.GELU()
        )

        # =========================
        # 3. Residual Decoder
        # =========================
        self.dec1 = UpResidualBlock(512, 256, skip_channels=64 if use_detail_branch else None)
        self.dec2 = UpResidualBlock(256, 128, skip_channels=32 if use_detail_branch else None)
        self.dec3 = UpResidualBlock(128, 64, skip_channels=16 if use_detail_branch else None)
        self.dec4 = UpResidualBlock(64, 32)

        # =========================
        # 4. Segmentation Head
        # =========================
        self.seg_head = nn.Conv2d(32, num_classes, 1)

        # =========================
        # 5. Boundary Head
        # =========================
        self.boundary_head = BoundaryHead(32)

    def forward(self, feats, output_size=None, details=None):

        """按层号读取骨干特征，并融合可选的局部细节特征。"""
        if self.use_detail_branch and details is None:
            raise ValueError("Detail-enabled decoder requires detail features")

        # =========================
        # Fusion
        # =========================
        x = self.fusion(feats)

        # =========================
        # Projection
        # =========================
        x = self.proj(x)

        # =========================
        # Decoder
        # =========================
        if self.use_detail_branch:
            x = self.dec1(x, skip=details['d8'])
            x = self.dec2(x, skip=details['d4'])
            x = self.dec3(x, skip=details['d2'])
            x = self.dec4(x, output_size=output_size)
        else:
            x = self.dec1(x)
            x = self.dec2(x)
            x = self.dec3(x)
            x = self.dec4(x)

        # =========================
        # Heads
        # =========================
        seg = self.seg_head(x)
        boundary = self.boundary_head(x)

        if output_size is not None:
            seg = F.interpolate(seg, size=output_size, mode="bilinear", align_corners=False)
            boundary = F.interpolate(boundary, size=output_size, mode="bilinear", align_corners=False)

        return seg, boundary
