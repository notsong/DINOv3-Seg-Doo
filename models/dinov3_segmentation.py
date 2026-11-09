"""
models/dinov3_segmentation.py

V3版本：
DINOv3 Encoder + DecoderV3
输出：
    seg logits
    boundary logits
"""

import torch
import torch.nn as nn

from models.dinov3_encoder import DINOv3Encoder
from models.decoder_v3 import DecoderV3
from models.detail_encoder import DetailEncoder


class DINOv3Seg(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.encoder = DINOv3Encoder(
            cfg.backbone_name,
            trainable=not cfg.freeze_backbone,
            feature_layers=cfg.feature_layers,
        )

        self.decoder = DecoderV3(
            num_classes=cfg.num_classes,
            feat_dim=768,
            use_detail_branch=cfg.use_detail_branch,
            feature_layers=cfg.feature_layers,
        )
        self.detail_encoder = DetailEncoder(cfg.in_channels) if cfg.use_detail_branch else None

    def forward(self, x):

        feats = self.encoder(x)
        details = self.detail_encoder(x) if self.detail_encoder is not None else None

        seg, boundary = self.decoder(
            feats,
            output_size=x.shape[2:],
            details=details
        )

        return seg, boundary
