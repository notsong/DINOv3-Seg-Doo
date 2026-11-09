"""
models/dinov3_encoder.py

DINOv3 Multi-layer Feature Extractor
从指定的四个 Transformer 层提取同分辨率的 patch 特征。
"""
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

from config import DEFAULT_FEATURE_LAYERS


class DINOv3Encoder(nn.Module):
    def __init__(self, model_name: str, trainable=False, feature_layers=DEFAULT_FEATURE_LAYERS):
        super().__init__()

        self.processor = AutoImageProcessor.from_pretrained(
            model_name, local_files_only=True
        )

        self.backbone = AutoModel.from_pretrained(
            model_name,
            dtype=torch.float32,
            trust_remote_code=True,
            local_files_only=True
        )

        self.feature_layers = tuple(feature_layers)
        self.num_layers = self.backbone.config.num_hidden_layers
        if (len(self.feature_layers) != 4
                or any(type(layer) is not int or not 1 <= layer <= self.num_layers
                       for layer in self.feature_layers)
                or tuple(sorted(set(self.feature_layers))) != self.feature_layers):
            raise ValueError(f"feature_layers must contain four increasing layer numbers in 1..{self.num_layers}")

        self.trainable = trainable
        if not trainable:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def set_trainable(self, trainable: bool):
        """动态切换backbone是否可训练"""
        self.trainable = trainable
        for p in self.backbone.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        """按 Transformer 层号返回同分辨率的特征图。"""
        outputs = self.backbone(
            pixel_values=x,
            output_hidden_states=True
        )

        # 只保留 Transformer 层输出，跳过可能存在的 embedding 状态。
        hs = outputs.hidden_states[-self.num_layers:]
        if len(hs) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} Transformer states, got {len(hs)}")

        patch_size = self.backbone.config.patch_size
        H, W = x.shape[-2] // patch_size, x.shape[-1] // patch_size
        prefix = 1 + self.backbone.config.num_register_tokens

        def reshape(feat):
            B, _, C = feat.shape
            # HuggingFace DINOv3: [CLS] + register tokens + patch tokens
            feat = feat[:, prefix:, :]
            if feat.shape[1] != H * W:
                raise ValueError(
                    f"Expected {H * W} patch tokens for grid {H}x{W}, got {feat.shape[1]}"
                )
            feat = feat.reshape(B, H, W, C).permute(0, 3, 1, 2)
            return feat.contiguous()

        return {layer: reshape(hs[layer - 1]) for layer in self.feature_layers}
