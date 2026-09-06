"""
models/dinov3_encoder.py

DINOv3 Multi-layer Feature Extractor
输出同分辨率的多层 feature maps（取最后3层hidden states）
"""
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel


class DINOv3Encoder(nn.Module):
    def __init__(self, model_name: str, trainable=False):
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
        """
        Returns:
            feats = {"f4": low-level, "f8": mid-level, "f16": high-level}
            f4/f8/f16 为兼容旧接口保留的层名，并非下采样倍率。
            三路网格均为输入高宽整除 backbone.config.patch_size（ViT-B/16 为16）。
        """
        outputs = self.backbone(
            pixel_values=x,
            output_hidden_states=True
        )

        hs = outputs.hidden_states

        # 取最后3层：low, mid, high
        f_low = hs[-3]
        f_mid = hs[-2]
        f_high = hs[-1]

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

        return {
            "f4": reshape(f_low),
            "f8": reshape(f_mid),
            "f16": reshape(f_high),
        }
