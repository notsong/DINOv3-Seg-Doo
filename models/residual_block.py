"""
models/residual_block.py

Residual Decoder Block
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()

        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.GroupNorm(8, out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    标准Residual Block

    Conv
        ↓
    GN
        ↓
    GELU
        ↓
    Conv
        ↓
    GN
        ↓
      +
      │
    Shortcut
        ↓
    GELU
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = ConvGNAct(in_channels, out_channels)

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(8, out_channels)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False
                ),
                nn.GroupNorm(8, out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.act = nn.GELU()

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)

        out = self.conv2(out)

        out = out + identity

        out = self.act(out)

        return out


class UpResidualBlock(nn.Module):
    """
    Decoder Block

    Upsample
        ↓
    ResidualBlock
    """

    def __init__(self, in_channels, out_channels, skip_channels=None):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )

        self.block = ResidualBlock(
            in_channels,
            out_channels
        )

        # Project before the existing residual block so it refines both paths.
        # None adds no state_dict keys, preserving the disabled baseline.
        self.skip_proj = (
            nn.Sequential(
                nn.Conv2d(skip_channels, in_channels, 1, bias=False),
                nn.GroupNorm(8, in_channels)
            ) if skip_channels is not None else None
        )

    def forward(self, x, skip=None, output_size=None):

        if self.skip_proj is not None:
            if skip is None:
                raise ValueError("This decoder stage requires a detail skip")
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = x + self.skip_proj(skip)
        elif output_size is not None:
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        else:
            x = self.up(x)

        x = self.block(x)

        return x
