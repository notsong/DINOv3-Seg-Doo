"""Shallow local-detail pyramid, independent of the frozen DINOv3 backbone."""
import torch.nn as nn

from models.residual_block import ConvGNAct


class DetailEncoder(nn.Module):
    """Return 1/2, 1/4, 1/8 grids (rounded up for odd input sizes)."""

    def __init__(self, in_channels=3):
        super().__init__()
        self.stage2 = nn.Sequential(ConvGNAct(in_channels, 16, stride=2), ConvGNAct(16, 16))
        self.stage4 = nn.Sequential(ConvGNAct(16, 32, stride=2), ConvGNAct(32, 32))
        self.stage8 = nn.Sequential(ConvGNAct(32, 64, stride=2), ConvGNAct(64, 64))

    def forward(self, x):
        d2 = self.stage2(x)
        d4 = self.stage4(d2)
        d8 = self.stage8(d4)
        return {'d2': d2, 'd4': d4, 'd8': d8}
