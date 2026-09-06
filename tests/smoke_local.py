"""Optional local-weight smoke check.

Run from repository root: python tests/smoke_local.py
Uses pretrained backbone and an untrained decoder; does not measure accuracy.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from config import Config
from models.dinov3_segmentation import DINOv3Seg
from infer import batched_sliding, single_infer
from losses.loss import TotalLoss
from unittest.mock import patch
import infer


def main():
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DINOv3Seg(Config()).eval().to(device)
    with torch.inference_mode():
        for h, w in ((64, 64), (64, 96), (67, 99)):
            x = torch.randn(1, 3, h, w, device=device)
            raw = model.encoder.backbone(pixel_values=x, output_hidden_states=True).hidden_states
            feats = model.encoder(x)
            for key, state in zip(('f4', 'f8', 'f16'), raw[-3:]):
                expected = state[:, 5:].transpose(1, 2).reshape(1, 768, h // 16, w // 16)
                torch.testing.assert_close(feats[key], expected)
            seg, boundary = model(x)
            assert seg.shape == (1, 2, h, w)
            assert boundary.shape == (1, 1, h, w)
            assert torch.isfinite(seg).all() and torch.isfinite(boundary).all()
            print(f'PASS real backbone + decoder: {h}x{w}')

    # One backward pass, no optimizer update and no dataset training.
    model.train()
    x = torch.randn(1, 3, 64, 96, device=device)
    with torch.amp.autocast(device, enabled=device == 'cuda'):
        seg, boundary = model(x)
    loss = TotalLoss(Config())(seg.float(), boundary.float(), torch.randint(0, 2, (1, 1, 64, 96), device=device))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None for p in model.encoder.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in model.detail_encoder.parameters())
    print('PASS real backbone + detail decoder: mixed-precision backward, backbone frozen')
    model.zero_grad(set_to_none=True)
    model.eval()
    model.half()
    image = np.random.default_rng(0).integers(0, 256, (40, 150, 3), dtype=np.uint8)
    with patch.object(infer.cfg, 'image_size', 64), patch.object(infer.cfg, 'crop_size', 64), patch.object(infer.cfg, 'stride', 48):
        for function in (single_infer, batched_sliding):
            prob = function(model, image, device)
            assert prob.shape == (40, 150) and np.isfinite(prob).all()
            print(f'PASS FP16 {function.__name__}: narrow image on {device}')


if __name__ == '__main__':
    main()
