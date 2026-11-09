import unittest
from unittest.mock import patch

import torch
from torch import nn

from config import Config, DEFAULT_FEATURE_LAYERS
from models.attention import AttentionFusion
from models.decoder_v3 import DecoderV3
from models.dinov3_segmentation import DINOv3Seg
from models.residual_block import UpResidualBlock


class FakeEncoder(nn.Module):
    """No weight files: trainability mirrors the production constructor."""
    def __init__(self, model_name, trainable=False, feature_layers=DEFAULT_FEATURE_LAYERS):
        super().__init__()
        self.feature_layers = feature_layers
        self.proj = nn.Conv2d(3, 768, 16, stride=16)
        self.proj.requires_grad_(trainable)

    def forward(self, x):
        f = self.proj(x)
        return {layer: f for layer in self.feature_layers}


def make_model(enabled):
    cfg = Config()
    cfg.use_detail_branch = enabled
    with patch('models.dinov3_segmentation.DINOv3Encoder', FakeEncoder):
        return DINOv3Seg(cfg)


class DetailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_default_enables_detail_branch(self):
        self.assertTrue(getattr(Config(), 'use_detail_branch', False))

    def test_detail_grids_and_channels(self):
        model = make_model(True)
        self.assertIsNotNone(getattr(model, 'detail_encoder', None))
        result = model.detail_encoder(torch.randn(2, 3, 35, 51))
        self.assertEqual({k: tuple(v.shape) for k, v in result.items()}, {
            'd2': (2, 16, 18, 26), 'd4': (2, 32, 9, 13), 'd8': (2, 64, 5, 7)})

    def test_skip_changes_output_and_receives_gradient(self):
        block = UpResidualBlock(32, 16, skip_channels=8)
        x = torch.randn(1, 32, 2, 3, requires_grad=True)
        skip = torch.randn(1, 8, 5, 7, requires_grad=True)
        out = block(x, skip=skip)
        self.assertEqual(tuple(out.shape), (1, 16, 5, 7))
        self.assertFalse(torch.allclose(out, block(x, skip=torch.zeros_like(skip))))
        out.square().mean().backward()
        self.assertGreater(skip.grad.abs().sum().item(), 0)
        self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_enabled_decoder_rejects_missing_details(self):
        model = make_model(True)
        feats = {layer: torch.randn(1, 768, 2, 3) for layer in DEFAULT_FEATURE_LAYERS}
        with self.assertRaisesRegex(ValueError, 'detail'):
            model.decoder(feats, output_size=(32, 48))

    def test_enabled_model_odd_shape_and_segmentation_gradient(self):
        model = make_model(True)
        self.assertIsNotNone(getattr(model, 'detail_encoder', None))
        x = torch.randn(1, 3, 35, 51)
        seg, boundary = model(x)
        self.assertEqual(tuple(seg.shape), (1, 2, 35, 51))
        self.assertEqual(tuple(boundary.shape), (1, 1, 35, 51))
        # Segmentation loss alone must train the entire CNN detail path.
        nn.functional.cross_entropy(seg, torch.randint(0, 2, (1, 35, 51))).backward()
        for name, param in model.detail_encoder.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
            self.assertGreater(param.grad.abs().sum().item(), 0, name)
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))

    def test_disabled_path_matches_decoder_without_detail_skips(self):
        model = make_model(False).eval()
        self.assertIsNone(getattr(model, 'detail_encoder', None))
        self.assertFalse(any('detail' in k or 'skip' in k for k in model.state_dict()))
        decoder = DecoderV3(num_classes=2).eval()
        decoder.load_state_dict(model.decoder.state_dict(), strict=True)
        x = torch.randn(1, 3, 35, 51)
        with torch.no_grad():
            feats = model.encoder(x)
            y = decoder.proj(decoder.fusion(feats))
            for block in (decoder.dec1, decoder.dec2, decoder.dec3, decoder.dec4):
                y = block.block(block.up(y))
            expected = tuple(nn.functional.interpolate(head(y), size=x.shape[2:], mode='bilinear', align_corners=False)
                             for head in (decoder.seg_head, decoder.boundary_head))
            actual = model(x)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_four_cbams_receive_gradients(self):
        fusion = AttentionFusion(16, DEFAULT_FEATURE_LAYERS)
        feats = {layer: torch.randn(1, 16, 3, 5, requires_grad=True) for layer in DEFAULT_FEATURE_LAYERS}
        self.assertEqual(tuple(fusion.cbam), ('3', '6', '9', '12'))
        torch.testing.assert_close(fusion.weight.softmax(0), torch.full((4,), 0.25))
        fusion(feats).square().mean().backward()
        self.assertGreater(fusion.weight.grad.abs().sum().item(), 0)
        for layer, feature in feats.items():
            self.assertGreater(feature.grad.abs().sum().item(), 0, layer)
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in fusion.cbam[str(layer)].parameters()), layer)

    def test_new_structure_strict_checkpoint_roundtrip(self):
        model = make_model(True).eval()
        restored = make_model(True).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        x = torch.randn(1, 3, 32, 48)
        with torch.no_grad():
            for a, b in zip(model(x), restored(x)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_detail_parameters_are_in_existing_optimizer(self):
        model = make_model(True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        ids = {id(p) for group in optimizer.param_groups for p in group['params']}
        self.assertTrue(all(p.requires_grad and id(p) in ids for p in model.detail_encoder.parameters()))


if __name__ == '__main__':
    unittest.main()
