import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import infer
from config import DEFAULT_FEATURE_LAYERS
from models.dinov3_encoder import DINOv3Encoder


class TokenBackbone(nn.Module):
    def __init__(self, registers=4, patch_size=16, extra=0, include_embedding=True):
        super().__init__()
        self.config = SimpleNamespace(num_register_tokens=registers, patch_size=patch_size, num_hidden_layers=12)
        self.extra = extra
        self.include_embedding = include_embedding

    def forward(self, pixel_values, output_hidden_states):
        b, _, h, w = pixel_values.shape
        n = (h // self.config.patch_size) * (w // self.config.patch_size)
        prefix = torch.full((b, 1 + self.config.num_register_tokens, 2), -999.)
        tokens = torch.arange(n + self.extra).float().view(1, -1, 1).expand(b, -1, 2)
        first = 0 if self.include_embedding else 1
        return SimpleNamespace(hidden_states=tuple(
            torch.cat((prefix, tokens + layer * 100), dim=1) for layer in range(first, 13)))


class EncoderTests(unittest.TestCase):
    def make_encoder(self, backbone, layers=DEFAULT_FEATURE_LAYERS):
        with patch('models.dinov3_encoder.AutoImageProcessor.from_pretrained'), patch(
            'models.dinov3_encoder.AutoModel.from_pretrained',
            return_value=backbone
        ):
            return DINOv3Encoder('unused', feature_layers=layers)

    def check_grid(self, h, w, registers=4, patch_size=16, include_embedding=True):
        encoder = self.make_encoder(TokenBackbone(registers, patch_size, include_embedding=include_embedding))
        result = encoder(torch.zeros(2, 3, h, w))
        gh, gw = h // patch_size, w // patch_size
        expected = torch.arange(gh * gw).float().reshape(gh, gw)
        self.assertEqual(tuple(result), (3, 6, 9, 12))
        for layer, feature in result.items():
            self.assertEqual(feature.shape, (2, 2, gh, gw))
            torch.testing.assert_close(feature[:, 0], (expected + layer * 100).expand(2, -1, -1))

    def test_square_spatial_order(self):
        self.check_grid(32, 32)

    def test_rectangular_spatial_order(self):
        self.check_grid(32, 48)
        self.check_grid(32, 48, include_embedding=False)

    def test_nondivisible_input_uses_convolution_grid(self):
        self.check_grid(35, 51)

    def test_configured_register_count_and_patch_size(self):
        self.check_grid(16, 24, registers=0, patch_size=8)
        self.check_grid(16, 24, registers=2, patch_size=8)

    def test_inconsistent_token_count_is_rejected(self):
        encoder = self.make_encoder(TokenBackbone(extra=1))
        with self.assertRaisesRegex(ValueError, 'patch tokens'):
            encoder(torch.zeros(1, 3, 32, 48))

    def test_invalid_layer_numbers_are_rejected(self):
        for layers in ((3, 6, 12), (3, 6, 6, 12), (0, 6, 9, 12), (3, 6, 9, 13)):
            with self.subTest(layers=layers), self.assertRaisesRegex(ValueError, 'feature_layers'):
                self.make_encoder(TokenBackbone(), layers=layers)


class PixelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, x):
        self.batch_sizes.append(len(x))
        score = x[:, :1].float()
        return torch.cat((torch.zeros_like(score), score), dim=1), None


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((64, 64, 3), dtype=np.uint8)
        self.image[:, :, 0] = np.arange(64, dtype=np.uint8)[None, :] * 4

    def test_single_has_one_forward_and_preserves_position(self):
        model = PixelModel()
        with patch.object(infer.cfg, 'image_size', 64):
            result = infer.single_infer(model, self.image, 'cpu')
        self.assertEqual(model.batch_sizes, [1])
        self.assertGreater(result[20, -1], result[20, 0])

    def test_sliding_has_one_forward_without_tta(self):
        model = PixelModel()
        result = infer.batched_sliding(model, self.image, 'cpu', crop=64, stride=64)
        self.assertEqual(model.batch_sizes, [1])
        expected = torch.sigmoid((torch.from_numpy(self.image[:, :, 0].astype(np.float32) / 255).half()
                                  - infer.MEAN[0, 0, 0, 0]) / infer.STD[0, 0, 0, 0]).float().numpy()
        np.testing.assert_allclose(result, expected, atol=3e-4)

    def test_narrow_images_keep_all_pixels(self):
        for shape in ((20, 150, 3), (150, 20, 3), (20, 20, 3)):
            with self.subTest(shape=shape):
                result = infer.batched_sliding(PixelModel(), np.full(shape, 128, np.uint8), 'cpu', crop=64, stride=48)
                self.assertEqual(result.shape, shape[:2])
                self.assertTrue(np.isfinite(result).all())
                np.testing.assert_allclose(result, result[0, 0], atol=1e-6)
                self.assertGreater(result.min(), 0)

    def test_narrow_image_preserves_spatial_values(self):
        image = np.tile(self.image, (1, 3, 1))[:20]
        result = infer.batched_sliding(PixelModel(), image, 'cpu', crop=64, stride=48)
        channel = torch.from_numpy(image[:, :, 0].astype(np.float32) / 255).half()
        expected = torch.sigmoid(((channel - infer.MEAN[0, 0, 0, 0]) / infer.STD[0, 0, 0, 0]).float()).numpy()
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_single_and_batch_agree_for_identical_preprocessing(self):
        # Remove letterbox differences to isolate the prediction paths.
        with patch.object(infer, 'letterbox', return_value=(self.image, (0, 0, 0, 0))):
            single = infer.single_infer(PixelModel(), self.image, 'cpu')
        batch = infer.batched_sliding(PixelModel(), self.image, 'cpu', crop=64, stride=64)
        np.testing.assert_allclose(single, batch, atol=1e-6)

    def test_dispatch_uses_configured_windows_after_scaling(self):
        image = np.full((20, 1300, 3), 128, np.uint8)
        model = PixelModel()
        with patch.object(infer.cv2, 'imread', return_value=image), patch.object(
            infer.cfg, 'crop_size', 64
        ), patch.object(infer.cfg, 'stride', 48), patch.object(infer.cfg, 'image_size', 64):
            mask, prob = infer.infer_single(model, 'unused', 'cpu')
        self.assertEqual(mask.shape, (20, 1300))
        self.assertEqual(prob.shape, (20, 1300))
        self.assertEqual(sum(model.batch_sizes), 23)

    def test_batches_are_bounded_and_results_independent_of_batch_size(self):
        image = np.tile(self.image, (2, 3, 1))
        results = []
        for size in (1, 2):
            model = PixelModel()
            with patch.object(infer.cfg, 'infer_batch_size', size, create=True):
                results.append(infer.batched_sliding(model, image, 'cpu', crop=64, stride=48))
            self.assertLessEqual(max(model.batch_sizes), size)
            self.assertEqual(sum(model.batch_sizes), 12)
        np.testing.assert_allclose(*results, atol=1e-6)

    def test_configured_window_parameters_are_used(self):
        model = PixelModel()
        with patch.object(infer.cfg, 'crop_size', 64), patch.object(infer.cfg, 'stride', 48):
            result = infer.batched_sliding(model, self.image, 'cpu')
        self.assertEqual(result.shape, (64, 64))
        self.assertEqual(model.batch_sizes, [1])

    def test_stride_cannot_leave_gaps(self):
        with self.assertRaises(ValueError):
            infer.batched_sliding(PixelModel(), self.image, 'cpu', crop=32, stride=48)


if __name__ == '__main__':
    unittest.main()
