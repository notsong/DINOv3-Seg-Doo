import copy
import random
import unittest

import numpy as np
import torch

from test_finetuning import TinyModel, policy
from utils.finetuning import build_optimizer, prepare_epoch


class CheckpointTests(unittest.TestCase):
    def test_restore_optimizer_policy_scaler_metrics_and_rng(self):
        from utils.training_checkpoint import make_checkpoint, restore_checkpoint
        cfg, model = policy(), TinyModel()
        optimizer = build_optimizer(model, cfg)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        prepare_epoch(model, optimizer, cfg, 2)
        model(torch.ones(2, 2)).square().mean().backward()
        optimizer.step()
        saved = copy.deepcopy(make_checkpoint(model, optimizer, scaler, cfg, 2, .4, .5, .6, .7))
        expected_rng = (random.random(), np.random.random(), torch.rand(3))
        restored = TinyModel()
        restored_optimizer = build_optimizer(restored, cfg)
        result = restore_checkpoint(saved, restored, restored_optimizer, scaler, cfg)
        self.assertEqual(result, (3, .6, .7))
        self.assertEqual(random.random(), expected_rng[0])
        self.assertEqual(np.random.random(), expected_rng[1])
        torch.testing.assert_close(torch.rand(3), expected_rng[2])
        torch.testing.assert_close(restored_optimizer.state[restored.decoder.weight]['exp_avg'],
                                   optimizer.state[model.decoder.weight]['exp_avg'])
        self.assertEqual(prepare_epoch(restored, restored_optimizer, cfg, result[0]), 'partial')

    def test_old_or_changed_policy_is_not_silently_resumed(self):
        from utils.training_checkpoint import make_checkpoint, restore_checkpoint
        cfg, model = policy(), TinyModel()
        optimizer = build_optimizer(model, cfg)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        saved = make_checkpoint(model, optimizer, scaler, cfg, 0, 0, 0, 0, 0)
        with self.assertRaisesRegex(ValueError, 'policy'):
            restore_checkpoint({}, model, optimizer, scaler, cfg)
        cfg.freeze_epochs += 1
        with self.assertRaisesRegex(ValueError, 'policy'):
            restore_checkpoint(saved, model, optimizer, scaler, cfg)


if __name__ == '__main__':
    unittest.main()
