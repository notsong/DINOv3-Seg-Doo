import copy
import unittest

import torch
from torch import nn

from config import Config


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.backbone = nn.Module()
        self.encoder.backbone.embeddings = nn.Linear(2, 2)
        self.encoder.backbone.model = nn.Module()
        self.encoder.backbone.model.layer = nn.ModuleList([nn.Linear(2, 2) for _ in range(4)])
        self.encoder.backbone.norm = nn.LayerNorm(2)
        self.decoder = nn.Linear(2, 1)

    def forward(self, x):
        x = self.encoder.backbone.embeddings(x)
        for layer in self.encoder.backbone.model.layer:
            x = torch.tanh(layer(x))
        return self.decoder(self.encoder.backbone.norm(x))


def policy():
    cfg = Config()
    cfg.epochs = 8
    cfg.freeze_epochs = 2
    cfg.partial_unfreeze_layers = 2
    cfg.full_unfreeze_epoch = 4
    cfg.warmup_epochs = 1
    cfg.finetune_warmup_epochs = 2
    cfg.layer_lr_decay = 0.8
    return cfg


class FinetuningTests(unittest.TestCase):
    def api(self):
        from utils.finetuning import build_optimizer, prepare_epoch
        return build_optimizer, prepare_epoch

    def test_stage_masks_and_layer_learning_rates(self):
        build, prepare = self.api()
        model, cfg = TinyModel(), policy()
        optimizer = build(model, cfg)
        self.assertEqual(prepare(model, optimizer, cfg, 0), 'frozen')
        self.assertFalse(any(p.requires_grad for p in model.encoder.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.decoder.parameters()))
        self.assertEqual(prepare(model, optimizer, cfg, 2), 'partial')
        layers = model.encoder.backbone.model.layer
        self.assertEqual([layer.weight.requires_grad for layer in layers], [False, False, True, True])
        lrs = {g['name']: g['lr'] for g in optimizer.param_groups}
        self.assertGreater(lrs['head'], lrs['layer.3'])
        self.assertAlmostEqual(lrs['layer.2'] / lrs['layer.3'], 0.8)
        self.assertEqual(lrs['embeddings'], 0)
        self.assertEqual(prepare(model, optimizer, cfg, 4), 'full')
        self.assertTrue(all(p.requires_grad for p in model.encoder.parameters()))

    def test_optimizer_covers_every_parameter_once_and_keeps_momentum(self):
        build, prepare = self.api()
        model, cfg = TinyModel(), policy()
        optimizer = build(model, cfg)
        ids = [id(p) for g in optimizer.param_groups for p in g['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters()})
        prepare(model, optimizer, cfg, 0)
        model(torch.ones(2, 2)).square().mean().backward()
        optimizer.step()
        momentum = optimizer.state[model.decoder.weight]['exp_avg'].clone()
        optimizer.zero_grad(set_to_none=True)
        prepare(model, optimizer, cfg, 2)
        torch.testing.assert_close(optimizer.state[model.decoder.weight]['exp_avg'], momentum)

    def test_resume_across_stages_matches_uninterrupted_updates(self):
        build, prepare = self.api()
        cfg = policy()
        for resume_epoch in (2, 3, 4, 5):
            with self.subTest(resume_epoch=resume_epoch):
                torch.manual_seed(7)
                model = TinyModel()
                optimizer = build(model, cfg)
                for epoch in range(resume_epoch):
                    prepare(model, optimizer, cfg, epoch)
                    optimizer.zero_grad(set_to_none=True)
                    model(torch.ones(2, 2)).square().mean().backward()
                    optimizer.step()
                restored = TinyModel()
                restored.load_state_dict(copy.deepcopy(model.state_dict()))
                restored_optimizer = build(restored, cfg)
                restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
                for epoch in range(resume_epoch, cfg.epochs):
                    for m, opt in ((model, optimizer), (restored, restored_optimizer)):
                        prepare(m, opt, cfg, epoch)
                        opt.zero_grad(set_to_none=True)
                        m(torch.ones(2, 2)).square().mean().backward()
                        opt.step()
                    self.assertEqual([g['lr'] for g in optimizer.param_groups], [g['lr'] for g in restored_optimizer.param_groups])
                for a, b in zip(model.parameters(), restored.parameters()):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_optional_full_unfreeze_and_no_freeze(self):
        build, prepare = self.api()
        model, cfg = TinyModel(), policy()
        cfg.full_unfreeze_epoch = -1
        optimizer = build(model, cfg)
        self.assertEqual(prepare(model, optimizer, cfg, 7), 'partial')
        cfg.freeze_backbone = False
        optimizer = build(model, cfg)
        self.assertEqual(prepare(model, optimizer, cfg, 0), 'full')

    def test_learning_rates_warm_up_and_do_not_invert_depth_order(self):
        build, prepare = self.api()
        model, cfg = TinyModel(), policy()
        optimizer = build(model, cfg)
        rates = []
        for epoch in range(cfg.epochs):
            prepare(model, optimizer, cfg, epoch)
            rates.append({g['name']: g['lr'] for g in optimizer.param_groups})
        self.assertGreater(rates[0]['head'], 0)
        self.assertGreater(rates[3]['layer.3'], rates[2]['layer.3'])
        self.assertGreater(rates[5]['layer.0'], rates[4]['layer.0'])
        for name in rates[-1]:
            self.assertGreater(rates[-1][name], 0)
        self.assertLess(rates[-1]['layer.0'], rates[-1]['layer.3'])
        self.assertAlmostEqual(rates[-1]['layer.0'] / rates[-1]['layer.3'], 0.8 ** 3)

    def test_invalid_policy_rejected(self):
        build, _ = self.api()
        for name, value in (('partial_unfreeze_layers', 5), ('layer_lr_decay', 1.1), ('full_unfreeze_epoch', 1)):
            cfg = policy()
            setattr(cfg, name, value)
            with self.assertRaises(ValueError):
                build(TinyModel(), cfg)

    def test_frozen_modules_stay_eval_after_model_train(self):
        from utils.finetuning import set_backbone_train_mode
        build, prepare = self.api()
        model, cfg = TinyModel(), policy()
        optimizer = build(model, cfg)
        prepare(model, optimizer, cfg, 2)
        model.train()
        set_backbone_train_mode(model)
        backbone = model.encoder.backbone
        self.assertFalse(backbone.embeddings.training)
        self.assertEqual([layer.training for layer in backbone.model.layer], [False, False, True, True])
        self.assertTrue(model.decoder.training)


if __name__ == '__main__':
    unittest.main()
