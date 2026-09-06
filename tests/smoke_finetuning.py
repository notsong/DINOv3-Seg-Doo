"""Three synthetic optimizer steps with real local backbone; no dataset training."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from config import cfg
from models.dinov3_segmentation import DINOv3Seg
from losses.loss import TotalLoss
from train import train_one_epoch
from utils.finetuning import build_optimizer, prepare_epoch, transformer_layers


def main():
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DINOv3Seg(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    criterion = TotalLoss(cfg)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    batch = [(torch.randn(1, 3, 32, 48), torch.randint(0, 2, (1, 1, 32, 48)), ['synthetic'])]
    for epoch in (0, cfg.freeze_epochs, cfg.full_unfreeze_epoch):
        stage = prepare_epoch(model, optimizer, cfg, epoch)
        loss = train_one_epoch(model, batch, optimizer, criterion, scaler, device)
        assert torch.isfinite(torch.tensor(loss))
        layers = transformer_layers(model.encoder.backbone)
        flags = [any(p.grad is not None and p.grad.abs().sum() > 0 for p in layer.parameters()) for layer in layers]
        expected = ([False] * len(layers) if stage == 'frozen' else
                    [False] * (len(layers) - cfg.partial_unfreeze_layers) + [True] * cfg.partial_unfreeze_layers
                    if stage == 'partial' else [True] * len(layers))
        assert flags == expected, (stage, flags)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.detail_encoder.parameters())
        print(f'PASS real train_one_epoch: {stage}, active transformer blocks={sum(flags)}, loss={loss:.6f}')


if __name__ == '__main__':
    main()
