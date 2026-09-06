"""
train.py
DINOv3 + DecoderV3 — 渐进解冻与分层学习率 + Boundary Head + 日志系统
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json, dataclasses
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import cfg
from data.dataset_loader import SegDataset
from data.transforms import get_train_transform, get_val_transform
from models.dinov3_segmentation import DINOv3Seg
from losses.loss import TotalLoss
from utils.metric import iou_score, dice_score
from utils.seed import set_seed
from utils.finetuning import build_optimizer, prepare_epoch, set_backbone_train_mode
from utils.training_checkpoint import make_checkpoint, restore_checkpoint
from utils.logger import TrainLogger


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    set_backbone_train_mode(model)
    total_loss = 0
    loop = tqdm(loader, desc="Train")

    for images, masks, _ in loop:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()

        optimizer.zero_grad()

        with torch.amp.autocast(device.type, enabled=cfg.amp and device.type == "cuda"):
            seg, boundary = model(images)

        loss = criterion(seg, boundary, masks)

        scaler.scale(loss).backward()

        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    iou_total = 0
    dice_total = 0

    for images, masks, _ in tqdm(loader, desc="Val"):
        images = images.to(device)
        masks = masks.to(device).long()

        seg, boundary = model(images)
        loss = criterion(seg, boundary, masks)

        preds = torch.argmax(seg, dim=1, keepdim=True)

        total_loss += loss.item()
        iou_total += iou_score(preds.cpu(), masks.cpu())
        dice_total += dice_score(preds.cpu(), masks.cpu())

    n = len(loader)
    return total_loss / n, iou_total / n, dice_total / n


def main():
    set_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Backbone: {cfg.backbone_name}")
    print(f"Config: image_size={cfg.image_size}, batch={cfg.batch_size}, "
          f"epochs={cfg.epochs}, lr={cfg.learning_rate}")
    print(f"Loss: bce={cfg.bce_weight}, dice={cfg.dice_weight}, "
          f"boundary={cfg.boundary_weight}")
    print(f"Progressive: frozen={cfg.freeze_epochs}, last_layers={cfg.partial_unfreeze_layers}, "
          f"full_epoch={cfg.full_unfreeze_epoch}, backbone_lr={cfg.unfreeze_lr}, decay={cfg.layer_lr_decay}")

    # ======================
    # Logger
    # ======================
    logger = TrainLogger(cfg.log_dir)
    logger.save_config(dataclasses.asdict(cfg))

    # ======================
    # Dataset
    # ======================
    train_dataset = SegDataset(
        cfg.train_image_dir, cfg.train_mask_dir,
        transform=get_train_transform(), is_train=True,
        image_size=cfg.image_size
    )
    val_dataset = SegDataset(
        cfg.val_image_dir, cfg.val_mask_dir,
        transform=get_val_transform(), is_train=False,
        image_size=cfg.image_size
    )
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    # ======================
    # Model
    # ======================
    model = DINOv3Seg(cfg).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}%)")

    # ======================
    # Loss, Optimizer, Scheduler
    # ======================
    criterion = TotalLoss(cfg)

    optimizer = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    best_iou = 0.0
    best_dice = 0.0
    start_epoch = 0

    # ======================
    # Resume
    # ======================
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device, weights_only=False)
        start_epoch, best_iou, best_dice = restore_checkpoint(
            ckpt, model, optimizer, scaler, cfg
        )
        del ckpt
        print(f"Resumed from epoch {start_epoch}")

    os.makedirs(cfg.save_dir, exist_ok=True)

    # ======================
    # Training Loop
    # ======================
    for epoch in range(start_epoch, cfg.epochs):
        stage = prepare_epoch(model, optimizer, cfg, epoch)
        current_lr = optimizer.param_groups[0]['lr']
        active_backbone_lrs = [g['lr'] for g in optimizer.param_groups if g['name'] != 'head' and g['lr'] > 0]
        print(f"Epoch [{epoch+1}/{cfg.epochs}] stage={stage} head_lr={current_lr:.2e} "
              f"backbone_lr={min(active_backbone_lrs, default=0):.2e}..{max(active_backbone_lrs, default=0):.2e}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device
        )
        val_loss, val_iou, val_dice = validate(
            model, val_loader, criterion, device
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val   Loss: {val_loss:.4f}")
        print(f"Val   IoU : {val_iou:.4f}")
        print(f"Val   Dice: {val_dice:.4f}")

        # ---- log ----
        logger.log_epoch(epoch, train_loss, val_loss, val_iou, val_dice, current_lr)

        improved_iou, improved_dice = val_iou > best_iou, val_dice > best_dice
        best_iou, best_dice = max(best_iou, val_iou), max(best_dice, val_dice)
        state = make_checkpoint(model, optimizer, scaler, cfg, epoch,
                                val_iou, val_dice, best_iou, best_dice)
        if improved_iou and cfg.save_best_iou:
            save_checkpoint(state, os.path.join(cfg.save_dir, "best_iou.pth"))
        if improved_dice and cfg.save_best_dice:
            save_checkpoint(state, os.path.join(cfg.save_dir, "best_dice.pth"))
        save_checkpoint(state, os.path.join(cfg.save_dir, "last.pth"))
        del state

    # ======================
    # Plot training curves
    # ======================
    logger.plot()
    print(f"\nTraining done! Best IoU: {best_iou:.4f}, Best Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
