"""Epoch-derived progressive unfreezing with stable AdamW parameter groups."""
import math

import torch


POLICY_FIELDS = (
    'epochs', 'freeze_backbone', 'freeze_epochs', 'partial_unfreeze_layers',
    'full_unfreeze_epoch', 'learning_rate', 'unfreeze_lr', 'layer_lr_decay',
    'warmup_epochs', 'finetune_warmup_epochs', 'min_lr', 'weight_decay',
    'use_detail_branch', 'feature_layers',
)


def policy_state(cfg):
    return {'version': 1, **{name: getattr(cfg, name) for name in POLICY_FIELDS}}


def transformer_layers(backbone):
    """DINOv3 HF releases expose the layer list under different containers."""
    for container in (getattr(backbone, 'model', None), getattr(backbone, 'encoder', None), backbone):
        layers = getattr(container, 'layer', None)
        if isinstance(layers, torch.nn.ModuleList):
            return layers
    raise ValueError('Unsupported DINOv3 transformer layer layout')


def build_optimizer(model, cfg):
    backbone = model.encoder.backbone
    layers = transformer_layers(backbone)
    if (cfg.epochs <= 0 or cfg.freeze_epochs < 0
            or not 1 <= cfg.partial_unfreeze_layers <= len(layers)
            or not 0 < cfg.layer_lr_decay <= 1
            or not 0 < cfg.unfreeze_lr <= cfg.learning_rate
            or not 0 <= cfg.min_lr <= cfg.learning_rate
            or cfg.warmup_epochs < 0 or cfg.finetune_warmup_epochs < 0
            or (cfg.full_unfreeze_epoch != -1 and cfg.full_unfreeze_epoch < cfg.freeze_epochs)):
        raise ValueError('Invalid progressive finetuning configuration')
    full_start = cfg.full_unfreeze_epoch if cfg.full_unfreeze_epoch >= 0 else cfg.epochs
    if not cfg.freeze_backbone:
        full_start = 0
    groups, assigned = [], set()

    def add(name, params, base_lr, start, warmup):
        params = list(params)
        if not params:
            return
        if any(id(p) in assigned for p in params):
            raise ValueError('Duplicate optimizer parameter')
        assigned.update(id(p) for p in params)
        groups.append(dict(name=name, params=params, lr=base_lr,
                           base_lr=base_lr, start_epoch=start, warmup=warmup))

    backbone_ids = {id(p) for p in backbone.parameters()}
    add('head', (p for p in model.parameters() if id(p) not in backbone_ids),
        cfg.learning_rate, 0, cfg.warmup_epochs)
    add('embeddings', backbone.embeddings.parameters(),
        cfg.unfreeze_lr * cfg.layer_lr_decay ** len(layers), full_start, cfg.finetune_warmup_epochs)
    for i, layer in enumerate(layers):
        start = (cfg.freeze_epochs if i >= len(layers) - cfg.partial_unfreeze_layers else full_start)
        if not cfg.freeze_backbone:
            start = 0
        add(f'layer.{i}', layer.parameters(), cfg.unfreeze_lr * cfg.layer_lr_decay ** (len(layers) - 1 - i),
            start, cfg.finetune_warmup_epochs)
    add('backbone.other', (p for p in backbone.parameters() if id(p) not in assigned),
        cfg.unfreeze_lr, full_start, cfg.finetune_warmup_epochs)
    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)


def prepare_epoch(model, optimizer, cfg, epoch):
    """Set trainability and LR before the epoch, also when resuming mid-stage."""
    if not 0 <= epoch < cfg.epochs:
        raise ValueError('Epoch outside configured training range')
    floor_ratio = cfg.min_lr / cfg.learning_rate
    for group in optimizer.param_groups:
        active = epoch >= group['start_epoch']
        for parameter in group['params']:
            parameter.requires_grad_(active)
            if not active:
                parameter.grad = None
        if not active:
            group['lr'] = 0.0
            continue
        local = epoch - group['start_epoch']
        duration = cfg.epochs - group['start_epoch']
        warmup = min(group['warmup'], duration)
        if local < warmup:
            factor = (local + 1) / warmup
        else:
            progress = min(1.0, (local - warmup) / max(1, duration - warmup - 1))
            factor = floor_ratio + (1 - floor_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        group['lr'] = group['base_lr'] * factor
    if not cfg.freeze_backbone or (cfg.full_unfreeze_epoch >= 0 and epoch >= cfg.full_unfreeze_epoch):
        return 'full'
    return 'partial' if epoch >= cfg.freeze_epochs else 'frozen'


def set_backbone_train_mode(model):
    """Call after model.train(): frozen embeddings/blocks stay in evaluation mode."""
    backbone = model.encoder.backbone
    backbone.eval()
    for module in (backbone.embeddings, *transformer_layers(backbone), backbone.norm):
        module.train(model.training and any(p.requires_grad for p in module.parameters()))
