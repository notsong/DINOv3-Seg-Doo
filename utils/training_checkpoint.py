"""Complete training state for policy-compatible epoch-boundary resumes."""
import random

import numpy as np
import torch

from utils.finetuning import policy_state


def make_checkpoint(model, optimizer, scaler, cfg, epoch, iou, dice, best_iou, best_dice):
    return dict(
        epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
        scaler=scaler.state_dict(), training_policy=policy_state(cfg),
        iou=iou, dice=dice, best_iou=best_iou, best_dice=best_dice,
        rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                 torch=torch.get_rng_state(),
                 cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
    )


def restore_checkpoint(checkpoint, model, optimizer, scaler, cfg):
    if checkpoint.get('training_policy') != policy_state(cfg):
        raise ValueError('Checkpoint training policy is missing or differs from config; '
                         'use matching config for resume. Legacy checkpoints cannot resume this optimizer policy.')
    model.load_state_dict(checkpoint['model'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
    scaler.load_state_dict(checkpoint['scaler'])
    rng = checkpoint['rng']
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch'].cpu())
    if torch.cuda.is_available() and rng['cuda'] is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in rng['cuda']])
    return checkpoint['epoch'] + 1, checkpoint['best_iou'], checkpoint['best_dice']
