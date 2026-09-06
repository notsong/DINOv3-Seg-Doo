"""
config.py
V3 — DINOv3 + Attention Fusion + Residual Decoder + Boundary Head + 渐进解冻 + 日志系统
"""
from dataclasses import dataclass
import os


@dataclass
class Config:

    # ===========================
    # Dataset
    # ===========================
    dataset_root: str = "./dataset"

    train_image_dir: str = os.path.join(dataset_root, "train/images")
    train_mask_dir: str = os.path.join(dataset_root, "train/masks")

    val_image_dir: str = os.path.join(dataset_root, "val/images")
    val_mask_dir: str = os.path.join(dataset_root, "val/masks")

    test_image_dir: str = "E:/Data/JLD/JLD_TOTAL/HQL_test/HQL_Small0629"

    unlabeled_dir: str = os.path.join(dataset_root, "unlabeled")

    # ===========================
    # Model — DINOv3
    # ===========================
    model_type: str = "dinov3_decoder_v3"

    backbone_name: str = "D:/dinov3_model"

    num_classes: int = 2

    in_channels: int = 3

    image_size: int = 1024  # ViT-B/16: 64×64 patch 网格

    use_multi_layer_features: bool = True

    # ===========================
    # Progressive Finetuning
    # ===========================
    epochs: int = 100

    # 阶段1：冻结backbone，训练decoder与CNN细节分支
    freeze_backbone: bool = True
    freeze_epochs: int = 10

    partial_unfreeze_layers: int = 4  # 阶段2：仅解冻末4层

    full_unfreeze_epoch: int = 30  # 阶段3起始epoch（从0计），-1禁用全量解冻

    layer_lr_decay: float = 0.8  # 越靠前的骨干层学习率越小

    finetune_warmup_epochs: int = 3  # 每组骨干参数解冻后的独立预热

    # 骨干最后一层基准LR；更早层按layer_lr_decay递减
    unfreeze_lr: float = 1e-5

    batch_size: int = 4

    num_workers: int = 4

    learning_rate: float = 1e-4

    weight_decay: float = 1e-4

    grad_clip: float = 1.0

    amp: bool = True

    seed: int = 42

    device: str = "cuda"

    # ===========================
    # Scheduler
    # ===========================
    warmup_epochs: int = 5

    min_lr: float = 1e-6

    # ===========================
    # V3 Decoder
    # ===========================
    use_detail_branch: bool = True  # CNN细节 + 逐级融合；旧结构权重需设False

    use_boundary_head: bool = True

    use_attention: bool = True

    deep_supervision: bool = True  # 预留标志，当前未接入模型/损失

    # ===========================
    # Loss
    # ===========================
    dice_weight: float = 0.4

    bce_weight: float = 0.25

    boundary_weight: float = 0.15

    bce_pos_weight: float = 8.0  # 预留参数，当前 CE/Boundary BCE 未使用

    # ===========================
    # Validation & Logging
    # ===========================
    val_interval: int = 1

    log_dir: str = "./logs"

    # ===========================
    # Checkpoint
    # ===========================
    save_dir: str = "./checkpoints"

    resume: str = ""

    pretrained: bool = True

    save_best_dice: bool = True
    save_best_iou: bool = True

    # ===========================
    # Sliding Window Inference
    # ===========================
    crop_size: int = 1024

    stride: int = 768

    infer_batch_size: int = 1  # 每次前向最多处理的滑窗数，可按显存调高

    # 推理阈值（可调低增强连续性）
    infer_threshold: float = 0.3

    # 推理后处理：形态学闭运算核大小（0=禁用）
    infer_close_kernel: int = 3

    # ===========================
    # ONNX Export
    # ===========================
    onnx_path: str = "./onnx/model.onnx"

    # ===========================
    # Output
    # ===========================
    output_dir: str = "./output"

    visualization_dir: str = "./output/vis"


cfg = Config()
