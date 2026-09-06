# CNN detail branch implementation

Goal: add local detail at 1/8, 1/4 and 1/2 decoder stages, preserving the corrected baseline when disabled.
Architecture: 16/32/64-channel shallow CNN; upsample semantic features to each skip grid, project detail to the stage input width, add, then apply the existing ResidualBlock. This uses the existing block to reduce to 256/128/64 without another high-resolution residual block.
Stack: existing PyTorch ConvGNAct, GroupNorm, GELU, bilinear interpolation.
User approved implementation and backed up existing model. Enable in Config by default; no training, commit or push.

- [x] Add failing tests for detail grids, skip influence/gradients, absent skips, disabled baseline state keys and numerical compatibility, odd shapes and frozen encoder gradients.
- [x] Add models/detail_encoder.py: three two-convolution stages at widths 16, 32, 64 with stride 2 then 1.
- [x] Extend UpResidualBlock with optional skip_channels and skip argument. Project to in_channels before existing residual block; disabled state has no new parameters.
- [x] Extend DecoderV3 with optional use_detail_branch (default false for direct legacy callers), enforce detail input if enabled, align final stage to output_size only in new mode.
- [x] Wire DINOv3Seg and Config.use_detail_branch (default true); maintain (seg, boundary) output and encoder freeze behavior.
- [x] Run unittest discovery, real backbone FP16 forward and one backward pass, parameter counts and bounded decoder benchmark.
- [x] Document switch, compatibility, new-train requirement, verification and measured performance limitations.

Verification: D:/work/anaconda/envs/unet_pro/python.exe -m unittest discover -s tests -v; python tests/smoke_local.py; git diff --check.
