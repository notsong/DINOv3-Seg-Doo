# CLAUDE.md — DINOv3 + FPN-UNet 工业边缘语义分割

## 项目概述

基于 DINOv3（ViT-B/16）作为特征提取骨干，结合 FPN + UNet 混合解码器的工业级边缘分割方案。专为工业图像的目标边缘检测设计。

## 技术栈

- **Python 3.10+** | **PyTorch 2.x** | **CUDA** (AMP 混合精度)
- **HuggingFace Transformers** — 加载 DINOv3 预训练权重
- **Albumentations** — 数据增强
- **OpenCV** — 图像读写、形态学后处理
- **Matplotlib** — 训练曲线绘制
- conda 环境: `D:/work/anaconda/envs/unet_pro/python`

## 项目结构

```
├── config.py                  # 全局配置 dataclass（唯一配置入口）
├── train.py                   # 训练入口（三阶段：冻结→末层→全量微调）
├── infer.py                   # 推理入口（FP16 + 批量滑窗 + 形态学后处理）
├── export_onnx.py             # ONNX 导出
├── data/
│   ├── dataset_loader.py       # SegDataset（有标注） + UnlabeledDataset
│   └── transforms.py          # Albumentations 训练/验证增强
├── models/
│   ├── dinov3_encoder.py      # DINOv3 多层特征提取（f4/f8/f16）
│   ├── fpn_unet_decoder.py    # FPN 投影 + 4级×2上采样解码器
│   └── dinov3_segmentation.py # 顶层模型 = Encoder + Decoder
├── losses/
│   └── loss.py                # TotalLoss = CE + Dice + Boundary
├── utils/
│   ├── metric.py              # IoU / Dice / Boundary F1
│   ├── scheduler.py           # Warmup + Cosine 学习率调度
│   ├── logger.py              # CSV 日志 + 训练曲线 PNG
│   ├── visualize.py           # 可视化工具
│   └── seed.py                # 随机种子固定
├── checkpoints/               # 模型权重 (best_iou.pth / best_dice.pth / last.pth)
├── logs/                      # config.json + metrics.csv + training_curves.png
├── dataset/                   # 数据集（按 README 结构组织）
└── output/                    # 推理输出
```

## 常用命令

```bash
# 激活环境并训练
D:/work/anaconda/envs/unet_pro/python train.py

# 推理
D:/work/anaconda/envs/unet_pro/python infer.py

# ONNX 导出
D:/work/anaconda/envs/unet_pro/python export_onnx.py

# 快速检查导入
D:/work/anaconda/envs/unet_pro/python -c "from config import cfg; print(cfg)"
```

## 架构关键细节

### Encoder: DINOv3 特征提取
- **Backbone**: `facebook/dinov3-vitb16-pretrain-lvd1689m`
- **patch_size=16**（以 backbone 配置为准），默认输入 `image_size=1024`，网格为64×64；非方形输入分别计算高宽
- 取最后 3 层 hidden_states，token 顺序为 CLS、register、patch；按配置去掉前面的 CLS 和 register tokens
- 所有层级输出相同空间分辨率 `[B, 768, H//16, W//16]`（ViT 不像 CNN 有下采样金字塔）

### Decoder: DecoderV3
- 默认 `use_detail_branch=True`：浅层CNN提供1/2、1/4、1/8细节，在前三次上采样后、残差块前投影相加。该开关已接入执行。
- 旧结构检查点需关闭细节开关；新结构需训练，不允许静默忽略缺失权重。
- 最后三层同分辨率特征分别经过 CBAM，再可学习加权融合。
- 1×1 投影至512通道，四级残差上采样：512→256→128→64→32。
- Segmentation Head 输出2通道，Boundary Head 输出1通道，最后插值到输入尺寸。

### 渐进解冻与分层学习率
- 默认epoch 0–9冻结骨干；10–29训练末4层；30起全量解冻。`full_unfreeze_epoch=-1`禁用最后阶段，`freeze_backbone=False`从起始全量微调。
- head基准LR=1e-4，骨干末层=1e-5，向前逐层乘0.8；新解冻组独立预热3轮。
- AdamW分组始终保留，阶段切换不丢动量。每轮开始从epoch计算LR与可训练掩码，按组比例设最低LR。
- 新断点保存training_policy、optimizer、scaler、最佳指标与RNG。恢复需保持策略配置一致；旧格式不能直接恢复新优化器。

### 损失函数
- `TotalLoss(cfg)` 使用 `bce_weight × CE + dice_weight × Dice + boundary_weight × Boundary`。
- 三个权重已接入 config；`bce_pos_weight` 未接入，不能视为正类加权已经生效。
- Boundary Head 使用 Laplacian 提取的 GT 边缘做 BCEWithLogits。

### 推理优化
- **FP16** 全流程（模型 + 归一化常量都用 half）
- **批量滑窗**: `crop_size`、`stride`、`infer_batch_size` 均从 config 读取，默认每批1个窗口；短边补灰后裁回，无翻转 TTA
- 大图自动降采样 0.85× 以减少窗口数
- 形态学闭运算后处理 (`cv2.MORPH_CLOSE`) 增强边缘连续性

## 编码约定

- 配置**只能**通过 `config.py` 的 `Config` dataclass 修改，不散落硬编码
- 模型通过 `cfg` 对象参数化，不直接 import 模块级常量
- 新增 loss 权重需同时更新 `config.py`（定义参数）和 `loss.py`（实际使用）
- 所有 tensor 操作优先使用 `torch.cuda.amp.autocast` 兼容混合精度
- 随机种子通过 `utils/seed.py` 统一设置
- 数据集 mask 统一为 `.png` 格式，像素值 0/1（非 0/255）

## 已知问题 / 待办

1. `bce_pos_weight`、`deep_supervision`、`use_attention`、`use_boundary_head`、`use_multi_layer_features` 等标志未接入对应执行路径；不要假设切换配置即可改变结构。
2. `batch_size=4` + `image_size=1024` + ViT-B 对显存要求较高（建议 ≥12GB），AMP 已开启。
3. 权重文件较大（DINOv3 backbone ≈ 350MB），不纳入版本控制。
