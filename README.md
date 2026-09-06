# DINOv3-Seg-Doo：工业边缘语义分割

基于 **DINOv3（ViT-B/16）** 作为特征提取骨干，结合 **Attention Fusion、CNN细节分支与残差解码器** 的工业级边缘分割方案。专为工业图像的目标边缘检测设计，支持训练、推理、ONNX 导出全流程。

---

## 目录

- [项目背景](#项目背景)
- [模型架构](#模型架构)
- [目录结构](#目录结构)
- [环境依赖](#环境依赖)
- [快速开始](#快速开始)
  - [1. 准备数据集](#1-准备数据集)
  - [2. 修改配置](#2-修改配置)
  - [3. 训练](#3-训练)
  - [4. 推理](#4-推理)
  - [5. ONNX 导出](#5-onnx-导出)
- [核心设计](#核心设计)
  - [Encoder：DINOv3 多层特征提取](#encoderdinov3-多层特征提取)
  - [Decoder：FPN + UNet 混合结构](#decoderfpn--unet-混合结构)
  - [损失函数：三合一组合 Loss](#损失函数三合一组合-loss)
  - [学习率调度：Warmup + Cosine](#学习率调度warmup--cosine)
  - [滑窗推理](#滑窗推理)
- [评估指标](#评估指标)
- [配置说明](#配置说明)
- [License](#license)

---

## 项目背景

工业检测中，目标边缘的自动提取是材料科学的关键任务。边缘具有以下特点：

- **细薄**：通常仅 1-3 像素宽，属于极端的细粒度分割
- **拓扑连续**：边缘必须形成闭合回路才有物理意义
- **尺度不一**：同一图像中目标大小差异可达数十倍

传统图像处理方法（Canny、分水岭等）在噪声大、对比度低的工业场景下鲁棒性不足。本项目采用深度学习方案，利用 DINOv3 自监督预训练的强语义特征，配合边界感知损失，实现高精度边缘分割。

---

## 模型架构

```text
输入图像 [B, 3, H, W]（默认1024×1024）
  ↓ DINOv3 ViT-B/16
最后三层 f4/f8/f16：均为 [B, 768, H//16, W//16]
  ↓ 每层 CBAM + 可学习加权融合
同分辨率特征融合 → 768通道
  ↓ 1×1 投影 → 512通道
四级残差上采样 → 256 → 128 → 64 → 32通道
  ↓ Segmentation Head / Boundary Head
输出 seg [B, 2, H, W] / boundary [B, 1, H, W]
```

### 关键参数

| 组件 | 参数 |
|------|------|
| Backbone | `facebook/dinov3-vitb16-pretrain-lvd1689m` |
| 输入尺寸 | 768×768 |
| FPN 通道数 | 256 |
| 解码器通道数 | 256→128→64→32 |
| 参数量（骨干冻结） | ~15M（仅解码器可训练） |

---

## 目录结构

```
├── datasets/
│   ├── dataset_loader.py      # 数据集加载（训练/验证/测试/无标注）
│   └── transforms.py         # 数据增强（Albumentations）
│
├── models/
│   ├── dinov3_encoder.py     # DINOv3 多层特征提取器
│   ├── fpn_unet_decoder.py   # FPN + UNet 混合解码器
│   └── dinov3_segmentation.py # 完整模型（Encoder + Decoder）
│
├── losses/
│   ├── dice.py               # Dice Loss（占位，实际实现在 loss.py）
│   ├── boundary_loss.py      # Boundary Loss（占位，实际实现在 loss.py）
│   └── loss.py               # 总损失：0.4×BCE + 0.4×Dice + 0.2×Boundary
│
├── utils/
│   ├── metric.py             # 评估指标：IoU、Dice、Boundary F1
│   ├── visualize.py          # 可视化工具：mask 着色、叠加显示
│   ├── scheduler.py          # Warmup + Cosine 学习率调度器
│   └── seed.py               # 随机种子固定（可复现性）
│
├── train.py                  # 训练入口
├── infer.py                  # 推理入口（单图 + 滑窗）
├── export_onnx.py            # ONNX 模型导出
├── config.py                 # 全局配置（dataclass）
│
├── checkpoints/              # 模型权重保存目录
└── README.md
```

---

## 环境依赖

```
torch >= 2.0
torchvision
transformers          # HuggingFace（加载 DINOv3）
albumentations        # 数据增强
opencv-python         # 图像读写
numpy
tqdm
```

一键安装：

```bash
pip install torch torchvision transformers albumentations opencv-python numpy tqdm
```

---

## 快速开始

### 1. 准备数据集

按以下结构组织数据：

```
dataset/
├── train/
│   ├── images/          # 训练原图 (.png/.jpg/.tif)
│   └── masks/           # 训练标注 (灰度图，边缘=255，背景=0)
├── val/
│   ├── images/          # 验证原图
│   └── masks/           # 验证标注
├── test/
│   └── images/          # 测试原图（无需标注）
└── unlabeled/           # 无标注数据（可选，用于伪标签扩展）
```

> **注意**：mask 文件名必须与对应 image 文件名一致。

### 2. 修改配置

编辑 [config.py](config.py)，调整以下关键参数：

```python
dataset_root: str = "./dataset"          # 数据集根目录
image_size: int = 768                    # 输入尺寸
batch_size: int = 4                      # 批次大小（显存不足可减小）
epochs: int = 80                         # 训练轮数
freeze_backbone: bool = True             # 是否冻结 DINOv3
```

### 3. 训练

```bash
python train.py
```

训练过程输出示例：

```
Epoch [1/80]
Train: 100%|██████████| 50/50 [02:30<00:00, loss=0.342]
Val: 100%|██████████| 10/10 [00:20<00:00]
Train Loss: 0.3512
Val Loss: 0.2891
Val IoU: 0.7823
Val Dice: 0.8534
Saved best model
```

- 每个 epoch 自动保存 `best.pth`（最佳 IoU）和 `last.pth`（最新）
- 支持断点续训：设置 `cfg.resume = "./checkpoints/last.pth"`

### 4. 推理

```bash
python infer.py
```

推理结果保存至 `./output/infer/`，每个输入生成：
- `*_mask.png`：二值分割结果
- `*_prob.png`：概率热力图

大图按 `infer_single` 的尺寸分支启用滑窗；默认 `crop_size=1024, stride=768, infer_batch_size=1`，重叠区域取平均。单图与滑窗均不使用翻转 TTA；0.85 缩放逻辑保留。短边不足窗口时补灰，回填时裁去补边。

### 5. ONNX 导出

```bash
python export_onnx.py
```

导出为 `./onnx/model.onnx`，支持动态 batch size 和输入尺寸，可直接部署到 Triton Inference Server 或其他 ONNX Runtime 环境。

---

## 核心设计

### Encoder：DINOv3 多层特征提取

不同于常见的仅取最后一层特征，本项目从 DINOv3 的 hidden states 中提取 **最后三层、同一空间分辨率** 的特征：

| 层级 | 来源 | 下采样倍率 | 语义 |
|------|------|-----------|------|
| f4 | hidden_states[-3] | 16× | 低级纹理（目标边缘细节） |
| f8 | hidden_states[-2] | 16× | 中级结构（目标形状） |
| f16 | hidden_states[-1] | 16× | 高级语义（目标区域） |

特征从 `[B, N, C]` (patch sequence) reshape 为 `[B, C, H, W]` 空间格式，按真实配置丢弃前面的 CLS 和 register tokens。网格高宽由输入高宽分别整除 patch_size 得到；f4/f8/f16 仅为兼容层名，不代表真实金字塔。默认冻结骨干网络，仅训练解码器，避免在小数据集上过拟合。

### Decoder：FPN + UNet 混合结构

1. **特征融合**：最后三层同分辨率特征分别经过 CBAM，再以可学习权重融合。
2. **残差解码**：投影到512通道，四级2×上采样，通道依次为256、128、64、32。
3. **输出头**：分割头和边界辅助头分别输出，最后插值到输入尺寸。

### 损失函数：三合一组合 Loss

针对边缘分割的特殊性，采用三种损失的加权组合：

```
Total Loss = 0.4 × BCE + 0.4 × Dice + 0.2 × Boundary
```

| 损失 | 作用 | 实现细节 |
|------|------|---------|
| **BCE Loss** | 像素级二分类监督 | `BCEWithLogitsLoss`，对 logits 的边缘通道 (channel=1) 计算 |
| **Dice Loss** | 缓解类别不平衡（边缘像素占比极小） | Smooth Dice，加入 Laplace 平滑（eps=1） |
| **Boundary Loss** | 强化目标边缘连续性 | 用 Laplacian 核 `[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]` 提取预测与真值的边缘，计算 L1 Loss |

> **为什么需要 Boundary Loss？** 边缘仅占图像像素的 1-3%，BCE + Dice 的组合倾向于给出模糊的边界预测。Boundary Loss 显式监督边缘梯度，迫使模型输出锐利、连续的边缘线。

### 学习率调度：Warmup + Cosine

训练入口使用 `utils/finetuning.py`，按epoch确定阶段：默认epoch 0–9冻结骨干，仅训练解码器和细节CNN；10–29解冻最后4层；30起全量微调。阶段边界从0计数，默认总轮数100。`full_unfreeze_epoch=-1`可保持末层微调，`freeze_backbone=False`从epoch 0全量微调。

解码器和CNN基准LR为 `learning_rate=1e-4`，骨干末层为 `unfreeze_lr=1e-5`，向前每层乘 `layer_lr_decay=0.8`，embedding再低一级。head预热5轮，骨干各组解冻后独立预热 `finetune_warmup_epochs=3` 轮，再余弦衰减。最低LR按 `min_lr/learning_rate` 比例应用到各组，保持层间比例。冻结组LR为0且不计算参数梯度。

优化器只建立一次，解冻不重置Adam动量。新checkpoint保存策略配置、optimizer、AMP scaler、最佳指标、Python/NumPy/Torch随机状态；每轮恢复时重算解冻状态和学习率。`resume`要求策略配置匹配，旧格式断点不能恢复新分组优化器。随机状态恢复不保证跨设备、不同worker配置或非确定性CUDA运算完全复现。

上述轮数是可调起点，不是经本项目精度验证的最优值。未启动完整训练。

### 滑窗推理

对于超过 `image_size` 的大图（如 2048×2048 的全景显微扫描），自动采用滑窗方式：

1. 以 `crop_size=1024, stride=768` 在图像上滑动
2. 每个窗口独立推理，得到概率图
3. 重叠区域累加概率并除以计数矩阵（加权平均融合）
4. 边界窗口自动对齐到图像边缘，避免越界

---

## 评估指标

| 指标 | 说明 | 代码位置 |
|------|------|---------|
| **IoU (Jaccard)** | 逐类 IoU 的平均值（背景 + 边缘），评估整体分割质量 | [utils/metric.py](utils/metric.py) |
| **Dice (F1)** | 边缘类的二值 Dice 系数，对细粒度目标更敏感 | [utils/metric.py](utils/metric.py) |
| **Boundary F1** | 用 Laplacian 核提取边缘后计算 F1，评估边缘连续性 | [utils/metric.py](utils/metric.py) |

---

## 配置说明

完整配置见 [config.py](config.py)，以下是核心配置项：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `dataset_root` | `"./dataset"` | 数据集根目录 |
| `backbone_name` | `"facebook/dinov3-vitb16-pretrain-lvd1689m"` | DINOv3 模型名称（HuggingFace） |
| `image_size` | `1024` | 训练/推理输入尺寸 |
| `batch_size` | `4` | 批次大小 |
| `epochs` | `100` | 总训练轮数 |
| `learning_rate` | `1e-4` | 初始学习率 |
| `warmup_epochs` | `5` | Warmup 轮数 |
| `min_lr` | `1e-6` | 最小学习率 |
| `dice_weight` | `0.4` | Dice Loss 权重 |
| `bce_weight` | `0.4` | BCE Loss 权重 |
| `boundary_weight` | `0.2` | Boundary Loss 权重 |
| `freeze_backbone` | `True` | 是否冻结 DINOv3 骨干 |
| `amp` | `True` | 混合精度训练 |
| `crop_size` | `1024` | 滑窗推理窗口大小 |
| `stride` | `768` | 滑窗步长 |
| `onnx_path` | `"./onnx/model.onnx"` | ONNX 导出路径 |

---

## License

本项目仅供学习和研究使用。


### 正确性回归验证

运行 `python -m unittest discover -s tests -v`，无需真实权重或 GPU。
可选运行 `python tests/smoke_local.py`，读取 config 中的本地 backbone 权重，检查非方形网格与 FP16 推理；解码器随机初始化，不用于精度评价。

特征排列修复改变了旧权重依赖的空间对应关系，需要重训后评估细晶界和小目标精度，不能据此承诺旧权重精度提升。
`bce_pos_weight`、`deep_supervision` 当前仅定义而未接入执行；本次未改变训练损失策略。


### CNN 细节分支与逐级融合

`config.use_detail_branch=True` 默认启用新结构。归一化图像经过轻量CNN，生成1/2、1/4、1/8分辨率的16、32、64通道特征。它们分别接入解码器前三个上采样阶段：先按细节网格对齐，再用1×1卷积投影到该阶段输入通道（128、256、512），相加后复用已有残差块细化，输出通道仍为64、128、256。最终一级直接对齐输入高宽，支持奇数和非方形尺寸。

CNN分支不随DINOv3冻结，现有优化器会自动训练其参数。分割和边界双输出接口不变，损失策略不变。新分支加跳跃投影共新增117,104个参数。

加载旧结构完整检查点时，将 `use_detail_branch=False`，该路径保留旧参数键与原始解码数值行为（编码器仍采用修复后的正确排列）。新结构需训练后使用，不能将旧检查点用 `strict=False` 静默加载后当作训练完成的新模型。新结构恢复训练/推理时应保持开关为True。

`python -m unittest discover -s tests -v` 验证开关、奇数尺寸、梯度、优化器参数与严格检查点兼容；`python tests/smoke_local.py` 另做真实骨干前向和一次混合精度反向（不更新参数、不启动训练）；`python tests/benchmark_decoder.py` 测量合成256×256输入的解码器与细节分支开销，不含骨干和数据预处理，不代表完整推理耗时。


### 本地轻量验证

- `python -m unittest discover -s tests -v`：正确性、结构、渐进解冻和断点恢复测试。
- `python tests/smoke_local.py`：真实本地骨干、非方形输入和FP16推理验证。
- `python tests/smoke_finetuning.py`：每阶段一个32×48合成batch，通过真实 `train_one_epoch` 执行共3次更新，验证0/4/12层分别获得梯度，不读取训练数据集。

本次实测环境为PyTorch 2.8.0+cu128、Transformers 5.13.0、RTX 3060。旧 `utils/scheduler.py` 保留供既有外部调用，训练入口已不使用它。
