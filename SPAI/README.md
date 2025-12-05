# SPAI: Spectral AI-Generated Image Detector

基于 [CVPR2025 论文](https://openaccess.thecvf.com/content/CVPR2025/html/Karageorgiou_Any-Resolution_AI-Generated_Image_Detection_by_Spectral_Learning_CVPR_2025_paper.html) 的 AI 生成图像检测器，使用自定义数据集进行训练。

## 项目概述

SPAI 通过频谱学习来检测 AI 生成的图像。它在自监督设置下学习真实图像的频谱分布，然后使用频谱重建相似度将 AI 生成的图像检测为该学习模型的分布外样本。

**本仓库基于原始 SPAI 代码进行了修改，支持自定义数据集训练。**

## 代码修改

### Bug 修复

- **`spai/__main__.py`**: 修复了 `lmdb_path` 参数处理的 bug
  ```python
  # 原代码（bug）
  "lmdb_path": str(lmdb_path),  # str(None) = "None" 导致错误
  
  # 修复后
  "lmdb_path": str(lmdb_path) if lmdb_path is not None else None,
  ```

### 新增文件

| 文件 | 说明 |
|------|------|
| `create_my_dataset_csv.py` | 自定义数据集 CSV 生成脚本 |
| `start_training.sh` | 训练启动脚本（含 Neptune 配置） |
| `dataset/my_dataset.csv` | 生成的数据集描述文件 |
| `CLAUDE.md` | Claude Code 开发指南 |

## 数据集

### 数据集结构

```
dataset/
└── mini_gen/
    ├── imagenet_ai_0419_biggan/
    │   ├── train/
    │   │   ├── ai/        # AI 生成图像
    │   │   └── nature/    # 真实图像
    │   └── val/
    │       ├── ai/
    │       └── nature/
    ├── imagenet_ai_0419_vqdm/
    ├── imagenet_ai_0424_sdv5/
    ├── imagenet_ai_0424_wukong/
    ├── imagenet_ai_0508_adm/
    ├── imagenet_glide/
    └── imagenet_midjourney/
```

### 数据集统计

| 类别 | 数量 |
|------|------|
| **总图像数** | 35,000 张 |
| 训练集 | 28,000 张 |
| 验证集 | 7,000 张 |
| AI 生成图像 | 17,500 张 |
| 真实图像 | 17,500 张 |

### 包含的生成器

- BigGAN
- VQDM
- Stable Diffusion v5
- Wukong
- ADM
- GLIDE
- Midjourney

## 环境安装

```bash
# 创建 conda 环境
conda create -n spai python=3.11
conda activate spai

# 安装 PyTorch（需要 CUDA 支持）
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# 安装其他依赖
pip install -r requirements.txt
```

**硬件要求**：
- 推理：< 8GB GPU 显存
- 训练：建议 24GB+ GPU 显存（使用梯度累积可降低要求）

## 预训练模型

下载预训练的 ViT-B/16 MFM 模型，放置到 `weights/` 目录：

```
weights/
└── mfm_pretrain_vit_base.pth
```

下载地址：[MFM GitHub](https://github.com/Jiahao000/MFM)

## 快速开始

### 1. 生成数据集 CSV（如需重新生成）

```bash
python create_my_dataset_csv.py
```

### 2. 配置训练脚本

编辑 `start_training.sh`，填入你的 Neptune 配置：

```bash
export NEPTUNE_API_TOKEN="your_api_token_here"
export NEPTUNE_PROJECT="yourname/spai-training"
```

如果不使用 Neptune，添加：
```bash
export NEPTUNE_MODE="offline"
```

### 3. 开始训练

```bash
./start_training.sh
```

或手动运行：

```bash
CUDA_VISIBLE_DEVICES=0 python -m spai train \
  --cfg ./configs/spai.yaml \
  --batch-size 4 \
  --accumulation-steps 18 \
  --pretrained ./weights/mfm_pretrain_vit_base.pth \
  --output ./output/train \
  --data-path ./dataset/my_dataset.csv \
  --tag "my_training" \
  --amp-opt-level "O0" \
  --data-workers 8 \
  --save-all \
  --opt "DATA.VAL_BATCH_SIZE" "4" \
  --opt "MODEL.FEATURE_EXTRACTION_BATCH" "32" \
  --opt "TRAIN.EPOCHS" "20"
```

## 训练参数说明

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch-size` | 4 | 训练批次大小 |
| `--accumulation-steps` | 18 | 梯度累积步数（有效batch = 4×18=72） |
| `--amp-opt-level` | O0 | 混合精度（O0=禁用，O1/O2=启用，需要APEX） |
| `--data-workers` | 8 | 数据加载进程数 |

### 配置覆盖参数（通过 `--opt`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TRAIN.EPOCHS` | 35 | 训练轮数 |
| `TRAIN.BASE_LR` | 5e-4 | 基础学习率 |
| `TRAIN.CLIP_GRAD` | None | 梯度裁剪（建议设为1.0） |
| `DATA.VAL_BATCH_SIZE` | 256 | 验证批次大小 |
| `MODEL.FEATURE_EXTRACTION_BATCH` | 400 | 特征提取批次 |

### 显存不足时的建议配置

```bash
--batch-size 4 \
--opt "DATA.VAL_BATCH_SIZE" "4" \
--opt "MODEL.FEATURE_EXTRACTION_BATCH" "32"
```

## 推理

使用训练好的模型进行推理：

```bash
python -m spai infer \
  --input <图像目录或CSV文件> \
  --output <输出目录> \
  --model ./output/train/finetune/<tag>/ckpt_epoch_<N>.pth
```

## 测试/评估

```bash
python -m spai test \
  --cfg ./configs/spai.yaml \
  --batch-size 8 \
  --model ./output/train/finetune/<tag>/ckpt_epoch_<N>.pth \
  --output ./output/test \
  --tag "test" \
  --test-csv ./dataset/my_dataset.csv \
  --opt "DATA.NUM_WORKERS" "8"
```

## 项目结构

```
spai/
├── configs/
│   └── spai.yaml              # 主配置文件
├── dataset/
│   ├── mini_gen/              # 图像数据
│   └── my_dataset.csv         # 数据集描述
├── spai/
│   ├── __main__.py            # CLI 入口
│   ├── models/                # 模型实现
│   │   ├── sid.py             # SPAI 模型
│   │   ├── vision_transformer.py
│   │   └── filters.py         # 频谱滤波
│   ├── data/                  # 数据加载
│   └── tools/                 # 工具脚本
├── weights/                   # 预训练权重
├── output/                    # 训练输出
├── create_my_dataset_csv.py   # 数据集生成脚本
├── start_training.sh          # 训练启动脚本
└── CLAUDE.md                  # 开发指南
```

## Neptune 实验跟踪

本项目集成了 Neptune.ai 进行实验跟踪：

1. 注册账号：https://neptune.ai/
2. 获取 API Token：https://app.neptune.ai/get_my_api_token
3. 设置环境变量：
   ```bash
   export NEPTUNE_API_TOKEN="your_token"
   export NEPTUNE_PROJECT="workspace/project"
   ```

