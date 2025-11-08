# train.py
from config import Config
from data.merge_dataset import merge_all_datasets
from models.model import ViTWithCustomHead, ViTWithInterpolation, ViTWithLocalPerception, SwinTransformerWithCustomHead, CvTWithCustomHead
from utils.train_utils import Trainer
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
# ----------------------------
# Dataset（简化版：不再需要频域支持）
# ----------------------------

class AIDetectionDataset(Dataset):
    def __init__(self, df, transform=None, data_root="/home/mdouab/vit_ai_detection/data/tiny-genimage/versions/1"):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.data_root = data_root  # ← 增加这个参数

    def __len__(self):
        return len(self.df)

    def load_image(self, path):
        full_path = os.path.join(self.data_root, path)
        with Image.open(full_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img.copy()

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = self.load_image(row['image_path'])
        if self.transform:
            image = self.transform(image)
        label = 1 if row['is_ai'] else 0
        return image, label


# ----------------------------
# 模型工厂函数
# ----------------------------
def build_model(config):
    if config.MODEL_TYPE == 'vit':
        return ViTWithCustomHead(freeze_backbone=True)
    elif config.MODEL_TYPE == 'vit_pos':
        return ViTWithInterpolation(freeze_backbone=True)
    elif config.MODEL_TYPE == 'vit_local':
        return ViTWithLocalPerception(freeze_backbone=True)
    elif config.MODEL_TYPE == 'swin':
        return SwinTransformerWithCustomHead(freeze_backbone=True)
    elif config.MODEL_TYPE == 'cvt':
        return CvTWithCustomHead(freeze_backbone=True)
    else:
        raise ValueError(f"不支持的 MODEL_TYPE: {config.MODEL_TYPE}")


# ----------------------------
# 获取图像尺寸和归一化参数
# ----------------------------
def get_transform_params(model_type):
    """
    返回 (image_size, mean, std)
    """
    if model_type == 'dino':
        # DINOv2 使用 ImageNet 统计（但你已不用 DINO，保留以防扩展）
        return 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        # ViT 系列使用 [-1, 1] 归一化
        if model_type == 'vit_pos':
            # 使用更高分辨率，例如 384（必须是 16 的倍数）
            image_size = 384
        elif model_type == 'vit_local':
            image_size = 224
        elif model_type == 'swin':
            image_size = 224
        elif model_type == 'cvt':
            image_size = 224
        else:
            image_size = 224
        return image_size, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]


# ----------------------------
# Main
# ----------------------------
def main():
    torch.manual_seed(Config.SEED)

    # 1. 加载数据
    df = merge_all_datasets()
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']

    # 2. 获取 transform 参数
    image_size, mean, std = get_transform_params(Config.MODEL_TYPE)

    # 3. 定义 transform
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # 4. 创建数据集（不再需要 return_raw_for_freq）
    train_dataset = AIDetectionDataset(train_df, transform=transform, data_root="/home/mdouab/vit_ai_detection/data/tiny-genimage/versions/1")
    val_dataset = AIDetectionDataset(val_df, transform=transform, data_root="/home/mdouab/vit_ai_detection/data/tiny-genimage/versions/1")

    # 5. 创建 DataLoader（不再需要自定义 collate_fn）
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )

    # 6. 构建模型
    model = build_model(Config)

    # 7. 启动训练
    trainer = Trainer(model, train_loader, val_loader, Config)
    trainer.fit()


if __name__ == "__main__":
    main()