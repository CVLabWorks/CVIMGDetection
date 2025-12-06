from config import Config
from data.merge_dataset import merge_all_datasets
from models.model import ViTWithCustomHead, ViTWithInterpolation, ViTWithLocalPerception
from utils.train_utils import TransferLearningTrainer
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import torch
from data.dataset import TransferLearningDataset  


class AIDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None, data_root="/home/mdouab/vit_ai_detection/data/tiny-genimage/versions/1"):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.data_root = data_root  

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


def build_transfer_model(config, num_classes):
    if config.MODEL_TYPE == 'vit':
        return ViTWithCustomHead(num_classes=num_classes, freeze_backbone=True)
    elif config.MODEL_TYPE == 'vit_pos':
        return ViTWithInterpolation(num_classes=num_classes, freeze_backbone=True)
    elif config.MODEL_TYPE == 'vit_local':
        return ViTWithLocalPerception(num_classes=num_classes, freeze_backbone=True)
    else:
        raise ValueError(f"不支持的 MODEL_TYPE: {config.MODEL_TYPE}")


def get_transform_params(model_type):

    if model_type == 'dino':
        return 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        if model_type == 'vit_pos':
            image_size = 384
        elif model_type == 'vit_local':
            image_size = 224
        else:
            image_size = 224
        return image_size, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]


def transfer_learning_main():
    torch.manual_seed(Config.SEED)

    df = merge_all_datasets()
    unique_models = df['ai_model'].unique()
    num_classes = len(unique_models)
    print(f" 迁移学习：识别 {num_classes} 个AI模型类型")
    print(f"AI模型列表: {list(unique_models)}")

    image_size, mean, std = get_transform_params(Config.MODEL_TYPE)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_dataset = TransferLearningDataset(
        csv_path=Config.MERGED_CSV,
        data_root="/home/mdouab/vit_ai_detection/data/tiny-genimage/versions/1",
        split="train",
        transform=transform
    )
    val_dataset = TransferLearningDataset(
        csv_path=Config.MERGED_CSV,
        data_root="/home/mdouab/vit_ai_detection/data/tiny-genimage/versions/1",
        split="val",
        transform=transform
    )

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

    model = build_transfer_model(Config, num_classes)

    trainer = TransferLearningTrainer(model, train_loader, val_loader, Config, num_classes=num_classes)
    trainer.fit()

    print(" 迁移学习阶段1完成！模型已保存为 transfer_best_model.pth")


def binary_classification_from_transfer():
    torch.manual_seed(Config.SEED)

    df = merge_all_datasets()
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']

    image_size, mean, std = get_transform_params(Config.MODEL_TYPE)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_dataset = AIDetectionDataset(train_df, transform=transform, data_root="/home/mdouab/vit_ai_detection/data/tiny-genimage/versions/1")
    val_dataset = AIDetectionDataset(val_df, transform=transform, data_root="/home/mdouab/vit_ai_detection/data/tiny-genimage/versions/1")

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

    model = build_transfer_model(Config, num_classes=2)

    transfer_model_path = "checkpoints/transfer_best_model.pth"
    try:
        checkpoint = torch.load(transfer_model_path, map_location=Config.DEVICE)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and 'classifier' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f" 已加载预训练模型权重: {transfer_model_path}")
    except FileNotFoundError:
        print(f"预训练模型不存在: {transfer_model_path}，使用随机初始化")

    from utils.train_utils import Trainer
    trainer = Trainer(model, train_loader, val_loader, Config)
    trainer.fit()

    print("二分类训练完成！")


if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) > 1 and sys.argv[1] == "binary":
        binary_classification_from_transfer()
    else:
        transfer_learning_main()
