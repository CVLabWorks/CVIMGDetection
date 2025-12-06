import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
import os

class AIDetectionDataset(Dataset):
    def __init__(self, csv_path, data_root, split="train", transform=None):
        self.transform = transform
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

        df = pd.read_csv(csv_path)
        self.df = df[df['split'] == split].reset_index(drop=True)
        self.data_root = data_root  
        print(f" [{split}] 加载 {len(self.df)} 个样本")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        full_img_path = os.path.join(self.data_root, row['image_path'])

        if not os.path.exists(full_img_path):
            print(f" 图像不存在: {full_img_path}")
            image = Image.new('RGB', (224, 224), (128, 128, 128))  # 灰色占位图
        else:
            try:
                image = Image.open(full_img_path).convert('RGB')
            except Exception as e:
                print(f"读取失败 {full_img_path}: {e}")
                image = Image.new('RGB', (224, 224), (0, 0, 0))  # 黑色占位图

        if self.transform:
            image = self.transform(image)

        label = 1 if row['is_ai'] else 0
        return image, torch.tensor(label, dtype=torch.long)


class TransferLearningDataset(Dataset):

    def __init__(self, csv_path, data_root=None, split="train", transform=None):
        self.transform = transform
        self.data_root = data_root  
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

        df = pd.read_csv(csv_path)
        self.df = df[df['split'] == split].reset_index(drop=True)

        unique_models = self.df['ai_model'].unique()
        self.model_to_idx = {model: idx for idx, model in enumerate(unique_models)}
        self.idx_to_model = {idx: model for model, idx in self.model_to_idx.items()}

        print(f"[{split}] 加载 {len(self.df)} 个样本")
        print(f" AI模型类别: {list(self.model_to_idx.keys())}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']

        if self.data_root is not None:
            full_img_path = os.path.join(self.data_root, img_path)
        else:
            full_img_path = img_path  

        if not os.path.exists(full_img_path):
            print(f"图像不存在: {full_img_path}")
            image = Image.new('RGB', (224, 224), (128, 128, 128)) 
        else:
            try:
                image = Image.open(full_img_path).convert('RGB')
            except Exception as e:
                print(f"读取失败 {full_img_path}: {e}")
                image = Image.new('RGB', (224, 224), (0, 0, 0))  

        if self.transform:
            image = self.transform(image)

        label = self.model_to_idx[row['ai_model']]
        return image, torch.tensor(label, dtype=torch.long)

    def get_num_classes(self):
        return len(self.model_to_idx)


def get_transforms():
    train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train, val
