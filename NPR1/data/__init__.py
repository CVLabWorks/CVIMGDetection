import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
import torchvision.transforms as transforms  # 正确导入
from PIL import Image  # 添加这个导入

from .datasets import dataset_folder

'''
def get_dataset(opt):
    dset_lst = []
    for cls in opt.classes:
        root = opt.dataroot + '/' + cls
        dset = dataset_folder(opt, root)
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst)
'''

import os
# def get_dataset(opt):
#     classes = os.listdir(opt.dataroot) if len(opt.classes) == 0 else opt.classes
#     if '0_real' not in classes or '1_fake' not in classes:
#         dset_lst = []
#         for cls in classes:
#             root = opt.dataroot + '/' + cls
#             dset = dataset_folder(opt, root)
#             dset_lst.append(dset)
#         return torch.utils.data.ConcatDataset(dset_lst)
#     return dataset_folder(opt, opt.dataroot)

def get_dataset(opt):
    # 过滤隐藏文件夹
    all_classes = os.listdir(opt.dataroot) if len(opt.classes) == 0 else opt.classes
    classes = [cls for cls in all_classes if not cls.startswith('.')]
    
    print(f"Found classes: {classes}")
    
    # 自定义数据集类来处理你的数据结构
    class CustomBinaryDataset(torch.utils.data.Dataset):
        def __init__(self, opt, root_dirs):
            self.opt = opt
            self.samples = []
            self.targets = []
            
            # 定义变换
            if opt.isTrain:
                crop_func = transforms.RandomCrop(opt.cropSize)
            elif opt.no_crop:
                crop_func = transforms.Lambda(lambda img: img)
            else:
                crop_func = transforms.CenterCrop(opt.cropSize)

            if opt.isTrain and not opt.no_flip:
                flip_func = transforms.RandomHorizontalFlip()
            else:
                flip_func = transforms.Lambda(lambda img: img)
            if not opt.isTrain and opt.no_resize:
                rz_func = transforms.Lambda(lambda img: img)
            else:
                rz_func = transforms.Resize((opt.loadSize, opt.loadSize))
                
            self.transform = transforms.Compose([
                rz_func,
                crop_func,
                flip_func,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            # 收集所有样本
            for root_dir in root_dirs:
                root_path = os.path.join(opt.dataroot, root_dir)
                for label_name in ['nature', 'ai']:
                    label_path = os.path.join(root_path, label_name)
                    if os.path.exists(label_path):
                        label = 0 if label_name == 'nature' else 1
                        for img_name in os.listdir(label_path):
                            if not img_name.startswith('.'):
                                img_path = os.path.join(label_path, img_name)
                                self.samples.append((img_path, label))
                                self.targets.append(label)
            
        def __getitem__(self, index):
            path, target = self.samples[index]
            img = Image.open(path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return img, target
            
        def __len__(self):
            return len(self.samples)
    
    # 使用自定义数据集
    dataset = CustomBinaryDataset(opt, classes)
    return dataset

def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler


def create_dataloader(opt):
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = get_dataset(opt)
    sampler = get_bal_sampler(dataset) if opt.class_bal else None

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              sampler=sampler,
                                              num_workers=int(opt.num_threads))
    return data_loader
