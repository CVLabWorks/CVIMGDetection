import os
import sys
import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader, ConcatDataset

# Directory paths
CHECKPOINT_DIR = '.checkpoint'
RESULTS_DIR = 'results'
LOGS_DIR = 'logs'

def init_checkpoint_dir():
    """Initialize checkpoint directory (create if doesn't exist). Call once at startup."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def init_results_dir():
    """Initialize results directory (create if doesn't exist)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

def init_logs_dir():
    """Initialize logs directory (create if doesn't exist)."""
    os.makedirs(LOGS_DIR, exist_ok=True)

class DualLogger:
    """Dual logger to write output to both console and log file."""
    def __init__(self, log_file):
        self.log_file = log_file
        self.terminal = sys.stdout
    
    def write(self, message):
        self.terminal.write(message)
        with open(self.log_file, 'a') as f:
            f.write(message)
    
    def flush(self):
        pass

class ImageFolderWithPaths(datasets.ImageFolder):
    """ImageFolder that also returns image paths."""
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path = self.samples[index][0]
        return img, label, path

def get_data_loaders(data_dir, batch_size=32):
    """
    Load training and validation data loaders with image paths.
    
    Args:
        data_dir: Root directory containing subdirectories for each AI type
        batch_size: Batch size for data loaders
    
    Returns:
        train_loader: Combined training DataLoader (yields images, labels, paths)
        val_loaders: Dict of validation DataLoaders keyed by AI type (yield images, labels, paths)
    """
    # Data transformations using v2
    transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Get all subfolders (different AI types)
    ai_types = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    train_datasets = []
    val_datasets = {}
    
    for ai_type in ai_types:
        train_dir = os.path.join(data_dir, ai_type, 'train')
        val_dir = os.path.join(data_dir, ai_type, 'val')
        
        # Load train data
        train_dataset = ImageFolderWithPaths(train_dir, transform=transform)
        train_datasets.append(train_dataset)
        
        # Load val data
        val_dataset = ImageFolderWithPaths(val_dir, transform=transform)
        val_datasets[ai_type] = val_dataset
    
    # Combine all train datasets
    combined_train_dataset = ConcatDataset(train_datasets)
    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create val loaders for each type
    val_loaders = {ai_type: DataLoader(val_datasets[ai_type], batch_size=batch_size, shuffle=False)
                   for ai_type in ai_types}
    
    return train_loader, val_loaders
