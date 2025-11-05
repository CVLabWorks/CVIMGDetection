import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from torchvision.transforms import v2
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import classification_report
import json

# Use pre-trained ResNet50 for better performance
def get_model(num_classes=2):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_data_loaders(data_dir, batch_size=32):
    # Data transformations using v2
    transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),                # 自动转换为 Image 类型
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    
    # Get all subfolders (different AI types)
    image_types = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    train_datasets = []
    val_datasets = {}

    for image_type in image_types:
        train_dir = os.path.join(data_dir, image_type, 'train')
        val_dir = os.path.join(data_dir, image_type, 'val')

        # Load train data
        train_dataset = datasets.ImageFolder(train_dir, transform=transform)
        train_datasets.append(train_dataset)

        # Load val data
        val_dataset = datasets.ImageFolder(val_dir, transform=transform)
        val_datasets[image_type] = val_dataset

    # Combine all train datasets
    combined_train_dataset = ConcatDataset(train_datasets)
    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True)

    # Create val loaders for each type
    val_loaders = {ai_type: DataLoader(val_datasets[ai_type], batch_size=batch_size, shuffle=False)
                   for ai_type in image_types}

    return train_loader, val_loaders

def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}')

def evaluate_model(model, val_loaders, device='cuda'):
    model.to(device)
    model.eval()
    results = {}
    with torch.no_grad():
        for ai_type, val_loader in val_loaders.items():
            all_preds = []
            all_labels = []
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            report = classification_report(all_labels, all_preds, target_names=['ai', 'nature'], output_dict=True)
            results[ai_type] = report
            print(f'Results for {ai_type}:')
            print(classification_report(all_labels, all_preds, target_names=['ai', 'nature']))
    return results

def main():
    data_dir = './datasets'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    train_loader, val_loaders = get_data_loaders(data_dir)

    # Initialize model, loss, optimizer
    model = get_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs=10, device=device)

    # Save model checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': 10,
        'loss': criterion,
    }, 'model_checkpoint.pth')
    print("Model checkpoint saved to model_checkpoint.pth")

    # Evaluate and record results
    results = evaluate_model(model, val_loaders, device=device)

    # Save results to JSON
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("Results saved to results.json")

if __name__ == '__main__':
    main()
