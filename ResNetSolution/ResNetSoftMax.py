import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from torchvision.transforms import v2
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import classification_report
import json

class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path = self.samples[index][0]
        return img, label, path

class ResNet50Softmax(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.resnet(x)
        probs = self.softmax(logits)
        return probs

def get_model(num_classes=2):
    return ResNet50Softmax(num_classes)

def get_data_loaders(data_dir, batch_size=32):
    # Data transformations using v2
    transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),                # Convert to Image type
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

def train_model(model, train_loader, optimizer, num_epochs=10, device='cuda'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()  # Note: expects logits, but our model outputs probs. Wait, problem.
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels, _ in train_loader:  # _ for paths
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # For training, we need logits, not probs.
            # So, temporarily modify forward or use logits.
            # Better to separate: model.resnet(x) for logits, then softmax in forward.
            # But to fix, let's make forward return logits, and apply softmax only in eval.
            # Wait, for simplicity, in train, use model.resnet(x), in eval use full model.
            logits = model.resnet(inputs)
            loss = criterion(logits, labels)
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
            for inputs, labels, paths in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                probs = model(inputs)  # This gives probabilities
                _, preds = torch.max(probs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                # Record per image probabilities
                for i in range(len(paths)):
                    path = paths[i]
                    prob_ai = probs[i, 0].item()  # Assuming class 0 is ai (fake)
                    prob_nature = probs[i, 1].item()  # class 1 nature (real)
                    print(f'{path}: AI(fake) prob: {prob_ai:.4f}, Nature(real) prob: {prob_nature:.4f}')
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
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, optimizer, num_epochs=10, device=device)

    # Save model checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': 10,
    }, 'model_softmax_checkpoint.pth')
    print("Model checkpoint saved to model_softmax_checkpoint.pth")

    # Evaluate and record results
    results = evaluate_model(model, val_loaders, device=device)

    # Save results to JSON
    with open('results_softmax.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("Results saved to results_softmax.json")

if __name__ == '__main__':
    main()
