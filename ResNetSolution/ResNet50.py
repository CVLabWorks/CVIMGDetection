import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import classification_report
import json
from utils import init_checkpoint_dir, init_results_dir, init_logs_dir, DualLogger, CHECKPOINT_DIR, RESULTS_DIR, LOGS_DIR, get_data_loaders

# Use pre-trained ResNet50 for better performance
def get_model(num_classes=2):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels, _ in train_loader:  # _ for paths
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
            for inputs, labels, _ in val_loader:  # _ for paths
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
    # Initialize directories
    init_checkpoint_dir()
    init_results_dir()
    init_logs_dir()
    
    # Setup logging
    log_file = os.path.join(LOGS_DIR, 'resnet.log')
    open(log_file, 'w').close()  # Clear log file
    import sys
    sys.stdout = DualLogger(log_file)
    
    data_dir = './datasets'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    train_loader, val_loaders = get_data_loaders(data_dir)

    # Initialize model, loss, optimizer
    model = get_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Check if checkpoint exists
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'model_resnet_checkpoint.pth')
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Checkpoint loaded successfully")
    else:
        # Train the model
        print("No checkpoint found, training model...")
        train_model(model, train_loader, criterion, optimizer, num_epochs=10, device=device)

        # Save model checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': 10,
            'loss': criterion,
        }, checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")

    # Evaluate and record results
    results = evaluate_model(model, val_loaders, device=device)

    # Save results to JSON
    results_file = os.path.join(RESULTS_DIR, 'results_resnet.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_file}")

if __name__ == '__main__':
    main()
