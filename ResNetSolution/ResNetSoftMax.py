import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import classification_report
import json
from utils import init_checkpoint_dir, init_results_dir, init_logs_dir, DualLogger, CHECKPOINT_DIR, RESULTS_DIR, LOGS_DIR, get_data_loaders

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
    log_file = os.path.join(LOGS_DIR, 'resnet_softmax.log')
    open(log_file, 'w').close()  # Clear log file
    import sys
    sys.stdout = DualLogger(log_file)
    
    data_dir = './datasets'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    train_loader, val_loaders = get_data_loaders(data_dir)

    # Initialize model, loss, optimizer
    model = get_model()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Check if checkpoint exists
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'model_resnet_softmax_checkpoint.pth')
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Checkpoint loaded successfully")
    else:
        # Train the model
        print("No checkpoint found, training model...")
        train_model(model, train_loader, optimizer, num_epochs=10, device=device)

        # Save model checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': 10,
        }, checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")

    # Evaluate and record results
    results = evaluate_model(model, val_loaders, device=device)

    # Save results to JSON
    results_file = os.path.join(RESULTS_DIR, 'results_resnet_softmax.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_file}")

if __name__ == '__main__':
    main()
