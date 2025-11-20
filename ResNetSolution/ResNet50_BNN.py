import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from sklearn.metrics import classification_report, roc_auc_score
import json
from torch.distributions import Normal, kl_divergence
from utils import init_checkpoint_dir, init_results_dir, init_logs_dir, DualLogger, CHECKPOINT_DIR, RESULTS_DIR, LOGS_DIR, get_data_loaders

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Variational parameters - weight mean and std
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).normal_(-3, 0.1))

        # Variational parameters - bias mean and std
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).normal_(-3, 0.1))

        # Prior distributions
        self.weight_prior = Normal(0, 1)
        self.bias_prior = Normal(0, 1)

    def forward(self, x, sample=True):
        if sample:
            # Reparameterization trick
            weight_epsilon = torch.randn_like(self.weight_mu)
            weight_sigma = torch.log1p(torch.exp(self.weight_rho))
            weight = self.weight_mu + weight_epsilon * weight_sigma

            bias_epsilon = torch.randn_like(self.bias_mu)
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias = self.bias_mu + bias_epsilon * bias_sigma
        else:
            # Use mean for deterministic prediction
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def kl_divergence(self):
        # Compute KL divergence between variational posterior and prior
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        q_weight = Normal(self.weight_mu, weight_sigma)
        kl_weight = kl_divergence(q_weight, self.weight_prior).sum()

        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        q_bias = Normal(self.bias_mu, bias_sigma)
        kl_bias = kl_divergence(q_bias, self.bias_prior).sum()

        return kl_weight + kl_bias

class ResNet50_BNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Remove the final fc layer, keep up to avgpool
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.classifier = BayesianLinear(2048, num_classes)

    def forward(self, x, sample=True):
        features = self.resnet(x).view(x.size(0), -1)  # Flatten after avgpool
        return self.classifier(features, sample)

    def kl_loss(self):
        return self.classifier.kl_divergence()

def get_model(num_classes=2):
    return ResNet50_BNN(num_classes)

def train_model(model, train_loader, optimizer, num_epochs=10, device='cuda'):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_kl = 0.0
        total_nll = 0.0
        for inputs, labels, _ in train_loader:  # _ for paths
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, sample=True)
            nll_loss = nn.functional.cross_entropy(outputs, labels)
            kl_loss = model.kl_loss()
            loss = nll_loss + kl_loss / len(train_loader.dataset)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total_nll += nll_loss.item()
            total_kl += kl_loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, '
              f'NLL: {total_nll/len(train_loader):.4f}, KL: {total_kl/len(train_loader):.4f}')

def evaluate_model(model, val_loaders, device='cuda', num_samples=50):
    model.to(device)
    model.eval()
    results = {}
    with torch.no_grad():
        for ai_type, val_loader in val_loaders.items():
            all_preds = []
            all_labels = []
            all_fake_probs = [] # Store prob of being fake for all samples
            fake_probs = []     # Store prob of being fake only for predicted fake samples
            for inputs, labels, paths in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                # Extract features from ResNet
                features = model.resnet(inputs).view(inputs.size(0), -1)
                # Get multiple samples for uncertainty
                predictions = []
                for _ in range(num_samples):
                    outputs = model.classifier(features, sample=True)
                    probs = nn.functional.softmax(outputs, dim=1)
                    predictions.append(probs)
                predictions = torch.stack(predictions)
                mean_probs = predictions.mean(dim=0)
                _, preds = torch.max(mean_probs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Collect probabilities for AUC
                all_fake_probs.extend(mean_probs[:, 1].cpu().numpy())

                # For predicted fake images (class 1), record prob of fake
                fake_mask = (preds == 1).cpu()
                if fake_mask.any():
                    fake_probs.extend(mean_probs[fake_mask, 1].cpu().numpy())
            
            report = classification_report(all_labels, all_preds, target_names=['nature', 'ai'], output_dict=True)
            
            # Calculate AUC
            try:
                auc = roc_auc_score(all_labels, all_fake_probs)
            except ValueError:
                auc = 0.0
            
            report['auc'] = auc
            results[ai_type] = report
            
            print(f'Results for {ai_type}:')
            print(classification_report(all_labels, all_preds, target_names=['nature', 'ai']))
            print(f'AUC for {ai_type}: {auc:.4f}')
            if fake_probs:
                print(f'Fake probabilities for {ai_type}: mean={sum(fake_probs)/len(fake_probs):.4f}, samples={len(fake_probs)}')
    return results

def main():
    # Initialize directories
    init_checkpoint_dir()
    init_results_dir()
    init_logs_dir()
    
    # Setup logging
    log_file = os.path.join(LOGS_DIR, 'resnet_bnn.log')
    open(log_file, 'w').close()  # Clear log file
    sys.stdout = DualLogger(log_file)
    
    data_dir = './datasets'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    train_loader, val_loaders = get_data_loaders(data_dir)

    # Initialize model, optimizer
    model = get_model()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Check if checkpoint exists
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'model_resnet_bnn_checkpoint.pth')
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
    results_file = os.path.join(RESULTS_DIR, 'results_resnet_bnn.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_file}")

if __name__ == '__main__':
    main()
