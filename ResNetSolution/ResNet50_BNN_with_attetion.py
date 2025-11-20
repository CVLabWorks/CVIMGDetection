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
import cv2
import numpy as np
from PIL import Image
import argparse
import matplotlib.pyplot as plt
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

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        x_cat = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        attention = self.sigmoid(self.conv(x_cat))  # [B, 1, H, W]
        return x * attention

class ResNet50_BNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Remove avgpool and fc, keep up to layer4
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        self.attention = SpatialAttention()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = BayesianLinear(2048, num_classes)

    def forward(self, x, sample=True):
        features_map = self.resnet(x)  # [B, 2048, 7, 7]
        features_map = self.attention(features_map)
        features = self.avgpool(features_map).view(x.size(0), -1)  # [B, 2048]
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
            all_fake_probs = []  # Store prob of being fake for all samples
            fake_probs = []      # Store prob of being fake only for predicted fake samples (legacy)
            for inputs, labels, paths in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                # Extract features from ResNet with attention and avgpool (same as forward)
                features_map = model.resnet(inputs)  # [B, 2048, 7, 7]
                features_map = model.attention(features_map)
                features = model.avgpool(features_map).view(inputs.size(0), -1)  # [B, 2048]
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
                
                # Collect probabilities for AUC (class 1 is fake)
                all_fake_probs.extend(mean_probs[:, 1].cpu().numpy())

                # For predicted fake images (class 1), record prob of fake
                fake_mask = (preds == 1).cpu()
                if fake_mask.any():
                    fake_probs.extend(mean_probs[fake_mask, 1].cpu().numpy())
            
            # Calculate Metrics
            report = classification_report(all_labels, all_preds, target_names=['nature', 'ai'], output_dict=True)
            
            # Calculate AUC
            try:
                auc = roc_auc_score(all_labels, all_fake_probs)
            except ValueError:
                auc = 0.0  # Handle edge cases (e.g., only one class in batch)
            
            report['auc'] = auc
            results[ai_type] = report
            
            print(f'Results for {ai_type}:')
            print(classification_report(all_labels, all_preds, target_names=['nature', 'ai']))
            print(f'AUC for {ai_type}: {auc:.4f}')
            if fake_probs:
                print(f'Fake probabilities for {ai_type}: mean={sum(fake_probs)/len(fake_probs):.4f}, samples={len(fake_probs)}')
    return results

def train():
    """Train the model and save checkpoint."""
    # Initialize directories
    init_checkpoint_dir()
    init_results_dir()
    init_logs_dir()
    
    # Setup logging
    log_file = os.path.join(LOGS_DIR, 'resnet_bnn_with_attention.log')
    open(log_file, 'w').close()  # Clear log file
    sys.stdout = DualLogger(log_file)
    
    data_dir = './datasets'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_loader, val_loaders = get_data_loaders(data_dir)

    # Initialize model, optimizer
    model = get_model()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, optimizer, num_epochs=10, device=device)

    # Save model checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'model_resnet_bnn_with_attention_checkpoint.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': 10,
    }, checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")

    # Evaluate and record results
    results = evaluate_model(model, val_loaders, device=device)

    # Save results to JSON
    results_file = os.path.join(RESULTS_DIR, 'results_resnet_bnn_with_attention.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_file}")

def main():
    """Load trained model from checkpoint, or train if not exists, then evaluate."""
    # Initialize directories
    init_checkpoint_dir()
    init_results_dir()
    init_logs_dir()
    
    # Setup logging
    log_file = os.path.join(LOGS_DIR, 'resnet_bnn_with_attention.log')
    open(log_file, 'w').close()  # Clear log file
    sys.stdout = DualLogger(log_file)
    
    data_dir = './datasets'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'model_resnet_bnn_with_attention_checkpoint.pth')
    
    # Load data
    train_loader, val_loaders = get_data_loaders(data_dir)
    
    # Initialize model and optimizer
    model = get_model()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Check if checkpoint exists
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
    results_file = os.path.join(RESULTS_DIR, 'results_resnet_bnn_with_attention.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {results_file}")

def generate_gradcam(model, input_tensor, target_class):
    """
    Generate Grad-CAM heatmap for the given input and target class.
    Uses deterministic forward (sample=False) for simplicity.
    """
    model.eval()

    gradients = []

    def save_gradient(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    activations = None

    def save_activation(module, input, output):
        nonlocal activations
        activations = output

    # Hook on attention layer
    hook_grad = model.attention.register_backward_hook(save_gradient)
    hook_act = model.attention.register_forward_hook(save_activation)

    # Forward pass
    output = model(input_tensor, sample=False)

    # Backward
    model.zero_grad()
    output[0, target_class].backward()

    # Get data
    grad = gradients[0].cpu().data.numpy()[0]  # [2048, 7, 7]
    act = activations.cpu().data.numpy()[0]   # [2048, 7, 7]

    # Compute weights
    weights = np.mean(grad, axis=(1, 2))  # [2048]

    # Generate CAM
    cam = np.zeros(act.shape[1:], dtype=np.float32)  # [7, 7]
    for i, w in enumerate(weights):
        cam += w * act[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    if np.max(cam) != 0:
        cam = cam / np.max(cam)

    # Remove hooks
    hook_grad.remove()
    hook_act.remove()

    return cam

def test_image(model, image_path, device='cuda', save_path='heatmap.jpg'):
    """
    End-to-end test interface: Given an image path, compute probability and save Grad-CAM heatmap.
    """
    model.to(device)
    model.eval()

    # Load and preprocess image
    transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    original_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(original_image).unsqueeze(0).to(device)

    # Predict probabilities with Monte Carlo sampling
    predictions = []
    for _ in range(50):  # num_samples
        output = model(input_tensor, sample=True)
        probs = F.softmax(output, dim=1)
        predictions.append(probs)
    predictions = torch.stack(predictions)
    mean_probs = predictions.mean(dim=0)[0]
    prob_fake = mean_probs[1].item()  # Class 1: fake
    prob_real = mean_probs[0].item()  # Class 0: real

    # Generate Grad-CAM heatmap for fake class
    heatmap = generate_gradcam(model, input_tensor, target_class=1)

    # Save heatmap overlaid on original image
    original_array = np.array(original_image.resize((224, 224)))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_array, 0.6, heatmap_colored, 0.4, 0)
    plt.imshow(overlay)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Image: {image_path}")
    print(f"Probability of being real (nature): {prob_real:.4f}")
    print(f"Probability of being fake (AI): {prob_fake:.4f}")
    print(f"Heatmap saved to: {save_path}")

    return prob_fake, heatmap

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or Test ResNet50-BNN with Attention and Explanation')
    parser.add_argument('--test_image', type=str, help='Relative path to image for testing')
    parser.add_argument('--train', action='store_true', help='Train the model (default is evaluate)')
    args = parser.parse_args()

    if args.test_image:
        if not os.path.exists(args.test_image):
            print(f"Image file {args.test_image} does not exist.")
            exit(1)
        model = get_model()
        checkpoint_path = os.path.join(CHECKPOINT_DIR, 'model_resnet_bnn_with_attention_checkpoint.pth')
        if not os.path.exists(checkpoint_path):
            print(f"Model checkpoint {checkpoint_path} not found. Please train first.")
            exit(1)
        model.load_state_dict(torch.load(checkpoint_path, weights_only=False)['model_state_dict'])
        save_path = os.path.splitext(args.test_image)[0] + '_heatmap.jpg'
        test_image(model, args.test_image, save_path=save_path)
    elif args.train:
        train()
    else:
        main()
