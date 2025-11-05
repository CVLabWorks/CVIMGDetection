import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, models
from torchvision.transforms import v2
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import classification_report
import json
from torch.distributions import Normal, kl_divergence

class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path = self.samples[index][0]
        return img, label, path

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
    image_types = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    train_datasets = []
    val_datasets = {}

    for image_type in image_types:
        train_dir = os.path.join(data_dir, image_type, 'train')
        val_dir = os.path.join(data_dir, image_type, 'val')

        # Load train data
        train_dataset = ImageFolderWithPaths(train_dir, transform=transform)
        train_datasets.append(train_dataset)

        # Load val data
        val_dataset = ImageFolderWithPaths(val_dir, transform=transform)
        val_datasets[image_type] = val_dataset

    # Combine all train datasets
    combined_train_dataset = ConcatDataset(train_datasets)
    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True)

    # Create val loaders for each type
    val_loaders = {ai_type: DataLoader(val_datasets[ai_type], batch_size=batch_size, shuffle=False)
                   for ai_type in image_types}

    return train_loader, val_loaders

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
    all_fake_probs = []  # To record probabilities for fake images
    fake_details = []  # List of dicts with path, prob, reasoning
    with torch.no_grad():
        for ai_type, val_loader in val_loaders.items():
            all_preds = []
            all_labels = []
            fake_probs = []
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
                # For predicted fake images (class 1), record prob of fake
                fake_mask = (preds == 1).cpu()
                if fake_mask.any():
                    fake_probs.extend(mean_probs[fake_mask, 1].cpu().numpy())
                    for idx in fake_mask.nonzero(as_tuple=True)[0]:
                        path = paths[idx]
                        prob = mean_probs[idx, 1].item()
                        feature = features[idx]
                        # Compute mean logit for class 1
                        mean_weight = model.classifier.weight_mu[1]
                        mean_bias = model.classifier.bias_mu[1]
                        logit = torch.dot(mean_weight, feature) + mean_bias
                        # Generate reasoning based on the image's specific features
                        # Find top contributing features (dimensions where weight * feature is high)
                        contributions = mean_weight * feature
                        top_contrib_indices = torch.topk(contributions, 5).indices.tolist()
                        reasoning = f"This image has a {prob:.1%} probability of being fake based on its ResNet50 features. The logit for the 'ai' class is {logit:.2f}, with strong contributions from feature dimensions {top_contrib_indices}, suggesting AI-generated characteristics in those aspects."
                        fake_details.append({
                            'path': path,
                            'probability_fake': prob,
                            'reasoning': reasoning
                        })
                        print(f"Image {path}: {reasoning}")
            all_fake_probs.extend(fake_probs)
            report = classification_report(all_labels, all_preds, target_names=['nature', 'ai'], output_dict=True)
            results[ai_type] = report
            print(f'Results for {ai_type}:')
            print(classification_report(all_labels, all_preds, target_names=['nature', 'ai']))
            if fake_probs:
                print(f'Fake probabilities for {ai_type}: mean={sum(fake_probs)/len(fake_probs):.4f}, samples={len(fake_probs)}')
    return results, all_fake_probs, fake_details

def main():
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
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': 10,
    }, 'model_bnn_checkpoint.pth')
    print("Model checkpoint saved to model_bnn_checkpoint.pth")

    # Evaluate and record results
    results, fake_probs, fake_details = evaluate_model(model, val_loaders, device=device)

    # Save results to JSON
    with open('results_bnn.json', 'w') as f:
        json.dump(results, f, indent=4)

    # Save fake probabilities and details
    with open('fake_details.json', 'w') as f:
        json.dump(fake_details, f, indent=4)

    print("Results saved to results_bnn.json")
    print("Fake details saved to fake_details.json")

if __name__ == '__main__':
    main()
