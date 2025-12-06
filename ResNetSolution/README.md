# ResNet50 Solution for AI-Generated Image Detection

This module implements ResNet50-based models for detecting AI-generated images, including baseline ResNet50, Bayesian Neural Network (BNN) variants, and Spatial Attention enhanced architectures.

## 1. Folder Structure

```
ResNetSolution/
├── datasets/                 # Dataset directory (not included in repo)
├── imgs/                     # Output images and visualizations
├── results/                  # Training logs and evaluation results
│   ├── results_resnet.json
│   ├── results_bnn.json
│   └── results_bnn_attention.json
├── ResNet50.py               # Baseline ResNet50 classifier
├── ResNet50_BNN.py           # ResNet50 + Bayesian Linear layer
├── ResNet50_BNN_with_attetion.py  # ResNet50 + Spatial Attention + BNN
├── utils.py                  # Data loading and utility functions
├── plot_individual_models.ipynb   # Visualization notebook
└── requirements.txt          # Python dependencies
```

## 2. Models Overview

| Model | Description |
|-------|-------------|
| **ResNet50** | Baseline CNN with pre-trained ImageNet weights |
| **ResNet50 + BNN** | Replaces FC layer with Bayesian Linear for uncertainty estimation |
| **ResNet50 + Attention + BNN** | Adds Spatial Attention before BNN for focused feature learning |

## 3. Getting Started

### 3.1 Install Dependencies

```bash
pip install -r requirements.txt
```

### 3.2 Prepare Dataset

Place dataset in `datasets/` directory with the following structure:
```
datasets/
├── train/
│   ├── nature/
│   └── ai/
└── val/
    ├── nature/
    └── ai/
```

### 3.3 Train & Evaluate

```bash
# Baseline ResNet50
python ResNet50.py

# ResNet50 + BNN
python ResNet50_BNN.py

# ResNet50 + Spatial Attention + BNN
python ResNet50_BNN_with_attetion.py
```

## 4. Technical Report

### 4.1 ResNet50 Baseline

ResNet50 extracts hierarchical features from low-level edges to high-level semantic patterns. Its skip connections effectively overcome the degradation problem in deep networks. Thus, it serves as an ideal CNN baseline for real/fake image detection. It demonstrates excellent fake image detection (>90% on most datasets) but moderate real image recall (~79-81%).

### 4.2 Enhanced Variants

We investigate the baseline ResNet50 model, introduce Bayesian Neural Network (BNN) modifications for uncertainty inference, and propose a hybrid architecture combining Spatial Attention with BNN layers.

#### 4.2.1 ResNet50 + Bayesian Linear Classifier

Traditional CNNs use fixed weights, limiting their ability to express uncertainty and may overfit on small datasets. By replacing the fully connected classifier with a Bayesian Linear layer, it may achieve better generalization.

**Architecture:**

| Stage | Component |
|-------|-----------|
| 1 | Input Image |
| 2 | ResNet50 Backbone |
| 3 | Global Avg Pool |
| 4 | Bayesian Classifier (Real/Fake) |

**Key Components:**
- **Variational Parameters**: Weights modeled as Gaussian distributions W ~ N(μ, σ²)
- **KL Divergence**: Regularization between posterior and prior
- **ELBO Loss**: Reconstruction Loss + KL Divergence
- **Monte Carlo Sampling**: Approximate posterior for inference

**Observation:** BNN exhibits *inverted behavior*—very high real image recall (>95% on some datasets) but lower fake detection. Overall accuracy slightly decreases compared to ResNet50.

**Analysis:**
- **Uncertainty-Aware Learning**: BNN's probabilistic weights are conservative, preferring to classify uncertain samples as "real"
- **Generalization Trade-off**: Reduced overfitting to fake artifacts comes at the cost of fake detection accuracy
- **Data Efficiency**: BNN requires more training data to learn discriminative distributions effectively

#### 4.2.2 ResNet50 + Spatial Attention + Bayesian Linear Classifier

**Architecture:**

| Stage | Component |
|-------|-----------|
| 1 | Input Image |
| 2 | ResNet50 Backbone |
| 3 | Spatial Attention |
| 4 | Avg Pool |
| 5 | Bayesian Classifier(Real/Fake)  |

**Spatial Attention Module:**
- **Compress**: Reduce 2048 channels into 2 spatial maps via Average and Max pooling
- **Learn**: Apply 7×7 convolution to generate spatial importance mask
- **Focus**: Multiply original features by attention mask to highlight artifact regions

**Result:** By integrating Spatial Attention, the model achieves the highest overall accuracy (~87.6%) among all variants. It maintains strong real image detection (>90% on most datasets) inherited from BNN, while significantly improving fake detection compared to BNN alone.

### 4.3 Results Summary

| Model | Avg Accuracy | Real Detection | Fake Detection | Uncertainty |
|-------|--------------|----------------|----------------|-------------|
| ResNet50 | 86.3% | ~79% | ~92% | ✗ |
| ResNet50+BNN | 81.5% | ~85% | ~81% | ✓ |
| **ResNet50+Attention+BNN** | **87.6%** | **~89%** | **~85%** | **✓** |

### 4.4 Conclusions

| Component | Contribution |
|-----------|--------------|
| **ResNet50** | Robust feature extraction from pre-trained backbone |
| **Spatial Attention** | Focuses on discriminative regions, reducing noise for BNN |
| **BNN** | Provides calibrated uncertainty and balanced predictions |

**Key Findings:**
- **Traditional CNN Limitations**: ResNet50's fixed weights lead to biased fake detection, lacking uncertainty quantification
- **BNN Advantages**: Probabilistic weights provide natural regularization and uncertainty estimation, improving real image detection
- **BNN Challenges**: Variational inference introduces approximation errors; requires more data for optimal performance
- **Attention-BNN Synergy**: Spatial attention compensates for BNN's data inefficiency by providing focused features
