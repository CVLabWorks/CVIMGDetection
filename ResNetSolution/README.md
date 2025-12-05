# ResNet50 Solution for AI-Generated Image Detection

This module implements ResNet50-based models for detecting AI-generated images, including baseline ResNet50, Bayesian Neural Network (BNN) variants, and Spatial Attention enhanced architectures.

## Folder Structure

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

## Models

| Model | Description |
|-------|-------------|
| **ResNet50** | Baseline CNN with pre-trained ImageNet weights |
| **ResNet50 + BNN** | Replaces FC layer with Bayesian Linear for uncertainty estimation |
| **ResNet50 + Attention + BNN** | Adds Spatial Attention before BNN for focused feature learning |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

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

### 3. Train & Evaluate

```bash
# Baseline ResNet50
python ResNet50.py

# ResNet50 + BNN
python ResNet50_BNN.py

# ResNet50 + Spatial Attention + BNN
python ResNet50_BNN_with_attetion.py
```

## Key Results

| Model | Avg Accuracy | Real Acc | Fake Acc |
|-------|--------------|----------|----------|
| ResNet50 | 86.3% | ~79% | ~92% |
| ResNet50 + BNN | 81.5% | ~85% | ~81% |
| ResNet50 + Attention + BNN | **87.6%** | ~89% | ~85% |

Through comprehensive ablation experiments on ResNet50 and its variants, we conclude:

| Component | Contribution |
|-----------|--------------|
| **ResNet50** | Robust feature extraction from pre-trained backbone |
| **Spatial Attention** | Focuses on discriminative regions, reducing noise for BNN |
| **BNN** | Provides calibrated uncertainty and balanced predictions |


- **Traditional CNN Limitations**: ResNet50's fixed weights lead to biased fake detection, lacking uncertainty quantification
- **BNN Advantages**: Probabilistic weights provide natural regularization and uncertainty estimation, improving real image detection
- **BNN Challenges**: Variational inference introduces approximation errors; requires more data for optimal performance
- **Attention-BNN Synergy**: Spatial attention compensates for BNN's data inefficiency by providing focused features
