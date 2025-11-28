# Neighboring Pixel Relationships (NPR) for AI-Generated Image Detection

A novel deepfake detection approach that analyzes upsampling artifacts in AI-generated images through neighboring pixel relationships.

## 📖 Overview

This method implements the NPR (Neighboring Pixel Relationships) method for detecting AI-generated images by analyzing the characteristic artifacts left by upsampling operations in generative model pipelines. The method achieves state-of-the-art performance across multiple generative models.

For more details, please visit [chuangchuangtan/NPR-DeepfakeDetection](https://github.com/chuangchuangtan/NPR-DeepfakeDetection)

## 🚀 Key Features

- **Multi-scale NPR Analysis**: Captures upsampling artifacts at different scales
- **Channel Attention Mechanism**: Adaptively focuses on discriminative color channels
- **Spatial Attention**: Identifies regions with prominent artifacts
- **ResNet50 Backbone**: Leverages pre-trained features for robust detection
- **Cross-Model Generalization**: Effective across 7 different generative models

## 📊 Performance

### Final Model Results (Average Across 7 Classes)

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **97.06%** |
| **Average Precision** | **99.62%** |
| **Real Image Accuracy** | 96.03% |
| **Fake Image Accuracy** | **98.09%** |

### Detailed Performance by Generator

| Generator | Accuracy | AP | Real Acc | Fake Acc |
|-----------|----------|-----|----------|-----------|
| ADM | 97.20% | 99.78% | 95.80% | 98.60% |
| BigGAN | 96.80% | 99.05% | 96.40% | 97.20% |
| GLIDE | 97.40% | 99.87% | 95.20% | 99.60% |
| Midjourney | 95.90% | 99.26% | 97.00% | 94.80% |
| SDv5 | 97.40% | 99.83% | 95.00% | 99.80% |
| VQDM | 98.00% | 99.91% | 96.40% | 99.60% |
| Wukong | 96.70% | 99.68% | 96.40% | 97.00% |

## 🛠️ Installation

### Requirements

```bash
pip install -r requirements.txt
pip install gdown==4.7.1
pip install tensorboardX
pip install transforms
```

## 📁Dataset Structure

Organize your dataset in the following structure:

<pre>
./genimage/
├── adm/
│   ├── nature/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── ai/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
├── biggan/
│   ├── nature/
│   └── ai/
├── glide/
│   ├── nature/
│   └── ai/
└── ... (other classes)
</pre>

## 🏃‍♂️ Usage

**Training:**

```bash
python train.py --model_path ./NPR.pth
```

Key training parameters (modify in `./options/`):

* Architecture: ResNet50
* Batch size: 64
* Learning rate: 0.0002
* Classes: 7 generative models
* Input size: 224×224

**Validation**:

```bash
python val.py --model_path ./checkpoints/final/model_epoch_last.pth
```

## 🔧 Configuration

* Model modifications: `./networks/resnet.py`
* Training options: `./options/`
* Hyperparameters: Learning rate scheduling, data augmentation, etc.





