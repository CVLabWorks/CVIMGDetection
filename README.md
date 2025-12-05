# AI-Generated Image Detection

This project implements multiple AI-generated image detection models, including Vision Transformer (ViT), ResNet50, SPAI, and a lightweight detector based on Neighboring Pixel Relationships (NPR). The models are trained and evaluated on the Tiny-GenImage dataset, with the enhanced NPR model selected as the final solution.

## Repository Structure

| Folder | Description |
|--------|-------------|
| **ResNetSolution** | Contains ResNet50-based implementations including standard ResNet50, Bayesian neural network variants, and spatial attention-enhanced models for AI-generated image detection. |
| **ViTsolution/vit_ai_detection** | Houses Vision Transformer implementations including standard ViT, ViT+CNN hybrids, Swin Transformer, and CvT models with various configurations for image forgery detection. |
| **SPAI** | Contains the Spectral Attention implementation that uses frequency-domain analysis and self-supervised learning to detect AI-generated images through spectral reconstruction similarity. |
| **NPR** | Includes the final NPR solution with multi-scale fusion, channel attention, and spatial attention mechanisms for detecting upsampling artifacts in synthetic images. |

More detailed instructions are available under each directory.

## Performance

The final NPR model achieves the following results on the Tiny-GenImage test set:

* **Overall Accuracy: 97.06%**
* **Average Precision (AP): 99.62%**
* **AI-generated Image Detection Accuracy: 98.09%**

For detailed performance across different generators, see Table 8 in the report.
