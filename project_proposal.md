


# Project Name
**SynthGuard: Universal Fake Image Detection System**

## 1.1 Project Background introduction
Recent advances in generative AI (StyleGAN, Stable Diffusion, DALL-E 3, Midjourney) have enabled creation of photorealistic synthetic images indistinguishable to human eyes. This poses critical challenges like:

- **Misinformation & Deepfakes**: AI-generated fake news images spreading across social media
- **Digital Evidence Integrity**: Authenticity verification in journalism and legal proceedings
- **Content Moderation**: Automated detection of synthetic content on platforms
- **Intellectual Property**: Distinguishing original artwork from AI-generated copies

Those definetely influence the public security and justice. So the detection between real information and the fake one becoming an ergency.

# Team Information

| Member  | Primary Responsibility |
|--------|---------------------|
|----|----|
|----|----|
|----|----|
| ZHANG, Shengqi | contribute to CNN solutions|


## Methodiligies


### 1.1  Convolutional Neural Networks (CNNs)

CNNs defenitely revolutionized the computer vision field since Their hierarchical feature extraction naturally aligns with image detection tasks by progressively learning from pixel-level patterns to high-level semantics, which is a natural fit for image detection tasks.

A well-performing CNN model can achieve hierarchical feature extraction while also being translation-invariant, making it highly sensitive to image microstructure. This allows it to effectively capture edges, texture, color distribution, local structure, and semantic information. Generated images, on the other hand, are prone to anomalies in details such as edges, lighting, color, and local texture; these differences can be used to distinguish generated images from real ones.

Also, CNN models have the benefits as below:

- **Efficiency**: Fast inference (10-15ms), low memory, deployable on edge devices
- **Reliability**: Proven track record, stable training, achieves 92-94% baseline accuracy
- **Simplicity**: Easy implementation, extensive pre-trained models, quick prototyping

[Sehngqi comments: After all possible tech proposal, we need to have a decision for the primary choice. ]

## Project Timeline

### Week 1: Dataset Construction (Oct 19 - Oct 26)
- Download ImageNet-1K validation set (50K real images)
- Generate synthetic images using 5 generative models (80K each = 400K total)
- Perform train (80%) / validation (10%) / test (10%) splits
- Prepare unseen generator test set (DALL-E 3, Midjourney, Firefly)

### Week 2-3: Baseline Implementation (Oct 27 - Nov 12)
- Implement ResNet-50 baseline for comparison
- Establish data loading pipeline with augmentations
- Initial training and validation (target: 93% accuracy)


### Week 4: Robustness Testing & Evaluation (Nov 13- Nov 20)
- Test on held-out test set (target: 95% accuracy)
- Evaluate on unseen generators (target: 88% accuracy)
- Robustness testing: JPEG compression, Gaussian noise, crops
- Grad-CAM visualization for interpretability

### Week 5: Report & Presentation
- Write 6-page CVPR-style technical report
- Prepare 15-minute presentation with demo
- Final code cleanup and documentation
- Submit deliverables


