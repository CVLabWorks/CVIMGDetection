# ViT models Result

## Data processing
The dataset is tiny_genimage, and two files in the data directory are responsible for reading and merging the data, ultimately generating mergedata

## Training setup
epoch：50
Opt：AMP

---

## 1. `ViTWithCustomHead`

- **Backbone**：`google/vit-base-patch16-224`
  - Patch size: 16×16  
  - Hidden size (D): **768**  
  - Layers: 12  
  - Attention heads: 12  

- **Training strategy**：freeze ViT backbone（`freeze_backbone=True`）

- **Classifier**：
Linear(768 → 512) → BatchNorm1d → GELU → Dropout(0.3)
Linear(512 → 256) → BatchNorm1d → GeLU → Dropout(0.3)
Linear(256 → num_classes)

- **Result**：

|               | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| Not AI        | 0.8812    | 0.8774 | 0.8793   | 3500    |
| AI            | 0.8577    | 0.8620 | 0.8599   | 3000    |
| **Accuracy**  |           |        | 0.8703   | 6500    |
| **Macro Avg** | 0.8695    | 0.8697 | 0.8696   | 6500    |
| **Weighted Avg** | 0.8704 | 0.8703 | 0.8703   | 6500    |

Overall Accuracy: 0.8703

---


## 2. `ViTWithLocalPerception`

- **Backbone**：ViT-Base

- **Local Perception module（CNN）**：
Conv2d(3 → 32, k=3, p=1) → BN → ReLU
Conv2d(32 → 64, k=3, p=1) → BN → ReLU
Conv2d(64 → 768, k=1)
→ AdaptiveAvgPool2d → [B, 768]


- **Feature Fusion Strategy**：Learnable weighted sum 
- `combined = α * ViT_cls + β * CNN_global`  
- `α`, `β` are scalar learnable parameters

- **Classifier**：same as ViT

- **Key feature**：Fuses global ViT features with global CNN texture features

- **Result**：

|               | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| Not AI        | 0.9016    | 0.8611 | 0.8809   | 3500    |
| AI            | 0.8461    | 0.8903 | 0.8676   | 3000    |
| **Accuracy**  |           |        | 0.8746   | 6500    |
| **Macro Avg** | 0.8738    | 0.8757 | 0.8743   | 6500    |
| **Weighted Avg** | 0.8760 | 0.8746 | 0.8748   | 6500    |

Overall Accuracy: 0.8746

---

## 3. `ViTWithLPMAndRegularizedHead` (Two-Stage Model)

- **Backbone**: ViT-Base

- **Local Perception Module (LPM)**:  
  Conv2d(3→32, k=3) → BN → ReLU  
  Conv2d(32→64, k=3) → BN → ReLU  
  Conv2d(64→768, k=1)  
  → AvgPool(kernel=16, stride=16) → [B, 768, H/16, W/16]  
  → Flatten → [B, N, 768] (**N = (H/16)×(W/16)**)

- **Feature Fusion**:  
  - Only enhances **patch tokens** (**CLS token** remains unchanged)  
  - `enhanced_patches = vit_patches + gated(LPM_output)`  
  - **Gating mechanism**: `gate = nn.Parameter(torch.zeros(...))`, initially suppresses LPM contribution

- **Two-Stage Training Support**:  
  - **Stage 1**: Multi-class classification (**by AI model type**)  
  - **Stage 2**: Binary classification + **KL distillation regularization**  
    - Load old head (**`old_head_state_dict`**)  
    - **KL loss**: temperature **T=2.0**, weight **α=0.5**

- **Classifier**：same
- **Result**：

|             | Precision | Recall  | F1-Score | Support |
|-------------|-----------|---------|----------|---------|
| Not AI      | 0.8812    | 0.8774  | 0.8793   | 3500    |
| AI          | 0.8577    | 0.8620  | 0.8599   | 3000    |
| **Accuracy**| —         | —       | **0.8703** | **6500** |
| **Macro Avg**| 0.8695   | 0.8697  | 0.8696   | 6500    |
| **Weighted Avg**| 0.8704| 0.8703  | 0.8703   | 6500    |

Overall Accuracy: 0.8703

---

## 4. `SwinTransformerWithCustomHead`

- **Backbone**：`microsoft/swin-base-patch4-window7-224`
- Patch size: 4×4
- Hidden size: **768**
- Stages: 4（block [2, 2, 6, 2]）
- Window size: 7

- **Feature Fusion**：Apply **mean pooling** to the final-layer token sequence (without CLS token).

- **Classifier**：same

- **Purpose**：Introduce a hierarchical local-global attention mechanism.

- **Result**：

|               | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| Not AI        | 0.9140    | 0.8443 | 0.8778   | 3500    |
| AI            | 0.8332    | 0.9073 | 0.8687   | 3000    |
| **Accuracy**  |           |        | 0.8734   | 6500    |
| **Macro Avg** | 0.8736    | 0.8758 | 0.8732   | 6500    |
| **Weighted Avg** | 0.8767 | 0.8734 | 0.8736   | 6500    |

Overall Accuracy: 0.8734

---

## 5. `CvTWithCustomHead`


- **Backbone**：`microsoft/cvt-13`

- **Feature Fusion**：**Global average pooling（GAP）** → `[B, 768]`

- **Classifier**：same

- **Result**：

|               | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| Not AI        | 0.7980    | 0.9243 | 0.8565   | 3500    |
| AI            | 0.8917    | 0.7270 | 0.8010   | 3000    |
| **Accuracy**  |           |        | 0.8332   | 6500    |
| **Macro Avg** | 0.8448    | 0.8256 | 0.8287   | 6500    |
| **Weighted Avg** | 0.8412 | 0.8332 | 0.8309   | 6500    |

Overall Accuracy: 0.8332

---

## Common Design Characteristics

| Component | Configuration |
|-----------|---------------|
| **Default Input Size** | 224×224 (`ViTWithInterpolation` supports higher resolutions) |
| **Normalization** | `BatchNorm1d(affine=False)` used in the classification head to avoid issues with single-sample batches |
| **Regularization** | Dropout(0.3) applied between MLP layers |
| **Backbone Freezing** | Pretrained backbone frozen by default in all models |
| **Output Dimension** | `num_classes` (typically 2; multi-class in Stage 1) |

---
