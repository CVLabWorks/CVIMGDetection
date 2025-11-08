# ViT模型训练和结果

## 训练设置
epoch：50
Opt：AMP

---

## 1. `ViTWithCustomHead`

- **骨干网络**：`google/vit-base-patch16-224`
  - Patch size: 16×16  
  - Hidden size (D): **768**  
  - Layers: 12  
  - Attention heads: 12  

- **训练策略**：冻结 ViT 主干（`freeze_backbone=True`）

- **分类头结构**：
Linear(768 → 512) → BatchNorm1d → ReLU → Dropout(0.3)
Linear(512 → 256) → BatchNorm1d → ReLU → Dropout(0.3)
Linear(256 → num_classes)

- **用途**：标准二分类基线（AI vs Real）
- **结果**：

|               | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| Not AI        | 0.8812    | 0.8774 | 0.8793   | 3500    |
| AI            | 0.8577    | 0.8620 | 0.8599   | 3000    |
| **Accuracy**  |           |        | 0.8703   | 6500    |
| **Macro Avg** | 0.8695    | 0.8697 | 0.8696   | 6500    |
| **Weighted Avg** | 0.8704 | 0.8703 | 0.8703   | 6500    |
总体准确率 (Accuracy): 0.8703

---

## 2. `ViTWithInterpolation`

- **骨干网络**：同上 ViT-Base，启用 **位置编码插值**
- 参数：`interpolate_pos_encoding=True`

- **支持输入分辨率**：任意 **16 的倍数**（如 224, 384, 512）

- **其他结构**：与 `ViTWithCustomHead` 完全一致

- **用途**：高分辨率图像训练（例如 384×384）

- **结果**：基本没提升，底层原理不对

---

## 3. `ViTWithLocalPerception`

- **骨干网络**：ViT-Base（冻结）

- **局部感知模块（CNN）**：
Conv2d(3 → 32, k=3, p=1) → BN → ReLU
Conv2d(32 → 64, k=3, p=1) → BN → ReLU
Conv2d(64 → 768, k=1)
→ AdaptiveAvgPool2d → [B, 768]


- **特征融合方式**：可学习加权求和  
- `combined = α * ViT_cls + β * CNN_global`  
- `α`, `β` 为标量可学习参数（初始化为 1）

- **分类头**：同三层 MLP（含 BN 和 Dropout）

- **特点**：融合全局 ViT 特征与全局 CNN 纹理特征

- **结果**：

|               | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| Not AI        | 0.9016    | 0.8611 | 0.8809   | 3500    |
| AI            | 0.8461    | 0.8903 | 0.8676   | 3000    |
| **Accuracy**  |           |        | 0.8746   | 6500    |
| **Macro Avg** | 0.8738    | 0.8757 | 0.8743   | 6500    |
| **Weighted Avg** | 0.8760 | 0.8746 | 0.8748   | 6500    |
总体准确率 (Accuracy): 0.8746

---

## 4. `ViTWithLPMAndRegularizedHead`（两阶段模型）

- **骨干网络**：ViT-Base（可冻结）

- **局部感知模块 LPM**（精细设计）：
Conv2d(3→32, k=3) → BN → ReLU
Conv2d(32→64, k=3) → BN → ReLU
Conv2d(64→768, k=1)
→ AvgPool(kernel=16, stride=16) → [B, 768, H/16, W/16]
→ Flatten → [B, N, 768] （N = (H/16)×(W/16)）

- **融合策略**：
- 仅增强 **patch tokens**（CLS token 保持不变）
- `enhanced_patches = vit_patches + gated(LPM_output)`
- **门控机制**：`gate = nn.Parameter(torch.zeros(...))`，初始抑制 LPM 贡献

- **两阶段训练支持**：
- **阶段1**：多类分类（按 AI 模型类型）
- **阶段2**：二分类 + **KL 蒸馏正则化**
  - 加载旧 head（`old_head_state_dict`）
  - KL loss：温度 T=2.0，权重 α=0.5

- **分类头**：三层 MLP（带 BN 和 Dropout）
- **结果**：

| 类别        | Precision | Recall  | F1-Score | Support |
|-------------|-----------|---------|----------|---------|
| Not AI      | 0.8812    | 0.8774  | 0.8793   | 3500    |
| AI          | 0.8577    | 0.8620  | 0.8599   | 3000    |
| **Accuracy**| —         | —       | **0.8703** | **6500** |
| **Macro Avg**| 0.8695   | 0.8697  | 0.8696   | 6500    |
| **Weighted Avg**| 0.8704| 0.8703  | 0.8703   | 6500    |

总体准确率 (Accuracy): 0.8703

---

## 5. `SwinTransformerWithCustomHead`

- **骨干网络**：`microsoft/swin-tiny-patch4-window7-224`
- Patch size: 4×4（分层下采样）
- Hidden size: **768**
- Stages: 4（block 数 [2, 2, 6, 2]）
- Window size: 7

- **特征聚合**：对最后一层 token 序列做 **mean pooling**（无 CLS token）

- **分类头**：三层 MLP（同上）

- **用途**：引入层次化局部-全局注意力机制

- **结果**：

|               | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| Not AI        | 0.9140    | 0.8443 | 0.8778   | 3500    |
| AI            | 0.8332    | 0.9073 | 0.8687   | 3000    |
| **Accuracy**  |           |        | 0.8734   | 6500    |
| **Macro Avg** | 0.8736    | 0.8758 | 0.8732   | 6500    |
| **Weighted Avg** | 0.8767 | 0.8734 | 0.8736   | 6500    |

总体准确率 (Accuracy): 0.8734

---

## 6. `CvTWithCustomHead`


- **骨干网络**：`microsoft/cvt-13`

- **特征聚合**：**全局平均池化（GAP）** → `[B, 768]`

- **分类头**：三层 MLP（含 BN 和 Dropout）

- **结果**：

|               | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| Not AI        | 0.7980    | 0.9243 | 0.8565   | 3500    |
| AI            | 0.8917    | 0.7270 | 0.8010   | 3000    |
| **Accuracy**  |           |        | 0.8332   | 6500    |
| **Macro Avg** | 0.8448    | 0.8256 | 0.8287   | 6500    |
| **Weighted Avg** | 0.8412 | 0.8332 | 0.8309   | 6500    |

总体准确率 (Accuracy): 0.8332

---

## 共同设计特点

| 组件 | 配置 |
|------|------|
| **默认输入尺寸** | 224×224（`ViTWithInterpolation` 支持更高） |
| **归一化** | 分类头中使用 `BatchNorm1d(affine=False)` 避免单样本问题 |
| **正则化** | Dropout(0.3) 应用于 MLP 层之间 |
| **主干冻结** | 所有模型默认冻结预训练主干 |
| **输出维度** | `num_classes`（通常为 2；阶段1为多类） |

---