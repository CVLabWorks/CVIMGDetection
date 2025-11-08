# test_run.py
import torch
from models.model import ViTWithLocalPerception
from config import Config

print("Testing new model...")
model = ViTWithLocalPerception(num_classes=2, freeze_backbone=True)
print("Model created successfully!")

# 创建模拟输入（使用批量大小>1来避免BatchNorm问题）
x = torch.randn(2, 3, 224, 224)  # 批量大小为2
model.eval()  # 设置为评估模式
with torch.no_grad():
    output = model(x)
print(f"Model output shape: {output.shape}")
print("✅ Model test passed!")