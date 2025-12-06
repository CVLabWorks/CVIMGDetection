import torch
from models.model import ViTWithLocalPerception
from config import Config

print("Testing new model...")
model = ViTWithLocalPerception(num_classes=2, freeze_backbone=True)
print("Model created successfully!")

x = torch.randn(2, 3, 224, 224) 
model.eval() 
with torch.no_grad():
    output = model(x)
print(f"Model output shape: {output.shape}")
print("✅ Model test passed!")
