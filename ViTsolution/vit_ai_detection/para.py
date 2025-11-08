from transformers import ViTModel, ViTConfig

# 下载并保存到本地目录
model = ViTModel.from_pretrained("google/vit-base-patch16-224")
config = ViTConfig.from_pretrained("google/vit-base-patch16-224")

save_dir = "./vit-base-patch16-224-local"
model.save_pretrained(save_dir)
config.save_pretrained(save_dir)

print(f"✅ 模型已保存到: {save_dir}")