import kagglehub

# Download latest version
path = kagglehub.dataset_download("yangsangtai/tiny-genimage")

print("Path to dataset files:", path)
