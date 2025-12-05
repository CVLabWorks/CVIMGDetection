#!/usr/bin/env python3
"""
Generate CSV file for SPAI training/validation from custom dataset.
"""
import csv
from pathlib import Path
import glob

# 数据集根目录
DATASET_ROOT = Path("/home/error/code/spai/dataset/mini_gen")

# 所有数据源
DATASETS = [
    "imagenet_ai_0419_biggan",
    "imagenet_ai_0419_vqdm",
    "imagenet_ai_0424_sdv5",
    "imagenet_ai_0424_wukong",
    "imagenet_ai_0508_adm",
    "imagenet_glide",
    "imagenet_midjourney"
]

def scan_directory(directory, class_label, split_label):
    """
    扫描目录并返回图像条目列表

    Args:
        directory: 要扫描的目录路径
        class_label: 类别标签 (0=真实, 1=AI生成)
        split_label: 数据集划分 ('train' 或 'val')

    Returns:
        包含图像信息的字典列表
    """
    entries = []

    # 支持的图像格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPEG', '*.JPG', '*.PNG', '*.BMP']

    directory_path = Path(directory)
    if not directory_path.exists():
        print(f"⚠️  目录不存在，跳过: {directory}")
        return entries

    # 扫描所有支持的图像格式
    for ext in image_extensions:
        for img_path in directory_path.rglob(ext):
            if img_path.is_file():
                entries.append({
                    'image': str(img_path),
                    'class': class_label,
                    'split': split_label
                })

    return entries

def main():
    """主函数：生成CSV文件"""
    all_entries = []

    print("🔍 开始扫描数据集...")
    print(f"📁 数据集根目录: {DATASET_ROOT}")
    print("-" * 60)

    # 遍历所有数据源
    for dataset_name in DATASETS:
        dataset_path = DATASET_ROOT / dataset_name

        if not dataset_path.exists():
            print(f"⚠️  数据集不存在，跳过: {dataset_name}")
            continue

        print(f"\n处理数据集: {dataset_name}")

        # 训练集 - AI图像
        train_ai_path = dataset_path / "train" / "ai"
        ai_train_entries = scan_directory(train_ai_path, class_label=1, split_label='train')
        all_entries.extend(ai_train_entries)
        print(f"  ✓ 训练集 AI图像: {len(ai_train_entries)} 张")

        # 训练集 - 真实图像
        train_nature_path = dataset_path / "train" / "nature"
        nature_train_entries = scan_directory(train_nature_path, class_label=0, split_label='train')
        all_entries.extend(nature_train_entries)
        print(f"  ✓ 训练集 真实图像: {len(nature_train_entries)} 张")

        # 验证集 - AI图像
        val_ai_path = dataset_path / "val" / "ai"
        ai_val_entries = scan_directory(val_ai_path, class_label=1, split_label='val')
        all_entries.extend(ai_val_entries)
        print(f"  ✓ 验证集 AI图像: {len(ai_val_entries)} 张")

        # 验证集 - 真实图像
        val_nature_path = dataset_path / "val" / "nature"
        nature_val_entries = scan_directory(val_nature_path, class_label=0, split_label='val')
        all_entries.extend(nature_val_entries)
        print(f"  ✓ 验证集 真实图像: {len(nature_val_entries)} 张")

    print("\n" + "=" * 60)

    if not all_entries:
        print("❌ 错误: 没有找到任何图像文件！")
        return

    # 写入CSV文件
    output_csv = Path('/home/error/code/spai/dataset/my_dataset.csv')
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'class', 'split'])
        writer.writeheader()
        writer.writerows(all_entries)

    # 统计信息
    train_count = sum(1 for e in all_entries if e['split'] == 'train')
    val_count = sum(1 for e in all_entries if e['split'] == 'val')
    ai_count = sum(1 for e in all_entries if e['class'] == 1)
    nature_count = sum(1 for e in all_entries if e['class'] == 0)

    print(f"\n✅ CSV文件已成功创建!")
    print(f"📄 文件路径: {output_csv}")
    print(f"\n📊 数据集统计:")
    print(f"  • 总图像数: {len(all_entries):,} 张")
    print(f"  • 训练集: {train_count:,} 张")
    print(f"  • 验证集: {val_count:,} 张")
    print(f"  • AI生成图像: {ai_count:,} 张")
    print(f"  • 真实图像: {nature_count:,} 张")
    print(f"\n💡 现在你可以使用以下命令进行训练:")
    print(f"   python -m spai train --cfg ./configs/spai.yaml \\")
    print(f"     --data-path {output_csv} \\")
    print(f"     --pretrained ./weights/mfm_pretrain_vit_base.pth \\")
    print(f"     --output ./output/train \\")
    print(f"     --tag my_training")
    print()

if __name__ == "__main__":
    main()
