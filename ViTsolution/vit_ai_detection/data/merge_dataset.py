# merge_dataset.py
import pandas as pd
from pathlib import Path


def merge_all_datasets(output_csv="merged_dataset.csv"):
    # 检查是否已存在缓存
    output_path = Path(output_csv)
    if output_path.exists():
        print(f"✅ 已存在合并数据集: {output_csv}")
        df = pd.read_csv(output_path)
        return df

    print("🔄 正在合并所有数据集...")

    # 数据根目录（Linux 上）
    data_root = "/home/mdouab/vit_ai_detection/data/tiny-genimage/versions/1"

    all_data = []
    for dataset_dir in Path(data_root).iterdir():
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name  # e.g., "imagenet_ai_0419_biggan"

        # 遍历 train 和 val
        for split in ["train", "val"]:
            split_dir = dataset_dir / split
            if not split_dir.exists():
                continue

            # 遍历 ai 和 nature 文件夹
            for label_dir in ["ai", "nature"]:
                label_path = split_dir / label_dir
                if not label_path.exists():
                    continue

                # 找到所有图像文件（支持 .JPEG, .jpg, .png）
                image_files = (
                    list(label_path.glob("*.JPEG")) +
                    list(label_path.glob("*.jpg")) +
                    list(label_path.glob("*.png"))
                )

                for img_path in image_files:
                    relative_path = img_path.relative_to(data_root)
                    is_ai = (label_dir == "ai")

                    # 🔧 关键修复：只有 AI 图像才标注模型名，nature 标为 "real"
                    if is_ai:
                        # 从 dataset_name 提取模型名，例如 "biggan", "vqdm", "sdv5"
                        ai_model = dataset_name.split('_')[-1]
                        # 特殊处理 glide 和 midjourney（虽然通常也能正确提取）
                        if ai_model in {"glide", "midjourney"}:
                            ai_model = ai_model
                        # 其他情况保留原逻辑（如 "biggan", "vqdm" 等）
                    else:
                        ai_model = "real"  # 👈 真实图像不归属任何生成模型

                    all_data.append({
                        "image_path": str(relative_path),
                        "split": split,
                        "is_ai": is_ai,
                        "ai_model": ai_model
                    })

    df = pd.DataFrame(all_data)
    df.to_csv(output_path, index=False)
    print(f"✅ 合并完成，共 {len(df)} 条记录")
    print("📊 is_ai 分布:")
    print(df['is_ai'].value_counts())
    print("🔍 ai_model 示例（前5个唯一值）:")
    print(df['ai_model'].unique()[:5])
    return df