import pandas as pd
from pathlib import Path


def merge_all_datasets(output_csv="merged_dataset.csv"):
    output_path = Path(output_csv)
    if output_path.exists():
        print(f" 已存在合并数据集: {output_csv}")
        df = pd.read_csv(output_path)
        return df

    print(" 正在合并所有数据集...")

    # 数据根目录
    data_root = "/home/mdouab/vit_ai_detection/data/tiny-genimage/versions/1"

    all_data = []
    for dataset_dir in Path(data_root).iterdir():
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name  # e.g., "imagenet_ai_0419_biggan"

        for split in ["train", "val"]:
            split_dir = dataset_dir / split
            if not split_dir.exists():
                continue

            for label_dir in ["ai", "nature"]:
                label_path = split_dir / label_dir
                if not label_path.exists():
                    continue

                image_files = (
                    list(label_path.glob("*.JPEG")) +
                    list(label_path.glob("*.jpg")) +
                    list(label_path.glob("*.png"))
                )

                for img_path in image_files:
                    relative_path = img_path.relative_to(data_root)
                    is_ai = (label_dir == "ai")

                    if is_ai:
                        ai_model = dataset_name.split('_')[-1]
                        if ai_model in {"glide", "midjourney"}:
                            ai_model = ai_model
                    else:
                        ai_model = "real"  

                    all_data.append({
                        "image_path": str(relative_path),
                        "split": split,
                        "is_ai": is_ai,
                        "ai_model": ai_model
                    })

    df = pd.DataFrame(all_data)
    df.to_csv(output_path, index=False)
    print(f"合并完成，共 {len(df)} 条记录")
    print(" is_ai 分布:")
    print(df['is_ai'].value_counts())
    print("ai_model 示例:")
    print(df['ai_model'].unique()[:5])
    return df
