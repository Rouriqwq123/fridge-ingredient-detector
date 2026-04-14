"""
合并多个数据集
支持任意数量的数据集合并

Usage: python src/merge_datasets.py
"""

import shutil
import yaml
import random
from pathlib import Path

# ============================================================
# 配置：在这里添加/删除数据集
# ============================================================
DATASETS = [
    {
        "path": Path("data/raw/fridge-object"),
        "prefix": "fridge_",
        "desc": "主数据集 (Fridge Object)",
    },
    {
        "path": Path("data/raw/custom"),
        "prefix": "custom_",
        "desc": "自定义标注 (中式食材)",
        "train_only": True,
    },
    {
        "path": Path("data/raw/ingredients-fixed"),
        "prefix": "ing_",
        "desc": "食材检测数据集 (Ingredients YoloV8)",
    },
]

OUTPUT_DIR = Path("data/processed/merged")
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ============================================================
# Step 1: 合并所有数据集的类别列表
# ============================================================
def merge_all_classes():
    print("=" * 55)
    print("📋 合并类别列表")
    print("=" * 55)

    all_classes_per_dataset = []
    merged_classes = []

    for ds in DATASETS:
        if not ds["path"].exists():
            print(f"  ⚠️  跳过 (路径不存在): {ds['path']}")
            all_classes_per_dataset.append([])
            continue

        yaml_path = ds["path"] / "data.yaml"
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        classes = config["names"]
        all_classes_per_dataset.append(classes)

        new = [c for c in classes if c not in merged_classes]
        merged_classes += new
        print(f"  {ds['desc']}: {len(classes)} 类，新增 {len(new)} 类")

    print(f"\n  合并后总类别数: {len(merged_classes)}")
    print()
    return merged_classes, all_classes_per_dataset

# ============================================================
# Step 2: 创建输出目录
# ============================================================
def create_output_dirs():
    for split in ["train", "valid", "test"]:
        (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
    print(f"  ✅ 输出目录: {OUTPUT_DIR}")

# ============================================================
# Step 3: 复制单个数据集（带类别重映射）
# ============================================================
def copy_dataset(src_dir, split, merged_classes, src_classes, prefix):
    img_src = src_dir / split / "images"
    lbl_src = src_dir / split / "labels"

    if not img_src.exists():
        return 0

    img_files = (list(img_src.glob("*.jpg")) +
                 list(img_src.glob("*.jpeg")) +
                 list(img_src.glob("*.png")))

    count = 0
    for img_path in img_files:
        new_name = prefix + img_path.name
        dst_img = OUTPUT_DIR / split / "images" / new_name
        dst_lbl = OUTPUT_DIR / split / "labels" / (new_name.rsplit(".", 1)[0] + ".txt")

        shutil.copy2(img_path, dst_img)

        lbl_path = lbl_src / (img_path.stem + ".txt")
        if lbl_path.exists():
            with open(lbl_path) as f:
                lines = f.readlines()
            new_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                old_id = int(parts[0])
                if old_id < len(src_classes):
                    cls_name = src_classes[old_id]
                    new_id = merged_classes.index(cls_name)
                    new_lines.append(f"{new_id} {' '.join(parts[1:])}\n")
            with open(dst_lbl, "w") as f:
                f.writelines(new_lines)
        else:
            open(dst_lbl, "w").close()

        count += 1
    return count

# ============================================================
# Step 4: 处理只有 train 的数据集（自动分割 valid）
# ============================================================
def split_and_copy(src_dir, merged_classes, src_classes, prefix, val_ratio=0.2):
    img_dir = src_dir / "train" / "images"
    lbl_dir = src_dir / "train" / "labels"

    if not img_dir.exists():
        return 0, 0

    img_files = (list(img_dir.glob("*.jpg")) +
                 list(img_dir.glob("*.jpeg")) +
                 list(img_dir.glob("*.png")))
    random.shuffle(img_files)

    n_val = max(1, int(len(img_files) * val_ratio))
    val_files = img_files[:n_val]
    train_files = img_files[n_val:]

    def _copy(files, split):
        for img_path in files:
            new_name = prefix + img_path.name
            shutil.copy2(img_path, OUTPUT_DIR / split / "images" / new_name)
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            dst_lbl = OUTPUT_DIR / split / "labels" / (new_name.rsplit(".", 1)[0] + ".txt")
            if lbl_path.exists():
                with open(lbl_path) as f:
                    lines = f.readlines()
                new_lines = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    old_id = int(parts[0])
                    if old_id < len(src_classes):
                        cls_name = src_classes[old_id]
                        new_id = merged_classes.index(cls_name)
                        new_lines.append(f"{new_id} {' '.join(parts[1:])}\n")
                with open(dst_lbl, "w") as f:
                    f.writelines(new_lines)
            else:
                open(dst_lbl, "w").close()

    _copy(train_files, "train")
    _copy(val_files, "valid")
    return len(train_files), len(val_files)

# ============================================================
# Step 5: 生成 data.yaml
# ============================================================
def write_yaml(merged_classes):
    config = {
        "path": str(OUTPUT_DIR.resolve()),
        "train": "train/images",
        "val":   "valid/images",
        "test":  "test/images",
        "nc":    len(merged_classes),
        "names": merged_classes,
    }
    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    print(f"  ✅ data.yaml: {yaml_path}")
    return yaml_path

# ============================================================
# Step 6: 统计结果
# ============================================================
def print_summary():
    print()
    print("=" * 55)
    print("📊 合并结果统计")
    print("=" * 55)
    total = 0
    for split in ["train", "valid", "test"]:
        count = len(list((OUTPUT_DIR / split / "images").glob("*")))
        total += count
        print(f"  {split:6s}: {count:6d} 张")
    print(f"  {'总计':6s}: {total:6d} 张")

# ============================================================
# Main
# ============================================================
def main():
    print()
    print("🧊 Fridge AI - 合并数据集 (多数据集版)")
    print()

    if OUTPUT_DIR.exists():
        print("  清空旧输出目录...")
        shutil.rmtree(OUTPUT_DIR)

    merged_classes, all_classes = merge_all_classes()
    create_output_dirs()

    print("=" * 55)
    print("📁 复制数据...")
    print("=" * 55)

    for ds, src_classes in zip(DATASETS, all_classes):
        if not ds["path"].exists() or not src_classes:
            continue

        if ds.get("train_only"):
            n_train, n_val = split_and_copy(
                ds["path"], merged_classes, src_classes, ds["prefix"])
            print(f"  {ds['desc']}: train={n_train}, valid={n_val}")
        else:
            for split in ["train", "valid", "test"]:
                n = copy_dataset(
                    ds["path"], split, merged_classes, src_classes, ds["prefix"])
                print(f"  {ds['desc']} {split}: {n} 张")

    print()
    print("=" * 55)
    print("📝 生成配置文件...")
    print("=" * 55)
    write_yaml(merged_classes)
    print_summary()

    print()
    print("=" * 55)
    print("✅ 合并完成！")
    print("=" * 55)
    print(f"  类别数: {len(merged_classes)}")
    print(f"  配置:   {OUTPUT_DIR / 'data.yaml'}")
    print()
    print("下一步: 重新训练")
    print("  python src/train.py")

if __name__ == "__main__":
    main()
