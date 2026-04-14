"""
修复 ingredients-detection 数据集
1. 修复以 '-' 开头的文件名
2. 清理 data.yaml 里的垃圾类别，只保留真实食材

Usage: python src/fix_ingredients_dataset.py
"""

import shutil
import yaml
import os
from pathlib import Path

SRC_DIR = Path("data/raw/ingredients-detection")
DST_DIR = Path("data/raw/ingredients-fixed")

# 只保留这些真实食材类别（过滤掉垃圾）
VALID_CLASSES = {
    'anchovy', 'annona', 'apple', 'artichoke', 'avocado', 'banana',
    'bay_leaf', 'beet', 'bell pepper', 'bell_pepper', 'blueberry',
    'broccoli', 'cabbage', 'carrot', 'cauliflower', 'cherry', 'chicken',
    'chicken breast', 'chickpeas', 'coriander', 'cranberry', 'cucumber',
    'egg', 'eggplant', 'fish', 'garlic', 'ginger', 'gooseberry', 'grape',
    'green chilli pepper', 'guava', 'kumquat', 'leek', 'lemon', 'lettuce',
    'long_pepper', 'mince', 'mulberry', 'mutton', 'okra', 'onion', 'orange',
    'papaya', 'parsley', 'pear', 'pineapple', 'pitaya', 'pork', 'potato',
    'pumpkin', 'radish', 'raspberry', 'rice', 'shrimp', 'spring_onion',
    'strawberry', 'tofu', 'tomato', 'white beans', 'white button mushroom',
    'zucchini',
    # 西班牙语类别映射
    'Carne', 'Cebolla', 'Mango', 'Papa', 'Pollo', 'Tomate', 'Verde',
}

def fix_filename(name: str) -> str:
    """修复以 '-' 或特殊字符开头的文件名"""
    name = name.strip()
    # 替换文件名开头的 '-'
    while name.startswith('-'):
        name = 'img' + name
    # 替换路径不允许的字符
    for ch in ['<', '>', ':', '"', '/', '\\', '|', '?', '*']:
        name = name.replace(ch, '_')
    return name

def get_valid_class_mapping(original_classes):
    """
    返回：
    - old_id → new_id 的映射（只包含有效食材）
    - 新的类别列表
    """
    new_classes = []
    old_to_new = {}

    for old_id, cls_name in enumerate(original_classes):
        if cls_name in VALID_CLASSES:
            new_id = len(new_classes)
            new_classes.append(cls_name)
            old_to_new[old_id] = new_id

    return old_to_new, new_classes

def process_split(split, old_to_new, src_dir, dst_dir):
    src_img = src_dir / split / "images"
    src_lbl = src_dir / split / "labels"
    dst_img = dst_dir / split / "images"
    dst_lbl = dst_dir / split / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    if not src_img.exists():
        return 0

    img_files = (list(src_img.glob("*.jpg")) +
                 list(src_img.glob("*.jpeg")) +
                 list(src_img.glob("*.png")))

    count = 0
    skipped = 0
    for img_path in img_files:
        new_img_name = fix_filename(img_path.name)
        dst_img_path = dst_img / new_img_name

        # 复制图片
        try:
            shutil.copy2(img_path, dst_img_path)
        except Exception as e:
            skipped += 1
            continue

        # 处理标注
        lbl_path = src_lbl / (img_path.stem + ".txt")
        new_lbl_name = fix_filename(img_path.stem) + ".txt"
        dst_lbl_path = dst_lbl / new_lbl_name

        if lbl_path.exists():
            try:
                with open(lbl_path) as f:
                    lines = f.readlines()
            except Exception:
                open(dst_lbl_path, "w").close()
                count += 1
                continue

            new_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                old_id = int(parts[0])
                if old_id in old_to_new:
                    new_id = old_to_new[old_id]
                    new_lines.append(f"{new_id} {' '.join(parts[1:])}\n")
                # 跳过垃圾类别的标注

            with open(dst_lbl_path, "w") as f:
                f.writelines(new_lines)
        else:
            open(dst_lbl_path, "w").close()

        count += 1

    return count

def main():
    print()
    print("🔧 修复 ingredients-detection 数据集")
    print()

    # 读取原始类别
    with open(SRC_DIR / "data.yaml") as f:
        config = yaml.safe_load(f)
    original_classes = config["names"]

    print(f"  原始类别数: {len(original_classes)}")

    # 生成类别映射
    old_to_new, new_classes = get_valid_class_mapping(original_classes)
    print(f"  有效食材类别: {len(new_classes)}")
    print(f"  过滤掉垃圾类别: {len(original_classes) - len(new_classes)} 个")
    print()
    print(f"  保留的类别: {new_classes[:20]}...")
    print()

    # 清空输出目录
    if DST_DIR.exists():
        shutil.rmtree(DST_DIR)

    # 处理每个 split
    print("=" * 55)
    print("📁 处理数据...")
    print("=" * 55)
    total = 0
    for split in ["train", "valid", "test"]:
        n = process_split(split, old_to_new, SRC_DIR, DST_DIR)
        print(f"  {split:6s}: {n} 张")
        total += n

    # 写新的 data.yaml
    new_config = {
        "train": "train/images",
        "val":   "valid/images",
        "test":  "test/images",
        "nc":    len(new_classes),
        "names": new_classes,
    }
    with open(DST_DIR / "data.yaml", "w") as f:
        yaml.dump(new_config, f, allow_unicode=True, default_flow_style=False)

    print()
    print("=" * 55)
    print("✅ 修复完成！")
    print("=" * 55)
    print(f"  输出目录: {DST_DIR}")
    print(f"  总图片数: {total}")
    print(f"  类别数:   {len(new_classes)}")
    print()
    print("下一步: 修改 merge_datasets.py 里的路径")
    print("  把 ingredients-detection 改成 ingredients-fixed")

if __name__ == "__main__":
    main()
