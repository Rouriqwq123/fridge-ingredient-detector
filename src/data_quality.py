"""
数据质量分析 + 清理脚本
1. 统计每个类别的标注数量
2. 找出需要补数据的类别（< 100个）
3. 删除稀有类别（< 20个），重建干净的数据集

Usage: python src/data_quality.py
"""

import yaml
import shutil
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR   = Path("data/processed/merged")
OUTPUT_DIR = Path("data/processed/merged_clean")
REPORT_DIR = Path("outputs/data_quality")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MIN_KEEP   = 20    # 少于这个数的类别直接删除
MIN_GOOD   = 100   # 少于这个数的类别需要补数据


# ============================================================
# Step 1: 统计每个类别的标注数量
# ============================================================
def count_annotations():
    with open(DATA_DIR / "data.yaml") as f:
        config = yaml.safe_load(f)
    class_names = config["names"]

    counts = defaultdict(int)
    for split in ["train", "valid", "test"]:
        label_dir = DATA_DIR / split / "labels"
        if not label_dir.exists():
            continue
        for lbl_file in label_dir.glob("*.txt"):
            with open(lbl_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        cls_id = int(line.split()[0])
                        if cls_id < len(class_names):
                            counts[class_names[cls_id]] += 1

    return dict(counts), class_names


# ============================================================
# Step 2: 分析报告
# ============================================================
def print_report(counts, class_names):
    print("=" * 60)
    print("📊 数据质量报告")
    print("=" * 60)

    sorted_counts = sorted(counts.items(), key=lambda x: x[1])
    total_classes = len(class_names)
    total_boxes   = sum(counts.values())

    # 分类统计
    delete_list = [(n, c) for n, c in sorted_counts if c < MIN_KEEP]
    warn_list   = [(n, c) for n, c in sorted_counts if MIN_KEEP <= c < MIN_GOOD]
    good_list   = [(n, c) for n, c in sorted_counts if c >= MIN_GOOD]

    print(f"\n  总类别数: {total_classes}")
    print(f"  总标注框: {total_boxes:,}")
    print()

    print(f"  ❌ 需要删除 (< {MIN_KEEP} 个标注): {len(delete_list)} 个类别")
    for name, count in delete_list:
        print(f"     {name:<30} {count:>5} 个")

    print()
    print(f"  ⚠️  需要补数据 ({MIN_KEEP}-{MIN_GOOD} 个标注): {len(warn_list)} 个类别")
    for name, count in warn_list:
        bar = "█" * int(count / 5)
        print(f"     {name:<30} {count:>5} 个  {bar}")

    print()
    print(f"  ✅ 数据充足 (>= {MIN_GOOD} 个标注): {len(good_list)} 个类别")

    print()
    print("=" * 60)
    print(f"  清理后保留类别: {len(good_list) + len(warn_list)} 个")
    print(f"  删除类别: {len(delete_list)} 个")
    print("=" * 60)

    return delete_list, warn_list, good_list


# ============================================================
# Step 3: 可视化
# ============================================================
def plot_distribution(counts, delete_list, warn_list, good_list):
    delete_names = {n for n, _ in delete_list}
    warn_names   = {n for n, _ in warn_list}

    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    names  = [n for n, _ in sorted_counts]
    values = [c for _, c in sorted_counts]
    colors = []
    for name in names:
        if name in delete_names:
            colors.append("#e74c3c")   # 红色 = 删除
        elif name in warn_names:
            colors.append("#f39c12")   # 橙色 = 需补数据
        else:
            colors.append("#2ecc71")   # 绿色 = 充足

    fig, ax = plt.subplots(figsize=(20, 8))
    bars = ax.bar(range(len(names)), values, color=colors)

    ax.axhline(y=MIN_KEEP, color='red',    linestyle='--', alpha=0.7, label=f'Delete threshold ({MIN_KEEP})')
    ax.axhline(y=MIN_GOOD, color='orange', linestyle='--', alpha=0.7, label=f'Warning threshold ({MIN_GOOD})')

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_title("Class Distribution - Data Quality Analysis", fontsize=14, fontweight='bold')
    ax.set_ylabel("Number of Annotations")
    ax.legend()

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label=f'Delete (< {MIN_KEEP})'),
        Patch(facecolor='#f39c12', label=f'Need more data ({MIN_KEEP}-{MIN_GOOD})'),
        Patch(facecolor='#2ecc71', label=f'Good (>= {MIN_GOOD})'),
    ]
    ax.legend(handles=legend_elements)
    plt.tight_layout()

    save_path = REPORT_DIR / "class_quality.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  💾 图表已保存: {save_path}")


# ============================================================
# Step 4: 清理数据集（删除稀有类别）
# ============================================================
def clean_dataset(counts, class_names):
    delete_names = {name for name, count in counts.items() if count < MIN_KEEP}

    if not delete_names:
        print("  ✅ 没有需要删除的类别！")
        return

    # 建立新的类别列表（保留顺序）
    new_classes   = [n for n in class_names if n not in delete_names]
    old_to_new    = {}
    for old_id, name in enumerate(class_names):
        if name in delete_names:
            continue
        new_id = new_classes.index(name)
        old_to_new[old_id] = new_id

    print(f"\n  🧹 开始清理，保留 {len(new_classes)} 个类别...")

    # 清空输出目录
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    total_removed_boxes = 0
    total_kept_boxes    = 0

    for split in ["train", "valid", "test"]:
        img_src = DATA_DIR / split / "images"
        lbl_src = DATA_DIR / split / "labels"
        img_dst = OUTPUT_DIR / split / "images"
        lbl_dst = OUTPUT_DIR / split / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        if not img_src.exists():
            continue

        img_files = (list(img_src.glob("*.jpg")) +
                     list(img_src.glob("*.jpeg")) +
                     list(img_src.glob("*.png")))

        kept_imgs = 0
        for img_path in img_files:
            lbl_path = lbl_src / (img_path.stem + ".txt")

            new_lines = []
            if lbl_path.exists():
                with open(lbl_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts  = line.split()
                        old_id = int(parts[0])
                        if old_id in old_to_new:
                            new_id = old_to_new[old_id]
                            new_lines.append(f"{new_id} {' '.join(parts[1:])}\n")
                            total_kept_boxes += 1
                        else:
                            total_removed_boxes += 1

            # 只保留有标注的图片（可选：注释掉这行也保留背景图）
            if new_lines:
                shutil.copy2(img_path, img_dst / img_path.name)
                with open(lbl_dst / (img_path.stem + ".txt"), "w") as f:
                    f.writelines(new_lines)
                kept_imgs += 1

        print(f"  {split:6s}: {kept_imgs} 张图片")

    # 写新的 data.yaml
    with open(DATA_DIR / "data.yaml") as f:
        old_config = yaml.safe_load(f)

    new_config = {
        "path":  str(OUTPUT_DIR.resolve()),
        "train": "train/images",
        "val":   "valid/images",
        "test":  "test/images",
        "nc":    len(new_classes),
        "names": new_classes,
    }
    with open(OUTPUT_DIR / "data.yaml", "w") as f:
        yaml.dump(new_config, f, allow_unicode=True, default_flow_style=False)

    print(f"\n  ✅ 清理完成！")
    print(f"  类别数: {len(class_names)} → {len(new_classes)}")
    print(f"  删除标注框: {total_removed_boxes:,}")
    print(f"  保留标注框: {total_kept_boxes:,}")
    print(f"  新数据集: {OUTPUT_DIR}")
    print(f"  新 data.yaml: {OUTPUT_DIR / 'data.yaml'}")


# ============================================================
# Step 5: 生成需要补数据的购物清单
# ============================================================
def print_shopping_list(warn_list):
    print()
    print("=" * 60)
    print("🛒 需要补充数据的类别清单")
    print("   (建议每个类别补到100+个标注框)")
    print("=" * 60)
    print()
    for name, count in sorted(warn_list, key=lambda x: x[1]):
        need = MIN_GOOD - count
        print(f"  {name:<30} 现有 {count:>4} 个  还需补 {need:>4} 个")
    print()
    print("建议去 Roboflow Universe 搜索对应类别补充数据")
    print("例如：搜索 'cucumber detection' 下载专项数据集")


# ============================================================
# Main
# ============================================================
def main():
    print()
    print("🧊 Fridge AI - 数据质量分析")
    print()

    counts, class_names = count_annotations()
    delete_list, warn_list, good_list = print_report(counts, class_names)

    print("\n生成可视化图表...")
    plot_distribution(counts, delete_list, warn_list, good_list)

    print_shopping_list(warn_list)

    print()
    answer = input("是否立即清理数据集（删除稀有类别）? [y/N]: ").strip().lower()
    if answer == 'y':
        clean_dataset(counts, class_names)
        print()
        print("下一步: 用清理后的数据集重新训练")
        print("  修改 train.py 里的 DATA_YAML:")
        print(f"  DATA_YAML = Path('data/processed/merged_clean/data.yaml')")
        print("  python src/train.py")
    else:
        print("  跳过清理，只生成报告。")


if __name__ == "__main__":
    main()
