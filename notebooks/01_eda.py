"""
Day 2: 数据集 EDA (探索性数据分析)
分析 Fridge Object 数据集的类别分布、图片数量、标注情况

Usage: python notebooks/01_eda.py
"""

import os
import yaml
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Windows 无头模式
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ============================================================
# 配置路径
# ============================================================
DATA_ROOT = Path("data/raw/fridge-object")
OUTPUT_DIR = Path("outputs/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Step 1: 读取 data.yaml
# ============================================================
def load_yaml():
    yaml_path = DATA_ROOT / "data.yaml"
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    print("=" * 55)
    print("📋 数据集配置 (data.yaml)")
    print("=" * 55)
    print(f"  类别数量: {config['nc']}")
    print(f"  类别列表:")
    for i, name in enumerate(config['names']):
        print(f"    [{i:2d}] {name}")
    print()
    return config

# ============================================================
# Step 2: 统计各 split 的图片数量
# ============================================================
def count_images(config):
    print("=" * 55)
    print("🖼️  图片数量统计")
    print("=" * 55)
    splits = {}
    for split in ["train", "valid", "test"]:
        img_dir = DATA_ROOT / split / "images"
        if img_dir.exists():
            count = len(list(img_dir.glob("*.jpg")) +
                        list(img_dir.glob("*.jpeg")) +
                        list(img_dir.glob("*.png")))
            splits[split] = count
            print(f"  {split:6s}: {count:5d} 张图片")
    total = sum(splits.values())
    print(f"  {'总计':6s}: {total:5d} 张图片")
    print()
    return splits

# ============================================================
# Step 3: 统计类别分布
# ============================================================
def count_classes(config):
    print("=" * 55)
    print("📊 类别分布统计")
    print("=" * 55)
    class_names = config['names']
    class_counts = defaultdict(int)

    for split in ["train", "valid", "test"]:
        label_dir = DATA_ROOT / split / "labels"
        if not label_dir.exists():
            continue
        for label_file in label_dir.glob("*.txt"):
            with open(label_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        cls_id = int(line.split()[0])
                        if cls_id < len(class_names):
                            class_counts[class_names[cls_id]] += 1

    # 排序显示
    sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"  {'类别':<20} {'数量':>8} {'占比':>8}")
    print(f"  {'-'*40}")
    total_boxes = sum(class_counts.values())
    for name, count in sorted_counts:
        pct = count / total_boxes * 100 if total_boxes > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {name:<20} {count:>8,} {pct:>7.1f}%  {bar}")
    print(f"\n  总标注框数: {total_boxes:,}")
    print()
    return dict(class_counts)

# ============================================================
# Step 4: 可视化 - 类别分布柱状图
# ============================================================
def plot_class_distribution(class_counts, class_names):
    # 按 class_names 顺序排列，没有的填0
    counts = [class_counts.get(name, 0) for name in class_names]

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    bars = ax.bar(class_names, counts, color=colors, edgecolor='white', linewidth=0.5)

    # 在每个柱上显示数值
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   str(count), ha='center', va='bottom', fontsize=8)

    ax.set_title("Fridge Object Dataset - Class Distribution", fontsize=14, fontweight='bold')
    ax.set_xlabel("Food Category")
    ax.set_ylabel("Number of Annotations")
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    save_path = OUTPUT_DIR / "class_distribution.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 类别分布图已保存: {save_path}")

# ============================================================
# Step 5: 可视化 - 随机抽样展示标注框
# ============================================================
def plot_sample_images(config, n_samples=6):
    class_names = config['names']
    # 随机选 n_samples 张训练图片
    img_dir = DATA_ROOT / "train" / "images"
    label_dir = DATA_ROOT / "train" / "labels"
    img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))

    if len(img_files) == 0:
        print("  ⚠️  找不到训练图片")
        return

    np.random.seed(42)
    selected = np.random.choice(img_files,
                                size=min(n_samples, len(img_files)),
                                replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # 颜色映射
    colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))

    for idx, (ax, img_path) in enumerate(zip(axes, selected)):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        ax.imshow(img)

        # 读取对应的标注文件
        label_path = label_dir / (img_path.stem + ".txt")
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])

                    # YOLO 格式转像素坐标
                    x1 = (cx - bw/2) * w
                    y1 = (cy - bh/2) * h
                    box_w = bw * w
                    box_h = bh * h

                    color = colors[cls_id % len(colors)]
                    rect = patches.Rectangle(
                        (x1, y1), box_w, box_h,
                        linewidth=2, edgecolor=color, facecolor='none'
                    )
                    ax.add_patch(rect)

                    label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                    ax.text(x1, y1 - 3, label,
                           color='white', fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8))

        ax.set_title(f"Sample {idx+1}", fontsize=10)
        ax.axis('off')

    # 隐藏多余的子图
    for ax in axes[len(selected):]:
        ax.axis('off')

    plt.suptitle("Fridge Object Dataset - Sample Annotations", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = OUTPUT_DIR / "sample_annotations.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 样本标注图已保存: {save_path}")

# ============================================================
# Step 6: 分析图片尺寸
# ============================================================
def analyze_image_sizes():
    print("=" * 55)
    print("📐 图片尺寸分析")
    print("=" * 55)
    widths, heights = [], []
    img_dir = DATA_ROOT / "train" / "images"
    img_files = list(img_dir.glob("*.jpg"))[:200]  # 抽样200张分析

    for img_path in img_files:
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)

    if widths:
        print(f"  宽度: min={min(widths)}, max={max(widths)}, avg={np.mean(widths):.0f}")
        print(f"  高度: min={min(heights)}, max={max(heights)}, avg={np.mean(heights):.0f}")
        unique_sizes = set(zip(widths, heights))
        print(f"  不同尺寸种类: {len(unique_sizes)} 种")
        if len(unique_sizes) <= 5:
            for s in sorted(unique_sizes):
                print(f"    {s[0]}x{s[1]}")
    print()

# ============================================================
# Main
# ============================================================
def main():
    print()
    print("🧊 Fridge AI - Day 2: 数据集 EDA")
    print()

    config = load_yaml()
    splits = count_images(config)
    class_counts = count_classes(config)
    analyze_image_sizes()

    print("=" * 55)
    print("📈 生成可视化图表...")
    print("=" * 55)
    plot_class_distribution(class_counts, config['names'])
    plot_sample_images(config)
    print()

    print("=" * 55)
    print("✅ EDA 完成！")
    print("=" * 55)
    print(f"  查看结果图表: outputs\\eda\\")
    print()
    print("关键发现:")
    if class_counts:
        most = max(class_counts, key=class_counts.get)
        least = min(class_counts, key=class_counts.get)
        print(f"  - 最多类别: {most} ({class_counts[most]:,} 个标注)")
        print(f"  - 最少类别: {least} ({class_counts[least]:,} 个标注)")
        imbalance = class_counts[most] / class_counts[least]
        print(f"  - 类别不平衡比: {imbalance:.1f}x")
        if imbalance > 10:
            print(f"  ⚠️  类别不平衡较严重，训练时需要注意")
    print()
    print("下一步 (Day 3): 拍摄自己的冰箱照片并标注")

if __name__ == "__main__":
    main()
