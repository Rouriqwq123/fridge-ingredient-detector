"""
Day 6: 模型评估脚本
在测试集上评估模型性能，生成详细报告

Usage: python src/evaluate.py
"""

import json
import yaml
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ============================================================
# 配置
# ============================================================
MODEL_PATH = Path("runs/detect/runs/train/fridge_v12/weights/best.pt")
DATA_YAML  = Path("data/processed/merged/data.yaml")
OUTPUT_DIR = Path("outputs/evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Step 1: 在测试集上评估
# ============================================================
def run_evaluation(model, data_yaml):
    print("=" * 55)
    print("🔍 在测试集上评估...")
    print("=" * 55)

    results = model.val(
        data=str(data_yaml),
        split="test",        # 用 test 集，不是 val
        imgsz=640,
        conf=0.25,           # 置信度阈值
        iou=0.5,             # IoU 阈值
        plots=True,
        save_json=False,
        verbose=False,
    )
    return results

# ============================================================
# Step 2: 打印核心指标
# ============================================================
def print_metrics(results):
    print()
    print("=" * 55)
    print("📊 核心评估指标")
    print("=" * 55)

    mp  = results.box.mp    # mean precision
    mr  = results.box.mr    # mean recall
    map50    = results.box.map50     # mAP@0.5
    map5095  = results.box.map       # mAP@0.5:0.95

    print(f"  mAP50:        {map50:.4f}  ({map50*100:.1f}%)")
    print(f"  mAP50-95:     {map5095:.4f}  ({map5095*100:.1f}%)")
    print(f"  Precision:    {mp:.4f}  ({mp*100:.1f}%)")
    print(f"  Recall:       {mr:.4f}  ({mr*100:.1f}%)")
    print()

    # F1 score
    f1 = 2 * mp * mr / (mp + mr + 1e-8)
    print(f"  F1 Score:     {f1:.4f}  ({f1*100:.1f}%)")
    print()

    # 简单评级
    print("  评级:")
    if map50 >= 0.7:
        print("  ⭐⭐⭐ 优秀！可以直接用于生产")
    elif map50 >= 0.5:
        print("  ⭐⭐  良好！适合 demo 展示，可继续优化")
    elif map50 >= 0.3:
        print("  ⭐   一般，建议增加数据或换更大模型")
    else:
        print("  ❌  需要改进，检查数据质量")

    return {"mAP50": map50, "mAP50-95": map5095,
            "precision": mp, "recall": mr, "f1": f1}

# ============================================================
# Step 3: 按类别分析，找出好/差的类别
# ============================================================
def analyze_per_class(results, data_yaml):
    print("=" * 55)
    print("📋 按类别分析 (Top10 最好 / Top10 最差)")
    print("=" * 55)

    with open(data_yaml) as f:
        config = yaml.safe_load(f)
    class_names = config['names']

    # 获取每个类别的 AP50
    ap_per_class = results.box.ap50  # shape: (num_classes,)

    if ap_per_class is None or len(ap_per_class) == 0:
        print("  ⚠️  无法获取每类别详细数据")
        return

    class_ap = [(class_names[i], float(ap_per_class[i]))
                for i in range(min(len(class_names), len(ap_per_class)))]
    class_ap = [(n, ap) for n, ap in class_ap if ap > 0]  # 过滤掉没有数据的类
    class_ap.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  {'🏆 Top 10 最好的类别':}")
    print(f"  {'类别':<25} {'AP50':>8}")
    print(f"  {'-'*35}")
    for name, ap in class_ap[:10]:
        bar = "█" * int(ap * 20)
        print(f"  {name:<25} {ap:>7.3f}  {bar}")

    print(f"\n  {'⚠️  Top 10 最差的类别':}")
    print(f"  {'类别':<25} {'AP50':>8}")
    print(f"  {'-'*35}")
    for name, ap in class_ap[-10:]:
        bar = "█" * int(ap * 20)
        print(f"  {name:<25} {ap:>7.3f}  {bar}")

    return class_ap

# ============================================================
# Step 4: 推理速度测试
# ============================================================
def test_inference_speed(model):
    print()
    print("=" * 55)
    print("⚡ 推理速度测试")
    print("=" * 55)

    import time
    import cv2
    import numpy as np

    # 创建一张随机测试图
    dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    tmp_path = OUTPUT_DIR / "dummy_test.jpg"
    cv2.imwrite(str(tmp_path), dummy_img)

    # 预热
    for _ in range(3):
        model(str(tmp_path), verbose=False)

    # 正式计时
    times = []
    for _ in range(20):
        start = time.time()
        model(str(tmp_path), verbose=False)
        times.append((time.time() - start) * 1000)

    tmp_path.unlink()  # 删除临时文件

    avg_ms = np.mean(times)
    fps = 1000 / avg_ms
    print(f"  平均推理时间: {avg_ms:.1f} ms")
    print(f"  FPS:         {fps:.1f} 帧/秒")
    print()

    if fps >= 30:
        print("  ✅ 实时推理（≥30fps），可用于视频流")
    elif fps >= 10:
        print("  ✅ 准实时，适合图片推理场景")
    else:
        print("  ⚠️  速度偏慢，考虑用 nano 模型或 TensorRT 加速")

    return avg_ms, fps

# ============================================================
# Step 5: 保存评估报告
# ============================================================
def save_report(metrics, class_ap, avg_ms, fps):
    report = {
        "model": str(MODEL_PATH),
        "metrics": metrics,
        "inference_speed_ms": avg_ms,
        "fps": fps,
        "top10_best_classes": class_ap[:10] if class_ap else [],
        "top10_worst_classes": class_ap[-10:] if class_ap else [],
    }
    report_path = OUTPUT_DIR / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 评估报告已保存: {report_path}")

# ============================================================
# Main
# ============================================================
def main():
    print()
    print("🧊 Fridge AI - Day 6: 模型评估")
    print()

    if not MODEL_PATH.exists():
        print(f"❌ 找不到模型: {MODEL_PATH}")
        print("   请先运行: python src/train.py")
        return

    print(f"  加载模型: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    # 评估
    results    = run_evaluation(model, DATA_YAML)
    metrics    = print_metrics(results)
    class_ap   = analyze_per_class(results, DATA_YAML)
    avg_ms, fps = test_inference_speed(model)
    save_report(metrics, class_ap, avg_ms, fps)

    print()
    print("=" * 55)
    print("✅ 评估完成！")
    print("=" * 55)
    print(f"  结果保存在: {OUTPUT_DIR}")
    print()
    print("下一步 (Day 8): 菜谱推荐模块")
    print("  python src/recommend.py")

if __name__ == "__main__":
    main()
