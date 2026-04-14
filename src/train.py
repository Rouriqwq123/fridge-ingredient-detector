"""
Day 4/5: 训练 YOLOv8 模型
Usage: python src/train.py
"""

from pathlib import Path
from ultralytics import YOLO

# ============================================================
# 配置
# ============================================================
DATA_YAML = Path("data/processed/merged/data.yaml")
MODEL     = "yolov8n.pt"   # nano，先跑通；效果不够再换 yolov8s.pt
EPOCHS    = 150
IMGSZ     = 1280
BATCH     = 4              # RTX 3050 4GB 显存，8 比较稳；跑满可改 16
PROJECT   = "runs/train"
NAME      = "fridge_v2"

# ============================================================
# 主函数
# ============================================================
def main():
    print("🧊 Fridge AI - 开始训练")
    print(f"   模型  : {MODEL}")
    print(f"   数据  : {DATA_YAML}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch : {BATCH}")
    print(f"   输出  : {PROJECT}/{NAME}")
    print()

    if not DATA_YAML.exists():
        print(f"❌ 找不到 data.yaml: {DATA_YAML}")
        print("   请先运行: python src/merge_datasets.py")
        return

    model = YOLO(MODEL)

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        project=PROJECT,
        name=NAME,
        # 数据增强
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.1,
        fliplr=0.5,
        mosaic=1.0,
        # 训练稳定性
        patience=20,        # 20轮没提升就提前停止
        save=True,
        save_period=10,     # 每10轮保存一次
        val=True,
        plots=True,         # 自动生成训练曲线图
        verbose=True,
    )

    print()
    print("=" * 55)
    print("✅ 训练完成！")
    print("=" * 55)
    best = Path(PROJECT) / NAME / "weights" / "best.pt"
    print(f"  最佳模型: {best}")
    print()
    print("下一步: 验证模型效果")
    print("  python src/evaluate.py")

if __name__ == "__main__":
    main()
