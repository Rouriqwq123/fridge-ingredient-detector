"""
Day 1: YOLOv8 Demo 检测脚本
用 COCO 预训练模型检测图片，验证整个 pipeline 能跑通

Usage:
  python src/demo_detect.py                        # 用内置示例图片
  python src/demo_detect.py --image your_photo.jpg # 用你自己的图片
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def run_demo(image_path: str | None = None):
    """
    下载 YOLOv8n 预训练模型并检测一张图片
    
    Args:
        image_path: 图片路径，为 None 则使用 ultralytics 内置示例
    """

    # ==========================================
    # Step 1: 加载预训练模型
    # ==========================================
    # 第一次运行会自动下载 yolov8n.pt (~6MB)
    # n=nano, s=small, m=medium, l=large, x=xlarge
    # 我们先用最小的 nano 版本
    print("📦 加载 YOLOv8n 预训练模型...")
    model = YOLO("yolov8n.pt")
    print(f"   模型加载完成，共 {len(model.names)} 个类别")
    print(f"   类别列表: {list(model.names.values())[:10]}...")
    print()

    # ==========================================
    # Step 2: 选择测试图片
    # ==========================================
    if image_path is None:
        # 使用 ultralytics 自带的示例图片
        # 这张图是一张包含行人和汽车的街景
        image_path = "https://ultralytics.com/images/bus.jpg"
        print(f"🖼️  使用示例图片: {image_path}")
    else:
        image_path = str(Path(image_path).resolve())
        print(f"🖼️  使用自定义图片: {image_path}")
    print()

    # ==========================================
    # Step 3: 运行推理
    # ==========================================
    print("🔍 开始检测...")
    results = model(image_path)
    print()

    # ==========================================
    # Step 4: 解析结果
    # ==========================================
    result = results[0]  # 只有一张图，取第一个结果

    print("=" * 50)
    print("📊 检测结果")
    print("=" * 50)

    if len(result.boxes) == 0:
        print("未检测到任何物体")
        return

    print(f"共检测到 {len(result.boxes)} 个物体:\n")

    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls[0])          # 类别 ID
        cls_name = model.names[cls_id]     # 类别名称
        confidence = float(box.conf[0])    # 置信度
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # 边界框坐标

        print(f"  [{i+1}] {cls_name}")
        print(f"      置信度: {confidence:.2%}")
        print(f"      位置: ({x1:.0f}, {y1:.0f}) → ({x2:.0f}, {y2:.0f})")
        print()

    # ==========================================
    # Step 5: 保存标注后的图片
    # ==========================================
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "demo_result.jpg"

    # result.plot() 返回一个带标注框的 numpy 数组
    annotated = result.plot()

    import cv2
    cv2.imwrite(str(output_path), annotated)
    print(f"💾 标注图片已保存: {output_path}")
    print()

    # ==========================================
    # Step 6: 模型信息汇总
    # ==========================================
    print("=" * 50)
    print("📋 模型信息")
    print("=" * 50)
    print(f"  模型: YOLOv8n (nano)")
    print(f"  输入尺寸: 640x640")
    print(f"  预训练数据集: COCO (80 类)")
    speed = result.speed
    print(f"  推理速度: 预处理 {speed['preprocess']:.1f}ms"
          f" + 推理 {speed['inference']:.1f}ms"
          f" + 后处理 {speed['postprocess']:.1f}ms")
    print()
    print("✅ Demo 完成！YOLOv8 环境验证通过。")
    print()
    print("=" * 50)
    print("🎯 下一步 (Day 2)")
    print("=" * 50)
    print("  1. 从 Roboflow 下载冰箱食材数据集")
    print("  2. 运行 EDA 分析数据集")
    print("  3. python notebooks/01_eda.py")


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Demo Detection")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="图片路径 (不指定则使用示例图片)",
    )
    args = parser.parse_args()
    run_demo(args.image)


if __name__ == "__main__":
    main()
