"""
Day 1: 环境验证脚本
运行此脚本确认所有依赖安装正确、GPU 可用
Usage: python check_env.py
"""

import sys


def check_python():
    """检查 Python 版本"""
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 10
    status = "✅" if ok else "❌"
    print(f"{status} Python 版本: {v.major}.{v.minor}.{v.micro}", end="")
    if not ok:
        print("  (需要 3.10+，请升级)")
    else:
        print()
    return ok


def check_torch():
    """检查 PyTorch 和 GPU"""
    try:
        import torch

        print(f"✅ PyTorch 版本: {torch.__version__}")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU 可用: {gpu_name} ({gpu_mem:.1f} GB)")
            print(f"   CUDA 版本: {torch.version.cuda}")

            # 简单的 GPU 运算测试
            x = torch.randn(1000, 1000, device="cuda")
            y = torch.matmul(x, x)
            print(f"   GPU 运算测试: 通过 (1000x1000 矩阵乘法)")
        else:
            print("⚠️  GPU 不可用，将使用 CPU 训练（会很慢）")
            print("   提示: 确认已安装 CUDA 版本的 PyTorch")
            print("   安装命令: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return True
    except ImportError:
        print("❌ PyTorch 未安装")
        print("   安装命令: pip install torch torchvision")
        return False


def check_ultralytics():
    """检查 YOLOv8"""
    try:
        import ultralytics

        print(f"✅ Ultralytics 版本: {ultralytics.__version__}")

        from ultralytics import YOLO

        print("   YOLO 模块导入成功")
        return True
    except ImportError:
        print("❌ Ultralytics 未安装")
        print("   安装命令: pip install ultralytics")
        return False


def check_other_deps():
    """检查其他依赖"""
    deps = {
        "cv2": "opencv-python",
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "gradio": "gradio",
        "PIL": "Pillow",
        "roboflow": "roboflow",
    }

    all_ok = True
    for module, package in deps.items():
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} 未安装  →  pip install {package}")
            all_ok = False
    return all_ok


def main():
    print("=" * 50)
    print("🧊 Fridge AI - 环境检查")
    print("=" * 50)
    print()

    results = []

    print("【1/4】Python 版本")
    results.append(check_python())
    print()

    print("【2/4】PyTorch + GPU")
    results.append(check_torch())
    print()

    print("【3/4】YOLOv8 (Ultralytics)")
    results.append(check_ultralytics())
    print()

    print("【4/4】其他依赖")
    results.append(check_other_deps())
    print()

    print("=" * 50)
    if all(results):
        print("🎉 所有检查通过！环境就绪。")
        print()
        print("下一步: 运行 YOLOv8 Demo")
        print("  python src/demo_detect.py")
    else:
        print("⚠️  部分检查未通过，请根据上面的提示修复。")
    print("=" * 50)


if __name__ == "__main__":
    main()
