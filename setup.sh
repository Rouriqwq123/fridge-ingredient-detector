#!/bin/bash
# ============================================
# 🧊 Fridge AI - 一键环境搭建脚本
# ============================================
# Usage: bash setup.sh
# ============================================

set -e  # 遇到错误立即退出

echo "========================================"
echo "🧊 Fridge AI - 环境搭建"
echo "========================================"
echo ""

# ----- Step 1: 创建 Conda 环境 -----
echo "【1/5】创建 Conda 虚拟环境..."
if conda info --envs | grep -q "fridge-ai"; then
    echo "  ⚠️  fridge-ai 环境已存在，跳过创建"
    echo "  (如需重建: conda remove -n fridge-ai --all)"
else
    conda create -n fridge-ai python=3.10 -y
    echo "  ✅ 环境创建完成"
fi
echo ""

# ----- Step 2: 激活环境 -----
echo "【2/5】激活环境..."
# 注意: 在脚本中 conda activate 需要先 init
eval "$(conda shell.bash hook)"
conda activate fridge-ai
echo "  ✅ 当前环境: $(python --version)"
echo ""

# ----- Step 3: 安装 PyTorch (CUDA) -----
echo "【3/5】安装 PyTorch (CUDA 12.1 版本)..."
echo "  如果你的 CUDA 版本不同，请修改下面的命令"
echo "  查看 CUDA 版本: nvidia-smi"
echo ""
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo "  ✅ PyTorch 安装完成"
echo ""

# ----- Step 4: 安装其他依赖 -----
echo "【4/5】安装项目依赖..."
pip install -r requirements.txt
echo "  ✅ 所有依赖安装完成"
echo ""

# ----- Step 5: 验证环境 -----
echo "【5/5】验证环境..."
python check_env.py
echo ""

echo "========================================"
echo "🎉 搭建完成！"
echo ""
echo "每次开始工作前运行:"
echo "  conda activate fridge-ai"
echo ""
echo "下一步:"
echo "  python src/demo_detect.py"
echo "========================================"
