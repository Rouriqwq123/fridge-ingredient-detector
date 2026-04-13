# 🧊 Fridge Ingredient Detector → Recipe Recommender

> 拍一张冰箱照片，AI 自动识别食材并推荐菜谱

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📖 项目简介

你是否经常打开冰箱，看着里面的食材却不知道能做什么菜？

本项目使用 **YOLOv8 目标检测模型**识别冰箱中的食材，并基于检测结果**智能推荐菜谱**，解决"有食材不知道做什么"的日常痛点。

## 🏗️ 系统架构

```
📸 拍照/上传图片
    ↓
🎯 YOLOv8 食材检测
    ↓
📋 食材列表 + 置信度
    ↓
🍳 菜谱推荐引擎
    ↓
📱 Gradio Web 界面展示
```

## 🔧 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| 目标检测 | YOLOv8 (Ultralytics) | 识别冰箱中的食材 |
| 深度学习框架 | PyTorch | 模型训练与推理 |
| 数据标注 | Roboflow | 标注自采集数据 |
| Web 界面 | Gradio | 用户交互界面 |
| 部署 | Hugging Face Spaces | 在线 Demo |

## 📊 模型性能

| 模型 | mAP50 | mAP50-95 | 推理速度 | 备注 |
|------|-------|----------|----------|------|
| YOLOv8n | - | - | - | Baseline |
| YOLOv8s | - | - | - | |
| YOLOv8m | - | - | - | |

> ⏳ 训练完成后更新

## 🚀 快速开始

### 1. 环境安装

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/fridge-ingredient-detector.git
cd fridge-ingredient-detector

# 创建虚拟环境
conda create -n fridge-ai python=3.10 -y
conda activate fridge-ai

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行推理

```bash
python src/detect.py --image path/to/your/fridge.jpg
```

### 3. 启动 Web 界面

```bash
python src/app.py
```

## 📁 项目结构

```
fridge-ingredient-detector/
├── data/
│   ├── raw/              # 原始数据集
│   ├── processed/        # 处理后的数据
│   └── augmented/        # 增强后的数据
├── models/               # 训练好的模型权重
├── notebooks/            # EDA 和实验 notebook
├── src/
│   ├── detect.py         # 推理脚本
│   ├── train.py          # 训练脚本
│   ├── recommend.py      # 菜谱推荐模块
│   └── app.py            # Gradio Web 应用
├── tests/                # 单元测试
├── outputs/              # 输出结果
├── requirements.txt
└── README.md
```

## 📈 训练过程

<!-- 训练完成后在这里添加 loss 曲线和检测效果图 -->

## 🤝 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Roboflow](https://roboflow.com/) - 数据集与标注工具
- Fridge Object Dataset - 基础训练数据

## 📄 License

MIT License
