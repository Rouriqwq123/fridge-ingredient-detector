<img width="1568" height="738" alt="image" src="https://github.com/user-attachments/assets/cf9af44f-07ce-4a1a-a7ce-c3b9bc449053" />

# 🧊 Fridge Ingredient Detector → Recipe Recommender

> Take a photo of your fridge, AI detects ingredients and recommends recipes you can cook right now.

[![HuggingFace Demo](https://img.shields.io/badge/🤗%20HuggingFace-Demo-blue)](https://huggingface.co/spaces/Rouriqwq/fridge-ingredient-detector)
[![Python](https://img.shields.io/badge/Python-3.10-green)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🎯 Live Demo

👉 **https://huggingface.co/spaces/Rouriqwq/fridge-ingredient-detector**

---

## 💡 Problem & Motivation

Ever open your fridge and have no idea what to cook? This project solves that.

Upload a fridge photo → AI identifies what ingredients you have → Get recipe recommendations that match your available ingredients, minimizing what you need to buy.

---

## 🏗️ System Architecture

```
📸 Upload fridge photo
        ↓
🎯 YOLOv8 Ingredient Detection
   - Fine-tuned on 45,265 images
   - 125 food categories
   - mAP50: 76.8%
        ↓
📋 Ingredient List + Confidence Scores
        ↓
🍳 Recipe Recommendation Engine
   - Jaccard similarity matching
   - Prioritizes recipes with fewest missing ingredients
   - 50,000 recipes from Food.com dataset
        ↓
📱 Gradio Web Interface (EN/ZH bilingual)
```

---

## 📊 Model Performance

| Model | mAP50 | mAP50-95 | Precision | Recall | FPS |
|-------|-------|----------|-----------|--------|-----|
| fridge_v1 (baseline) | 51.0% | 37.0% | 68.0% | 47.0% | 49.5 |
| fridge_v12 (final) | **76.8%** | **58.6%** | **82.1%** | **76.4%** | **53.0** |

### Best Performing Classes (AP50 > 0.98)
`leek` · `spinach` · `bamboo shoots` · `kimchi` · `ham` · `mushrooms` · `bean` · `flour` · `corn` · `butter`

### Needs Improvement (AP50 < 0.15)
`longan` · `orange juice` · `green chilies` — insufficient training data (<5 samples per class)

---

## 🔧 Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Object Detection | YOLOv8n (Ultralytics) | Detect ingredients in fridge photos |
| Deep Learning | PyTorch 2.5.1 + CUDA 12.1 | Model training & inference |
| Data Annotation | Roboflow | Dataset labeling & management |
| Recipe Matching | Jaccard Similarity | Match ingredients to recipes |
| Recipe Dataset | Food.com (50k recipes) | Recipe recommendation source |
| Web UI | Gradio | Bilingual web interface |
| Deployment | Hugging Face Spaces | Free online hosting |

---

## 📁 Dataset

### Training Data (3 sources merged)

| Dataset | Images | Classes | Source |
|---------|--------|---------|--------|
| Fridge Object Dataset | 7,358 | 61 | Roboflow Universe |
| Custom Annotations | 29 | 30 new | Self-labeled (Chinese ingredients) |
| Ingredients Detection YoloV8 | 47,000+ | 68 | Roboflow Universe |
| **Total** | **~54,000** | **125** | — |

### Why Custom Annotations?
The base dataset lacked Chinese-specific ingredients. I personally photographed and annotated 29 images adding categories like `tofu`, `rice`, `kimchi`, `bamboo shoots`, `pork` — common in Asian households but absent from Western datasets.

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/fridge-ingredient-detector
cd fridge-ingredient-detector

conda create -n fridge-ai python=3.10 -y
conda activate fridge-ai

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. Download Data

- [Fridge Object Dataset](https://universe.roboflow.com/fridgeingredients/fridge-object) → `data/raw/fridge-object/`
- [Food.com Recipes](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) → `data/raw/recipes/`

### 3. Merge Datasets & Train

```bash
python src/merge_datasets.py
python src/train.py
```

### 4. Evaluate

```bash
python src/evaluate.py
```

### 5. Run Web App

```bash
python src/app.py
# Open http://localhost:7860
```

---

## 📂 Project Structure

```
fridge-ingredient-detector/
├── data/
│   ├── raw/
│   │   ├── fridge-object/        # Base dataset
│   │   ├── custom/               # Self-labeled Chinese ingredients
│   │   ├── ingredients-fixed/    # Additional ingredients dataset
│   │   └── recipes/              # Food.com recipe dataset
│   └── processed/
│       └── merged/               # Final merged training dataset
├── notebooks/
│   └── 01_eda.py                 # Exploratory data analysis
├── src/
│   ├── merge_datasets.py         # Multi-dataset merging with class remapping
│   ├── train.py                  # YOLOv8 fine-tuning script
│   ├── evaluate.py               # Model evaluation & per-class AP analysis
│   ├── recommend.py              # Recipe recommendation engine
│   └── app.py                    # Gradio bilingual web application
├── outputs/
│   └── eda/                      # EDA charts (class distribution, sample annotations)
├── requirements.txt
└── README.md
```

---

## 🔍 Key Technical Decisions

**Why YOLOv8 over other detectors?**
Single-stage detector with excellent speed/accuracy tradeoff. At 53 FPS on RTX 3050, it's real-time capable. The Ultralytics API also streamlined the full pipeline from training to deployment.

**Why Transfer Learning?**
Fine-tuning YOLOv8n pretrained on COCO achieved 76.8% mAP50 with ~45k images. Training from scratch would require millions of samples to reach comparable performance.

**Why prioritize "fewest missing ingredients" in recommendations?**
Initial version sorted by Jaccard similarity, which recommended recipes requiring many extra ingredients. Redesigned to sort by missing count first — much more practical for real cooking decisions.

**Why AMP (Automatic Mixed Precision)?**
RTX 3050 has only 4GB VRAM. AMP halves memory usage by computing in float16 where possible, allowing batch_size=8 at 640px resolution without OOM.

---

## 📈 Training Configuration

```python
model    = YOLO("yolov8n.pt")   # Transfer learning from COCO pretrained
epochs   = 50
imgsz    = 640
batch    = 8
device   = "cuda"               # RTX 3050 Laptop GPU (4GB)

# Key augmentation params
mosaic   = 1.0   # 4-image mosaic — handles cluttered fridge scenes
hsv_v    = 0.4   # Brightness jitter — accounts for varying fridge lighting
fliplr   = 0.5   # Horizontal flip
patience = 20    # Early stopping to prevent overfitting
```

---

## 🤝 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Roboflow Universe](https://universe.roboflow.com) — Fridge Object Dataset
- [Food.com Recipes Dataset](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) on Kaggle
- [Gradio](https://gradio.app)

---

## 📄 License

MIT License
