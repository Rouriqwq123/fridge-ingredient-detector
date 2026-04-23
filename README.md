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
   - Fine-tuned on 54,000+ images
   - 88 food categories (after quality filtering)
   - mAP50: 78.2%
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

| Model | Dataset | mAP50 | mAP50-95 | Precision | Recall | Note |
|-------|---------|-------|----------|-----------|--------|------|
| fridge_v1 | 7K images, 91 classes | 51.0% | 37.0% | 68.0% | 47.0% | Baseline |
| fridge_v12 | 54K images, 125 classes | 76.8% | 58.6% | 82.1% | 76.4% | +3 datasets |
| fridge_v44 | 54K images, 121 classes | **78.2%** | **62.6%** | **81.9%** | **75.7%** | Resume training |

### Iteration Summary
- **v1 → v12 (+25.8%)**: Added 2 extra datasets (47K images), merged with class remapping
- **v12 → v44 (+1.4%)**: Continued training from checkpoint, switched to RTX 5070

### Best Performing Classes (AP50 > 0.95)
`leek` · `spinach` · `ham` · `mushrooms` · `bean` · `flour` · `corn` · `butter` · `annona` · `kumquat`

### Removed Low-Quality Classes
37 classes with < 20 annotations were removed after data quality analysis, improving overall mAP stability.

---

## 🔧 Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Object Detection | YOLOv8n (Ultralytics) | Detect ingredients in fridge photos |
| Deep Learning | PyTorch 2.12 + CUDA 12.8 | Model training & inference |
| Data Annotation | Roboflow | Dataset labeling & management |
| Recipe Matching | Jaccard Similarity | Match ingredients to recipes |
| Recipe Dataset | Food.com (50k recipes) | Recipe recommendation source |
| Web UI | Gradio | Bilingual (EN/ZH) web interface |
| Deployment | Hugging Face Spaces | Free online hosting |

---

## 📁 Dataset

### Training Data (3 sources merged)

| Dataset | Images | Classes | Source |
|---------|--------|---------|--------|
| Fridge Object Dataset | 7,358 | 61 | Roboflow Universe |
| Custom Annotations | 29 | 30 new | Self-labeled (Chinese ingredients) |
| Ingredients Detection YoloV8 | 47,000+ | 68 | Roboflow Universe (cleaned) |
| **Total (after quality filter)** | **~54,000** | **88** | — |

### Data Quality Pipeline
After merging, ran automated quality analysis:
- Removed **37 classes** with < 20 annotations (e.g. `longan`, `milkshake`, `sauce`)
- Flagged **7 classes** needing more data (e.g. `sausage`, `red bell pepper`)
- Final dataset: **88 classes**, all with 20+ annotations

### Why Custom Annotations?
The base dataset lacked Chinese-specific ingredients. I personally photographed and annotated 29 images adding categories like `tofu`, `rice`, `kimchi`, `bamboo shoots`, `pork` — common in Asian households but absent from Western datasets.

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/Rouriqwq123/fridge-ingredient-detector
cd fridge-ingredient-detector

conda create -n fridge-ai python=3.10 -y
conda activate fridge-ai

# For CUDA 12.1 (RTX 30xx series)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# For CUDA 12.8 (RTX 50xx series)
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

pip install -r requirements.txt
```

### 2. Download Data

- [Fridge Object Dataset](https://universe.roboflow.com/fridgeingredients/fridge-object) → `data/raw/fridge-object/`
- [Ingredients Detection YoloV8](https://universe.roboflow.com/visual-captioning-for-food/ingredients-detection-yolov8-npkkb) → `data/raw/ingredients-detection/`
- [Food.com Recipes](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) → `data/raw/recipes/`

### 3. Fix, Merge & Clean Datasets

```bash
python src/fix_ingredients_dataset.py   # Fix filenames & remove junk classes
python src/merge_datasets.py            # Merge all 3 datasets
python src/data_quality.py              # Remove rare classes, analyze distribution
```

### 4. Train

```bash
python src/train.py
```

### 5. Evaluate

```bash
python src/evaluate.py
```

### 6. Run Web App

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
│   │   ├── fridge-object/          # Base dataset (Roboflow)
│   │   ├── custom/                 # Self-labeled Chinese ingredients
│   │   ├── ingredients-detection/  # Raw (has junk classes)
│   │   ├── ingredients-fixed/      # Cleaned version
│   │   └── recipes/                # Food.com recipe dataset
│   └── processed/
│       ├── merged/                 # Raw merged dataset (125 classes)
│       └── merged_clean/           # Quality-filtered dataset (88 classes)
├── notebooks/
│   └── 01_eda.py                   # Exploratory data analysis
├── src/
│   ├── fix_ingredients_dataset.py  # Fix filenames & filter junk classes
│   ├── merge_datasets.py           # Multi-dataset merging with class remapping
│   ├── data_quality.py             # Per-class annotation analysis & cleanup
│   ├── train.py                    # YOLOv8 fine-tuning script
│   ├── evaluate.py                 # Model evaluation & per-class AP analysis
│   ├── recommend.py                # Recipe recommendation engine
│   └── app.py                      # Gradio bilingual web application
├── outputs/
│   ├── eda/                        # Class distribution charts
│   └── data_quality/               # Quality analysis charts
├── requirements.txt
└── README.md
```

---

## 🔍 Key Technical Decisions

**Why YOLOv8 over other detectors?**
Single-stage detector with excellent speed/accuracy tradeoff. Real-time capable at 53 FPS. The Ultralytics API streamlined the full pipeline from training to deployment.

**Why Transfer Learning?**
Fine-tuning YOLOv8n pretrained on COCO achieved 78.2% mAP50 with ~54k images — training from scratch would require millions of samples.

**Why remove rare classes instead of keeping them?**
Classes with < 20 annotations hurt overall mAP. A model that consistently gets 88 classes right is more useful than one that occasionally misidentifies 125 classes.

**Why prioritize "fewest missing ingredients" in recommendations?**
Initial version sorted by Jaccard similarity, which recommended recipes requiring many extra ingredients. Redesigned to sort by missing count first — much more practical for real cooking.

**Why AMP (Automatic Mixed Precision)?**
Halves VRAM usage by computing in float16, allowing larger batch sizes on consumer GPUs.

---

## 📈 Training Configuration

```python
# Latest run (fridge_v44, RTX 5070)
model    = YOLO("yolov8n.pt")   # Transfer learning from COCO pretrained
epochs   = 100
imgsz    = 640
batch    = 8
device   = "cuda"               # RTX 5070 Laptop GPU (8GB)

# Key augmentation — chosen for fridge-specific challenges
mosaic   = 1.0   # Handles cluttered fridge scenes with overlapping items
hsv_v    = 0.4   # Brightness variation for different fridge lighting conditions
fliplr   = 0.5   # Horizontal flip — food orientation is arbitrary
patience = 20    # Early stopping
```

---

## 🤝 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Roboflow Universe](https://universe.roboflow.com) — Fridge Object Dataset & Ingredients Detection Dataset
- [Food.com Recipes Dataset](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) on Kaggle
- [Gradio](https://gradio.app)

---

## 📄 License

MIT License
