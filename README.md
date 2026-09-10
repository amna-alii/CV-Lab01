# CV-Lab01
# Skin Lesion Classification — Computer Vision Lab Task

This repository contains the implementation for a Computer Vision lab task comparing **transfer learning models**, **classical classifiers on deep features**, and **computational efficiency** for multi-class skin lesion classification.

## 📌 Task Overview

Using a subset of a skin disease image dataset, this project addresses three sub-tasks:

1. **Transfer Learning Model Comparison** — Fine-tune 8 pretrained CNN architectures and compare their classification performance.
2. **Classifier Comparison on Deep Features** — Extract deep features from a fine-tuned CNN and train 7 classical machine learning classifiers on top of them.
3. **Computational Efficiency Comparison** — Compare the same backbones on parameter count, model size, FLOPs, and inference speed.

## 🩺 Dataset

- **Source:** [Skin Disease Classification Image Dataset (Kaggle)](https://www.kaggle.com/datasets/riyaelizashaju/skin-disease-classification-image-dataset)
- **Classes used (5 of 9 available):**
  - Melanoma
  - Melanocytic nevus
  - Benign keratosis
  - Dermatofibroma
  - Vascular lesion
- **Split:** Pre-organized into `train/` (~80 images/class) and `val/` (~20 images/class)
- **Format:** JPEG images, organized in a Kaggle `ImageFolder`-style directory structure:

```
Split_smol/
├── train/
│   ├── Melanoma/
│   ├── Melanocytic nevus/
│   ├── Benign keratosis/
│   ├── Dermatofibroma/
│   └── Vascular lesion/
└── val/
    ├── Melanoma/
    ├── Melanocytic nevus/
    ├── Benign keratosis/
    ├── Dermatofibroma/
    └── Vascular lesion/
```

> The raw dataset zip is not included in this repo due to size/licensing — download it from the Kaggle link above and place it as described in [Usage](#-usage).

## 🧠 Methods

### Table 1 — Transfer Learning Models
Pretrained (ImageNet) backbones fine-tuned end-to-end with a new classification head for 5 classes:
`AlexNet`, `VGG16`, `VGG19`, `ResNet18`, `ResNet50`, `ResNet101`, `DenseNet121`, `EfficientNet-B0`

### Table 2 — Classifiers on Deep Features
Deep features (2048-D) extracted from the fine-tuned **ResNet50**, standardized, then classified using:
`Logistic Regression`, `Decision Tree`, `Random Forest`, `K-Nearest Neighbors (KNN)`, `Linear SVM`, `RBF-SVM`, `XGBoost`

### Table 3 — Computational Efficiency
For each backbone: parameter count (M), model size on disk (MB), FLOPs (G), and average inference time per image (ms), alongside its Table 1 accuracy.

## 📊 Evaluation Metrics

All models are evaluated on the validation set using:
- Accuracy (%)
- Precision (%) — macro-averaged
- Recall (%) — macro-averaged
- F1-Score (%) — macro-averaged
- AUC (%) — macro-averaged, one-vs-rest

## 🗂️ Repository Structure

```
.
├── Skin_Lesion_Classification_Lab.ipynb   # Main notebook (run in Google Colab)
├── Task_01.docx                           # Lab task sheet with result tables
└── README.md
```

## 🚀 Usage

1. Open `Skin_Lesion_Classification_Lab.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Set **Runtime → Change runtime type → GPU (T4)**.
3. Run all cells top to bottom:
   - Upload the dataset zip when prompted.
   - The notebook automatically filters it down to the 5 selected classes.
   - All three tables are trained, evaluated, printed, and exported as CSV files.
4. Copy the generated CSV values into the corresponding tables in `Task_01.docx`, or place the CSVs in `results/` in this repo.

## 🛠️ Requirements

Installed automatically inside the notebook (Colab already has most of these):

```
torch
torchvision
scikit-learn
pandas
numpy
xgboost
thop
```

## 📈 Results

_Add your final numbers here after running the notebook, e.g.:_

| Model | Accuracy (%) | F1-Score (%) |
|---|---|---|
| ResNet50 | — | — |
| EfficientNet-B0 | — | — |
| ... | | |

## 📄 License

This project is for academic/educational purposes as part of a Computer Vision lab assignment. Dataset used under its original Kaggle license.
