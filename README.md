# 🎗️ PyTorch Breast Cancer Detection & Explainability System

An end-to-end medical deep learning system for automated breast cancer classification from B-mode ultrasound imaging, complete with clinical performance evaluation and Grad-CAM visual interpretability.

---

## 🩺 Clinical Overview

Breast cancer is the most prevalent malignancy among women worldwide. Early and accurate diagnosis is pivotal—early-stage localized detection yields a 5-year survival rate exceeding **99%**.

### Ultrasound Diagnostic Characteristics
Ultrasound (US) imaging is the primary modality used to differentiate solid tumors from fluid-filled cysts, guiding biopsy decisions:
* **Benign Lesions** (e.g., Fibroadenomas, Simple Cysts):
  - Well-circumscribed, oval or round margins.
  - Wider-than-tall orientation.
  - Posterior acoustic enhancement.
* **Malignant Tumors** (e.g., Invasive Ductal Carcinoma):
  - Irregular, non-circumscribed, microlobulated or spiculated margins.
  - Taller-than-wide orientation (vertical growth axis).
  - Posterior acoustic shadowing and microcalcifications.
* **Normal Tissue**:
  - Regular layered glandular and subcutaneous echotexture without focal mass.

---

## 🚀 Key Features

* **Dual Model Architectures**:
  * **`BreastNet`**: Custom medical Convolutional Neural Network with **Squeeze-and-Excitation (SE)** channel attention blocks for acoustic feature refinement.
  * **Transfer Learning Backbones**: Pre-trained ResNet-18, ResNet-50, and EfficientNet with customized diagnostic dropout heads.
* **Medical Data Augmentations**:
  * Simulates clinical transducer angle variations, acoustic gain changes, affine rotations, and flips.
  * Automatic inverse-frequency **class weighting** to mitigate medical class imbalance.
* **Clinical Performance Metrics**:
  * Sensitivity (Recall) — prioritizes eliminating False Negatives.
  * Specificity — prevents unnecessary invasive biopsies.
  * Macro F1-score and multi-class One-vs-Rest ROC-AUC.
* **Visual Interpretability with Grad-CAM**:
  * Highlights suspect hypoechoic acoustic zones directly on the ultrasound scan to aid radiological review.
* **Dual Execution Modes**:
  * **Interactive Jupyter Notebook** ([salah.ipynb](file:///d:/breast_cancer/salah.ipynb)): Step-by-step guided exploration, training, and visualization.
  * **Modular CLI Scripts**: Clean production-ready Python modules for automation.

---

## 📂 Directory Structure

```text
breast_cancer/
├── data/
│   ├── raw/                       # Raw source images (benign, malignant, normal)
│   ├── processed/                 # Stratified splits (train: 70%, val: 15%, test: 15%)
│   └── sample_generator.py        # Realistic B-mode ultrasound synthesizer
├── src/
│   ├── __init__.py
│   ├── config.py                  # Hyperparameters, device settings, and class mapping
│   ├── dataset.py                 # PyTorch Dataset, DataLoaders, and Augmentations
│   ├── models.py                  # BreastNet (SE-Attention) and Transfer Learning architectures
│   ├── train.py                   # Training loop with Early Stopping & LR Scheduling
│   ├── evaluate.py                # Test set evaluation, Confusion Matrix, and ROC-AUC
│   ├── explainability.py          # Grad-CAM (Gradient-weighted Class Activation Mapping)
│   ├── predict.py                 # Single and batch image inference engine
│   └── utils.py                   # Checkpointing, metrics calculation, and plotting
├── scripts/
│   ├── download_data.py           # Ingestion and stratified split script
│   ├── run_pipeline.py            # Master CLI runner executing data prep, training, eval & Grad-CAM
│   └── make_notebook.py           # Programmatic generator for salah.ipynb
├── tests/
│   └── test_pipeline.py           # Unit and integration tests
├── models_saved/                  # Saved model checkpoints (.pth)
├── reports/
│   └── figures/                   # Learning curves, confusion matrices, ROC curves, Grad-CAM overlays
├── salah.ipynb                    # Complete interactive Jupyter Notebook
├── requirements.txt               # Project dependencies
└── README.md                      # Documentation
```

---

## 🛠️ Installation & Setup

1. **Clone or Navigate to the Workspace**:
   ```bash
   cd d:\breast_cancer
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🖥️ Usage Guide

### Streamlit Diagnostic Assistant
Launch the focused image-analysis interface:
```bash
streamlit run app.py
```
Upload a PNG or JPG, select **Analyze**, and review the prediction with its Grad-CAM overlay. The app uses the validation-selected ResNet-18 BUSI checkpoint. Its local image-level split contains some near-duplicate images across splits, so reported metrics may be optimistic; it has not been independently or clinically validated and must not be used to diagnose, rule out cancer, or guide treatment.

### Train on the BUSI dataset
Use the official class folders (`benign`, `malignant`, `normal`) and exclude segmentation masks:
```powershell
python scripts/train_busi.py --source "C:\path\to\Dataset_BUSI_with_GT" --epochs 15
```
This creates a separate 70/15/15 split under `data/processed_busi/`, writes the selected checkpoint under `models_saved/`, and reports per-class metrics plus macro-F1 and ROC-AUC in `reports/`. To use pretrained ResNet-18, add `--model resnet18`; its ImageNet weights are cached by torchvision. If the output split already exists, the script stops rather than replacing it; reuse a verified split with `--reuse-splits`.

### 1. Run the Entire End-to-End Pipeline
Execute data preparation, training, evaluation, and Grad-CAM generation in one command:
```bash
python scripts/run_pipeline.py --model breastnet --epochs 10 --batch-size 16
```

### 2. Train a Model
Train `BreastNet` or a transfer learning backbone:
```bash
# Train BreastNet
python src/train.py --model breastnet --epochs 15 --batch-size 16 --lr 0.0001

# Train ResNet-18 Transfer Learning
python src/train.py --model resnet18 --epochs 15 --batch-size 16 --lr 0.0001
```

### 3. Evaluate Model on Test Set
Evaluate the trained checkpoint and generate the clinical confusion matrix and ROC curves:
```bash
python src/evaluate.py --model breastnet --checkpoint models_saved/best_breast_cancer_model.pth
```

### 4. Run Diagnosis on a New Ultrasound Scan
Predict diagnosis and generate an explainability heatmap for any image:
```bash
python src/predict.py --image data/processed/test/malignant/malignant_0001.png --model breastnet --explain
```

---

## 📓 Interactive Jupyter Notebook

Open and run [salah.ipynb](file:///d:/breast_cancer/salah.ipynb) inside JupyterLab, VS Code, or Antigravity IDE:
* Full markdown tutorials on ultrasound interpretation.
* Interactive visualizations of scans across diagnostic classes.
* Live training loss & accuracy graphs.
* High-resolution Confusion Matrix heatmap & ROC curves.
* Interactive Grad-CAM widget visualizing where the model focuses.

---

## 🌐 Using the Official BUSI Dataset

To use the official **Breast Ultrasound Images (BUSI)** benchmark:
1. Download from Kaggle: [Breast Ultrasound Images Dataset (BUSI)](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset).
2. Unzip into `data/raw/` so the folders are organized as:
   ```text
   data/raw/
   ├── benign/
   ├── malignant/
   └── normal/
   ```
3. Run the partitioning script:
   ```bash
   python scripts/download_data.py --source data/raw
   ```
4. Re-run training:
   ```bash
   python src/train.py --model breastnet --epochs 25
   ```

---

## 📊 Evaluation Metrics Guide

| Metric | Clinical Importance |
| :--- | :--- |
| **Sensitivity (Recall)** | Minimizes **False Negatives** — ensures malignant cancers are not missed. |
| **Specificity** | Minimizes **False Positives** — avoids unnecessary biopsy trauma and patient anxiety. |
| **F1-Score** | Harmonic mean balancing precision and recall across imbalanced medical classes. |
| **ROC-AUC** | Evaluates diagnostic discrimination threshold independence across all classes. |
| **Grad-CAM** | Provides transparent visual validation of AI reasoning for clinical auditability. |
