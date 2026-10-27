# 🎗️ Experimental Breast Ultrasound Classification Research Project

This project contains an experimental image classifier. It is not a medical
device, has not been clinically validated, and must not be used to diagnose,
exclude, or guide treatment for breast cancer.

> **Safety status:** Classification is disabled in the Streamlit app and
> inference APIs. The project does not have a validated detector to verify that
> an uploaded image is a breast ultrasound. Until such a detector is trained
> and independently evaluated, every image is rejected rather than assigned a
> cancer class.

---

## Research context

This project experiments with classifying BUSI dataset images. Its outputs
must not be interpreted as cancer diagnoses or used to guide clinical care.

---

## 🚀 Key Features

* **Experimental Model Architectures**:
  * `BreastNet` and transfer-learning backbones are research code, not validated
    diagnostic models.
* **Training Augmentations**: Rotations, flips, affine transforms, and color
  jitter are experimental choices, not a claim of clinical realism.
* **Experimental Metrics**: Historical image-level split measurements are
  included for research context only. They are not patient-independent or
  clinical validation.
* **Visual Interpretability with Grad-CAM**:
  * Exploratory visualization only; Grad-CAM is not validated lesion detection
    or localization.
* **Dual Execution Modes**:
  * **Interactive Jupyter Notebook** ([salah.ipynb](file:///d:/breast_cancer/salah.ipynb)): Guided exploration and training; inference cells now display the safety gate instead of a diagnosis.
  * **Modular CLI Scripts**: Research and experiment utilities.

---

## 📂 Directory Structure

```text
breast_cancer/
├── data/
│   ├── raw/                       # Raw source images (benign, malignant, normal)
│   ├── processed/                 # Stratified splits (train: 70%, val: 15%, test: 15%)
│   └── sample_generator.py        # Synthetic examples for software tests only
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
│   ├── run_pipeline.py            # Research data prep, training, and evaluation
│   └── make_notebook.py           # Programmatic generator for salah.ipynb
├── tests/
│   └── test_inference_safety.py   # Fail-closed inference regression tests
├── models_saved/                  # Saved model checkpoints (.pth)
├── reports/
│   └── figures/                   # Research learning curves, confusion matrices, and ROC curves
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

### Streamlit research interface
Launch the research interface:
```bash
streamlit run app.py
```
The upload flow currently fails closed and displays
`Unsupported image: Unable to verify that this is a breast ultrasound image.`
for every image. No prediction, confidence, or Grad-CAM overlay is generated.
The model's softmax scores are uncalibrated and cannot establish whether an
image is in-distribution.

See [RELIABILITY_AUDIT.md](./RELIABILITY_AUDIT.md) for the root-cause analysis,
historical metrics, regression tests, and remaining validation requirements.

### Train on the BUSI dataset
Use the official class folders (`benign`, `malignant`, `normal`) and exclude segmentation masks:
```powershell
python scripts/train_busi.py --source "C:\path\to\Dataset_BUSI_with_GT" --epochs 15
```
This creates a separate 70/15/15 split under `data/processed_busi/`, writes the selected checkpoint under `models_saved/`, and reports per-class metrics plus macro-F1 and ROC-AUC in `reports/`. To use pretrained ResNet-18, add `--model resnet18`; its ImageNet weights are cached by torchvision. If the output split already exists, the script stops rather than replacing it; reuse a verified split with `--reuse-splits`.

### 1. Run the Entire End-to-End Pipeline
Execute data preparation, training, and experimental evaluation in one command.
Grad-CAM and image prediction are intentionally disabled:
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

### 3. Evaluate Model on the existing image-level split
Generate experimental classification metrics and plots. These are not
independent patient-level or clinical validation:
```bash
python src/evaluate.py --model breastnet --checkpoint models_saved/best_breast_cancer_model.pth
```

### 4. Try the prediction CLI (currently blocked)
The CLI rejects all images until a validated modality detector is available:
```bash
python src/predict.py --image data/processed_busi/test/malignant/malignant_0001.png
```

---

## 📓 Interactive Jupyter Notebook

Open and run [salah.ipynb](file:///d:/breast_cancer/salah.ipynb) inside JupyterLab, VS Code, or Antigravity IDE:
* Full markdown tutorials on ultrasound interpretation.
* Interactive visualizations of scans across diagnostic classes.
* Live training loss & accuracy graphs.
* High-resolution Confusion Matrix heatmap & ROC curves.
* Inference and Grad-CAM are disabled until image eligibility is validated.

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

| Metric | Research interpretation (not clinical validation) |
| :--- | :--- |
| **Sensitivity (Recall)** | Fraction of reference-labeled examples recovered for a class. |
| **Specificity** | Fraction of other reference-labeled examples not assigned to a class. |
| **F1-Score** | Harmonic mean of precision and recall on the sampled image split. |
| **ROC-AUC** | Ranking metric on the sampled image split; not evidence of clinical utility. |
| **PR-AUC / average precision** | Class-ranking summary on the sampled image split; not validation of eligibility or calibration. |
| **Grad-CAM** | Exploratory activation visualization only; it does not prove model reasoning or localize a lesion. |
