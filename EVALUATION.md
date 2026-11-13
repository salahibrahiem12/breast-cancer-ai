# Evaluation & Cross-Source Generalization Report

This document reports the empirical validation metrics for BreastCare AI across locked evaluation experiments.

---

## 1. Locked Evaluation Methodology

1. **Protocol Lock:** Evaluation scripts (`ml/locked_evaluation.py`) are cryptographically locked and deterministic (`torch.manual_seed(42)`).
2. **Patient Isolation:** No duplicate or patient-level data crosses train and test folds.
3. **Leave-One-Source-Out (LOSO):**
   - Model is trained exclusively on Dataset $A$ (e.g., BUS-BRA).
   - Evaluated without fine-tuning on Dataset $B$ (TCIA) and Dataset $C$ (Mendeley).

---

## 2. Experimental Metrics Comparison

### Frozen Linear Baseline (v1)
- **Feature Extractor:** Frozen ImageNet-pretrained ResNet-18.
- **Classifier:** L2-regularized logistic regression with balanced class weights.

| Source Cohort | ROC-AUC | Balanced Accuracy | Sensitivity | Specificity | Brier Score (Calibration) |
|---|---|---|---|---|---|
| TCIA BrEaST | 0.6821 | 0.6235 | 0.6681 | 0.5789 | 0.2214 |
| Mendeley Data | 0.6718 | 0.5462 | 0.6014 | 0.4910 | 0.2458 |
| BUS-BRA | 0.5896 | 0.5669 | 0.6318 | 0.5020 | 0.2489 |

### Neural Adapter Head (v1 Finetune)
- **Architecture:** 2-layer MLP adapter head (512 -> 128 -> 2) with Dropout (p=0.3) on frozen encoder.
- **Result:** Did not show statistically significant improvement over linear baseline on held-out TCIA (ROC-AUC 0.679 vs 0.682), confirming baseline preservation rules.

### GPU Encoder Fine-Tuning (v3 Colab Protocol)
- **Protocol State:** Packaged and verified in `reports/gpu_encoder_finetune_v3/breastcare_colab_v3_bundle.zip` for remote execution on Google Colab GPU instances. Local CPU execution correctly bypassed.

