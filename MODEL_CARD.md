# Model Card: BreastCare AI Deep Feature Encoders

## Model Overview
- **Model Name:** BreastCare AI ResNet-18 Deep Feature Encoder
- **Architecture:** 18-layer Deep Residual Convolutional Neural Network (ResNet-18) with layer4 spatial feature extraction (512-dim continuous embedding).
- **Secondary Modality Classifier:** Rule-based statistical physics descriptor engine measuring frequency-domain entropy, aspect ratio, air margin boundary, and bilateral lung column structures.
- **Explainability Mechanism:** Gradient-weighted Class Activation Mapping (Grad-CAM) computed on `layer4` conv activations.

---

## Intended Use & Target Users
- **Primary Domain:** Investigational decision-support research on breast ultrasound (B-mode) and digital mammography (CC/MLO).
- **Target Audience:** Medical imaging researchers, clinical data scientists, and healthcare AI engineers.
- **Out of Scope & Prohibited Uses:**
  - Primary diagnosis or autonomous staging of breast lesions.
  - Triage or referral decisions in clinical practice without direct radiologist oversight.
  - Analysis of non-breast medical scans (chest X-rays, abdominal CT, brain MRI) or non-medical consumer photography.

---

## Performance Summary (Frozen Linear Baseline v1)

Evaluated under locked cross-source protocol:

| Test Dataset (Held-Out) | Sample Count | ROC-AUC | Balanced Accuracy | Sensitivity (Malignant Recall) | Specificity (Benign Recall) |
|---|---|---|---|---|---|
| **TCIA BrEaST** | 541 | 0.6821 | 0.6235 | 66.8% | 57.9% |
| **Mendeley Data** | 256 | 0.6718 | 0.5462 | 60.1% | 49.1% |
| **BUS-BRA (Cross-fit)** | 1,418 | 0.5896 | 0.5669 | 63.2% | 50.2% |

*Key Takeaway:* Domain shift across different ultrasound hardware manufacturers and clinical acquisition settings causes substantial performance degradation, underscoring why fail-closed gating is mandatory.

---

## Safety Controls
- Default State: `BREASTCARE_CLASSIFICATION_ENABLED=false`.
- The system returns `"Unable to provide a reliable assessment. Specialist review is recommended."` whenever out-of-distribution or quality criteria fail.

