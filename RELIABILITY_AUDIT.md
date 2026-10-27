# BreastCare AI reliability audit

**Status:** Classification is disabled until a breast-ultrasound eligibility
detector has been independently validated. This is a research prototype, not a
medical device or a substitute for clinical assessment.

## Root cause

The deployed classifier is a closed-set ResNet-18 with three outputs:
`benign`, `malignant`, and `normal`. The app decoded an uploaded file, resized
it to 224 × 224, converted it to RGB, applied ImageNet normalization, and
always selected the largest softmax output. There was no out-of-distribution
(OOD) or modality gate. A book photograph therefore still received one of the
three classes; softmax does not verify that an input is a breast ultrasound.

No separate image-modality model, training data for such a detector, validated
eligibility threshold, or OOD evaluation set was found. File type, dimensions,
grayscale conversion, and basic image statistics cannot prove image modality.
Training a detector or claiming an OOD rejection rate without appropriate,
diverse labeled examples would be unsupported. Accordingly, the app and every
image prediction/explanation API now fail closed with:

> Unsupported image: Unable to verify that this is a breast ultrasound image.

This currently rejects valid ultrasound images too. That is an intentional
safety limitation, not a claim that the requested modality detection has been
solved.

## Implementation review and changes

- **Class mapping:** `CLASS_NAMES` is `benign`, `malignant`, `normal`; dataset
  labels and output indices use this same order. No label-order mismatch was
  identified in the inspected inference/training code.
- **Image preprocessing:** training augmentation and deterministic evaluation
  both resize, convert to tensor, and apply ImageNet normalization. Inference
  and Grad-CAM use the deterministic evaluation transform. The RGB handling is
  consistent; it does not provide modality validation.
- **Model behavior:** softmax on three closed-set logits always sums to one.
  The top value is not a calibrated probability. No temperature scaling,
  reliability diagram, Brier score, or expected calibration error was found.
  Confidence and per-class softmax percentages were removed from the app.
- **Malignant bias:** historical test predictions contain 31 malignant calls
  among 119 images (26.1%), versus 32 malignant ground-truth images (26.9%).
  There were 26 true-positive malignant predictions and 5 false-positive
  malignant calls; 6 malignant cases were missed (5 called benign, 1 called
  normal). These image-level results do not explain or make safe a malignant
  prediction on an unrelated image.
- **Class balance:** the saved image-level split contains train counts
  305/146/93, validation 65/31/19, and test 66/32/21 (benign/malignant/normal).
  The BUSI trainer computes inverse-frequency weights from training data.
- **Overfitting/leakage:** saved ResNet-18 train accuracy/macro-F1 were 95.4% /
  0.952, versus validation 86.1% / 0.845 and test 83.2% / 0.818. The split
  script randomly partitions images, not patients. Near-duplicate image pairs
  were previously found across train/validation and train/test. Patient IDs
  are unavailable, so patient-level independence and residual leakage cannot
  be verified.
- **Clinical metrics:** the existing test split is not a genuinely independent
  patient-level set. It was not reused to tune this fix. The historical values
  below are descriptive only; they do not justify diagnostic use.
- **Lesions and size:** no validated detection/segmentation model or known
  image scale was found. The interface no longer claims lesion detection,
  lesion sizing, or clinically validated Grad-CAM interpretation.

## Historical image-level metrics

These are from the saved ResNet-18 report and are not recalculated as part of
this safety change. PR-AUC and calibration metrics were not saved, so they are
reported below only as additional descriptive calculations on the same
previously used image-level test split. Neither is independent validation.

| Class | Support | Precision | Recall / Sensitivity | F1 | Specificity | OvR ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Benign | 66 | 0.862 | 0.848 | 0.855 | 0.830 | 0.915 |
| Malignant | 32 | 0.839 | 0.813 | 0.825 | 0.943 | 0.941 |
| Normal | 21 | 0.739 | 0.810 | 0.773 | 0.939 | 0.939 |
| **Overall / macro** | **119** | — | — | **0.818 macro-F1** | **0.904 macro** | **0.931 macro** |

Accuracy: **0.832**. Confusion matrix (rows = actual; columns = predicted;
class order benign, malignant, normal):

```text
[[56,  5,  5],
 [ 5, 26,  1],
 [ 4,  0, 17]]
```

Calibration is **not validated**. OOD/modality rejection sensitivity and
specificity are **not measurable from the available data**. Independent
patient-level performance is **not available**.

Additional one-pass, descriptive computations from the same 119 test images:

| Metric | Value |
|---|---:|
| Macro average precision (OvR; stepwise PR-AUC) | 0.894 |
| Benign average precision | 0.933 |
| Malignant average precision | 0.854 |
| Normal average precision | 0.894 |
| Multiclass Brier score (sum over classes) | 0.248 |
| Top-label ECE (10 equal-width bins) | 0.047 |

These are not patient-independent estimates or evidence of satisfactory
calibration. The ECE is bin-dependent and estimated from only 119 images; no
calibration method or threshold was fitted using this set.

## Regression coverage and verification

`tests/test_inference_safety.py` exercises the prediction and explanation
entrypoints with generated illustrative non-ultrasound-like images,
low-quality/corrupted images, and (when local BUSI files are available)
benign/malignant/normal images. It also checks the exact Streamlit error. A
sentinel model verifies that classification is never invoked. This tests the
fail-closed behavior, **not** the accuracy of ultrasound detection or rejection
of real-world book/natural/other-medical photographs.

## Required work before enabling classification

1. Obtain a representative, consented modality dataset containing breast
   ultrasound scans and diverse non-ultrasound/other-medical images, with
   patient-level identifiers and independent sites/devices where possible.
2. Train a separate eligibility/OOD detector using patient-grouped splits.
   Choose any operating threshold only on validation data and predefine the
   acceptable false-accept rate.
3. Measure sensitivity/specificity and confidence intervals on a locked,
   patient-independent external test set, including book/photos, screenshots,
   unrelated medical images, blur, corruption, and low-quality scans.
4. Evaluate class calibration, PR-AUC, per-class confusion/error slices and
   malignant false negatives on an independent patient-level test set.
5. Only then consider enabling classification, with clinical oversight and
   prospective validation. Do not treat softmax scores or Grad-CAM as
   calibrated uncertainty or validated lesion localization.

No model was retrained, no threshold was invented, and no accuracy improvement
is claimed in this audit.
