# TCIA pipeline audit and separate development comparison

Completed 2026-11-03. Classification remains disabled. TCIA is an **inspected
external evaluation set**, not an untouched final test. No threshold or model
selection was performed on it. The replay below checks the same fixed model;
neither new development candidate was evaluated on TCIA.

## Pipeline and ground-truth audit

- Checkpoint: `mendeley_binary_resnet18_v1.pth`, SHA-256
  `69b0ef99a4d0c0f744ed1f809d12742250baa250c43969b85c0f9a5e720c6045`.
  Strict state-dictionary loading succeeded for torchvision ResNet-18 with
  its two-output head. Selected epoch 6; temperature 1.0633265716371612.
- Class order is exactly `benign` index 0, `malignant` index 1. Ground truth
  was joined by the source `Image_filename` and checked against the original
  workbook's `Classification` and `Verification`, not inferred from folder
  order or mask colors. All **197 mappings matched**.
- Each prepared image's SHA-256 matched the audited original image inventory.
  The preparation/inventory tools verify original ZIP and workbook hashes.
  Primary PNGs are loaded unchanged; masks are separate files and never model
  inputs or independent samples. Multiple `_other` annotations do not create
  additional cases or observations.
- Evaluation converts the full frame to RGB, stretches it to **224x224** using
  torchvision/PIL bilinear resize with antialiasing, scales pixels to [0,1],
  and normalizes by mean (0.485,0.456,0.406), std (0.229,0.224,0.225).
  There is no evaluation augmentation, lesion crop, mask overlay, or automatic
  border removal. These settings match the checkpoint metadata and existing
  training/evaluation code. Aspect ratio is not preserved.
- Malignant score is softmax(logits / recorded temperature), index 1. The
  fixed decision rule is malignant score >= **0.5**. There were no exact ties;
  argmax and the threshold therefore produced identical labels.
- A latent exact-tie inconsistency was corrected for future evaluations:
  binary prediction and calibration calculations now consistently use the
  documented >=0.5 rule instead of argmax's first-index tie break. Regression
  coverage checks either class order. TCIA has zero ties, so this correction
  does not change its recorded results or explain the observed failures.
- Exact-checkpoint replay reproduced all saved decisions. Maximum absolute
  probability difference was **4.993e-9**, within eight-decimal CSV rounding.
  No class reversal, threshold mismatch, or mask-as-image error was found.

Manifest SHA-256:
`7cac20c3a84afcdabf9b348c08e07dfdc481f324395919707a2454fbf5109fe7`.
Saved prediction CSV SHA-256:
`3ddc3257fbcc0c69192128b800de9aa191676e8c3f63e5bec4b85c06b8da829d`.

## What 197 means

The [source paper](https://pubmed.ncbi.nlm.nih.gov/38297002/) and official TCIA
collection describe **256 exported scan images from 256 patients**. The
workbook contains one primary image and one distinct case identifier per row.
The selected subset has **197 unique case images and 197 source-reported
patient groups**, one observation per group: 99 benign and 98 malignant.
It is not a count of separate lesions. Separate examination identifiers are
not provided, so an independently verified examination count cannot be claimed.
Case identifiers are source-backed pseudonymous group keys, not independently
validated identities, and cross-source identity matching is unavailable.

Only `confirmed by biopsy` cases with binary benign/malignant labels were
included. The 55 follow-up-only cases and four normal cases were excluded
before predictions. The paper notes that biopsy results for 12 BI-RADS 3
cases precede the follow-up scan; contemporaneous biopsy timing should not be
assumed for every image. This is a biopsy-selected population, not screening
prevalence, and source labels have not been independently clinically adjudicated.

| Actual class | Predicted benign | Predicted malignant | Total |
|---|---:|---:|---:|
| Benign | 6 | 93 | 99 |
| Malignant | 9 | 89 | 98 |
| Total | 15 | 182 | 197 |

## Measured classifier metrics

Malignant is the positive class. Intervals are **2,000-resample nonparametric
patient-group percentile bootstrap**, seed 42. All groups contain one image,
so this equals patient-level resampling under the publisher's grouping claim.
All reported metrics had 2,000 valid resamples. These intervals quantify
sampling variability within this selected cohort, not clinical transportability
or label uncertainty. Calculations use saved, eight-decimal probabilities.

| Metric | Estimate | 95% interval |
|---|---:|---:|
| Sensitivity | 0.9082 (89/98) | 0.8468–0.9600 |
| Specificity | 0.0606 (6/99) | 0.0190–0.1100 |
| Precision / PPV | 0.4890 (89/182) | 0.4171–0.5604 |
| NPV | 0.4000 (6/15) | 0.1429–0.6667 |
| Accuracy | 0.4822 (95/197) | 0.4162–0.5533 |
| Balanced accuracy | 0.4844 | 0.4471–0.5205 |
| ROC-AUC | 0.4630 | 0.3792–0.5431 |
| Malignant PR average precision | 0.4736 | 0.3936–0.5785 |
| Malignant PR-AUC, trapezoidal | 0.4673 | 0.3835–0.5636 |
| Binary Brier score | 0.3836 | 0.3346–0.4314 |
| Two-class summed Brier score | 0.7671 | 0.6692–0.8629 |
| Log loss | 1.1419 | 0.9994–1.2880 |
| Top-label ECE, 10 equal-width bins | 0.3489 | 0.2835–0.4185 |

The earlier 0.4818 PR result was **macro average precision over both classes**.
It is not malignant-class PR-AUC. Both AP and trapezoidal PR integration are
named explicitly here. Malignant prevalence is 98/197 = 0.4975, the reference
prevalence for interpreting AP and predictive values. PPV and especially NPV
should not be transferred to a screening population; NPV has only 15
predicted-negative observations. The two-class Brier convention is exactly
twice binary Brier. ECE depends on binning/sample size and is descriptive.
No calibration was fitted on TCIA.

## Local error and acquisition review

All **93 false positives and nine false negatives** were reviewed in seven
local contact-sheet pages using neutral FP/FN numbers rather than case IDs.
A deterministic 16-image development sample was also reviewed. The sheets
omit the outer 15% left/right, 20% top and 15% bottom for privacy; these are
review crops only, not model inputs. They can obscure peripheral findings and
metadata. This was non-clinical artifact review, not radiologist adjudication,
relabeling, or a check that each lesion is diagnostically visible.

Measured and observed differences:

- TCIA median native dimensions: **503x537**, range width 363–1024 and height
  338–618. Mendeley development median: **139x92**. Development thumbnails
  show much coarser pixelation and tighter fields than the larger TCIA images.
  The same 224x224 resize therefore upsamples much of development data and
  downsamples TCIA, while distorting different native aspect ratios.
- False-negative median width/aspect ratio: **831 / 1.545**; false-positive:
  **545 / 1.024**. True negatives also tend to be wide (median aspect 1.680),
  so width alone is not a proven explanation of missed malignant cases.
- Median whole-image near-black fraction (gray <10/255): **22.2% in FN**,
  **0.42% in FP**, **21.3% in TN**, **0.40% in TP**. This includes dark tissue
  and lesion regions as well as borders; it is not a pure frame detector.
- Across the inspected cohort, score correlations (Spearman) were **-0.432**
  with near-black fraction and **-0.369** with aspect ratio. Near-black
  fraction's correlation with ground truth was approximately **0.001**.
  These exploratory associations are consistent with nuisance sensitivity,
  but acquisition, lesion appearance and other confounding remain unresolved.
  No causal attribution, multiplicity-adjusted inference, or TCIA-based
  cropping/threshold optimization is claimed.
- In the outer 10% band, median near-black fraction was **54.0% in FN**,
  **52.6% in TN**, **0.57% in FP**, and **0.37% in TP**. Its score correlation
  was **-0.411**, versus **0.021** with ground truth. This provides a more
  frame-focused descriptive check, but dark peripheral anatomy can still
  contribute; it is not proof that borders caused any individual error.
- False positives were often high-score: range **0.551–0.985**, median
  **0.883**. False negatives ranged **0.302–0.490**, median **0.460**. The
  failure is not confined to a few exact-threshold ties.
- The primary source documents two Polish centers and five scanners (Hitachi,
  Esaote, Samsung, and two Philips models). The clinical workbook does not
  supply a per-case scanner/site field, preventing device-stratified analysis.
  Corresponding Mendeley acquisition metadata are unavailable.
- Byte identity and code inspection establish that colored segmentation masks
  were not composited into classifier inputs. Some source images contain
  colored pixels; color fraction alone cannot identify annotations. Bright
  border pixel counts are recorded but do not identify text or prove its
  absence. Outer-frame text was not displayed in the privacy-cropped review.

Source-paper snapshot, error sheets, per-outcome dimension/border/intensity
summaries, and artifact associations remain local under
`reports/tcia_pipeline_audit_v1/` and `reports/tcia_audit_2026-11-03/`.
No source patient/case identifiers or raw images are published in this report.

## Feasible separate development work

The [development protocol](DEVELOPMENT_COMPARISON_PROTOCOL_2026-11-03.md)
was recorded before fitting the two new classical models. Both use only the
existing duplicate-controlled Mendeley development split: **121 train images
in 39 groups; 41 validation images in 12 groups**. All **88 old-test-linked
images in 21 groups** remain excluded from feature fitting and validation.
All 250 source image hashes and perceptual hashes were checked. Cross-split
dHash distances <=5 or exact matches cause rejection. Patient IDs remain
unavailable, so none of these groups can be called independent patients.

Baseline: grayscale stretch to 32x32, train-only StandardScaler, balanced L2
logistic regression, C=1, lbfgs, max_iter=2000, seed 42. Candidate: 1% per-image
autocontrast plus aspect-preserving resize/padding, with identical classifier
settings. Full images are retained; no lesion crop or annotation is used.
Both use fixed threshold 0.5. There was no parameter sweep or TCIA scoring.

| Validation metric | Baseline | Aspect/contrast candidate |
|---|---:|---:|
| TN / FP / FN / TP | 8 / 10 / 0 / 23 | 5 / 13 / 2 / 21 |
| Accuracy | 0.7561 | 0.6341 |
| Balanced accuracy | 0.7222 | 0.5954 |
| Sensitivity | 1.0000 | 0.9130 |
| Specificity | 0.4444 | 0.2778 |
| ROC-AUC | 0.9420 | 0.8357 |
| Malignant average precision | 0.9550 | 0.8870 |
| Binary Brier | 0.1733 | 0.2573 |

**The candidate did not improve validation performance and was not adopted.**
The baseline is a reproducible development reference, not a clinically
validated replacement. Validation was reused in earlier experiments and is
not independent testing. Duplicate-group bootstrap intervals are very broad:
baseline balanced accuracy 0.50–1.00; candidate 0.4130–0.8949. A zero observed
baseline FN count produces a degenerate sensitivity bootstrap interval; this
does not establish perfect sensitivity. Coefficients and scaler parameters
were saved without executable pickle, and reconstructed validation
probabilities matched exactly for both candidates.

Local experiment outputs: `reports/development_baseline_comparison_v1/`.
No production checkpoint, serving configuration, or eligibility model changed.

## Remaining blockers and cohort reservation

The [cohort ledger](COHORT_USAGE_LEDGER.md) marks TCIA as inspected and all
Mendeley partitions as development or previously inspected data. **No fresh
independent final cohort is available or reserved.** The 59 excluded TCIA
cases do not provide a separately sourced fresh final cohort.

Further development needs patient-mapped data, broader acquisition coverage,
independent clinical error review, and a new untouched final cohort under a
fixed protocol. No claim of improved external performance can be made from
this development comparison. Eligibility-gate performance remains completely
separate and unmeasured; no classifier result establishes modality rejection.
Classification remains disabled and cannot be enabled by these scripts.

## Reproduction and software checks

- `python -m ml.audit_tcia_evaluation --help`: verify fixed-checkpoint mappings,
  replay probabilities, statistics, image summaries and local error sheets.
- `python -m ml.development_baseline --help`: fit both predeclared candidates
  using train/validation only. Commands refuse existing output directories.
- Full software regression after the threshold consistency fix: **70 passed,
  29 subtests passed**, one upstream Starlette/AnyIO deprecation warning,
  67.79 seconds. Local XML/log:
  `reports/software_regression_audit_final_2026-11-03.*`.
- These tests check statistics, grouping, data integrity, feature extraction
  and software behavior. They do not establish clinical reliability.
- Final targeted checks after the calibration tie fix: **8 passed, 3 subtests
  passed**. The original 197-case scores were not changed or retuned.

## Files changed for this audit

| File | Change |
|---|---|
| `ml/audit_tcia_evaluation.py` | Fixed-checkpoint replay, ground-truth/unit checks, complete binary metrics and group intervals, local privacy-cropped error sheets, artifact summaries. |
| `ml/development_baseline.py` | Separate development-only baseline/candidate fitting, integrity/duplicate guards, saved numerical parameters and replay verification. |
| `ml/evaluate.py` | Consistent binary >=0.5 tie handling for decisions and calibration. |
| `tests/test_audit_development.py` | Metrics, undefined inputs, clustered resampling, split safeguards, feature extraction and tie regression tests. |
| `.gitignore` | Include the new regression test in version control. |
| `docs/DEVELOPMENT_COMPARISON_PROTOCOL_2026-11-03.md` | Predeclared separate development comparison. |
| `docs/COHORT_USAGE_LEDGER.md` | Inspected-set and unavailable-final-cohort status. |
| `docs/TCIA_PIPELINE_AUDIT_2026-11-03.md` | This audit and measured results. |
| `docs/TCIA_EXTERNAL_EVALUATION_PROTOCOL_2026-11-03.md` | Current inspected-cohort status; original protocol retained. |
| `docs/MODEL_CARD.md`, `docs/PROGRESS.md`, `README.md` | Audit findings, AP definition, negative development result and remaining limitations. |

Aggregate figure: `reports/tcia_pipeline_audit_v1/audit_summary.png`.
Local audit data: `reports/tcia_pipeline_audit_v1/audit.json`.
No serving configuration, frontend code, or production checkpoint was changed.
