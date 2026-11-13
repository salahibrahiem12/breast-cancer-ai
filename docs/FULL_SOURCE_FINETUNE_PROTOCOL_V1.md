# Full-eligible source-held-out fine-tuning protocol v1

Predefined on 4 November 2026 after completing and verifying the full-eligible
frozen baseline ([`FULL_SOURCE_BASELINE_RESULTS_V1.md`](FULL_SOURCE_BASELINE_RESULTS_V1.md)).
All three breast-ultrasound sources (Mendeley, TCIA BrEaST, BUS-BRA) have already
been inspected. This experiment is an exploratory cross-source model-development
comparison, not an untouched final clinical test. Public classification remains
disabled (`BREASTCARE_CLASSIFICATION_ENABLED=false`).

## 1. Objective and Justification

The earlier 220-image cross-source fine-tuning experiment (`reports/cross_source_v1`)
was a compute-limited pilot: it sampled only 16 groups per source/class and ran
only two epochs (8–18 gradient updates) without ultrasound augmentations,
epoch-level learning curves, or validation early stopping. The subsequent
full-data frozen baseline (`reports/full_source_baseline_v1`) used all 2,232
eligible images and verified L-BFGS-B convergence across all 15 linear fits,
achieving held-out group ROC-AUC / balanced accuracy of:
- Mendeley: ROC-AUC 0.706, BA 0.573
- TCIA: ROC-AUC 0.688, BA 0.644
- BUS-BRA: ROC-AUC 0.594, BA 0.570

To test whether nonlinear adaptation with anatomy-preserving ultrasound
augmentations and training-side validation early stopping improves cross-source
generalization when given all 2,232 eligible images and an adequate epoch budget,
this protocol defines one controlled fine-tuning experiment (`full_source_finetune_v1`).

## 2. Dataset, Grouping, and Partitions

- **Eligible inventory:** All **2,232 eligible images** in **1,298 global duplicate/case groups**
  from `reports/design_audit_v1/eligible_manifest.json` (reconciled in
  `data/prepared/unified_dataset_manifest_v1.csv`):
  - **Mendeley v1:** 162 images / 38 duplicate groups (71 benign / 17 groups; 91 malignant / 21 groups).
    88 images in duplicate groups linked to the historical test split remain excluded.
  - **TCIA BrEaST v1:** 197 biopsy-confirmed images / 197 publisher-reported patient IDs
    (99 benign, 98 malignant). 55 follow-up-only benign and 4 normal cases remain excluded.
  - **BUS-BRA v1:** 1,873 images / 1,063 source Cases (1,266 benign / 721 Cases; 607 malignant / 342 Cases).
    Two benign images in Case 54 remain conservatively excluded for a near-duplicate hash match.
- **Outer evaluation splits:** Three source-held-out folds (`mendeley`, `tcia`, `busbra`).
  Every group belonging to the held-out source is completely absent from feature
  standardization, sample weighting, neural training, early stopping, checkpoint
  selection, Platt calibration, and threshold selection.
- **Inner training/validation splits:** Exactly the same 4-fold
  `StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)` stratified by
  `(source, label)` and grouped by `global_group` used in `full_source_baseline_v1`.

## 3. Pretrained Backbone, Preprocessing, and Ultrasound Augmentations

- **Pretrained backbone:** Official ImageNet `ResNet-18` weights
  (`reports/bounded_development_v1/preserved/pretrained_resnet18.pth`,
  SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`).
- **Class order:** `["benign", "malignant"]` (`benign=0`, `malignant=1`).
- **Deterministic evaluation transform:** Convert to RGB, full-frame bilinear
  resize to `224 × 224`, `ToTensor()`, and ImageNet normalization
  `mean=(0.485, 0.456, 0.406)`, `std=(0.229, 0.224, 0.225)`.
- **Anatomy-preserving ultrasound augmentations (training splits only):**
  - B-mode breast ultrasound has directional depth semantics (skin line at top,
    chest wall and posterior acoustic shadowing/enhancement at bottom). Vertical
    flips and aggressive crops that risk excising peripheral lesions are prohibited.
  - Training images use an ultrasound-safe augmentation pipeline:
    1. Horizontal reflection (mirroring left/right transducer orientation),
    2. Mild in-plane rotation within `[-6°, +6°]` (simulating probe tilt),
    3. Mild aspect-preserving central crop (`94%` field of view resized to `224 × 224`),
    4. Mild B-mode acoustic brightness/contrast gain adjustment (`±8%`).
  - To enable deterministic, resumable execution on local CPU without sub-sampling
    the 2,232-image dataset, a deterministic ultrasound-augmented image view (`seed=101`
    combined with row index) is extracted through the frozen `ResNet-18` trunk alongside the
    canonical view (`reports/full_source_baseline_v1/features.npz`), verified by
    input payload SHA-256. During each training epoch, the trainer stochastically
    samples between canonical and ultrasound-augmented views, applies within-image
    view interpolation (`alpha ~ U(0, 0.35)`), and applies mild multiplicative
    acoustic speckle/gain jitter (`N(1.0, 0.03^2)`) on training batches only.
  - Validation folds and held-out sources use strictly the unaugmented canonical view.
  - For GPU environments (e.g., Google Colab / Kaggle T4), the companion script and
    notebook (`notebooks/colab_full_source_finetune.ipynb`) also provide an
    on-the-fly raw-image `layer4 + head` unfreezing mode (`--mode full_layer4_gpu`).

## 4. Fine-Tuned Model Architecture and Imbalance Policy

- **Architecture (`ResNet18-ResidualAdapter`):**
  - Training-only sample-weighted `StandardScaler` fit on canonical training features.
  - Trainable residual bottleneck adaptation block + classification head
    (`265,474` trainable parameters):
    - `LayerNorm(512) -> Linear(512, 256) -> GELU -> Dropout(0.25) -> Linear(256, 512)`
    - Learnable zero-initialized residual gate `g`: `h = z + tanh(g) * adapter(z)`
    - Classification head: `Dropout(0.20) -> Linear(512, 2)`
  - Zero-initialization of `g` ensures epoch 0 starts smoothly from the pretrained
    feature representation before learning nonlinear adaptations.
- **Imbalance weighting policy (`sample_weights`):**
  - Identical to `full_source_baseline_v1`: equal total weight mass to each
    training source (`1 / n_sources`), equal weight mass to each diagnosis class
    (`benign`, `malignant`) within each source (`1 / 2`), and equal weight to each
    global group within `(source, diagnosis)`, divided equally among repeated images
    in that group. Total weight sums to the number of training groups.
- **Optimization budget:**
  - Optimizer: `AdamW(lr=3e-4, weight_decay=1e-2)` with cosine annealing learning
    rate schedule (`eta_min=1e-5`) and gradient norm clipping (`max_norm=2.0`).
  - Batch size: `64`.
  - Maximum epochs per fit: `35` epochs (compared to 2 epochs in the prior pilot).
  - Minimum warmup epochs before early stopping: `8`.
  - Early stopping patience: `10` epochs on the inner validation fold.

## 5. Predeclared Selection Metric, Calibration, and Threshold Policy

- **Inner checkpoint selection metric:** Lowest inner-validation sample-weighted
  binary cross-entropy loss (`val_weighted_bce`), with ties broken by higher
  inner-validation equal-source-weighted group ROC-AUC (`val_group_roc_auc`),
  then higher equal-source-weighted group balanced accuracy (`val_group_ba`).
- **Epoch-level tracking:** Every inner fit (`{source}_inner0..3`) saves its full
  per-epoch history (`epoch`, `train_loss`, `train_roc_auc`, `val_loss`,
  `val_group_roc_auc`, `val_group_ba`, `learning_rate`).
- **Out-of-fold (OOF) calibration and threshold selection:**
  - Out-of-fold validation logits from the 4 selected inner checkpoints are
    collected across all training-side images.
  - Platt scaling (`LogisticRegression(C=1, max_iter=2000, tol=1e-6, random_state=42)`)
    is fit on training-side OOF logits using natural-prevalence equal-source/group
    weights (`balance_diagnosis=False`).
  - Group probabilities are computed as the arithmetic mean of calibrated image
    probabilities within each pure global group.
  - The operating threshold is selected on training-side OOF group probabilities to
    maximize equal-source-weighted group balanced accuracy (ties broken by higher
    weighted specificity, then higher threshold).
- **Final training-side refit:**
  - Refit on all training-side eligible images (`2,070` for Mendeley-held,
    `2,035` for TCIA-held, `359` for BUS-BRA-held) for the mean selected epoch
    count of the 4 inner folds (`round(mean(inner_best_epochs))`), saving the full
    epoch-level training loss/AUC curve and final checkpoint.

## 6. Evaluation, Uncertainty, and Promotion Rules

- **Primary endpoint:** Held-out source group balanced accuracy at the fixed
  training-side threshold, reported separately for Mendeley, TCIA, and BUS-BRA.
- **Secondary endpoints:** Group and image-level ROC-AUC, malignant average
  precision (PR-AUC), sensitivity, specificity, PPV, NPV, confusion matrices,
  binary Brier score, log loss, and 10-bin top-label ECE.
- **Uncertainty:** 1,000 seeded (`seed=42`) percentile group-bootstrap replicates
  resampling whole global groups.
- **No automatic promotion or clinical claim:** Because Mendeley, TCIA, and BUS-BRA
  have all been previously inspected and no eligibility gate dataset is available,
  no outcome from this experiment will enable `BREASTCARE_CLASSIFICATION_ENABLED`
  or be claimed as clinical validation. Poor or negative transfer results must be
  preserved in full.
