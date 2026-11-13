# Full-eligible neural-head adaptation on frozen features (`full_source_finetune_v1`): measured results

Completed 4 November 2026 under the predeclared protocol in
[`FULL_SOURCE_FINETUNE_PROTOCOL_V1.md`](FULL_SOURCE_FINETUNE_PROTOCOL_V1.md)
and compared directly against the full-eligible frozen linear baseline
([`FULL_SOURCE_BASELINE_RESULTS_V1.md`](FULL_SOURCE_BASELINE_RESULTS_V1.md)).
Public classification remains disabled (`BREASTCARE_CLASSIFICATION_ENABLED=false`).
All three breast-ultrasound cohorts (Mendeley, TCIA BrEaST, BUS-BRA) have already
been inspected; these are exploratory cross-source development results, not an
untouched clinical validation.

## 1. Exact Scope: Neural-Head Training on Cached Frozen Features vs. Encoder Fine-Tuning

To be completely transparent about what is trained in `full_source_finetune_v1` and how verification distinguishes parameter flags from historical training updates:
1. **Four distinct concepts verified in `ml/verify_full_source_finetune.py`:**
   - **(a) `requires_grad` flags on standalone module objects (and resolution of `task-329`):**
     `net()` in `ml/cross_source.py` instantiates `resnet18(weights=None)`, loads `pretrained_resnet18.pth`
     (SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`), replaces `fc` with
     `Identity()`, and calls `.eval()`. In PyTorch, `.eval()` sets `module.training = False` (disabling dropout
     and freezing BatchNorm running statistics) but leaves `param.requires_grad == True` on the standalone
     `net()` object (`11,176,512` parameters). In `task-329`, an initial assertion in `ml/verify_full_source_finetune.py`
     (`assert sum(p.numel() for p in backbone.parameters() if p.requires_grad) == 0`) failed because it checked
     the default `requires_grad` flag on a newly constructed `net()` object rather than checking whether `net()`
     participated in optimization. That failure is preserved in `reports/full_source_finetune_v1/lock_audit_trail.json`
     and `reports/full_source_finetune_v1/verification.json`.
   - **(b) Optimizer parameter-group membership:**
     During `fit_adapter()` (`ml/full_source_finetune.py` line 394), `net()` is never instantiated;
     `torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)` receives only the `264,963` parameters
     of `ResidualAdapterHead` (`0` `ResNet-18` parameters).
   - **(c) Autograd gradient computation:**
     During `aug_features()`, `net()` executes inside `with torch.inference_mode():` and writes detached
     `float32` numpy arrays (`aug_features.npz`), so no autograd graph or backward pass ever touches `ResNet-18`.
     Within `ResidualAdapterHead`, because `gate` is initialized to `0.0` (`tanh(0) = 0`), a synthetic 2-step
     architectural check shows that `norm`, `fc1`, and `fc2` receive `0.0` gradient on step 0 while `gate` and
     `classifier` receive non-zero gradients, and all 9 parameter tensors receive non-zero gradients on step ≥ 1
     once `gate != 0`. **Note:** A freshly constructed diagnostic probe only illustrates this mathematical
     property of the architecture; it does **not** prove historical training behavior.
   - **(d) Actual historical parameter changes in the 15 saved checkpoints on disk:**
     Because `fit_adapter()` seeds PyTorch (`torch.manual_seed(seed)`, where `seed = 42 + source_idx * 10 + fold`
     for inner folds `0..3` and `42 + source_idx * 10 + 9` for `busbra_final`) immediately before instantiating
     `ResidualAdapterHead()`, `ml/verify_full_source_finetune.py` reconstructs the exact initial parameter tensors
     `theta_init` for all 15 fits and compares them directly against the saved historical checkpoints
     `reports/full_source_finetune_v1/models/*.npz`. Every one of the 9 parameter tensors (`gate`, `norm.weight`,
     `norm.bias`, `fc1.weight`, `fc1.bias`, `fc2.weight`, `fc2.bias`, `classifier.weight`, `classifier.bias`)
     has verified non-zero L2 change `||theta_saved - theta_init||_2 > 0` across all 15 saved checkpoints.
2. **Trainable module breakdown (`264,963` parameters in `ResidualAdapterHead`):**

| Trainable Module | Shape / Definition | Parameters in `AdamW` | Architectural Step-0 Grad (`gate=0`) | Historical Saved Checkpoints `||theta_saved - theta_init||_2 > 0` |
|---|---|---:|---|---|
| `norm` | `LayerNorm(512)` (`weight` + `bias`) | `1,024` | `0.0` (gated by `tanh(0)=0`) | Verified (`>0` in all 15 saved checkpoints on disk) |
| `fc1` | `Linear(512 -> 256)` (`weight` + `bias`) | `131,328` | `0.0` (gated by `tanh(0)=0`) | Verified (`>0` in all 15 saved checkpoints on disk) |
| `fc2` | `Linear(256 -> 512)` (`weight` + `bias`) | `131,584` | `0.0` (gated by `tanh(0)=0`) | Verified (`>0` in all 15 saved checkpoints on disk) |
| `gate` | Scalar `nn.Parameter(zeros(1))` (`h = z + tanh(gate) * delta`) | `1` | Non-zero (`sech^2(0) = 1.0`) | Verified (`|gate| > 0` in all 15 saved checkpoints on disk) |
| `classifier` | `Linear(512 -> 2)` (`weight` + `bias`) | `1,026` | Non-zero on step 0 | Verified (`>0` in all 15 saved checkpoints on disk) |
| **Total (`ResidualAdapterHead`)** | **Nonlinear residual bottleneck head** | **`264,963`** | **Architectural property** | **All 15 historical checkpoints verified** |

- **Separately versioned GPU encoder fine-tuning (`gpu_encoder_finetune_v2` / `v3`):**
  Because local hardware is CPU-only (`torch.cuda.is_available() == False`), end-to-end convolutional
  encoder fine-tuning has **not** run locally and is packaged as a separately versioned GPU experiment
  ([`GPU_ENCODER_FINETUNE_PROTOCOL_V3.md`](GPU_ENCODER_FINETUNE_PROTOCOL_V3.md), `ml/gpu_encoder_finetune_v2.py`,
  `notebooks/colab_full_source_finetune.ipynb`) without relabeling `full_source_finetune_v1`.

## 2. Protocol Lock Audit Trail and Augmentation / Fold Isolation

1. **Preserved protocol locks (`lock_initial_20261104T090149Z.json` and `lock.json`):**
   - Initial lock (`task-204`, `2026-11-04T09:01:49.061990+00:00`, SHA-256 `2e92dcb8a96d9c1b51f5088cca7a31dc2e861742d1cc00814230276ff58fb60f`)
     recorded `Path(__file__)` as an absolute path key (`D:/breast_cancer/ml/full_source_finetune.py`).
   - Augmented feature extraction (`task-212`) completed across all 2,232 eligible images in
     `1.2s` before any model fitting began.
   - `task-278` aborted immediately inside `check_lock()` before loading features or creating
     `models/`; **zero model fits or outcomes existed before the lock refresh**.
   - Active lock (`task-292`, `2026-11-04T09:15:33.338715+00:00`, SHA-256 `f8656aa5c96ce9c801d77ee11720cb08e24cbbf8a0f1a2e16548f20abcb0800d`)
     normalized the script key to relative `ml/full_source_finetune.py` and added a guard
     preventing relocking once `models/` exists. Every dataset manifest hash, protocol hash,
     pretrained weight hash, canonical feature hash, and hyperparameter is **100% identical**
     between both preserved lock files (`reports/full_source_finetune_v1/lock_audit_trail.json`).
2. **Augmentation and partition isolation:**
   - Every original image (`x_can[i]`) and its deterministic ultrasound-augmented view (`x_aug[i]`)
     are verified by input payload `sha256` at identical row index `i`, keeping them in the
     exact same `global_group` and training partition (`tr`).
   - Inner validation (`va`) and held-out source evaluation (`held`) use strictly the
     unaugmented canonical features (`x_can`) and locked group-mean probability aggregation.
   - Weighted `StandardScaler`, early-stopping checkpoint selection (`selected_epoch`),
     OOF Platt calibration (`LogisticRegression(C=1)`), and operating threshold selection
     (`choose_threshold`) are fit strictly on training-side data (`s != held`).

## 3. Epoch-Level Training, Early Stopping, and Historical Parameter Deltas Across All 15 Fits

| Held source | Inner selected epochs (folds 0–3) | Inner epochs ran (folds 0–3) | Final refit epochs | Final `tanh(gate)` | Final `||delta_fc1||_2` / `||delta_fc2||_2` / `||delta_cls||_2` from init | Training OOF group AUC |
|---|---|---|---:|---:|---|---:|
| mendeley | `[4, 5, 4, 4]` | `[14, 15, 14, 14]` | 8 | `-0.0491` | `3.374` / `6.982` / `0.214` | 0.720 |
| tcia | `[5, 4, 8, 7]` | `[15, 14, 18, 17]` | 8 | `-0.0500` | `4.066` / `8.241` / `0.242` | 0.725 |
| busbra | `[35, 35, 19, 20]` | `[35, 35, 29, 30]` | 27 | `+0.0322` | `4.139` / `7.677` / `0.238` | 0.701 |

## 4. Primary Group-Level Results and Baseline Comparison

Group balanced accuracy (`BA`) at the fixed training-side threshold is the
predeclared primary endpoint. Confidence intervals are 95% percentile group-bootstrap
intervals from 1,000 seeded draws (`seed=42`) resampling whole global groups.

| Held source | Model | Group BA [95% CI] | Group ROC-AUC [95% CI] | Sensitivity | Specificity | Malignant AP | Binary Brier | Top-label ECE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| mendeley | Frozen linear baseline (`v1`) | 0.573 [0.421, 0.711] | 0.706 [0.519, 0.875] | 0.381 | 0.765 | 0.753 | 0.252 | 0.150 |
| mendeley | Neural adapter head (`v1`) | 0.465 [0.367, 0.545] | 0.448 [0.277, 0.651] | 0.048 | 0.882 | 0.538 | 0.355 | 0.319 |
| tcia | Frozen linear baseline (`v1`) | 0.644 [0.578, 0.712] | 0.688 [0.615, 0.761] | 0.520 | 0.768 | 0.657 | 0.258 | 0.161 |
| tcia | Neural adapter head (`v1`) | 0.562 [0.504, 0.620] | 0.671 [0.593, 0.745] | 0.276 | 0.848 | 0.633 | 0.259 | 0.146 |
| busbra | Frozen linear baseline (`v1`) | 0.570 [0.538, 0.600] | 0.594 [0.554, 0.631] | 0.576 | 0.564 | 0.408 | 0.305 | 0.196 |
| busbra | Neural adapter head (`v1`) | 0.518 [0.497, 0.541] | 0.573 [0.536, 0.611] | 0.857 | 0.179 | 0.419 | 0.341 | 0.292 |

### Confusion Matrices and Calibration Transfer

| Held source | Group confusion `[[TN, FP], [FN, TP]]` | Threshold | Calibration coef / intercept | PPV | NPV | Held majority accuracy | Constant-prevalence Brier |
|---|---|---:|---|---:|---:|---:|---:|
| mendeley | `[[15, 2], [20, 1]]` | 0.434231 | 0.6470 / -0.3524 | 0.333 | 0.429 | 0.553 | 0.289 |
| tcia | `[[84, 15], [71, 27]]` | 0.517346 | 0.7036 / -0.5638 | 0.643 | 0.542 | 0.503 | 0.278 |
| busbra | `[[129, 592], [49, 293]]` | 0.519190 | 0.8057 / -0.1857 | 0.331 | 0.725 | 0.678 | 0.252 |

## 5. Image-Level Secondary Endpoints

| Held source | Image BA [95% CI] | Image ROC-AUC [95% CI] | Sensitivity | Specificity | Malignant AP | Binary Brier | Top-label ECE |
|---|---|---|---:|---:|---:|---:|---:|
| mendeley | 0.459 [0.361, 0.528] | 0.427 [0.264, 0.608] | 0.044 | 0.873 | 0.530 | 0.364 | 0.330 |
| tcia | 0.562 [0.504, 0.620] | 0.671 [0.593, 0.745] | 0.276 | 0.848 | 0.633 | 0.259 | 0.146 |
| busbra | 0.516 [0.497, 0.537] | 0.559 [0.525, 0.595] | 0.842 | 0.190 | 0.394 | 0.351 | 0.308 |

## 6. Scientific Interpretation: Measured External Degradation vs. Unproved Hypotheses

1. **Measured external outcome (the neural adapter performed worse externally than the linear baseline):**
   Across the inner validation folds (`StratifiedGroupKFold(4)` within the two
   training sources), `ResidualAdapterHead` achieved `0.69–0.89` validation group
   ROC-AUC and drove training weighted BCE loss down to `0.26–0.55`. However, when
   evaluated on the unseen held-out third source, `ResidualAdapterHead` performed
   worse externally than the simpler L2-regularized linear logistic regression baseline
   (`full_source_baseline_v1`) across all three held-out sources:
   - Mendeley: group ROC-AUC `0.448` vs `0.706`; group BA `0.465` vs `0.573`.
   - TCIA: group ROC-AUC `0.671` vs `0.688`; group BA `0.562` vs `0.644`.
   - BUS-BRA: group ROC-AUC `0.573` vs `0.594`; group BA `0.518` vs `0.570`.
2. **Hypothesis vs. established explanation:**
   Although [`EXPERIMENT_DESIGN_AUDIT_V1.md`](EXPERIMENT_DESIGN_AUDIT_V1.md) showed that
   acquisition source (`mendeley`, `tcia`, `busbra`) is `99.1%` predictable from
   frozen `ResNet-18` features even within a single diagnosis class, **source-specific
   overfitting in frozen feature space is an unproved hypothesis, not an established
   causal explanation** for why `ResidualAdapterHead` performed worse externally. Other
   plausible contributors include optimization dynamics of the zero-initialized residual
   gate (`tanh(gate)`), feature-space view mixing / speckle perturbation hyperparameters,
   inner-fold early-stopping variance, and fixed-feature representation limits.
3. **Why public classification remains disabled (`BREASTCARE_CLASSIFICATION_ENABLED=false`):**
   Neither the frozen linear baseline nor the nonlinear neural-head adaptation
   meets a defensible standard for clinical or public diagnostic use; all three
   cohorts are inspected development sources, GPU encoder fine-tuning has not yet
   been executed on CUDA hardware, and input-eligibility / OOD gating remains an
   unresolved workstream (the failed leave-one-source-out Mahalanobis threshold
   diagnostic in `reports/eligibility_ood_candidate_audit_v1.json` only shows that a
   simple 95% in-distribution Mahalanobis cutoff fails under cross-source shift, not
   that every OOD method fails).

## 7. Reproducibility and Verification Commands

```powershell
.\.venv\Scripts\python.exe -m ml.verify_full_source_finetune --intervals
.\.venv\Scripts\python.exe -m ml.report_full_source_finetune
```
