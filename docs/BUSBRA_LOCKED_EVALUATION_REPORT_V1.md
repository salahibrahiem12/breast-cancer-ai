# Locked BUS-BRA offline evaluation v1 — 4 November 2026

**The locked model failed to demonstrate useful external discrimination. Public classification remains disabled.** No retry of model selection, threshold tuning, calibration or aggregation was performed on BUS-BRA. This source is now an inspected evaluation cohort, not a fresh future test set.

## Grouping, permissions and pre-inference checks

Official documentation and the accessible original-paper abstract were reviewed; see `BUSBRA_GROUPING_REVIEW_V1.md`. They give overall patient/case counts but do not establish whether the CSV Case field represents a patient, breast, lesion or examination. Full-text access was unavailable through checked routes. We therefore report source-Case results, **not patient-independent results**. Intervals condition on recorded Case grouping; undetected dependence between Cases may make them too narrow. Matching counts or published folds were not treated as proof of patient identity.

The official Zenodo API declares CC BY 4.0, compatible with this local research evaluation and the paper’s research-use description. Cite Gomez-Flores, Gregorio-Calas and Coelho de Albuquerque Pereira; dataset DOI 10.5281/zenodo.8231412 and paper DOI 10.1002/mp.16812, link the license, and identify derivatives. No clinical or public-serving authorization is inferred. Source images remain local and Git-ignored.

The expanded overlap audit scanned 707 raw, 260 processed and 778 legacy BUSI-processed classifier images, plus the TCIA inventory, producing 1,544 unique reference pixel arrays. Legacy/unverified images were used only for overlap inventory. One dHash distance-5 candidate (pHash 24, no exact byte/pixel match) triggered the predefined conservative exclusion of its whole Case: **two benign images, one benign Case**. This is a possible similarity, not confirmed duplication or patient overlap. No within-BUS-BRA cross-Case dHash/pHash candidates at distance <=5 were found. These methods do not exclude every possible derivative or patient overlap.

Included: **1,873 images (1,266 benign, 607 malignant), grouped into 1,063 source Cases (721 benign, 342 malignant)**. All eligible images and Cases were scored. There were no missing/corrupt image, unknown-label, or contradictory-label failures. No Cases were excluded based on model predictions. Multiple images contribute one arithmetic mean score per Case, fixed before inference; source pathology labels agree within each included Case.

## Frozen model and temporal lock

Exactly one final refit used the 162 eligible Mendeley development images, retaining 88 old-test-linked exclusions. Architecture: frozen ImageNet ResNet-18, fc replaced by Identity, 512 pooled features; full-frame RGB, bilinear 224x224 antialiased resize, ToTensor and ImageNet normalization; train-only StandardScaler and balanced L2 logistic regression C=1, lbfgs, max_iter=2000, seed 42. The encoder never trained on BUS-BRA. No augmentation, new hyperparameter search, temperature fit or calibration. Raw logistic probability, fixed threshold **0.5**; no clinical acceptance criterion. Selection followed the previously completed development comparison, not external results.

- Final model frozen (Dubai): 2026-11-04T00:44:42.851712+04:00
- Protocol receipt (UTC): 2026-11-03T20:45:50.200011+00:00
- Single inference started (UTC): 2026-11-03T20:46:14.419056+00:00
- Single inference completed (UTC): 2026-11-03T20:55:02.236579+00:00
- These are locally recorded timestamps and hash receipts, not an independently notarized registration.

| Artifact | SHA-256 |
|---|---|
| weights | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |
| linear | `3fd25a0a2d14a7c4c9d60c0000b7311652cdfde3ca5b283fbdab5614fbb8abfa` |
| development_manifest | `eee71039b79b4ade2cf6ef6e927c93bda37b074cb0b741a0587bb6a46c550ab9` |
| reserved_archive | `ba3e6ed19cc37c682d8d39e25435bbf8a555a12cb7e641b5f2117685c95580ff` |
| eligible_inventory | `c7e4fb66a5d64a8c3ca81e92cd3eaffe301b93b4dca46a02a8912d400436da8d` |
| runner | `abdd9f459c3d690fc40b4a5e19e3a53ff42f9bed5aee2a9aad7f58ac844a37ea` |
| Protocol Markdown | `5966bae2bfff9205a1ba0354e7446c515ceee468c6f738c6130358db31653bdc` |
| Machine lock | `aefc3eb35010af97bd49c890057ad9588b007b5b2ba5ce4710b78934ad5f53f0` |

Dependency versions: torch 2.15.0+cpu, torchvision 0.29.2+cpu, sklearn 1.9.1, numpy 2.5.4, Pillow 13.0.0. Complete hashes and settings are in `reports/busbra_locked_v1/lock.json`; the protocol is `docs/BUSBRA_LOCKED_PROTOCOL_V1.md`.

## Primary source-Case results

| Actual | Predicted benign | Predicted malignant | Total |
|---|---:|---:|---:|
| Benign | 11 | 710 | 721 |
| Malignant | 8 | 334 | 342 |
| Total | 19 | 1044 | 1063 |

One observation per recorded Case. Intervals: 2,000 percentile bootstrap resamples of source Cases, seed 42. These are conditional source-case intervals, **not verified patient-level intervals**. No claim of clinical independence is made.

| Metric | Estimate | Conditional 95% Case-bootstrap interval |
|---|---:|---|
| sensitivity | 0.976608 | 0.959502–0.990992 |
| specificity | 0.015257 | 0.006925–0.024624 |
| precision_ppv | 0.319923 | 0.290381–0.346825 |
| npv | 0.578947 | 0.352941–0.800000 |
| accuracy | 0.324553 | 0.296331–0.350917 |
| balanced_accuracy | 0.495932 | 0.486942–0.504618 |
| roc_auc | 0.470351 | 0.431336–0.508135 |
| pr_auc_average_precision | 0.303286 | 0.269222–0.344138 |
| pr_auc_trapezoid | 0.300866 | 0.266047–0.339819 |
| binary_brier | 0.644006 | 0.617839–0.671887 |
| two_class_brier | 1.288012 | 1.235679–1.343775 |
| log_loss | 5.284571 | 5.017770–5.582618 |
| top_label_ece_10_bins | 0.641600 | 0.614665–0.671601 |

The model called 1,044 of 1,063 Cases malignant. High sensitivity accompanied specificity of only 1.53%; this is not useful discrimination. ROC-AUC was 0.4704 and its interval includes 0.5. Malignant AP was 0.3033 versus cohort malignant prevalence 0.3217. Poor Brier/log-loss/ECE indicate severe miscalibration in this selected cohort. PPV/NPV depend on cohort composition and do not transfer to screening. Binary Brier is half the summed two-class convention. ECE is a descriptive ten-bin top-label estimate; no recalibration was applied.

## Secondary image-level and descriptive subgroup results

Image-level confusion matrix (actual rows / predicted columns; benign, malignant): `[[36, 1230], [25, 582]]`. Image-level balanced accuracy 0.493625, sensitivity 0.958814, specificity 0.028436, ROC-AUC 0.458138, AP 0.296002 and binary Brier 0.648162. The machine report includes source-Case-clustered intervals for these secondary metrics; images were not bootstrapped as unrelated observations.

| Reported device | Source Cases | Benign / malignant | Balanced accuracy | ROC-AUC |
|---|---:|---|---:|---:|
| GE Logiq 5 @10-12MHz | 489 | 349 / 140 | 0.496429 | 0.408903 |
| GE Logiq 7 @10-14MHz | 488 | 327 / 161 | 0.496752 | 0.560640 |
| Toshiba Aplio 300 @12-14 MHz | 72 | 37 / 35 | 0.499228 | 0.375290 |
| U-Systems @10-14MHz | 14 | 8 / 6 | 0.583333 | 0.520833 |

Subgroups are descriptive, not causal or multiplicity-adjusted comparisons. The smallest device group has only 14 Cases. No mixed-device group arose, and no subgroup outcome was used to choose or modify a model.

## Failures, restrictions and next evidence

No execution failure was recorded. The scientific result is poor. Do not try a different model, threshold, calibration or aggregation on BUS-BRA and describe that as untouched testing. BUS-BRA and TCIA are both now inspected audit/evaluation sources. Further development needs different training/validation data with verified patient/exam mappings and another fresh external cohort. Authoritative Case semantics remain unresolved.

This was an **offline classifier evaluation** on lesion-containing source images. It does not evaluate eligibility-gate rejection, healthy/normal screening, the upload-to-result system, prospective care or clinical readiness. The absent gate did not prevent the research evaluation. Neither good nor bad classifier results can enable public classification automatically.

## Reproduction, verification and deliverables

Original commands, executed once in order:

```powershell
python -m ml.busbra_locked prepare
python -m ml.busbra_locked evaluate
```

Preparation refuses an existing locked output directory. Evaluation refuses any second invocation after its start marker, even if a run fails. Do not remove those guards to search for a better result. For a separate technical reproduction, use an isolated copy of the repository/data in which reports/busbra_locked_v1 does not yet exist; identical library versions and source artifacts are required. Timestamps/artifact serialization metadata may differ on refitting.

Reproduce metrics and Case intervals from the saved predictions without new image inference:

```powershell
python -m ml.verify_busbra_lock --intervals
```

Local artifacts under `reports/busbra_locked_v1/`: final_linear.npz, lock.json, protocol_receipt.json, eligible_inventory.csv, precheck/exclusion records, image_predictions.csv, case_predictions.csv, evaluation.json, start/completion markers and verification.json. Predictions contain source pseudonymous grouping keys for local reproduction; no patient identifiers or raw images are printed in this public-facing report. Encoder weights remain at `reports/bounded_development_v1/preserved/pretrained_resnet18.pth`, frozen by the hash above.

Software regression: **74 passed, 29 subtests passed**, one upstream Starlette/AnyIO warning, 117.66 seconds. Two new guard tests passed separately. These establish software behavior, not clinical reliability. Logs/XML: `reports/software_regression_locked_v1.*`.
