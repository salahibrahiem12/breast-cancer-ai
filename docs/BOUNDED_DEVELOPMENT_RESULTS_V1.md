# Bounded model-development experiment v1

Software verification: **72 tests passed, 29 subtests passed**, with one
upstream Starlette/AnyIO deprecation warning (67.13 seconds). These are
software checks, not evidence of clinical reliability. Local log/XML:
`reports/software_regression_bounded_v1.*`.

Changed implementation: `ml/bounded_development.py`, `ml/reserve_busbra.py`,
`tests/test_bounded_development.py`, and its `.gitignore` inclusion. Added this
report and the predefined protocol; updated README, progress, cohort ledger
and dataset register. No serving code, flags, failed checkpoint, or TCIA
predictions changed.

Public classification remains disabled. TCIA is an inspected audit cohort and was not scored by any new candidate. No clinical readiness or eligibility rejection performance is claimed.

## Preservation and predefined budget

The failed checkpoint, TCIA outputs, original/development/external manifests and preparation records were copied with SHA-256 hashes into `reports/bounded_development_v1/preserved/`. All 13 original/copy pairs were verified unchanged after training. The cached pretrained encoder weights were also copied for reproduction.

Protocol: `BOUNDED_DEVELOPMENT_PROTOCOL_V1.md`, recorded before fitting. Three arms, three identical StratifiedGroupKFold folds (seed 42), six total fine-tuning epochs. Primary selection: mean fold balanced accuracy at threshold 0.5. No hyperparameter sweep or repeat search. Local CPU threads 2; elapsed 247.9 seconds. No Colab fallback was necessary.

## Data and leakage audit

Only 162 eligible Mendeley images (71 benign, 91 malignant; 51 duplicate groups) were used. All 88 old-test-linked images remained excluded. Each fold has 108 train and 54 validation images. Existing integrity guards verify all source hashes and dHashes and reject exact/near duplicates crossing partitions at Hamming distance <=5. Patient IDs, site and device mappings are absent: duplicate groups are not independent patients. Repartitioning reused development data does not create independent testing.

The logistic baseline uses 1,024 full-frame grayscale intensities: bilinear stretch to 32x32, divide by 255, StandardScaler fit on fold training only, then balanced L2 logistic regression C=1, lbfgs, max_iter=2000. It receives no filenames, case IDs, metadata, masks, diagnoses or split indicators. Labels supervise fitting only. No code-level feature leakage was identified; unknown patient relationships and source artifacts remain possible. The earlier AUC 0.942 on 41 reused images was exploratory; the new pixel baseline pooled grouped-OOF AUC is 0.8256.

Frozen encoder: cached official ImageNet ResNet-18, RGB/224/ImageNet normalization, 512 pooled features, train-only scaler and the same linear classifier. Fine-tuning: same pretrained start, layer4 and binary head only, two epochs per fold, AdamW lr=1e-4, weight_decay=1e-4, weighted cross-entropy, batch 16; BatchNorm running statistics frozen. No epoch selection, augmentation or temperature fitting.

## Fold-level results at fixed 0.5

| Fold | Arm | Balanced accuracy | Sensitivity | Specificity | ROC-AUC | Binary Brier | ECE |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | pixels | 0.6667 | 0.5000 | 0.8333 | 0.7653 | 0.2430 | 0.2627 |
| 0 | frozen_encoder | 0.7333 | 0.8000 | 0.6667 | 0.8583 | 0.1981 | 0.1963 |
| 0 | fine_tuned | 0.8333 | 0.8333 | 0.8333 | 0.8833 | 0.1511 | 0.1570 |
| 1 | pixels | 0.7938 | 0.9355 | 0.6522 | 0.9691 | 0.1206 | 0.1227 |
| 1 | frozen_encoder | 0.8920 | 0.8710 | 0.9130 | 0.9663 | 0.0693 | 0.0487 |
| 1 | fine_tuned | 0.8429 | 0.9032 | 0.7826 | 0.9397 | 0.1216 | 0.1267 |
| 2 | pixels | 0.6458 | 0.6667 | 0.6250 | 0.7361 | 0.2774 | 0.2910 |
| 2 | frozen_encoder | 0.8750 | 0.8333 | 0.9167 | 0.8889 | 0.1333 | 0.1360 |
| 2 | fine_tuned | 0.7500 | 1.0000 | 0.5000 | 0.9472 | 0.1364 | 0.1354 |

## Aggregate exploratory comparison

| Arm | Mean fold balanced accuracy ± SD | Pooled balanced accuracy (95% group-bootstrap CI) | Sensitivity / specificity | ROC-AUC | AP | Brier / ECE |
|---|---:|---|---|---:|---:|---|
| pixels | 0.7021 ± 0.0801 | 0.7038 (0.5665–0.8383) | 0.7033 / 0.7042 | 0.8256 | 0.8974 | 0.2136 / 0.2169 |
| frozen_encoder | 0.8334 ± 0.0871 | 0.8331 (0.7282–0.9184) | 0.8352 / 0.8310 | 0.8977 | 0.9366 | 0.1336 / 0.1136 |
| fine_tuned | 0.8088 ± 0.0511 | 0.8082 (0.6809–0.9234) | 0.9121 / 0.7042 | 0.9245 | 0.9558 | 0.1364 / 0.1041 |

The frozen encoder ranks first under the predefined primary metric. Fine-tuning has higher sensitivity and AUC but lower specificity at 0.5; it was not substituted based on a different post-hoc metric. This is a development-only choice, with overlapping fold training sets and previously reused data. The three-fold SD is descriptive, not a confidence interval from independent trials. Pooled intervals use 1,000 duplicate-group bootstrap resamples, seed 42; they do not quantify patient-independent generalization. Full metric intervals and log loss are in the machine report.

## Development-only threshold trade-offs

| Arm | Exploratory OOF-selected threshold | Sensitivity | Specificity | Balanced accuracy |
|---|---:|---:|---:|---:|
| pixels | 0.915154 | 0.6703 | 0.9859 | 0.8281 |
| frozen_encoder | 0.726333 | 0.8352 | 0.9014 | 0.8683 |
| fine_tuned | 0.882919 | 0.8352 | 0.9577 | 0.8965 |

Thresholds maximize pooled development OOF balanced accuracy under the recorded rule; their performance is optimistic because the same scores select and assess the threshold. They were not used for primary model ranking, production, or external testing. Calibration estimates are uncalibrated out-of-fold scores; no external calibration was fitted.

## Controlled development-only artifact ablations

Intervene only on held-out fold images for the already fitted pixel baseline, with its original scaler and fixed threshold. Replace the outer 10% band with each image median, or replace the central 80% with that median. No refit.

| Input | Balanced accuracy | ROC-AUC | Mean absolute score change | Decisions changed |
|---|---:|---:|---:|---:|
| Original | 0.7038 | 0.8256 | 0 | 0 |
| border_removed | 0.7038 | 0.8191 | 0.0993 | 26 |
| center_removed | 0.5313 | 0.5123 | 0.3165 | 50 |

Border removal changed individual decisions despite identical aggregate confusion counts. The center-removed result was close to chance. These interventions alter anatomy and image statistics as well as possible artifacts; they neither prove causation nor establish absence of artifacts. No per-device/source causal ablation is possible with missing Mendeley metadata. Encoder/fine-tuning artifact ablations were outside this fixed budget.

## Data acquisition and external reservation

Official BUS-BRA v1: https://zenodo.org/records/8231412 . The saved authoritative API metadata declares CC BY 4.0, and the authors expressly permit research reuse with citation (Gomez-Flores et al., DOI 10.1002/mp.16812). Acquisition is limited to local research reservation; earlier questions about broader release scopes remain unresolved. No external messages were sent.

Archive: 133,917,740 bytes; official MD5 verified, SHA-256 `ba3e6ed19cc37c682d8d39e25435bbf8a555a12cb7e641b5f2117685c95580ff`.

Inspected metadata/decoded-image hashes: 1875 images, 1064 source Case groups, 1268 benign and 607 malignant images, four device strings. Source cases do not cross published 5-fold test assignments and have no conflicting binary labels. No repeated decoded arrays or within-source dHash/pHash pairs within distance 3 were found. Against 506 Mendeley/TCIA images, no byte/pixel matches or dHash/pHash pairs within distance 3 were found. Hash checks do not prove all overlap absent.

BUS-BRA is quarantined as a prediction-free reserved external source. **No model predictions were generated.** Case is a real source field, but equivalence to stable patient/exam identity is not independently documented by the inspected data dictionary. A matching count of 1,064 cases and reported patients is insufficient to infer independence. It is therefore not yet a qualified patient-independent final cohort. Raw images and metadata remain local and Git-ignored. Cite creators, license, DOI and modifications before any permitted derivative publication.

Other feasibility: Mendeley permits the existing limited development use but lacks patient mapping; TCIA is licensed but already inspected; BUS-UCLM access was previously 403; BUSI mirror rights/patient mapping remain unresolved. These were not silently converted into approved fresh test sets.

## Remaining blockers and next evidence-producing step

Resolve the authoritative BUS-BRA Case-to-patient/exam definition and retain a frozen reservation manifest. Lock one final model/refit procedure, aggregation unit, development-derived threshold and reporting protocol before scoring the reserved cohort once. If patient identity cannot be established, report only the weaker source-case/image-level external evaluation or obtain a different patient-mapped cohort. Do not reuse TCIA to choose a winner. No clinical reliability, prospective performance or eligibility-gate rejection performance is established.

## Reproduction and artifacts

```powershell
python -m ml.bounded_development --source data/raw/mendeley_v1/official-v1 --manifest data/prepared/mendeley_dev_v2/split_manifest.csv --output reports/bounded_development_reproduction_v1 --weights reports/bounded_development_v1/preserved/pretrained_resnet18.pth
python -m ml.reserve_busbra
```

Outputs: `reports/bounded_development_v1/comparison.json`, OOF predictions, per-fold linear coefficients/scalers and fine-tuning checkpoints; original artifacts in `preserved/` with `preservation.json`. BUS-BRA inventory/reservation/rights snapshot: `data/research_data/busbra_reserved_v1/`. No model is auto-promoted or eligible for serving.
