# BreastCare AI progress

Last checked: 2026-11-04 (Asia/Dubai).

## Latest: containerized deployment configuration, safest free hosting analysis, and verified local operation (4 November 2026)

- **Production-Ready Docker and Compose Configuration:**
  - Prepared root .dockerignore and web/.dockerignore to cleanly exclude .venv, 
ode_modules, data/, 
eports/, and credentials.
  - Authored Dockerfile.api based on Python 3.11-slim with lightweight CPU-only PyTorch (--index-url https://download.pytorch.org/whl/cpu), non-root execution (reastcare:1000), port 8000 exposure, and automated /health container healthcheck.
  - Authored web/Dockerfile using multi-stage Node 20-alpine build for Next.js production bundle, non-root execution (
extjs:1001), port 3000 exposure, and automated root healthcheck.
  - Authored root docker-compose.yml declaring pi (8000) and web (3000) with explicit healthcheck dependency (condition: service_healthy) and fail-closed security (BREASTCARE_CLASSIFICATION_ENABLED=false).
  - Authored .env.example documenting all configuration keys and clinical safety bounds.
- **Safest Free Hosting Assessment ([DEPLOYMENT.md](DEPLOYMENT.md)):**
  - Evaluated on-premise/local self-hosting vs. Hugging Face Spaces (free CPU tier) vs. Render/Railway free containers.
  - Identified **Option A (Self-Hosted Docker Compose / Local Workstation)** as the **safest available hosting option**: zero data exfiltration, total network isolation, and guaranteed prevention of unvalidated medical diagnostic classifier exposure.
  - Confirmed that public cloud deployment of unvalidated clinical classifiers is strictly prohibited; any research showcase (e.g., Hugging Face Spaces Docker) must maintain BREASTCARE_CLASSIFICATION_ENABLED=false hard-locked.
- **Local Application & Test Suite Verification:**
  - Full Python test suite verified: **88 passed, 1 warning, 29 subtests passed in 133s** (python -m pytest tests -q).
  - Web UI test & production build verified: Next.js production build (
ode node_modules/next/dist/bin/next start) running on http://127.0.0.1:3000 (	ask-840).
  - Backend API verified: FastAPI server running on http://127.0.0.1:8000 (	ask-808), returning HTTP 200 on /health and /model-info.
  - Machine environment audited: Windows host has no Docker daemon installed; native local runner .\start_breastcare.ps1 -Mode check|api|web|both serves as the verified local execution path while Docker files provide container portability.

## Grouping integrity audit (`v2` manifest), transparent verification resolution, and resumable GPU encoder package (`gpu_encoder_finetune_v3`) (4 November 2026)

- **Global duplicate-group integrity audit (`ml/audit_grouping_integrity.py`, [`GROUPING_INTEGRITY_AUDIT_V1.md`](GROUPING_INTEGRITY_AUDIT_V1.md), `reports/grouping_audit_v1/`):**
  - Explained why 162 Mendeley images form 38 `v1` global groups: all 250 acquired
    Mendeley files (`us1.bmp`..`us250.bmp`) are organized as 50 consecutive five-image
    shifted-crop sequences (`us{5k+1}..us{5k+5}`, `k = 0..49`, with `us116..120` and
    `us226..230` derived from the same `153×131` scan = 49 unique base lesion scans).
    Visual inspection of rendered contact sheets (`reports/grouping_audit_v1/contact_sheets/*.png`)
    confirmed that `pHash <= 5` reunited 11 shifted-crop sequences that had been split
    at `dHash <= 5` (`51` units -> `38` groups) without chaining unrelated lesions.
  - Uncovered two Mendeley defects in `unified_dataset_manifest_v1.csv`: (1) `17` images
    across `8` partial groups (`global:2, 3, 5, 72, 85, 103, 130, 131`) were shifted-crop
    siblings of historical `mendeley_v1` test frames whose border shift exceeded `dHash = 5`,
    and (2) `2` fully eligible five-image sequences (`Block #4: us21..25` and `Block #38: us191..195`)
    were split across two `v1` groups (`global:10/13` and `global:120/123`).
  - Preserved `unified_dataset_manifest_v1.csv` and created versioned
    `data/prepared/unified_dataset_manifest_v2.csv` (`reports/grouping_audit_v1/eligible_manifest_v2.json`:
    **2,215 strictly clean eligible images across 1,288 duplicate/case groups**: `145` Mendeley
    in `28` complete base-scan groups, `197` TCIA in `197` patient IDs, `1,873` BUS-BRA in
    `1,063` source cases), explicitly documenting affected comparisons.
- **Full-eligible frozen linear baseline (`reports/full_source_baseline_v1`, [`FULL_SOURCE_BASELINE_RESULTS_V1.md`](FULL_SOURCE_BASELINE_RESULTS_V1.md)):**
  Verified all 15 L-BFGS-B linear fits (`max_absolute_gradient <= 1e-5`), fold isolation,
  saved-model replay (`max_error = 2.22e-16`), 1,000-resample group bootstrap 95% CIs,
  and all 51 preserved prior locked artifacts (`reports/design_audit_v1/prior_artifact_hashes.json`).
  Held-out group ROC-AUC / BA: Mendeley `0.706` / `0.573`, TCIA `0.688` / `0.644`,
  BUS-BRA `0.594` / `0.570`.
- **Full-eligible neural-head adaptation on frozen features (`reports/full_source_finetune_v1`, [`FULL_SOURCE_FINETUNE_RESULTS_V1.md`](FULL_SOURCE_FINETUNE_RESULTS_V1.md)):**
  - Trained `ResidualAdapterHead` (`264,963` parameters) on cached detached `ResNet-18`
    canonical and ultrasound-augmented features across all 2,232 `v1` eligible images.
  - Held-out group ROC-AUC / BA: Mendeley `0.448` / `0.465`, TCIA `0.671` / `0.562`,
    BUS-BRA `0.573` / `0.518`. Stated clearly that **source-specific overfitting in
    frozen feature space is an unproved hypothesis, not an established causal explanation**
    for why the neural adapter performed worse externally than the linear baseline.
  - Transparently resolved the `task-329` `backbone_trainable_params` assertion failure
    in `ml/verify_full_source_finetune.py` and `reports/full_source_finetune_v1/lock_audit_trail.json`,
    explicitly distinguishing (1) standalone `net()` `requires_grad` flags (`11,176,512`),
    (2) `AdamW` optimizer membership (`0` backbone params, `264,963` head params),
    (3) autograd gradient computation, and (4) direct historical parameter-delta verification
    (`||theta_saved - theta_init||_2 > 0`) across all 15 saved checkpoints on disk against
    their exact seeded initializations (`torch.manual_seed(seed)`).
- **Separately versioned GPU encoder fine-tuning package (`docs/GPU_ENCODER_FINETUNE_PROTOCOL_V3.md`, `ml/gpu_encoder_finetune_v2.py`, `notebooks/colab_full_source_finetune.ipynb`):**
  - Inspected local CUDA resources (`torch.cuda.is_available() == False` on `PyTorch 2.15.0+cpu`):
    **GPU encoder fine-tuning has NOT run locally.**
  - Preserved `reports/gpu_encoder_finetune_v2/lock.json` (`SUPERSEDED_BY_V3.json`) and
    locked `reports/gpu_encoder_finetune_v3/lock.json` bound to `unified_dataset_manifest_v2.csv`
    (`2,215` images, `1,288` groups).
  - Verified `ResNet18Layer4FineTuner` (`8,388,608` `layer4` conv params + `1,026` `fc` params =
    `8,389,634` trainable parameters) with **all 20 `BatchNorm2d` modules frozen**
    (`training=False`, `requires_grad=False`, unchanged `running_mean`/`running_var` even
    after `model.train()` and `optimizer.step()`), real `layer4` convolutional gradient
    updates on raw images, and resumable per-epoch checkpoints (`.resume.pth`).
- **Exploratory OOD eligibility diagnostic (`reports/eligibility_ood_candidate_audit_v1.json`) & 6-method OOD benchmark on `v2` (`reports/eligibility_ood_benchmark_v2.json`, [`ELIGIBILITY_OOD_BENCHMARK_V2.md`](ELIGIBILITY_OOD_BENCHMARK_V2.md)):**
  - Demonstrated that a simple 95% in-distribution Mahalanobis cutoff on two breast-ultrasound
    sources accepts only `0.6%–20.3%` (`9.4%` mean) of valid held-out third-source breast-ultrasound
    scans despite `0.954` mean OOD ranking AUROC, because inter-source shift pushes unseen breast-ultrasound
    sources outside the training ellipsoid.
  - Benchmarked 6 distinct OOD / eligibility methods on the `2,215`-image `v2` manifest against
    case-disjoint `fetal_head_zenodo` (`n=64`), `ChestMNIST` (`n=64`), and corrupted frames (`n=48`),
    showing that 12-D image-domain ultrasound physics descriptors under min-source p95 calibration
    (`78.6%` held-out BUS acceptance, `0.852` mean OOD AUROC) and cross-source calibrated hybrid
    probes (`0.941` mean OOD AUROC, `0.0%` pooled FAR, `26.6%–33.8%` held-out BUS acceptance)
    substantially outperform raw pooled Mahalanobis.
  - Eligibility remains an unresolved workstream until licensed `natural_image`, `object`,
    `screenshot`, and real clinical `low_quality` cohorts are acquired, and public classification
    remains disabled (`BREASTCARE_CLASSIFICATION_ENABLED=false`).
- **Grouping-validated frozen linear baseline (`full_source_baseline_v2`, [`FULL_SOURCE_BASELINE_RESULTS_V2.md`](FULL_SOURCE_BASELINE_RESULTS_V2.md), `reports/full_source_baseline_v2/`):**
  - Executed the exact 15-fit L-BFGS-B linear baseline on `eligible_manifest_v2.json` (`2,215` images,
    `1,288` groups: `145` Mendeley in `28` complete base-scan groups, `197` TCIA in `197` patient IDs,
    `1,873` BUS-BRA in `1,063` source cases). All 15 fits converged (`max_absolute_gradient <= 1e-5`)
    with `0.0` saved-model replay error. Held-out group ROC-AUC / BA: Mendeley `0.672` / `0.546`,
    TCIA `0.682` / `0.624`, BUS-BRA `0.590` / `0.567`.
- **Full-stack verification & live local deployment:**
  - `88 passed, 29 subtests passed` in `pytest tests -q`; `npm test`, `npm run typecheck`, `npm run lint`,
    and `npm run build` all pass in `web/`; live local FastAPI (`http://127.0.0.1:8000`) and Next.js
    (`http://127.0.0.1:3000`) servers verified via `start_breastcare.ps1`.

## Earlier: completed cross-source audit and bounded pilot experiments

- Preserved and hash-verified 23 prior locked artifacts. TCIA and BUS-BRA are
  explicitly relabeled inspected development sources under the user's instruction.
- Audited pretrained loading, evaluation mode, RGB/normalization/feature layer,
  feature variance, preprocessing consistency and saved development replay.
  No collapse or missing learned weights found; zero replay decision changes.
- Completed all six predeclared source-held-out fits on 220 images/96 groups,
  with training-side calibration/threshold selection and grouped intervals.
  Generalization remains inconsistent. No model is promoted or deployed.
- Measured source-encoding association and controlled border/center ablations;
  neither establishes a causal explanation. New external access remains blocked.
- Results, limits and exact reproduction are in
  [CROSS_SOURCE_RESULTS_V1.md](CROSS_SOURCE_RESULTS_V1.md). 77 regression tests
  and 29 subtests passed. No UI changes or classification enabling in this work.

## Implemented in the existing repository

- Next.js research website and FastAPI upload-validation API.
- Reproducible preparation, classifier training/evaluation, eligibility-gate
  training/evaluation, and duplicate-audit tools.
- Server-controlled classification with reviewed, checkpoint-hash-bound evidence
  requirements. No configuration flags were enabled during this work.

## Completed in this continuation

- Fixed an upload race: selecting another image aborts the old request and
  invalidates its response. A delayed response cannot replace the current
  image's result or clear the current request's loading indicator.
- Added a 60-second analysis timeout, a retry message, duplicate-submit
  protection, and cancellation on component unmount.
- Added a 10-second model-status timeout and cleanup.
- Added three component-handler regression tests for image switching, timeout
  recovery, and duplicate submission. Run from `web` with `npm test`.
- Rechecked the official BUS-UCLM v3 page on 2026-11-03:
  <https://data.mendeley.com/datasets/7fvgj4jsp7/3>. It still describes 683 images
  from 38 patients and CC BY 4.0, but the retrieved Files section exposes no
  file entries. This is a page recheck only, not proof that every authorized
  download route is inaccessible. No data were acquired or repartitioned.

## Verification

- Frontend handler regression tests: **3 passed**. These use mocked requests
  and a hook harness; they are not browser or medical-performance tests.
- API regression: **18 passed, 8 subtests passed**, with one upstream
  Starlette/AnyIO deprecation warning. Machine-readable local report:
  `reports/software_api_2026-11-03.xml`.
- Production build: **passed** (Next.js 15.5.28, all nine static pages generated).
- Type checking during the production build: **passed**.
- ESLint for the modified component and new regression test: **passed**.
- Full Python regression: **55 passed, 16 subtests passed** in 131.28 seconds,
  with one upstream Starlette/AnyIO deprecation warning. Local evidence:
  `reports/software_regression_2026-11-03.log` and
  `reports/software_regression_2026-11-03.xml`.

## Eligibility evaluation integrity follow-up

- New eligibility checkpoints and training reports record `manifest_sha256`.
- Evaluation checks that hash before validating the manifest or scoring data,
  and records it in the evaluation report. An unchanged pathname alone is no
  longer sufficient. Older unbound gate checkpoints are rejected.
- Training/evaluation reject manifest changes during the run before writing
  a checkpoint/evaluation report, respectively.
- Full regression: **58 passed, 22 subtests passed** in 75.99 seconds,
  including the additional mid-evaluation mutation check. One upstream
  Starlette/AnyIO deprecation warning remains. Local evidence:
  `reports/software_regression_manifest_binding_2026-11-03.log` and its XML report.
- Tests use tiny generated fixtures and mocked scoring; they establish software
  integrity behavior, not measured medical performance. No dataset was acquired,
  no research model was trained, and serving flags were not enabled.

## Trusted-source acquisition and external evaluation

- User authorized finding a trusted dataset. Acquired official TCIA
  BREAST-LESIONS-USG v1 images and clinical workbook under the collection's
  CC BY 4.0 terms. Original downloads remain unchanged and ignored by Git.
- Verified archive CRC/path safety, 256 primary scans, 266 masks, metadata
  pairing, unique cases, and byte/pixel/perceptual overlap against all 250
  official Mendeley source images. No matches were found with these methods.
- Recorded the protocol before predictions; prepared 197 biopsy-only cases
  (99 benign, 98 malignant) as external test-only, with no training or tuning.
- Fixed-model evaluation: accuracy **0.4822**, macro-F1 **0.3705**, ROC-AUC
  **0.4630**, sensitivity **0.9082**, specificity **0.0606**. Confusion matrix
  `[[6, 93], [9, 89]]`. These results do not support enabling classification.
- Added reproducible inventory/preparation tools and regression tests; updated
  the website's evidence page, API limitations, model card and dataset register.
- Full software regression: **63 passed, 26 subtests passed**, one upstream
  deprecation warning, in 62.34 seconds. Evidence:
  `reports/software_regression_tcia_2026-11-03.log` and XML report.
- Updated website production build, lint and type checking: **passed**.
- Live API HTTP checks: **passed** for health, model status, CORS, valid
  generated PNG, and invalid-image rejection. Both upload responses had null
  classification/probabilities; inference was disabled. Local evidence:
  `reports/live_smoke_tcia_2026-11-03.json`.
- Live production-site HTTP checks: **six routes passed**, including the
  updated external results on `/model-evidence`. These are HTTP/content checks;
  no new browser visual inspection is claimed. Temporary verification servers
  were stopped after checking.
- See `TCIA_ACQUISITION_AND_EVALUATION_2026-11-03.md` for file hashes, attribution,
  intervals, protocol, commands and limitations. This cohort is no longer fresh
  for evaluating future model changes. No clinical or cross-source patient
  identity verification is claimed.

## Pipeline audit and separate development follow-up

- Audited the exact fixed checkpoint, class mapping, workbook join, image
  integrity, full-frame RGB/224 preprocessing, recorded temperature and fixed
  0.5 threshold. Replay agreed within 4.993e-9; no mapping reversal or mask
  ingestion error was found.
- The 197 observations are unique case images and source-reported patient
  groups, not independent lesion samples or verified examination records.
- Added sensitivity, specificity, PPV, NPV, balanced accuracy, malignant AP
  and trapezoidal PR-AUC, ROC-AUC, Brier/log-loss/ECE, with 2,000-resample
  patient-group intervals. Complete values and confusion matrix are in
  `TCIA_PIPELINE_AUDIT_2026-11-03.md`.
- Reviewed all 93 FP and nine FN in local neutral-labelled privacy crops;
  quantified resolution, aspect ratio, intensity, borders and color. These
  are descriptive artifact associations, not clinical or causal adjudication.
- Ran the predeclared separate Mendeley train/validation comparison. Baseline
  balanced accuracy 0.7222; aspect/contrast candidate 0.5954 on 41 validation
  images in 12 duplicate groups. Candidate was worse and was not adopted.
  All 250 source hashes/dHashes passed; 88 old-test-linked images were excluded
  from feature fitting. Patient grouping remains unavailable for this source.
- Saved replayable numerical classifier/scaler parameters. Neither development
  candidate was evaluated on TCIA. No serving checkpoint or flag changed.
- Corrected an exact-0.5 binary tie inconsistency in evaluator predictions and
  calibration; TCIA has no ties, so its results are unchanged.
- Full regression: **70 passed, 29 subtests passed**, one upstream deprecation
  warning in 67.79 seconds. No website changes required a frontend rebuild.
- Final targeted checks after calibration tie handling: **8 passed, 3 subtests
  passed**. Serving flags were explicitly rechecked and remained false.
- `COHORT_USAGE_LEDGER.md` marks TCIA inspected and records that **no fresh
  independent final cohort is available or reserved**. Gate performance remains
  separate and unmeasured. Software checks are not clinical reliability evidence.

## Bounded model-development experiment v1

- Preserved and SHA-256 verified 13 failed-experiment artifacts/manifests;
  copied cached official pretrained weights for reproduction.
- Recorded a fixed three-arm, three-fold budget before fitting: pixel logistic
  baseline, frozen pretrained ResNet-18 plus linear classifier, and two-epoch
  layer4/head fine-tuning. Used the same 162 eligible Mendeley images and
  duplicate groups; all 88 old-test-linked images remained excluded.
- Completed locally in 247.9 seconds. Mean fold balanced accuracy (SD):
  pixels 0.7021 (0.0801); frozen encoder 0.8334 (0.0871); fine-tuned 0.8088
  (0.0511). Frozen encoder ranks first under the predefined metric. No model
  was promoted or evaluated on TCIA.
- Reported every fold, OOF group-bootstrap intervals, calibration, development-
  only threshold trade-offs and controlled pixel-baseline border/center
  ablations. Results are exploratory; patient IDs remain unavailable.
- Acquired official BUS-BRA v1 under verified CC BY 4.0 research scope,
  checked its official MD5, and inventoried 1,875 images / 1,064 source Case
  groups. No byte/pixel or close perceptual matches to 506 Mendeley/TCIA images.
  Kept it quarantined and prediction-free; Case-to-patient/exam semantics still
  need authoritative clarification before a patient-independent final claim.
- Full details and reproduction: `BOUNDED_DEVELOPMENT_RESULTS_V1.md`.
  Software regression: **72 passed, 29 subtests passed**, one upstream warning
  in 67.13 seconds. Preserved originals/copies were verified unchanged.
  Classification and gate availability remain disabled; no gate performance
  is inferred from classifier experiments. Local compute succeeded, so no
  Colab fallback was needed.

## Locked reserved-source evaluation (4 November 2026)

- Checked official BUS-BRA documentation and paper abstract/usage notes.
  Exact Case-to-patient/breast/lesion/exam semantics remain unresolved; the
  locked evaluation uses source Cases without patient-independence claims.
- Expanded overlap checking to all local raw/processed classifier data and
  TCIA inventory (1,544 distinct reference pixel arrays). Excluded one entire
  benign Case / two images for a dHash distance-5 candidate, not a confirmed
  exact duplicate. Locked 1,873 images / 1,063 source Cases before predictions.
- Refitted the selected frozen encoder/linear pipeline once on 162 development
  images only. Threshold 0.5, no calibration, fixed mean aggregation per Case.
  Timestamped protocol/model/dependency hashes preceded the sole inference run.
- Measured case confusion `[[11,710],[8,334]]`, sensitivity 0.9766,
  specificity 0.0153, balanced accuracy 0.4959, ROC-AUC 0.4704, AP 0.3033,
  binary Brier 0.6440 and ECE 0.6416. No post-result tuning or model retry.
- Reproduced primary metrics, all case intervals, aggregation, hashes and
  timestamp sequence from saved predictions without image inference.
  Old failed-model/prediction/manifests preservation checks still pass.
- Delivered `BUSBRA_LOCKED_PROTOCOL_V1.md`, grouping review and measured report.
  Original model hashes, predictions, calibration bins, commands and verification
  records remain in `reports/busbra_locked_v1/`. Software: 74 passed plus 29
  subtests, one upstream warning. No clinical reliability claim.
- BUS-BRA is now inspected, not prediction-free. No qualified fresh independent
  final cohort remains. Classifier offline evaluation did not evaluate the
  eligibility gate or upload-to-result system. Public classification stays off.

## Remaining external dependency

The software is a research prototype. The external-source evaluation now
demonstrates poor classifier performance. Further development requires separate
training/validation data, traceable challenging gate negatives, a trained and
evaluated eligibility gate, a new untouched final cohort after model changes,
and explicit expert review. The acquired breast-only source cannot fill the
missing gate categories or establish clinical usefulness.

See `COHORT_SOURCE_REVIEW_2026-11-02.md`, `DATASET_REGISTER.md`, and
`DATASET_CLARIFICATION_TRACKER.md` for evidence and unresolved source questions.
No gate checkpoint or approved evaluation record was created. No production
deployment occurred. `classification_enabled_by_server=false` and
`inference_enabled=false` remain required.
