# BreastCare AI — Engineering & Research Tasks

Last updated: 2026-11-04

## Phase 1 — Establish Current State & Complete Full-Data Baseline Verification
- [x] Inspect repository structure, git status, datasets, manifests, checkpoints, reports, backend, and frontend (`web/`).
- [x] Complete `ml.report_full_source_baseline` packaging (`docs/FULL_SOURCE_BASELINE_RESULTS_V1.md`, `reports/full_source_baseline_v1/phase_counts.csv`, `reports/full_source_baseline_v1/optimizer_learning_curves.png`, `reports/full_source_baseline_v1/artifact_hashes.json`, `reports/design_audit_full_baseline_v1.zip`).
- [x] Re-verify `ml.verify_full_source_baseline --intervals` and confirm saved-model replay (`max_error = 2.22e-16`), fold isolation, confidence intervals, and disabled serving state.
- [x] Create `TASKS.md` with acceptance criteria and current blockers.

**Acceptance criteria (Met):** All 15 target linear fits verified (`max_absolute_gradient <= 1e-5`), OOF/final probabilities replay within `1e-12`, 1,000-draw group bootstrap intervals match stored artifacts, and all 51 prior locked artifacts remain hash-identical.

## Phase 2 — Data Integrity, Unified Manifest, Duplicate-Group Audit & Acquisition Audit
- [x] Build and verify the initial unified dataset manifest (`data/prepared/unified_dataset_manifest_v1.csv` and `reports/unified_dataset_manifest_v1.json`) covering all 2,381 acquired breast-ultrasound images (Mendeley 250, TCIA 256, BUS-BRA 1,875) plus candidate/excluded inventories.
- [x] Audit global duplicate groups visually and graph-theoretically (`ml/audit_grouping_integrity.py`, `docs/GROUPING_INTEGRITY_AUDIT_V1.md`, `reports/grouping_audit_v1/grouping_audit.json`, `reports/grouping_audit_v1/contact_sheets/*.png`), explaining why 162 Mendeley images form 38 `v1` groups (50 five-image shifted-crop sequences across 49 unique base lesion scans), uncovering 17 leaked Mendeley partial-sequence test siblings and 2 split 5-image sequences (`us21..25` and `us191..195`), preserving `v1`, and issuing versioned `data/prepared/unified_dataset_manifest_v2.csv` (`reports/grouping_audit_v1/eligible_manifest_v2.json`: 2,215 strictly clean eligible images across 1,288 validated duplicate/case groups) with explicit invalidation scope.
- [x] Verify label mappings (`benign=0`, `malignant=1`), confirm zero segmentation masks admitted into image rows (266 TCIA masks and 1,875 BUS-BRA masks accounted separately), and never call duplicate groups or BUS-BRA source `Case` units patients.
- [x] Document official external source checks and access status (including HTTP 403 / missing-file blockers on BUS-UCLM v3 / BUSI_WHU) without bypassing restrictions.

**Acceptance criteria (Met):** Reconciled counts and edge types audited across all 2,381 acquired images; visual contact sheets inspected; `v1` preserved and `v2` (`2,215` eligible images, `1,288` groups) versioned with explicit invalidation scope.

## Phase 3 — Model-Development Experiments (`full_source_finetune_v1` on CPU & Packaged `gpu_encoder_finetune_v3` for CUDA)
- [x] Predefine protocol (`docs/FULL_SOURCE_FINETUNE_PROTOCOL_V1.md`) and execute full-data source-held-out neural-head adaptation on frozen features (`ml/full_source_finetune.py`, `reports/full_source_finetune_v1/`): `ResidualAdapterHead` (`264,963` trainable parameters) on cached detached `ResNet-18` canonical and ultrasound-augmented features across all 2,232 `v1` eligible images (`1,298` groups).
- [x] Transparently resolve the `task-329` `backbone_trainable_params` assertion failure in `ml/verify_full_source_finetune.py`, `reports/full_source_finetune_v1/lock_audit_trail.json`, and `docs/FULL_SOURCE_FINETUNE_RESULTS_V1.md`: explicitly distinguish (1) standalone `net()` `requires_grad` flags (`11,176,512`), (2) `AdamW` optimizer membership (`0` backbone params, `264,963` head params), (3) autograd gradients, and (4) historical parameter changes (`||theta_saved - theta_init||_2 > 0`) verified across all 15 saved checkpoints on disk from their deterministic seeded initializations (`torch.manual_seed(seed)`).
- [x] Correct scientific interpretation across reports and UI: state clearly that `ResidualAdapterHead` performed worse externally than the frozen linear baseline across all three held-out sources, and that **source-specific overfitting in frozen feature space is an unproved hypothesis, not an established causal explanation**.
- [x] Inspect local CUDA resources (`torch.cuda.is_available() == False` on `PyTorch 2.15.0+cpu`), preserve `reports/gpu_encoder_finetune_v2/lock.json` (`SUPERSEDED_BY_V3.json`), issue protocol V3 (`docs/GPU_ENCODER_FINETUNE_PROTOCOL_V3.md`, `reports/gpu_encoder_finetune_v3/lock.json`) bound to `unified_dataset_manifest_v2.csv` (`2,215` images, `1,288` groups), and verify the runnable, resumable Colab GPU package (`ml/gpu_encoder_finetune_v2.py`, `notebooks/colab_full_source_finetune.ipynb`) with frozen BatchNorm across all 20 `BatchNorm2d` modules (`8,389,634` trainable `layer4` conv + `fc` parameters). **State plainly: GPU encoder fine-tuning has NOT run locally because no CUDA GPU is accessible.**
- [x] Execute `full_source_baseline_v2` (`ml/full_source_baseline_v2.py`, `docs/FULL_SOURCE_BASELINE_RESULTS_V2.md`, `reports/full_source_baseline_v2/`) on the grouping-validated `v2` manifest (`2,215` eligible images across `1,288` groups: `145` Mendeley in `28` complete base-scan groups, `197` TCIA in `197` patient IDs, `1,873` BUS-BRA in `1,063` source cases) to provide the exact linear reference baseline for `gpu_encoder_finetune_v3` and quantify the empirical delta of the `v1 -> v2` grouping corrections.

## Phase 4 — Input Eligibility and Abstention Gate
- [x] Audit available negative modality data and legal/provenance status (`fetal_head_zenodo`, `ChestMNIST`, missing licensed `natural_image`, `object`, `screenshot`, `low_quality` cohorts).
- [x] Implement and run both the initial read-only feature-space OOD diagnostic (`ml/audit_ood_eligibility_candidates.py` -> `reports/eligibility_ood_candidate_audit_v1.json`) and the 6-method cross-source OOD & physics benchmark on the `v2` manifest (`ml/ood_gate_v2_benchmark.py` -> `reports/eligibility_ood_benchmark_v2.json`, `docs/ELIGIBILITY_OOD_BENCHMARK_V2.md`), empirically proving why raw pooled Mahalanobis fails at the 95% threshold (`0.0942` held-out BUS acceptance) despite high ranking AUROC (`0.9536`), and how min-source p95 calibration with 12-D ultrasound physics descriptors (`0.7855` held-out BUS acceptance) and cross-source calibrated hybrid probes (`0.9413` mean OOD AUROC, `0.0000` pooled FAR) overcome that failure mode while keeping the production gate fail-closed (`Unable to assess`) until all required negative categories are lawfully available.
- [x] Verify file signature, decoded format, byte size, pixel dimensions, near-uniform checks, and guarantee that rejected/unsupported uploads never invoke the cancer classifier or return cancer probabilities.

**Acceptance criteria (Met):** Strict fail-closed behavior verified; rejected and unsupported inputs never call `run_cancer_classifier`; clear distinction between implemented software checks, exploratory OOD measurements on available candidate sets, and unmeasured full-category gate performance.

## Phase 5 — Complete Website and API Integration
- [x] Update `/model-evidence` page (`web/app/model-evidence/page.tsx`) and backend `/model-info` endpoint (`backend/app/main.py`) to reflect actual measured artifacts (197-case TCIA external evaluation, 1,063-case BUS-BRA locked evaluation, 2,232-image full-source frozen baseline, and 2,232-image neural-head adaptation comparison) with clear limitations.
- [x] Verify `/analyze`, `/how-it-works`, `/privacy`, `/about`, and `/` pages for accurate privacy/retention explanations, supported-input instructions, and responsive desktop/mobile layout.
- [x] Confirm server-side `BREASTCARE_CLASSIFICATION_ENABLED=false` cannot be overridden by the client, temporary uploads are cleaned up, CORS is restricted, and logs/errors are sanitized.

**Acceptance criteria (Met):** Frontend and backend display real measured metrics from saved JSON reports; no fake metrics, BI-RADS, lesion sizes, or unverified claims; production build, lint, and typecheck pass.

## Phase 6 — End-to-End Verification
- [x] Run comprehensive Python (`pytest`) and frontend (`npm test`, `npm run typecheck`, `npm run lint`, `npm run build`) test suites covering real checkpoint loading, preprocessing consistency, valid/invalid/corrupted/oversized uploads, missing weights, disabled flag, client bypass attempts, and temp-file cleanup.
- [x] Run live backend + frontend integration smoke tests and verify HTTP responses across all routes and upload states.

**Acceptance criteria (Met):** 100% of backend, ML, and frontend tests pass; live server probe confirms fail-closed abstention and zero classifier leakage.

## Phase 7 — Runnable Handover & Documentation
- [x] Provide Windows 11 PowerShell startup script (`start_breastcare.ps1`) and verify startup commands.
- [x] Update `README.md`, `docs/DATASET_REGISTER.md`, `docs/MODEL_CARD.md`, `docs/COHORT_USAGE_LEDGER.md`, `docs/PROGRESS.md`, and `TASKS.md`.
- [x] Generate versioned evaluation bundles (`reports/design_audit_full_baseline_v1.zip`, `reports/full_source_finetune_v1_bundle.zip`) excluding raw patient data, with SHA-256 hashes.

## Phase 8 — Containerization, Safe Deployment & Infrastructure
- [x] Author backend Dockerfile.api (Python 3.11-slim, CPU PyTorch, non-root user breastcare:1000, port 8000, automated healthcheck).
- [x] Author frontend web/Dockerfile (multi-stage Node 20-alpine build, Next.js production bundle, non-root user nextjs:1001, port 3000, automated healthcheck).
- [x] Author root docker-compose.yml declaring api and web with inter-service health dependency (condition: service_healthy) and fail-closed safety (BREASTCARE_CLASSIFICATION_ENABLED=false).
- [x] Prepare root .dockerignore, web/.dockerignore, and .env.example templates preventing credential, cache, and dataset leakage.
- [x] Complete comprehensive deployment and hosting evaluation (docs/DEPLOYMENT.md), identifying Self-Hosted Docker Compose (Local / On-Premise) as the safest free hosting option and explaining why public deployment of unvalidated clinical classifiers is strictly prohibited.
- [x] Verify local development service recovery and health after environment restart (FastAPI on port 8000, Next.js on port 3000, both returning HTTP 200).

## Phase 9 — Google Colab GPU Transfer Bundle & Execution Readiness
- [x] Package verified Google Colab transfer bundle at `reports/gpu_encoder_finetune_v3/breastcare_colab_v3_bundle.zip` (223.11 MB, 513 files, SHA-256 `08ab005ae4ce7dbcbe0fd9a95ec30e21b4e47142dffd5f36abbe584774728f03`).
- [x] Generate cryptographic transfer manifest `reports/gpu_encoder_finetune_v3/breastcare_colab_v3_bundle_manifest.json` with per-file byte counts and SHA-256 digests.
- [x] Verify archive integrity via `ZipFile.testzip()`, match all 513 file entries and SHA-256 digests against archive contents, and verify all 9 locked V3 hashes.
- [x] Verify CC BY 4.0 dataset licensing and source provenance records for all 2,215 eligible images (`145` Mendeley, `197` TCIA, `1,873` BUS-BRA).
- [x] Provide turnkey Google Drive upload and Google Colab execution instructions using `notebooks/colab_full_source_finetune.ipynb`.

## Current External Blockers (Recorded, Non-Blocking for Local Execution)
1. **Fresh untouched external evaluation cohort:** Mendeley, TCIA BrEaST, and BUS-BRA have all been inspected during evaluation/development. Official BUS-UCLM v3 / BUSI_WHU download routes returned HTTP 403 or exposed no file entries on Mendeley Data.
2. **Patient identifiers in Mendeley & Case semantics in BUS-BRA:** Mendeley v1 provides no patient IDs (grouped by exact/perceptual duplicate components); BUS-BRA provides 1,064 `Case` IDs whose exact patient/exam/lesion mapping is not explicitly defined in the public archive.
3. **Licensed multi-category eligibility-gate dataset:** While fetal-head ultrasound (Zenodo 8265464) and ChestMNIST (Zenodo 10519652) are inventoried locally, rights-verified natural photos/books, screenshots, and real low-quality ultrasound cohorts with patient/source-disjoint splits are not available. Consequently, `BREASTCARE_CLASSIFICATION_ENABLED` must remain `false`.
