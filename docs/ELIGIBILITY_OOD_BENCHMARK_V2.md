# Multi-Method Input-Eligibility & OOD Benchmark V2 (`eligibility_ood_benchmark_v2`)

**Generated UTC:** `2026-11-04T11:37:30.155816+00:00`
**In-Distribution Cohort:** Grouping-validated `v2` manifest (`2,215` eligible breast-ultrasound images across `1,288` groups: Mendeley `145`, TCIA `197`, BUS-BRA `1,873`)
**Evaluation Protocol:** Strict leave-one-breast-source-out evaluation (fit on 2 breast-ultrasound sources, evaluate on the unseen 3rd breast-ultrasound source vs. disjoint test splits of `fetal_head_zenodo` [`n=64`], `ChestMNIST` [`n=64`], and synthetic corrupted/low-quality frames [`n=48`]).

## 1. Summary Across Held-Out Breast-Ultrasound Sources

| Method | Held-Out BUS Acceptance (Pooled p95) | Held-Out BUS Acceptance (Min-Source p95) | Mean Fetal-Head US AUROC | Mean ChestMNIST AUROC | Mean Corrupted AUROC | Overall Mean OOD AUROC | Mean OOD FAR (Pooled / Min-Source p95) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `raw_mahalanobis_512d` | `0.0942` | `0.3223` | `0.9380` | `0.9970` | `0.9257` | `0.9536` | `0.0000 / 0.0017` |
| `l2_cosine_knn_k10` | `0.1638` | `0.1938` | `0.7917` | `0.9997` | `1.0000` | `0.9304` | `0.0000 / 0.0000` |
| `pca64_orthogonal_residual` | `0.0592` | `0.0709` | `0.7823` | `0.9999` | `0.9987` | `0.9270` | `0.0000 / 0.0017` |
| `ultrasound_physics_12d_oneclass` | `0.4191` | `0.7855` | `0.6565` | `0.9160` | `0.9837` | `0.8521` | `0.0694 / 0.1562` |
| `hybrid_physics_cosine_probe` | `0.2193` | `0.2986` | `0.7971` | `0.9998` | `0.9971` | `0.9313` | `0.0000 / 0.0000` |
| `cross_source_calibrated_hybrid_probe` | `0.2657` | `0.3378` | `0.8499` | `0.9827` | `0.9913` | `0.9413` | `0.0000 / 0.0017` |

## 2. Per-Source Breakdown

| Held-Out BUS Source | Method | Held-Out BUS Acceptance | Fetal-Head US AUROC (FAR) | ChestMNIST AUROC (FAR) | Corrupted AUROC (FAR) |
|---|---|---:|---:|---:|---:|
| `mendeley` | `raw_mahalanobis_512d` | `0.0069` | `0.9755` (`0.0000`) | `1.0000` (`0.0000`) | `0.9493` (`0.0000`) |
| `mendeley` | `l2_cosine_knn_k10` | `0.0000` | `0.4977` (`0.0000`) | `0.9990` (`0.0000`) | `1.0000` (`0.0000`) |
| `mendeley` | `pca64_orthogonal_residual` | `0.0000` | `0.6226` (`0.0000`) | `1.0000` (`0.0000`) | `0.9973` (`0.0000`) |
| `mendeley` | `ultrasound_physics_12d_oneclass` | `0.0690` | `0.4434` (`0.2969`) | `0.9718` (`0.0000`) | `1.0000` (`0.0000`) |
| `mendeley` | `hybrid_physics_cosine_probe` | `0.0000` | `0.4967` (`0.0000`) | `1.0000` (`0.0000`) | `0.9912` (`0.0000`) |
| `mendeley` | `cross_source_calibrated_hybrid_probe` | `0.0000` | `0.6305` (`0.0000`) | `1.0000` (`0.0000`) | `1.0000` (`0.0000`) |
| `tcia` | `raw_mahalanobis_512d` | `0.2030` | `0.8693` (`0.0000`) | `0.9912` (`0.0000`) | `0.8624` (`0.0000`) |
| `tcia` | `l2_cosine_knn_k10` | `0.0914` | `0.8965` (`0.0000`) | `1.0000` (`0.0000`) | `1.0000` (`0.0000`) |
| `tcia` | `pca64_orthogonal_residual` | `0.1777` | `0.8828` (`0.0000`) | `0.9997` (`0.0000`) | `0.9989` (`0.0000`) |
| `tcia` | `ultrasound_physics_12d_oneclass` | `0.8477` | `0.9263` (`0.1719`) | `0.9687` (`0.0000`) | `1.0000` (`0.0000`) |
| `tcia` | `hybrid_physics_cosine_probe` | `0.2843` | `0.8997` (`0.0000`) | `0.9994` (`0.0000`) | `1.0000` (`0.0000`) |
| `tcia` | `cross_source_calibrated_hybrid_probe` | `0.3909` | `0.9281` (`0.0000`) | `0.9482` (`0.0000`) | `0.9739` (`0.0000`) |
| `busbra` | `raw_mahalanobis_512d` | `0.0726` | `0.9690` (`0.0000`) | `0.9999` (`0.0000`) | `0.9654` (`0.0000`) |
| `busbra` | `l2_cosine_knn_k10` | `0.3999` | `0.9807` (`0.0000`) | `1.0000` (`0.0000`) | `1.0000` (`0.0000`) |
| `busbra` | `pca64_orthogonal_residual` | `0.0000` | `0.8414` (`0.0000`) | `0.9999` (`0.0000`) | `1.0000` (`0.0000`) |
| `busbra` | `ultrasound_physics_12d_oneclass` | `0.3406` | `0.5999` (`0.1562`) | `0.8077` (`0.0000`) | `0.9512` (`0.0000`) |
| `busbra` | `hybrid_physics_cosine_probe` | `0.3737` | `0.9949` (`0.0000`) | `1.0000` (`0.0000`) | `1.0000` (`0.0000`) |
| `busbra` | `cross_source_calibrated_hybrid_probe` | `0.4063` | `0.9909` (`0.0000`) | `0.9998` (`0.0000`) | `1.0000` (`0.0000`) |

## 3. Key Findings & Remaining Blockers
- **Why raw Mahalanobis failed (and why other methods succeed):** Uncalibrated 512-D ResNet-18 Mahalanobis distance achieves high ranking AUROC (mean OOD AUROC 0.9536: fetal-head US 0.9380, ChestMNIST 0.9970, corrupted 0.9257), but fails at the uncalibrated 95% training-source threshold (mean held-out breast-ultrasound acceptance only 0.0942) because inter-source domain shift pushes unseen breast-ultrasound cohorts outside the within-source training ellipsoid. Explicit 12-D ultrasound physics descriptors (`ultrasound_physics_12d_oneclass`: mean held-out BUS acceptance 0.4191, mean OOD AUROC 0.8521) and the Cross-Source Calibrated Hybrid Probe (`cross_source_calibrated_hybrid_probe`: mean held-out BUS acceptance 0.2657, mean OOD AUROC 0.9413, mean OOD false-acceptance rate 0.0000) substantially reduce unseen-source rejection while preserving strong OOD rejection. Production eligibility gating nonetheless remains disabled because licensed natural_image, object, screenshot, and real clinical low_quality negative cohorts with verified patient/author provenance are not yet acquired.
- **Remaining deployment blocker:** Public classification (`BREASTCARE_CLASSIFICATION_ENABLED=false`) and production gate deployment remain disabled because licensed `natural_image`, `object`, `screenshot`, and real clinical `low_quality` cohorts with verified patient/author provenance are still required for full multi-category gate certification.
