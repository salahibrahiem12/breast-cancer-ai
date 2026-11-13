# Full-Source Frozen Baseline Protocol V2 (`full_source_baseline_v2`)

## Purpose
Execute the frozen ResNet-18 linear baseline (`StandardScaler` + group/source/class-weighted `L2` logistic regression via `L-BFGS-B`, Platt calibration, and training-side balanced-accuracy threshold selection) on the grouping-validated **`v2` manifest** ([`reports/grouping_audit_v1/eligible_manifest_v2.json`](../reports/grouping_audit_v1/eligible_manifest_v2.json) / [`data/prepared/unified_dataset_manifest_v2.csv`](../data/prepared/unified_dataset_manifest_v2.csv)):
- **Total eligible images:** `2,215` (down from `2,232` in `v1` after excluding `17` Mendeley frames leaked from 8 partial five-image sequences whose sibling frames were in the historical `mendeley_v1` test split).
- **Total validated groups:** `1,288` (`28` Mendeley complete base-scan duplicate groups [27 of size 5, 1 of size 10], `197` TCIA `patient_id` groups, `1,063` BUS-BRA source `Case` groups after separating the accidental `dHash = 5` merge `bus_0426-l` / `bus_0888-r`).

## Fixed Scientific Settings (Unchanged from V1)
- **Backbone:** Frozen `ResNet18_Weights.IMAGENET1K_V1` (`512`-D `avgpool` embedding).
- **Classifier:** Group-, source-, and diagnosis-balanced `L2` logistic regression (`C = 1.0`, `L-BFGS-B`, `gtol = 1e-6`, `ftol = 1e-12`, `max_absolute_gradient <= 1e-5`).
- **Outer evaluation:** Leave-one-source-out across `mendeley`, `tcia`, and `busbra`.
- **Inner validation:** 4-fold `StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)` stratified by `source:label` and grouped by `v2_global_group_id`.
- **Calibration & thresholding:** Platt scaling (`LogisticRegression(C=1.0)`) and group-level source-weighted balanced-accuracy threshold fit strictly on inner out-of-fold predictions of the two training sources.
- **Serving policy:** `BREASTCARE_CLASSIFICATION_ENABLED=false` remains enforced.

