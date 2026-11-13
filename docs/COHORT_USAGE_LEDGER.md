# Cohort usage ledger

Updated 2026-11-04. Classification remains disabled. This ledger describes
actual use; it does not grant rights or establish clinical reliability.

## Full-eligible cross-source baseline & neural-head adaptation (4 November 2026)

All 2,381 acquired images across Mendeley v1 (250), TCIA BrEaST v1 (256), and
BUS-BRA v1 (1,875) are reconciled in `data/prepared/unified_dataset_manifest_v1.csv`
(`reports/unified_dataset_manifest_v1.json`): **2,232 eligible images in 1,298 global
duplicate/case groups** and **149 excluded images** (88 Mendeley historical-test-linked
duplicates, 55 TCIA follow-up-only benign, 4 TCIA normal, and 2 BUS-BRA Case 54
cross-source near-duplicate images). Masks (266 TCIA, 1,875 BUS-BRA) remain strictly
excluded from classification.

All 2,232 eligible images were used in two predeclared source-held-out experiments
where each source is held out in turn while the other two supply 4-fold grouped OOF
calibration, threshold selection, and final refitting:
- **Frozen linear baseline (`full_source_baseline_v1`):** [`FULL_SOURCE_BASELINE_RESULTS_V1.md`](FULL_SOURCE_BASELINE_RESULTS_V1.md)
- **Neural-head adaptation on frozen features (`full_source_finetune_v1`):** [`FULL_SOURCE_FINETUNE_RESULTS_V1.md`](FULL_SOURCE_FINETUNE_RESULTS_V1.md)

All three breast-ultrasound cohorts remain inspected exploratory development sources.
No fresh qualified independent final cohort is available (`BUSI_WHU` v3 official
download endpoints remain blocked with HTTP 403).

## Earlier role change: cross-source development pilot, 4 November 2026

The user explicitly authorized TCIA and BUS-BRA reuse as **inspected development
sources** after their failed evaluations. Their old locked artifacts remain
unchanged, but neither source can be called an untouched final test again.
The six predeclared source-held-out fits use 220 images in 96 global groups
from Mendeley, TCIA and BUS-BRA. All threshold/calibration fitting stays inside
each training side. These are exploratory development results; see
[CROSS_SOURCE_RESULTS_V1.md](CROSS_SOURCE_RESULTS_V1.md). BUS-BRA Case semantics
and Mendeley patient IDs remain unresolved. No fresh qualified external cohort
is reserved; BUSI_WHU is a prediction-free but inaccessible metadata lead.
The table and sections below retain the history preceding this role change.

| Cohort | Recorded use | Grouping and limits | Future final-test status |
|---|---|---|---|
| Official Mendeley v1, 250 images | Original model training/validation and an already inspected 50-image test. The later development manifest excludes 88 old-test-linked images, leaving 121 train and 41 validation. | Patient IDs absent. dHash distance <=5 groups are separated; 39 training and 12 validation duplicate groups. Hash groups cannot establish patient independence. | Ineligible as a fresh final test. |
| TCIA BrEaST v1, 256 case images | The 197 biopsy-confirmed cases were used for failed external evaluation and audit; 32 subsequently sampled for exploratory source-held-out development under the explicit role change above. The 55 follow-up and four normal cases remain excluded. | Publisher documents one image per patient. Cross-source identity and distinct examination IDs cannot be verified. | **Inspected development source**, never a fresh final test. The other 59 cases are not a separately sourced fresh cohort. |
| ChestMNIST and fetal-head candidates | Existing inventory/candidate work only; not part of this classifier audit or development comparison. | Different modalities/targets and unresolved readiness conditions. | Not a breast-cancer final test. |
| New independent cohort | **Unavailable; none has been reserved.** | Requires traceable rights, patient/exam mappings, suitable reference standards, duplicate/source checks, and a protocol fixed before predictions. | Required before a future final evaluation of any improved model. |

## Current locked BUS-BRA outcome (4 November 2026)

The prediction-free reservation was consumed by one locked offline evaluation
of the selected frozen encoder, refitted once on development data only.
Expanded overlap checks conservatively excluded one source Case / two benign
images. Evaluated 1,873 images aggregated into 1,063 source Cases (721 benign,
342 malignant). Case semantics remain unverified: this is not labelled
patient-independent. Balanced accuracy 0.4959, ROC-AUC 0.4704; confusion matrix
`[[11,710],[8,334]]`. No model/threshold retry occurred. BUS-BRA is now inspected
and cannot be treated as a fresh future test set. See
`BUSBRA_LOCKED_EVALUATION_REPORT_V1.md` and the timestamped protocol.

## Bounded experiment and source reservation (historical, 3 November)

The 162 eligible Mendeley development images were used for a predeclared
three-arm, three-fold duplicate-grouped comparison. All 88 old-test-linked
images remained excluded. No candidate was scored on TCIA.

Official BUS-BRA v1 was subsequently acquired under its CC BY 4.0 record for
limited research reservation: 1,875 images, 1,064 explicit source Case groups.
The inventory and metadata hashes are frozen in its local reservation record.
No predictions were generated. No byte/pixel or close dHash/pHash matches were
found against 506 Mendeley/TCIA images. This is a **quarantined reserved source**,
not yet a qualified patient-independent final cohort: authoritative patient/exam
semantics of Case remain unresolved. Thus the earlier unavailability statement
now applies to a qualified patient-independent cohort, not to an unscored
candidate archive. See `BOUNDED_DEVELOPMENT_RESULTS_V1.md`.

Both new development candidates load only Mendeley train/validation rows for
features and fitting. Excluded old-test images may be read only for hash-based
integrity/duplicate checks. Neither candidate was scored on TCIA. Validation
has been reused in previous development and remains exploratory.

Eligibility-gate development/evaluation is a separate task. No classifier
metric in this ledger measures eligibility or unrelated-image rejection.
