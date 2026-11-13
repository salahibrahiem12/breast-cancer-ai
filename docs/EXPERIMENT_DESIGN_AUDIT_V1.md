# Experiment-design audit v1: counts and training diagnosis

Completed 4 November 2026, before a new full-eligible baseline fit. Public
classification remains disabled. Prior pilot and locked external results are
preserved, not corrected or replaced. All existing sources are inspected
development sources. This audit does not evaluate the eligibility gate.

## Count reconciliation

Each cell is **images / known source units / global connected groups**.
Mendeley known units are its old duplicate IDs, not patients. TCIA known units
are publisher-reported one-image-per-patient IDs. BUS-BRA known units are
source Cases with unresolved patient/breast/lesion/exam semantics. Global
groups union known units with byte/pixel and dHash/pHash distance<=5 links.
A dash means not recomputed at acquisition, not zero. Full eligible grouping
was rebuilt and agrees with every sampled pilot group ID.

| Source | Class | Acquired | Eligible | Noneligible | Compute-only omitted | Pilot sampled |
|---|---|---|---|---|---|---|
| mendeley | benign | 100/29/— | 71/22/17 | 29/7/— | 5/1/1 | 66/21/16 |
| mendeley | malignant | 150/43/— | 91/29/21 | 59/14/— | 23/6/5 | 68/23/16 |
| tcia | benign | 154/154/— | 99/99/99 | 55/55/— | 83/83/83 | 16/16/16 |
| tcia | malignant | 98/98/— | 98/98/98 | 0 | 82/82/82 | 16/16/16 |
| tcia | normal | 4/4/— | 0 | 4/4/— | 0 | 0 |
| busbra | benign | 1268/722/— | 1266/721/721 | 2/1/— | 1240/705/705 | 26/16/16 |
| busbra | malignant | 607/342/— | 607/342/342 | 0 | 579/326/326 | 28/16/16 |

Acquired classifier-source images total **2,381** (250+256+1,875); masks and
other non-image metadata are not separate classification observations.
Eligible total **2,232 images / 1,298 global groups**: Mendeley162/38,
TCIA197/197, BUS-BRA1873/1063. Acquired Mendeley has72 original duplicate units,
eligible51; additional pHash/dHash connectivity merges these into38 global
components. No new cross-source or contradictory-label components were found
over this eligible inventory. Hash screening cannot establish complete absence
of duplicate examinations, shared patients or source overlap.

The 149 noneligible images are not all bad-quality images or mandatory legal
exclusions: 88 Mendeley images (29 benign/59 malignant, 21 old duplicate units)
remain excluded because they link to the historical test partition; 55 TCIA
benign follow-up-only images fail the chosen biopsy reference-standard rule,
and four normal images are outside the benign/malignant target; two BUS-BRA
benign images in one Case were conservatively excluded for a close perceptual
match to previously available data. That match was not an exact duplicate.
These integrity, role and reference-standard rules were retained consistently.
None is a compute cap. Future reconsideration requires a separately documented
role/reference-standard decision, not silent reinstatement.

The pilot then randomly selected **16 pure global groups per class/source**,
seed42, and included every image in each selected group. That gives96 groups
and220 images (134 Mendeley,32 TCIA,54 BUS-BRA), not220 unique patients.
Exactly **2,012 otherwise eligible images / 1,202 groups** were omitted solely
to bound CPU work: Mendeley28/6, TCIA165/165, BUS-BRA1819/1031. Those reductions
were not integrity exclusions. The pilot was class-balanced by group rather
than representative of natural prevalence and must be called compute-limited.

## Per-fold training, validation and evaluation

Rows below show **images/global groups** separately for benign (B) and malignant
(M). The complete `reports/design_audit_v1/counts.csv` additionally records known
source-unit counts for every stage, source and class. Both pilot arms used
identical partitions. Each sampled image is evaluated once per arm, but belongs
to a training-side split in the two other source-held-out folds; these phase
counts therefore must not be summed as unique acquisitions.

| Held source | Contributing source | Class | Trained | Validation/calibration/threshold | Evaluated |
|---|---|---|---|---|---|
| mendeley | mendeley | benign | 0/0 | 0/0 | 66/16 |
| mendeley | mendeley | malignant | 0/0 | 0/0 | 68/16 |
| mendeley | tcia | benign | 12/12 | 4/4 | 0/0 |
| mendeley | tcia | malignant | 12/12 | 4/4 | 0/0 |
| mendeley | busbra | benign | 19/12 | 7/4 | 0/0 |
| mendeley | busbra | malignant | 21/12 | 7/4 | 0/0 |
| tcia | mendeley | benign | 49/12 | 17/4 | 0/0 |
| tcia | mendeley | malignant | 51/13 | 17/3 | 0/0 |
| tcia | tcia | benign | 0/0 | 0/0 | 16/16 |
| tcia | tcia | malignant | 0/0 | 0/0 | 16/16 |
| tcia | busbra | benign | 20/13 | 6/3 | 0/0 |
| tcia | busbra | malignant | 21/12 | 7/4 | 0/0 |
| busbra | mendeley | benign | 49/12 | 17/4 | 0/0 |
| busbra | mendeley | malignant | 51/13 | 17/3 | 0/0 |
| busbra | tcia | benign | 12/12 | 4/4 | 0/0 |
| busbra | tcia | malignant | 12/12 | 4/4 | 0/0 |
| busbra | busbra | benign | 0/0 | 0/0 | 26/16 |
| busbra | busbra | malignant | 0/0 | 0/0 | 28/16 |

The audit asserts disjoint global groups and complete source absence from
all training-side phases, including validation rows used for calibration and
threshold selection. The preserved runner hash agrees with the completed
result. It uses fixed C=1, fixed two epochs, fixed seeds and a final checkpoint,
with no outer-score branch selecting a model. Encoder ImageNet weights are
fixed before these experiments. Other folds intentionally use each inspected
source as development data; this does not make any fold an untouched final test.

Aggregation checks: all global groups are single-source and label-pure.
Sampled TCIA groups equal individual source IDs; BUS-BRA groups equal Cases;
Mendeley32 components contain44 old duplicate units. Mean calibrated image
probabilities are aggregated first, then compared with the fixed training-side
threshold. The label chosen for each mean is valid only after purity checks;
these checks were confirmed over both the pilot and full eligible inventory.

## Training adequacy

Frozen arm: fixed512-dimensional ResNet18 features, train-only StandardScaler,
balanced L2 logistic C=1/max_iter2000, default lbfgs. No encoder layers trained.
Solver iteration state/warnings were not retained with the pilot NPZ, so
convergence cannot be retroactively certified from an iteration count.

Fine-tuning: ImageNet ResNet18 layer4 and binary head only, **8,394,754 trainable
parameters**, AdamW lr1e-4/weight_decay1e-4, weighted cross entropy, two epochs,
batch at most16. One deterministic shuffled pass per epoch; no replacement,
oversampling or augmentation. BatchNorm running statistics stay frozen.
Class weights are N/(2*N_class), computed on each inner training side only.
Seeds42/43/44 follow held-source order. The final epoch2 is saved; no validation
early stopping or best-checkpoint selection. No per-epoch losses, optimizer
state or intermediate checkpoints were saved. Actual learning curves cannot
be reconstructed without retraining; initial/final evaluations below are
endpoints, not invented learning curves.

| Held source | Train B/M images | Class weights B/M | Updates | Train weighted CE initial→final | Validation weighted CE initial→final | Final train/validation image AUC |
|---|---|---|---:|---|---|---|
| mendeley | 31/33 | 1.0323/0.9697 | 8 | 1.0361→0.4862 | 1.0282→0.6339 | 0.952/0.752 |
| tcia | 69/72 | 1.0217/0.9792 | 18 | 0.7258→0.2477 | 0.7395→0.4585 | 0.982/0.873 |
| busbra | 61/63 | 1.0164/0.9841 | 16 | 0.7102→0.0560 | 0.7069→0.5543 | 1.000/0.825 |

Training clearly occurred: loss decreased and final training AUC ranges
.952–1.000. Thus uniformly inadequate fitting is not an established explanation
for external failure. The BUS-BRA-held fold nearly memorized the training side
(train CE.056/AUC1), with validation CE.554/AUC.825 and held image AUC.223.
This is a substantial generalization gap, compatible with overfitting/domain
shift; two endpoints cannot prove convergence or identify its cause.
Raw training predictions contain both classes in every final checkpoint;
BUS-BRA held images collapse to malignant, not all training features or all
training decisions. Calibration/thresholds also produce all-positive BUS-BRA
Case calls. TCIA raw calls31/32 malignant, with calibrated calls29/32 malignant.
The tiny number of updates and missing curves prevent a claim of adequately
trained fine-tuning, but the observed near-perfect training ranking also
prevents simply blaming lack of training for every failure.

## Why TCIA BA .594 and AUC .375 coexist

The saved threshold is0.6393432021. Confusion is `[[3,13],[0,16]]`:
sensitivity16/16=1; specificity3/16=.1875; BA=(1+.1875)/2=.59375.
All16 malignant scores lie above this threshold. Three benign scores lie
below it, while many of the other13 benign scores exceed malignant scores.
Of all256 malignant-versus-benign pairs, malignant scores win only96, with
zero ties: ROC-AUC=96/256=.375. Benign median=.847809; malignant median=.796252.
BA describes one operating point; AUC describes ranking across all thresholds.
There is no contradiction. Scores were not reversed and the threshold was not
retuned. Saved score summaries are in `tcia_ba_auc_explanation.json`.

## Source prediction diagnostic

Replayed from the same220 fixed embeddings. Three-fold shuffled seed42
StratifiedGroupKFold uses the same global connected groups; train-only scaling,
logistic C=1/max_iter2000. All fold train/validation group intersections are
asserted empty, and fold membership/source counts are saved. The within-class
repeats use three folds within benign only and malignant only, with48 groups
each (16/source), preserving duplicate components. These are diagnostics on
inspected sampled data, not new independent source-transfer tests.

| Diagnosis subset | Images/groups | Embedding source accuracy | Balanced accuracy | Training-majority accuracy | Diagnosis-only accuracy |
|---|---|---:|---:|---:|---:|
| all | 220/96 | 0.9909 | 0.9792 | 0.6091 | 0.6091 |
| benign | 108/48 | 0.9907 | 0.9792 | 0.6111 | 0.6111 |
| malignant | 112/48 | 0.9911 | 0.9792 | 0.6071 | 0.6071 |

The majority-source and diagnosis-only baselines are fitted on each training
fold. The latter uses only diagnosis to predict the most frequent source,
without embeddings. Full source-by-diagnosis image counts are Mendeley66/68,
TCIA16/16, BUS-BRA26/28. Aggregate majority-source accuracy is.6091 and balanced
accuracy1/3. High source prediction persists within diagnoses, so the99.1%
result cannot be explained solely by diagnosis prevalence differences. Shared
patients not represented by known IDs remain possible; no patient-independent
claim follows. Source information does not prove that a cancer classifier
uses that information or establish a causal shortcut mechanism.

## Decision before the next experiment

Do not search architectures. A measured CPU forward batch16 takes about1.73s,
and no CUDA GPU is available. Full fixed-encoder feature extraction is feasible
locally in minutes, followed by a convex linear optimization. Therefore define
one full-eligible frozen baseline, not another tiny fine-tuning substitute.
The separate timestamped protocol specifies all2232 images, four inner OOF
folds per source-held-out training side, all-training-side refit, training-only
weights/scaling/calibration/thresholds, convergence checks and optimizer curves.
Its15 target fits are one fixed baseline, not15 competing architectures.
Full curves certify optimization of that objective only; they cannot establish
clinical validity or adequate diversity of the development sources.

Resumable feature chunks and coefficient/model checkpoints support interruption
without silently reducing the dataset. No cloud run or patient-data upload is
needed for this baseline. Substantial neural training on this CPU is a separate
compute question, not performed or claimed here.

## Acquisition and blockers

No new acquisition or access-control bypass was performed. The official
BUSI_WHU v3 record is a CC BY4.0 lead; previous permitted routes returned403.
It remains inaccessible and unqualified, with no model predictions or verified
patient map. Existing sources are inspected. Fresh independent validation
remains blocked. Any newly obtained cohort stays prediction-free until rights,
labels, grouping and overlap are checked and a future protocol is locked.

Reproduce the audit without fitting target models:

```powershell
.\.venv\Scripts\python.exe -m ml.design_audit
.\.venv\Scripts\python.exe -m ml.pilot_training_diagnosis
.\.venv\Scripts\python.exe -m ml.report_design_audit
```

These commands write only `reports/design_audit_v1` and this report. Previous
artifact hashes are recorded there; previous model results remain unchanged.
