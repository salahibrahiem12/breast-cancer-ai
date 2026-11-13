# Cross-source exploratory development: measured results v1

Completed 4 November 2026. Public classification remains disabled. No deployment,
serving model replacement, eligibility-gate evaluation or UI changes were made.
These experiments use inspected sources and are exploratory development results,
not independent final testing or evidence of clinical reliability.

## Outcome

The encoder was loaded and replayed correctly, with noncollapsed features.
Cross-source discrimination remains inconsistent. Fine-tuning improved the
Mendeley fold but did not recover ranking on TCIA or BUS-BRA. No model is promoted,
no held-out threshold is retuned, and no low AUC is reversed after observing labels.

## Preservation, rights and roles

All 23 snapshots in `reports/cross_source_v1/preservation.json` match both the
original files and preserved copies. This includes failed checkpoints, locked
BUS-BRA reports/predictions/inventory and the earlier TCIA evaluation artifacts.
The old timestamped BUS-BRA protocol and failed results were not rewritten.

The user's current instruction explicitly authorizes TCIA and BUS-BRA reuse as
**inspected development sources**. They cease to be untouched external tests.
Recorded CC BY 4.0 dataset terms support this attributed local research work;
this is not a legal clearance for a clinical product or redistributed dataset.
Dataset citations and earlier rights evidence remain in the dataset register
and original reservation records.

Authoritative grouping evidence and limitations:

- TCIA BrEaST publisher documentation describes one image per patient; the
  selected 32 source IDs therefore provide source-reported patient units.
  Cross-source patient identity cannot be checked.
- BUS-BRA authors report 1,064 patients, and supplied metadata has 1,064 `Case`
  values. Accessible documentation does not explicitly define whether `Case`
  represents patient, breast, lesion or examination. Counts and fold coherence
  are supporting observations, not a verified dictionary. Use source Cases,
  never a patient-independent claim.
- Mendeley patient identifiers are absent. Its global duplicate components
  are split-control units, not known independent patients or clinical cases.

Primary references: [Mendeley v1](https://data.mendeley.com/datasets/wmy84gzngw/1),
[BrEaST dataset](https://www.cancerimagingarchive.net/collection/breast-lesions-usg/),
[BUS-BRA author repository](https://github.com/wgomezf/BUS-BRA),
[BUS-BRA release](https://zenodo.org/records/8231412),
[BUS-BRA paper](https://doi.org/10.1002/mp.16812).

## Encoder audit

Official cached ImageNet ResNet18 weights SHA256:
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
All supplied pretrained tensors match the loaded encoder exactly. Evaluation
mode is active; input is RGB; features are 512-dimensional global average
pooling after layer4 with an Identity head. Each source has zero constant
dimensions; centered ranks are 133/134, 31/32 and 53/54 images respectively,
the maximum centered ranks for these sample sizes. No collapse was observed.

Actual preprocessing is full-frame bilinear resize to 224x224, ToTensor and
ImageNet normalization mean (.485,.456,.406), std (.229,.224,.225). Development
and inference agree. The official pretrained reference instead resizes to 256
then center-crops to 224: normalization agrees, geometry differs. This is a
future training-side design question; current preprocessing was not changed
after observing held-out results.

Repeated batches, single-image versus batch extraction and an independently
assembled PIL transform have maximum error 0. Original development OOF
predictions replay within 1.031e-7 probability, with zero decisions changed.
The saved full-development scaler mean/scale replay exactly; float32 versus
float64 standardization differs by at most 8.096e-8 probability and changes
zero development decisions. No external predictions were needed for that check.

The initial pre-fit audit harness incorrectly assumed all legacy weight files
contain BatchNorm bookkeeping counters. It was corrected before any comparison
fit: 20 omitted `num_batches_tracked` counters initialize to zero, while all
supplied tensors must match. This was a harness compatibility issue, not
missing learned weights. The metrics verifier likewise restores original
float32 CSV probabilities before means; no predictions or fitted models changed.

## Locked bounded design and partitions

`docs/CROSS_SOURCE_PROTOCOL_V1.md` was written before fitting; its SHA256 and
runner SHA256 are stored in `result.json`. Six fits total: three held-out sources
times two arms, no hyperparameter sweep or retries based on scores. The run used
CPU with two threads. Split/sampling seed 42; fine-tuning seeds 42/43/44 in fixed
source order Mendeley/TCIA/BUS-BRA, as implemented before fitting. The protocol's
base-seed shorthand is expanded here; no seed was selected by outcome.

All 2,232 eligible images (162 Mendeley, 197 TCIA, 1,873 BUS-BRA) were considered
for known-group, SHA256/pixel, dHash and pHash distance <=5 connectivity. Original
old-test-linked Mendeley exclusions and prior BUS-BRA overlap exclusion remain.
No new cross-source or conflicting-label components were found in this eligible
inventory. This finite fingerprint search does not prove absence of duplicates,
source overlap or shared patients. Known groups and all detected links remain
together. Sixteen pure groups per class/source were sampled before prediction.

| Held-out source | Images benign/malignant | Global groups benign/malignant | Grouping endpoint | Image majority accuracy |
|---|---:|---:|---|---:|
| Mendeley | 66/68 | 16/16 | 32 duplicate components; 44 old duplicate units | .507 |
| TCIA | 16/16 | 16/16 | 32 source patient/case IDs, one image each | .500 |
| BUS-BRA | 26/28 | 16/16 | 32 source Cases, patient semantics unresolved | .519 |

All group-majority accuracies and majority-class balanced accuracies are .500.
The sample is deliberately class-enriched and small: neither calibration nor
precision/NPV estimates represent deployment prevalence. No population-wide
performance estimate is justified.

Frozen arm: ImageNet ResNet18 pooled features, training-only StandardScaler,
balanced L2 logistic C=1/max_iter=2000. Fine-tuned arm: same initial weights,
train layer4 and binary head for exactly two epochs, AdamW lr1e-4, weight decay
1e-4, weighted cross entropy, batch16, frozen BatchNorm running statistics,
no augmentation or early stopping. Class order is benign=0, malignant=1.
Both arms use the identical inner partitions. One source/label-stratified
grouped four-fold split supplies the first inner validation fold. Calibration
is regularized Platt logistic C=1 on that training-side validation only;
threshold maximizes its group balanced accuracy, ties specificity then higher
threshold. Models are not refitted afterward. Thus no held-out source contributes
to fitting, scaling, calibration, threshold or hyperparameter choice.

| Held source | Inner train images/groups | Inner validation images/groups | Frozen threshold | Fine-tuned threshold |
|---|---:|---:|---:|---:|
| mendeley | 64/48 | 22/16 | 0.528478 | 0.488221 |
| tcia | 141/50 | 47/14 | 0.445920 | 0.639343 |
| busbra | 124/49 | 42/15 | 0.407013 | 0.330739 |

## Primary mean-group results

Balanced accuracy (BA) is the predefined primary endpoint. Square brackets
are 95% percentile intervals from 1,000 seeded group bootstrap replicates.
For Mendeley these are duplicate-component intervals; TCIA uses source IDs;
BUS-BRA uses Cases. Unresolved identity prevents a general patient-independent
claim. Group scores are mean calibrated image probabilities, followed by the
training-side fixed threshold. Each pure group receives equal weight.

| Held source | Arm | BA [95% CI] | ROC-AUC [95% CI] | Sensitivity | Specificity | PR average precision | Binary Brier | ECE (10 bins) |
|---|---|---|---|---:|---:|---:|---:|---:|
| mendeley | frozen | 0.469 [0.312, 0.625] | 0.367 [0.182, 0.571] | 0.250 | 0.688 | 0.477 | 0.321 | 0.179 |
| mendeley | fine_tuned | 0.688 [0.541, 0.820] | 0.648 [0.441, 0.834] | 0.438 | 0.938 | 0.752 | 0.242 | 0.146 |
| tcia | frozen | 0.531 [0.361, 0.698] | 0.523 [0.302, 0.737] | 0.750 | 0.312 | 0.578 | 0.294 | 0.195 |
| tcia | fine_tuned | 0.594 [0.500, 0.695] | 0.375 [0.175, 0.585] | 1.000 | 0.188 | 0.420 | 0.352 | 0.261 |
| busbra | frozen | 0.500 [0.500, 0.500] | 0.453 [0.250, 0.663] | 1.000 | 0.000 | 0.528 | 0.314 | 0.226 |
| busbra | fine_tuned | 0.500 [0.500, 0.500] | 0.219 [0.067, 0.423] | 1.000 | 0.000 | 0.369 | 0.329 | 0.220 |

The two BUS-BRA arms call every Case malignant: sensitivity 1, specificity 0.
Their empirical BA intervals collapse to [.5,.5] because every observed
resample has the same behavior. This is not zero population uncertainty.
Only 32 groups/source produce wide, unstable intervals; they do not include
training/seed uncertainty or unobserved between-patient dependence. There is one
fixed training seed per outer fold, so seed variability was not estimated.
Report each source separately rather than pooling them into a favorable score.

Complete confusion matrices, precision, NPV, accuracy, log loss, fixed-.5
operating points and intervals for every reported scalar metric are in
`result.json`. Undefined denominator metrics are null, not invented zero values.
ECE is top-label confidence calibration with ten fixed bins, not a fitted
calibration slope. Brier here is binary mean (p-y)^2, not its doubled two-class
version. AP is malignant average precision, not trapezoidal PR area.

## Image-level secondary endpoints and calibration

Image intervals in `result.json` resample the same whole groups, keeping
multiple images together. Image weighting can differ from the primary endpoint.
TCIA image and group results coincide; BUS-BRA image and source-Case results
are both supported. Mendeley components cannot be called clinical cases.

| Held source | Arm | BA | ROC-AUC | Sensitivity | Specificity | AP | Binary Brier | ECE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| mendeley | frozen | 0.436 | 0.354 | 0.206 | 0.667 | 0.451 | 0.330 | 0.221 |
| mendeley | fine_tuned | 0.623 | 0.620 | 0.368 | 0.879 | 0.699 | 0.245 | 0.101 |
| tcia | frozen | 0.531 | 0.523 | 0.750 | 0.312 | 0.578 | 0.294 | 0.195 |
| tcia | fine_tuned | 0.594 | 0.375 | 1.000 | 0.188 | 0.420 | 0.352 | 0.261 |
| busbra | frozen | 0.519 | 0.424 | 1.000 | 0.038 | 0.477 | 0.317 | 0.250 |
| busbra | fine_tuned | 0.500 | 0.223 | 1.000 | 0.000 | 0.388 | 0.332 | 0.249 |

Training-side calibration parameters (logit-to-probability):

| Held source | Arm | Coefficient | Intercept |
|---|---|---:|---:|
| mendeley | frozen | 0.384158 | 0.011967 |
| mendeley | fine_tuned | 1.094480 | -0.402109 |
| tcia | frozen | 0.177649 | 0.450120 |
| tcia | fine_tuned | 1.548874 | -0.116710 |
| busbra | frozen | 0.221768 | 0.202544 |
| busbra | fine_tuned | 0.629254 | -0.333636 |

All fitted slopes are positive. Threshold effects cannot explain away poor
ROC ranking: for example TCIA fine-tuning achieves BA .594 but AUC .375 at its
training-side threshold. Calibration was fitted on very small reused inner
data and does not transfer reliably. No held-out recalibration was performed.

## Source and border diagnostics

A training-only-scaled, three-fold grouped source classifier predicts the
dataset from frozen embeddings with image accuracy .9909 (218/220), balanced
accuracy .9792, versus majority-image accuracy .6091. This documents strong
source information; it does not prove that source shortcuts caused target
failure. The PCA plot is a descriptive visualization of all inspected sampled
embeddings, not a transform used by the target classifiers.

Frozen models only were evaluated with the original calibration and threshold
after median-filling either outer 10% on each edge or center 80% of width and
height. Those remove approximately 36% or 64% of area respectively; they are
not equal-area interventions. Integer rounding depends on image resolution.

| Held source | Intervention | Group BA | Mean absolute image probability change | Image calls changed |
|---|---|---:|---:|---:|
| mendeley | border_removed | 0.531 | 0.279 | 39 |
| mendeley | center_removed | 0.500 | 0.382 | 36 |
| tcia | border_removed | 0.500 | 0.139 | 5 |
| tcia | center_removed | 0.750 | 0.289 | 19 |
| busbra | border_removed | 0.469 | 0.202 | 9 |
| busbra | center_removed | 0.562 | 0.217 | 15 |

Predictions are sensitive to both interventions. TCIA center removal happens
to increase BA to .750; it also deletes anatomy and introduces artificial
boundaries. This is an inspected-development observation, not evidence to
adopt that preprocessing. The interventions cannot isolate annotations from
anatomy, resolution or acquisition differences and do not establish causation.

## Fresh cohort and next evidence-producing step

[BUSI_WHU v3](https://data.mendeley.com/datasets/k6cpmwybk3/3),
DOI 10.17632/k6cpmwybk3.3, is an official CC BY 4.0 lead, described as 927
ultrasound images. It received no model predictions. Checked dataset/API
access routes returned HTTP403; a provider-cache naming-pattern probe also
returned403 and is not a verified download route. No files or patient mapping
were acquired. Consequently this is a metadata lead, **not a qualified reserved
external cohort**. Final independent validation remains blocked. BUS-UCLM
also remains an access-limited previously reviewed lead; neither a new source
name nor a license alone establishes independent patients or usable labels.

Next: obtain an official accessible archive and dictionary, verify reference
labels, patient/exam IDs, rights and overlap against all available sources;
reserve it without model-based inspection. Separately, expand training-side
source diversity and sample size before another predefined source-held-out
comparison, including reference-geometry versus full-frame preprocessing.
Do not choose a model or crop from the current outer scores or repeatedly
search these small partitions. The current two-epoch result is a bounded
candidate, not proof that all fine-tuning approaches fail.

## Reproduction and checks

From the repository root with the recorded data and old artifacts present:

```powershell
.\.venv\Scripts\python.exe -m ml.verify_cross_source --intervals
.\.venv\Scripts\python.exe -m ml.audit_cross_source_precision
.\.venv\Scripts\python.exe -m ml.report_cross_source
```

This verifies payload hashes, disjoint training/validation/held-out groups,
all preserved original hashes, all image/group metrics and group intervals;
it fits no model. For a complete rerun, use an isolated copy of this workspace
with the same input data, old reports and dependency versions, but an empty
`reports/cross_source_v1` directory, then run:

```powershell
.\.venv\Scripts\python.exe -m ml.cross_source
```

The runner refuses to overwrite an existing `result.json`. Do not delete this
run to reproduce it. Reruns are software reproduction, not fresh scientific
tests. Loaded runtime versions and distribution metadata are both recorded in
`environment.json`; use loaded runtime versions if they differ. Seed mapping is
also recorded there. Input payload hashes and group membership are in
`selection.json`; prior
manifests remain preserved. Each saved model is named for its held-out source,
not its training source. Scaler/linear coefficients are in NPZ; fine-tuned
state dictionaries are in PTH; calibration/thresholds are in `result.json`.
Use the cached official encoder plus each NPZ for the frozen arm. `artifact_hashes.json`
binds new model/results/code/protocol artifacts. Raw source images are not
included in the results bundle.

Software regression: 77 tests and 29 subtests passed (184.99 seconds), with
one upstream Starlette/AnyIO deprecation warning. The three new focused tests
also passed. These are software checks and say nothing about clinical reliability.
No measured eligibility-gate rejection performance is claimed. This offline
classifier comparison does not evaluate the upload-to-result system.
