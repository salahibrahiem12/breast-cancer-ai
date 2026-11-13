# Full-eligible frozen baseline v1: measured results

Completed 4 November 2026 after the count reconciliation and training diagnosis
in [EXPERIMENT_DESIGN_AUDIT_V1.md](EXPERIMENT_DESIGN_AUDIT_V1.md).
Public classification remains disabled. No architecture search, deployment,
eligibility-gate measurement or raw patient-data upload was performed.

## Design and completion

The earlier220-image experiment is a compute-limited pilot, not conclusive
model development. Its missing epoch learning curves prevent a convergence
claim; saved endpoints also show nearly perfect training ranking in some folds,
so lack of fitting is not a sufficient explanation for poor transfer.

One fixed baseline was subsequently locked and run on **all2,232 eligible
images /1,298 global groups**, without a compute sampling cap. It uses the same
fixed ImageNet ResNet18 encoder, input transform and512 pooled features. Each
source-held-out side has four grouped inner fits for OOF scores, then one fit
on all images of the other two sources: **15 target linear fits total**.
All15 optimize the same predefined weighted regularized logistic objective.
No alternate hyperparameters, architecture, seed or held-out threshold was tried.

Source/diagnosis/group weighting prevents repeated images or the larger source
from automatically dominating the classifier objective. Calibration preserves
natural group diagnosis proportions with equal source/group weights. All
weights, StandardScaler fits, calibration and thresholds use only training-side
rows. Four-fold OOF scores supply calibration and threshold selection; every
training-side image then enters the final refit. No held-out source supplies
fitted preprocessing, labels or predictions for selection. The calibrated
OOF-to-final logit transfer is an approximation, not validated calibration.

This differs from the pilot in data volume, weighting and OOF/refit procedure;
any result difference cannot be attributed solely to more images or epochs.
All existing sources have already been inspected. These are exploratory
source-held-out development results, not an untouched final test.

| Held source | Final training images/groups | OOF calibration images/groups | Evaluated images/groups | Held images B/M | Held groups B/M |
|---|---|---|---|---|---|
| mendeley | 2070/1260 | 2070/1260 | 162/38 | 71/91 | 17/21 |
| tcia | 2035/1101 | 2035/1101 | 197/197 | 99/98 | 99/98 |
| busbra | 359/235 | 359/235 | 1873/1063 | 1266/607 | 721/342 |

`phase_counts.csv` reconciles every source/class for final training, all four
inner training/validation folds, OOF calibration, OOF threshold and evaluation,
with both known-unit and global-group counts. Phase totals are not distinct
acquisitions: OOF rows recur in three inner fits and in the final training fit.

## Primary mean-group endpoints

BA is the predefined primary endpoint at each training-side fixed threshold.
Intervals are95% percentile group bootstrap intervals from1,000 seeded draws.
Mendeley units are duplicate components, not clinical cases; TCIA units are
publisher-reported patient IDs, one image each; BUS-BRA uses source Cases whose
patient/breast/lesion/exam semantics remain unresolved. No general patient-
independent claim is made. Bootstrap intervals do not capture unresolved shared
patients, training-seed uncertainty or new-source variability.

| Held source | BA [95% CI] | ROC-AUC [95% CI] | Sensitivity | Specificity | Malignant AP | Binary Brier | Top-label ECE |
|---|---|---|---:|---:|---:|---:|---:|
| mendeley | 0.573 [0.421, 0.711] | 0.706 [0.519, 0.875] | 0.381 | 0.765 | 0.753 | 0.252 | 0.150 |
| tcia | 0.644 [0.578, 0.712] | 0.688 [0.615, 0.761] | 0.520 | 0.768 | 0.657 | 0.258 | 0.161 |
| busbra | 0.570 [0.538, 0.600] | 0.594 [0.554, 0.631] | 0.576 | 0.564 | 0.408 | 0.305 | 0.196 |

Cross-source performance remains modest: the fixed operating points detect
only38%–58% of malignant source groups. None of these outcomes establishes
clinical reliability or promotes a checkpoint. Additional data, weighting and
OOF calibration changed together relative to the pilot, so this is not a causal
estimate of the effect of longer training or increased sample size.
As an additional descriptive reference, a constant probability of.5 has binary
Brier.25 on any binary sample. All three measured Brier scores are larger;
training-side calibration has not demonstrated reliable probability transfer.

Complete confusion matrices and PPV/NPV, accuracy, log loss, all metric intervals,
and calibration bins with counts/mean probability/observed frequency are in
`result.json`. AP means average precision, not trapezoidal PR area; binary Brier
is mean(p-y)^2, not its doubled two-class form. Undefined denominators are null.
Calibration bin summaries are descriptive and receive no held-out refit.

| Held source | Group confusion [[TN,FP],[FN,TP]] | Threshold | Calibration coefficient/intercept | Majority-group accuracy | Training-majority accuracy | Constant training-prevalence group Brier |
|---|---|---:|---|---:|---:|---:|
| mendeley | [[13, 4], [13, 8]] | 0.421908 | 0.143157/-0.298094 | 0.553 | 0.447 | 0.289 |
| tcia | [[76, 23], [47, 51]] | 0.407195 | 0.263851/-0.183455 | 0.503 | 0.503 | 0.278 |
| busbra | [[407, 314], [145, 197]] | 0.619502 | 0.322889/0.087644 | 0.678 | 0.322 | 0.252 |

The held-source majority accuracy is an outcome-distribution reference, not a
deployable prediction rule. Training-majority and training-prevalence baselines
use training-side group labels only. Constant-class BA=.5. Any degenerate
bootstrap interval reflects the empirical observed prediction behavior, not
zero population uncertainty. No clinical acceptance threshold was invented.

## Image-level secondary results

Image CIs in the machine report resample whole global groups; repeated images
are not treated as independent. Group scores mean calibrated probabilities
before thresholding. Label purity and source-unit consistency were checked.

| Held source | BA | ROC-AUC | Sensitivity | Specificity | AP | Binary Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| mendeley | 0.569 | 0.646 | 0.407 | 0.732 | 0.722 | 0.262 | 0.140 |
| tcia | 0.644 | 0.688 | 0.520 | 0.768 | 0.657 | 0.258 | 0.161 |
| busbra | 0.565 | 0.573 | 0.600 | 0.531 | 0.375 | 0.327 | 0.224 |

## Optimization adequacy and limitations

All15 target fits passed the predefined optimizer-success and maximum absolute
gradient<=1e-5 checks. Actual optimizer curves are saved for every fit; the plot
shows the three final refits. This certifies optimization of the frozen-feature
regularized objective, not adequacy of the encoder, data diversity or clinical
use. A frozen encoder has no neural training epochs. No fully trained neural
fine-tuning candidate is claimed in this continuation.

| Held source | Final optimizer iterations | Final objective | Maximum absolute gradient | Final train image AUC | Training-side OOF group AUC |
|---|---:|---:|---:|---:|---:|
| mendeley | 252 | 0.256034 | 8.11e-07 | 0.911 | 0.742 |
| tcia | 257 | 0.194624 | 9.48e-07 | 0.936 | 0.780 |
| busbra | 96 | 0.063354 | 8.85e-07 | 1.000 | 0.754 |

Feature extraction took 450.8 seconds on the local CPU, reusing 220 identical pilot rows and hashing every input payload before new chunk creation. All2232 rows were retained.

Training-side OOF metrics share the scores used for calibration and threshold
selection, so they are diagnostic and optimistic for those fitted steps.
They are not additional independent validation. No outcome in this report
automatically promotes a model or justifies another architecture search.

## Reproducibility, preservation and blockers

The lock records the pre-fit UTC timestamp, protocol/runner/input hashes and
runtime versions. `LOCK_SHA256.txt` binds it. New chunks/checkpoints are
resumable; previous pilot/external reports and models were not overwritten.
Verification recomputes all15 training-only weighted scalers, all15 objectives
and gradients, OOF/final logits, calibration/threshold replay, metric tables and
both group/image intervals. All2,232 eligible rows have exactly one held-out
prediction. Prior artifact hashes and the disabled serving state also pass.

From the project root, existing run:

```powershell
.\.venv\Scripts\python.exe -m ml.verify_full_source_baseline --intervals
.\.venv\Scripts\python.exe -m ml.report_full_source_baseline
```

To reproduce training, use an isolated copy of the project with identical
inputs and the recorded environment, an empty `reports/full_source_baseline_v1`
directory, and the completed design audit present. Then:

```powershell
.\.venv\Scripts\python.exe -m ml.full_source_baseline lock
.\.venv\Scripts\python.exe -m ml.full_source_baseline features
.\.venv\Scripts\python.exe -m ml.full_source_baseline run
```

Resume features/run after interruption with the same commands and lock.
Completed fits are reused; partial coefficient checkpoints reset quasi-Newton
history, as recorded, without changing objective or hyperparameters. Do not
delete or overwrite the delivered run. Model NPZs include scaler and linear
parameters; training-side selection JSON holds calibration and threshold.
`artifact_hashes.json` binds delivered outputs/code, and the results ZIP omits
raw ultrasound images. The saved feature matrix supports fitting reproduction
without neural re-extraction when input receipts remain available.

No fresh external dataset was acquired and no access restrictions were bypassed.
BUSI_WHU remains an inaccessible, prediction-free licensed metadata lead with
unverified grouping. Final independent validation remains blocked pending an
official accessible archive and verified labels/grouping/overlap. Any new cohort
stays prediction-free until its protocol is locked. Existing sources remain
inspected development data. Eligibility-gate performance and the upload-to-
result system were not evaluated. Software tests do not establish clinical
reliability.
Six focused software tests passed in31.47s, covering the numerical objective
gradient, source/diagnosis/group weighting and existing aggregation/threshold
behavior. The earlier77-test regression run is historical, not a rerun here.
