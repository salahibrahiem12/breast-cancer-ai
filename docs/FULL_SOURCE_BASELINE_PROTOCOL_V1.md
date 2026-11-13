# Full-eligible source-held-out baseline protocol v1

Predefined after the pilot design audit and before new target-model fitting.
All three sources have already been inspected. This is exploratory development,
not a new independent test. Public classification remains disabled.

## Justification and fixed budget

The prior 220-image, two-epoch experiment was a compute-limited pilot, with
8–18 fine-tuning updates and no saved learning curves. It cannot establish
fine-tuning convergence or justify an architecture search. The next experiment
is one fully optimized, frozen ImageNet ResNet18 plus regularized logistic
baseline, using the entire eligible inventory. No alternative architecture,
preprocessing comparison, hyperparameter search or seed search is authorized
by this protocol. Three outer sources, four inner folds plus one final fit per
outer training side: exactly 15 target linear fits. Calibrators are separate.

## Data and grouping

Use all 2,232 previously eligible images: Mendeley162, TCIA197, BUS-BRA1873.
No per-source or per-class image cap. Retain 88 Mendeley historical-test-linked
exclusions, 55 TCIA follow-up-only benign and four normal exclusions, and two
BUS-BRA images in one conservatively excluded Case. These are integrity,
reference-standard/target-scope or historical-role exclusions, not compute
reductions. They are not all intrinsic image-quality failures.

Use the audited full known-unit plus SHA256/pixel/dHash/pHash-distance<=5
connected components. Cross-source or conflicting-label components are
quarantined, not split across sides. The current inventory has no such new
components. There are 1,298 global groups: Mendeley38, TCIA197, BUS-BRA1063.
Mendeley groups are duplicate components, TCIA IDs are publisher-reported
one-image-per-patient units, and BUS-BRA Cases have unresolved patient semantics.
Do not claim general patient independence or verified cross-source identities.

## Features and optimization

Use the preserved official ImageNet ResNet18 weights and exact pilot input
transform: RGB, full-frame bilinear224, ImageNet normalization, eval mode,
512 pooled features. No encoder fitting, augmentation or target-feature-driven
transform selection. Existing identical fixed-encoder embeddings may be reused
by payload hash; all other payload hashes are checked during extraction.
Feature computation on a held-out source performs no fitted preprocessing.

For each outer source, all its groups stay absent from every target fit,
scaler, calibration and threshold decision. Divide the other two sources with
four source/label-stratified grouped folds, shuffle seed42. Fit four inner
models and obtain one out-of-fold raw logit for every training-side image.
Then refit one final model on every eligible image of those two sources.

Train-only weighted StandardScaler and L2 logistic C=1. Image weights give
equal total mass to each training source, then each diagnosis within source,
then each known global group in that source/diagnosis, divided equally across
its images. Normalize total weight to the number of training groups. These
weights and scaler statistics are recomputed using only each fit's own rows.
No over/undersampling or held-out prevalence adjustment.

Optimize weighted mean binary cross entropy plus ||coef||²/(2*C*sum(weights)),
unpenalized intercept, using deterministic L-BFGS-B from zero. maxiter1000,
gtol1e-6, ftol1e-12, maxls50. Record objective/gradient learning curves,
iterations, final gradient, optimizer status and train loss for every fit.
Stop and report failure if the optimizer does not succeed or final maximum
absolute gradient exceeds1e-5; do not compensate with a smaller sampled run.
CPU two threads. Fixed completed fits and feature chunks can resume; partial
coefficient checkpoints can restart optimization with documented reset of
quasi-Newton history, not a new hyperparameter trial.

## Training-side calibration and threshold

Fit one Platt logistic C=1 to the inner out-of-fold logits only, with equal
source/group image weights (without diagnosis balancing), total weight equal
to training group count. Record coefficient, intercept and convergence.
No calibration labels or predictions from the held-out source are used.
No post-hoc score reversal. Apply this calibrator to final-refit logits;
acknowledge the OOF-to-final logit-distribution approximation as a limitation.

Aggregate calibrated OOF image probabilities by arithmetic mean per pure
global group. Choose the threshold maximizing equal-source-weighted group
balanced accuracy on these training-side OOF predictions. Ties favor weighted
specificity then higher threshold. No model selection from outer scores.

## Evaluation and reporting

Primary endpoint: mean-group balanced accuracy at the fixed training-side
threshold, each outer source separately. Report ROC-AUC, malignant average
precision, sensitivity, specificity, PPV, NPV, confusion matrix, binary Brier,
log loss and ten-bin calibration summaries. Report image-level metrics also,
with group-cluster uncertainty. TCIA/BUS-BRA permit source-ID/Case aggregation;
Mendeley components are not clinical cases. Compare held-out majority-class
accuracy/BA, training-majority calls and constant training-prevalence Brier.

1,000 seeded percentile group bootstrap replicates; no training-seed uncertainty
or unresolved within-person dependence is captured. Full source class counts
and natural eligible prevalence replace the pilot's artificial class-balanced
sample. Degenerate intervals must be identified. Report all exclusions,
optimization failures and calibration limitations, without choosing a favorable
source or combining them into a clinical reliability claim.

## Reproducibility and fresh data

Timestamp/hash the protocol, runner, full manifest, weights and previous-result
hash ledger before fitting. Save models, OOF scores, fold manifests, optimizer
curves, calibration, thresholds, held-out predictions and dependency versions.
No new cohort is qualified. BUSI_WHU remains a prediction-free official-record
lead whose checked download routes returned403; do not bypass access controls.
Any genuinely new evaluation data remain prediction-free until qualified and
its future protocol locked. Eligibility-gate performance remains unmeasured.
