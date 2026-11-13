# Bounded development comparison v1

Recorded before fitting these models. Public classification stays disabled.
TCIA is inspected audit-only and will not be scored by any candidate.

Data: the 162 eligible Mendeley development images only; retain all 88
old-test-linked exclusions. No patient identifiers exist. Use three fixed
StratifiedGroupKFold folds, seed 42, on existing duplicate groups. Verify source
hashes and cross-partition dHash distance >5 using existing guards. All arms
use identical folds. These reused development images are exploratory.

Arms, one seed and no hyperparameter search:
1. Existing 32x32 grayscale stretched pixel features, train-only scaler,
   balanced L2 logistic regression C=1, lbfgs, max_iter=2000.
2. Frozen ImageNet ResNet-18 encoder, standard RGB/224/ImageNet normalization,
   512 pooled features; same train-only scaler and logistic classifier.
3. Modest fine-tuning: same ImageNet ResNet-18, seed 42 plus fold index,
   unfreeze layer4 and binary head only; two epochs, AdamW lr=0.0001,
   weight_decay=0.0001, weighted cross-entropy, batch 16, CPU threads 2.
   BatchNorm running statistics frozen; no augmentation, early stopping,
   validation-based epoch selection, temperature fitting, or pretrained downloads.

Budget: 3 arms x 3 folds; six total fine-tuning epochs. Cached official
pretrained weights only. Primary selection metric: mean fold balanced accuracy
at fixed threshold 0.5. Report all fold results and standard deviation, pooled
OOF metrics/calibration and duplicate-group bootstrap intervals. No replacement
of the production model. If compute fails, record failure and provide a Colab
runner for the same budget.

Threshold analysis: select one exploratory threshold on pooled development
OOF scores maximizing balanced accuracy, ties favor higher specificity then
higher threshold. This reuses validation scores and is optimistic; it is not
independent evaluation. Report sensitivity/specificity at fixed 0.5 separately.

Controlled ablations: for each trained pixel baseline, evaluate held-out fold
images after replacing the outer 10% band with image median gray, and after
replacing the central 80% with that median. No refitting or new threshold.
Report paired score changes and balanced accuracy. The intervention can remove
anatomy as well as artifacts; it cannot establish border causation. Per-source
causal analysis is unavailable with one development source and absent device IDs.

Fresh external cohort search: inspect official BUS-BRA rights/metadata and
cross-source duplicates, but generate no model predictions. Reserve the source
only if access, patient mapping and overlap evidence support that disposition.
No final inference until the development choice and protocol are locked.
