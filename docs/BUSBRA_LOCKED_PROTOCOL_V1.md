# Locked reserved BUS-BRA evaluation v1

The model and protocol below were frozen before reserved-cohort inference.
Public classification remains disabled. This is offline image/case research,
not an upload-to-result system or eligibility-gate evaluation.

Primary endpoint: source-Case balanced accuracy at 0.5. Aggregate all eligible
image probabilities by arithmetic mean within the recorded Case; count each
Case once with its consistent source pathology label. Patient, breast, lesion
and examination semantics are unresolved; do not claim patient independence.
Confidence intervals resample recorded Cases and are conditional on their
unverified independence. They cannot establish patient-level uncertainty.

Report sensitivity, specificity, PPV, NPV, balanced accuracy, accuracy,
ROC-AUC, malignant PR average precision and trapezoidal PR-AUC, binary and
summed two-class Brier, log loss, and 10-bin top-label ECE. Use 2,000 source-Case
percentile bootstrap replicates with seed 42; undefined metrics remain null.
Image-level metrics are secondary with Case-clustered intervals. Describe
source-case Device subgroups, including missing/mixed categories if present;
no subgroup-based selection, multiplicity claim, or clinical acceptance rule.

All source cases are included unless the pre-inference overlap exclusion
applies. Missing/corrupt images, unknown labels or contradictory within-Case
labels fail the run rather than silently removing observations. The expanded
reference check covers all available raw/processed classifier image data,
including historical/unverified sources for overlap only, plus TCIA inventory.
One entire Case (two images) was excluded before predictions. A near-hash
candidate is not proof of shared patients. Internal cross-Case dHash/pHash
candidate pairs at distance <=5 were also checked and none were found.

The official Zenodo metadata declares CC BY 4.0. The source paper's accessible
abstract/usage notes permit academic/research use. Cite Gomez-Flores,
Gregorio-Calas and Coelho de Albuquerque Pereira, DOI 10.1002/mp.16812, dataset
DOI 10.5281/zenodo.8231412, link CC BY 4.0, and identify derived changes.
This local research evaluation does not assert broader clinical authorization.
Official sources checked: https://zenodo.org/records/8231412 and
https://aapm.onlinelibrary.wiley.com/doi/abs/10.1002/mp.16812 .
The accessible paper does not define Case; no inference from matching counts
or published fold consistency substitutes for a data dictionary.

No calibration is fitted: use raw logistic probabilities. The threshold is
fixed at 0.5, consistent with model selection in development. No model, epoch,
threshold, aggregation rule or preprocessing may be changed based on BUS-BRA.
A single-refit frozen encoder is used, not an ensemble selected after results.

Machine-readable locked specification:

```json
{
  "created_utc": "2026-11-03T20:44:42.851668+00:00",
  "created_dubai": "2026-11-04T00:44:42.851712+04:00",
  "architecture": "ImageNet ResNet18 frozen, fc Identity, 512 pooled features + train-only StandardScaler + balanced L2 logistic C=1 lbfgs max_iter=2000",
  "class_order": [
    "benign",
    "malignant"
  ],
  "preprocessing": "RGB; bilinear Resize224x224 antialias True; ToTensor; mean .485,.456,.406 / std .229,.224,.225; full frame; no augmentation",
  "development_images": 162,
  "final_training": "one refit on all 162 eligible development images; 88 old-test-linked exclusions retained; no reserved data used",
  "threshold": 0.5,
  "calibration": "none; raw logistic probabilities; no external calibration",
  "seed": 42,
  "primary_endpoint": "balanced accuracy at fixed .5 over source Case mean image probability",
  "unit": "source Case, not verified patient, breast, lesion or examination",
  "aggregation": "arithmetic mean of all eligible image probabilities within Case; one consistent source binary label required",
  "intervals": "2000 percentile bootstrap resamples of source Case, seed42; conditional case-level uncertainty, not verified patient-independent CI",
  "secondary": "image metrics with source-Case cluster bootstrap; descriptive source-case Device subgroup, mixed devices labelled mixed; no multiplicity claims",
  "exclusions": "entire Case excluded for any byte/pixel match or dHash/pHash distance<=5 to any available classifier-source image; unknown/missing labels, missing/corrupt images or contradictory Case labels cause failure, no silent exclusion",
  "grouping_limit": "Official record and accessible paper abstract give patient counts but do not define the Case field. Full article was not accessible. No patient-independence claim; within-person correlation across distinct Case values cannot be ruled out.",
  "clinical_acceptance_threshold": null,
  "public_classification_enabled": false,
  "hashes": {
    "weights": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
    "linear": "3fd25a0a2d14a7c4c9d60c0000b7311652cdfde3ca5b283fbdab5614fbb8abfa",
    "development_manifest": "eee71039b79b4ade2cf6ef6e927c93bda37b074cb0b741a0587bb6a46c550ab9",
    "reserved_archive": "ba3e6ed19cc37c682d8d39e25435bbf8a555a12cb7e641b5f2117685c95580ff",
    "eligible_inventory": "c7e4fb66a5d64a8c3ca81e92cd3eaffe301b93b4dca46a02a8912d400436da8d",
    "precheck": "94dc29f9abd89f3989b1706f1417479516faf114c6ea32b3cf74fd2dbf2c26c9",
    "runner": "abdd9f459c3d690fc40b4a5e19e3a53ff42f9bed5aee2a9aad7f58ac844a37ea"
  },
  "software": {
    "torch": "2.15.0+cpu",
    "torchvision": "0.29.2+cpu",
    "sklearn": "1.9.1",
    "numpy": "2.5.4",
    "Pillow": "13.0.0"
  }
}
```

Lock SHA-256: `aefc3eb35010af97bd49c890057ad9588b007b5b2ba5ce4710b78934ad5f53f0`
