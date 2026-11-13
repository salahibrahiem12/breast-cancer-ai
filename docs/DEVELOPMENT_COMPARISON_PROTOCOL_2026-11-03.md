# Development-only comparison protocol

Recorded before fitting the following new comparison. TCIA is already an
inspected external set and will not be scored by either development candidate.

- Use the existing `mendeley_dev_v2` manifest: 121 training images, 41 validation
  images, and 88 images excluded because they are in or linked to the previously
  inspected original test set. Keep its duplicate groups and all exclusions.
- Patient IDs are unavailable. This is duplicate-grouped image-level
  development, not patient-independent development. No fresh final test exists.
- Verify file hashes, group separation, and dHash distances greater than 5
  between development partitions and excluded groups before fitting.
- Reproducible baseline: full-frame grayscale stretched to 32x32; training-only
  StandardScaler; balanced L2 logistic regression, C=1, lbfgs, max_iter=2000,
  seed=42. No augmentation or fitted calibration.
- Improvement hypothesis: reduce dependence on source contrast and aspect-ratio
  distortion using 1% per-image autocontrast followed by aspect-preserving
  resize/padding to 32x32, filled with that image's median gray intensity.
  Use exactly the same scaler/classifier settings as the baseline.
- Use the fixed 0.5 threshold and report both candidates, including negative
  results. Do not sweep thresholds/hyperparameters or choose a model using TCIA.
- Save coefficients, scaler parameters, source/manifest hashes, software
  versions, split counts, and validation predictions. Reconstruct probabilities
  from saved parameters to verify reproducibility.
- Treat validation differences as exploratory: these 41 images were already
  used for earlier development, and no independent final performance claim is
  possible. A candidate is not a replacement for the existing classifier or
  eligibility gate and cannot enable serving.
