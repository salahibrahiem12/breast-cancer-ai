# Full-eligible end-to-end GPU encoder fine-tuning protocol v2 (`gpu_encoder_finetune_v2`)

Predefined on 4 November 2026 as a separately versioned GPU protocol following
the local CPU neural-head adaptation run (`full_source_finetune_v1`,
[`FULL_SOURCE_FINETUNE_RESULTS_V1.md`](FULL_SOURCE_FINETUNE_RESULTS_V1.md)).
Public classification remains disabled (`BREASTCARE_CLASSIFICATION_ENABLED=false`).

## 1. Distinction from `full_source_finetune_v1`

- **`full_source_finetune_v1` (completed on local CPU):** Trained a 264,963-parameter
  nonlinear residual adapter head (`ResidualAdapterHead`: `LayerNorm(512) ->
  Linear(512, 256) -> GELU -> Dropout(0.25) -> Linear(256, 512)` + `gate` +
  `Linear(512, 2)`) on cached, detached 512-dimensional `ResNet-18` canonical and
  ultrasound-augmented embeddings while keeping all `11,176,512` convolutional
  backbone parameters frozen.
- **`gpu_encoder_finetune_v2` (this separately versioned GPU protocol):** Unfreezes
  the convolutional `layer4` residual stage of `ResNet-18` (`8,393,728` parameters,
  including `conv1`, `bn1`, `conv2`, `bn2`, and `downsample` across both BasicBlocks)
  together with the classification head (`Linear(512, 2)`, `1,026` parameters;
  total **8,394,754 trainable parameters** out of `11,177,538`), backpropagating
  gradients directly into the spatial convolutional filters on raw images at every
  mini-batch step.

## 2. Dataset, Grouping, and Partitions

- Identical 2,232 eligible images in 1,298 global duplicate/case groups from
  `reports/design_audit_v1/eligible_manifest.json` (`data/prepared/unified_dataset_manifest_v1.csv`).
- Outer splits: `mendeley`, `tcia`, `busbra` source-held-out folds.
- Inner splits: 4-fold `StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)`
  stratified by `(source, label)` and grouped by `global_group`.
- Every original image and all of its on-the-fly augmented views remain strictly
  inside the same `global_group` and partition.

## 3. On-the-Fly Raw-Image Ultrasound Augmentations and Optimization

- **Training transform (raw images per epoch):**
  - Random horizontal reflection (`p = 0.5`),
  - Random in-plane probe tilt rotation in `[-6°, +6°]` (bilinear),
  - Random aspect-preserving central crop (`[0.92, 1.00]` scale resized to `224 × 224`),
  - Random B-mode brightness/contrast gain jitter (`±8%`),
  - `ToTensor()` and ImageNet normalization (`mean=(0.485, 0.456, 0.406)`, `std=(0.229, 0.224, 0.225)`).
- **Validation and held-out transform:** Deterministic `Resize((224, 224))`, `ToTensor()`,
  and ImageNet normalization.
- **Differential learning rates (`AdamW`, `weight_decay=1e-2`):**
  - Unfrozen `layer4` convolutional/BN parameters (`8,393,728` params): `lr = 1e-4`
  - Classification head (`fc`, `1,026` params): `lr = 3e-4`
  - Frozen `conv1..layer3` (`2,782,784` params): `requires_grad = False` (BatchNorm in `conv1..layer3` kept in `eval()` mode so running statistics do not drift across small batches).
- **Budget & early stopping:** Up to 20 epochs per fit (`batch_size = 32`), cosine
  annealing schedule, gradient norm clipping (`max_norm = 2.0`), warmup 4 epochs,
  early stopping `patience = 5` on inner-validation sample-weighted BCE loss.
- **Resumable execution:** Saves per-epoch checkpoints (`.resume.pth`) and curves
  (`.curve.json`) under `reports/gpu_encoder_finetune_v2/`.

## 4. Exact Execution & Resume Commands (CUDA GPU / Colab / Kaggle)

```bash
python -m ml.gpu_encoder_finetune_v2 lock
python -m ml.gpu_encoder_finetune_v2 run --device cuda
```

