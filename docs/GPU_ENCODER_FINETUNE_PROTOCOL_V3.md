# Full-eligible end-to-end GPU encoder fine-tuning protocol v3 (`gpu_encoder_finetune_v3`)

Predefined on 4 November 2026 after the global duplicate-group integrity audit
([`GROUPING_INTEGRITY_AUDIT_V1.md`](GROUPING_INTEGRITY_AUDIT_V1.md)) invalidated
the Mendeley grouping in `unified_dataset_manifest_v1.csv` that had been locked by
[`GPU_ENCODER_FINETUNE_PROTOCOL_V2.md`](GPU_ENCODER_FINETUNE_PROTOCOL_V2.md)
(`reports/gpu_encoder_finetune_v2/lock.json`).

Both `docs/GPU_ENCODER_FINETUNE_PROTOCOL_V2.md` and
`reports/gpu_encoder_finetune_v2/lock.json` are preserved intact in the audit
trail (`reports/gpu_encoder_finetune_v2/SUPERSEDED_BY_V3.json`). Zero GPU
training fits were executed under V2 before it was superseded by V3.
Public classification remains disabled (`BREASTCARE_CLASSIFICATION_ENABLED=false`).

## 1. Why Protocol V2 Was Superseded Before Training

The global duplicate-group audit (`ml/audit_grouping_integrity.py`,
`reports/grouping_audit_v1/grouping_audit.json`) established two concrete
defects in the Mendeley portion of `unified_dataset_manifest_v1.csv`:
1. **17 leaked Mendeley images from 8 historical test sequences:** The 250 acquired
   Mendeley images (`us1.bmp`..`us250.bmp`) are organized as 50 five-image
   shifted-crop sequences (`us{5k+1}..us{5k+5}`, `k = 0..49`). In `mendeley_dev_v2`,
   excluding `dHash <= 5` components touching the 50-image historical `mendeley_v1`
   test split left behind 17 sibling crop images across 8 test-linked five-image
   sequences whose border shift exceeded `dHash = 5` (`dHash = 6..16`), forming 8
   partial global groups of size `1..4` in the 162-image `v1` Mendeley set.
2. **2 split Mendeley five-image sequences (`us21..us25.bmp` and `us191..us195.bmp`):**
   Benign sequence Block `#4` (`us21..us25.bmp`, `93x57`) remained split into two
   `v1` global groups (`global:10` [`us21..23`] and `global:13` [`us24..25`]), and
   malignant sequence Block `#38` (`us191..us195.bmp`, `119x101`) remained split
   into `global:120` (`us191..194`) and `global:123` (`us195`).

`data/prepared/unified_dataset_manifest_v2.csv` (`reports/grouping_audit_v1/eligible_manifest_v2.json`)
excludes the 17 leaked Mendeley test-sibling images and reunites `us21..us25.bmp`
and `us191..us195.bmp` into their complete 5-image base-scan groups, yielding
**2,215 strictly clean eligible images across 1,288 validated duplicate/case groups**:
- **Mendeley:** `145` images in `28` complete base-scan duplicate groups (`13` benign
  groups [65 images], `15` malignant groups [80 images]: 27 groups of size 5 and
  1 group of size 10 [`us116..120` + `us226..230`]).
- **TCIA BrEaST:** `197` biopsy-confirmed images in `197` publisher-reported patient IDs.
- **BUS-BRA:** `1,873` images in `1,063` source-case groups.

## 2. Architecture and Frozen BatchNorm Specification (`ResNet18Layer4FineTuner`)

- **Backbone initialization:** ImageNet-pretrained `ResNet-18` loaded from
  `reports/bounded_development_v1/preserved/pretrained_resnet18.pth`
  (SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`).
- **Frozen early stages and ALL BatchNorm2d layers (`2,787,904` frozen parameters):**
  - `conv1`, `bn1`, `layer1`, `layer2`, `layer3` parameters have `requires_grad = False`.
  - **Every `nn.BatchNorm2d` module in the entire network—including `layer4.0.bn1`,
    `layer4.0.bn2`, `layer4.0.downsample.1`, `layer4.1.bn1`, and `layer4.1.bn2` (`9,600`
    total BatchNorm affine parameters across 20 modules)—has `requires_grad = False`
    on its affine `weight`/`bias` parameters AND is forced into `.eval()` mode
    (`bn.training == False`) inside `model.train()`** so that neither BatchNorm
    running statistics (`running_mean`, `running_var`) nor BatchNorm affine
    parameters drift across cross-source mini-batches.
  - `verify_frozen_batchnorm()` asserts after every `model.train()` call and after
    every epoch that all 20 `BatchNorm2d` modules have `training == False`,
    `grad is None`, and exact bitwise equality of `running_mean` and `running_var`
    against their initial pretrained buffers.
- **Trainable encoder and head modules (`8,389,634` trainable parameters):**
  - `layer4` convolutional filters (`layer4.0.conv1.weight`, `layer4.0.conv2.weight`,
    `layer4.0.downsample.0.weight`, `layer4.1.conv1.weight`, `layer4.1.conv2.weight`):
    `8,388,608` trainable parameters (`lr = 1e-4`).
  - Classification head `fc` (`Dropout(0.25) -> Linear(512, 2)`): `1,026` trainable
    parameters (`lr = 3e-4`).

## 3. Training, Validation, Calibration, Thresholding, and Resumable Checkpoints

- **Outer splits:** Leave-one-source-out across `mendeley`, `tcia`, `busbra`.
- **Inner splits:** 4-fold `StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)`
  stratified by `(source, label)` and grouped by `v2_duplicate_group` on the
  training-side sources (`s != held`).
- **On-the-fly raw-image training augmentations:** Horizontal flip (`p = 0.5`),
  rotation `[-6°, +6°]`, aspect-preserving central crop `[0.92, 1.00]` resized to
  `224 × 224`, brightness/contrast jitter `[0.92, 1.08]`, and ImageNet normalization.
  Every original image and its augmented views remain in the same group and fold.
- **Evaluation transform:** Deterministic `Resize((224, 224))`, `ToTensor()`, and
  ImageNet normalization, aggregated by arithmetic mean of probabilities per group.
- **Training-side selection:**
  - Early-stopping checkpoint (`selected_epoch`) is chosen strictly on inner-fold
    validation weighted BCE loss (`warmup_epochs = 4`, `patience = 5`, `max_epochs = 20`).
  - Final source model is refit on all training-side images for the median of the
    4 inner-fold `selected_epoch` values.
  - Platt scaling (`LogisticRegression(C=1)`) and operating threshold (`choose_threshold`)
    are fit strictly on training-side 4-fold out-of-fold (`OOF`) validation logits.
- **Resumable execution:** After every epoch, `fit_encoder_fold()` atomically saves
  `models/{name}.resume.pth` (containing `epoch`, `model_state_dict`,
  `optimizer_state_dict`, `scheduler_state_dict`, `best_epoch`, `best_key`,
  `best_state_dict`, `epochs_since_improvement`, `rng_state`, and `curve`) and
  `models/{name}.curve.json`. Re-running `python -m ml.gpu_encoder_finetune_v2 run --device cuda`
  automatically resumes from the last completed epoch of any interrupted fit.

## 4. Exact Minimal Launch Commands (Colab / Kaggle CUDA GPU)

```bash
python -m ml.gpu_encoder_finetune_v2 verify_package
python -m ml.gpu_encoder_finetune_v2 run --device cuda
```
