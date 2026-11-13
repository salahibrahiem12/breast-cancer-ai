# Model card: BreastCare AI research prototype

## Status

**Full-data cross-source baseline, neural-head adaptation, grouping audit & GPU package update, 4 November 2026:**
Following the design audit ([`EXPERIMENT_DESIGN_AUDIT_V1.md`](EXPERIMENT_DESIGN_AUDIT_V1.md))
and unified manifest reconciliation (`data/prepared/unified_dataset_manifest_v1.csv`,
2,381 acquired images -> 2,232 eligible in 1,298 global duplicate/case groups, 149 excluded),
two predeclared full-data source-held-out experiments were completed and verified on local CPU,
followed by a visual and graph audit of all global duplicate groups ([`GROUPING_INTEGRITY_AUDIT_V1.md`](GROUPING_INTEGRITY_AUDIT_V1.md)):
1. **Frozen linear baseline (`full_source_baseline_v1`, [`FULL_SOURCE_BASELINE_RESULTS_V1.md`](FULL_SOURCE_BASELINE_RESULTS_V1.md)):**
   L2-regularized logistic regression on 512-d frozen ImageNet `ResNet-18` embeddings
   across all 2,232 eligible images (all 15 L-BFGS-B fits converged with max gradient ≤ 1e-5).
   Held-out group ROC-AUC / balanced accuracy / sensitivity / specificity:
   - Mendeley (38 duplicate groups; no patient IDs): ROC-AUC `0.706 [0.519, 0.875]`, BA `0.573 [0.421, 0.711]`, Sens `0.381`, Spec `0.765`
   - TCIA BrEaST (197 publisher-reported patient IDs): ROC-AUC `0.688 [0.615, 0.761]`, BA `0.644 [0.578, 0.712]`, Sens `0.520`, Spec `0.768`
   - BUS-BRA (1,063 source Cases; unresolved patient semantics): ROC-AUC `0.594 [0.554, 0.631]`, BA `0.570 [0.538, 0.600]`, Sens `0.576`, Spec `0.564`
2. **Neural-head adaptation on frozen features (`full_source_finetune_v1`, [`FULL_SOURCE_FINETUNE_RESULTS_V1.md`](FULL_SOURCE_FINETUNE_RESULTS_V1.md)):**
   Trains `ResidualAdapterHead` (`264,963` trainable parameters across `norm`, `fc1`,
   `fc2`, `gate`, and `classifier`) on cached detached `ResNet-18` canonical and
   anatomy-preserving ultrasound-augmented features while keeping all `11,176,512`
   convolutional backbone parameters frozen (this is neural-head training on frozen
   features, not end-to-end convolutional backbone fine-tuning). Despite achieving
   `0.69–0.89` inner validation group ROC-AUC, held-out external performance was worse
   than the linear baseline across all three sources (source-specific overfitting in
   frozen feature space is an unproved hypothesis, not an established causal explanation):
   - Mendeley: ROC-AUC `0.448 [0.277, 0.651]`, BA `0.465 [0.367, 0.545]`, Sens `0.048`, Spec `0.882`
   - TCIA BrEaST: ROC-AUC `0.671 [0.593, 0.745]`, BA `0.562 [0.504, 0.620]`, Sens `0.276`, Spec `0.848`
   - BUS-BRA: ROC-AUC `0.573 [0.536, 0.611]`, BA `0.518 [0.497, 0.541]`, Sens `0.857`, Spec `0.179`
3. **Global duplicate-group audit & GPU encoder fine-tuning package ([`GROUPING_INTEGRITY_AUDIT_V1.md`](GROUPING_INTEGRITY_AUDIT_V1.md), [`GPU_ENCODER_FINETUNE_PROTOCOL_V3.md`](GPU_ENCODER_FINETUNE_PROTOCOL_V3.md)):**
   Visual and graph inspection showed that the 250 Mendeley files are 50 five-image
   shifted-crop sequences (49 unique base lesion scans), that 17 images in 8 partial
   `v1` groups were shifted-crop siblings of historical `mendeley_v1` test frames that
   escaped `dHash <= 5`, and that 2 five-image sequences (`us21..25` and `us191..195`)
   were split across two `v1` groups. Versioned manifest `data/prepared/unified_dataset_manifest_v2.csv`
   (`2,215` eligible images across `1,288` groups) corrects both issues and supersedes
   preserved lock `reports/gpu_encoder_finetune_v2/lock.json` with `reports/gpu_encoder_finetune_v3/lock.json`.
   Because local hardware is CPU-only (`torch.cuda.is_available() == False`), GPU
   encoder fine-tuning (`8,389,634` trainable `layer4` conv + `fc` parameters with all
   20 `BatchNorm2d` layers frozen) has **not** run locally and is packaged in
   `notebooks/colab_full_source_finetune.ipynb`.
4. **Grouping-validated frozen linear baseline (`full_source_baseline_v2`, [`FULL_SOURCE_BASELINE_RESULTS_V2.md`](FULL_SOURCE_BASELINE_RESULTS_V2.md)) & 6-method OOD benchmark ([`ELIGIBILITY_OOD_BENCHMARK_V2.md`](ELIGIBILITY_OOD_BENCHMARK_V2.md)):**
   On the cleaned `v2` manifest (`2,215` images across `1,288` groups), `full_source_baseline_v2`
   achieved held-out group ROC-AUC / BA of `0.672 / 0.546` on Mendeley (`28` complete base-scan groups),
   `0.682 / 0.624` on TCIA (`197` patient IDs), and `0.590 / 0.567` on BUS-BRA (`1,063` source Cases).
   A 6-method leave-one-breast-source-out OOD benchmark on `v2` showed that while raw pooled 512-D
   Mahalanobis accepts only `9.4%` of held-out breast-ultrasound images at its 95% training threshold
   despite `0.954` mean OOD AUROC, 12-D image-domain ultrasound physics descriptors under min-source
   p95 calibration raise held-out breast-ultrasound acceptance to `78.6%` (`0.852` mean OOD AUROC) and
   a cross-source calibrated hybrid probe achieves `0.941` mean OOD AUROC with `0.0%` pooled false-acceptance
   rate.
   No model is promoted and public classification remains disabled (`BREASTCARE_CLASSIFICATION_ENABLED=false`).

**Earlier pilot update, 4 November 2026:** TCIA and BUS-BRA
are now explicitly inspected development sources. A bounded six-fit comparison
on 220 sampled images found inconsistent transfer: frozen/fine-tuned group
ROC-AUC was .367/.648 on Mendeley, .523/.375 on TCIA and .453/.219 on BUS-BRA.
The encoder loaded correctly, features did not collapse, and original
development predictions replayed within 1.031e-7 with zero decision changes.
Dataset source was strongly encoded; this association does not establish the
cause of failure. No candidate is promoted. No qualified fresh final cohort
is reserved. See [CROSS_SOURCE_RESULTS_V1.md](CROSS_SOURCE_RESULTS_V1.md).
These exploratory results do not measure eligibility-gate or end-to-end system
performance. Public classification remains disabled.

**4 November 2026 offline update:** A separately selected frozen ImageNet
ResNet-18 encoder plus a single development-only refit linear classifier was
evaluated once on BUS-BRA. Source-Case balanced accuracy was 0.4959, ROC-AUC
0.4704 and specificity 0.0153 on 1,063 source Cases. Patient/exam semantics
remain unresolved. This pipeline is not enabled or clinically validated;
it is distinct from the older CNN evaluated on TCIA. See
`BUSBRA_LOCKED_EVALUATION_REPORT_V1.md`. No fresh final cohort remains.

**Public inference: disabled by default.** The server-side
`BREASTCARE_CLASSIFICATION_ENABLED` flag defaults to `false`; the browser cannot
override it. Inference also requires classifier and eligibility-gate
evaluation records that are explicitly reviewed for research use and bound to
the exact checkpoint hashes. No eligibility gate or approved evaluation record
is available, so public cancer classifications remain unavailable.

## Intended use and limitations

This project is for software and research workflow development only. It is not
a medical device and is not intended to diagnose, screen, triage, exclude
disease, or guide clinical decisions. A model output, when enabled in a future
research build, would require professional interpretation.

The API validates PNG/JPEG signatures, decoding, size, dimensions, and basic
quality flags. These safeguards are not modality proof. If a future gate
accepts an image, its result means only **ultrasound-like**; it does not verify
breast anatomy or diagnostic suitability.

The Mendeley-trained binary classifier now has a retrospective external-source
evaluation on 197 biopsy-confirmed BrEaST cases, with poor results described
below. No model has validated clinical calibration or a clinically validated
operating threshold. Cross-source patient identities cannot be verified.
The historical BUSI metrics documented in `RELIABILITY_AUDIT.md` came from an
image-level split with known near-duplicate candidates. They are descriptive
historical results only and are not results for a model offered by this site.

## Data and training

Dataset facts and rights status are recorded in
[`DATASET_REGISTER.md`](DATASET_REGISTER.md). The pipeline supports a
pretrained torchvision ResNet-18, deterministic seeds, fixed class ordering,
class-weighted loss, augmentation restricted to training, validation-based
checkpoint selection and temperature fitting, and a final test report.

Patient grouping is used when a complete image-to-patient CSV is supplied.
Without that mapping, split creation requires an explicit opt-in and reports
image-level uncertainty. Exact and perceptual near-duplicates are joined
before splitting. This does not establish external validity or eliminate all
possible leakage.

Mendeley version 1 (250 images, 100 benign, 150 malignant) was obtained from
its CC BY 4.0 record and prepared into an image-level, duplicate-grouped split.
Its source does not document patient IDs or Normal images. The split and
archive are local, ignored by Git, and detailed in `DATASET_REGISTER.md`.

The trained binary ResNet-18 was evaluated on a locked internal test partition
of 50 images. Accuracy was 0.76 and macro-F1 was 0.7024. Benign specificity at
the predeclared 0.5 malignant-probability threshold was 0.40 (12/20 benign
images were called malignant); malignant sensitivity was 1.00 (30/30). These
results are image-level, not patient-independent or external validation, and
do not establish clinical usefulness.

## Evaluation

The internal test report is stored at
`reports/mendeley_binary_v1/test_metrics.json`. The confusion matrix (rows =
actual benign/malignant, columns = predicted benign/malignant) is
`[[8, 12], [0, 30]]`. Macro ROC-AUC is 0.99 and macro average precision is
0.9900, but each estimate is based on this small, non-patient-independent
hold-out. Multiclass Brier score is 0.2349 and top-label ECE is 0.1411; these
do not demonstrate adequate calibration. No gate rejection metrics are
available. The subsequent external-source classifier results are reported below.

Software regression tests verify fail-closed state transitions and that the
cancer model is not called when the server flag is disabled or a gate rejects.
Generated fixture drawings are software test inputs only; they are not
real-world OOD evaluation images.

## External-source evaluation (2026-11-03)

The subsequent [pipeline and error audit](TCIA_PIPELINE_AUDIT_2026-11-03.md)
verified mappings and replayed the fixed checkpoint, added patient-group
intervals for all requested binary metrics, and reviewed errors locally. TCIA
is now explicitly an inspected evaluation set. Malignant average precision
is 0.4736; the earlier 0.4818 value is macro average precision over both classes.
Separate development experiments did not score new candidates on TCIA. No
fresh independent final cohort is currently reserved.

The fixed `mendeley_binary_resnet18_v1.pth` checkpoint (SHA-256
`69b0ef99a4d0c0f744ed1f809d12742250baa250c43969b85c0f9a5e720c6045`)
was evaluated once on 197 biopsy-confirmed TCIA BrEaST cases: 99 benign and
98 malignant. Preprocessing, validation-fitted temperature, and the existing
0.5 decision threshold were unchanged. No model selection or tuning used
this cohort. All 59 follow-up/normal cases were excluded before predictions.

Accuracy was **0.4822**, macro-F1 **0.3705**, ROC-AUC **0.4630**. The confusion
matrix was `[[6, 93], [9, 89]]` (rows actual, columns predicted; benign,
malignant). Sensitivity was **0.9082** and benign specificity **0.0606**.
Deterministic 1,000-iteration group-bootstrap 95% intervals were
0.8485–0.9608 for sensitivity and 0.0187–0.1089 for specificity.
Multiclass Brier score was 0.7671 and top-label ECE was 0.3489.

These results do not support enabling classification. The publisher documents
one scan per patient, but cross-source patient identities cannot be checked.
No byte/pixel matches or dHash/pHash pairs within distance 3 were found against
the 250-image Mendeley archive. This is not proof that all overlap is absent.
The biopsy-selected cohort cannot establish normal-class or eligibility-gate
performance. It is no longer an untouched future final test after this run.

Protocol: `TCIA_EXTERNAL_EVALUATION_PROTOCOL_2026-11-03.md`. Local report:
`reports/tcia_breast_biopsy_external_v1/test_metrics.json`. Source details:
`TCIA_ACQUISITION_AND_EVALUATION_2026-11-03.md`.

## Safety and output restrictions

- The API fails closed unless the server flag is on, both checkpoints exist,
  and their separately reviewed evaluation records match their SHA-256 hashes.
- No probability is shown when inference is unavailable.
- A gate-accepted image is called ultrasound-like only; it is not verified as
  breast anatomy or diagnostically suitable.
- Grad-CAM is not included in the website API. If used in separate research
  code, it is only an attention visualization, not a lesion boundary or proof.
- No lesion size, BI-RADS category, stage, or treatment recommendation is
  produced.
- Uploaded image bytes are not retained by the application. The multipart
  framework may temporarily spool larger request bodies to operating-system
  temporary storage and closes the upload after processing. Hosting, proxy,
  temporary-storage, and operator retention policies remain outside the
  app's control.

## Versioning

Model version is reported as `unavailable` unless a trained checkpoint with
its preparation and evaluation records is present. Existing legacy checkpoints
are not automatically loaded by the new service.
