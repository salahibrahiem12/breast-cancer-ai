# BreastCare AI reliability audit

**2026-11-03 external-source update:** An official TCIA BrEaST acquisition
enabled a locked 197-case biopsy-only evaluation of the fixed Mendeley-trained
classifier. Accuracy was 48.2%, ROC-AUC 0.463, and benign specificity 6.1%:
93 false positives and nine false negatives. These results do not support
enabling classification. Earlier entries retain their original scope/date.
See [source integrity, protocol and measured results](docs/TCIA_ACQUISITION_AND_EVALUATION_2026-11-03.md).

**Status:** Classification is disabled until a breast-ultrasound eligibility
detector has been independently validated. This is a research prototype, not a
medical device or a substitute for clinical assessment.

## Root cause

The deployed classifier is a closed-set ResNet-18 with three outputs:
`benign`, `malignant`, and `normal`. The app decoded an uploaded file, resized
it to 224 × 224, converted it to RGB, applied ImageNet normalization, and
always selected the largest softmax output. There was no out-of-distribution
(OOD) or modality gate. A book photograph therefore still received one of the
three classes; softmax does not verify that an input is a breast ultrasound.

No separate image-modality model, training data for such a detector, validated
eligibility threshold, or OOD evaluation set was found. File type, dimensions,
grayscale conversion, and basic image statistics cannot prove image modality.
Training a detector or claiming an OOD rejection rate without appropriate,
diverse labeled examples would be unsupported. Accordingly, the app and every
image prediction/explanation API now fail closed with:

> Unsupported image: Unable to verify that this is a breast ultrasound image.

The initial legacy Streamlit path uses this message for rejected uploads. The
new API distinguishes unsupported/unsuitable, unable-to-assess,
model-unavailable, and research-classification-disabled states; it never
classifies unless the server flag and hash-bound evaluation approvals pass.

## Implementation review and changes

- **Class mapping:** `CLASS_NAMES` is `benign`, `malignant`, `normal`; dataset
  labels and output indices use this same order. No label-order mismatch was
  identified in the inspected inference/training code.
- **Image preprocessing:** training augmentation and deterministic evaluation
  both resize, convert to tensor, and apply ImageNet normalization. Inference
  and Grad-CAM use the deterministic evaluation transform. The RGB handling is
  consistent; it does not provide modality validation.
- **Model behavior:** softmax on three closed-set logits always sums to one.
  The top value is not a calibrated probability. No temperature scaling,
  reliability diagram, Brier score, or expected calibration error was found.
  Confidence and per-class softmax percentages were removed from the app.
- **Malignant bias:** historical test predictions contain 31 malignant calls
  among 119 images (26.1%), versus 32 malignant ground-truth images (26.9%).
  There were 26 true-positive malignant predictions and 5 false-positive
  malignant calls; 6 malignant cases were missed (5 called benign, 1 called
  normal). These image-level results do not explain or make safe a malignant
  prediction on an unrelated image.
- **Class balance:** the saved image-level split contains train counts
  305/146/93, validation 65/31/19, and test 66/32/21 (benign/malignant/normal).
  The BUSI trainer computes inverse-frequency weights from training data.
- **Overfitting/leakage:** saved ResNet-18 train accuracy/macro-F1 were 95.4% /
  0.952, versus validation 86.1% / 0.845 and test 83.2% / 0.818. The split
  script randomly partitions images, not patients. Near-duplicate image pairs
  were previously found across train/validation and train/test. Patient IDs
  are unavailable, so patient-level independence and residual leakage cannot
  be verified.
- **Clinical metrics:** the existing test split is not a genuinely independent
  patient-level set. It was not reused to tune this fix. The historical values
  below are descriptive only; they do not justify diagnostic use.
- **Lesions and size:** no validated detection/segmentation model or known
  image scale was found. The interface no longer claims lesion detection,
  lesion sizing, or clinically validated Grad-CAM interpretation.

## Historical image-level metrics

These are from the saved ResNet-18 report and are not recalculated as part of
this safety change. PR-AUC and calibration metrics were not saved, so they are
reported below only as additional descriptive calculations on the same
previously used image-level test split. Neither is independent validation.

| Class | Support | Precision | Recall / Sensitivity | F1 | Specificity | OvR ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Benign | 66 | 0.862 | 0.848 | 0.855 | 0.830 | 0.915 |
| Malignant | 32 | 0.839 | 0.813 | 0.825 | 0.943 | 0.941 |
| Normal | 21 | 0.739 | 0.810 | 0.773 | 0.939 | 0.939 |
| **Overall / macro** | **119** | — | — | **0.818 macro-F1** | **0.904 macro** | **0.931 macro** |

Accuracy: **0.832**. Confusion matrix (rows = actual; columns = predicted;
class order benign, malignant, normal):

```text
[[56,  5,  5],
 [ 5, 26,  1],
 [ 4,  0, 17]]
```

Calibration is **not validated**. OOD/modality rejection sensitivity and
specificity are **not measurable from the available data**. Independent
patient-level performance is **not available**.

Additional one-pass, descriptive computations from the same 119 test images:

| Metric | Value |
|---|---:|
| Macro average precision (OvR; stepwise PR-AUC) | 0.894 |
| Benign average precision | 0.933 |
| Malignant average precision | 0.854 |
| Normal average precision | 0.894 |
| Multiclass Brier score (sum over classes) | 0.248 |
| Top-label ECE (10 equal-width bins) | 0.047 |

These are not patient-independent estimates or evidence of satisfactory
calibration. The ECE is bin-dependent and estimated from only 119 images; no
calibration method or threshold was fitted using this set.

## Regression coverage and verification

`tests/test_inference_safety.py` exercises the prediction and explanation
entrypoints with generated illustrative non-ultrasound-like images,
low-quality/corrupted images, and (when local BUSI files are available)
benign/malignant/normal images. It also checks the exact Streamlit error. A
sentinel model verifies that classification is never invoked. This tests the
fail-closed behavior, **not** the accuracy of ultrasound detection or rejection
of real-world book/natural/other-medical photographs.

## Required work before enabling classification

1. Obtain a representative, consented modality dataset containing breast
   ultrasound scans and diverse non-ultrasound/other-medical images, with
   patient-level identifiers and independent sites/devices where possible.
2. Train a separate eligibility/OOD detector using patient-grouped splits.
   Choose any operating threshold only on validation data and predefine the
   acceptable false-accept rate.
3. Measure sensitivity/specificity and confidence intervals on a locked,
   patient-independent external test set, including book/photos, screenshots,
   unrelated medical images, blur, corruption, and low-quality scans.
4. Evaluate class calibration, PR-AUC, per-class confusion/error slices and
   malignant false negatives on an independent patient-level test set.
5. Only then consider enabling classification, with clinical oversight and
   prospective validation. Do not treat softmax scores or Grad-CAM as
   calibrated uncertainty or validated lesion localization.

At the time this audit was first written, no model was retrained, no threshold
was invented, and no accuracy improvement was claimed. The later classifier
experiment is reported separately below.

## Research website and pipeline implementation

The Next.js website and FastAPI service are separate from the legacy Streamlit
app. The API validates file extension, actual decoded PNG/JPEG format, image
dimensions, request size, and basic quality flags. These checks are upload
safeguards, not proof of breast-ultrasound modality. Classification requires a
server-only opt-in flag, both model artifacts, hash-matching evaluation
records, and explicit research approvals; browser inputs cannot enable the
flag. Unsupported/rejected images never receive cancer classifications,
probabilities, or explanations. The present system remains disabled because
there is no trained eligibility gate or approved evidence record.

The new PyTorch pipeline is versioned independently from the legacy model. It
supports two- or three-class data only when those labels are present, fixed
class ordering, patient grouping when a complete mapping is supplied,
SHA-256 and perceptual-hash duplicate grouping, training-only augmentation,
validation-based selection and temperature fitting, and an untouched final
test evaluation with class metrics, calibration metrics, plots, and bootstrap
intervals. A missing patient map requires explicit image-level split opt-in.
An external evaluation mode requires a distinct source/manifest and matching
class set.

### Dataset provenance and current results

`docs/DATASET_REGISTER.md` records the BUSI article's 780-image, 600-woman
dataset facts (487 benign, 210 malignant, 133 normal, with masks), absent
usable patient IDs in the examined public distribution, and the unresolved
rights behind its third-party Kaggle mirror. It also records Mendeley version
1 (250 images: 100 benign, 150 malignant) and its DataCite CC BY 4.0 record.
Mendeley does not document Normal images or patient IDs. The official archive
was later downloaded and ZIP-validated (3,264,338 bytes; SHA-256
`87dbd3aaa89e2d8886839983be3a985b7b190d5d8cb167d95a8dcd5d224b63f7`); all
250 BMP images were present. The earlier incomplete archive remains separate
and was not used.

The first prepared seed-42 manifest grouped exact and near-duplicate
candidates before an explicitly image-level split. It was later regenerated
at the correct source root with train 150 (60 benign, 90 malignant), validation
50 (20 benign, 30 malignant), and test 50 (20 benign, 30 malignant), with 126
groups total. Patient IDs were not supplied, so none of these partitions are
patient-independent.

The initial CPU run took several minutes per epoch; the completed run selected
epoch 6, stopped early after epoch 12, and saved a versioned checkpoint. It was
then evaluated on the fixed test partition. The results and limitations are in
the 2026-10-30 update below; no external or patient-independent classifier
evaluation is available.

### Software checks and boundaries

The regression tests cover API status/results, file decoding and format
mismatch, oversize requests, local ultrasound inputs when available,
model-unavailable state, upload-spool cleanup, classifier call ordering,
duplicate-group splits, and validation preprocessing. Generated fixtures test
software behavior only; they do not measure real-world rejection quality.
See the repository README for setup and exact commands. Passing these tests
does not validate clinical reliability, privacy infrastructure outside the
application, or general OOD detection.

Verification completed for this implementation:

- `python -m pytest -q tests\test_ml_pipeline.py`: **10 passed**.
- `.venv\Scripts\python.exe -m pytest -q tests\test_backend_api.py`:
  **12 passed**, with 7 upload subtests.
- `python -m pytest -q tests\test_pipeline.py tests\test_inference_safety.py`:
  **12 passed**.
- `python -m py_compile` on changed Python entrypoints and `git diff
  --check`: passed.
- A live Uvicorn smoke check confirmed `/health`, `/model-info`, abstention for
  a generated unrelated image, and rejection of corrupt bytes. The API reports
  inference disabled and returns no classification.
- Combined ML and legacy safety tests: **22 passed**. A separate full-suite
  run in the API-only `.venv` could not collect ML/legacy tests because that
  environment lacks NumPy, PyTorch, and Streamlit; the targeted test groups
  passed in their respective environments.
- Frontend dependencies are locked in `web/package-lock.json`. Next.js and
  `eslint-config-next` were upgraded together to 15.5.28 after npm flagged
  15.3.3 with critical and high advisories. PostCSS 8.5.32 and Sharp 0.35.5
  overrides address remaining vulnerable transitive versions.
- `npm audit --omit=dev --audit-level=high`: **0 vulnerabilities**.
- `npm run build`, `npm run lint`, and `npm run typecheck`: passed. A local
  production-server smoke test returned HTTP 200 for all six pages.

These are software checks only. External validation and OOD rejection
performance remain unavailable.

## 2026-10-30 implementation and evaluation update

The official Mendeley binary experiment did complete after this earlier audit
draft. Its seed-42 split is duplicate-grouped but image-level because patient
IDs are not available: 150 train, 50 validation, and 50 locked test images.
The selected checkpoint was evaluated once on that hold-out; metrics are
recorded in `reports/mendeley_binary_v1/test_metrics.json`:

- Accuracy 0.76; macro-F1 0.7024.
- Benign: precision 1.00, recall 0.40, F1 0.5714.
- Malignant: precision 0.7143, recall/sensitivity 1.00, F1 0.8333.
- Confusion matrix (actual rows benign/malignant, predicted columns
  benign/malignant): `[[8, 12], [0, 30]]`.
- Malignant specificity 0.40 at the predeclared 0.5 probability threshold;
  12 benign cases were falsely called malignant.
- Macro ROC-AUC 0.99; macro average precision 0.9900. Calibration: multiclass
  Brier 0.2349 and top-label ECE 0.1411.

This is not patient-independent or external validation. The high AUC estimates
from only 50 images do not offset the poor benign specificity or establish
clinical performance. This classifier is not approved for website use.

The API now checks a server-only `BREASTCARE_CLASSIFICATION_ENABLED` flag
(default `false`), classifier checkpoint/evaluation approval and matching
hash, and an independently evaluated eligibility checkpoint/evaluation
approval and matching hash before attempting cancer inference. The eligibility
preparation/training/evaluation code is implemented, but no licensed
non-breast images, other-anatomy ultrasound negatives, trained gate checkpoint,
threshold-selection set, or gate test results were available. Generated
book/nature/medical-like fixtures are contract tests only. Therefore the
eligibility gate is unavailable, no gate acceptance or false-acceptance
metrics can honestly be reported, and public classification remains disabled.

Quality-limit correction: the first real Mendeley benign image inspected was
95x75 pixels and was incorrectly rejected by the former 96-pixel-side check.
The default minimum side is now 64 pixels so that sample passes basic file
quality checks. This simple size floor is not a quality classifier and does
not validate the image or its modality.

## 2026-10-30 classifier audit and development-only evaluation

The Mendeley binary pipeline was traced end to end:

- Manifest labels are `benign` and `malignant`; the loaded checkpoint records
  the same explicit order, and inference maps output indices through that
  checkpoint order. No label-encoding mismatch was found.
- Training and inference both convert PIL images to RGB, resize to 224 × 224,
  convert to tensors, and normalize with ImageNet mean `(0.485, 0.456, 0.406)`
  and standard deviation `(0.229, 0.224, 0.225)`. The loader now rejects
  checkpoints whose recorded size or normalization differs from inference.
- The legacy checkpoint selected epoch 6 because its selection policy
  lexicographically prioritizes malignant recall over macro-F1 and loss.
  At that epoch validation malignant recall was 1.00 and macro-F1 was 0.790.
  Later epochs in the same training run recorded validation macro-F1 0.919
  but malignant recall 0.867. This is direct evidence that the prior selection
  objective preferred a high-recall operating point with more benign
  false-positive risk; it is not evidence that the test split should be used
  to retune the model.
- The split has 250 images and 126 duplicate groups, with no patient IDs.
  Cross-partition shared exact SHA-256 values and shared groups are both zero;
  no cross-split dHash pairs at distance ≤3 were found. At distance 4 there
  are 29 cross-partition pairs; at distance 5 there are 30 additional pairs.
  These are near-duplicate candidates not caught by the original threshold.
  Patient-level overlap cannot be ruled out, and this image-level split is not
  certified leak-free.
- The already-inspected 50-image test split was not reloaded or used for any
  new model/threshold choice. The current checkpoint's validation-only report
  is `reports/mendeley_binary_v1/validation_ablation/baseline.json`. It has
  accuracy 0.82, macro-F1 0.790, confusion matrix `[[11, 9], [0, 30]]`,
  malignant sensitivity 1.00, and benign specificity 0.55 at 0.5. At fixed
  threshold 0.75, validation sensitivity remained 1.00 and specificity was
  0.75 (5/20 benign false positives). The same validation partition selected
  the checkpoint and fit temperature; these are development measurements,
  not independent performance estimates or a deployment threshold.

The most plausible observed contributor to weak benign specificity is the
recall-first checkpoint selection policy on a small, imbalanced, non-patient-
grouped dataset, alongside rapid train-set saturation. The data do not support
a more specific causal claim. Validation-only ablations for class weights,
augmentation, and a frozen pretrained backbone were subsequently run on the
leak-safer development split; their results are recorded below. No public
setting is changed, and no new cancer-model performance is claimed from the
old test set.

The Zenodo fetal-head archive was checksum-verified (129,809,023 bytes;
published MD5 `52f24da73f61cab9f3b1cfafa6c69cf2`) and all 999 image files
decoded. Its perceptual-hash inventory produced 991 dHash<=3 groups; it has
no patient IDs, and all split assignments remain unassigned. This is only one
`other_ultrasound` source. The archive is not cleared for training until
curator review resolves original-image provenance and attribution.

The official MedMNIST+ ChestMNIST 64 archive was downloaded under its stated
CC BY 4.0 record terms; its published MD5
`9de6cd0b934ebb5b7426cfba5efbae16` was verified (401,604,127 bytes). It is a
candidate `other_medical` source. The archive contains 78,468 training,
11,219 validation, and 22,433 test images, each 64x64 with 14 multi-label
targets; patient identifiers are not exposed in this archive. Its source
rights/provenance review is pending. It does not cover natural photographs,
objects, screenshots, other-anatomy ultrasound beyond fetal head, or real
low-quality scans. These acquired candidate datasets do not support training
or evaluating a general eligibility gate; public classification remains
disabled.

### 2026-10-30 leak-safer development split

The Mendeley binary dataset was regrouped at perceptual-hash distance <=5,
then every group connected to the previously inspected 50-image test split
was excluded from development. The generated manifest at
`data/prepared/mendeley_dev_v2/split_manifest.csv` has 250 source rows and 72
duplicate groups; 88 images in test-linked groups were excluded. Remaining
development partitions contain 53 benign / 68 malignant training images and
18 benign / 23 malignant validation images. No final test partition was
generated. There are no patient IDs, so these duplicate-grouped image-level
partitions are not patient-independent and are not an independent performance
evaluation.

### Measured validation-only ablation results (development split only)

All results below are duplicate-grouped, image-level development metrics from
the leak-safer Mendeley development manifest (`data/prepared/mendeley_dev_v2/`).
They are **not** independent test results: validation was used for checkpoint
selection and temperature fitting, patient IDs are unavailable, and the
previously inspected test partition is excluded. No public classification is
enabled by these results.

| Configuration | Macro-F1 | Malignant recall | Benign specificity | Confusion matrix (benign/malignant rows) | Brier | Top-label ECE |
|---|---:|---:|---:|---|---:|---:|
| Baseline: no class weights, no augmentation, random init | 0.783 | 1.000 | 0.556 | `[[10, 8], [0, 23]]` | 0.281 | 0.155 |
| Class weights, no augmentation, random init | **0.818** | **0.957** | **0.667** | `[[12, 6], [1, 22]]` | 0.241 | 0.160 |
| Class weights + augmentation, random init | 0.305 | 0.000 | 1.000 | `[[18, 0], [23, 0]]` | 0.495 | 0.111 |
| Class weights + augmentation, frozen pretrained backbone | 0.606 | 0.913 | 0.333 | `[[6, 12], [2, 21]]` | 0.448 | 0.069 |

The class-weighting ablation is the best measured configuration: it improves
macro-F1 from 0.783 to 0.818 while keeping malignant recall at 0.957 and
raising benign specificity from 0.556 to 0.667. Augmentation caused model
collapse on this small development split, and the frozen-pretrained backbone
did not recover benign specificity. These are development measurements only;
no threshold was selected from them and no final test set was used.

## 2026-11-01 source and model-state review

The latest source review and dataset disposition are in
[`docs/DATASET_REGISTER.md`](docs/DATASET_REGISTER.md). Primary-source checks:

- The [Mendeley dataset record](https://data.mendeley.com/datasets/wmy84gzngw/1)
  and [DataCite rights metadata](https://api.datacite.org/dois/10.17632/wmy84gzngw.1)
  declare CC BY 4.0 and describe 250 breast-ultrasound images (100 benign,
  150 malignant). The local copy lacks patient identifiers and does not
  document a normal class.
- The [BUSI paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6906728/)
  reports 780 images from 600 women, but no usable image-to-patient mapping
  or verified license for the circulated image copy was found. The 260 images
  under the root `data\raw` class directories have no attached source/license
  manifest and remain excluded as provenance-unverified.
- The [Zenodo fetal dataset record](https://zenodo.org/records/8265464) and
  its [included README](https://zenodo.org/records/8265464/files/Readme.txt?download=1)
  state CC BY 4.0, 3,832 fetal-head ultrasound images, no predefined splits,
  and require attribution to original image author(s). The local archive
  contains 999 decoded images; authorship mapping and patient IDs are absent.
  It is inventory-only, not admitted to model data.
- The [MedMNIST+ record](https://zenodo.org/records/10519652) and official
  [MedMNIST metadata](https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py)
  state CC BY 4.0 for the derivative archive and identify ChestX-ray14 as the
  source (112,120 frontal radiographs, 30,805 patients, 14 text-mined labels,
  official source split). The local 64x64 archive has 78,468/11,219/22,433
  train/validation/test arrays but no patient or image IDs; the NIH source
  distribution terms were not independently verified in this review. It is
  a medical-negative candidate only, not partitioned or used.
- Wikimedia Commons provides item-level licensing/attribution information
  but explicitly cautions that uploader-supplied rights are not warranted;
  Open Images provides image-level labels and URL-based assets. No natural
  photo/object assets were acquired, and neither source is treated as
  globally licensed without per-item review. No permitted and traceable
  source was verified for screenshots or real low-quality scans.

The root data inventory contains the official Mendeley 250-image set, the
fetal candidate archive/inventory (999 images), and ChestMNIST 64 NPZ. There
are also 260 images in the root `benign` / `malignant` / `normal` folders
(100/100/60); their origin is not recorded, so they are excluded. No new data
were downloaded or incorporated in this review.

The effective backend configuration was checked via the application’s
`model_availability()` and reports:
`classification_enabled_by_server=false`,
`inference_enabled=false`, classifier artifact/evaluation approval absent,
and eligibility artifact/evaluation approval absent. No flag, deployment,
or model configuration was changed.

**No eligibility gate training, threshold selection, classifier test
evaluation, or new model-performance evaluation was performed.** The current
data cannot support leakage-safe patient/source partitions across required
categories; patient/source identifiers or verifiable patient-disjoint
source splits, broader negative categories, and a fresh independent cancer
test cohort remain blockers. Public classification remains disabled.

## 2026-11-01 additional dataset-candidate review

The expanded source register is in
[`docs/DATASET_REGISTER.md`](docs/DATASET_REGISTER.md). The review checked
primary dataset records, source papers, and applicable terms for MIMIC-CXR,
CheXpert, CAMUS, EchoNet-Dynamic, Open Images V6, and Wikimedia Commons, in
addition to the previously reviewed BUSI, Mendeley, fetal-ultrasound, and
ChestMNIST sources.

Key dispositions:

- **Eligible for acquisition, with narrow limits:** Mendeley Breast Ultrasound
  Image v1 declares CC BY 4.0 and describes 250 breast-ultrasound images
  (100 benign, 150 malignant); MedMNIST+ / ChestMNIST declares CC BY 4.0 for
  its derivative 64x64 archive; individually selected Wikimedia Commons
  files can be used only after per-file license/source/attribution review.
  Mendeley has no documented patient IDs; ChestMNIST lacks IDs in the local
  archive and cannot be patient-leakage audited. None is a fresh independent
  breast-ultrasound test cohort.
- **Needs permission or clarification:** fetal-head ultrasound has a CC BY
  4.0 record but its README requests credit to original image authors not
  mapped to files; MIMIC-CXR is controlled-access and requires its current
  access and use terms to be satisfied; CheXpert terms and usable
  patient-to-file mapping were not established by the reviewed official
  page; CAMUS links a 500-patient echocardiography collection but its
  applicable license, access terms, and released patient mapping were not
  confirmed; Open Images needs per-image provenance/license/attribution
  verification; and no rights-cleared screenshot or low-quality breast-scan
  cohort was found.
- **Excluded for the proposed use unless terms change:** BUSI’s circulated
  copy has no verified image-data license, and the 260 local class-folder
  images have no source manifest. EchoNet-Dynamic’s posted agreement limits
  use to personal non-commercial research and prohibits derivative works
  and transfer/publication; written authorization would be required before
  considering it for a trained eligibility model.

The source record facts, release/version information, known identifiers,
license limits, and per-candidate evidence links are recorded in the
register. No dataset was downloaded or incorporated during this review.
Existing local files are not newly cleared by this research: the 260
unattributed class-folder images remain excluded; the fetal archive remains
inventory-only; ChestMNIST remains only a derivative medical-negative
candidate; and no natural-photo/object/screenshot/low-quality examples have
been admitted to a gate dataset.

**Acquisition stop decision:** available data do not establish both (a)
permitted, traceable diverse positives and negatives and (b) patient/source-
safe, duplicate-checked partitions with a fresh untouched final test cohort.
Secure a rights-cleared breast-ultrasound cohort with patient mapping and
site/device metadata; obtain separate lawful negative sources for each
required category; resolve access and attribution terms; and record
patient/source IDs and per-image provenance before preparing any split. Real
low-quality scans require documented quality labels from an appropriate
source. The previously inspected 50-image classifier test set is not fresh
and must not be used for a later final claim.

No eligibility gate training, classifier evaluation, threshold tuning, or
model-performance measurement was performed. The runtime server-side
configuration remains `BREASTCARE_CLASSIFICATION_ENABLED=false` by default;
no flag, inference configuration, or deployment was changed. Public
classification remains disabled.

## 2026-11-01 limited acquisition and inventory

After the source review above, only the two candidates already marked
eligible for limited acquisition were rechecked against current authoritative
records and downloaded: Mendeley Breast Ultrasound Image v1 (Mendeley and
DataCite, CC BY 4.0) and MedMNIST+ ChestMNIST 64 (Zenodo record 10519652,
version 3.0, CC BY 4.0). Their original archives are stored unchanged under
`data\research_data\limited_acquisition_2026-11-01\`, outside application
assets and model/inference paths. Details, source URLs, licenses, and
attribution requirements are in
[`docs/LIMITED_DATA_ACQUISITION_2026-11-01.md`](docs/LIMITED_DATA_ACQUISITION_2026-11-01.md);
machine-readable inventories are adjacent to the local archives.

- **Mendeley:** `us-dataset.zip`, 3,264,338 bytes, SHA-256
  `87dbd3aaa89e2d8886839983be3a985b7b190d5d8cb167d95a8dcd5d224b63f7`.
  This fresh download matches the previously verified official archive.
  Safe-path checks and ZIP CRC validation passed before extraction. Inventory:
  250/250 decoded BMP files; 100 benign and 150 malignant; 46 distinct
  dimensions (width 75–199, height 57–161 px). Source file paths are
  preserved, but no patient-to-file mapping, separate source accession ID,
  or site/device metadata is supplied.
- **ChestMNIST:** `chestmnist_64.npz`, 401,604,127 bytes, SHA-256
  `e28fbed21400c50040d6a4e51f379dce89bcb24819d3878c252ca78d52d92c3e`.
  Its size and MD5 `9de6cd0b934ebb5b7426cfba5efbae16` match the current
  Zenodo record. Inventory validated all six arrays and loaded 112,120
  grayscale 64×64 `uint8` image tensors with 14-column labels (78,468 train,
  11,219 val, 22,433 test). The splits are those supplied in the source;
  no project partitions were created. The reduced archive has no patient,
  source-image, site, or device mapping. This is chest X-ray data, not breast
  ultrasound.

Both sources are limited development/inventory only and neither supports
patient-independent evaluation with the identifiers available. ChestMNIST
does not fill the missing breast-ultrasound positive cohort. No other
candidate was accessed or downloaded.

No training, project partition creation, threshold selection, classifier or
eligibility-gate evaluation, model-performance measurement, deployment, or
configuration change occurred. The backend still reports
`classification_enabled_by_server=false` and `inference_enabled=false`;
public classification remains disabled.

## 2026-11-01 follow-up source verification

Primary records and authoritative metadata were rechecked without downloading
data, contacting data owners, or applying for access. Details, source links,
versions, terms, identifiers, dispositions, and the staged acquisition plan
are recorded in
[`docs/DATASET_REGISTER.md`](docs/DATASET_REGISTER.md#follow-up-source-verification-and-acquisition-plan-2026-11-01).

- **BUS-BRA is the strongest breast-ultrasound candidate found, not yet
  approved for acquisition or evaluation.** Zenodo v1.0 reports 1,875 images
  from 1,064 female patients and declares CC BY 4.0. The paper reports four
  scanners at one institution, 722 benign and 342 malignant biopsy-proven
  cases, and supplied 5-/10-fold partitions. The paper also describes use as
  “for academic and research purposes.” The record’s license/copyright
  statement and the paper’s use wording require reconciliation. Patient- and
  scanner-to-image mapping and fold grouping were not inspected, so patient-
  disjointness is unverified.
- **BUS-UCLM** DataCite metadata reports 683 images from 38 patients
  (174 benign, 90 malignant, 419 normal), one Siemens scanner, and CC BY-NC
  3.0. Its Mendeley landing page returned “Dataset Not Found” on
  2026-11-01; no usable patient-to-image map was confirmed. **BUS_UC** reports
  CC BY 4.0 but states the images originated at ultrasoundcases.info; the
  upstream image rights and patient identifiers remain unverified.
- **Negative-source paths remain conditional.** MIMIC-CXR v2.1.0 documents
  chest radiographs with subject/study/image identifiers but requires
  controlled access and is single-institution. CAMUS documents 500-patient
  echocardiography but its linked download terms and released patient map
  remain unverified. Open Images V6 can support a per-asset curated
  natural/object set only after checking each image’s source and rights.
  Fetal-head ultrasound remains blocked by missing patient mapping and
  unmapped original-author attribution. No suitable verified source was
  found for real screenshots or low-quality/corrupted breast scans.
- **No data-readiness milestone was claimed.** No new data were acquired;
  no partitions, model training, threshold tuning, gate evaluation,
  classifier evaluation, or clinical performance analysis was performed.
  The previously acquired Mendeley and ChestMNIST archives remain limited
  development/inventory only and cannot establish patient-independent
  performance.

The minimum acquisition plan is a rights-cleared, patient-linked
breast-ultrasound development cohort; a separate fresh patient-level
breast-ultrasound test cohort with pathology labels; and traceable, lawfully
usable real negatives for each required category. Patient/source IDs,
source-label definitions, rights and attribution, duplicate checks, and
site/device metadata where available must be verified before creating
grouped train/validation/final-test partitions. Public classification remains
disabled; no configuration or deployment change was made.

## 2026-11-01 clarification request planning

Created the unsent source-specific questions, six separate draft
communications, and an institutional collection checklist in
[`docs/DATASET_CLARIFICATION_PACKAGE.md`](docs/DATASET_CLARIFICATION_PACKAGE.md).
The questionnaire covers each source’s research/model-development rights,
paper-versus-record terms, redistribution and derivative-model/demo/
commercial restrictions, patient/image IDs, patient-disjoint split evidence,
repeat/duplicate overlap, site/device metadata, label standards and limits,
attribution, and access/ethics/institutional approvals.

Specific unresolved items remain unchanged: BUS-BRA’s CC BY 4.0 versus
“academic and research purposes” wording and unverified CSV grouping; BUS-
UCLM’s unavailable landing page, non-commercial terms, and unverified
patient mapping; BUS_UC’s upstream image rights and absent patient IDs;
MIMIC-CXR’s controlled-access requirements and intended-use approval; CAMUS
download terms and patient mapping; and fetal-ultrasound per-image
attribution and patient grouping. The checklist addresses authorized
collection, consent/waiver review, de-identification, provenance, category
and quality labels, patient/source IDs, duplicate grouping, and secure
handling for genuine screenshots and naturally poor/corrupted scans. It is
explicitly planning guidance, not ethics or legal approval.

No messages or applications were sent; no data were acquired; no model,
inference, or deployment behavior was changed. Effective server-side
classification remains disabled.

Response status and evidence requirements for BUS-BRA, BUS-UCLM, BUS_UC,
MIMIC-CXR, CAMUS, and fetal-head ultrasound are tracked in
[`docs/DATASET_CLARIFICATION_TRACKER.md`](docs/DATASET_CLARIFICATION_TRACKER.md).
All responses and decisions are initially Pending. The tracker keeps legal
permission separate from limited-development and patient/source-independent
evaluation suitability; no source disposition is changed without recorded
supporting evidence.

## Acquired-data duplicate audit (2026-11-01)

The Mendeley and ChestMNIST archives were reverified against their recorded
SHA-256 hashes and inventories. Exact-byte matching found no duplicate
Mendeley files among 250 images. Reproduction from the unchanged ChestMNIST
archive confirmed 25 exact-pixel groups / 27 pairs (24 two-image groups and
one three-image group), with 26 repeated sample occurrences. Comparing the
source 14-label vectors found 14 groups with identical labels and 11 groups
with conflicting labels.

There are 112,120 source images and 112,094 unique pixel arrays after
global exact deduplication. Exact unique counts within the published splits
are train 78,452, validation 11,214, and test 22,431; three further
cross-split equivalences reduce the sum of those within-split counts to the
global unique count. The exact cross-split pairs are train–validation twice
and validation–test once; there is no train–test exact pair. This is
source-split image overlap, not proof of shared patients or patient leakage.
No patient-to-image identifiers are available to establish either.

pHash accounting was independently checked: Mendeley has 44 non-exact
same-hash image pairs and 142 nearby-code image pairs (111 unique hash-code
pairs); ChestMNIST has 26 and 1,732 image pairs, respectively (1,717 unique
hash-code pairs). Every reported nearby-code distance was recomputed and all
were 2. Contact sheets visually cover all 186 Mendeley candidates, all
ChestMNIST exact groups and 26 same-hash candidates, plus a deterministic
45-pair sample of 1,732 nearby-code pairs. These similarity candidates do
not identify repeat exams or patients. Details, per-group labels, split
counts, sampling method, and machine outputs are in
[`docs/DUPLICATE_IMAGE_AUDIT_2026-11-01.md`](docs/DUPLICATE_IMAGE_AUDIT_2026-11-01.md).

ChestMNIST is chest X-ray data and cannot validate a breast-ultrasound
eligibility gate. Both archives remain limited development/inventory only.
No training, partitioning, model evaluation, or configuration change was
performed. `classification_enabled_by_server=false` and
`inference_enabled=false` remain required.

## New cohort and negative-source review (2026-11-02)

The primary-source review is in
[`docs/COHORT_SOURCE_REVIEW_2026-11-02.md`](docs/COHORT_SOURCE_REVIEW_2026-11-02.md).
It adds new evidence for BUS-UCLM v3: the 2025 paper documents stable
pseudonymous patient grouping by filename, five-fold patient-level
partitions in a comparative experiment, biopsy-verified malignant labels,
and 38 patients from one hospital/system. The current v3 record states CC BY
4.0 for the dataset, distinct from the article's CC BY-NC-ND license, but
the public Mendeley page showed no files and the author-linked archive URL
returned HTTP 403 to HEAD and a one-byte range request. No archive was
downloaded. This cohort may support small patient-grouped development if
authorized files become available, but not a standalone fresh final test.

BUS-CoT reports a larger, multi-source breast-ultrasound dataset and an
aggregate count of 4,838 patients, but the inspected paper does not establish
stable patient-to-image identifiers. Its source mix, per-image permissions,
source overlap, label/count consistency, and reported split unit remain
unverified. Other new leads (ALN-Ultra, BUS-Large, the Taichung
breast/lymph-node release, and FOCUS-7D) remain blocked by access, provenance,
rights, or identifier gaps. MURA lacks verified data-use terms; no new
negative-image source is approved. Open Images retains its existing
per-asset review status. No source is eligible for patient-independent
evaluation based on this review.

No data were acquired; no archive hashes or inventories were produced for
these candidates. No training, partitioning, model/gate evaluation,
threshold tuning, deployment, or configuration change occurred. Effective
serving flags remain `classification_enabled_by_server=false` and
`inference_enabled=false`.
