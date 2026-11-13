# BreastCare AI research prototype

BreastCare AI is an experimental breast-ultrasound research website with a
Next.js frontend, FastAPI service, and reproducible PyTorch data/training/
evaluation pipeline.

> **Safety status: public classification is disabled.** The current build
> cannot reliably verify breast-ultrasound modality, and no new independently
> evaluated model is available. The Analyze page returns **Unable to assess**
> and never displays Benign/Malignant classifications. This software is not a
> medical device, has no clinical validation, and must not be used for
> diagnosis, screening, triage, or treatment decisions.

Latest research (4 November 2026): following the [experiment design audit](docs/EXPERIMENT_DESIGN_AUDIT_V1.md)
and unified dataset manifest (`data/prepared/unified_dataset_manifest_v1.csv`, 2,381 acquired
images -> 2,232 eligible in 1,298 global duplicate/case groups, 149 excluded), we completed
and verified on local CPU both the [**full-eligible frozen linear baseline (`full_source_baseline_v1`)**](docs/FULL_SOURCE_BASELINE_RESULTS_V1.md)
(held-out group ROC-AUC / BA: Mendeley **0.706 / 0.573**, TCIA **0.688 / 0.644**, BUS-BRA **0.594 / 0.570**)
and [**full-eligible neural-head adaptation on frozen features (`full_source_finetune_v1`)**](docs/FULL_SOURCE_FINETUNE_RESULTS_V1.md)
(264,963-parameter `ResidualAdapterHead` on cached detached `ResNet-18` canonical and ultrasound-augmented
features with frozen convolutional trunk; held-out group ROC-AUC / BA: Mendeley **0.448 / 0.465**,
TCIA **0.671 / 0.562**, BUS-BRA **0.573 / 0.518** — performing worse externally than the linear baseline;
source-specific overfitting is an unproved hypothesis, not an established causal explanation).
We then completed a [**global duplicate-group integrity audit**](docs/GROUPING_INTEGRITY_AUDIT_V1.md)
(`data/prepared/unified_dataset_manifest_v2.csv`: 2,215 strictly clean eligible images across 1,288
validated groups after excluding 17 Mendeley partial-sequence images linked to historical `mendeley_v1`
test frames and reuniting 2 split five-image sequences) and packaged the separately versioned,
resumable **GPU convolutional encoder fine-tuning experiment** ([`docs/GPU_ENCODER_FINETUNE_PROTOCOL_V3.md`](docs/GPU_ENCODER_FINETUNE_PROTOCOL_V3.md),
`ml/gpu_encoder_finetune_v2.py`, `notebooks/colab_full_source_finetune.ipynb`; `8,389,634` trainable
`layer4` conv + `fc` parameters with all 20 `BatchNorm2d` layers frozen), which has **not** run locally
because this Windows machine has no CUDA GPU (`torch.cuda.is_available() == False`).
For one-command Windows 11 verification or startup, run `.\start_breastcare.ps1 -Mode check` or `.\start_breastcare.ps1`.
Earlier pilot: the [cross-source audit and measured experiments](docs/CROSS_SOURCE_RESULTS_V1.md)
found inconsistent transfer despite correctly loaded, noncollapsed encoder
features. TCIA and BUS-BRA are now inspected development sources. No candidate
is promoted, no fresh qualified final cohort is reserved, and classification
remains disabled. The [predefined protocol](docs/CROSS_SOURCE_PROTOCOL_V1.md)
and local results bundle preserve the six bounded experiments.

## What is included

- **Website:** Home, Analyze, Model & Evidence, How It Works, Privacy, and About.
- **API:** `GET /health`, `GET /model-info`, `POST /analyze`; bounded PNG/JPEG
  validation and structured abstentions.
- **Research ML pipeline:** class-folder scanning, patient/duplicate grouping,
  reproducible split manifest, training-only augmentation, class-weighted
  ResNet-18 training, validation checkpoint selection and temperature fitting,
  plus final test metrics and bootstrap intervals.
- **Evidence:** dataset register, model card, and historical reliability audit.

The API does not persist uploads or transmit images to a third-party AI
provider. The multipart framework may temporarily spool larger request bodies
to OS temporary storage and closes the upload after processing. Hosting and
proxy logs are controlled separately and must be reviewed before deployment.

## Windows 11 setup

Prerequisites: Python 3.12 or 3.13, Node.js 20.9+, npm, and Git.

In PowerShell at the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Start the API in this terminal:

```powershell
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

In a second PowerShell terminal:

```powershell
Set-Location D:\breast_cancer\web
npm ci
Copy-Item .env.example .env.local
npm run dev
```

Open `http://localhost:3000`. The API docs are at `http://127.0.0.1:8000/docs`.
Uploads return Unable to assess until the safety gates and model are supported;
this is expected behavior, not a setup failure.

To check the source:

```powershell
Set-Location D:\breast_cancer
python -m pytest tests
Set-Location D:\breast_cancer\web
npm run typecheck
npm run lint
npm run build
```

### Docker and Container Deployment

To run both API and Frontend in isolated production containers using Docker Compose:

`ash
docker compose up --build -d
`

- Web UI: http://localhost:3000
- API Health: http://localhost:8000/health
- See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for complete container configurations, environment variables, health checks, and the safest free hosting analysis.

### Configuration

`BREASTCARE_MAX_UPLOAD_BYTES`, `BREASTCARE_MAX_IMAGE_PIXELS`,
`BREASTCARE_MIN_IMAGE_SIDE`, `BREASTCARE_CORS_ORIGINS`,
`BREASTCARE_MODEL_PATH`, `BREASTCARE_CLASSIFIER_EVALUATION_PATH`,
`BREASTCARE_ELIGIBILITY_MODEL_PATH`,
`BREASTCARE_ELIGIBILITY_EVALUATION_PATH`, and
`BREASTCARE_CLASSIFICATION_ENABLED` are read by the API. Classification is
server-controlled and defaults to `false`; there is no browser request field
that can override it. Even with the flag enabled, inference remains blocked
unless both exact-checkpoint evaluation records are present and explicitly
reviewed. File and quality checks do not prove modality. A gate result is only
“ultrasound-like,” not verification of breast anatomy or diagnostic
suitability.

Set `NEXT_PUBLIC_API_URL` in `web\.env.local` to the API's browser-reachable
origin when it is not `http://127.0.0.1:8000`. Configure CORS to exactly match
the deployed website origin.

## Dataset provenance and access

Read [`docs/DATASET_REGISTER.md`](docs/DATASET_REGISTER.md) before obtaining or
using any images. It distinguishes the authors' BUSI paper from a third-party
Kaggle mirror and documents a separate Mendeley 250-image, two-class dataset.
The source papers report dataset details, but license and provenance status
must be checked independently.

- BUSI's commonly circulated mirror claims CC0, but that claim has not been
  independently verified as authorization for the original patient images.
  Do not redistribute or train with it for a public/commercial release without
  rights confirmation.
- Mendeley version 1 has a DataCite CC BY 4.0 rights record, and documents 100
  benign / 150 malignant images. It does not document Normal images or patient
  IDs. It cannot train the ultrasound eligibility gate.
- Zenodo record 8265464 documents 3,832 fetal-head ultrasound images and CC BY
  4.0 terms. Its original-image archive was downloaded and validated against
  the published size and MD5. It is an `other_ultrasound` challenge source,
  with 999 images inventoried. It is not enough to train/validate the gate;
  see provenance and split limitations in
  [`docs/DATASET_REGISTER.md`](docs/DATASET_REGISTER.md).
- The official MedMNIST+ ChestMNIST 64 archive was downloaded and its
  published checksum verified. Its 78,468/11,219/22,433 train/validation/test
  images are 64x64 chest radiographs with 14 multi-label targets. It is only
  an `other_medical` candidate; the reduced archive has no patient IDs, and
  source-split leakage plus NIH source-use terms are not verified for this
  project. It is not admitted to gate partitions.
- The 260 images in top-level `data\raw\benign`, `malignant`, and `normal`
  folders have no recorded provenance/license manifest and are excluded.
- No rights-reviewed natural-photo/object, screenshot, or real low-quality
  image set has been acquired. Fetal ultrasound and ChestMNIST remain
  inventory/candidate sources only; no eligibility gate training or evaluation
  is supported. See the latest review in
  [`docs/DATASET_REGISTER.md`](docs/DATASET_REGISTER.md).
- The official version 1 archive was downloaded and ZIP-validated in this
  workspace. Its exact hash, the image-level split, and limitations are in
  [`docs/DATASET_REGISTER.md`](docs/DATASET_REGISTER.md). The prepared manifest
  is local and ignored by Git.
- No dataset is downloaded automatically by the application or training
  scripts, and missing real data is never replaced with synthetic samples.
  Keep authorized image files outside Git under `data\raw\`.

For BUSI, download the source archive manually from its
[Kaggle record](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
only after checking the current terms and your authorization. For Mendeley,
download version 1 from
[Mendeley Data](https://data.mendeley.com/datasets/wmy84gzngw/1) and retain its
citation/license details. Organize images in class folders:

```text
data\raw\busi\
├── benign\
├── malignant\
└── normal\
```

For the Mendeley binary experiment, the validated archive is
`data\raw\mendeley_v1\us-dataset-official.zip` and was extracted into
`data\raw\mendeley_v1\official-v1\`. Its archive nests images under class
directories in `originals\`; keep the benign and malignant folders intact.
The pipeline recursively reads BMP, PNG, and JPEG files and records only these
two supported classes. Do not use the older incomplete `us-dataset.zip`.

The splitter ignores image files with `_mask` in their name. If patient
identifiers are legally available, create `data\patient_map.csv` with columns
`image_path,patient_id`, using paths relative to `data\raw\busi`. Never infer
patient IDs.

The verified fetal-ultrasound archive can be inventoried separately (it is
never added to the cancer-classifier training set):

```powershell
python -m ml.prepare_fetal_negative --archive "data\raw\fetal_head_zenodo\Diverse Fetal Head Images-original-image.full.zip" --output data\prepared\fetal_candidate_v1
```

Its source inventory records duplicate groups and hashes but deliberately
leaves split assignments unassigned and `rights_verified=false` until curator
review confirms provenance/attribution. It is not accepted by the eligibility
trainer as-is. Do not mix it into gate training without the remaining negative
categories and independent patient/source-aware partitions.

Verify and inventory the ChestMNIST challenge archive without extracting it:

```powershell
python -m ml.inspect_chestmnist --archive data\raw\medmnist\chestmnist_64.npz --output data\prepared\medmnist_candidate_v1\inventory.json
```

This checks the published archive checksum, array shapes, and binary target
encoding, and records the source-provided split sizes and per-column label
counts. The report deliberately leaves rights approval false and does not
reassign source train/validation/test rows to eligibility-gate partitions.

## Reproducible ML experiments

Create a fixed, duplicate-grouped patient split when complete patient metadata
is available:

```powershell
python -m ml.prepare_dataset --source data\raw\busi --output data\prepared\busi --metadata-csv data\patient_map.csv --seed 42
```

Without patient IDs, the pipeline requires explicit opt-in and labels all
results as image-level:

```powershell
python -m ml.prepare_dataset --source data\raw\busi --output data\prepared\busi --allow-image-level-split --seed 42
```

The preparation step checks image decoding, SHA-256 duplicates, and
perceptual-hash near duplicates; joins those groups before making deterministic
stratified partitions; writes `split_manifest.csv` and a JSON report. It stops
if groups contain conflicting labels or there are too few independent groups.
Patient-level grouping does not establish external validation.

Train a versioned ResNet-18. ImageNet weights are downloaded by torchvision
when `--pretrained` is used (default); `--no-pretrained` works offline:

```powershell
python -m ml.train --source data\raw\busi --manifest data\prepared\busi\split_manifest.csv --output models_saved\versioned\breastcare_resnet18_v1.pth --epochs 25 --batch-size 16 --seed 42
```

Training uses only the training partition for augmentation and class weights.
It selects checkpoints on validation malignant recall, then macro-F1, then
validation loss; temperature scaling is fit on validation logits. Existing
output checkpoints are not overwritten unless `--overwrite` is explicitly
passed. The final test partition is not used for selection or threshold
tuning.

The previously inspected Mendeley test partition is retired and must not be
used for further tuning or evaluation. The new development manifest excludes
that partition and all duplicate groups linked to it at dHash distance <=5:

```powershell
python -m ml.prepare_mendeley_development --source data\raw\mendeley_v1\official-v1 --previous-manifest data\prepared\mendeley_v1\split_manifest.csv --output data\prepared\mendeley_dev_v2 --seed 42 --near-duplicate-distance 5
python -m ml.train --source data\raw\mendeley_v1\official-v1 --manifest data\prepared\mendeley_dev_v2\split_manifest.csv --output models_saved\versioned\mendeley_dev_v2_baseline.pth --epochs 25 --batch-size 16 --seed 42
python -m ml.evaluate_validation --source data\raw\mendeley_v1\official-v1 --manifest data\prepared\mendeley_dev_v2\split_manifest.csv --checkpoint models_saved\versioned\mendeley_dev_v2_baseline.pth --output reports\mendeley_binary_v1\validation_ablation\dev_v2_baseline.json
```

The generated split has no final test partition and no patient IDs. All
50 previously inspected test images and all dHash<=5-linked groups are
excluded; this leaves 53 benign / 68 malignant training images and 18 benign /
23 malignant validation images. All resulting metrics are duplicate-grouped,
image-level development metrics, not independent estimates. The validation
evaluator is restricted to validation rows and reports fixed
malignant-probability operating points. Class-weight,
augmentation, and backbone-freezing comparisons should use this same
development manifest; `--no-augmentation`, `--no-class-weights`, and
`--freeze-backbone` select the respective ablations. The validation set has
already been used for checkpoint selection/calibration, so it cannot serve as
a final test set.

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

Run the final internal evaluation once after selection:

```powershell
python -m ml.evaluate --source data\raw\busi --manifest data\prepared\busi\split_manifest.csv --checkpoint models_saved\versioned\breastcare_resnet18_v1.pth --output-dir reports\research_v1
```

The evaluator writes per-class precision/recall/F1, confusion matrix, ROC-AUC,
PR-AUC, malignant sensitivity/specificity at a predeclared calibrated
probability threshold of 0.5, Brier score, ECE, bootstrap intervals using
patient groups where supplied (otherwise duplicate groups/images), ROC and
reliability plots, and per-image predictions. These measurements are not
external or clinical validation. No result should be described as
patient-independent when the manifest says `image_level_internal_holdout`.

### Mendeley binary experiment

The Mendeley record does not document patient identifiers, so this experiment
must use the explicit image-level opt-in. It cannot produce a Normal class or
validate ultrasound eligibility:

```powershell
python -m ml.prepare_dataset --source data\raw\mendeley_v1\official-v1 --output data\prepared\mendeley_v1 --allow-image-level-split --seed 42
python -m ml.train --source data\raw\mendeley_v1\official-v1 --manifest data\prepared\mendeley_v1\split_manifest.csv --output models_saved\versioned\mendeley_binary_resnet18_v1.pth --epochs 25 --batch-size 16 --seed 42
python -m ml.evaluate --source data\raw\mendeley_v1\official-v1 --manifest data\prepared\mendeley_v1\split_manifest.csv --checkpoint models_saved\versioned\mendeley_binary_resnet18_v1.pth --output-dir reports\mendeley_binary_v1
```

Treat its held-out metrics as image-level research results only. The public
website remains disabled even after this experiment.

The current trained model's held-out internal results are recorded in
`reports\mendeley_binary_v1\test_metrics.json`: accuracy 0.76, macro-F1
0.7024, malignant sensitivity 1.00, and benign specificity 0.40 at the
predeclared 0.5 threshold. Its confusion matrix is `[[8, 12], [0, 30]]`
(actual benign/malignant by predicted benign/malignant). The set has 50
images and no patient IDs. The later TCIA external evaluation is reported in
the verification section below. In this internal set, 12 of 20
benign samples were called malignant; this model must not be presented as a
reliable cancer classifier.

### Ultrasound-like eligibility gate

Eligibility training binds each new checkpoint to the exact manifest bytes
using SHA-256. Evaluation rejects an edited manifest or an older, unbound
checkpoint, even if its filename is unchanged. Training and evaluation also
check for manifest changes before saving their artifacts. Keep the prepared
manifest unchanged; legacy eligibility checkpoints require a new training run
to establish this binding. This integrity check does not establish dataset
rights, patient independence, or model performance.

No eligible positive/negative gate dataset exists in this workspace. The
Mendeley and BUSI sources contain breast ultrasound, not representative
non-ultrasound negatives. The preparation step requires `other_ultrasound`,
`other_medical`, `natural_image`, `object`, `screenshot`, and `low_quality`
categories in every split. Do not train a gate using generated drawings,
grayscale/texture rules, or the cancer model's confidence. The following
reproducible pipeline is ready for a properly licensed, curator-reviewed
dataset. It fails unless every row includes an explicit rights attestation.

Prepare a CSV with these columns:

```text
image_path,label,category,patient_id,license,source_url,rights_verified
```

`label` is `breast_ultrasound` or `negative`; positive rows must use category
`breast_ultrasound`. Negative categories are required for other ultrasound,
other medical images, natural images, objects/books, screenshots, and
low-quality scans; corrupted files are rejected before the gate. `patient_id`
may be blank only when the dataset does not provide one; the preparation
report then clearly marks its evaluation as
duplicate-grouped image-level. `rights_verified=true` is a curator attestation,
not independent legal verification. Never fill these fields from assumptions.

```powershell
python -m ml.prepare_eligibility --source data\raw\eligibility --input-csv data\raw\eligibility.csv --output data\prepared\eligibility --seed 42
python -m ml.train_eligibility --source data\raw\eligibility --manifest data\prepared\eligibility\eligibility_manifest.csv --output models_saved\versioned\eligibility_v1.pth --epochs 25 --batch-size 16 --seed 42
python -m ml.evaluate_eligibility --source data\raw\eligibility --manifest data\prepared\eligibility\eligibility_manifest.csv --checkpoint models_saved\versioned\eligibility_v1.pth --output models_saved\versioned\eligibility_v1.evaluation.json --max-invalid-acceptance $MaxInvalidAcceptance
```

The threshold is selected on a separate `threshold` partition using the
predeclared `--max-invalid-acceptance`; the locked `test` partition is used
only afterward. Define `$MaxInvalidAcceptance` from a preregistered research
protocol before inspecting threshold/test results; no target or image-score
threshold is selected in this repository. Reports include valid-image
acceptance, false acceptance by negative category, sample/group counts, and
group-bootstrap 95% intervals.
The evaluation output always sets `research_inference_approved=false` and
`deployment_gate_pass=false`; tests cannot auto-enable the site. Small test
sets, missing patient IDs, unrepresented devices/sites, and unobserved negative
categories remain limitations. The gate can at most say “ultrasound-like,” not
confirm breast anatomy or diagnostic suitability.

Classifier enablement also requires a reviewed evaluation JSON bound to that
exact checkpoint hash. The current classifier report explicitly records weak
benign specificity and is not approved. Even after suitable data are obtained,
do not set `research_inference_approved=true` in either evaluation record or
set `BREASTCARE_CLASSIFICATION_ENABLED=true` until the locked evidence and
independent review justify the research deployment. None of these settings is
controlled by the frontend.

The CPU-default inference utility is intentionally limited to an already
prepared, labeled research manifest; it cannot score arbitrary uploads and is
not connected to the public API:

```powershell
python -m ml.inference --source data\raw\busi --manifest data\prepared\busi\split_manifest.csv --checkpoint models_saved\versioned\breastcare_resnet18_v1.pth --split test --device cpu --acknowledge-research-only
```

Its calibrated probabilities are research estimates only and do not establish
modality, clinical uncertainty, or diagnostic certainty.

For a genuinely separate, license-compatible source with the same class set,
prepare that source in its own directory and pass `--external` to the evaluator.
The evaluator verifies that the source and manifest differ from the training
data and that the class mapping matches. A separate folder alone does not prove
independence; check provenance and overlap before calling it external.

Training a classifier does **not** enable website inference. A separate
ultrasound eligibility/OOD model needs diverse breast ultrasound, unrelated
photographs and other imaging modalities, patient/site-aware splits, and a
locked independent test set. Until those requirements are met, the service
continues to abstain.

### Google Colab with a T4

1. Use a Colab GPU runtime with a T4.
2. Clone the repository and install dependencies:

   ```python
   !git clone https://github.com/salahibrahiem12/breast-cancer-ai.git
   %cd breast-cancer-ai
   !pip install -r requirements.txt
   ```

3. Obtain the data manually under its verified terms. Do not put credentials in
   a notebook or commit image data. Upload or mount the authorized source at
   `/content/breast-cancer-ai/data/raw/busi`.
4. Run preparation, then use a larger batch if memory permits:

   ```python
   !python -m ml.prepare_dataset --source data/raw/busi --output data/prepared/busi --allow-image-level-split
   !python -m ml.train --source data/raw/busi --manifest data/prepared/busi/split_manifest.csv --output models_saved/versioned/breastcare_resnet18_v1.pth --epochs 25 --batch-size 32
   !python -m ml.evaluate --source data/raw/busi --manifest data/prepared/busi/split_manifest.csv --checkpoint models_saved/versioned/breastcare_resnet18_v1.pth --output-dir reports/research_v1
   ```

5. The supplied BUSI distribution does not provide usable patient IDs; this
   run is explicitly image-level. Upload the generated artifacts to private,
   access-controlled storage; do not treat Colab as an independent test source.

## API contract

- `GET /health` reports service health and inference status.
- `GET /model-info` reports artifact availability, supported modality, class
  labels, server-side classification flag, and limitations.
- `POST /analyze` accepts one multipart `file`. Responses explicitly report
  unsupported/unsuitable, unable-to-assess, model-unavailable, classification-
  disabled, or research-classification states. The server validates file
  signature, decoding, size, dimensions, and basic quality before any model.
  Unsupported/rejected images never receive cancer probabilities or
  explanations. An accepted gate result is only ultrasound-like.

No uploaded image or filename is logged or retained by application code.
Application logs do not include image bytes or patient metadata. No
authentication, rate limiting, or hosting-provider privacy controls are
claimed; configure these before exposing a deployment publicly.

## Existing Streamlit prototype

The previous `app.py` remains available for compatibility and also fails
closed. It uses the legacy model only in code paths blocked by the current
eligibility gate. The new website and API do not silently load legacy weights.
See [`RELIABILITY_AUDIT.md`](RELIABILITY_AUDIT.md) for historical results and
the original root-cause analysis.

## Verification and limitations

The [locked BUS-BRA offline evaluation](docs/BUSBRA_LOCKED_EVALUATION_REPORT_V1.md)
completed once on 4 November 2026 after protocol/model hashes were frozen.
The selected frozen encoder failed external source-Case discrimination:
balanced accuracy 0.4959, ROC-AUC 0.4704, specificity 0.0153. No threshold or
model retry occurred. Grouping semantics remain unresolved; these are not
patient-independent results. Public classification remains disabled.

The subsequent [bounded three-model comparison](docs/BOUNDED_DEVELOPMENT_RESULTS_V1.md)
completed on identical development folds. A frozen pretrained encoder ranked
first on the predefined metric, but these are exploratory image-level results.
BUS-BRA was acquired as a prediction-free reserved source; patient/exam
group semantics remain to be qualified. No new model was evaluated on TCIA,
promoted to serving, or described as clinically ready.

The [TCIA pipeline audit and separate development comparison](docs/TCIA_PIPELINE_AUDIT_2026-11-03.md)
records complete binary metrics and patient-group intervals, exact checkpoint
replay, privacy-preserving error review, and a negative preprocessing
experiment on separate development data. TCIA is an inspected evaluation set;
no fresh independent final cohort is available. See the
[cohort usage ledger](docs/COHORT_USAGE_LEDGER.md). Classification stays disabled.

Current continuation status and remaining data dependencies are recorded in
[`docs/PROGRESS.md`](docs/PROGRESS.md). The upload flow cancels outdated requests
when the selected image changes and reports a timeout after 60 seconds.
Run the frontend upload regression checks from `web` with `npm test`.

Run the test/build commands above. Software tests use generated fixtures only
to verify API contracts, validation behavior, label mappings, and data-split
invariants. They are **not** measured model performance or proof of general
non-ultrasound rejection.

The prepared Mendeley split, trained checkpoint, and test metrics are local
and ignored by Git. A subsequent fixed-model external evaluation on 197
biopsy-confirmed TCIA BrEaST cases measured **48.2% accuracy**, **0.463 ROC-AUC**,
and **6.1% benign specificity**: 93 false positives and nine false negatives.
These results do not support enabling classification. See the
[protocol](docs/TCIA_EXTERNAL_EVALUATION_PROTOCOL_2026-11-03.md) and
[source inventory and measured results](docs/TCIA_ACQUISITION_AND_EVALUATION_2026-11-03.md).
No licensed negative modality dataset, eligibility
checkpoint, or measured eligibility/rejection performance is available. Before
enabling classification, obtain patient-mapped and diverse modality data,
build and externally evaluate the gate, review errors with qualified experts,
and validate the complete pipeline.
