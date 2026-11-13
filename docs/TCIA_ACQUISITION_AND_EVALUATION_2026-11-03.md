# TCIA BrEaST acquisition and external research evaluation

Completed 2026-11-03 after the user authorized finding a trusted dataset.
This supersedes earlier statements that external-source classifier results
were unavailable. Public inference remains disabled.

## Authoritative source

[TCIA BREAST-LESIONS-USG](https://www.cancerimagingarchive.net/collection/breast-lesions-usg/),
version 1, updated 2024-01-08, DOI
[10.7937/9WKK-Q141](https://doi.org/10.7937/9WKK-Q141).
The collection declares **CC BY 4.0** for images and clinical metadata.
The [source paper](https://pubmed.ncbi.nlm.nih.gov/38297002/) documents public
research reuse and 256 scans from 256 patients. The checked
[TCIA policy](https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/)
requires dataset/repository acknowledgement and citation under the stated
collection license. Local snapshots accompany the acquisition record.

Credit Pawlowska, Cwierz-Pienkowska, Domalik, Jagus, Kasprzak, Matkowski,
Fura, Nowicki, and Zolek; cite the dataset and paper DOI
10.1038/s41597-024-02984-z, link
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), and indicate changes.
Original downloads are unchanged. Inventories, selection, manifests, and
reports are project-created derivatives. No raw data were added to Git or
served by the website. This official acquisition is distinct from the
previously unverified third-party repository; BUS-BRA questions are unchanged.

## Verified downloads and inventory

Local root: `data/research_data/tcia_breast_lesions_usg_2026-11-03/` (ignored).
Both downloads came from links on the official TCIA collection page.

| File | Bytes | Locally computed SHA-256 |
|---|---:|---|
| BrEaST-Lesions_USG-images_and_masks-Dec-15-2023.zip | 69,862,076 | `c32e49cdfa54042e065582d0cb35978b5fc5e5b5e461b33bf773fb50cfdad355` |
| BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx | 40,173 | `89a8874496a6f1f93960390b963f7369b90f7e72ee08a738f2d17e37f5eff8bf` |

ZIP CRC passed. Member checks cover traversal, absolute/drive paths,
backslashes, symlinks, duplicate members, and expansion limits. All 522 PNGs
decoded: **256 scans and 266 masks**. Masks were excluded from sample counts.
All primary images matched the workbook; case IDs and image names were unique.

Classes: **154 benign, 98 malignant, four normal**. Verification: **197 biopsy,
55 follow-up, four not applicable**. Derived inventory excludes age, symptoms,
and other clinical descriptors. Namespaced CaseIDs supply grouping based on
the publisher's one-scan-per-patient statement, not verified identities.

No repeated decoded RGB arrays occurred within the cohort. Against all 250
official local Mendeley images, there were zero byte/pixel matches and zero
dHash/pHash pairs within Hamming distance 3. These checks cannot prove absence
of all overlap or establish patient independence across sources.

## Locked evaluation and measured result

The [protocol](TCIA_EXTERNAL_EVALUATION_PROTOCOL_2026-11-03.md) was recorded
before predictions. All **197 biopsy-confirmed cases (99 benign, 98 malignant)**
were test-only. The 55 follow-up cases and four normals were excluded before
scoring. This is a biopsy-selected population with selection/spectrum limits.
The fixed Mendeley-trained model, preprocessing, validation-fitted temperature,
and 0.5 decision threshold were unchanged; no training or tuning used BrEaST.

Checkpoint SHA-256:
`69b0ef99a4d0c0f744ed1f809d12742250baa250c43969b85c0f9a5e720c6045`.
Manifest SHA-256:
`7cac20c3a84afcdabf9b348c08e07dfdc481f324395919707a2454fbf5109fe7`.

| Metric | Result |
|---|---:|
| Accuracy | 0.482234 |
| Macro-F1 | 0.370489 |
| ROC-AUC | 0.462997 |
| Macro average precision | 0.481764 |
| Malignant sensitivity at 0.5 | 89/98 = 0.908163 |
| Benign specificity at 0.5 | 6/99 = 0.060606 |
| False negatives / false positives | 9 / 93 |
| Multiclass Brier / top-label ECE, 10 bins | 0.767129 / 0.348925 |

Confusion matrix (actual rows, predicted columns; benign/malignant):
`[[6, 93], [9, 89]]`. Deterministic 1,000-iteration patient-group bootstrap
95% intervals: sensitivity 0.848468–0.960821; specificity 0.018692–0.108935;
macro-F1 0.318950–0.426678.

Metrics, predictions, errors, ROC and reliability plots are local in
`reports/tcia_breast_biopsy_external_v1/`. The CPU run used two threads.

## Decision and reproduction

These results do not support enabling classification. Serving flags remain
false. No clinician review, clinical validation, or gate evaluation is claimed.
This breast-only source supplies none of the six required negative gate
categories and has too few normals for normal-class evaluation. It is no
longer a fresh future final test cohort after this run. Further development
needs separate training/validation data and a new untouched external cohort.

Inventory: `python -m ml.inspect_tcia_breast --help`. Biopsy-only preparation:
`python -m ml.prepare_tcia_external --help`. Both refuse existing output
directories. Generated-fixture tests cover archive safety, metadata binding,
mask exclusion, test-only preparation and overlap.

```powershell
python -m ml.evaluate --source data/raw/tcia_breast_biopsy_external_v1 --manifest data/prepared/tcia_breast_biopsy_external_v1/split_manifest.csv --checkpoint models_saved/versioned/mendeley_binary_resnet18_v1.pth --output-dir reports/tcia_breast_biopsy_external_v1 --bootstrap-iterations 1000 --external
```

This documents the completed run. Do not use repeat runs or predictions to
select a threshold or a better-performing model.
