# Limited research-data acquisition and inventory

**Access date:** 2026-11-01
**Scope:** acquisition and file inventory only. These sources are for limited
development/inventory; no training, project splits, threshold selection,
classifier/gate evaluation, or performance claims were made.

The original archives are stored outside the application and model paths in
`data\research_data\limited_acquisition_2026-11-01\`. The `data/` tree is
gitignored. Archive bytes are retained unchanged; inventories are separate
sidecars next to each archive.

## Mendeley Breast Ultrasound Image v1

- **Authoritative record:** [Mendeley Data, version 1](https://data.mendeley.com/datasets/wmy84gzngw/1);
  DOI [10.17632/wmy84gzngw.1](https://doi.org/10.17632/wmy84gzngw.1).
- **Version/date:** version 1, issued 2017-12-31. Checked on 2026-11-01.
- **Modality/labels:** breast ultrasonography; source record reports 100 benign
  and 150 malignant images.
- **Terms:** the current [DataCite rights metadata](https://api.datacite.org/dois/10.17632%2Fwmy84gzngw.1)
  declares CC BY 4.0. Provide appropriate creator credit, link the license,
  indicate changes, and do not imply endorsement. The license does not
  establish patient consent or waive privacy, publicity, or other applicable
  rights.
- **Attribution:** Rodrigues, Paulo Sergio (2017), *Breast Ultrasound Image*,
  version 1, Mendeley Data, DOI:10.17632/wmy84gzngw.1. License:
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The archive was
  downloaded unchanged; the accompanying inventory is a project-created
  description, not a modified data archive.
- **Source file URL:**
  [Mendeley public archive](https://data.mendeley.com/public-files/datasets/wmy84gzngw/files/b63daee9-78de-4122-8475-9b3aa22ffd64/file_downloaded).
- **Local archive:** `Mendeley_Breast_Ultrasound_Image_v1\us-dataset.zip`
  (3,264,338 bytes; SHA-256
  `87dbd3aaa89e2d8886839983be3a985b7b190d5d8cb167d95a8dcd5d224b63f7`;
  locally computed MD5
  `64601c9b0247dba24d67a1de597b749e`). The SHA-256 matches the previously
  verified official archive. The current public source URL returned this
  same archive on the access date.
- **Extraction safety/integrity:** checked all 253 ZIP entries for absolute
  paths, traversal, backslash/drive paths, symlinks, duplicate member paths,
  and unreasonable expansion before extraction; no unsafe entries were
  found. ZIP CRC validation passed. The original archive remains unchanged.
- **Inventory:** 250 image files; 250 decoded successfully; all BMP. Source
  folder labels: benign 100, malignant 150. There are 46 distinct image
  dimensions; width range 75–199 px, height range 57–161 px. The detailed
  per-dimension and per-file inventory is in
  `Mendeley_Breast_Ultrasound_Image_v1\inventory.json`.
- **Missing identifiers:** source file paths/names are preserved in the
  inventory, but there is no patient-to-file mapping, separate source
  accession ID, or site/device metadata documented in the archive. There is
  no Normal class documented.
- **Use boundary:** limited development/inventory only. Without patient IDs,
  this source cannot support patient-independent evaluation. It is not a
  fresh independent final test cohort and does not provide negative
  out-of-distribution categories.

## MedMNIST+ ChestMNIST 64

- **Authoritative record:** [Zenodo record 10519652](https://zenodo.org/records/10519652);
  DOI [10.5281/zenodo.10519652](https://doi.org/10.5281/zenodo.10519652).
- **Version/date:** Zenodo version 3.0, published 2024-01-16; checked on
  2026-11-01.
- **Modality/labels:** 64×64 grayscale frontal chest radiographs, derived
  from NIH ChestX-ray14. The source defines 14 multi-label targets in this
  order: atelectasis, cardiomegaly, effusion, infiltration, mass, nodule,
  pneumonia, pneumothorax, consolidation, edema, emphysema, fibrosis,
  pleural, hernia. **This is chest radiography, not breast ultrasound.**
- **Terms:** the Zenodo record and official
  [MedMNIST metadata](https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py)
  declare CC BY 4.0 for the derivative dataset. Attribute MedMNIST, link the
  license, and indicate changes. These terms apply to the MedMNIST
  redistribution assessed here; they do not grant rights to separately
  download or redistribute the underlying NIH source archive.
- **Attribution:** Yang J, Shi R, Wei D, Liu Z, Zhao L, Ke B, Pfister H, Ni B.
  “MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical
  image classification.” *Scientific Data* 10, 41 (2023),
  [doi:10.1038/s41597-022-01721-8](https://doi.org/10.1038/s41597-022-01721-8);
  also cite the original ChestX-ray14 paper, Wang et al.,
  [arXiv:1705.02315](https://arxiv.org/abs/1705.02315). License:
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Source file URL:**
  [Zenodo ChestMNIST 64 archive](https://zenodo.org/records/10519652/files/chestmnist_64.npz?download=1).
- **Local archive:** `MedMNIST_Plus_ChestMNIST_v3.0\chestmnist_64.npz`
  (401,604,127 bytes; SHA-256
  `e28fbed21400c50040d6a4e51f379dce89bcb24819d3878c252ca78d52d92c3e`).
  The local MD5 is `9de6cd0b934ebb5b7426cfba5efbae16`, matching the current
  Zenodo record and official metadata.
- **Archive safety/integrity:** the NPZ is a ZIP container. Internal member
  paths were checked for absolute/traversal paths; the file was read without
  extracting it. Its published MD5 and byte size both match exactly.
- **Inventory:** one NPZ file containing six source arrays. Pixel arrays
  loaded and passed shape, dtype, finite-value, and label-shape checks:

  | Source-provided split | Images | Image tensor | Label tensor |
  |---|---:|---|---|
  | train | 78,468 | 78,468 × 64 × 64, uint8 | 78,468 × 14, uint8 |
  | val | 11,219 | 11,219 × 64 × 64, uint8 | 11,219 × 14, uint8 |
  | test | 22,433 | 22,433 × 64 × 64, uint8 | 22,433 × 14, uint8 |

  Total: 112,120 decoded pixel arrays. The “image format” is grayscale
  `uint8` NumPy arrays inside NPZ, not individual PNG/JPEG files. Exact array
  names and shapes are in
  `MedMNIST_Plus_ChestMNIST_v3.0\inventory.json`. The source-provided split
  arrays are inventoried as-is; no project split or evaluation was created.
- **Missing identifiers:** the reduced archive contains no per-image or
  patient identifiers, site/device metadata, or patient-to-split mapping.
  Although the upstream dataset describes 30,805 source patients, that
  aggregate count does not identify which released array rows share a
  patient.
- **Use boundary:** limited development/inventory only. The absent
  patient-to-image mapping means this archive cannot support a
  patient-independent evaluation for this project. It is a possible
  other-medical image source only, not breast-ultrasound positive data or a
  diverse medical/ultrasound challenge set.

## Shared limitations and serving status

Neither dataset supports patient-independent evaluation with the identifiers
available. Their acquisition does not close the need for a rights-cleared,
patient-linked breast-ultrasound cohort, traceable diverse negative
categories, real screenshots/low-quality examples, or a fresh untouched
evaluation cohort. These archives were not put in application assets or model
inference paths and were not used for training or evaluation.

Public classification remains disabled. No deployment or configuration
change was made.
