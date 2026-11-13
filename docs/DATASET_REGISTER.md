# Dataset register

**Latest cross-source development, 4 November 2026:** TCIA and BUS-BRA have been
explicitly relabeled inspected development sources under the user's instruction.
Recorded dataset CC BY 4.0 terms and attribution obligations support the limited
local research comparison; the BUS-BRA repository code's GPL license is separate
from the dataset release license. Known source groups and detected duplicate
components remain together. BUS-BRA Case semantics and Mendeley patient IDs
remain unresolved. The historical locked evaluations below remain preserved.
See [measured cross-source report](CROSS_SOURCE_RESULTS_V1.md).

BUSI_WHU v3 is a new prediction-free lead: its official record
<https://data.mendeley.com/datasets/k6cpmwybk3/3> describes 927 images and
CC BY 4.0. Direct checked access routes returned HTTP403; neither files nor
grouping were acquired. It is not yet a qualified reserved external cohort.
Independent final validation remains blocked.

**4 November 2026:** The BUS-BRA reservation was consumed by one locked
source-Case evaluation of the selected frozen encoder. One Case / two images
were conservatively excluded before scoring. Grouping semantics remain
unresolved, and no patient-independent claim is made. The model performed
poorly and was not promoted. BUS-BRA is now inspected; see
[locked measured report](BUSBRA_LOCKED_EVALUATION_REPORT_V1.md).

**Subsequent BUS-BRA reservation:** Under explicit user authorization to acquire
research data, the authoritative Zenodo CC BY 4.0 metadata and authors' stated
research-use permission were checked. BUS-BRA v1 was acquired for limited local
research reservation, checksum/inventory/overlap audited, and kept prediction-free.
This supersedes the earlier acquisition block for this defined scope only.
Case is an explicit metadata field; patient/exam semantics remain unverified,
so no patient-independent evaluation or broader release permission is inferred.
See [bounded experiment and reservation](BOUNDED_DEVELOPMENT_RESULTS_V1.md).

**2026-11-03 update:** Official TCIA BREAST-LESIONS-USG v1 was acquired,
inventoried and used for a locked, biopsy-only external research evaluation.
See [acquisition and measured results](TCIA_ACQUISITION_AND_EVALUATION_2026-11-03.md).
This authoritative source is separate from the unverified repository mentioned
below. It supplies no gate negatives or clinical approval. Earlier reviews
retain their original date and scope.

This register distinguishes the dataset described by its authors from
third-party download records. A platform's license field is not independent
proof that the uploader had rights to relicense every image.

## BUSI (Breast Ultrasound Images)

- **Primary citation:** Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. “Dataset of
  breast ultrasound images.” *Data in Brief*. 2020;28:104863.
  [DOI](https://doi.org/10.1016/j.dib.2019.104863) |
  [PubMed](https://pubmed.ncbi.nlm.nih.gov/31867417/) |
  [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC6906728/).
- **Original-data facts reported by the paper:** 780 cropped PNG images from
  600 women (ages 25–75), collected in 2018; 487 benign, 210 malignant, and
  133 normal images. The article describes a mask for each image and
  radiologist review. It says the source images were collected at Baheya
  Hospital and that duplicates were removed during preparation.
- **Patient identifiers:** no usable patient-to-image identifier is supplied
  with the public folder distribution examined by this project. The paper's
  count of 600 women is not a substitute for an image-level patient mapping.
- **Download record:** the commonly circulated
  [Kaggle copy](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
  is a third-party mirror. Its page claims `CC0: Public Domain` and points to
  the paper, but the paper does not establish that the uploader owns or may
  relicense the underlying patient images. The Kaggle license field is
  therefore **not treated as verified authorization** for this project.
- **License / permitted use:** underlying image-data redistribution and
  commercial-training rights are **unverified**. The article's open-access
  license covers the article, not automatically its separate patient-image
  dataset. Do not redistribute or train a release intended for public or
  commercial use until the data steward or original authors confirm terms.
- **Overlap / limitations:** existing local copies have reported
  cross-split near-duplicate candidates and do not provide patient IDs.
  Separate BUSI-derived mirrors may overlap exactly; do not combine this
  dataset with BUSI mirrors or use a mirror as an external test set.
- **Project decision:** source is documented, but it is not cleared for
  training or redistribution by this project. The pipeline can inspect a
  user-authorized local copy; its manifest will mark the evaluation as
  image-level unless usable patient IDs are supplied.

## Breast Ultrasound Image (Mendeley Data)

- **Dataset record:** Rodrigues PS. *Breast Ultrasound Image*, version 1,
  published 2017-12-31.
  [Mendeley Data](https://data.mendeley.com/datasets/wmy84gzngw/1) |
  [DOI](https://doi.org/10.17632/wmy84gzngw.1) |
  [DataCite metadata](https://api.datacite.org/dois/10.17632%2Fwmy84gzngw.1).
- **Download:** the live version 1 record exposes `us-dataset.zip` (listed as
  3.11 MB), available from
  [the Mendeley public file URL](https://data.mendeley.com/public-files/datasets/wmy84gzngw/files/b63daee9-78de-4122-8475-9b3aa22ffd64/file_downloaded).
- **Documented contents:** 250 breast-ultrasound images: 100 benign and 150
  malignant. No normal class, patient identifiers, or mask counts are
  documented in the dataset description inspected.
- **License:** the DataCite record reports
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribution is
  required. The record's rights statement applies to this dataset version;
  verify the current record and cite Rodrigues before reuse.
- **Permitted use:** CC BY 4.0 allows reuse, including commercial reuse, subject
  to attribution and other applicable law. It does not provide missing normal
  examples or patient-group metadata.
- **Overlap / limitations:** exact overlap with BUSI or other reposts has not
  been ruled out. It is not a modality-negative dataset and cannot train or
  validate a breast-ultrasound eligibility gate.
- **Project decision:** selected for a binary research experiment because the
  dataset record and CC BY 4.0 rights metadata are publicly accessible. It is
  not used for a three-class model or modality gate. The website remains
  disabled regardless of this experiment.
- **Local verification (2026-10-29):** the official archive downloaded as
  `data\raw\mendeley_v1\us-dataset-official.zip` (3,264,338 bytes); ZIP CRC
  validation passed and all 250 BMP images were present. SHA-256:
  `87dbd3aaa89e2d8886839983be3a985b7b190d5d8cb167d95a8dcd5d224b63f7`.
- **Prepared split:** the deterministic seed-42 manifest groups exact and
  near-duplicate candidates (perceptual-hash distance at most 3) into 126
  groups before splitting. The current manifest contains 150 training, 50
  validation, and 50 test images. Patient IDs are absent, so this is not patient-independent;
  near-duplicate grouping cannot establish patient independence or eliminate
  all leakage.

## Download and preparation

1. For BUSI, open the Kaggle record above, sign in if prompted, review its
   current terms, and download the archive only if your intended use is
   authorized. Do not treat this project's record as a rights clearance.
2. For another copy of the Mendeley dataset, open the version 1 record and
   download `us-dataset.zip` (or use its public file link above). The verified
   local archive used in this experiment is ignored by Git; preserve the
   citation and license record with any downloaded data.
3. Keep downloaded data outside Git under `data\raw\`. Do not merge sources.
4. Organize a source as `data\raw\<source>\<class>\image.png` (BUSI class
   folders: `benign`, `malignant`, `normal`). The Mendeley archive observed
   here contains BMP images below `originals\<class>\`; the preparation script
   recursively reads BMP, PNG, and JPEG files. BUSI masks containing `_mask`
   are excluded from classification.
5. If the source provides a legally usable patient mapping, create a CSV with
   columns `image_path,patient_id`; paths may be relative to the source root.
   Never infer patient IDs from image appearance or class labels.
6. Prepare splits with `python ml\prepare_dataset.py --source data\raw\busi
   --output data\prepared\busi --metadata-csv data\patient_map.csv`. Without
   patient metadata, explicitly pass `--allow-image-level-split`; resulting
   metrics are image-level and are not patient-independent.

The application does not download datasets. Missing files, authentication, and
license checks are surfaced rather than silently replaced with synthetic data.

## Eligibility/OOD-gate data status

The locally verified Mendeley version 1 dataset contains breast-ultrasound
positive images only (benign and malignant); it has no documented negative
examples such as books, natural photographs, unrelated medical images, or
other ultrasound anatomies. BUSI is also breast-ultrasound-only and the
examined third-party mirror's rights are unresolved. Neither source can train
or evaluate a modality/OOD gate. No appropriately licensed negative set, gate
checkpoint, threshold-selection results, or locked gate test results are
available in this workspace. The gate is therefore unavailable and
classification remains disabled.

### Fetal ultrasound challenge source (candidate; not yet gate-ready)

- **Primary source:** Alzubaidi M, et al. *Large-Scale Annotation Dataset for
  Fetal Head Biometry in Ultrasound Images*, Zenodo record 8265464, v2.
  [Record and files](https://zenodo.org/records/8265464) |
  [DOI](https://doi.org/10.5281/zenodo.8265464) |
  [Dataset paper](https://doi.org/10.1016/j.dib.2023.109708).
- **Source-published facts:** 3,832 fetal-head ultrasound images, stated
  dimensions 959 × 661, anonymized/de-identified per the record README, and
  no predefined data splits or patient IDs. The source offers an
  original-image archive separately from annotation-format packages.
- **License:** the Zenodo record and included README state CC BY 4.0. Reuse
  requires attribution, indication of changes, and a link to the license.
  The README also asks users to credit original image authors; it does not
  identify each original author or provide patient IDs. Therefore the source
  needs curator review of provenance and attribution before redistribution
  or public model use.
- **Potential use:** negative category `other_ultrasound` only. Fetal scans do
  not represent the full range of other ultrasound anatomies, acquisition
  devices, or users. One-source negatives risk source-artifact shortcuts and
  cannot validate breast-ultrasound eligibility.
- **Selected file:** the downloaded original-image archive,
  `Diverse Fetal Head Images-original-image.full.zip`, source-listed size
  129,809,023 bytes and MD5
  `52f24da73f61cab9f3b1cfafa6c69cf2`. Do not treat a partial download as
  valid; verify exact size and checksum before extraction. No patient IDs or
  split manifest are supplied by the source.
- **Local inventory:** the archive checksum and ZIP integrity were verified;
  999/999 image files decoded. Dimensions were 800x540 (975), 800x542 (5),
  and 19 other dimensions (one image each). Perceptual hashing at dHash
  distance <=3 produced 991 groups. There are no patient identifiers and all
  split assignments remain unassigned.
- **Status:** source terms and metadata were reviewed on 2026-10-30. The
  archive is an inventory-only candidate; `rights_verified=false` pending
  curator review of original-image provenance and attribution. It is not
  included in an eligibility training/evaluation split. This dataset alone
  does not satisfy required category coverage.

### ChestMNIST medical-image challenge source (candidate; not gate-ready)

- **Primary source:** MedMNIST+ record
  [10.5281/zenodo.10519652](https://zenodo.org/records/10519652), ChestMNIST
  dataset; official MedMNIST metadata is in
  [the project's dataset information](https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py).
- **Source/provenance:** MedMNIST identifies NIH ChestX-ray14 as the source:
  112,120 frontal chest X-rays from 30,805 unique patients, with 14
  text-mined disease labels, and says it uses the official source split.
  The original dataset paper is Wang et al.,
  [ChestX-ray8](https://arxiv.org/abs/1705.02315), whose abstract describes
  108,948 frontal images from 32,717 patients and eight NLP-derived labels;
  MedMNIST documents the later ChestX-ray14 version. The NIH download is
  linked from the paper to
  [NIH Box](https://nihcc.app.box.com/v/ChestXray-NIHCC).
- **License:** the MedMNIST+ record and official MedMNIST repository state
  CC BY 4.0 for MedMNIST datasets (DermaMNIST is the stated exception).
  MedMNIST asks users to cite its paper and the source-dataset paper. This
  verifies the license claim for the redistributed MedMNIST archive, not a
  blanket license for unmodified NIH files or any other source copy.
- **Labels/modality:** single-channel frontal chest radiographs with 14
  multi-label targets: atelectasis, cardiomegaly, effusion, infiltration,
  mass, nodule, pneumonia, pneumothorax, consolidation, edema, emphysema,
  fibrosis, pleural, and hernia. This is a challenging `other_medical`
  category candidate only, not ultrasound.
- **Local archive:** `data\raw\medmnist\chestmnist_64.npz`, 401,604,127 bytes;
  published MD5 `9de6cd0b934ebb5b7426cfba5efbae16`. The checksum was verified
  after download.
- **Local inventory:** 78,468 training, 11,219 validation, and 22,433 test
  images; each image is 64x64, with a 14-column multi-label target. The
  archive does not expose patient identifiers. Preserve the provided splits.
- **Potential use:** candidate `other_medical` negatives (chest radiographs),
  not breast-ultrasound positives. The 64-pixel benchmark version is
  downsampled and does not represent the diversity of medical images or
  ultrasound-like hard negatives. Preserve the source's train/validation/test
  partitions; do not merge images or relabel them as ultrasound.
- **Identifiers/limitations:** the local MedMNIST NPZ has no patient IDs,
  original image IDs, site/device data, or explicit patient-to-split mapping.
  The upstream source states it uses the official split, but the reduced
  archive does not permit this project to audit patient-disjointness or
  duplicate leakage. The NIH Box distribution's own detailed reuse terms
  were not independently verified in this review.
- **Status:** archive and array inventory completed. The declared MedMNIST
  CC BY 4.0 terms are recorded, but the derived archive is not admitted to
  this project’s gate partitions until its source-split leakage can be
  verified; no patient-level claim or gate metric is made.
- **Reproducible inventory:** run
  `python -m ml.inspect_chestmnist --archive data\raw\medmnist\chestmnist_64.npz --output data\prepared\medmnist_candidate_v1\inventory.json`.

## Source review and data eligibility (2026-11-01)

The following distinguishes source records from data actually admitted to
model development. The terms in a repository or dataset record do not replace
review of provenance for underlying medical images.

| Need | Source checked and documented facts | Current data/identifiers | Decision |
|---|---|---|---|
| Breast-ultrasound positive | Mendeley Data `wmy84gzngw/1` and its [DataCite metadata](https://api.datacite.org/dois/10.17632/wmy84gzngw.1) state CC BY 4.0 and 250 breast-ultrasound images (100 benign, 150 malignant). | 250 local images; filenames are preserved, but no patient-to-file mapping, separate source accession ID, site/device metadata, or Normal class is documented. | Reuse permitted under the deposited CC BY 4.0 record with attribution; not adequate for patient-grouped split or final independent cancer evaluation. |
| Breast-ultrasound positive | BUSI paper reports 780 images from 600 women (487 benign, 210 malignant, 133 normal); see [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6906728/). | The publicly circulated mirror does not provide usable patient mapping; current top-level `data\raw\benign`, `malignant`, and `normal` folders have 100/100/60 files and no source manifest tying them to a licensed origin. | BUSI image rights remain unverified; the 260 top-level files are provenance-unverified and excluded. Do not equate paper access with a license for image data. |
| Other-anatomy ultrasound | Zenodo 8265464 and its [README](https://zenodo.org/records/8265464/files/Readme.txt?download=1) declare CC BY 4.0 and 3,832 fetal-head ultrasound images, with no predefined splits and no patient IDs; the README asks users to credit original image author(s). | The downloaded original-size archive contains 999 decoded images; 991 dHash<=3 groups; no patient IDs or per-image original-author mapping. | Do not admit to training yet: the bundle does not identify the original image authors needed by its attribution instructions, nor provide leakage-safe partitions. Inventory only. |
| Other medical | MedMNIST+ / ChestMNIST is declared CC BY 4.0 for the derivative archive by the official [record](https://zenodo.org/records/10519652) and [repository](https://github.com/MedMNIST/MedMNIST); source labels and modality described above. | Local 64x64 archive with source-provided arrays but no patient/image identifiers or site metadata. | Eligible for limited acquisition/inventory under the derivative archive’s terms with attribution; not admitted to gate partitions because patient grouping and duplicate isolation cannot be audited. The derivative license does not clear separately downloaded NIH originals. |
| Natural photographs and objects | [Wikimedia Commons reuse guidance](https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia) explains per-file creator/license terms, requires per-item review, and disclaims warranty of the uploader-supplied copyright status. [Open Images](https://storage.googleapis.com/openimages/web/index.html) provides image-level labels; its [download project](https://github.com/cvdfoundation/open-images-dataset) primarily distributes source URLs. | No images downloaded or incorporated. | Potential sources for manually curated book/object/natural-photo examples only after checking each asset’s origin, license, attribution, and image availability. No blanket license is assumed. |
| Screenshots, low-quality scans, additional ultrasound anatomies | No qualifying primary source with clear reuse terms and traceable source/patient identifiers was verified in this review. | No real examples present. | Missing; do not replace with synthetic fixtures or hand-crafted visual heuristics. |

### Stop decision

The available positives and candidate negatives do not meet the minimum
requirements for source/patient-level training, validation, and an untouched
test set. Required before any eligibility training:

- a rights-cleared breast-ultrasound cohort with patient IDs and site/device
  metadata, adequate normal/benign/malignant or ultrasound-suitability labels,
  and permission for the intended research/model use;
- source-attributed negative cohorts for natural photos, objects/books,
  unrelated medical images, multiple non-breast ultrasound anatomies,
  screenshots, and real low-quality scans;
- patient/source IDs or verifiable source-defined patient-disjoint splits,
  plus duplicate checks across all categories and sites;
- an untouched, independently sourced final test cohort with sufficient
  examples per category to report rates and confidence intervals.

No new dataset was downloaded or incorporated during this source-assessment
stage. The separate limited acquisition and inventory completed on
2026-11-01 is recorded below. No gate training, threshold selection,
classifier test evaluation, or new metrics were performed.

The eligibility preparation pipeline requires curator-supplied source
metadata for every real image: `license`, `source_url`, and an explicit
`rights_verified=true` attestation, along with label, negative category, and
optional patient ID. This is an auditable declaration, not independent legal
verification. Negative categories should include separately held-out
`other_ultrasound`, `other_medical`, `natural_image`, `object`, `screenshot`,
and `low_quality` examples where lawful and representative. A simple generated
drawing is not an acceptable substitute for those data.

## Additional source assessment (2026-11-01)

This assessment records source-page facts separately from project eligibility.
“Eligible for acquisition” means the reviewed source terms permit obtaining
the specified material for the stated limited research use; it does not mean
that the data are adequate for patient-independent evaluation, that every
asset has been acquired, or that a model trained on it is approved for release.

| Candidate and primary source | Version/date, modality, labels, and documented scale | Identifiers and provenance relevant to leakage checks | Rights and status |
|---|---|---|---|
| **Mendeley Breast Ultrasound Image** — [version 1](https://data.mendeley.com/datasets/wmy84gzngw/1), [DataCite rights metadata](https://api.datacite.org/dois/10.17632/wmy84gzngw.1) | Published 2017-12-31; 250 breast-ultrasound images (100 benign, 150 malignant); no Normal class documented. | No patient, site, device, or image-to-patient IDs documented by the record. Exact overlap with reposted BUSI images remains unruled out. | DataCite reports CC BY 4.0. **Eligible for acquisition** with attribution and applicable license notices. Limited to research development after deduplication; not sufficient for patient-level splits or a fresh final test cohort. A fresh archive download and inventory were completed 2026-11-01; see [limited acquisition manifest](LIMITED_DATA_ACQUISITION_2026-11-01.md). |
| **BUSI** — [dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6906728/) and [third-party Kaggle copy](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) | Paper published 2020; reports 780 cropped breast-ultrasound images from 600 women: 487 benign, 210 malignant, 133 normal. The paper describes image masks and radiologist review. | No usable image-to-patient mapping was found in the examined public folder distribution. The 260 root `data\raw` class-folder images have no source manifest and are not confirmed as BUSI. | The mirror claims CC0, but that does not establish authority to relicense the underlying patient images; the paper’s article license is not an image-data license. **Excluded** pending direct rights confirmation from the data steward/rightsholder. Do not acquire or use the unverified local 260 images. |
| **Fetal-head ultrasound** — [Zenodo record 8265464](https://zenodo.org/records/8265464), [included README](https://zenodo.org/records/8265464/files/Readme.txt?download=1) | Version 2; 3,832 fetal-head ultrasound images; no predefined splits or patient IDs. | No patient IDs or per-image original-author mapping. The local selected archive has 999 decoded images and 991 dHash<=3 groups; it is inventory only. | Record and README state CC BY 4.0, while the README also requests credit to original image authors not mapped to files. **Needs permission or clarification** on satisfying attribution and provenance; do not admit to model partitions meanwhile. |
| **ChestMNIST / MedMNIST+** — [Zenodo record](https://zenodo.org/records/10519652), [official MedMNIST metadata](https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py) | Zenodo version 3.0; the derivative archive is labeled as 64x64 frontal chest radiographs with 14 multi-label targets, sourced from ChestX-ray14. Local arrays contain 78,468 train / 11,219 validation / 22,433 test images. | The local NPZ has no patient, image, site, or device identifiers. It preserves source-provided arrays, but this project cannot audit patient separation or duplicates across the underlying source. | The derivative MedMNIST archive is declared CC BY 4.0; this is not blanket authorization for other NIH copies. **Eligible for acquisition** only for the licensed derivative archive with attribution. A fresh archive download and inventory were completed 2026-11-01; it remains limited development/inventory only and is not included in gate partitions. See [manifest](LIMITED_DATA_ACQUISITION_2026-11-01.md). |
| **MIMIC-CXR** — [PhysioNet v2.1.0 record](https://physionet.org/content/mimic-cxr/2.1.0/), [database paper](https://doi.org/10.1038/s41597-019-0322-0) | Version 2.1.0 (2024); chest radiographs and reports. The source documents 377,110 images and 227,835 studies from one institution. | Source documentation describes de-identified `subject_id`, `study_id`, and per-image identifiers/hashes, which can support grouping and duplicate checks after authorized access. Single-institution data do not provide site diversity. | PhysioNet access is controlled. **Needs permission or clarification**: complete its current access requirements and review dataset-specific terms for the intended gate-training/model use before acquisition. No data downloaded. |
| **CheXpert** — [official Stanford dataset page](https://stanfordmlgroup.github.io/competitions/chexpert/) and [paper](https://arxiv.org/abs/1901.07031) | Paper (2019) reports 224,316 chest radiographs from 65,240 patients, with 14 report-derived labels and radiologist reference-standard subsets. | The official landing page reviewed here does not establish the patient-to-file mapping or site/device metadata needed by this project; inspect the authorized source manifest before relying on patient grouping. | A dataset-specific license/terms grant for this project’s intended model use was not verified from the checked source page. **Needs permission or clarification**; do not download or use until terms and identifier availability are confirmed. |
| **CAMUS echocardiography** — [official challenge page](https://www.creatis.insa-lyon.fr/Challenge/camus/), [2019 paper](https://doi.org/10.1109/TMI.2019.2900516) | Challenge page describes 2D two- and four-chamber echocardiography sequences from 500 patients, with segmentation/volume-estimation annotations; page links to a download portal. | The source page does not establish whether the released files expose patient-level identifiers suitable for group splitting, nor device/site metadata. | A dataset license and the download portal’s applicable access terms were not verified on the reviewed page. **Needs permission or clarification** before acquisition or model training. |
| **EchoNet-Dynamic** — [official dataset page and research-use agreement](https://echonet.github.io/dynamic/) | 10,030 apical-four-chamber echocardiography videos acquired at Stanford during 2016–2018; labels/annotations include ejection fraction, ventricular volumes, and expert tracings. | Patient-level identifiers and a reusable patient-to-file map were not confirmed in the reviewed public documentation. One institution; not multi-site evidence. | The posted agreement limits use to personal, non-commercial research, prohibits commercial use and derivative works, and restricts transfer/publication. **Excluded** for eligibility-gate training or model release under the posted terms unless Stanford grants written permission changing those terms. |
| **Open Images V6** — [official download page](https://storage.googleapis.com/openimages/web/download.html), [project repository](https://github.com/cvdfoundation/open-images-dataset) | V6 released February 2020; the official page describes 1,743,042 training images annotated for 600 boxable object classes, plus validation/test sets. Useful candidate classes may include people, animals, books, and ordinary objects; this is not a curated gate-negative cohort. | Open Images image IDs and source URLs support asset tracking, not patient grouping. Class labels and source provenance must be reviewed per asset; URL availability and image characteristics may vary. | No single project-level grant for every underlying asset was established in this review. **Needs permission or clarification** until each selected image’s source, license, attribution, availability, and suitability are recorded. No assets acquired. |
| **Wikimedia Commons individual files** — [reuse guidance](https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia) | Not a fixed dataset; modality, labels, and counts depend on the individually selected files. Potential source for selected natural/object images and screenshots. | Commons file pages can provide file/source and uploader/license information; there are no patient identifiers. Commons cautions that uploader-supplied copyright information is not warranted. | **Eligible for acquisition only file-by-file** when the actual file page, source, license, attribution, and reuse conditions are verified and recorded. No screenshot/natural/object assets were selected or acquired. This is not a blanket clearance for Commons content. |
| **Screenshots and real low-quality breast-ultrasound scans** | No suitable, rights-cleared, documented collection with the required labels and source metadata was verified. | No source IDs, patient IDs, collection sites, quality labels, or leakage-safe partitions are available. | **Needs permission or clarification** through a rights-cleared clinical/research collection or individually licensed screenshot source. No substitute data should be fabricated. |

### Acquisition plan and stop conditions

The source review does not identify a complete, permitted, traceable set from
which to make patient/source-safe train, validation, threshold-selection, and
fresh final-test partitions. Before eligibility-gate training, obtain and
document all of the following:

1. A rights-cleared breast-ultrasound cohort with a file-to-patient mapping,
   class/suitability labels, source/site and device metadata where available,
   and written permission for the planned research and model use. Existing
   Mendeley images can support only limited development; BUSI images require
   rights clarification and a usable patient map.
2. Multiple-source negatives for non-breast ultrasound anatomies, unrelated
   medical imaging, natural photographs, books/objects, and genuine
   screenshots. Preserve source IDs, per-file rights/attribution, modality or
   category labels, and source/site/device metadata where applicable. MIMIC,
   CheXpert, CAMUS, and EchoNet are not cleared by this review; the EchoNet
   terms presently exclude the intended derivative model use.
3. Real low-quality breast-ultrasound examples with a documented, defensible
   quality label and permission for use. A model-generated degradation or
   synthetic screenshot is not evidence for this category.
4. Patient/source-disjoint manifests and exact/near-duplicate checks across
   all candidate sources. For image-only sources, group at the source-item
   level and clearly limit the claims; do not describe the resulting estimate
   as patient-independent.
5. A fresh, untouched final test cohort that has not been used for training,
   source selection, threshold choice, or model selection. Obtain enough
   observations in each required category to report counts and confidence
   intervals. The previously inspected 50-image classifier test set is not a
   fresh final evaluation set.

Until these conditions are met, stop before training or evaluation. No
candidate status alone is a claim of acquisition or rights adjudication.
Actual limited acquisitions are documented below. No model performance was
measured.

## Limited acquisition and inventory (2026-11-01)

The only datasets acquired in this stage were Mendeley Breast Ultrasound
Image v1 and the CC BY 4.0 MedMNIST+ ChestMNIST 64 derivative, the candidates
already marked eligible for limited acquisition. Current authoritative
records were rechecked before download: Mendeley/DataCite reports version 1
and CC BY 4.0; the Zenodo record reports MedMNIST+ version 3.0 and CC BY 4.0.
No candidates marked “needs permission or clarification” or “excluded” were
accessed or acquired.

Both original source archives were downloaded unchanged into
`data\research_data\limited_acquisition_2026-11-01\`, outside application
assets and model/inference paths. Mendeley’s fresh download (3,264,338 bytes,
SHA-256
`87dbd3aaa89e2d8886839983be3a985b7b190d5d8cb167d95a8dcd5d224b63f7`)
matches the previously verified official archive hash. Its 253 ZIP entries
passed path traversal/absolute-path/symlink/duplicate-path checks and CRC
validation before extraction. It contains 250 BMPs, all decoded: 100 benign,
150 malignant, 46 distinct dimensions, 75–199 px wide and 57–161 px high.
Source file paths are preserved, but no patient-to-file mapping, separate
source accession ID, or site/device metadata is present.

The ChestMNIST 64 archive (401,604,127 bytes, SHA-256
`e28fbed21400c50040d6a4e51f379dce89bcb24819d3878c252ca78d52d92c3e`)
matches the current Zenodo record’s published byte size and MD5
`9de6cd0b934ebb5b7426cfba5efbae16`. Its NPZ contains six published arrays;
all 112,120 `uint8` grayscale 64×64 image tensors loaded and their 14-column
label arrays passed structural/value checks: 78,468 train, 11,219 val, and
22,433 test. Those are source-provided arrays, inventoried in place; no
project split was created. ChestMNIST is chest X-ray data, not breast
ultrasound. It contains no per-image/patient identifiers, site/device data,
or patient-to-split mapping.

The exact download URLs, access date, version, license and attribution terms,
archive sizes and hashes, detailed dimensions, labels, and missing identifiers
are recorded in the [limited acquisition manifest](LIMITED_DATA_ACQUISITION_2026-11-01.md)
and adjacent `inventory.json` sidecars. Both datasets are **limited
development/inventory only**. Neither supports patient-independent
evaluation with the identifiers available; ChestMNIST is not breast-
ultrasound data and cannot replace the missing positive cohort.

No training, partition creation, threshold tuning, eligibility-gate or
classifier evaluation, performance measurement, deployment, or configuration
change occurred. Public classification remains disabled.

## Duplicate-image audit and label review (2026-11-01/02)

The local source archives were re-audited without changing archive bytes,
published arrays, or source splits. Mendeley has zero exact file-duplicate
pairs among 250 images. ChestMNIST has 25 exact pixel groups / 27 pairs
(24 pairs plus one three-image group); its 112,120 source samples contain
112,094 globally unique pixel arrays after exact deduplication. Fourteen
duplicate groups have identical 14-label vectors; 11 have conflicting
vectors. Three exact pairs cross published splits (two train–validation,
one validation–test); no exact train–test pair was found.

The exact cross-split image overlap is a source-split image-quality concern,
not proof that patients overlap or that patient leakage occurred. The archive
has no patient-to-image map, so patient-level split validity cannot be
determined. The audit also validates pHash accounting and visually reviews
all Mendeley candidate pairs, every exact ChestMNIST group, all 26
non-exact same-hash ChestMNIST pairs, and a deterministic sample of 45/1,732
nearby-code ChestMNIST pairs. Similarity is not proof of duplicate
acquisitions or shared patients. Full group labels, split counts, methods,
sample selection, contact-sheet locations, and machine-readable results are
in the [duplicate-image audit report](DUPLICATE_IMAGE_AUDIT_2026-11-01.md).

ChestMNIST is chest X-ray data, not breast ultrasound, and cannot train or
validate a breast-ultrasound eligibility gate. Both datasets remain limited
development/inventory only. No partitions, training, or model evaluation
were performed; dataset dispositions and source splits remain unchanged.

## Follow-up source verification and acquisition plan (2026-11-01)

This follow-up checked current primary records and authoritative metadata
pages. “Potentially suitable pending verification” does not mean permission
or acquisition is approved. No dataset owner was contacted; no access
application, download, or archive inspection was performed in this review.

| Candidate and authoritative source | Verified record facts | Identifiers, access, and evidence gaps | Disposition |
|---|---|---|---|
| **BUS-BRA** — [Zenodo v1.0 record 8231412](https://zenodo.org/records/8231412), [record metadata API](https://zenodo.org/api/records/8231412), [paper abstract](https://pubmed.ncbi.nlm.nih.gov/37937827/) | Zenodo record published 2023-03-13 (last modified 2024-09-25); breast ultrasound; 1,875 images from 1,064 female patients; paper reports four scanners at one institution (National Institute of Cancer, Rio de Janeiro), with biopsy-proven 722 benign and 342 malignant cases and BI-RADS 2–5. The paper says 5- and 10-fold partitions are supplied and CSV files contain dataset information. | Zenodo declares CC BY 4.0, states PEB/COPPE-UFRJ holds copyright and requires citation. The paper calls availability “for academic and research purposes.” Whether the CC BY declaration controls all intended uses needs clarification. Neither the file contents nor the CSVs were inspected; patient-to-image mapping, scanner-to-image mapping, and whether folds are patient-disjoint are unverified. One institution limits external-site generalization. | **Requires access/permission** before acquisition: clarify the rights/use wording and inspect permitted files/metadata. It may be suitable as a breast-ultrasound research cohort if patient mapping and grouping are confirmed, but is not approved as a fresh final test set. |
| **BUS-UCLM** — [DataCite DOI 10.17632/7fvgj4jsp7.1](https://api.datacite.org/dois/10.17632/7fvgj4jsp7.1), [Mendeley version 1 landing page](https://data.mendeley.com/datasets/7fvgj4jsp7/1) | DataCite metadata issued 2024-02-26 describes 683 breast-ultrasound images from 38 patients: 174 benign, 90 malignant, 419 normal; Siemens ACUSON S2000 scans acquired 2022–2023. It declares CC BY-NC 3.0. The Mendeley landing page returned “Dataset Not Found” when checked 2026-11-01. | DataCite’s aggregate patient count does not establish a usable patient-to-image map, and site/device is one system. Current data availability is unconfirmed. Thirty-eight patients are inadequate as the sole final evaluation cohort; non-commercial terms would also need to match the planned use. | **Requires access/permission**: availability, patient mapping, and intended-use terms must be confirmed. Do not rely on it as a standalone independent test. |
| **BUS_UC** — [Mendeley version 1](https://data.mendeley.com/datasets/3ksd7w7jkx/1), [DataCite metadata](https://api.datacite.org/dois/10.17632/3ksd7w7jkx.1) | Published 2023-10-20; 811 256×256 breast-ultrasound images: 358 benign and 453 malignant. The author says the images were obtained from ultrasoundcases.info and then annotated with radiologist assistance. DataCite/Mendeley declares CC BY 4.0. | No patient count, patient-to-file identifiers, or site/device mapping documented in the reviewed record. The repository-level license does not resolve rights in the upstream third-party clinical images. | **Requires access/permission**: verify upstream image rights and obtain traceable identifiers before use. Not suitable for patient-independent evaluation on current evidence. |
| **MIMIC-CXR v2.1.0** — [PhysioNet record](https://physionet.org/content/mimic-cxr/2.1.0/) | PhysioNet’s current record identifies version 2.1.0 (2024); its database description reports 377,110 chest radiographs across 227,835 studies, from Beth Israel Deaconess Medical Center. This is chest X-ray, not ultrasound. | Controlled access applies. Patient (`subject_id`), study, and image identifiers are described by the database documentation, subject to authorized access and the data-use agreement. Single institution; no ultrasound anatomy diversity. | **Requires access/permission**: complete current credentialing and data-use terms before acquisition. Potential other-medical negatives only, not breast-ultrasound positives or multi-site evidence. |
| **CAMUS** — [official challenge page](https://www.creatis.insa-lyon.fr/Challenge/camus/) | Official page describes 2D two- and four-chamber echocardiography sequences from 500 patients and requires citation of its 2019 paper. | The page links to a download portal but does not state the applicable data license or establish released patient-to-file/site/device mappings. | **Requires access/permission**: confirm current portal access, permitted use, and patient grouping. Potential other-anatomy ultrasound negatives only. |
| **Fetal-head ultrasound v2** — [Zenodo record 8265464](https://zenodo.org/records/8265464) and its [README](https://zenodo.org/records/8265464/files/Readme.txt?download=1) | Zenodo identifies 3,832 fetal-head ultrasound images, 959×661 pixels, and declares CC BY 4.0. The project’s earlier inventory of the already acquired bundle is recorded above and in the prior source review. | The record offers an aggregate image count, not patient IDs or patient-to-file mapping. README attribution to original image authors is not mapped to individual files in the prior inventory. | **Requires access/permission** to resolve attribution/provenance and patient grouping before model partitions. Do not treat it as independent patient-level negative evidence. |
| **Open Images V6** — [official download page](https://storage.googleapis.com/openimages/web/download.html) | V6 released February 2020. The official page lists 1,743,042 training images annotated for 600 boxable object classes, plus 41,620 validation and 125,436 test images; image IDs are available. | This is a general object-image source, not a medical dataset. Image IDs/source URLs permit traceability but are not patient identifiers. The page distributes source assets and annotations; the rights and attribution for each chosen image must be checked individually. | **Potentially suitable pending verification** for a curated natural/object negative set only after per-image source, license, attribution, availability, and category checks. No assets selected or acquired. |
| **Natural photographs, screenshots, and real low-quality/corrupted scans** | No fixed source with verified image-level terms, suitable labels, and traceable provenance was confirmed for screenshots or naturally low-quality/corrupted breast ultrasound. | Real examples need reliable suitability/quality labels and source provenance. Artificially corrupted copies can test code handling, but are not evidence of real-world gate performance. | **Unresolved blocker**: obtain an explicitly licensed, curated source or approved institutional collection; do not substitute synthetic fixtures as evaluation evidence. |

**Unverified repository lead (not acquired):** a
[third-party GitHub repository](https://github.com/ii50-txt/BrEaST-Lesions_USG-images_and_masks)
describes a BUSBRA-based segmentation project and asserts case-ID-based
splits in its README. The GitHub API reports no repository license
(`license: null`); the README is not an authoritative rights record, and
neither original image provenance nor permitted use was verified. Repository
claims about its contents and grouping were not independently checked. No
files were downloaded or incorporated. This does not change the official
BUS-BRA disposition above: its terms and patient/source mappings still need
verification before acquisition or evaluation.

### Concrete acquisition plan and minimum evidence

1. **Breast-ultrasound development cohort:** first resolve BUS-BRA rights and
   obtain written/recorded confirmation of intended research/model use. Before
   any acquisition for modeling, inspect the released CSV schema and verify
   stable patient-to-image and scanner-to-image mappings, biopsy labels,
   normal/non-lesion availability, and the patient unit for its published
   folds. Retain the source-defined patient mapping and license terms.
2. **Fresh classifier test cohort:** obtain a separate, rights-cleared
   breast-ultrasound cohort, not used for training or threshold/model
   selection, with patient-to-image mapping and pathology-backed benign and
   malignant outcomes. Prefer a different institution and device mix. Set
   sample targets in advance from the desired confidence-interval precision;
   do not infer adequacy from image totals alone.
3. **Eligibility-gate training and test sources:** acquire lawfully usable,
   separately labeled real images for natural photos, objects/books,
   unrelated medical modalities, and multiple non-breast ultrasound
   anatomies. Obtain genuine screenshots and low-quality/corrupted scans
   with documented source and quality labels. Keep patient/source IDs and
   source-defined acquisition groups for split isolation; check exact and
   near duplicates across every source before locking the final test cohort.
4. **Per-asset provenance:** for general image collections such as Open
   Images, record each image ID, original source, exact license, required
   attribution, availability, and label before inclusion; a dataset-level
   landing page is not a blanket rights grant.
5. **Access and audit trail:** record version/date, source and access URLs,
   terms/approval evidence, archive digest, identifiers available, label
   definitions, and exclusions in the manifest. Use separate training,
   validation/threshold-selection, and untouched final-test partitions
   grouped by patient/source (and site/device where possible). Select
   thresholds on validation data only and report each negative category
   separately with sample counts and confidence intervals.

**Stop condition:** until those rights, identifiers, labels, category-specific
negative examples, and independent test cohorts are available and verified,
there is no basis to create leakage-safe partitions, train/evaluate the gate,
or make patient-independent classifier claims. The existing Mendeley and
ChestMNIST archives remain limited development/inventory only.

## Clarification and institutional collection planning package (2026-11-01)

The review-ready, unsent question set and separate draft requests are in
[DATASET_CLARIFICATION_PACKAGE.md](DATASET_CLARIFICATION_PACKAGE.md). Its
common questionnaire applies to every source and asks about intended
research/model-development use, conflicts between papers and dataset terms,
redistribution and derived-model/demo/commercial restrictions,
patient-to-image mapping, patient-disjoint splits, repeat/duplicate overlap,
site/device/acquisition metadata, label references and limitations,
attribution, and access/ethics/institutional approvals.

Source-specific questions to resolve before acquisition are:

- **BUS-BRA:** reconcile Zenodo CC BY 4.0 and the paper’s “academic and
  research purposes” language; verify image/annotation coverage, patient,
  exam, and scanner mappings in the CSVs, and confirm whether supplied folds
  are patient-disjoint with repeat/duplicate checks.
- **BUS-UCLM:** establish the current official access route after the
  “Dataset Not Found” response; verify exact CC BY-NC 3.0 scope and intended
  model use, stable patient/exam mapping, label definitions, and any splits.
- **BUS_UC:** establish depositor authority for third-party
  ultrasoundcases.info images; identify per-image source and patient/exam
  grouping, label reference standard, and upstream restrictions.
- **MIMIC-CXR:** use only the official controlled-access pathway; confirm
  current credentialing, training, institutional/ethics, and data-use
  requirements, permitted derived weights/demo/commercial use, identifier
  grouping, labels, and available acquisition metadata.
- **CAMUS:** verify current download terms/license and approvals; establish
  patient grouping across views/sequences, available device/site metadata,
  label scope, and split/duplicate handling.
- **Fetal ultrasound:** confirm that CC BY 4.0 covers all files, resolve
  per-image original-author attribution requested by the README, and verify
  patient/acquisition grouping, repeated images, label limitations, and
  downstream model-use conditions.
- **Institutional screenshots and naturally low-quality/corrupted scans:**
  use the package’s planning checklist to obtain authorization, consent or
  approved waiver, privacy review, provenance, category/quality labels,
  pseudonymous grouping IDs, and secure handling. The checklist is not an
  ethics or legal approval.

No source disposition is changed by this planning package. No external
message or access application was sent, no data were downloaded, and no
application or model configuration was changed.

Response tracking for the six datasets is maintained in
[DATASET_CLARIFICATION_TRACKER.md](DATASET_CLARIFICATION_TRACKER.md). It
records each question as Pending, the evidence to retain, separate legal
permission and evaluation-suitability decisions, and the criteria required
before any future disposition change.

## New cohort and negative-source review (2026-11-02)

The expanded review is documented in
[COHORT_SOURCE_REVIEW_2026-11-02.md](COHORT_SOURCE_REVIEW_2026-11-02.md).
New primary evidence from the 2025 BUS-UCLM paper confirms 38 pseudonymous
patient groups and describes patient-level five-fold partitions, but the
v3 archive was not reachable through the checked author-linked download
URL (HTTP 403). The dataset record and paper state CC BY 4.0 for v3; the
article itself has a separate CC BY-NC-ND license. BUS-UCLM is still too
small and single-site to serve as a standalone fresh final test cohort.

BUS-CoT is a larger candidate, but its aggregate patient count is not a
patient-to-image map; image-level source licensing, grouping, and the
reported split unit remain unverified. ALN-Ultra, BUS-Large, the Taichung
breast/lymph-node release, and FOCUS-7D have unresolved access, provenance,
rights, or identifier questions. No new source is approved for acquisition
or patient-independent evaluation. MURA remains an unlicensed/terms-unverified
medical-negative lead; Open Images remains subject to per-image rights
review. No suitable source was found for genuine screenshots or naturally
low-quality scans.

No new data were acquired, and existing source dispositions remain unchanged
except that BUS-UCLM's patient-mapping evidence is updated by the 2025 paper.
The clarified BUS-UCLM evidence and unresolved owner-response questions are
recorded in the [clarification tracker](DATASET_CLARIFICATION_TRACKER.md).
