# New cohort and negative-source review (2026-11-02)

**Scope:** public-source research only. No dataset archives were downloaded,
no access requests or external messages were sent, and no data were extracted,
partitioned, or evaluated. Dataset-level license statements were not treated
as proof that third-party source images may be relicensed.

## Outcome

No new dataset is currently verified as eligible for patient-independent
evaluation. **BUS-UCLM v3 is the clearest patient-linked lead**, but its
published patient mapping does not overcome the small, single-site cohort,
and its files were not reachable through the checked download path. BUS-CoT
reports a much larger patient count but does not establish stable
patient-to-image identifiers in the inspected public paper; it also combines
third-party images whose per-source permissions and attribution need to be
checked before acquisition.

No candidate was downloaded. No SHA-256 hashes were computed for these
candidates. Their reported record sizes/checksums below are source metadata,
not locally verified archive hashes.

For records declaring **CC BY 4.0**, the stated license permits sharing and
adaptation, including commercial use, subject to attribution, a license link,
and an indication of changes. It grants only rights the licensor has
authority to grant; it does not itself resolve privacy/publicity rights or
third-party material. No separate public-demo restriction was stated in
those records, but unresolved source rights and controlled-access terms
remain controlling blockers where noted.

## Breast-ultrasound candidates

### BUS-UCLM, Mendeley Data v3

- **Authoritative records:** [Mendeley version 3](https://data.mendeley.com/datasets/7fvgj4jsp7/3),
  DOI [10.17632/7fvgj4jsp7.3](https://doi.org/10.17632/7fvgj4jsp7.3);
  the 2025 *Scientific Data* paper, [BUS-UCLM: Breast ultrasound lesion
  segmentation dataset](https://doi.org/10.1038/s41597-025-04562-3)
  ([full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11814256/)); and the
  authors' [download/code instructions](https://github.com/noeliavallez/BUS-UCLM-Dataset).
- **Version/date and contents:** v3, published 2025-01-27. The paper reports
  683 768×1024 PNG images from 38 patients: 419 normal, 174 benign, and 90
  malignant. Images were acquired in 2022–2023 at Ciudad Real General
  University Hospital using one Siemens ACUSON S2000 system and an 18L6 HD
  probe. Lesion malignancy was biopsy-confirmed; radiologists annotated the
  masks. The paper reports ethics approval and informed consent.
- **Patient and split evidence:** the paper states that the first four
  characters in each image filename identify one anonymized patient, and the
  sequence suffix identifies an image within that patient's study. It reports
  five-fold cross-validation with patient-level partitions for a comparative
  experiment. The public fold assignments and a complete split manifest were
  not verified. The paper reports one site/system and only 38 patients; this
  is not adequate as the sole fresh final test cohort or for broad site/device
  generalization.
- **Terms and attribution:** the v3 DataCite record and the paper's dataset
  record section state CC BY 4.0; the paper separately has a CC BY-NC-ND
  article license. Attribute Vallez, Bueno, Deniz, Rienda, and Pastor, cite
  the dataset DOI and paper DOI, link CC BY 4.0, and indicate changes. The
  older v1 metadata had different rights; use only the v3 terms for v3.
- **Access result:** the v3 Mendeley page displayed dataset metadata but no
  file listing. The authors' README links to
  `https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/7fvgj4jsp7-1.zip`;
  both an unauthenticated HEAD and a one-byte Range GET returned HTTP 403.
  No alternate credentials or access route was used.
- **Disposition — Requires access or clarification.** Patient grouping and
  the paper's patient-level split method are documented, but the archive was
  not accessible for inventory, the published fold files were not verified,
  and 38 patients at one site cannot supply a robust fresh independent test.
  If the official v3 archive becomes accessible under the stated terms, it
  may be useful for limited patient-grouped development, not as the sole
  external test cohort.

### BUS-CoT, Figshare v1

- **Authoritative records:** [Figshare dataset record v1](https://doi.org/10.6084/m9.figshare.30838715.v1)
  and the 2026 *Scientific Data* paper,
  [A Chain-of-thought Reasoning Breast Ultrasound Dataset Covering All
  Histopathology Categories](https://doi.org/10.1038/s41597-026-06702-9)
  ([PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12988125/)).
- **Version/date and reported contents:** Figshare v1 was published
  2025-12-09; the paper appeared 2026-02-06. The paper/record report 11,439
  ultrasound images, 11,850 lesions, and 4,838 patients; modalities include
  B-mode, Doppler, and elastography, with five device types. Labels include
  benign, malignant, and uncertain malignant potential; the curated
  BUS-Lesion subset has 5,163 images (2,694 benign, 2,469 malignant) and
  reported train/test counts of 4,338/825. The paper's pathology table lists
  4,856 benign, 4,514 malignant, and 2,069 other lesions (11,439 total),
  which does not reconcile with its 11,850-lesion claim. Split grouping and
  duplicate handling for the 4,338/825 subset were not verified.
- **Provenance and identifiers:** the paper says images were gathered from
  scientific publications, public datasets, and Radiopaedia/Eurorad cases;
  images were cropped/preprocessed, and per-image source information is
  described as available in dataset metadata and supplementary tables.
  Aggregate patient count does not establish stable patient-to-image IDs.
  Source-level duplicates, patient overlap across sources, and site/device
  mappings were not verified. The dataset is lesion-focused and does not
  provide a normal-breast class.
- **Terms and attribution:** the Figshare record identifies CC BY 4.0 and
  provides a 6,942,371,401-byte `BUSCoT.zip` (record MD5
  `832db89dd4095f49a1a4182a72c75be8`). The paper says its sources comply with
  applicable licenses and are cited, but the underlying sources are
  heterogeneous and include third-party published/case images. The
  source-by-source license and attribution crosswalk in the supplement was
  not verified. Attribute the deposited work to Youcheng Li, cite the
  dataset DOI and 2026 paper DOI, link CC BY 4.0, and indicate changes. The
  record-level CC BY statement does not by itself resolve rights the
  depositor may not control.
- **Disposition — Requires access or clarification.** Promising for
  development and source-held-out research only after verifying the
  per-image rights/provenance, stable grouping fields, label references, and
  split unit. No download was made.

### ALN-Ultra, Zenodo v1

- **Authoritative record:** [Zenodo v1 record 18483501](https://zenodo.org/records/18483501),
  DOI [10.5281/zenodo.18483501](https://doi.org/10.5281/zenodo.18483501),
  published 2026-02-04.
- **Reported scope:** paired breast/axillary lymph-node ultrasound images
  and videos from 257 breast-cancer patients, with expert assessments and
  histopathological biopsy results for axillary-node metastasis. The
  endpoint is nodal metastasis, not normal/benign/malignant breast-lesion
  classification. No image count, patient-to-file map, site/device fields,
  or patient-disjoint split evidence is stated in the record.
- **Terms/access:** the Zenodo metadata has a CC BY 4.0 license field, but
  the authoritative record marks access as **restricted** and lists no
  files, despite describing the dataset as open-access. No access request
  was submitted. Cite Hong Zhiliang and the Zenodo DOI and follow CC BY 4.0
  attribution terms if the license is confirmed to apply to the released
  files; no separate public-demo or commercial restriction is stated in the
  record.
- **Disposition — Requires access or clarification.** Resolve the access
  and rights mismatch and verify identifiers before considering use.

### BUS-Large, KTH data repository

- **Authoritative record:** [KTH record](https://datarepository.kth.se/records/frfd6-ymw69),
  DOI [10.71775/kth.kxrjk-gq496](https://doi.org/10.71775/kth.kxrjk-gq496);
  the related record is [10.71775/kth.frfd6-ymw69](https://doi.org/10.71775/kth.frfd6-ymw69),
  published 2026-07-15.
- **Reported scope:** a harmonized derivative of 21 publicly available
  breast-ultrasound datasets, with masks and clinical labels. The record
  text says “more than 21,00 images”; the count is malformed and was not
  normalized to an assumed value. No new patient count, stable patient
  mapping, or split-level duplicate evidence is stated. Aggregating prior
  datasets creates substantial source overlap risk.
- **Terms/access:** DataCite declares CC BY 4.0, but the KTH record says the
  files are restricted and requires an access request assessed under
  Swedish laws and regulations. The record also describes controlled access
  through an IRB-approved data-use agreement. No request was submitted.
  Cite KTH and the repository DOI under the stated CC BY 4.0 metadata; exact
  attribution and controlling use terms remain to be confirmed through the
  authorized access route.
- **Disposition — Requires access or clarification.** Access, controlling
  terms, source-level permissions, patient mapping, and cross-source
  duplicate handling remain unresolved.

### Breast and Lymph Node Ultrasound Image Dataset, Figshare v2

- **Authoritative record:** [Figshare record 30112000 v2](https://doi.org/10.6084/m9.figshare.30112000.v2),
  published 2026-04-24.
- **Reported scope:** a 100-image privacy-preserving breast/axillary-node
  ultrasound subset from Taichung Veterans General Hospital. The record
  also describes annotation files for approximately 3,800 images, without
  reconciling that number with the 100-image subset. It says images were
  anonymized and no patient-identifying information was preserved; no
  patient mapping, sample counts by label, or site/device breakdown is
  provided. The public image archive is 31,626,325 bytes and the annotation
  archive is 518,534 bytes according to Figshare metadata.
- **Terms/access:** the Figshare record declares CC BY 4.0 and calls the
  subset public illustration/methodological material, while saying access
  to the complete curated set is controlled under an IRB-approved DUA.
  Scope of the public license and the relationship between the image and
  annotation files need clarification. Attribute Meng-Hsiun Tsai,
  Chih-Chiang Hung, Chun-Ming Yen, Chih-Chan Yang, Wei-Hsin Hung, and
  Yun-Hsin Li; cite Figshare v2, link CC BY 4.0, and indicate changes for
  any use authorized by the applicable terms.
- **Disposition — Requires access or clarification.** No patient-independent
  evaluation is possible from the published metadata, and the public subset,
  annotation counts, and controlled-access statement are not sufficiently
  reconciled.

### BUSI curated derivative, Zenodo v1

- **Authoritative record:** [Zenodo record 21128640](https://zenodo.org/records/21128640),
  published 2026-07-02, declares CC BY 4.0 and contains a 161,657,476-byte
  `BUSI.zip` (record MD5 `c6d336ffa4b0314b1488c7e4c686d7d8`).
- **Reported scope and limits:** the record describes 629 images/masks as a
  curated version of the original BUSI dataset. It provides no patient
  identifiers or patient-to-image mapping and does not establish new
  patients. The underlying BUSI image rights remain unresolved in the
  existing register.
- **Disposition — Excluded from the patient-linked cohort shortlist.** This
  is a BUSI-derived repackage, not independent patient-linked evidence; its
  record-level license does not resolve rights in the source images. The
  record names Joaquín Ortiz de Murua, Adrián Inés Armas, and Jônathan Heras
  and links the original BUSI paper; source-image rights and any required
  attribution remain unresolved.

## Negative-image source review

- **Natural photographs and objects:** the existing register's
  [Open Images V6 review](DATASET_REGISTER.md#source-review-and-data-eligibility-2026-11-01)
  remains the only potentially usable path found: select assets individually
  and retain image ID, original source, exact license, attribution, and
  availability. No new assets were selected or downloaded.
- **Other medical imaging — MURA:** the [official Stanford MURA page](https://stanfordmlgroup.github.io/competitions/mura/)
  describes musculoskeletal radiographs and notes the competition is closed.
  The [original paper](https://arxiv.org/abs/1712.06957) reports 40,561 images
  from 14,863 studies with study-level normal/abnormal labels. The checked
  official page did not state a data license or current download terms;
  patient-to-image and site/device mappings were not verified.
  **Disposition — Requires access or clarification.** No MURA data were
  downloaded.
- **Non-breast ultrasound — FOCUS-7D:** [Zenodo record 21806452](https://zenodo.org/records/21806452),
  v1 published 2026-08-05, declares CC BY 4.0 and reports 6,432 ultrasound
  images: 2,192 cardiac/fetal-head/kidney images and 4,240 challenge-source
  images spanning appendix, breast, heart, fetal head, kidney, liver, and
  thyroid. Its `FOCUS-7D.rar` is listed as 1,949,457,591 bytes with MD5
  `27bf9a71023df1ce167816deaf4156b0`. The record does not state patient
  identifiers, per-category counts, or image-level site/device metadata;
  challenge-source rights and provenance were not independently verified.
  Cite the listed creators Zhou Ying, Lin Zehui, Huang Sihan, Zhang Yanming,
  Xu Dong, and Tan Tao and follow CC BY 4.0 attribution/change terms if
  rights to all included components are confirmed.
  **Disposition — Requires access or clarification.** It is a potential
  multi-anatomy negative-source lead, not a verified acquisition.
- **Screenshots and naturally low-quality/corrupted scans:** no source with
  verified terms, provenance, and category/quality labels was found. Use
  the existing institutional collection checklist; it is planning guidance,
  not legal or ethics approval. Synthetic corruptions are not substitutes
  for genuine examples.
- **Existing sources:** previously reviewed MIMIC-CXR, CAMUS, fetal
  ultrasound, and ChestMNIST dispositions are unchanged; they were not
  reacquired. ChestMNIST remains chest X-ray, not breast ultrasound.

## Prioritized next steps

1. Recheck BUS-UCLM v3 through the official Mendeley page or an authorized
   route. If files become accessible under v3 terms, preserve the original
   archive, verify SHA-256, inspect ZIP paths before extraction, and inventory
   the documented patient-group keys, image/mask pairing, class labels,
   duplicates, and any source split files. Do not treat 38 patients as a
   sufficient fresh final test cohort.
2. Before acquiring BUS-CoT, verify the supplementary per-source rights and
   attribution crosswalk, the archive's actual patient/image/source fields,
   whether its 4,338/825 split is patient/source-disjoint, and the apparent
   lesion-count inconsistency. The paper's aggregate count is not a
   patient-to-image map.
3. Keep the final cancer-classifier test cohort separate, fresh, and
   patient-disjoint, with pathology-backed labels and adequate patient counts
   determined before evaluation. Prefer an independent institution/device.
4. For gate negatives, use only individually rights-checked natural/object
   assets, or resolve source-specific rights and grouping for FOCUS-7D and
   other-medical candidates. Keep each negative category separately
   traceable. Obtain genuine screenshots and naturally low-quality scans
   only through the institution's approved collection process.

No source in this review is currently eligible for a patient-independent
final evaluation. No model was trained or evaluated; no partitions or
thresholds were created or tuned; no app, runtime configuration, or
deployment was changed.
