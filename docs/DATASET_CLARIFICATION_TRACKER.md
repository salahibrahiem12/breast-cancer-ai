# Dataset clarification response tracker

**Prepared:** 2026-11-01  
**Status:** All source responses and decisions are **Pending**. This tracker
does not grant permission, establish data suitability, or change any
disposition in the [dataset register](DATASET_REGISTER.md).

Use this tracker with the detailed questions and unsent drafts in
[DATASET_CLARIFICATION_PACKAGE.md](DATASET_CLARIFICATION_PACKAGE.md). For
each source, record the actual response or a link to it; do not convert an
assumption, an unanswered email, or an inaccessible page into evidence.

## How to update

1. Keep every unanswered response marked **Pending**. When answered, record
   the date, concise response, evidence location/link, and reviewer notes.
2. Retain the underlying reply, current license/terms and version, official
   metadata/data dictionary, access agreement, approval evidence, and any
   split/duplicate verification supplied by the source. Store restricted
   records only in an authorized location; link to them here without copying
   sensitive content into the repository.
3. Decide **legal permission** and **evaluation suitability** separately.
   A license alone does not establish patient-level evaluation suitability;
   patient IDs alone do not establish permission.
4. Do not change any disposition in this tracker or the register until
   supporting evidence is recorded, reviewed, and sufficient for the
   applicable criteria below. Preserve unresolved or conflicting terms as
   blocked and request an authorized clarification.
5. Changing a tracker decision does not itself authorize acquisition,
   partitioning, training, evaluation, deployment, or public classification.
   Obtain any required institutional approval first.

## Decision framework

Record each decision independently:

- **Legal permission**
  - **Pending:** no adequate evidence reviewed.
  - **Confirmed for defined scope:** authoritative current terms or written
    confirmation from an authorized rights/access authority covers the exact
    proposed use and output scope.
  - **Blocked / excluded:** terms conflict, do not cover the proposed use,
    underlying image rights are unclear, or required permission is denied.
- **Limited development use**
  - Mark eligible only when legal permission is confirmed for a specific
    non-clinical development use; source/version and provenance are
    traceable; labels and use limitations are documented; attribution,
    access, security, and retention conditions can be met; and any required
    institutional approvals are in place.
  - This decision is not evidence of patient-independent performance and
    does not approve public demos, distribution, or commercial use unless
    those uses are separately confirmed.
- **Patient/source-independent evaluation**
  - Mark potentially eligible only after limited-use permission is resolved
    for evaluation and all applicable evidence below is documented:
    1. Stable, privacy-preserving image-to-patient and image-to-exam/source
       mappings exist and can be used to keep related images together.
    2. Train, validation/threshold-selection, and untouched final-test
       partitions are distinct at the patient/source level. Any published
       split must state its grouping unit; patient disjointness cannot be
       inferred from fold names or image counts.
    3. Exact and near-duplicate checks, repeat exams, derived crops/screenshots,
       and known cross-source overlaps are checked across partitions; method,
       findings, and unresolved limitations are recorded.
    4. Site/device identifiers are available and documented where provided;
       site/device separation is used where feasible, and its absence or
       single-site scope is explicitly recorded.
    5. Labels, label provenance, reference standards, class/category meaning,
       missing/uncertain labels, and known limitations are defined and fit
       the proposed endpoint. Image totals alone do not demonstrate adequate
       independent sample size; define precision/CI needs before evaluation.
    6. The final cohort is genuinely untouched by model/threshold selection,
       and its source is independent for the intended claim. Report each
       required negative category separately where relevant.
  - This is a data-readiness decision only, not a result or clinical claim.
- **Excluded or still blocked**
  - Leave blocked when permission, provenance, patient/source grouping,
    suitable labels, split evidence, or required approvals are absent or
    conflicting. Exclude from the proposed use when an authoritative source
    disallows it or a critical gap cannot reasonably be resolved. State the
    evidence and scope; do not silently infer or widen permission.

## 1. BUS-BRA

Source records: [Zenodo v1.0 record 8231412](https://zenodo.org/records/8231412),
[metadata API](https://zenodo.org/api/records/8231412), [paper abstract](https://pubmed.ncbi.nlm.nih.gov/37937827/).
Source-specific prompts are in the BUS-BRA section of the
[clarification package](DATASET_CLARIFICATION_PACKAGE.md#1-bus-bra).

### Questions and responses

| ID | Question requiring an answer | Response |
|---|---|---|
| BB-1 | Does CC BY 4.0 cover every v1.0 image and annotation, and does the paper’s “academic and research purposes” language add a non-commercial or other restriction? Which terms control? | **Pending** |
| BB-2 | Are research model development, retained/redistributed images, derived weights, publications, public demos/hosted inference, and commercial use each permitted, restricted, or subject to separate approval? | **Pending** |
| BB-3 | Do the released CSVs map each image/annotation to stable privacy-preserving patient, exam, scanner, and fold IDs? Provide the data dictionary and identify absent fields. | **Pending** |
| BB-4 | Are the supplied 5-/10-fold partitions patient-disjoint? What grouping unit and verification method were used? | **Pending** |
| BB-5 | Were repeat exams, exact/near duplicates, and derived crops checked across folds? Provide method, findings, and limitations. | **Pending** |
| BB-6 | What site, scanner make/model, acquisition, probe/view, and protocol metadata are available at an allowed granularity? | **Pending** |
| BB-7 | Define biopsy labels, BI-RADS labels, normal/non-lesion regions, annotation authorship, reference standards, uncertain cases, and known limitations. Are normal regions labeled exams or only regions within cases? | **Pending** |
| BB-8 | What exact dataset, paper, license, author, and image-level attribution/citation wording and change notices are required? | **Pending** |
| BB-9 | What access, application, ethics/IRB, institutional, data-protection, agreement, retention, or other approval steps are required? | **Pending** |
| BB-10 | Is the respondent authorized to clarify rights and data terms? Identify the controlling record/version and provide written confirmation or an authoritative link for each proposed use. | **Pending** |

### Evidence and review

- **Evidence to retain:** authorized written clarification of CC BY/paper
  relationship; current license/terms; Zenodo record/version; CSV/data
  dictionary; patient/exam/scanner/fold mapping description; split and
  duplicate verification; label/reference-standard documentation; required
  citation; and approval/access records.
- **Legal permission decision:** **Pending** — do not infer model-use or
  output rights from the Zenodo license while the paper wording is unresolved.
- **Limited development decision:** **Pending** — requires confirmed use
  scope, traceable version/provenance, adequate labels, and required
  approvals.
- **Evaluation suitability decision:** **Pending** — requires image-to-
  patient/exam mapping, patient-disjoint partition evidence, duplicate/repeat
  handling, scanner mapping where available, suitable labels, and a
  preplanned untouched evaluation cohort. One institution must be stated as
  a generalization limitation.
- **Date reviewed:** Pending
- **Reviewer notes:** Pending

## 2. BUS-UCLM

Source records: [Mendeley version 3](https://data.mendeley.com/datasets/7fvgj4jsp7/3),
[DataCite DOI 10.17632/7fvgj4jsp7.3](https://api.datacite.org/dois/10.17632/7fvgj4jsp7.3),
and the 2025 [Scientific Data paper](https://doi.org/10.1038/s41597-025-04562-3).
The earlier version 1 record is
[10.17632/7fvgj4jsp7.1](https://api.datacite.org/dois/10.17632/7fvgj4jsp7.1).
Source-specific prompts are in the BUS-UCLM section of the
[clarification package](DATASET_CLARIFICATION_PACKAGE.md#2-bus-uclm).

### Questions and responses

| ID | Question requiring an answer | Response |
|---|---|---|
| UCLM-1 | Is version 3 currently downloadable through an official route, and can the source provide the exact archive/version and its file manifest? | **Pending** |
| UCLM-2 | Confirm that CC BY 4.0 applies to every v3 image and mask, including any exceptions; identify whether model weights, publications, demos, hosted inference, and commercial use are permitted. | **Pending** |
| UCLM-3 | Reconcile version-specific dataset terms with the separate article license and any institutional or data-protection restrictions. | **Pending** |
| UCLM-4 | Confirm the paper-described four-letter patient keys are stable across all released images and masks, and clarify whether the sequence suffix groups exams or only images within a study. Are repeat exams identifiable? | **Pending** |
| UCLM-5 | Provide the five-fold patient-level assignments used in the paper, confirm the grouping unit, and document duplicate/repeat checks across folds. | **Pending** |
| UCLM-6 | What site, Siemens ACUSON S2000/device, acquisition, probe/view, and protocol fields are available, and what metadata is absent? | **Pending** |
| UCLM-7 | Define benign, malignant, and normal labels; distinguish normal exams from non-lesion/background pixels; identify annotators/reference standards, uncertainty, and known limitations. | **Pending** |
| UCLM-8 | What exact attribution, citation, license notices, and change notices are required? | **Pending** |
| UCLM-9 | What access, application, ethics/IRB, institutional, data-protection, agreement, retention, or non-commercial-use conditions apply? | **Pending** |
| UCLM-10 | Is the respondent authorized to clarify current access and rights? Identify the controlling terms/version and provide written confirmation or official links. | **Pending** |

### Evidence and review

- **Evidence to retain:** current working official record/access instructions;
  license and full terms; file/data dictionary; patient/exam/image mapping;
  label and reference-standard documentation; split and duplicate evidence;
  attribution wording; and required approvals.
- **Legal permission decision:** **Pending** — the v3 DataCite record and
  paper's dataset record section state CC BY 4.0, while the article has a
  separate CC BY-NC-ND license. Retain confirmation of any exceptions for
  images/masks and model outputs.
- **Limited development decision:** **Pending** — requires available
  authorized files, confirmed intended use, source traceability, documented
  labels, and approvals.
- **Evaluation suitability decision:** **Pending** — requires image-to-
  patient/exam mapping, patient-grouped split/duplicate evidence, acquisition
  metadata where available, and sufficient independently held-out patients.
  The reported 38 patients are not presumed sufficient as a final test set.
- **Date reviewed:** 2026-11-02
- **Reviewer notes:** Desk research (not an owner response) found that the
  2025 paper describes 683 PNG images from 38 patients, four-letter
  pseudonymous patient keys, a single hospital and Siemens ACUSON S2000
  system, biopsy confirmation for lesion malignancy, informed consent, and
  patient-level five-fold partitions in a comparative experiment. The v3
  Mendeley page showed no file listing; the author-linked download URL
  returned HTTP 403 to both HEAD and a one-byte Range GET. No data were
  acquired. Source responses and legal/evaluation decisions remain Pending;
  38 patients are not presumed sufficient for a fresh final test cohort.

## 3. BUS_UC

Source records: [Mendeley version 1](https://data.mendeley.com/datasets/3ksd7w7jkx/1),
[DataCite metadata](https://api.datacite.org/dois/10.17632/3ksd7w7jkx.1).
Source-specific prompts are in the BUS_UC section of the
[clarification package](DATASET_CLARIFICATION_PACKAGE.md#3-bus_uc).

### Questions and responses

| ID | Question requiring an answer | Response |
|---|---|---|
| UC-1 | Did the depositor have authority from ultrasoundcases.info and relevant rightsholders to redistribute every underlying image under CC BY 4.0? Identify third-party restrictions or exceptions. | **Pending** |
| UC-2 | Does the repository license cover all images, masks, annotations, and any source material? Do paper, upstream, or institutional terms add conditions? | **Pending** |
| UC-3 | Are research model development, retaining/redistributing images, derived weights, publications, public demos/hosted inference, and commercial use permitted or separately approved? | **Pending** |
| UC-4 | Can each released image be linked to its upstream case/source and a stable privacy-preserving patient/exam group? Are repeat cases identifiable? | **Pending** |
| UC-5 | Are any source-defined splits patient-disjoint? What grouping and verification method was used? Were exact/near duplicates or repeat exams checked across splits and source cases? | **Pending** |
| UC-6 | What institution/site, scanner/device, acquisition, probe/view, and protocol metadata are available? Which fields are missing? | **Pending** |
| UC-7 | Who assigned benign/malignant labels, what clinical reference standard was used, how were disagreements handled, and what label limitations or missing normal class apply? | **Pending** |
| UC-8 | What exact attribution/citation, upstream source credit, license notice, and change notices are required? | **Pending** |
| UC-9 | What access, ethics/IRB, institutional, data-protection, agreement, or retention approvals are required from the depositor or upstream source? | **Pending** |
| UC-10 | Is the respondent authorized to confirm both repository and underlying-image rights? Identify the controlling terms/version and provide written confirmation or authoritative links. | **Pending** |

### Evidence and review

- **Evidence to retain:** written authority for upstream-image use;
  ultrasoundcases.info terms or rights evidence; current repository license;
  image/source and patient/exam mapping; label/reference-standard
  documentation; split/duplicate evidence; required attribution; and
  approvals.
- **Legal permission decision:** **Pending** — the repository CC BY notice
  alone is insufficient to establish rights in third-party source images.
- **Limited development decision:** **Pending** — requires verified
  underlying-image permission as well as defined model/output scope and
  source provenance.
- **Evaluation suitability decision:** **Pending** — requires stable
  image-to-patient/source mapping, patient-disjoint partitions, repeat and
  duplicate checks, documented acquisition/site/device fields where
  available, and appropriate labels. No patient count is currently
  established.
- **Date reviewed:** Pending
- **Reviewer notes:** Pending

## 4. MIMIC-CXR v2.1.0

Source records: [PhysioNet v2.1.0](https://physionet.org/content/mimic-cxr/2.1.0/)
and [database paper](https://doi.org/10.1038/s41597-019-0322-0).
Source-specific prompts are in the MIMIC-CXR section of the
[clarification package](DATASET_CLARIFICATION_PACKAGE.md#4-mimic-cxr-v210).

### Questions and responses

| ID | Question requiring an answer | Response |
|---|---|---|
| MIMIC-1 | Do current dataset terms permit the exact non-clinical eligibility research and model-development use, including training, validation, and evaluation? | **Pending** |
| MIMIC-2 | Do PhysioNet policies, paper terms, institutional rules, or the data-use agreement add or control restrictions? Identify the current controlling versions. | **Pending** |
| MIMIC-3 | Are source-image retention/redistribution, derived weights, publications, aggregate outputs, public demos/hosted inference, and commercial use permitted or restricted? | **Pending** |
| MIMIC-4 | Which authorized files expose stable subject, study, and image IDs, and how should related views/studies be grouped without re-identification? | **Pending** |
| MIMIC-5 | What split unit is used in supplied partitions? Are patient groups disjoint, and are repeat studies, exact/near duplicates, and leakage across splits checked? | **Pending** |
| MIMIC-6 | Which site/device and acquisition metadata are available or suppressed? State the limits of this single-institution chest-radiography source for the intended negative category. | **Pending** |
| MIMIC-7 | Which labels are report-derived, uncertain, or radiologist-validated? Define reference standards, missing labels, and known limitations. | **Pending** |
| MIMIC-8 | What exact PhysioNet/database/paper citations, attribution, and acknowledgements are required? | **Pending** |
| MIMIC-9 | What current credentialing, training, application, ethics/IRB determination, institutional sign-off, DUA, data-protection review, fees, and retention terms apply? | **Pending** |
| MIMIC-10 | Which PhysioNet authority can confirm the proposed uses? Provide official documentation or written confirmation and the required application route. | **Pending** |

### Evidence and review

- **Evidence to retain:** current PhysioNet access requirements, DUA and
  applicable terms; credentialing/training and institutional approval
  records; official data dictionary and ID/split documentation; label
  references; attribution wording; and authorized access decision.
- **Legal permission decision:** **Pending** — no access or use until required
  credentials, approvals, and terms are satisfied.
- **Limited development decision:** **Pending** — requires authorization for
  this project and explicit derived-model/output scope. Chest radiographs
  would only serve the documented unrelated-medical-image negative role.
- **Evaluation suitability decision:** **Pending** — requires subject-to-
  study/image grouping, patient-disjoint partitions and duplicate/repeat
  checks, label suitability, and documented metadata. Single-institution
  scope must be disclosed; this source cannot establish breast-ultrasound
  positive or multi-site ultrasound performance.
- **Date reviewed:** Pending
- **Reviewer notes:** Pending

## 5. CAMUS

Source: [official CAMUS challenge page](https://www.creatis.insa-lyon.fr/Challenge/camus/)
and [2019 paper](https://doi.org/10.1109/TMI.2019.2900516).
Source-specific prompts are in the CAMUS section of the
[clarification package](DATASET_CLARIFICATION_PACKAGE.md#5-camus).

### Questions and responses

| ID | Question requiring an answer | Response |
|---|---|---|
| CAMUS-1 | What current license/access agreement governs the image sequences and annotations, and does it permit the intended research/model development? | **Pending** |
| CAMUS-2 | Do challenge, paper, institutional, or portal terms add restrictions? Which terms control and what version/date applies? | **Pending** |
| CAMUS-3 | Are source images, derived weights, publications, public demos/hosted inference, and commercial use permitted, restricted, or separately approved? | **Pending** |
| CAMUS-4 | Does each sequence map to a stable patient ID? Can all views/sequences/exams from a patient be grouped without identity disclosure? | **Pending** |
| CAMUS-5 | Are any supplied/recommended splits patient-disjoint? How was this verified? Were repeat exams and exact/near duplicates checked across splits? | **Pending** |
| CAMUS-6 | What site, scanner/device, view, acquisition, and protocol metadata are available, and which are absent? | **Pending** |
| CAMUS-7 | Define sequence and annotation labels, annotator/reference standards, uncertain cases, and limitations for use as other-anatomy ultrasound negatives. | **Pending** |
| CAMUS-8 | What exact dataset/paper citations, acknowledgements, license notices, and change notices are required? | **Pending** |
| CAMUS-9 | What portal access, application, ethics/IRB, institutional, data-protection, training, or agreement steps are required? | **Pending** |
| CAMUS-10 | Which data authority can confirm the terms and data structure? Provide official documentation or written answers for each proposed use. | **Pending** |

### Evidence and review

- **Evidence to retain:** current portal terms/license and access decision;
  data dictionary; patient/sequence/view mapping; split and duplicate
  verification; acquisition/device metadata; annotation documentation;
  required citation; and approval records.
- **Legal permission decision:** **Pending** — the challenge-page citation
  requirement is not by itself a data-use license.
- **Limited development decision:** **Pending** — requires confirmed
  sequence/annotation rights, permitted model/output scope, official access,
  and required approvals.
- **Evaluation suitability decision:** **Pending** — requires stable
  patient-to-sequence mapping, patient-disjoint grouping, duplicate/repeat
  checks, relevant metadata, and a preplanned independent cohort. CAMUS
  represents echocardiography only, not breast ultrasound.
- **Date reviewed:** Pending
- **Reviewer notes:** Pending

## 6. Fetal-head ultrasound v2

Source records: [Zenodo record 8265464](https://zenodo.org/records/8265464),
[included README](https://zenodo.org/records/8265464/files/Readme.txt?download=1),
and [dataset paper](https://doi.org/10.1016/j.dib.2023.109708).
Source-specific prompts are in the fetal-ultrasound section of the
[clarification package](DATASET_CLARIFICATION_PACKAGE.md#6-fetal-head-ultrasound-v2).

### Questions and responses

| ID | Question requiring an answer | Response |
|---|---|---|
| FETAL-1 | Does CC BY 4.0 cover every image, annotation, and third-party component, and does it permit this research/model-development use? | **Pending** |
| FETAL-2 | Does the paper, README, original-source agreement, or other policy add terms beyond the Zenodo record? Which terms control? | **Pending** |
| FETAL-3 | Are images/annotations redistributable, and are derived weights, publication examples, public demos/hosted inference, or commercial use restricted? | **Pending** |
| FETAL-4 | Can each image be mapped to its original source/author and a stable privacy-preserving patient or acquisition group? | **Pending** |
| FETAL-5 | Are any suggested splits patient/source-disjoint? What grouping unit and verification were used? Were repeated frames and exact/near duplicates checked across splits? | **Pending** |
| FETAL-6 | What acquisition institution/site, scanner/device, date/period, view, and protocol metadata are available or absent? | **Pending** |
| FETAL-7 | Define image and annotation labels, annotators/reference standards, uncertainty, image-quality limitations, and scope limits when used as fetal rather than breast ultrasound. | **Pending** |
| FETAL-8 | What exact per-image original-author credits, dataset/paper citations, license notices, and change notices are required? | **Pending** |
| FETAL-9 | What access, ethics/IRB, institutional, privacy, or other approvals are required for use or linkage metadata? | **Pending** |
| FETAL-10 | Which authorized contributor can confirm the rights, attribution mapping, and metadata? Provide written confirmation or an authoritative manifest/terms link. | **Pending** |

### Evidence and review

- **Evidence to retain:** current Zenodo record/files and license; README and
  controlling terms; per-image source/author mapping; patient/acquisition
  grouping documentation; split and duplicate evidence; label/annotation
  documentation; exact attribution instructions; and required approvals.
- **Legal permission decision:** **Pending** — resolve the README’s
  original-author attribution requirement and third-party coverage before
  relying on the repository-level CC BY statement.
- **Limited development decision:** **Pending** — requires satisfiable
  attribution, confirmed image/annotation rights and model-use scope,
  traceable provenance, documented labels, and approvals.
- **Evaluation suitability decision:** **Pending** — requires stable
  patient/source grouping, split and duplicate verification, appropriate
  labels, and independent sample groups. Fetal anatomy is only a negative
  category candidate and cannot serve as breast-ultrasound positive evidence.
- **Date reviewed:** Pending
- **Reviewer notes:** Pending

## Decision record

| Source | Legal permission | Limited development use | Patient/source-independent evaluation | Date reviewed | Reviewer notes |
|---|---|---|---|---|---|
| BUS-BRA | Pending | Pending | Pending | Pending | Pending |
| BUS-UCLM | Pending | Pending | Pending | Pending | Pending |
| BUS_UC | Pending | Pending | Pending | Pending | Pending |
| MIMIC-CXR v2.1.0 | Pending | Pending | Pending | Pending | Pending |
| CAMUS | Pending | Pending | Pending | Pending | Pending |
| Fetal-head ultrasound v2 | Pending | Pending | Pending | Pending | Pending |
