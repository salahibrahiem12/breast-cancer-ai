# Dataset clarification and access-request package

**Prepared:** 2026-11-01  
**Purpose:** Draft questions and communications for rights, access, and
metadata review before any new dataset acquisition or use.  
**Status:** Planning drafts only. No message was sent, application submitted,
or data downloaded. This package is not legal advice, ethics approval, or
authorization to use data.

## How to use this package

For each source, send the draft only to a recipient verified from the
authoritative source or institutional process. Leave recipient and sender
fields blank until verified. Include the relevant source record and request a
written response or an authoritative link to applicable terms. Do not send
patient data in the request.

The **common question set** applies separately to every source below. Pair it
with that source’s specific questions and draft. A response about one source
does not establish permission or metadata availability for another.

## Common clarification questionnaire

Request explicit answers for each item; “open access” alone does not resolve
the questions.

1. **Permitted purpose:** Does the current dataset license, access agreement,
   and any source-specific policy permit this project’s non-clinical research
   and machine-learning model development, including training, validation,
   and evaluation? Are there restrictions on automated analysis?
2. **Conflicting terms:** Does the associated paper, repository record,
   institutional policy, upstream image source, or another agreement add
   restrictions beyond the dataset’s stated license? Which document controls
   if terms differ?
3. **Outputs and redistribution:** May the project retain or redistribute
   source images? May it publish or demonstrate outputs using images? Are
   trained or fine-tuned model weights, embeddings, or other derivatives
   considered restricted derivatives? Are public demos, hosted inference, or
   commercial use prohibited or subject to separate permission?
4. **Patient/image grouping:** Is a stable, privacy-preserving patient-to-
   image mapping available, or can the data steward provide group IDs that
   keep a patient’s images together without disclosing identity? Are repeat
   exams or multiple images per exam identifiable?
5. **Splits and leakage:** If source-defined folds/splits are supplied, what
   unit was split (patient, exam, lesion, or image), and how was patient
   separation verified? Were exact/near-duplicate images or repeat exams
   checked across splits? Can the verification method and any overlap
   limitations be documented?
6. **Acquisition metadata:** Are site, scanner/device make and model,
   acquisition date or period, probe, view, and relevant protocol metadata
   available at a privacy-safe level? Which fields are absent or suppressed?
7. **Labels and references:** What does each label mean, who assigned it,
   and what reference standard was used (for example pathology, radiologist
   consensus, report-derived label, or technical quality review)? How are
   uncertain, missing, normal, repeat, and technically inadequate cases
   represented? What known label limitations should be cited?
8. **Required credit:** What exact attribution, citation, license notice,
   acknowledgement, and change notice are required for data and resulting
   publications or demos? Are there source-author or third-party credits
   required at image level?
9. **Access process:** What account, application, training, data-use
   agreement, ethics review, institutional approval, or data-protection
   review is required before access or use? Are there costs, renewal terms,
   geographic restrictions, or limits on retention?
10. **Written confirmation:** Who is authorized to clarify rights and data
    terms? Please identify the controlling terms/version and confirm which
    proposed uses are permitted, prohibited, or require separate approval.

## Source-specific clarification questions and draft communications

### 1. BUS-BRA

**Records:** [Zenodo v1.0 record 8231412](https://zenodo.org/records/8231412),
[record metadata](https://zenodo.org/api/records/8231412), and
[dataset paper abstract](https://pubmed.ncbi.nlm.nih.gov/37937827/).

**Resolve before acquisition:** The record declares CC BY 4.0, identifies
PEB/COPPE-UFRJ as copyright holder/distributor, and requires citation; the
paper describes access for “academic and research purposes.” Determine
whether those statements conflict or narrow intended use. Verify actual CSV
fields, patient and scanner mappings, and split unit without assuming the
published folds are patient-disjoint.

**Additional questions**

- Does CC BY 4.0 govern all image files and annotations in v1.0, and does
  “academic and research purposes” impose an additional non-commercial or
  other limitation?
- Do the CSVs link each image and lesion annotation to a stable patient,
  exam, scanner, and source-defined fold? Can a data dictionary be shared?
- Are 5-/10-fold partitions grouped by patient, and were repeat exams and
  duplicate/near-duplicate images checked across folds? What patient-level
  grouping or overlap evidence can be provided?
- Are normal/non-lesion regions equivalent to labeled normal breast exams,
  or are the labels limited to regions within cases?
- What approvals or permissions are needed for model weights, public
  demonstration, publication figures, or commercial use?

**Draft**

> **To:**  
> **From:**  
> **Subject:** Clarification request: BUS-BRA v1.0 research and data structure
>
> Dear BUS-BRA data steward,
>
> I am preparing a research project on image eligibility and breast-ultrasound
> analysis and am reviewing BUS-BRA v1.0 before considering any access or use.
> I have not downloaded or incorporated the dataset. Could you please answer
> the attached common clarification questionnaire and the BUS-BRA-specific
> questions in the accompanying request?
>
> In particular, I would appreciate clarification on how the Zenodo CC BY 4.0
> record relates to the paper’s “academic and research purposes” wording,
> whether model development and release/demo of derived weights or outputs
> are permitted, and whether the supplied folds are patient-disjoint. Please
> also indicate which CSV fields link images to stable patient, exam, and
> scanner groups, and the required attribution and approval process.
>
> A written response or authoritative terms/documentation link would help us
> determine whether the dataset is appropriate to request. We will not infer
> permission from public availability alone.
>
> Thank you,  
> 

### 2. BUS-UCLM

**Records:** [Mendeley version 3](https://data.mendeley.com/datasets/7fvgj4jsp7/3),
[DataCite DOI 10.17632/7fvgj4jsp7.3](https://api.datacite.org/dois/10.17632/7fvgj4jsp7.3),
and the 2025 [Scientific Data paper](https://doi.org/10.1038/s41597-025-04562-3).

**Resolve before acquisition:** the v3 record is published and states CC BY
4.0; the paper reports 683 images from 38 patients and documents a
pseudonymous patient-key pattern. However, the page did not show a file
listing, and the author-linked download URL returned HTTP 403 to HEAD and a
one-byte range request on 2026-11-02. Confirm a working official access route,
that v3 terms cover all images/masks, and that patient and fold metadata are
available. The article itself has a separate CC BY-NC-ND license. The
38-patient single-site cohort is not sufficient by itself for a fresh final
test.

**Additional questions**

- Is v3 currently downloadable through an official route, and what are the
  archive filename, version, and file manifest?
- Does CC BY 4.0 apply to every image and mask, and are any exceptions or
  institutional restrictions present? Are model weights, publications,
  public demos/hosted inference, and commercial use permitted?
- Are the paper-described four-letter patient keys stable across all images
  and masks, and does each sequence suffix identify an image within an exam
  or a patient study? Are repeat exams identifiable?
- Can the five-fold patient-level assignments described in the paper be
  provided, with the exact grouping unit and verification method? Were
  repeats and duplicates checked across folds?
- Are scanner/device identifiers available beyond the reported Siemens
  ACUSON S2000 and 18L6 HD probe?
- How were benign, malignant, and normal labels established, and do the
  annotations distinguish a normal exam from non-lesion/background pixels?

**Draft**

> **To:**  
> **From:**  
> **Subject:** Access and metadata clarification: BUS-UCLM version 3
>
> Dear BUS-UCLM dataset contact,
>
> I am assessing BUS-UCLM for a research data-readiness project and have not
> downloaded or used the files. The v3 Mendeley record states CC BY 4.0, and
> the 2025 paper describes 683 images from 38 patients, pseudonymous
> patient-group keys, and patient-level five-fold partitions. The v3 page did
> not show files, and the author-linked download URL returned HTTP 403 when
> checked on 2026-11-02.
>
> Could you identify a working official access route and confirm the
> applicable terms for research model development, derived weights,
> publications/demonstrations, and any commercial use? Please also confirm
> that the v3 license covers all images and masks, and clarify whether stable
> patient/exam-to-image identifiers, repeat-exam links, scanner metadata, and
> the five-fold patient-grouped split assignments are available.
> The attached common questionnaire lists the remaining rights, labels,
> attribution, and approval questions.
>
> A link to the current data dictionary, license, or access agreement would
> be appreciated. We will not rely on the metadata record alone as proof of
> file-level rights or patient grouping.
>
> Thank you,  
> 

### 3. BUS_UC

**Records:** [Mendeley version 1](https://data.mendeley.com/datasets/3ksd7w7jkx/1)
and [DataCite metadata](https://api.datacite.org/dois/10.17632/3ksd7w7jkx.1).

**Resolve before acquisition:** The repository declares CC BY 4.0, but its
description says images were obtained from ultrasoundcases.info. Confirm
authority to license those upstream clinical images and whether stable
patient/exam IDs and source/site/device fields exist.

**Additional questions**

- Did the depositor have permission from ultrasoundcases.info and the
  relevant image rightsholders to redistribute the images under CC BY 4.0
  and permit model development? Are any images third-party-restricted?
- Can the upstream source pages or original case identifiers be mapped to
  each released image, and do they reveal repeats or duplicates?
- Are labels original source labels or new radiologist annotations? Who
  annotated them, what reference standard was used, and were disagreements
  adjudicated?
- Is a stable pseudonymous patient or exam grouping available? Are site,
  scanner, acquisition, or split identifiers available?

**Draft**

> **To:**  
> **From:**  
> **Subject:** Rights and provenance clarification: BUS_UC version 1
>
> Dear BUS_UC dataset contact,
>
> I am reviewing the BUS_UC version 1 record for possible research use and
> have not downloaded or incorporated the dataset. The record lists CC BY
> 4.0 and states that the images came from ultrasoundcases.info.
>
> Before considering access, could you confirm that the depositor had
> authority to license each underlying clinical image for redistribution and
> machine-learning development, and explain whether any upstream terms
> restrict derived models, public demonstrations, publication, or commercial
> use? Please also clarify whether image-level source case IDs can be linked
> to pseudonymous patient/exam groups, whether repeat exams or duplicates
> exist, and how the benign/malignant labels were established.
>
> Please see the attached common questionnaire for the additional split,
> acquisition metadata, attribution, and approval questions. A current
> data dictionary and authoritative terms link would be helpful.
>
> Thank you,  
> 

### 4. MIMIC-CXR v2.1.0

**Records:** [PhysioNet v2.1.0 record](https://physionet.org/content/mimic-cxr/2.1.0/)
and [database paper](https://doi.org/10.1038/s41597-019-0322-0).

**Resolve before access:** Follow the current PhysioNet credentialing and
data-use process; do not request or receive restricted files by email.
Confirm that the intended research/model-development workflow is allowed
under current terms, including model outputs/weights and public demo
restrictions. The dataset is chest radiography, not ultrasound.

**Additional questions**

- Which current credentialing, training, institutional sign-off, IRB/ethics
  determination, and data-use agreement steps apply to this project?
- Do the current terms permit training a separate image-eligibility model
  and retaining derived model parameters? Are public hosted demos, outputs,
  or commercial use allowed, disallowed, or separately reviewable?
- Which files provide the current subject, study, and image identifiers?
  Can those IDs be used to group all views/studies from the same person
  without attempting re-identification?
- What site/device or acquisition fields are available, and which are
  suppressed? Are split definitions and duplicate/repeat-study checks
  documented?
- If labels are used, which labels are report-derived, uncertain, or
  radiologist-validated, and what limitations should be carried into an
  evaluation?

**Draft access statement**

> **To:** PhysioNet access authority / MIMIC-CXR data steward  
> **From:**  
> **Subject:** Request for guidance on MIMIC-CXR v2.1.0 eligibility research
>
> Dear PhysioNet access team,
>
> I am reviewing the authorized-access requirements for MIMIC-CXR v2.1.0 as
> a potential source of unrelated medical-image negatives in a research
> image-eligibility project. I have not applied for access or downloaded
> data. I understand this dataset is controlled access and will use only
> the official credentialing and agreement process.
>
> Please direct me to the current requirements for credentialing, training,
> institutional approval, ethics/IRB determination, and the applicable
> data-use agreement. Before applying, I would also appreciate confirmation
> whether the intended non-clinical model-development use, retention of
> derived weights, publication of aggregate results, public demonstration,
> and any commercial use are permitted or require separate approval.
>
> Could you point me to documentation for subject/study/image grouping,
> available acquisition/site/device fields, split construction, duplicate
> checks, and label limitations? The attached common questionnaire captures
> the full set of rights, metadata, citation, and approval questions.
>
> Thank you,  
> 

### 5. CAMUS

**Record:** [official CAMUS challenge page](https://www.creatis.insa-lyon.fr/Challenge/camus/)
and the linked [2019 paper](https://doi.org/10.1109/TMI.2019.2900516).

**Resolve before access:** Confirm current portal access conditions and
license; the challenge page requires citation but does not itself settle
data-use rights or confirm the released patient mapping.

**Additional questions**

- What license or click-through agreement governs the image sequences and
  annotations, and does it permit model development, derived weights,
  public demo, publication figures, or commercial use?
- Does each sequence/file map to a stable patient ID? Are multiple views,
  studies, or sequences from the same patient identifiable and grouped in
  any recommended split?
- Are site, scanner/device, acquisition, and image-quality metadata
  available? Were duplicates or repeat exams checked between splits?
- What do the annotations label, who created them, and what limitations
  apply if the images are used only as non-breast ultrasound examples?

**Draft**

> **To:**  
> **From:**  
> **Subject:** CAMUS dataset terms and patient-grouping clarification
>
> Dear CAMUS data steward,
>
> I am assessing the CAMUS echocardiography dataset as a possible source of
> other-anatomy ultrasound images for research on image eligibility. I have
> not accessed the download portal or obtained any files.
>
> Could you point me to the current license/access agreement and confirm
> whether it permits research model development, retention of derived
> weights, publications, public demos, and commercial use? Please also
> clarify whether the released files expose stable patient-level grouping,
> whether all sequences/views from a patient can be kept together, and
> whether site/device/acquisition metadata and split-leakage checks are
> documented.
>
> The attached common questionnaire includes questions about label
> definitions, attribution, and any required institutional or ethics
> approvals. Please identify the required process before any access request.
>
> Thank you,  
> 

### 6. Fetal-head ultrasound v2

**Records:** [Zenodo record 8265464](https://zenodo.org/records/8265464),
[included README](https://zenodo.org/records/8265464/files/Readme.txt?download=1),
and [dataset paper](https://doi.org/10.1016/j.dib.2023.109708).

**Resolve before additional use:** The Zenodo record states CC BY 4.0 and
asks users to credit contributors; the included README requests credit to
original image authors. Existing inventory did not establish a per-image
mapping to those original authors or patient IDs. Clarify whether the
license and attribution can be satisfied for every image and whether usable
group IDs can be provided.

**Additional questions**

- Does CC BY 4.0 cover every image, annotation, and third-party component?
  How should users satisfy the README’s original-author credit when author
  attribution is not mapped per image?
- Can the steward provide a source/image manifest, original-author
  attribution, and stable patient or acquisition-group IDs without
  disclosing patient identities?
- Are repeated images/frames or patient-level duplicates represented, and
  are any published partitions grouped by patient?
- What labels/annotations were created by whom, what reference standard was
  used, and what limitations apply to using these images as fetal—not
  breast—ultrasound negatives?
- Are model development, derived weights, redistributed examples, public
  demo, and commercial use allowed under the record and any paper terms?

**Draft**

> **To:**  
> **From:**  
> **Subject:** Attribution, provenance, and grouping questions: fetal-head
> ultrasound v2
>
> Dear dataset contributor,
>
> I am reviewing Zenodo record 8265464 and its included README before any
> further use. I have not downloaded additional data. The record declares
> CC BY 4.0, while the README requests credit to original image authors;
> our existing inventory did not identify a per-image author mapping or
> patient identifiers.
>
> Could you confirm which terms govern the image files and annotations,
> provide the required attribution wording or per-image provenance mapping,
> and clarify whether stable patient/acquisition grouping IDs are available?
> Please also explain whether repeated images or patient overlap can occur
> across any suggested splits and whether model weights, public demos, or
> commercial use have additional restrictions.
>
> The attached common questionnaire includes remaining questions on label
> definitions, acquisition metadata, citation, and approval requirements.
> We will not assume that the repository license resolves third-party
> attribution or patient-grouping gaps.
>
> Thank you,  
> 

## Institutional collection checklist: screenshots and naturally poor scans

**Planning checklist only.** This is not ethics approval, legal advice,
consent, a waiver, or authorization to collect or use clinical data. Obtain
institutional privacy, security, legal, and ethics review before collection.
Do not capture identifiable clinical screens or transfer clinical images to
personal devices or unapproved services.

### Authorization and governance

- [ ] Identify the data controller, custodian, collection site, and approved
  research purpose; document institutional authorization and data-transfer
  terms.
- [ ] Obtain ethics/IRB review or a documented determination of exemption or
  non-human-subject status as applicable; do not self-declare exemption.
- [ ] Determine whether informed consent is required or whether an authorized
  waiver exists; document the approved scope, use, retention, and withdrawal
  conditions.
- [ ] For screenshots, obtain explicit authorization for the screen-capture
  method and source application/workstation. Confirm that capture does not
  violate clinical-system policy, vendor terms, or patient privacy rules.
- [ ] Define whether examples are for development, validation, or a locked
  test cohort before collection; prevent test examples from being reused
  during tuning.

### Privacy, de-identification, and secure handling

- [ ] Exclude direct identifiers from images, overlays, filenames, screen
  chrome, reports, and accompanying metadata. Review burned-in text and
  hidden metadata, not just filenames.
- [ ] Have an authorized privacy reviewer validate de-identification and
  assess re-identification risk, including dates, rare conditions, facility
  names, device overlays, and contextual details.
- [ ] Keep any re-identification key or linkage table only with the
  authorized custodian, separately from research images; use stable
  pseudonymous patient/source IDs for grouping where approved.
- [ ] Use institution-approved encrypted storage and transfer, least-
  privilege access, access logging, backup controls, retention limits, and
  secure deletion procedures.
- [ ] Document incident reporting, data correction/removal, retention
  expiry, and publication review procedures.

### Provenance, labels, and cohort integrity

- [ ] Record source institution, authorized source system, collection date
  at an approved granularity, image/exam provenance, modality, anatomy, and
  device/acquisition fields allowed by the protocol.
- [ ] Assign separate, reviewable categories: genuine screenshot; naturally
  low-quality scan; corrupted/unreadable clinical source; non-breast
  ultrasound anatomy; or other medical image. Do not label synthetic
  corruptions as real examples.
- [ ] Define “low quality” and “corrupted” before collection, with
  operational criteria (for example, clinically documented acquisition
  defect or unreadable image) and a qualified reviewer. Keep distinct
  artifact types and severity levels rather than collapsing them.
- [ ] Record who assigned each category, the adjudication process,
  disagreement/uncertainty, label version, and any reference standard.
- [ ] Maintain stable pseudonymous patient, exam, image, and source IDs;
  document repeated exams, multiple images per exam, derived screenshots,
  and related/duplicate frames so groups cannot cross partitions.
- [ ] Store consent/waiver and data-use evidence in a restricted governance
  location, not in the model-data directory. Keep a separate manifest of
  permitted uses, provenance, license/terms, and required attribution.
- [ ] Before any future evaluation, perform documented exact and
  near-duplicate checks, group by patient/source (and site/device where
  feasible), and reserve an untouched test set.
