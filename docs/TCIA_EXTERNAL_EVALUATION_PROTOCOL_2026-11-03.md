# Locked BrEaST research evaluation protocol

**Current status:** The run and subsequent pipeline/error audit are complete.
This is now an **inspected external evaluation set**, not an untouched final
cohort. See `TCIA_PIPELINE_AUDIT_2026-11-03.md` and `COHORT_USAGE_LEDGER.md`.
No threshold or model selection is permitted on this cohort's results.

Recorded before generating any BrEaST model predictions on 2026-11-03.

## Source and permitted scope

Use the original TCIA BREAST-LESIONS-USG version 1 downloads, updated
2024-01-08, DOI 10.7937/9WKK-Q141, under the collection's CC BY 4.0 terms.
The collection and original paper explicitly describe public research reuse.
Credit Pawlowska and colleagues and cite the dataset and paper
10.1038/s41597-024-02984-z. Source page:
https://www.cancerimagingarchive.net/collection/breast-lesions-usg/.

Archive SHA-256:
`c32e49cdfa54042e065582d0cb35978b5fc5e5b5e461b33bf773fb50cfdad355`.
Clinical workbook SHA-256:
`89a8874496a6f1f93960390b963f7369b90f7e72ee08a738f2d17e37f5eff8bf`.
Original files remain unchanged in the ignored local data directory.

## Prespecified analysis

- Use only the 197 cases whose source Verification is `confirmed by biopsy`
  and whose Classification is benign or malignant. Exclude the 55 follow-up
  cases and four normal cases before any predictions. This restricts inference
  to the biopsy-selected population and may introduce spectrum/selection bias.
- Select the already trained checkpoint
  `models_saved/versioned/mendeley_binary_resnet18_v1.pth`; do not compare or
  select alternative checkpoints on this cohort.
- Preserve its recorded benign/malignant class order, preprocessing, and
  validation-fitted temperature. Use malignant score >= 0.5 as already
  specified by the evaluation pipeline. Do not fit or tune anything on BrEaST.
- Assign every included case to the external `test` split. Keep each source
  CaseID as a namespaced patient group because the collection/paper document
  one scan per patient. No cross-source patient identity matching is possible.
- Run exact-byte and pixel checks and dHash/pHash candidate scans against all
  250 local Mendeley source images before evaluation. The inventory found zero
  exact matches and zero perceptual pairs at Hamming distance <= 3. This does
  not prove absence of all overlap or establish clinical independence.
- Report class counts, confusion matrix, accuracy, macro-F1, ROC/PR AUC,
  malignant sensitivity/specificity at 0.5, and calibration. Use the existing
  deterministic 1,000-iteration patient/duplicate-group bootstrap intervals.
- Do not use observed results for further model or threshold selection. Once
  evaluated, this cohort is no longer a fresh future final-test dataset.

## Interpretation and release

This is a retrospective external-source research evaluation, not clinical
validation. It does not evaluate an ultrasound eligibility gate, general
negative-image rejection, normal-class performance, prospective outcomes,
or broad device/site generalization. No pass threshold or deployment approval
is assigned. Public classification stays disabled regardless of results.
