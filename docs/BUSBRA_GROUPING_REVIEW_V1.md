# BUS-BRA grouping and access review

Reviewed 4 November 2026, before reserved-cohort predictions were available.

## Authoritative evidence

- Official Zenodo record/API, DOI 10.5281/zenodo.8231412, describes 1,064
  patients and declares CC BY 4.0. The local API snapshot is retained with
  its SHA-256 in the reservation record.
- The original paper, DOI 10.1002/mp.16812, accessible publisher abstract and
  Data Format and Usage Notes describe 1,875 images, 1,064 female patients,
  722 benign and 342 malignant cases, four scanners, biopsy-proven labels,
  and academic/research reuse. Its full-text route redirected to the abstract;
  Europe PMC and OpenAlex did not provide an open full-text copy.
- The authors' official GitHub README provides download/reproduction
  instructions but no explicit definition of the metadata Case field.
- The actual archive contains ID, Case, Histology, Pathology, BIRADS, Device,
  Width, Height, Side and BBOX fields. Case has 1,064 distinct values; image
  IDs are distinct. No Case has conflicting binary labels or crosses the
  published five-fold test assignment.

Sources:
https://zenodo.org/records/8231412
https://aapm.onlinelibrary.wiley.com/doi/abs/10.1002/mp.16812
https://pubmed.ncbi.nlm.nih.gov/37937827/
https://github.com/wgomezf/BUS-BRA

## Conclusion fixed before inference

Case is a verified source grouping key. Its equivalence to a stable patient,
breast, lesion or examination identifier is **not established** by the
accessible authoritative documentation. Matching counts and coherent folds
are not a substitute for an explicit data dictionary. No patient-independent
claim is made, and Side values are not used to infer anatomical independence.

Evaluate one mean probability per source Case, as locked before inference.
Report source-Case bootstrap intervals conditional on that grouping. Dependence
between distinct Cases belonging to one person cannot be ruled out, so those
intervals are not verified patient-level uncertainty. An authoritative mapping
or full data dictionary remains necessary for a patient-independent claim.

## Permission and cohort checks

This local offline research evaluation is consistent with both the dataset's
CC BY 4.0 statement and the paper's research-use description. Cite Gomez-Flores,
Gregorio-Calas and Coelho de Albuquerque Pereira, dataset and paper DOIs,
license, and derived changes. No clinical/public-serving authorization is
inferred. Images and source metadata remain local and Git-ignored.

The expanded check scanned 707 raw, 260 processed and 778 legacy BUSI-processed
images and included the full TCIA inventory: 1,544 distinct reference pixel
arrays after deduplication. Historical/unverified images were used for
inventory overlap checks only, never model fitting. One dHash distance-5
candidate was found in raw reference data (pHash distance 24; no exact byte
or pixel match). The predeclared conservative rule excluded its whole Case,
two benign images, before scoring. This is not confirmed duplication or proof
of patient overlap. No internal cross-Case dHash/pHash candidates at distance
<=5 were found; hash checks still cannot establish absence of all overlap.

Locked eligible cohort: 1,873 images / 1,063 source Cases, with 721 benign and
342 malignant Cases. No eligible Case had conflicting binary labels.
