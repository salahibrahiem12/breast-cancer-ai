# Acquired dataset duplicate-image audit

**Audit date:** 2026-11-01
**Scope:** local acquired archives only. No data were downloaded or modified;
no partitions, training, model evaluation, or threshold tuning were performed.

## Archive and inventory verification

| Dataset | Archive SHA-256 | Archive / inventory result |
|---|---|---|
| Mendeley Breast Ultrasound Image v1 | `87dbd3aaa89e2d8886839983be3a985b7b190d5d8cb167d95a8dcd5d224b63f7` | 3,264,338 bytes; ZIP CRC passed; all 250 archive images match the 250 extracted inventory files and recorded formats/dimensions. |
| MedMNIST+ ChestMNIST 64 | `e28fbed21400c50040d6a4e51f379dce89bcb24819d3878c252ca78d52d92c3e` | 401,604,127 bytes; all six NPZ members and source-provided split shapes/dtypes match the inventory; 112,120 image arrays decoded. |

These are the unchanged archives and inventories described in the
[acquisition manifest](LIMITED_DATA_ACQUISITION_2026-11-01.md). ChestMNIST
is chest radiography, not breast ultrasound.

## Method

Exact matches were counted using SHA-256 of original BMP file bytes for
Mendeley and SHA-256 of each contiguous `uint8` 64×64 pixel array for
ChestMNIST. This detects byte-identical Mendeley files and pixel-identical
ChestMNIST samples, respectively; it does not detect every re-encoding or
image transformation.

The similarity screen used a 64-bit DCT perceptual hash: convert to grayscale,
resize to 32×32 using BOX/area resampling, compute the top-left 8×8
coefficients of the orthonormal 2D DCT, and threshold each coefficient at the
median of those 64 coefficients. Distinct hashes at direct Hamming distance
at most 3 were counted as candidate near matches; identical-hash, non-exact
image pairs are reported separately. Pairs were not expanded into transitive
components. A five-band index (13/13/13/13/12 bits, all ten pairs of bands)
reduces candidate comparisons while finding every unique-hash pair within the
stated distance. ChestMNIST required 3,495,875 indexed distance checks across
112,068 unique hashes rather than an all-pairs scan.

The pHash threshold is a screening choice, not a validated duplicate rule.
Similarity can yield false matches and miss transformed copies. It is not
proof that two files show the same acquisition or patient.

## Results

### Mendeley Breast Ultrasound Image v1

- **Exact file matches:** 0 groups, 0 pairs among 250 images.
- **Identical pHash, non-identical files:** 28 hash groups, 44 image pairs:
  18 benign–benign and 26 malignant–malignant.
- **Different pHash codes within distance 3:** 111 unique hash-code pairs and 142
  image pairs. All were at distance 2; no pairs were at distance 1 or 3.
  The pairs were 45 benign–benign and 97 malignant–malignant.
- Full hash groups, candidate hash pairs, image identifiers, per-image hashes,
  and class labels are retained in the machine-readable output. For example,
  one identical-pHash group contains
  `originals/malignant/us114.bmp`, `us115.bmp`, and `us221.bmp`; this is only
  a similarity group, not a patient grouping.

### MedMNIST+ ChestMNIST 64

- **Exact pixel matches:** 25 groups and 27 image pairs: 24 two-image groups
  and one three-image group. This represents 26 repeated sample occurrences
  beyond the first copy in each group. The source splits remain unchanged.
- **Fourteen-label comparison:** 14 groups have identical 14-element binary
  label vectors; 11 groups have conflicting vectors. All 27 labels were
  compared directly from the source NPZ arrays. The names below abbreviate
  the positive target positions in the source order; “all 14 targets=0”
  means all entries are zero, not a verified normal diagnosis.

| Group | Source-provided members | Label-vector comparison |
|---|---|---|
| G01 | `test[3242]`, `test[17956]` | **Conflict:** `test[3242]` = nodule, edema; `test[17956]` = nodule |
| G02 | `test[5411]`, `test[14567]` | Identical: infiltration |
| G03 | `train[11181]`, `train[17715]` | Identical: effusion |
| G04 | `train[11800]`, `train[43643]` | **Conflict:** `train[11800]` = effusion, infiltration, pneumothorax; `train[43643]` = pneumothorax, pleural |
| G05 | `train[18920]`, `train[63791]` | Identical: atelectasis, infiltration |
| G06 | `train[21954]`, `train[29343]` | **Conflict:** `train[21954]` = all 14 targets=0; `train[29343]` = infiltration |
| G07 | `train[31395]`, `train[60807]` | **Conflict:** `train[31395]` = all 14 targets=0; `train[60807]` = atelectasis |
| G08 | `train[35331]`, `train[40086]`, `train[75334]` | Identical: all 14 targets=0 |
| G09 | `train[38395]`, `train[61195]` | **Conflict:** `train[38395]` = pneumothorax; `train[61195]` = atelectasis, pneumothorax |
| G10 | `train[3930]`, `train[35093]` | Identical: all 14 targets=0 |
| G11 | `train[39438]`, `train[53058]` | **Conflict:** `train[39438]` = nodule; `train[53058]` = atelectasis, infiltration |
| G12 | `train[42998]`, `train[52519]` | Identical: all 14 targets=0 |
| G13 | `train[6003]`, `train[18301]` | Identical: all 14 targets=0 |
| G14 | `train[60444]`, `train[74858]` | **Conflict:** `train[60444]` = all 14 targets=0; `train[74858]` = infiltration |
| G15 | `train[6164]`, `train[74404]` | Identical: all 14 targets=0 |
| G16 | `train[65650]`, `val[1517]` | Identical across splits: all 14 targets=0 |
| G17 | `train[76551]`, `val[5194]` | Identical across splits: all 14 targets=0 |
| G18 | `train[8935]`, `train[64077]` | **Conflict:** `train[8935]` = atelectasis, infiltration, pneumothorax; `train[64077]` = fibrosis |
| G19 | `train[9134]`, `train[74178]` | Identical: atelectasis |
| G20 | `val[2373]`, `val[9387]` | **Conflict:** `val[2373]` = all 14 targets=0; `val[9387]` = infiltration, nodule |
| G21 | `val[3418]`, `val[11214]` | Identical: all 14 targets=0 |
| G22 | `val[3477]`, `val[10307]` | Identical: nodule |
| G23 | `val[4957]`, `val[5832]` | **Conflict:** `val[4957]` = all 14 targets=0; `val[5832]` = infiltration |
| G24 | `val[8331]`, `val[10388]` | Identical: all 14 targets=0 |
| G25 | `val[8332]`, `test[5546]` | **Conflict:** `val[8332]` = all 14 targets=0; `test[5546]` = infiltration, emphysema |

Label targets are in this source order: atelectasis, cardiomegaly, effusion,
infiltration, mass, nodule, pneumonia, pneumothorax, consolidation, edema,
emphysema, fibrosis, pleural, hernia.

**Unique pixel samples:** the source arrays contain 78,468 train, 11,219
validation, and 22,433 test samples (112,120 total). After exact deduplication
within each published split, counts are 78,452 train, 11,214 validation, and
22,431 test (112,097 summed across splits). Three further exact equivalences
span splits, leaving **112,094 globally unique pixel arrays**, a reduction of
26 sample occurrences. These are counts for audit interpretation only; no
array or source split was changed.

- Exact duplicate pairs within source splits: train–train 17 pairs,
  validation–validation 5, and test–test 2.
- **Exact cross-split pairs:** `train[65650]`–`val[1517]` and
  `train[76551]`–`val[5194]` (both label vectors all-zero), plus
  `val[8332]`–`test[5546]` (all-zero versus infiltration and emphysema).
  There is no exact train–test pair.
- **Identical pHash, non-identical pixels:** 26 hash groups, 26 image pairs:
  train–train 15, validation–validation 2, test–test 2, train–validation 1,
  train–test 5, and validation–test 1.
- **Different pHash codes within distance 3:** 1,717 unique hash-code pairs
  and 1,732 image pairs. All were at distance 2; none were at distances 1 or 3.
  Pair counts by source split: train–train 877, validation–validation 27,
  test–test 92, train–validation 240, train–test 443, and validation–test 53.

The exact cross-split image matches are a **source-split image-overlap
concern**, not proof of patient overlap or patient leakage. A single patient
may contribute distinct images, and the archive lacks patient-to-image IDs;
neither identity nor patient-level split validity can be inferred from these
pixel matches.

## Pair accounting and visual review

The regenerated audit was run from the unchanged local archives to a separate
output file. The result was independently checked against each source array:
IDs, SHA-256 values, group membership, and label vectors matched. Pair
accounting was also checked: same-hash group counts are distinct from their
image-pair totals, nearby-code records have unique code pairs, every stored
Hamming distance was recomputed from its two 64-bit codes, and weighted
image-pair totals match the stored per-split counts. This confirms the
reported 44 / 142 Mendeley and 26 / 1,732 ChestMNIST same-hash / nearby-code
image-pair counts; it does not make pHash a validated duplicate classifier.

Contact sheets were visually reviewed from
`data\research_data\limited_acquisition_2026-11-01\duplicate_review_2026-11-02\`:

- All 25 ChestMNIST exact-pixel groups (51 images) are in five
  [`exact-group sheets`](../data/research_data/limited_acquisition_2026-11-01/duplicate_review_2026-11-02/chestmnist_exact_groups_01.png)
  (pages 01–05). All exact groups and every cross-split exact pair are shown.
- All 186 Mendeley pHash candidate image pairs (44 same-hash and 142
  nearby-code pairs) are in eight
  [`Mendeley candidate sheets`](../data/research_data/limited_acquisition_2026-11-01/duplicate_review_2026-11-02/mendeley_phash_candidates_01.png)
  (pages 01–08).
- ChestMNIST includes all 26 non-exact same-hash pairs and a deterministic,
  evenly spaced sample of 45/1,732 nearby-code pairs: up to 10 from each of
  the three cross-split strata and up to 5 from each within-split stratum.
  The remaining 1,687 nearby-code image pairs were not visually reviewed.
  The 71 selected pairs are in three
  [`ChestMNIST candidate sheets`](../data/research_data/limited_acquisition_2026-11-01/duplicate_review_2026-11-02/chestmnist_phash_candidates_01.png)
  (pages 01–03). This includes a sample from every split-pair category; it
  does not include every cross-split near-match candidate.

Visual resemblance is a candidate flag only. The sheets contain similar
images as well as visibly different images sharing a pHash threshold;
visual review cannot establish that files are the same acquisition or
patient.

The original machine result remains
[`duplicate_audit.json`](../data/research_data/limited_acquisition_2026-11-01/duplicate_audit.json).
The reproduced output is
[`duplicate_audit_reproduced_2026-11-01.json`](../data/research_data/limited_acquisition_2026-11-01/duplicate_audit_reproduced_2026-11-01.json);
label comparisons, pHash validation, and contact-sheet inventory are in
[`duplicate_validation_and_review.json`](../data/research_data/limited_acquisition_2026-11-01/duplicate_review_2026-11-02/duplicate_validation_and_review.json).
Reproduce the archive audit with
`python -m ml.audit_acquired_duplicates --output data\research_data\limited_acquisition_2026-11-01\duplicate_audit_reproduced_2026-11-01.json`,
then run
`python -m ml.validate_duplicate_review --audit-json data\research_data\limited_acquisition_2026-11-01\duplicate_audit_reproduced_2026-11-01.json --output-dir data\research_data\limited_acquisition_2026-11-01\duplicate_review_2026-11-02`.
The validation and review implementation and focused tests are in
[`ml/validate_duplicate_review.py`](../ml/validate_duplicate_review.py) and
[`tests/test_duplicate_audit.py`](../tests/test_duplicate_audit.py).

## Limitations and use boundary

Neither dataset supplies usable patient-to-image identifiers. Exact or
perceptual matching cannot determine patient identity, repeat examinations,
or all possible split leakage; it cannot establish patient-independent
evaluation. The ChestMNIST cross-split exact pairs are image-level
source-split overlap observations only—not proof that the images belong to
one patient or that a patient crossed splits. Do not infer patient identity
from filenames, array indices, labels, or visual similarity.

Both datasets remain **limited development/inventory only**. ChestMNIST is
chest X-ray data, not breast-ultrasound data; it cannot train or validate a
breast-ultrasound eligibility gate. It is not a substitute for a patient-
linked breast-ultrasound cohort or diverse, traceable negative evaluation
data. Public classification and inference remain disabled.
