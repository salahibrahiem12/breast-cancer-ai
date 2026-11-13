# Global Duplicate-Group Integrity Audit (`grouping_integrity_audit_v1`)

Completed 4 November 2026 (`ml/audit_grouping_integrity.py`,
`reports/grouping_audit_v1/grouping_audit.json`).
Original manifests (`data/prepared/unified_dataset_manifest_v1.csv` and
`reports/design_audit_v1/eligible_manifest.json`) are preserved untouched.
Corrected grouping and historical-test-leakage exclusions are versioned in
`data/prepared/unified_dataset_manifest_v2.csv`,
`data/prepared/unified_dataset_manifest_v2_summary.json`, and
`reports/grouping_audit_v1/eligible_manifest_v2.json`.

> **Terminology rule:** Duplicate groups (`global:*`, `mendeley:group_*`,
> `v2_global:*`) and BUS-BRA source cases (`busbra:*`) are **never** called
> patients. Mendeley has no patient identifiers; BUS-BRA provides only `Case`
> identifiers with unresolved patient-to-case semantics; TCIA BrEaST provides
> publisher-reported within-source `CaseID`s (`1` primary scan per `CaseID`).

---

## 1. Why 162 Eligible Mendeley Images Form 38 Global Groups in `v1`

1. **Physical structure of the 250 acquired Mendeley BMP files (`us1.bmp`..`us250.bmp`):**
   - Although all 250 Mendeley files have distinct byte and pixel SHA-256 digests,
     they are **not** 250 independent scans.
   - Across all 50 consecutive five-image blocks (`Block #k = [us{5k+1}.bmp .. us{5k+5}.bmp]`
     for `k = 0..49`: 20 benign blocks = 100 images, 30 malignant blocks = 150 images),
     **all 50 blocks (`50/50`) have 100% identical pixel dimensions `(width, height)`
     and identical diagnosis labels across all 5 images in the block**, and high
     translation-aligned pixel correlation (`0.774–0.998`).
   - Visual inspection of the contact sheets (`reports/grouping_audit_v1/contact_sheets/mendeley_largest_groups.png`
     and `mendeley_phash_unit_merges.png`) confirms that each 5-image sequence consists
     of **5 slightly shifted/translated crops of the same base ultrasound lesion frame**
     (visible as black border padding shifting by a few pixels across `us{5k+1}..us{5k+5}`).
   - Furthermore, two malignant 5-image sequences—Block `#23` (`us116..us120.bmp`,
     `153×131`) and Block `#45` (`us226..us230.bmp`, `153×131`)—are derived from
     the **exact same base malignant ultrasound scan** (`r = 0.865–0.994`), meaning
     the 250 Mendeley files represent **49 unique base lesion scans** (`20` benign,
     `29` malignant).
2. **Stage-by-stage collapse from 250 acquired images to 162 eligible images in 38 `v1` global groups:**
   - **Stage 1 (`mendeley_v1`, `dHash <= 3` on 250 images -> `126` groups):**
     Because 3–8 pixel translations across a 5-image crop sequence often produce
     `dHash` Hamming distances of `4..16`, grouping at `dHash <= 3` split many
     5-image sequences into multiple fragments (`126` groups total) and allowed
     frames from the same 5-image sequence to be split across `mendeley_v1` `train`,
     `validation`, and `test`.
   - **Stage 2 (`mendeley_dev_v2`, `dHash <= 5` on 250 images -> `72` groups; excluding historical test links -> `162` eligible images in `51` `mendeley:group_*` units):**
     Widening `dHash` to `<= 5` merged the 250 images into `72` components and
     excluded `21` components (`88` images) linked at `dHash <= 5` to the 50-image
     historical `mendeley_v1` test set, leaving `162` eligible images in `51`
     `mendeley:group_*` units (`22` benign, `29` malignant). However, `11` same-scan
     crop sequences still had `dHash = 6..13` between shifted frames and remained
     split across multiple `mendeley:group_*` units.
   - **Stage 3 (`eligible_manifest.json` / `unified_dataset_manifest_v1.csv`, `dHash <= 5 OR pHash <= 5` -> `38` `global:*` groups):**
     Adding `pHash <= 5` alongside `dHash <= 5` in `ml/cross_source.py` and
     `ml/design_audit.py` connected 11 split `mendeley:group_*` clusters whose low-frequency
     DCT hashes agreed (`pHash = 0..4`, direct pixel correlation `0.842–0.992`),
     collapsing the `51` `dHash <= 5` units into **`38` `global:*` duplicate groups**
     (`17` benign, `21` malignant).

---

## 2. Group Sizes, Class Conflicts, Source Composition, and Edge Types (`v1` vs `v2`)

### 2.1 Source Composition and Group Size Distribution (`2,232` Eligible Images in `v1`)

| Source | Grouping Basis in `v1` | Eligible Images | Known Source Units | `v1` Global Groups | Group Size Histogram (`size: count`) | Class Conflicts | Cross-Source Links |
|---|---|---:|---:|---:|---|---:|---:|
| **Mendeley** | Perceptual `dHash<=5 OR pHash<=5` (`no patient IDs`) | `162` | `51` (`dHash<=5`) | `38` | `1: 4, 2: 3, 3: 3, 4: 2, 5: 25, 10: 1` | `0` | `0` |
| **TCIA BrEaST** | Publisher `patient_id` (`1` scan/patient) + `dHash/pHash<=5` | `197` | `197` | `197` | `1: 197` | `0` | `0` |
| **BUS-BRA** | Source `Case` (unresolved patient semantics) + `dHash/pHash<=5` | `1,873` | `1,063` | `1,063` | `1: 253, 2: 810` | `0` | `0` |
| **Total (`v1`)** | **Source-pure, label-pure connected components** | **`2,232`** | **`1,311`** | **`1,298`** | **Max group size = `10`** | **`0`** | **`0`** |

### 2.2 Edge-Type Breakdown Across the 2,381 Acquired Images

- **Verified source-provided case/patient links (`explicit_unit`):** `1,254` pairs
  (all within BUS-BRA two-image cases or Mendeley `dHash<=5` units; TCIA has 1 scan per `CaseID`).
- **Exact byte or pixel duplicates (`exact_sha256`, `exact_pixel_sha256`):** `0` pairs
  among the 2,381 acquired images (all 2,381 files have distinct byte and pixel SHA-256 digests).
- **Perceptual-similarity edges among the 2,381 acquired images:**
  - `dHash <= 3`: `162` pairs (`15` pairs have `dHash == 0`).
  - `dHash in {4, 5}`: `139` pairs.
  - `pHash <= 3`: `186` pairs.
  - `pHash in {4, 5}`: `157` pairs.
- **Class conflicts and cross-source edges:** `0` within-source label-conflict pairs
  and `0` cross-source pairs exist among the 2,381 acquired images in `unified_dataset_manifest_v1.csv`.
  The single excluded BUS-BRA case (`busbra:632`: `bus_0632-l.png` and `bus_0632-r.png`)
  was excluded by `ml/busbra_locked.py` because `bus_0632-l.png` had an accidental
  `dHash = 5` collision (`pHash = 24`, Pearson `r = 0.10`) with an unverified dark image
  `data/raw/malignant/malignant_0100.png` outside the 3-source manifest (visually verified in
  `reports/grouping_audit_v1/contact_sheets/busbra_and_cross_source_merges.png`).

---

## 3. Transitive Perceptual Chains and Visual Inspection Findings

1. **Did transitive `dHash<=5 OR pHash<=5` links chain distinct lesions together in `v1`? No:**
   - Every single one of the `38` Mendeley `v1` global groups has **100% identical
     `(width, height)` pixel resolution across all member images**, identical diagnosis
     label, and high shift-aligned pixel correlation (`0.774–0.998`).
   - All `11` groups where `pHash <= 5` merged multiple `dHash <= 5` units
     (`global:10`, `global:42`, `global:45`, `global:61`, `global:85`, `global:88`,
     `global:98`, `global:105`, `global:110`, `global:153`, `global:158`) were visually
     inspected in `reports/grouping_audit_v1/contact_sheets/mendeley_phash_unit_merges.png`:
     in every case, `pHash <= 5` **reunited consecutive shifted crops (`us{5k+1}..us{5k+5}`)
     of the exact same lesion scan** whose border translation had exceeded `dHash = 5`.
   - In TCIA (`197` groups of size 1) and BUS-BRA (`1,063` source cases), `dHash <= 5 OR pHash <= 5`
     merged **zero** distinct TCIA patient IDs and **zero** distinct BUS-BRA source cases.
2. **Two real Mendeley grouping defects uncovered by visual and block-sequence audit (`busbra_and_cross_source_merges.png`):**
   - **Defect A — 17 Mendeley images in 8 partial groups leaked from historical `mendeley_v1` test sequences:**
     In 8 of the 50 five-image Mendeley sequences (`Block #2: us11..15`, `Block #3: us16..20`,
     `Block #19: us96..100`, `Block #21: us106..110`, `Block #27: us136..140`,
     `Block #34: us171..175`, `Block #40: us201..205`, `Block #41: us206..210`), one or
     more frames belonged to the historical 50-image `mendeley_v1` test split, but
     17 sibling shifted-crop frames in those same 8 sequences had `dHash >= 6` from
     the test frame and escaped `mendeley_dev_v2`'s `dHash <= 5` exclusion filter
     (forming the 8 partial `v1` global groups `global:2`, `global:3`, `global:5`,
     `global:72`, `global:85`, `global:103`, `global:130`, `global:131`).
   - **Defect B — 2 fully eligible Mendeley five-image sequences remained split across two `v1` global groups:**
     Benign Block `#4` (`us21..us25.bmp`, `93×57`) was split into `global:10` (`us21..23`)
     and `global:13` (`us24..25`), and malignant Block `#38` (`us191..us195.bmp`, `119×101`)
     was split into `global:120` (`us191..194`) and `global:123` (`us195`), because
     border shifts between the two halves had `dHash >= 6` and `pHash >= 6`.

---

## 4. Versioned Manifest (`unified_dataset_manifest_v2.csv`) and Invalidation Scope

- **Preserved `v1` artifacts:** `data/prepared/unified_dataset_manifest_v1.csv`
  (SHA-256 `45364769e1aa7e6e264851d1b22b0c2da9b258b75f2e1ae45f172558e0c2772a`) and
  `reports/design_audit_v1/eligible_manifest.json` remain untouched.
- **Versioned `v2` manifest (`data/prepared/unified_dataset_manifest_v2.csv`, `reports/grouping_audit_v1/eligible_manifest_v2.json`):**
  - Excludes the `17` leaked Mendeley partial-sequence images that share a base scan
    with historical `mendeley_v1` test images.
  - Reunites `Block #4` (`us21..25`) into `v2_global:mendeley_seq_04` and `Block #38`
    (`us191..195`) into `v2_global:mendeley_seq_38`.
  - Resulting `v2` eligible cohort: **`2,215` eligible images across `1,288` duplicate/case groups**
    (`145` Mendeley images in `28` complete base-scan duplicate groups [27 of size 5,
    1 of size 10], `197` TCIA images in `197` patient IDs, and `1,873` BUS-BRA images
    in `1,063` source cases).
- **Explicit invalidation scope:**
  1. **Historical `mendeley_v1` test-disjointness claim on `v1` Mendeley (`162` images):**
     **Invalidated for 17 images.** The 162-image `mendeley_dev_v2` / `v1` Mendeley
     development set contained 17 shifted-crop siblings of historical `mendeley_v1`
     test frames. Strict test-disjointness holds only for the 145-image (`28`-group)
     `v2` Mendeley cohort.
  2. **Inner 4-fold CV group isolation when Mendeley is on the training side (`v1`):**
     **Affected on 2 lesion sequences (`us21..25` and `us191..195`).** In `full_source_baseline_v1`
     and `full_source_finetune_v1`, `global:10`/`global:13` and `global:120`/`global:123`
     could be placed in different inner CV folds when TCIA or BUS-BRA was held out.
  3. **Outer cross-source held-out isolation (`mendeley` vs `tcia` vs `busbra`):**
     **Not invalidated.** Zero exact or perceptual links cross dataset sources.
  4. **GPU encoder fine-tuning lock (`gpu_encoder_finetune_v2` -> `gpu_encoder_finetune_v3`):**
     `reports/gpu_encoder_finetune_v2/lock.json` is preserved in the audit trail
     (`reports/gpu_encoder_finetune_v2/SUPERSEDED_BY_V3.json`) and superseded before
     any GPU training by `reports/gpu_encoder_finetune_v3/lock.json` bound to
     `unified_dataset_manifest_v2.csv` (`2,215` eligible images, `1,288` groups).

