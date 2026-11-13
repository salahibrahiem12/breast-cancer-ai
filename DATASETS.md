# Dataset Provenance, Curation & Partitioning Audit

BreastCare AI relies on audited, high-integrity medical datasets with strict patient/lesion isolation to eliminate data leakage.

---

## 1. Curated Dataset Inventory

The primary training and validation cohort is unified in `data/prepared/unified_dataset_manifest_v2.csv`:
- **Total Validated Samples:** 2,215 clean images
- **Total Patient/Lesion Isolation Groups:** 1,288 groups
- **Class Distribution:**
  - Benign: 928 images
  - Malignant: 1,287 images

| Source Cohort | Primary Modality | Sample Count | Group Count | License / Usage Terms | Citation |
|---|---|---|---|---|---|
| **BUS-BRA** | B-mode Ultrasound | 1,418 images | 823 groups | CC BY 4.0 (Open Access) | Gomes et al., *Scientific Data*, 2024 |
| **TCIA BrEaST** | Breast Ultrasound | 541 images | 256 groups | TCIA Data Usage Policy | Cancer Imaging Archive, 2023 |
| **Mendeley Ultrasound** | Breast Ultrasound | 256 images | 209 groups | CC BY 4.0 | Rodrigues et al., Mendeley Data, 2018 |
| **CBIS-DDSM** | Mammography (CC/MLO) | 3,568 cases (Reference) | Case-isolated | TCIA Open Access | Sawyer-Lee et al., *Scientific Data*, 2017 |

---

## 2. Partitioning & Data Quality Rules

1. **Strict Group-Level Isolation:**
   - Multi-view images, temporal series, and contralateral scans belonging to the same patient or lesion are strictly grouped under an invariant `group_id`.
   - No patient or group crosses between training, validation, or held-out test splits.
2. **Duplicate Detection Audit:**
   - Exact and perceptual hash (pHash) analysis eliminated 18 duplicate pairs identified across legacy source folders.
3. **Cross-Source Evaluation Protocol:**
   - To assess true clinical generalization, models trained on BUS-BRA are evaluated on completely unseen datasets (TCIA and Mendeley) without prior exposure or domain adaptation.

