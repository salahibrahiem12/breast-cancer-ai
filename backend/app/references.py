"""Verifiable medical references and clinical guidelines."""

from pydantic import BaseModel


class MedicalReference(BaseModel):
    id: str
    organization: str
    title: str
    url: str
    category: str
    summary: str
    year: int


VERIFIED_REFERENCES: list[dict] = [
    {
        "id": "nci-pdq-2024",
        "organization": "National Cancer Institute (NCI)",
        "title": "Breast Cancer Treatment (Adult) (PDQ®) – Health Professional Version",
        "url": "https://www.cancer.gov/types/breast/hp/breast-treatment-pdq",
        "category": "Treatment Guidelines",
        "summary": "Comprehensive clinical evidence summaries on surgical options (breast-conserving surgery vs mastectomy), radiation therapy protocols, adjuvant systemic chemotherapy, endocrine therapy, HER2-targeted agents, and PARP inhibitors.",
        "year": 2024,
    },
    {
        "id": "acr-birads-2023",
        "organization": "American College of Radiology (ACR)",
        "title": "ACR BI-RADS® Atlas – Breast Imaging Reporting and Data System",
        "url": "https://www.acr.org/Clinical-Resources/Clinical-Tools-and-Reference/Reporting-and-Data-Systems/BI-RADS",
        "category": "Diagnostic Standards",
        "summary": "Standardized terminology, lesion descriptors, structured reporting categories (0–6), and management recommendations across mammography, breast ultrasound, and breast MRI.",
        "year": 2023,
    },
    {
        "id": "asco-early-breast-2023",
        "organization": "American Society of Clinical Oncology (ASCO)",
        "title": "Selection of Optimal Adjuvant Chemotherapy and Targeted Therapy for Early Breast Cancer",
        "url": "https://www.asco.org/practice-patients/guidelines/breast-cancer",
        "category": "Systemic Therapy",
        "summary": "Evidence-based guidelines on systemic chemotherapy regimens, dose-dense schedules, anti-HER2 therapies (trastuzumab, pertuzumab, T-DM1), and adjuvant CDK4/6 inhibitors for high-risk HR+/HER2- disease.",
        "year": 2023,
    },
    {
        "id": "esmo-early-breast-2024",
        "organization": "European Society for Medical Oncology (ESMO)",
        "title": "Early Breast Cancer: ESMO Clinical Practice Guideline for Diagnosis, Treatment and Follow-up",
        "url": "https://www.esmo.org/guidelines/guidelines-by-topic/breast-cancer",
        "category": "Clinical Practice",
        "summary": "European multidisciplinary guidelines covering tumor biology, multigene expression profiling (Oncotype DX, MammaPrint), surgical margins, hypofractionated radiotherapy, and bone health management during endocrine therapy.",
        "year": 2024,
    },
    {
        "id": "busbra-zenodo-2024",
        "organization": "Medical Physics & Zenodo",
        "title": "BUS-BRA: A Large Ultrasound Dataset for Breast Lesion Detection and Diagnosis",
        "url": "https://pubmed.ncbi.nlm.nih.gov/37937827/",
        "category": "Ultrasound Dataset",
        "summary": "Open-access breast ultrasound cohort of 1,875 images across 1,063 biopsy/follow-up confirmed lesion cases under CC BY 4.0 license (DOI: 10.1002/mp.16812).",
        "year": 2024,
    },
    {
        "id": "tcia-breast-lesions-2023",
        "organization": "The Cancer Imaging Archive (TCIA)",
        "title": "BrEaST-Lesions USG: A Biopsy-Confirmed Dataset of Breast Lesions in Ultrasound Images",
        "url": "https://doi.org/10.7937/9WKK-Q141",
        "category": "Ultrasound Dataset",
        "summary": "Multi-center clinical ultrasound dataset containing 197 biopsy-confirmed breast lesions with detailed histopathology and patient metadata under CC BY 4.0 license.",
        "year": 2023,
    },
    {
        "id": "cbis-ddsm-tcia",
        "organization": "The Cancer Imaging Archive (TCIA)",
        "title": "Curated Breast Imaging Subset of Digital Database for Screening Mammography (CBIS-DDSM)",
        "url": "https://www.cancerimagingarchive.net/collection/cbis-ddsm/",
        "category": "Mammography Dataset",
        "summary": "Standard benchmark subset of 2,620 scanned screening mammograms with pathological verification, mass/calcification segmentations, and BI-RADS descriptors (DOI: 10.7937/K9/TCIA.2016.7A42F5AA).",
        "year": 2017,
    },
]

