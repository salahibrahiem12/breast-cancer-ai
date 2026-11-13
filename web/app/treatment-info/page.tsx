const treatmentModalities = [
  {
    title: "Surgical Resection",
    badge: "Local Therapy",
    detail: "Primary intervention to remove localized tumor tissue with negative surgical margins.",
    options: [
      {
        name: "Breast-Conserving Surgery (Lumpectomy)",
        desc: "Excision of the primary tumor with a surrounding margin of normal tissue, typically followed by whole-breast adjuvant radiotherapy.",
      },
      {
        name: "Mastectomy",
        desc: "Complete removal of the breast tissue, performed for multicentric disease, large tumor-to-breast ratio, or patient preference, with immediate or delayed reconstruction options.",
      },
      {
        name: "Axillary Staging (SLNB vs ALND)",
        desc: "Sentinel lymph node biopsy (SLNB) to evaluate nodal involvement with minimal lymphedema risk, avoiding full axillary dissection in selected early-stage presentations.",
      },
    ],
  },
  {
    title: "Radiation Therapy",
    badge: "Locoregional Control",
    detail: "High-energy photon beams destroy microscopic residual cancer cells post-surgery.",
    options: [
      {
        name: "Adjuvant Whole-Breast Irradiation",
        desc: "Standard of care following breast-conserving surgery to reduce local recurrence rates by up to 50–70% (EBCTCG meta-analysis).",
      },
      {
        name: "Post-Mastectomy Radiation Therapy (PMRT)",
        desc: "Indicated for patients with positive surgical margins, tumors >5 cm (T3/T4), or involvement of ≥4 axillary lymph nodes.",
      },
    ],
  },
  {
    title: "Systemic Chemotherapy",
    badge: "Systemic Therapy",
    detail: "Cytotoxic drug combinations targeting rapidly dividing cells throughout the body.",
    options: [
      {
        name: "Neoadjuvant Chemotherapy",
        desc: "Administered prior to surgery to downstage locally advanced tumors, facilitate breast conservation, and assess in vivo pathologic complete response (pCR).",
      },
      {
        name: "Standard Anthracycline & Taxane Regimens",
        desc: "Common regimens include AC-T (Doxorubicin + Cyclophosphamide followed by Paclitaxel) or Docetaxel + Cyclophosphamide (TC).",
      },
    ],
  },
  {
    title: "Endocrine (Hormone) Therapy",
    badge: "Receptor-Targeted",
    detail: "Indicated for Hormone Receptor-Positive (ER+ and/or PR+) breast cancers, suppressing estrogenic signaling.",
    options: [
      {
        name: "Selective Estrogen Receptor Modulators (SERMs)",
        desc: "Tamoxifen (20 mg daily for 5–10 years) blocks estrogen binding in breast tissue, suitable for premenopausal and postmenopausal patients.",
      },
      {
        name: "Aromatase Inhibitors (AIs)",
        desc: "Anastrozole, Letrozole, or Exemestane block peripheral estrogen synthesis in postmenopausal women, superior to tamoxifen in disease-free survival.",
      },
      {
        name: "Ovarian Function Suppression (OFS)",
        desc: "GnRH agonists (Goserelin, Leuprolide) paired with tamoxifen or AI for high-risk premenopausal women (SOFT/TEXT trials).",
      },
    ],
  },
  {
    title: "HER2-Targeted Biologics",
    badge: "Targeted Therapy",
    detail: "For HER2-overexpressing or gene-amplified (IHC 3+ or FISH+) breast tumors.",
    options: [
      {
        name: "Trastuzumab (Herceptin)",
        desc: "Humanized monoclonal antibody binding extracellular domain IV of HER2, substantially reducing recurrence and mortality.",
      },
      {
        name: "Pertuzumab & Antibody-Drug Conjugates (ADCs)",
        desc: "Dual HER2 blockade (Pertuzumab + Trastuzumab) or ADCs such as T-DM1 (Trastuzumab emtansine) and T-DXd (Trastuzumab deruxtecan) for residual or metastatic disease.",
      },
    ],
  },
  {
    title: "Immunotherapy & PARP Inhibitors",
    badge: "Biomarker-Driven",
    detail: "Emerging targeted modalities for specific molecular subtypes.",
    options: [
      {
        name: "Immune Checkpoint Blockade",
        desc: "Pembrolizumab (anti-PD-1) combined with chemotherapy for high-risk early-stage or PD-L1 positive metastatic Triple-Negative Breast Cancer (KEYNOTE-522).",
      },
      {
        name: "PARP Inhibitors",
        desc: "Olaparib or Talazoparib for patients with germline pathogenic BRCA1/2 mutations, targeting homologous recombination deficiency via synthetic lethality.",
      },
    ],
  },
];

export default function TreatmentInfoPage() {
  return (
    <main className="page-shell mx-auto max-w-5xl px-5 pb-16 pt-10 sm:pt-14 lg:px-8">
      {/* Header */}
      <div className="border-b border-pink-rose/60 pb-6 mb-8">
        <div className="inline-flex items-center gap-2 rounded-full border border-pink-rose bg-pink-soft px-3 py-1 text-xs font-semibold text-pink-deep mb-3">
          Clinical Oncology Guidelines & Pathways
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-ink tracking-tight">
          Breast Cancer <span className="text-pink-primary">Treatment Pathways</span>
        </h1>
        <p className="mt-3 text-base text-muted max-w-3xl leading-relaxed">
          Breast cancer treatment is highly individualized. It depends on anatomic staging (TNM), tumor grade,
          biomarker receptor expression (ER, PR, HER2, Ki-67), and genomic risk profiles. This overview compiles standard-of-care
          protocols established by NCCN, ASCO, ESMO, and St. Gallen consensus panels.
        </p>
      </div>

      {/* Critical Medical Disclaimer Banner */}
      <div className="rounded-2xl border-2 border-pink-rose bg-pink-soft/60 p-6 mb-10 shadow-sm">
        <div className="flex items-start gap-4">
          <span className="text-2xl">⚠️</span>
          <div>
            <h2 className="text-base font-bold text-pink-deep">Mandatory Clinical Decision Disclaimer</h2>
            <p className="mt-1 text-xs leading-5 text-slate-700">
              Treatment strategies cannot be selected from imaging or artificial intelligence predictions alone.
              Therapeutic decisions require definitive tissue histopathology, immunohistochemistry (IHC) / in-situ hybridization (FISH)
              receptor quantification, multidisciplinary tumor board review, and an in-person oncology consultation.
            </p>
          </div>
        </div>
      </div>

      {/* Treatment Modalities Grid */}
      <div className="space-y-8">
        {treatmentModalities.map((modality, idx) => (
          <section key={idx} className="rounded-2xl border border-pink-rose/60 bg-white p-7 shadow-sm">
            <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
              <h2 className="text-xl font-bold text-ink flex items-center gap-2">
                <span className="text-pink-primary">{idx + 1}.</span> {modality.title}
              </h2>
              <span className="rounded-full bg-pink-soft px-3 py-1 text-xs font-semibold text-pink-deep">
                {modality.badge}
              </span>
            </div>
            <p className="text-sm text-slate-600 mb-5">{modality.detail}</p>

            <div className="grid gap-3 sm:grid-cols-2">
              {modality.options.map((opt, oIdx) => (
                <div key={oIdx} className="rounded-xl bg-pink-bg p-4 border border-pink-rose/40">
                  <h3 className="font-bold text-ink text-xs sm:text-sm">{opt.name}</h3>
                  <p className="mt-1 text-xs text-slate-600 leading-5">{opt.desc}</p>
                </div>
              ))}
            </div>
          </section>
        ))}
      </div>

      {/* Verified Clinical Guidelines Citation */}
      <div className="mt-12 rounded-2xl bg-white border border-pink-rose/60 p-6 shadow-sm">
        <h3 className="text-base font-bold text-ink mb-3">Authoritative Guidelines & Clinical Literature</h3>
        <ul className="space-y-2 text-xs text-slate-700 list-disc list-inside">
          <li><strong>NCCN Guidelines:</strong> National Comprehensive Cancer Network Clinical Practice Guidelines in Oncology: Breast Cancer, Version 4.2024.</li>
          <li><strong>ASCO Guidelines:</strong> Burstein HJ, et al. Endocrine Therapy and Targeted Therapy for Patients With Hormone Receptor-Positive HER2-Negative Breast Cancer: ASCO Guideline Update, JCO 2023.</li>
          <li><strong>ESMO Guidelines:</strong> Loibl S, et al. Early Breast Cancer: ESMO Clinical Practice Guideline for diagnosis, treatment and follow-up, Ann Oncol 2024.</li>
          <li><strong>ACR Appropriateness Criteria:</strong> Breast Cancer Management, American College of Radiology 2023.</li>
        </ul>
      </div>
    </main>
  );
}

