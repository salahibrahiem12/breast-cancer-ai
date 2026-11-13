import Link from "next/link";

const features = [
  {
    title: "Dual Modality Verification",
    badge: "Ultrasound & Mammography",
    description: "Automatic visual feature analysis determines whether scans are Breast Ultrasound or Mammography before any analysis. Strictly rejects chest X-rays, natural photos, selfies, and non-medical images.",
  },
  {
    title: "4-Stage Quality & Safety Gate",
    badge: "Fail-Closed Protocol",
    description: "Checks file integrity, modality discrimination, image contrast/blur quality control, and out-of-distribution (OOD) boundary assessment to prevent erroneous classification.",
  },
  {
    title: "Interpretability & Explainability",
    badge: "Layer4 Grad-CAM",
    description: "Generates attention heatmaps highlighting spatial regions analyzed by deep feature encoders. Accompanied by strict notices that heatmaps are mathematical visual cues, not certified tumor boundaries.",
  },
  {
    title: "Clinical Reference Alignment",
    badge: "NCI · ACR · ASCO · ESMO",
    description: "Educational modules and report summaries cite published, peer-reviewed clinical guidelines, including ACR BI-RADS Atlas and NCI Physician Data Query (PDQ).",
  },
];

const safetyNotices = [
  "Investigational research prototype only — not approved by the FDA or CE-marked for primary diagnostic use.",
  "Cancer classification is strictly fail-closed by default (BREASTCARE_CLASSIFICATION_ENABLED=false) pending certified clinical trials.",
  "Never classifies chest X-rays, CT scans, MRIs, or consumer photographs as breast imaging.",
  "Borderline or out-of-distribution uploads display: 'Unable to provide a reliable assessment. Specialist review is recommended.'",
];

export default function HomePage() {
  return (
    <main className="page-shell mx-auto max-w-7xl px-5 pb-16 pt-10 sm:pt-16 lg:px-8">
      {/* Hero Section */}
      <section className="grid items-center gap-10 lg:grid-cols-[1.15fr_0.85fr]">
        <div>
          <div className="inline-flex items-center gap-2 rounded-full border border-pink-rose bg-pink-soft/80 px-4 py-1.5 text-xs font-semibold uppercase tracking-wider text-pink-deep">
            <span>🎗️ Breast Cancer Awareness & Research</span>
          </div>
          <h1 className="mt-5 max-w-3xl text-4xl font-extrabold leading-tight tracking-tight text-ink sm:text-6xl">
            Intelligent Medical Imaging Decision-Support for <span className="text-pink-primary">Breast Health</span>.
          </h1>
          <p className="mt-6 max-w-2xl text-lg leading-8 text-muted">
            BreastCare AI provides an open, rigorous research environment combining multi-stage image verification,
            dual-modality (Ultrasound & Mammography) quality validation, and deep learning feature interpretability.
          </p>
          <div className="mt-8 flex flex-wrap gap-4">
            <Link
              href="/analyze"
              className="rounded-xl bg-pink-primary px-7 py-3.5 font-semibold text-white shadow-md hover:bg-pink-dark transition-all transform hover:-translate-y-0.5"
            >
              Analyze Medical Scan
            </Link>
            <Link
              href="/cancer-info"
              className="rounded-xl border border-pink-rose bg-white px-6 py-3.5 font-semibold text-pink-deep transition hover:bg-pink-soft/50 shadow-sm"
            >
              Learn Symptoms & Screening
            </Link>
            <Link
              href="/treatment-info"
              className="rounded-xl border border-line bg-white px-6 py-3.5 font-semibold text-slate-text transition hover:border-pink-primary"
            >
              Treatment Pathways
            </Link>
          </div>
          <p className="mt-5 text-xs font-semibold uppercase tracking-wider text-pink-accent">
            ⚠️ Experimental Research Prototype · Strict Fail-Closed Safeguards Active
          </p>
        </div>

        {/* Safety & Protocol Card */}
        <aside className="rounded-3xl border border-pink-rose bg-white p-7 shadow-card sm:p-8">
          <div className="flex items-center justify-between pb-4 border-b border-pink-soft">
            <h2 className="text-base font-bold text-ink flex items-center gap-2">
              <span className="h-3 w-3 rounded-full bg-emerald-500 animate-pulse" />
              Operational Safeguards Status
            </h2>
            <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-pink-soft text-pink-deep">
              v3.2 Fail-Closed Active
            </span>
          </div>

          <div className="mt-5 rounded-2xl bg-pink-bg border border-pink-rose/50 p-5">
            <p className="text-xs font-bold uppercase tracking-wider text-pink-deep">Current Clinical Protocol State</p>
            <p className="mt-1 text-sm font-semibold text-ink">
              Multi-Stage Validation Enabled · Diagnostic Inference Restrained
            </p>
            <p className="mt-2 text-xs leading-5 text-slate-600">
              The platform executes file structure verification, dual-modality validation, image quality checks, and Grad-CAM interpretability. Diagnostic classification remains restrained until prospective multi-center trial clearance.
            </p>
          </div>

          <ul className="mt-6 space-y-3.5">
            {safetyNotices.map((notice, idx) => (
              <li key={idx} className="flex items-start gap-3 text-xs leading-5 text-slate-600">
                <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-pink-primary" aria-hidden="true" />
                <span>{notice}</span>
              </li>
            ))}
          </ul>
        </aside>
      </section>

      {/* Feature Grid */}
      <section className="mt-16">
        <div className="text-center max-w-2xl mx-auto mb-10">
          <h2 className="text-2xl sm:text-3xl font-bold text-ink">Engineering & Clinical Methodology</h2>
          <p className="mt-3 text-sm text-muted">
            Engineered with strict clinical ethics, patient-level isolation, and transparent scientific metrics.
          </p>
        </div>

        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {features.map((feat) => (
            <article key={feat.title} className="rounded-2xl border border-pink-rose/60 bg-white p-6 shadow-sm hover:shadow-card transition-shadow">
              <span className="inline-block rounded-md bg-pink-soft px-2.5 py-1 text-xs font-semibold text-pink-deep mb-3">
                {feat.badge}
              </span>
              <h3 className="font-bold text-ink text-base">{feat.title}</h3>
              <p className="mt-2 text-xs leading-6 text-slate-600">{feat.description}</p>
            </article>
          ))}
        </div>
      </section>

      {/* Supported Modalities Banner */}
      <section className="mt-14 rounded-3xl bg-gradient-to-r from-pink-soft via-pink-rose/40 to-pink-soft border border-pink-rose p-8">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="space-y-2">
            <h3 className="text-xl font-bold text-ink">Supported Diagnostic Imaging Modalities</h3>
            <p className="text-sm text-slate-700 max-w-2xl">
              <strong>Modality A: Breast Ultrasound</strong> (B-mode ultrasound scans, hypoechoic/hyperechoic acoustic patterns).<br />
              <strong>Modality B: Mammography</strong> (Full-Field Digital Mammography or film digitization, CC and MLO views).<br />
              <span className="text-xs text-pink-deep font-semibold">
                * Note: Chest X-rays, abdominal scans, selfies, screenshots, and non-breast anatomical scans are detected and rejected at Stage 2.
              </span>
            </p>
          </div>
          <Link
            href="/analyze"
            className="shrink-0 rounded-xl bg-pink-primary px-6 py-3 font-semibold text-white shadow hover:bg-pink-dark transition-colors"
          >
            Go to Image Analysis →
          </Link>
        </div>
      </section>
    </main>
  );
}
