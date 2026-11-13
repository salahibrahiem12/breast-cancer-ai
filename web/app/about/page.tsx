import Link from "next/link";

export default function AboutPage() {
  return (
    <main className="page-shell mx-auto max-w-4xl px-5 pb-16 pt-10 sm:pt-14 lg:px-8">
      {/* Header */}
      <div className="border-b border-pink-rose/60 pb-6 mb-8">
        <div className="inline-flex items-center gap-2 rounded-full border border-pink-rose bg-pink-soft px-3 py-1 text-xs font-semibold text-pink-deep mb-3">
          Governance, Mission & Ethics
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-ink tracking-tight">
          About <span className="text-pink-primary">BreastCare AI</span>
        </h1>
        <p className="mt-3 text-base text-muted max-w-2xl leading-relaxed">
          Advancing safe, verifiable, and transparent artificial intelligence research for early breast disease detection.
        </p>
      </div>

      <div className="space-y-8">
        {/* Mission Statement */}
        <section className="rounded-2xl border border-pink-rose/60 bg-white p-7 shadow-sm">
          <h2 className="text-xl font-bold text-ink mb-3">Our Mission</h2>
          <p className="text-sm text-slate-700 leading-relaxed">
            BreastCare AI is designed to address a critical flaw in modern medical AI: deep learning models that blindly classify
            unsuitable, corrupted, out-of-distribution, or inappropriate non-medical images without clinical validation or uncertainty estimation.
            Our platform demonstrates that medical AI must be <strong>rigorous, fail-closed, explainable, and multi-stage validated</strong> before any inference is generated.
          </p>
        </section>

        {/* Safety First Architecture */}
        <section className="rounded-2xl border border-pink-rose/60 bg-white p-7 shadow-sm">
          <h2 className="text-xl font-bold text-ink mb-3">Safety-First Design Principles</h2>
          <div className="space-y-4">
            <div className="rounded-xl bg-pink-bg p-4 border border-pink-rose/40">
              <h3 className="font-bold text-ink text-sm">1. Multi-Stage Gating Before Inference</h3>
              <p className="mt-1 text-xs text-slate-600 leading-5">
                Every upload undergoes rigorous file integrity checks, dual-modality classification (Ultrasound vs Mammography), image quality screening (blur, clipping, contrast collapse), and out-of-distribution detection. Non-medical uploads (selfies, screenshots, documents) and non-breast medical images (chest X-rays) are rejected immediately.
              </p>
            </div>
            <div className="rounded-xl bg-pink-bg p-4 border border-pink-rose/40">
              <h3 className="font-bold text-ink text-sm">2. Fail-Closed Operation (BREASTCARE_CLASSIFICATION_ENABLED=false)</h3>
              <p className="mt-1 text-xs text-slate-600 leading-5">
                Until formal multi-center prospective clinical trials and regulatory body approvals are granted, the diagnostic classification engine remains restrained by default. The system provides structural modality identification and feature interpretability while preserving human clinical primacy.
              </p>
            </div>
            <div className="rounded-xl bg-pink-bg p-4 border border-pink-rose/40">
              <h3 className="font-bold text-ink text-sm">3. Zero Data Retention & Patient Privacy</h3>
              <p className="mt-1 text-xs text-slate-600 leading-5">
                All uploaded medical image bytes are processed ephemerally in RAM. No patient imagery or diagnostic data is written to disk or transmitted to third-party services during analysis.
              </p>
            </div>
          </div>
        </section>

        {/* Clinical Disclaimer */}
        <section className="rounded-2xl border-2 border-pink-rose bg-pink-soft/50 p-6">
          <h2 className="text-base font-bold text-pink-deep mb-2">Legal & Clinical Disclaimer</h2>
          <p className="text-xs text-slate-700 leading-5">
            BreastCare AI is strictly an investigational research prototype. It is not an FDA-cleared, CE-marked, or MHRA-approved medical device. It must not be used as a substitute for professional medical advice, clinical diagnosis, or therapeutic decision-making. Individuals experiencing symptoms, palpable changes, or receiving abnormal screening results should consult a licensed physician, radiologist, or certified breast specialist immediately.
          </p>
        </section>
      </div>
    </main>
  );
}
