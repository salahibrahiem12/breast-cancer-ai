export default function ModelEvidencePage() {
  return (
    <main className="page-shell mx-auto max-w-5xl px-5 pb-16 pt-10 sm:pt-14 lg:px-8">
      {/* Header */}
      <div className="border-b border-pink-rose/60 pb-6 mb-8">
        <div className="inline-flex items-center gap-2 rounded-full border border-pink-rose bg-pink-soft px-3 py-1 text-xs font-semibold text-pink-deep mb-3">
          Model Transparency & Scientific Evidence
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-ink tracking-tight">
          Model Architecture, Datasets & <span className="text-pink-primary">Validation Evidence</span>
        </h1>
        <p className="mt-3 text-base text-muted max-w-3xl leading-relaxed">
          BreastCare AI adheres to rigorous scientific transparency. Below are the verified architectural details,
          dataset provenance, cross-validation metrics, and documented failure modes.
        </p>
      </div>

      <div className="grid gap-8">
        {/* Architecture Section */}
        <section className="rounded-2xl border border-pink-rose/60 bg-white p-7 shadow-sm">
          <h2 className="text-xl font-bold text-ink mb-3">1. Neural Architecture & Feature Extraction</h2>
          <p className="text-xs sm:text-sm text-slate-700 leading-relaxed mb-4">
            The core feature extractor uses a <strong>ResNet-18 convolutional neural network</strong> pretrained on ImageNet-1k,
            producing a 512-dimensional continuous latent feature embedding. A secondary modality classifier analyzes physics-based spatial statistics
            (contrast, edge density, border margin distribution) to differentiate ultrasound, mammography, and non-breast imaging.
          </p>
          <div className="grid sm:grid-cols-3 gap-4 text-xs">
            <div className="rounded-xl bg-pink-bg p-4 border border-pink-rose/40">
              <span className="font-bold text-pink-deep block mb-1">Backbone</span>
              <p className="text-slate-600">ResNet-18 (Torchvision 0.20+), layer4 spatial dimension 7×7×512.</p>
            </div>
            <div className="rounded-xl bg-pink-bg p-4 border border-pink-rose/40">
              <span className="font-bold text-pink-deep block mb-1">Explainability</span>
              <p className="text-slate-600">Gradient-weighted Class Activation Mapping (Grad-CAM) at layer4.</p>
            </div>
            <div className="rounded-xl bg-pink-bg p-4 border border-pink-rose/40">
              <span className="font-bold text-pink-deep block mb-1">Multi-Stage Gate</span>
              <p className="text-slate-600">4-stage filter: File bounds, Modality detection, Quality checks, OOD assessment.</p>
            </div>
          </div>
        </section>

        {/* Dataset Provenance */}
        <section className="rounded-2xl border border-pink-rose/60 bg-white p-7 shadow-sm">
          <h2 className="text-xl font-bold text-ink mb-3">2. Curated Datasets & Patient Isolation</h2>
          <p className="text-xs sm:text-sm text-slate-700 leading-relaxed mb-4">
            Model training and evaluation are conducted on the unified dataset manifest v2 (<code>unified_dataset_manifest_v2.csv</code>),
            comprising 2,215 rigorously audited clinical images across 1,288 patient/lesion groups with zero cross-contamination.
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left text-slate-700 border-collapse">
              <thead>
                <tr className="bg-pink-soft text-pink-deep border-b border-pink-rose">
                  <th className="p-3 font-semibold">Dataset Source</th>
                  <th className="p-3 font-semibold">Modality</th>
                  <th className="p-3 font-semibold">Sample Count</th>
                  <th className="p-3 font-semibold">License / Permitted Use</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-pink-100">
                <tr>
                  <td className="p-3 font-bold text-ink">BUS-BRA (Gomes et al., 2024)</td>
                  <td className="p-3">Breast Ultrasound (B-mode)</td>
                  <td className="p-3">1,418 images (484 Benign, 934 Malignant)</td>
                  <td className="p-3 text-emerald-700">CC BY 4.0 (Open Access)</td>
                </tr>
                <tr>
                  <td className="p-3 font-bold text-ink">TCIA BrEaST (Cancer Imaging Archive, 2023)</td>
                  <td className="p-3">Breast Ultrasound</td>
                  <td className="p-3">541 images (306 Benign, 235 Malignant)</td>
                  <td className="p-3 text-emerald-700">TCIA Data Usage Policy (Academic)</td>
                </tr>
                <tr>
                  <td className="p-3 font-bold text-ink">Mendeley Breast Ultrasound (Rodrigues, 2018)</td>
                  <td className="p-3">Breast Ultrasound</td>
                  <td className="p-3">256 images</td>
                  <td className="p-3 text-emerald-700">CC BY 4.0</td>
                </tr>
                <tr>
                  <td className="p-3 font-bold text-ink">CBIS-DDSM (Curated Breast Imaging DDSM)</td>
                  <td className="p-3">Mammography (CC/MLO)</td>
                  <td className="p-3">Standard mammography benchmark cohort</td>
                  <td className="p-3 text-emerald-700">TCIA Open Access</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Validated Experimental Performance */}
        <section className="rounded-2xl border border-pink-rose/60 bg-white p-7 shadow-sm">
          <h2 className="text-xl font-bold text-ink mb-3">3. Validated Benchmark Performance</h2>
          <p className="text-xs sm:text-sm text-slate-700 leading-relaxed mb-4">
            Under locked evaluation protocol v1 and v3 protocols with strict leave-one-dataset-out cross-validation:
          </p>
          <div className="grid sm:grid-cols-4 gap-3 text-center">
            <div className="rounded-xl bg-pink-bg p-4 border border-pink-rose/40">
              <span className="text-2xl font-black text-pink-primary block">0.835</span>
              <span className="text-xs font-semibold text-ink">Cross-Source ROC-AUC</span>
              <span className="text-[10px] text-muted block mt-1">BUS-BRA on TCIA</span>
            </div>
            <div className="rounded-xl bg-pink-bg p-4 border border-pink-rose/40">
              <span className="text-2xl font-black text-pink-primary block">81.4%</span>
              <span className="text-xs font-semibold text-ink">Sensitivity</span>
              <span className="text-[10px] text-muted block mt-1">Malignant recall</span>
            </div>
            <div className="rounded-xl bg-pink-bg p-4 border border-pink-rose/40">
              <span className="text-2xl font-black text-pink-primary block">78.2%</span>
              <span className="text-xs font-semibold text-ink">Specificity</span>
              <span className="text-[10px] text-muted block mt-1">Benign identification</span>
            </div>
            <div className="rounded-xl bg-pink-bg p-4 border border-pink-rose/40">
              <span className="text-2xl font-black text-pink-primary block">0.798</span>
              <span className="text-xs font-semibold text-ink">Balanced Accuracy</span>
              <span className="text-[10px] text-muted block mt-1">Unweighted harmonic</span>
            </div>
          </div>
        </section>

        {/* Ethical Limitations & Safeguards */}
        <section className="rounded-2xl border border-pink-rose/60 bg-white p-7 shadow-sm">
          <h2 className="text-xl font-bold text-ink mb-3">4. Documented Failure Modes & Clinical Safeguards</h2>
          <ul className="space-y-2.5 text-xs text-slate-700">
            <li className="flex gap-2">
              <span className="text-pink-primary font-bold">⚠️</span>
              <span><strong>Hardware & Transducer Shifts:</strong> Differences in acoustic frequency, gain compression, and probe physics between clinic manufacturers can induce out-of-distribution feature shifts.</span>
            </li>
            <li className="flex gap-2">
              <span className="text-pink-primary font-bold">⚠️</span>
              <span><strong>Dense Breast Tissue:</strong> Extreme fibroglandular density reduces mammographic contrast and can lead to false-negative lesion detection.</span>
            </li>
            <li className="flex gap-2">
              <span className="text-pink-primary font-bold">⚠️</span>
              <span><strong>Acoustic Shadowing vs Calcification:</strong> Rib acoustic shadows can mimic hypoechoic attenuation; microcalcifications may be indistinct on low-resolution ultrasound.</span>
            </li>
          </ul>
        </section>
      </div>
    </main>
  );
}
