export default function CancerInfoPage() {
  return (
    <main className="page-shell mx-auto max-w-5xl px-5 pb-16 pt-10 sm:pt-14 lg:px-8">
      {/* Page Header */}
      <div className="border-b border-pink-rose/60 pb-6 mb-8">
        <div className="inline-flex items-center gap-2 rounded-full border border-pink-rose bg-pink-soft px-3 py-1 text-xs font-semibold text-pink-deep mb-3">
          Medical Education · Peer-Reviewed Oncology Sources
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-ink tracking-tight">
          Understanding <span className="text-pink-primary">Breast Cancer</span>
        </h1>
        <p className="mt-3 text-base text-muted max-w-3xl leading-relaxed">
          Breast cancer is the most diagnosed cancer worldwide in women, yet early detection dramatically
          improves long-term survival. This guide synthesizes educational consensus from the National Cancer Institute (NCI),
          the American College of Radiology (ACR), and the American Society of Clinical Oncology (ASCO).
        </p>
      </div>

      <div className="grid gap-10">
        {/* Section 1: Overview */}
        <section className="rounded-2xl border border-pink-rose/60 bg-white p-7 shadow-sm">
          <h2 className="text-2xl font-bold text-ink mb-4 flex items-center gap-2">
            <span className="text-pink-primary">1.</span> Pathophysiology & Biology
          </h2>
          <p className="text-sm text-slate-700 leading-relaxed">
            Breast cancer develops when epithelial cells lining the ductal or lobular units of breast tissue acquire
            genetic and epigenetic alterations, leading to unregulated proliferation, evasion of apoptosis, and tissue invasion.
          </p>
          <div className="mt-5 grid sm:grid-cols-2 gap-4">
            <div className="rounded-xl bg-pink-bg p-4 border border-pink-rose/40">
              <h3 className="font-bold text-ink text-sm">Ductal Carcinoma</h3>
              <p className="mt-1 text-xs text-slate-600 leading-5">
                Originates in the milk ducts (tubes carrying milk to the nipple). Accounts for approximately 70–80% of all breast cancers, either as non-invasive Ductal Carcinoma In Situ (DCIS) or Invasive Ductal Carcinoma (IDC).
              </p>
            </div>
            <div className="rounded-xl bg-pink-bg p-4 border border-pink-rose/40">
              <h3 className="font-bold text-ink text-sm">Lobular Carcinoma</h3>
              <p className="mt-1 text-xs text-slate-600 leading-5">
                Originates in the milk-producing glands (lobules). Invasive Lobular Carcinoma (ILC) accounts for about 10–15% of cases and often presents with diffuse, non-distinct architectural distortion.
              </p>
            </div>
          </div>
        </section>

        {/* Section 2: Benign vs Malignant */}
        <section className="rounded-2xl border border-pink-rose/60 bg-white p-7 shadow-sm">
          <h2 className="text-2xl font-bold text-ink mb-4 flex items-center gap-2">
            <span className="text-pink-primary">2.</span> Benign vs. Malignant Lesions
          </h2>
          <p className="text-sm text-slate-700 leading-relaxed mb-4">
            The majority of breast lumps and imaging abnormalities are benign (non-cancerous). Proper diagnostic evaluation
            distinguishes harmless changes from malignancies requiring intervention.
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left text-slate-700 border-collapse">
              <thead>
                <tr className="bg-pink-soft text-pink-deep border-b border-pink-rose">
                  <th className="p-3 font-semibold">Lesion Type</th>
                  <th className="p-3 font-semibold">Key Ultrasound Features</th>
                  <th className="p-3 font-semibold">Key Mammography Features</th>
                  <th className="p-3 font-semibold">Typical Clinical Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-pink-100">
                <tr className="hover:bg-pink-50/50">
                  <td className="p-3 font-bold text-ink">Simple Cyst (Benign)</td>
                  <td className="p-3">Anechoic (black), round/oval, circumscribed margins, posterior acoustic enhancement</td>
                  <td className="p-3">Round or oval circumscribed mass with uniform low-to-medium density</td>
                  <td className="p-3 text-emerald-700 font-medium">BI-RADS 2: Routine screening, no biopsy needed</td>
                </tr>
                <tr className="hover:bg-pink-50/50">
                  <td className="p-3 font-bold text-ink">Fibroadenoma (Benign)</td>
                  <td className="p-3">Hypoechoic, wider-than-tall, circumscribed or macrolobulated margins</td>
                  <td className="p-3">Circumscribed mass, may have coarse &quot;popcorn&quot; calcifications</td>
                  <td className="p-3 text-emerald-700 font-medium">BI-RADS 2 or 3: Routine or short-interval follow-up</td>
                </tr>
                <tr className="hover:bg-pink-50/50">
                  <td className="p-3 font-bold text-ink">Invasive Carcinoma (Malignant)</td>
                  <td className="p-3">Hypoechoic mass, taller-than-wide, angular/spiculated margins, posterior shadowing</td>
                  <td className="p-3">Spiculated high-density mass, pleomorphic microcalcifications, architectural distortion</td>
                  <td className="p-3 text-rose-700 font-bold">BI-RADS 4/5: Core needle biopsy required for histology</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Section 3: Signs & Symptoms */}
        <section className="rounded-2xl border border-pink-rose/60 bg-white p-7 shadow-sm">
          <h2 className="text-2xl font-bold text-ink mb-4 flex items-center gap-2">
            <span className="text-pink-primary">3.</span> Warning Signs & Symptoms
          </h2>
          <p className="text-sm text-slate-700 leading-relaxed mb-4">
            While early-stage breast cancer often presents no symptoms (which is why routine imaging screening is vital),
            individuals should be aware of physical signs warranting prompt medical evaluation:
          </p>
          <div className="grid sm:grid-cols-3 gap-4">
            {[
              {
                title: "Palpable Lump or Mass",
                desc: "A hard, painless lump with irregular edges is most common, though tender or round lumps can also be cancerous.",
              },
              {
                title: "Skin Changes & Dimpling",
                desc: "Skin retraction, pitting or puckering (like an orange peel, termed 'peau d'orange'), redness, or scaly texture.",
              },
              {
                title: "Nipple Alterations",
                desc: "Inversion or retraction of the nipple, spontaneous clear or bloody discharge, or persistent scaling and itching.",
              },
            ].map((symptom, i) => (
              <div key={i} className="rounded-xl border border-pink-rose/40 bg-pink-bg/60 p-4">
                <h4 className="font-bold text-ink text-sm">{symptom.title}</h4>
                <p className="mt-1.5 text-xs text-slate-600 leading-5">{symptom.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Section 4: Risk Factors & Screening Guidelines */}
        <section className="rounded-2xl border border-pink-rose/60 bg-white p-7 shadow-sm">
          <h2 className="text-2xl font-bold text-ink mb-4 flex items-center gap-2">
            <span className="text-pink-primary">4.</span> Risk Factors & Recommended Screening Guidelines
          </h2>
          <div className="grid sm:grid-cols-2 gap-6">
            <div>
              <h3 className="font-bold text-ink text-sm mb-2">Key Risk Factors</h3>
              <ul className="space-y-2 text-xs text-slate-700">
                <li className="flex gap-2">
                  <span className="text-pink-primary font-bold">•</span>
                  <span><strong>Genetic Predisposition:</strong> Pathogenic germline mutations in <em>BRCA1</em>, <em>BRCA2</em>, <em>PALB2</em>, <em>TP53</em>, or <em>PTEN</em>.</span>
                </li>
                <li className="flex gap-2">
                  <span className="text-pink-primary font-bold">•</span>
                  <span><strong>Family History:</strong> First-degree relative (mother, sister, daughter) diagnosed before age 50.</span>
                </li>
                <li className="flex gap-2">
                  <span className="text-pink-primary font-bold">•</span>
                  <span><strong>High Breast Density:</strong> Dense fibroglandular tissue obscures mammographic lesions and increases independent risk.</span>
                </li>
                <li className="flex gap-2">
                  <span className="text-pink-primary font-bold">•</span>
                  <span><strong>Hormonal Exposure:</strong> Early menarche (&lt;12), late menopause (&gt;55), nulliparity, or prolonged combined hormone replacement therapy.</span>
                </li>
              </ul>
            </div>
            <div>
              <h3 className="font-bold text-ink text-sm mb-2">Clinical Screening Guidelines</h3>
              <ul className="space-y-2 text-xs text-slate-700">
                <li className="flex gap-2">
                  <span className="text-pink-primary font-bold">•</span>
                  <span><strong>USPSTF & ACS (Average Risk):</strong> Biennial or annual screening mammography starting at age 40 through age 74.</span>
                </li>
                <li className="flex gap-2">
                  <span className="text-pink-primary font-bold">•</span>
                  <span><strong>Dense Breasts (ACR):</strong> Supplemental screening with Automated Whole-Breast Ultrasound or Contrast-Enhanced MRI.</span>
                </li>
                <li className="flex gap-2">
                  <span className="text-pink-primary font-bold">•</span>
                  <span><strong>High-Risk Individuals (Lifetime risk &gt;20%):</strong> Annual breast MRI screening combined with mammography starting at age 30.</span>
                </li>
              </ul>
            </div>
          </div>
        </section>

        {/* References Callout */}
        <div className="rounded-xl bg-pink-soft/80 border border-pink-rose p-5 text-xs text-slate-700 space-y-2">
          <p className="font-bold text-pink-deep">Authoritative Clinical Sources:</p>
          <ul className="list-disc list-inside space-y-1">
            <li>National Cancer Institute (NCI) Physician Data Query (PDQ) Breast Cancer Screening & Treatment, 2024.</li>
            <li>American College of Radiology (ACR) BI-RADS® Atlas 5th Edition: Ultrasound and Mammography lexicon.</li>
            <li>American Society of Clinical Oncology (ASCO) Clinical Practice Guidelines on Breast Cancer Management.</li>
          </ul>
        </div>
      </div>
    </main>
  );
}

