import type { Metadata } from "next";
import { AnalyzeUploader } from "@/components/analyze-uploader";

export const metadata: Metadata = {
  title: "AI Medical Image Analysis | BreastCare AI",
  description: "Multi-stage validation and deep learning feature analysis for breast ultrasound and mammography imaging.",
};

export default function AnalyzePage() {
  return (
    <main className="page-shell mx-auto max-w-7xl px-5 py-10 sm:py-14 lg:px-8">
      <div className="border-b border-pink-rose/50 pb-6 mb-8">
        <div className="inline-flex items-center gap-2 rounded-full border border-pink-rose bg-pink-soft px-3 py-1 text-xs font-semibold text-pink-deep mb-3">
          Clinical Decision-Support Research
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-ink">
          Dual-Modality <span className="text-pink-primary">Breast Imaging Analysis</span>
        </h1>
        <p className="mt-3 max-w-3xl text-sm leading-relaxed text-muted">
          Upload a breast ultrasound or mammography scan for 4-stage validation (file bounds, modality discrimination,
          quality control, and out-of-distribution assessment) and deep convolutional Grad-CAM interpretability.
        </p>
      </div>

      <div>
        <AnalyzeUploader />
      </div>
    </main>
  );
}
