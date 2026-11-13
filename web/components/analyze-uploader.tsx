"use client";

import { useEffect, useRef, useState } from "react";

type ValidateImageResponse = {
  is_valid: boolean;
  detected_modality: string;
  supported_modality: string | null;
  stages_passed: string[];
  rejection_reason: string | null;
  message: string;
  quality_metrics: Record<string, number>;
  dimensions: { width: number; height: number };
};

type AnalyzeResponse = {
  request_id: string;
  model_version: string;
  detected_modality: string;
  status:
    | "unsupported_or_unsuitable"
    | "unable_to_assess"
    | "model_unavailable"
    | "research_classification_disabled"
    | "classification_available";
  input_suitability: "unable_to_assess" | "rejected" | "ultrasound_like" | "mammogram_like";
  abstention_reason: string;
  message: string;
  classification: string | null;
  confidence: number | null;
  probabilities: Record<string, number> | null;
  gradcam_overlay_url: string | null;
  explanation: string | null;
  segmentation: string | null;
  scientific_limitations: string[];
  research_disclaimer: string;
};

type StageDisplay = {
  stage: number;
  key: string;
  name: string;
  passed: boolean;
  detail: string;
};

const MAX_FILE_BYTES = 10 * 1024 * 1024; // 10MB
const UNSUPPORTED_REJECTION_TEXT = "Unsupported image. Please upload a valid breast ultrasound or mammogram.";
const UNABLE_TO_ASSESS_TEXT = "Unable to provide a reliable assessment. Specialist review is recommended.";

function isAnalyzeResponse(value: unknown): value is AnalyzeResponse {
  if (typeof value !== "object" || value === null) return false;
  const r = value as Record<string, unknown>;
  return (
    typeof r.request_id === "string" &&
    typeof r.model_version === "string" &&
    typeof r.status === "string" &&
    typeof r.message === "string"
  );
}

function buildStageBreakdown(
  validation: ValidateImageResponse | null,
  analysis: AnalyzeResponse,
): StageDisplay[] {
  const passedSet = new Set(validation?.stages_passed ?? []);
  const hasValidation = validation !== null;
  const isRejected = analysis.status === "unsupported_or_unsuitable" || analysis.input_suitability === "rejected";

  const stageDefs = [
    { stage: 1, key: "stage1_file_validation", name: "File Integrity & Format" },
    { stage: 2, key: "stage2_modality_classification", name: "Modality Classification" },
    { stage: 3, key: "stage3_quality_control", name: "Image Quality Control" },
    { stage: 4, key: "stage4_ood_assessment", name: "OOD Boundary Assessment" },
  ];

  return stageDefs.map((def) => {
    const passed = hasValidation ? passedSet.has(def.key) : !isRejected;
    let detail = passed ? "Passed" : "Blocked";
    if (!passed && analysis.abstention_reason) {
      detail = analysis.abstention_reason.replace(/_/g, " ");
    } else if (passed && def.stage === 2) {
      detail = (analysis.detected_modality || "verified").replace(/_/g, " ");
    }
    return { ...def, passed, detail };
  });
}

function formatModalityLabel(modality: string | undefined): string {
  if (!modality) return "UNKNOWN";
  return modality.replace(/_/g, " ").toUpperCase();
}

export function AnalyzeUploader() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [validation, setValidation] = useState<ValidateImageResponse | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [showGradCam, setShowGradCam] = useState(true);
  const analysisRequest = useRef<AbortController | null>(null);

  useEffect(() => () => {
    analysisRequest.current?.abort();
    analysisRequest.current = null;
  }, []);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function chooseFile(nextFile: File | null) {
    analysisRequest.current?.abort();
    analysisRequest.current = null;
    setLoading(false);
    setValidation(null);
    setResult(null);
    setError(null);

    if (!nextFile) {
      setFile(null);
      return;
    }

    if (!["image/png", "image/jpeg", "image/jpg"].includes(nextFile.type)) {
      setFile(null);
      setError(UNSUPPORTED_REJECTION_TEXT);
      return;
    }

    if (nextFile.size > MAX_FILE_BYTES) {
      setFile(null);
      setError("File exceeds 10MB limit. " + UNSUPPORTED_REJECTION_TEXT);
      return;
    }

    setFile(nextFile);
  }

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file || analysisRequest.current) return;

    const controller = new AbortController();
    analysisRequest.current = controller;
    let timedOut = false;
    const timeout = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, 60_000);

    setError(null);
    setValidation(null);
    setResult(null);
    setLoading(true);

    try {
      const baseUrl = (process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
      const analyzeForm = new FormData();
      analyzeForm.append("file", file);

      const analyzeResponse = await fetch(`${baseUrl}/api/analyze`, {
        method: "POST",
        body: analyzeForm,
        signal: controller.signal,
      });

      const analyzeBody: unknown = await analyzeResponse.json();
      if (!analyzeResponse.ok && analyzeResponse.status !== 400 && analyzeResponse.status !== 413) {
        throw new Error("The analysis service could not process this request. No classification was made.");
      }
      if (!isAnalyzeResponse(analyzeBody)) {
        throw new Error("The service returned an unexpected response. No classification was made.");
      }

      // Optionally enrich with /api/validate-image stage details (best-effort)
      let valBody: ValidateImageResponse | null = null;
      try {
        const valForm = new FormData();
        valForm.append("file", file);
        const valResponse = await fetch(`${baseUrl}/api/validate-image`, {
          method: "POST",
          body: valForm,
          signal: controller.signal,
        });
        if (valResponse.ok || valResponse.status === 400) {
          const parsed = (await valResponse.json()) as ValidateImageResponse;
          if (Array.isArray(parsed?.stages_passed)) {
            valBody = parsed;
          }
        }
      } catch {
        valBody = null;
      }

      if (analysisRequest.current === controller) {
        setValidation(valBody);
        setResult(analyzeBody);
      }
    } catch (reason) {
      if (analysisRequest.current !== controller) return;
      setError(
        timedOut
          ? "The image check timed out. No classification was made. Please try again."
          : reason instanceof TypeError
          ? "The research analysis service is currently unavailable. Ensure the backend is running."
          : reason instanceof Error
          ? reason.message
          : "The request failed. No classification was made."
      );
    } finally {
      clearTimeout(timeout);
      if (analysisRequest.current === controller) {
        analysisRequest.current = null;
        setLoading(false);
      }
    }
  }

  function downloadReport() {
    if (!result) return;
    const reportData = {
      timestamp: new Date().toISOString(),
      breastcare_version: "3.2.0",
      validation_stages: buildStageBreakdown(validation, result),
      quality_metrics: validation?.quality_metrics ?? null,
      analysis_results: result,
      clinical_disclaimer:
        "RESEARCH PROTOTYPE REPORT ONLY. This report is generated by BreastCare AI for investigational decision-support purposes and DOES NOT constitute a medical diagnosis.",
    };
    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `BreastCare_Analysis_Report_${result.request_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const isVerifiedMedical =
    result !== null &&
    (result.input_suitability === "ultrasound_like" || result.input_suitability === "mammogram_like");
  const isRejected =
    result !== null &&
    (result.status === "unsupported_or_unsuitable" || result.input_suitability === "rejected");
  const stages = result ? buildStageBreakdown(validation, result) : [];

  return (
    <div className="grid gap-8 lg:grid-cols-[1.05fr_0.95fr]">
      {/* Upload & Form Section */}
      <form onSubmit={submit} className="rounded-3xl border border-pink-rose bg-white p-6 shadow-card sm:p-8">
        <div className="mb-6 rounded-2xl border border-pink-rose bg-pink-soft/50 p-4">
          <p className="font-bold text-pink-deep text-sm flex items-center gap-2">
            <span>🎗️</span> Supported Diagnostic Modalities
          </p>
          <p className="mt-1 text-xs leading-5 text-slate-700">
            Accepts <strong>Breast Ultrasound</strong> (hypoechoic/hyperechoic B-mode) and <strong>Mammography</strong> (CC/MLO X-ray).
            Chest X-rays, photos, selfies, screenshots, and non-breast anatomical images are automatically rejected.
          </p>
        </div>

        {/* Drag and Drop Zone */}
        <label
          htmlFor="scan-upload"
          className="flex min-h-60 cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-pink-rose bg-pink-bg px-5 text-center transition hover:border-pink-primary hover:bg-pink-soft/40"
        >
          <input
            id="scan-upload"
            type="file"
            accept="image/png,image/jpeg,image/jpg"
            className="sr-only"
            onChange={(e) => chooseFile(e.target.files?.[0] ?? null)}
          />
          <div className="flex flex-col items-center">
            <span className="text-4xl mb-2">📁</span>
            <span className="font-bold text-ink text-sm sm:text-base">
              {file ? file.name : "Select or drag & drop breast scan"}
            </span>
            <span className="mt-1 text-xs text-muted">
              PNG, JPG or JPEG up to 10MB (DICOM scans converted to PNG/JPG)
            </span>
          </div>
        </label>

        {previewUrl && (
          <div className="mt-5 rounded-2xl border border-pink-rose/50 bg-pink-bg p-3 text-center">
            <p className="text-xs font-semibold text-slate-600 mb-2">Uploaded Scan Preview</p>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={previewUrl}
              alt="Scan preview"
              className="mx-auto max-h-56 rounded-xl object-contain border border-pink-rose/40"
            />
          </div>
        )}

        <div className="mt-6 flex flex-wrap items-center justify-between gap-3">
          <button
            type="submit"
            disabled={!file || loading}
            className="rounded-xl bg-pink-primary px-6 py-3 font-semibold text-white shadow-md hover:bg-pink-dark transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {loading ? (
              <>
                <span className="animate-spin text-sm">⏳</span> Verifying & Analyzing...
              </>
            ) : (
              "Run Multi-Stage Analysis"
            )}
          </button>

          {file && (
            <button
              type="button"
              onClick={() => chooseFile(null)}
              className="text-xs font-semibold text-slate-500 hover:text-pink-primary transition"
            >
              Clear Selection
            </button>
          )}
        </div>

        {error && (
          <div className="mt-5 rounded-xl border border-rose-300 bg-rose-50 p-4 text-xs font-semibold text-rose-800">
            ⚠️ {error}
          </div>
        )}
      </form>

      {/* Results & Explainability Section */}
      <div className="space-y-6">
        {result ? (
          <div className="rounded-3xl border border-pink-rose bg-white p-6 shadow-card sm:p-8 space-y-6">
            {/* Results Header */}
            <div className="flex items-center justify-between gap-3 border-b border-pink-soft pb-4">
              <div>
                <span className="text-xs font-bold uppercase tracking-wider text-muted">Assessment Result</span>
                <h3 className="text-xl font-bold text-ink mt-0.5">
                  {result.status === "classification_available"
                    ? "Research Assessment Complete"
                    : isVerifiedMedical
                    ? "Verified Medical Scan"
                    : isRejected
                    ? "Upload Rejected"
                    : "Unable to Assess"}
                </h3>
                <p className="mt-1 text-xs text-slate-600 font-medium">{result.message}</p>
              </div>
              <span
                className={`shrink-0 rounded-full px-3 py-1 text-xs font-bold ${
                  isVerifiedMedical
                    ? "bg-emerald-100 text-emerald-800 border border-emerald-300"
                    : isRejected
                    ? "bg-rose-100 text-rose-800 border border-rose-300"
                    : "bg-amber-100 text-amber-800 border border-amber-300"
                }`}
              >
                {formatModalityLabel(result.detected_modality)}
              </span>
            </div>

            {/* Rejection / Abstention Messaging */}
            {result.status !== "classification_available" && (
              <div
                className={`rounded-2xl p-4 border text-sm font-semibold ${
                  isRejected
                    ? "bg-rose-50 border-rose-200 text-rose-900"
                    : "bg-amber-50 border-amber-200 text-amber-900"
                }`}
              >
                {isRejected ? UNSUPPORTED_REJECTION_TEXT : UNABLE_TO_ASSESS_TEXT}
                {result.abstention_reason && result.abstention_reason !== "none" && (
                  <p className="mt-1 text-xs font-normal text-slate-600">
                    Reason code: {result.abstention_reason}
                  </p>
                )}
              </div>
            )}

            {/* Research Classification (only if explicitly server-enabled) */}
            {result.status === "classification_available" && result.classification && (
              <div className="rounded-2xl border border-pink-rose bg-pink-soft/50 p-4 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold uppercase tracking-wider text-pink-deep">
                    Research Pattern Output
                  </span>
                  <span className="text-sm font-bold text-ink">{result.classification}</span>
                </div>
                {result.probabilities && (
                  <div className="grid grid-cols-2 gap-2 pt-1 text-xs">
                    {Object.entries(result.probabilities).map(([label, prob]) => (
                      <div key={label} className="rounded-lg bg-white px-3 py-2 border border-pink-rose/40 flex justify-between">
                        <span className="font-semibold text-slate-700">{label}</span>
                        <span className="font-mono text-pink-deep">{(prob * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Validation Stages Breakdown */}
            <div className="space-y-2">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
                Multi-Stage Safety Gate
              </span>
              <div className="space-y-2">
                {stages.map((stage) => (
                  <div
                    key={stage.stage}
                    className="flex items-center justify-between rounded-xl bg-pink-bg p-3 text-xs border border-pink-rose/30"
                  >
                    <span className="font-semibold text-slate-700 flex items-center gap-2">
                      <span>{stage.passed ? "✅" : "❌"}</span>
                      <span>Stage {stage.stage}: {stage.name}</span>
                    </span>
                    <span className="text-slate-500 capitalize">{stage.detail}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Grad-CAM Heatmap Visualization */}
            {result.gradcam_overlay_url && (
              <div className="rounded-2xl border border-pink-rose/50 bg-pink-bg p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-ink">Deep Feature Interpretability (Layer4 Grad-CAM)</span>
                  <button
                    type="button"
                    onClick={() => setShowGradCam(!showGradCam)}
                    className="text-xs font-semibold text-pink-deep hover:underline"
                  >
                    {showGradCam ? "Hide Heatmap" : "Show Heatmap"}
                  </button>
                </div>

                {showGradCam && (
                  <div>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={result.gradcam_overlay_url}
                      alt="Grad-CAM Heatmap"
                      className="w-full rounded-xl border border-pink-rose/40 max-h-64 object-contain mx-auto bg-black"
                    />
                    <p className="mt-2 text-[11px] leading-4 text-slate-500 italic">
                      Notice: Feature attention heatmaps reflect deep convolutional neural activation clusters. They
                      do not represent certified tumor boundaries or pathological margins.
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* Scientific Limitations */}
            {result.scientific_limitations?.length > 0 && (
              <div className="rounded-xl bg-white p-4 border border-pink-rose/40 text-xs text-slate-600 space-y-1.5">
                <p className="font-bold text-ink">Scientific Limitations</p>
                <ul className="list-disc list-inside space-y-1">
                  {result.scientific_limitations.map((item, idx) => (
                    <li key={idx}>{item}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Research Disclaimer & Fail-Closed Status */}
            <div className="rounded-xl bg-pink-soft/60 p-4 border border-pink-rose text-xs text-slate-700 space-y-2">
              <p className="font-bold text-pink-deep">⚠️ Protocol Governance Notice</p>
              <p>{result.research_disclaimer}</p>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-3">
              <button
                type="button"
                onClick={downloadReport}
                className="flex-1 rounded-xl border border-pink-rose bg-white px-4 py-2.5 text-xs font-semibold text-pink-deep hover:bg-pink-soft/50 transition shadow-sm"
              >
                📥 Download Research Report (.JSON)
              </button>
            </div>
          </div>
        ) : (
          /* Empty State */
          <div className="rounded-3xl border border-pink-rose/50 bg-white p-8 text-center shadow-card text-muted flex flex-col items-center justify-center min-h-[360px]">
            <span className="text-5xl mb-3">🩺</span>
            <h4 className="font-bold text-ink text-base">Awaiting Medical Image Upload</h4>
            <p className="mt-2 text-xs max-w-sm leading-5">
              Select or drop a breast ultrasound or mammography scan on the left to execute the 4-stage validation and Grad-CAM interpretability pipeline.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
