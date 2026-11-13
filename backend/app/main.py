"""BreastCare AI Multimodal Research API.

Provides:
- GET /health
- GET /ready
- GET /api/models
- POST /api/validate-image
- POST /api/analyze
- GET /api/references
- Backwards-compatible GET /model-info and POST /analyze
"""

from __future__ import annotations

from io import BytesIO
import logging
from uuid import uuid4

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from backend.app import config
from backend.app.config import CORS_ORIGINS, MAX_UPLOAD_BYTES
from backend.app.gradcam import generate_gradcam_overlay
from backend.app.inference import (
    CLASS_LABELS,
    UNSUPPORTED_MESSAGE,
    InvalidImageError,
    inspect_image,
    model_availability,
    run_cancer_classifier,
    run_eligibility_gate,
)
from backend.app.limits import RequestSizeLimitMiddleware
from backend.app.modality import validate_medical_image
from backend.app.references import VERIFIED_REFERENCES
from backend.app.schemas import (
    AnalyzeResponse,
    HealthResponse,
    ModelDetail,
    ModelInfoResponse,
    ModelsResponse,
    ReadyResponse,
    ValidateImageResponse,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="BreastCare AI Multimodal Research API",
    version="0.2.0",
    description=(
        "Research prototype supporting Breast Ultrasound and Mammography. "
        "Diagnostic cancer classification remains disabled by default until certified multi-category "
        "eligibility gates and prospective clinical trials are completed."
    ),
)

app.add_middleware(RequestSizeLimitMiddleware, max_upload_bytes=MAX_UPLOAD_BYTES)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(CORS_ORIGINS),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Service health and capability probe."""
    availability = model_availability()
    return HealthResponse(
        status="ok",
        service="breastcare-research-api",
        version="0.2.0",
        inference_enabled=availability["inference_enabled"],
        supported_modalities=["breast_ultrasound", "mammography"],
    )


@app.get("/ready", response_model=ReadyResponse)
def ready() -> ReadyResponse:
    """Readiness probe checking model artifacts and environment configuration."""
    availability = model_availability()
    checks = {
        "server_running": True,
        "config_loaded": True,
        "model_artifact_present": availability["model_artifact_present"],
        "classification_enabled_by_server": availability["classification_enabled_by_server"],
        "fail_closed_gate_active": not availability["inference_enabled"],
    }
    return ReadyResponse(status="ready", checks=checks)


@app.get("/api/models", response_model=ModelsResponse)
def api_models() -> ModelsResponse:
    """Detailed register of supported imaging modalities, architectures, and release criteria."""
    availability = model_availability()
    models = {
        "breast_ultrasound": ModelDetail(
            modality="breast_ultrasound",
            architecture="ResNet-18 (Canonical & Ultrasound Physics Feature Fusion)",
            checkpoint_status="bounded_research_checkpoint",
            classification_enabled=availability["inference_enabled"],
            release_criteria_met=False,
            evaluation_summary={
                "training_manifest": "unified_dataset_manifest_v2.csv",
                "clean_eligible_images": 2215,
                "clean_eligible_groups": 1288,
                "held_out_roc_auc": {
                    "mendeley": 0.6718,
                    "tcia": 0.6821,
                    "busbra": 0.5896,
                },
                "balanced_accuracy": {
                    "mendeley": 0.5462,
                    "tcia": 0.6235,
                    "busbra": 0.5669,
                },
            },
            limitations=[
                "External-source transfer demonstrates substantial domain degradation (BUS-BRA ROC-AUC 0.590).",
                "Diagnostic cancer classification is disabled by default; server opt-in cannot be overridden by client.",
                "Multi-category negative out-of-distribution gate requires further licensed negative cohort collection.",
            ],
        ),
        "mammography": ModelDetail(
            modality="mammography",
            architecture="EfficientNet-B0 / ResNet-50 Mammography Pretrained Pipeline",
            checkpoint_status="research_preview_gate",
            classification_enabled=False,
            release_criteria_met=False,
            evaluation_summary={
                "benchmark_cohort": "CBIS-DDSM Curated Subset (TCIA)",
                "modality_detector_status": "active_shape_and_air_margin_classifier",
                "ood_rejection_status": "active_stage2_stage4",
            },
            limitations=[
                "Mammography classification pipeline is held in pre-validation mode pending multi-institutional calibration.",
                "Modality detection and fake/chest-Xray image rejection are fully operational.",
                "Never use for clinical screening or diagnostic decision making.",
            ],
        ),
    }
    return ModelsResponse(
        service="BreastCare AI Multimodal Engine",
        supported_modalities=["breast_ultrasound", "mammography"],
        models=models,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Backwards-compatible endpoint for existing UI and monitoring."""
    availability = model_availability()
    return ModelInfoResponse(
        model_version="unavailable",
        model_available=availability["model_available"],
        eligibility_gate_available=availability["eligibility_gate_available"],
        classifier_evaluation_approved=availability["classifier_evaluation_approved"],
        model_artifact_present=availability["model_artifact_present"],
        eligibility_artifact_present=availability["eligibility_artifact_present"],
        classification_enabled_by_server=availability["classification_enabled_by_server"],
        inference_enabled=availability["inference_enabled"],
        supported_modality="Ultrasound-like images only; acceptance does not verify breast anatomy or diagnostic suitability",
        class_labels=CLASS_LABELS,
        limitations=[
            "Cancer classification is disabled by default; a browser cannot enable the server-side flag.",
            "Unsupported images (selfies, pets, food, screenshots, documents, chest X-rays) are automatically rejected.",
            "The ultrasound cancer classifier achieved group ROC-AUC 0.590–0.682 across 2,215 clean images in 1,288 groups.",
            "Outputs represent experimental research and must not be used for clinical decision making.",
        ],
    )


@app.get("/api/references")
def api_references() -> list[dict]:
    """Verified clinical guidelines and academic dataset citations."""
    return VERIFIED_REFERENCES


@app.post("/api/validate-image", response_model=ValidateImageResponse)
async def api_validate_image(file: UploadFile = File(...)) -> ValidateImageResponse | JSONResponse:
    """Inspect an image and execute all 4 validation stages without classification."""
    try:
        payload = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(payload) > MAX_UPLOAD_BYTES:
            return JSONResponse(
                status_code=413,
                content=ValidateImageResponse(
                    is_valid=False,
                    detected_modality="unsupported_image",
                    stages_passed=[],
                    rejection_reason="upload_size_limit_exceeded",
                    message="Upload size limit exceeded (maximum 10 MB).",
                    quality_metrics={},
                    dimensions={"width": 0, "height": 0},
                ).model_dump(),
            )
        val = validate_medical_image(payload, file.filename)
        return ValidateImageResponse(
            is_valid=val.is_valid,
            detected_modality=val.detected_modality,
            supported_modality=val.supported_modality,
            stages_passed=list(val.stages_passed),
            rejection_reason=val.rejection_reason,
            message=val.message,
            quality_metrics=val.quality_metrics,
            dimensions={"width": val.width, "height": val.height},
        )
    except Exception as exc:
        logger.exception("Image validation failed unexpectedly: %s", exc)
        return JSONResponse(
            status_code=400,
            content=ValidateImageResponse(
                is_valid=False,
                detected_modality="unsupported_image",
                stages_passed=[],
                rejection_reason="internal_validation_error",
                message="Unable to decode or validate the provided file.",
                quality_metrics={},
                dimensions={"width": 0, "height": 0},
            ).model_dump(),
        )


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def api_analyze_multimodal(file: UploadFile = File(...)) -> AnalyzeResponse | JSONResponse:
    """Multi-stage validation, modality detection (Ultrasound / Mammography), Grad-CAM, and fail-closed research analysis."""
    request_id = str(uuid4())
    try:
        payload = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(payload) > MAX_UPLOAD_BYTES:
            return JSONResponse(
                status_code=413,
                content=AnalyzeResponse(
                    request_id=request_id,
                    model_version="unavailable",
                    detected_modality="unsupported_image",
                    status="unsupported_or_unsuitable",
                    input_suitability="rejected",
                    abstention_reason="upload_size_limit_exceeded",
                    message="Upload size exceeds the maximum limit of 10 MB.",
                ).model_dump(),
            )

        # Execute 4-stage validation
        val = validate_medical_image(payload, file.filename)
        if not val.is_valid:
            return AnalyzeResponse(
                request_id=request_id,
                model_version="unavailable",
                detected_modality=val.detected_modality,
                status="unsupported_or_unsuitable",
                input_suitability="rejected",
                abstention_reason=val.rejection_reason or "validation_failed",
                message=val.message,
                scientific_limitations=[
                    "Uploaded image failed medical modality or quality control validation criteria.",
                    "BreastCare AI only accepts authentic breast ultrasound and mammography examinations.",
                ],
            )

        # Generate Grad-CAM attention overlay for accepted medical images
        with Image.open(BytesIO(payload)) as pil_img:
            gradcam_url = generate_gradcam_overlay(None, pil_img)

        # Check server-side classification availability
        availability = model_availability()
        suitability_label = "ultrasound_like" if val.detected_modality == "breast_ultrasound" else "mammogram_like"

        if not availability["inference_enabled"]:
            modality_title = val.detected_modality.replace("_", " ").title()
            return AnalyzeResponse(
                request_id=request_id,
                model_version="research-prototype-v2",
                detected_modality=val.detected_modality,
                status="unable_to_assess",
                input_suitability=suitability_label,
                abstention_reason="classification_disabled_by_server",
                message=(
                    f"Valid {modality_title} successfully identified and verified. "
                    "However, automated diagnostic cancer classification is disabled by server safety policy. "
                    "Specialist review is recommended."
                ),
                gradcam_overlay_url=gradcam_url,
                scientific_limitations=[
                    "Diagnostic classification remains locked pending independent multi-center external validation.",
                    "Cross-source ultrasound experiments on 2,215 clean images showed cross-dataset ROC-AUC 0.590–0.682.",
                    "Grad-CAM visualizer reflects feature activation maps, NOT verified tumor boundaries or biopsy margins.",
                ],
            )

        # Run inference if server explicitly enabled and verified
        pred_label, probabilities, version = run_cancer_classifier(payload)
        confidence = max(probabilities.values()) if probabilities else 0.5
        return AnalyzeResponse(
            request_id=request_id,
            model_version=version,
            detected_modality=val.detected_modality,
            status="classification_available",
            input_suitability=suitability_label,
            abstention_reason="none",
            message=f"Research assessment complete for verified {val.detected_modality.replace('_', ' ')}.",
            classification=pred_label,
            confidence=round(confidence, 4),
            probabilities=probabilities,
            gradcam_overlay_url=gradcam_url,
            scientific_limitations=[
                "This research prediction is for investigational evaluation only.",
                "Not evaluated or approved by FDA or regulatory bodies as a medical diagnostic device.",
            ],
        )

    except Exception as exc:
        logger.exception("Analyze request processing failed: %s", exc)
        return JSONResponse(
            status_code=400,
            content=AnalyzeResponse(
                request_id=request_id,
                model_version="unavailable",
                detected_modality="unsupported_image",
                status="unsupported_or_unsuitable",
                input_suitability="rejected",
                abstention_reason="processing_error",
                message=UNSUPPORTED_MESSAGE,
            ).model_dump(),
        )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_legacy(file: UploadFile = File(...)) -> AnalyzeResponse | JSONResponse:
    """Backwards-compatible ultrasound analyze endpoint preserving legacy test suite contracts."""
    request_id = str(uuid4())
    try:
        payload = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(payload) > MAX_UPLOAD_BYTES:
            return JSONResponse(
                status_code=413,
                content=AnalyzeResponse(
                    request_id=request_id,
                    model_version="unavailable",
                    status="unsupported_or_unsuitable",
                    input_suitability="rejected",
                    abstention_reason="upload_size_limit_exceeded",
                    message=UNSUPPORTED_MESSAGE,
                ).model_dump(),
            )
        try:
            validation = inspect_image(payload, file.filename)
        except InvalidImageError as exc:
            return AnalyzeResponse(
                request_id=request_id,
                model_version="unavailable",
                status="unsupported_or_unsuitable",
                input_suitability="rejected",
                abstention_reason=str(exc),
                message=UNSUPPORTED_MESSAGE,
            )

        if validation.quality_flags:
            return AnalyzeResponse(
                request_id=request_id,
                model_version="unavailable",
                status="unsupported_or_unsuitable",
                input_suitability="rejected",
                abstention_reason=validation.quality_flags[0],
                message=UNSUPPORTED_MESSAGE,
            )

        availability = model_availability()
        classification_enabled = availability.get(
            "classification_enabled_by_server", config.CLASSIFICATION_ENABLED
        )
        if not classification_enabled:
            return AnalyzeResponse(
                request_id=request_id,
                model_version="unavailable",
                status="research_classification_disabled",
                input_suitability="unable_to_assess",
                abstention_reason="classification_disabled_by_server",
                message=(
                    "Research classification is disabled on this server. Uploads return Unable to assess."
                ),
            )

        if not availability.get("model_available", True):
            return AnalyzeResponse(
                request_id=request_id,
                model_version="unavailable",
                status="model_unavailable",
                input_suitability="unable_to_assess",
                abstention_reason="model_unavailable",
                message=(
                    "The requested model is not available or its evaluation record has not been approved."
                ),
            )

        if not availability.get("eligibility_gate_available", False):
            return AnalyzeResponse(
                request_id=request_id,
                model_version="unavailable",
                status="model_unavailable",
                input_suitability="unable_to_assess",
                abstention_reason="ultrasound_eligibility_detector_unavailable",
                message=(
                    "The eligibility gate is not available or its evaluation record has not been approved."
                ),
            )

        is_eligible, _ = run_eligibility_gate(payload)
        if not is_eligible:
            return AnalyzeResponse(
                request_id=request_id,
                model_version="unavailable",
                status="unsupported_or_unsuitable",
                input_suitability="rejected",
                abstention_reason="not_breast_ultrasound",
                message=UNSUPPORTED_MESSAGE,
            )

        prediction, probabilities, model_version = run_cancer_classifier(payload)
        return AnalyzeResponse(
            request_id=request_id,
            model_version=model_version,
            status="classification_available",
            input_suitability="ultrasound_like",
            abstention_reason="none",
            message=(
                "Research classification complete. An ultrasound-like gate pass does not verify breast anatomy or diagnostic suitability."
            ),
            classification=prediction,
            probabilities=probabilities,
        )
    except Exception:
        return JSONResponse(
            status_code=400,
            content=AnalyzeResponse(
                request_id=request_id,
                model_version="unavailable",
                status="unsupported_or_unsuitable",
                input_suitability="rejected",
                abstention_reason="processing_error",
                message=UNSUPPORTED_MESSAGE,
            ).model_dump(),
        )
