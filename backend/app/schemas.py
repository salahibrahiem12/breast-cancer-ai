"""Typed API response schemas and contracts for BreastCare AI."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str = "breastcare-research-api"
    version: str = "0.2.0"
    inference_enabled: bool = False
    supported_modalities: list[str] = Field(default_factory=lambda: ["breast_ultrasound", "mammography"])


class ReadyResponse(BaseModel):
    status: Literal["ready", "initializing"] = "ready"
    checks: dict[str, bool] = Field(default_factory=dict)


class ModelDetail(BaseModel):
    modality: str
    architecture: str
    checkpoint_status: str
    classification_enabled: bool
    release_criteria_met: bool
    evaluation_summary: dict[str, Any]
    limitations: list[str]


class ModelsResponse(BaseModel):
    service: str = "BreastCare AI Multimodal Engine"
    supported_modalities: list[str]
    models: dict[str, ModelDetail]


class ModelInfoResponse(BaseModel):
    """Backwards-compatible model info schema."""
    model_version: str
    model_available: bool
    eligibility_gate_available: bool
    classifier_evaluation_approved: bool
    model_artifact_present: bool
    eligibility_artifact_present: bool
    classification_enabled_by_server: bool
    inference_enabled: bool
    supported_modality: str
    class_labels: list[str]
    limitations: list[str]


class ValidateImageResponse(BaseModel):
    is_valid: bool
    detected_modality: str
    supported_modality: str | None = None
    stages_passed: list[str]
    rejection_reason: str | None = None
    message: str
    quality_metrics: dict[str, float]
    dimensions: dict[str, int]


class AnalyzeResponse(BaseModel):
    request_id: str
    model_version: str
    detected_modality: str = "unknown"
    status: Literal[
        "unsupported_or_unsuitable",
        "unable_to_assess",
        "model_unavailable",
        "research_classification_disabled",
        "classification_available",
    ]
    input_suitability: Literal["unable_to_assess", "rejected", "ultrasound_like", "mammogram_like"]
    abstention_reason: str
    message: str
    classification: str | None = None
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    gradcam_overlay_url: str | None = None
    explanation: str | None = None
    segmentation: str | None = None
    scientific_limitations: list[str] = Field(default_factory=list)
    research_disclaimer: str = (
        "BreastCare AI is an experimental research prototype and NOT a cleared medical device. "
        "Outputs do NOT constitute a medical diagnosis, BI-RADS categorization, or treatment recommendation. "
        "All findings require independent clinical evaluation by qualified breast specialists."
    )
