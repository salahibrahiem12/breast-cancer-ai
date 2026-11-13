"""Multi-stage medical image validation, modality classification, and OOD detection.

Stages:
1. File Validation (MIME, bytes, bounds, decoding)
2. Modality Classification (Breast Ultrasound, Mammogram, Chest X-ray, or Unsupported)
3. Image Quality Control (Blur, contrast, clipping, near-uniform)
4. Out-of-Distribution & Uncertainty Assessment
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import logging
import math
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageFilter, ImageStat, UnidentifiedImageError

from backend.app.config import (
    ELIGIBILITY_PROBE_PATH,
    MAX_IMAGE_PIXELS,
    MAX_UPLOAD_BYTES,
    MIN_IMAGE_SIDE,
    SUPPORTED_FORMATS,
)

logger = logging.getLogger(__name__)

# Cached ResNet-18 feature extractor and linear probe
_PROBE_WEIGHTS: dict | None = None
_RESNET18_MODEL = None
_TORCH_TRANSFORM = None


SupportedModality = Literal["breast_ultrasound", "mammography"]
DetectedModality = Literal[
    "breast_ultrasound",
    "mammography",
    "chest_xray",
    "natural_image",
    "document_or_screenshot",
    "unsupported_image",
]


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    detected_modality: DetectedModality
    supported_modality: SupportedModality | None
    stages_passed: tuple[str, ...]
    rejection_reason: str | None
    message: str
    quality_metrics: dict[str, float]
    width: int
    height: int
    image_format: str


class ImageValidationError(ValueError):
    """Raised for malformed, dangerous, or unreadable uploads."""


def stage1_file_validation(payload: bytes, filename: str | None = None) -> tuple[Image.Image, str, int, int]:
    """Stage 1: Validate payload size, format, decoding, and dimensions."""
    if not payload:
        raise ImageValidationError("empty_upload")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ImageValidationError("upload_size_limit_exceeded")

    suffix = (filename or "").rsplit(".", 1)[-1].lower() if filename and "." in filename else ""
    expected_format = "PNG" if suffix == "png" else ("JPEG" if suffix in {"jpg", "jpeg"} else None)

    try:
        with Image.open(BytesIO(payload)) as probe:
            fmt = (probe.format or "").upper()
            if fmt not in SUPPORTED_FORMATS:
                raise ImageValidationError("unsupported_image_format")
            if expected_format and fmt != expected_format:
                raise ImageValidationError("format_extension_mismatch")
            w, h = probe.size
            if w <= 0 or h <= 0 or (w * h) > MAX_IMAGE_PIXELS:
                raise ImageValidationError("invalid_image_dimensions")
            probe.verify()

        img = Image.open(BytesIO(payload))
        img.load()
        return img, fmt, w, h
    except ImageValidationError:
        raise
    except (EOFError, UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ImageValidationError("image_decode_failed") from exc


def compute_quality_metrics(img: Image.Image) -> dict[str, float]:
    """Stage 3: Compute objective image quality descriptors."""
    gray = img.convert("L")
    w, h = gray.size
    arr = np.asarray(gray, dtype=np.float32)

    # Pixel intensity statistics
    stat = ImageStat.Stat(gray)
    mean_val = float(stat.mean[0])
    std_val = float(stat.stddev[0])

    # Dynamic range / contrast
    p5, p95 = float(np.percentile(arr, 5)), float(np.percentile(arr, 95))
    contrast_range = p95 - p5

    # Blur detection via Laplacian edge variance
    laplacian_kernel = ImageFilter.Kernel(
        (3, 3),
        [0, 1, 0, 1, -4, 1, 0, 1, 0],
        scale=1,
        offset=0,
    )
    edges = gray.filter(laplacian_kernel)
    lap_var = float(ImageStat.Stat(edges).var[0])

    # Near-black border fraction (common in ultrasound / mammography background)
    black_border_fraction = float(np.mean(arr < 12.0))

    # Color saturation: check if RGB channels diverge significantly
    rgb = img.convert("RGB")
    rgb_arr = np.asarray(rgb, dtype=np.float32)
    rg_diff = np.abs(rgb_arr[:, :, 0] - rgb_arr[:, :, 1])
    gb_diff = np.abs(rgb_arr[:, :, 1] - rgb_arr[:, :, 2])
    pixel_chroma = rg_diff + gb_diff
    chroma_std = float(np.mean(pixel_chroma))
    colored_pixel_fraction = float(np.mean(pixel_chroma > 20.0))

    return {
        "mean_intensity": mean_val,
        "intensity_std": std_val,
        "contrast_range": contrast_range,
        "sharpness_laplacian_var": lap_var,
        "black_border_fraction": black_border_fraction,
        "chroma_divergence": chroma_std,
        "colored_pixel_fraction": colored_pixel_fraction,
        "aspect_ratio": round(w / max(1, h), 4),
    }


def _get_probe_components():
    global _PROBE_WEIGHTS, _RESNET18_MODEL, _TORCH_TRANSFORM
    if _PROBE_WEIGHTS is None and ELIGIBILITY_PROBE_PATH.exists():
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms

            _PROBE_WEIGHTS = torch.load(ELIGIBILITY_PROBE_PATH, map_location="cpu")
            m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            m.fc = torch.nn.Identity()
            m.eval()
            _RESNET18_MODEL = m
            _TORCH_TRANSFORM = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            logger.info("Loaded Breast Eligibility Feature Probe from %s", ELIGIBILITY_PROBE_PATH)
        except Exception as exc:
            logger.warning("Failed to initialize eligibility probe: %s", exc)
    return _PROBE_WEIGHTS, _RESNET18_MODEL, _TORCH_TRANSFORM


def evaluate_breast_ultrasound_probe(img: Image.Image) -> float | None:
    """Evaluate deep ResNet-18 feature probe probability for authentic breast ultrasound."""
    weights, model, tf = _get_probe_components()
    if weights is None or model is None or tf is None:
        return None
    try:
        import torch
        t = tf(img.convert("RGB")).unsqueeze(0)
        with torch.inference_mode():
            feat = model(t).numpy()[0]
        coef = np.asarray(weights["coef"], dtype=np.float32)
        intercept = float(weights["intercept"])
        logit = float(np.dot(coef, feat) + intercept)
        prob = 1.0 / (1.0 + math.exp(-logit))
        return prob
    except Exception as exc:
        logger.warning("Eligibility probe inference error: %s", exc)
        return None


def is_obstetric_sector_geometry(img: Image.Image, q: dict[str, float]) -> bool:
    """Identify curved transducer sector geometry characteristic of obstetric/fetal ultrasound."""
    w, h = img.size
    aspect = q.get("aspect_ratio", w / max(1, h))
    arr = np.asarray(img.convert("L"), dtype=np.float32)

    cw, ch = int(w * 0.16), int(h * 0.16)
    tl = float(np.mean(arr[:ch, :cw] < 12))
    tr = float(np.mean(arr[:ch, w - cw:] < 12))
    corners = (tl + tr) / 2.0

    row0_black = float(np.mean(arr[0, :] < 10))
    top10_black = float(np.mean(arr[:max(1, int(h * 0.1)), :] < 10))
    top_center = float(np.mean(arr[:max(1, int(h * 0.08)), int(w * 0.35) : int(w * 0.65)] < 12))

    # Curved transducer sector signature: top dead corners radiating from apex
    if (corners > 0.65 and (row0_black > 0.90 or top10_black > 0.60)) or (aspect > 1.42 and top_center > 0.85 and corners > 0.60):
        return True
    return False


def classify_modality(img: Image.Image, q: dict[str, float]) -> tuple[DetectedModality, str | None]:
    """Stage 2: Differentiate breast ultrasound, mammography, chest X-ray, and non-medical images."""
    w, h = img.size
    chroma = q["chroma_divergence"]
    colored_frac = q.get("colored_pixel_fraction", 0.0)
    std_val = q["intensity_std"]
    black_border = q["black_border_fraction"]
    contrast = q["contrast_range"]
    aspect = q["aspect_ratio"]

    # 1. Non-medical color images (selfies, food, animals, cars, drawings)
    # Allow sparse Doppler/caliper overlays on otherwise grayscale ultrasound scans.
    if chroma > 22.0 or (chroma > 14.0 and colored_frac > 0.35):
        return "natural_image", "High color saturation detected; uploaded image is a natural or non-medical photo."

    # 2. Text / Screenshot / Document check
    # Documents typically have near-white background, very high mean intensity (>210) with sharp black text
    if q["mean_intensity"] > 215.0 and black_border < 0.02 and q["sharpness_laplacian_var"] > 80.0:
        return "document_or_screenshot", "Document or screenshot detected."

    # 3. Detect Chest X-Ray vs Mammography
    # Chest X-ray characteristics:
    # - Bilateral symmetrical dark lung fields separated by central mediastinal spine/heart column
    # - Low black border fraction (chest radiograph fills the detector)
    # - Aspect ratio ~ 0.8 - 1.2
    gray_arr = np.asarray(img.convert("L"), dtype=np.float32)
    left_half = gray_arr[:, : w // 2]
    right_half = gray_arr[:, w // 2 :]
    symmetry_diff = float(np.mean(np.abs(left_half[:, ::-1] - right_half[:, : left_half.shape[1]])))

    # Chest X-rays have relatively low border black fraction (< 0.12), distinct bilateral lung lucency, and central spine
    if black_border < 0.12 and symmetry_diff < 32.0 and aspect >= 0.75 and aspect <= 1.35 and contrast > 80.0:
        # Check central column intensity (spine/sternum/heart) vs lung zones
        mid_third = gray_arr[:, w // 3 : 2 * w // 3]
        outer_third = np.concatenate([gray_arr[:, : w // 3], gray_arr[:, 2 * w // 3 :]], axis=1)
        mid_minus_outer = float(np.mean(mid_third) - np.mean(outer_third))
        if (symmetry_diff < 16.0 and mid_minus_outer > 8.0) or mid_minus_outer > 18.0:
            return "chest_xray", "Chest X-ray detected. BreastCare AI only evaluates breast ultrasound and mammograms."

    # 4. Mammography (Breast X-ray):
    # - Pure black background on one side (air), prominent soft-tissue breast curve
    # - Typically single lateralized breast projection (MLO or CC view), portrait format (aspect <= 1.10)
    # - High black border fraction (air around breast typically 20% to 75%)
    # - Asymmetrical intensity between left and right sides
    lateral_diff = abs(float(np.mean(left_half)) - float(np.mean(right_half)))
    if aspect <= 1.10 and black_border >= 0.15 and lateral_diff > 25.0 and contrast >= 60.0:
        return "mammography", None

    # 5. Acoustic Sector check: Obstetric / Fetal Curved Sector Ultrasound
    if is_obstetric_sector_geometry(img, q):
        return "unsupported_image", "Non-breast acoustic sector geometry detected (curvilinear transducer or non-breast study)."

    # 6. Breast Ultrasound:
    # - Full-frame acoustic field with dark scan borders, OR
    # - Cropped B-mode ultrasound field with characteristic speckle texture and mid/dark luminance
    is_us_candidate = False
    if black_border >= 0.05 and q["sharpness_laplacian_var"] >= 6.0 and contrast >= 25.0:
        is_us_candidate = True
    elif q["mean_intensity"] <= 145.0 and std_val >= 15.0 and contrast >= 45.0 and q["sharpness_laplacian_var"] >= 20.0:
        is_us_candidate = True

    if is_us_candidate:
        # Check ResNet-18 breast ultrasound eligibility probe
        probe_score = evaluate_breast_ultrasound_probe(img)
        if probe_score is not None:
            # Conservative calibrated cutoff
            threshold = 0.20
            if _PROBE_WEIGHTS and "threshold" in _PROBE_WEIGHTS:
                threshold = float(_PROBE_WEIGHTS["threshold"])
            if probe_score < threshold:
                return "unsupported_image", f"Ultrasound features do not match authentic breast imaging criteria (probe score {probe_score:.3f} < {threshold:.2f})."
        return "breast_ultrasound", None

    # Fallback to unsupported if confidence is low
    return "unsupported_image", "Image does not exhibit characteristic breast ultrasound or mammography morphology."


def validate_medical_image(payload: bytes, filename: str | None = None) -> ValidationResult:
    """Execute all 4 validation stages and return structured result."""
    stages: list[str] = []

    # Stage 1: File container and byte validation
    try:
        img, fmt, w, h = stage1_file_validation(payload, filename)
        stages.append("stage1_file_validation")
    except ImageValidationError as exc:
        return ValidationResult(
            is_valid=False,
            detected_modality="unsupported_image",
            supported_modality=None,
            stages_passed=(),
            rejection_reason=str(exc),
            message="Unsupported image. Please upload a valid breast ultrasound or mammogram.",
            quality_metrics={},
            width=0,
            height=0,
            image_format="UNKNOWN",
        )

    # Stage 2 & 3: Quality Metrics and Modality Classification
    q = compute_quality_metrics(img)

    # Stage 3 Checks:
    if min(w, h) < MIN_IMAGE_SIDE:
        return ValidationResult(
            is_valid=False,
            detected_modality="unsupported_image",
            supported_modality=None,
            stages_passed=tuple(stages),
            rejection_reason="image_dimensions_below_minimum",
            message="Image dimensions are too low for diagnostic evaluation (minimum 64x64).",
            quality_metrics=q,
            width=w,
            height=h,
            image_format=fmt,
        )

    if q["intensity_std"] < 2.0 or q["contrast_range"] < 8.0:
        return ValidationResult(
            is_valid=False,
            detected_modality="unsupported_image",
            supported_modality=None,
            stages_passed=tuple(stages),
            rejection_reason="near_uniform_or_empty_image",
            message="Empty or near-uniform image detected; unusable diagnostic content.",
            quality_metrics=q,
            width=w,
            height=h,
            image_format=fmt,
        )

    if q["sharpness_laplacian_var"] < 3.5:
        return ValidationResult(
            is_valid=False,
            detected_modality="unsupported_image",
            supported_modality=None,
            stages_passed=tuple(stages),
            rejection_reason="severe_blur_detected",
            message="Unable to provide a reliable assessment. Specialist review is recommended.",
            quality_metrics=q,
            width=w,
            height=h,
            image_format=fmt,
        )

    stages.append("stage3_quality_control")

    # Stage 2 Modality classification
    modality, reject_msg = classify_modality(img, q)
    if modality in {"natural_image", "chest_xray", "document_or_screenshot", "unsupported_image"}:
        return ValidationResult(
            is_valid=False,
            detected_modality=modality,
            supported_modality=None,
            stages_passed=tuple(stages),
            rejection_reason=f"modality_rejected_{modality}",
            message="Unsupported image. Please upload a valid breast ultrasound or mammogram.",
            quality_metrics=q,
            width=w,
            height=h,
            image_format=fmt,
        )

    stages.append("stage2_modality_classification")

    # Stage 4 Out-of-Distribution boundary check
    # Check if contrast and feature variance lie within acceptable medical ultrasound / mammography envelope
    if q["contrast_range"] < 15.0 or q["intensity_std"] < 6.0:
        return ValidationResult(
            is_valid=False,
            detected_modality=modality,
            supported_modality=modality,  # type: ignore
            stages_passed=tuple(stages),
            rejection_reason="borderline_distribution_uncertainty",
            message="Unable to provide a reliable assessment. Specialist review is recommended.",
            quality_metrics=q,
            width=w,
            height=h,
            image_format=fmt,
        )

    stages.append("stage4_ood_verification")

    return ValidationResult(
        is_valid=True,
        detected_modality=modality,
        supported_modality=modality,  # type: ignore
        stages_passed=tuple(stages),
        rejection_reason=None,
        message=f"Valid {modality.replace('_', ' ').title()} successfully accepted for research assessment.",
        quality_metrics=q,
        width=w,
        height=h,
        image_format=fmt,
    )

