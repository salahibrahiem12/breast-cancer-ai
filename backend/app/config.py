"""Environment-backed API configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

MAX_UPLOAD_BYTES = int(os.getenv("BREASTCARE_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv("BREASTCARE_MAX_IMAGE_PIXELS", "25000000"))
MIN_IMAGE_SIDE = int(os.getenv("BREASTCARE_MIN_IMAGE_SIDE", "64"))
MODEL_PATH = Path(
    os.getenv(
        "BREASTCARE_MODEL_PATH",
        str(PROJECT_ROOT / "models_saved" / "versioned" / "breastcare_resnet18_v1.pth"),
    )
)
ELIGIBILITY_PROBE_PATH = Path(
    os.getenv(
        "BREASTCARE_ELIGIBILITY_PROBE_PATH",
        str(PROJECT_ROOT / "models_saved" / "eligibility" / "breast_eligibility_probe.pth"),
    )
)
ELIGIBILITY_PROBE_META_PATH = Path(
    os.getenv(
        "BREASTCARE_ELIGIBILITY_PROBE_META_PATH",
        str(PROJECT_ROOT / "models_saved" / "eligibility" / "breast_eligibility_probe.json"),
    )
)
ELIGIBILITY_MODEL_PATH = Path(
    os.getenv(
        "BREASTCARE_ELIGIBILITY_MODEL_PATH",
        str(PROJECT_ROOT / "models_saved" / "versioned" / "eligibility_v1.pth"),
    )
)
ELIGIBILITY_EVALUATION_PATH = Path(
    os.getenv(
        "BREASTCARE_ELIGIBILITY_EVALUATION_PATH",
        str(PROJECT_ROOT / "models_saved" / "versioned" / "eligibility_v1.evaluation.json"),
    )
)
CLASSIFIER_EVALUATION_PATH = Path(
    os.getenv(
        "BREASTCARE_CLASSIFIER_EVALUATION_PATH",
        str(PROJECT_ROOT / "models_saved" / "versioned" / "breastcare_resnet18_v1.evaluation.json"),
    )
)
CLASSIFICATION_ENABLED = os.getenv("BREASTCARE_CLASSIFICATION_ENABLED", "false").strip().lower()
if CLASSIFICATION_ENABLED not in {"true", "false"}:
    raise ValueError("BREASTCARE_CLASSIFICATION_ENABLED must be set to 'true' or 'false'.")
CLASSIFICATION_ENABLED = CLASSIFICATION_ENABLED == "true"
CORS_ORIGINS = tuple(
    origin.strip()
    for origin in os.getenv(
        "BREASTCARE_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if origin.strip()
)
SUPPORTED_FORMATS = {"PNG", "JPEG"}

if MAX_UPLOAD_BYTES <= 0 or MAX_IMAGE_PIXELS <= 0 or MIN_IMAGE_SIDE <= 0:
    raise ValueError("Upload byte, image pixel, and minimum-side limits must be positive.")
if not CORS_ORIGINS:
    raise ValueError("BREASTCARE_CORS_ORIGINS must contain at least one explicit origin.")
