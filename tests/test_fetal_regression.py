"""Automated regression suite verifying fetal/obstetric ultrasound rejection and no Grad-CAM generation."""

import glob
from pathlib import Path
from fastapi.testclient import TestClient
import pytest

from backend.app.main import app
from backend.app.modality import validate_medical_image

client = TestClient(app)

FETAL_IMAGES = sorted(glob.glob("data/prepared/fetal_candidate_v1/images/*.png"))[:10]
TCIA_IMAGES = sorted(glob.glob("data/raw/tcia_breast_biopsy_external_v1/*.png"))[:5]


@pytest.mark.parametrize("fetal_path", FETAL_IMAGES)
def test_fetal_image_rejected_at_validation(fetal_path: str):
    """Confirm fetal/obstetric ultrasound is rejected as unsupported."""
    with open(fetal_path, "rb") as f:
        payload = f.read()
    res = validate_medical_image(payload, Path(fetal_path).name)
    assert not res.is_valid, f"Expected fetal scan {fetal_path} to be rejected, but was validated as {res.detected_modality}"
    assert res.detected_modality != "breast_ultrasound"


@pytest.mark.parametrize("fetal_path", FETAL_IMAGES)
def test_fetal_image_rejected_at_api_analyze(fetal_path: str):
    """Confirm /api/analyze rejects fetal scans and generates NO Grad-CAM."""
    with open(fetal_path, "rb") as f:
        resp = client.post("/api/analyze", files={"file": (Path(fetal_path).name, f, "image/png")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "unsupported_or_unsuitable"
    assert data["input_suitability"] == "rejected"
    assert data.get("gradcam_overlay_url") is None
    assert data.get("classification") is None


@pytest.mark.parametrize("tcia_path", TCIA_IMAGES)
def test_authentic_breast_ultrasound_accepted(tcia_path: str):
    """Confirm authentic breast ultrasound passes validation."""
    with open(tcia_path, "rb") as f:
        payload = f.read()
    res = validate_medical_image(payload, Path(tcia_path).name)
    assert res.is_valid, f"Expected authentic breast scan {tcia_path} to be valid, but got: {res.rejection_reason}"
    assert res.detected_modality == "breast_ultrasound"

