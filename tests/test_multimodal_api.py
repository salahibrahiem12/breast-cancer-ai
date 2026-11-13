"""Comprehensive automated test suite for BreastCare AI Multimodal API."""

import concurrent.futures
from io import BytesIO
import numpy as np
from PIL import Image
import pytest
from starlette.testclient import TestClient

from backend.app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def make_ultrasound_bytes(width=224, height=224) -> bytes:
    """Return an authentic breast ultrasound image from local dataset if available, else synthetic."""
    sample_path = "data/raw/tcia_breast_biopsy_external_v1/case002.png"
    import os
    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            return f.read()
    rng = np.random.default_rng(42)
    arr = np.zeros((height, width), dtype=np.uint8)
    margin_y = height // 8
    margin_x = width // 8
    speckle = rng.normal(85, 30, (height - 2 * margin_y, width - 2 * margin_x))
    speckle = np.clip(speckle, 10, 240).astype(np.uint8)
    arr[margin_y : height - margin_y, margin_x : width - margin_x] = speckle
    img = Image.fromarray(arr, mode="L").convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_mammogram_bytes(width=224, height=280) -> bytes:
    """Generate a synthetic mammogram with dark air background and lateralized breast tissue curve."""
    arr = np.zeros((height, width), dtype=np.uint8)
    y, x = np.ogrid[:height, :width]
    # Lateralized breast arc originating from chest wall on the left
    breast_mask = ((x / (width * 0.85)) ** 2 + ((y - height * 0.45) / (height * 0.55)) ** 2) <= 1.0
    arr[breast_mask] = np.clip(np.random.normal(120, 35, np.sum(breast_mask)), 20, 245).astype(np.uint8)
    img = Image.fromarray(arr, mode="L").convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_color_selfie_bytes(width=200, height=200) -> bytes:
    """Generate a natural color photo with high RGB divergence (selfie/pet/car/food)."""
    rng = np.random.default_rng(101)
    r = rng.integers(180, 255, (height, width), dtype=np.uint8)
    g = rng.integers(90, 160, (height, width), dtype=np.uint8)
    b = rng.integers(40, 100, (height, width), dtype=np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    img = Image.fromarray(rgb, mode="RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_screenshot_bytes(width=300, height=200) -> bytes:
    """Generate a synthetic text document/screenshot with white background and black text bars."""
    arr = np.full((height, width), 245, dtype=np.uint8)
    for row in range(20, height - 20, 18):
        arr[row : row + 4, 30 : width - 30] = 15  # black text line
    img = Image.fromarray(arr, mode="L").convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_chest_xray_bytes(width=240, height=240) -> bytes:
    """Generate a synthetic chest X-ray with bilateral symmetrical lung fields and central spine."""
    arr = np.full((height, width), 160, dtype=np.uint8)
    y, x = np.ogrid[:height, :width]
    # Bilateral dark lung fields
    left_lung = ((x - width * 0.3) / 28) ** 2 + ((y - height * 0.5) / 55) ** 2 <= 1.0
    right_lung = ((x - width * 0.7) / 28) ** 2 + ((y - height * 0.5) / 55) ** 2 <= 1.0
    arr[left_lung] = 35
    arr[right_lung] = 35
    # Central mediastinum / spine (brighter)
    spine = (np.abs(x - width * 0.5) <= 12) & (y >= height * 0.2)
    arr[spine] = 210
    img = Image.fromarray(arr, mode="L").convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_blurry_image_bytes(width=100, height=100) -> bytes:
    """Generate an extremely blurry / near-uniform low-contrast image."""
    arr = np.full((height, width), 80, dtype=np.uint8)
    arr[40:60, 40:60] = 81
    img = Image.fromarray(arr, mode="L").convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "breast_ultrasound" in data["supported_modalities"]
    assert "mammography" in data["supported_modalities"]


def test_ready_endpoint(client):
    res = client.get("/ready")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ready"
    assert data["checks"]["server_running"] is True


def test_api_models_endpoint(client):
    res = client.get("/api/models")
    assert res.status_code == 200
    data = res.json()
    assert "breast_ultrasound" in data["models"]
    assert "mammography" in data["models"]
    assert data["models"]["breast_ultrasound"]["release_criteria_met"] is False


def test_api_references_endpoint(client):
    res = client.get("/api/references")
    assert res.status_code == 200
    refs = res.json()
    assert len(refs) >= 5
    orgs = {r["organization"] for r in refs}
    assert any("National Cancer Institute" in o for o in orgs)
    assert any("American College of Radiology" in o for o in orgs)


def test_scenario_1_valid_breast_ultrasound(client):
    payload = make_ultrasound_bytes()
    res = client.post(
        "/api/validate-image",
        files={"file": ("ultrasound_sample.png", payload, "image/png")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["is_valid"] is True
    assert data["detected_modality"] == "breast_ultrasound"


def test_scenario_2_valid_mammogram(client):
    payload = make_mammogram_bytes()
    res = client.post(
        "/api/validate-image",
        files={"file": ("mammogram_sample.jpg", payload, "image/jpeg")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["is_valid"] is True
    assert data["detected_modality"] == "mammography"


def test_scenario_3_random_human_selfie_rejected(client):
    payload = make_color_selfie_bytes()
    res = client.post(
        "/api/validate-image",
        files={"file": ("selfie.jpg", payload, "image/jpeg")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["is_valid"] is False
    assert data["detected_modality"] == "natural_image"
    assert "Unsupported image" in data["message"]


def test_scenario_4_pet_animal_rejected(client):
    payload = make_color_selfie_bytes()
    res = client.post(
        "/api/analyze",
        files={"file": ("cat.jpg", payload, "image/jpeg")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "unsupported_or_unsuitable"
    assert data["input_suitability"] == "rejected"
    assert "Unsupported image" in data["message"]


def test_scenario_5_document_screenshot_rejected(client):
    payload = make_screenshot_bytes()
    res = client.post(
        "/api/validate-image",
        files={"file": ("notes.png", payload, "image/png")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["is_valid"] is False
    assert data["detected_modality"] in {"document_or_screenshot", "unsupported_image"}


def test_scenario_6_chest_xray_rejected(client):
    payload = make_chest_xray_bytes()
    res = client.post(
        "/api/validate-image",
        files={"file": ("chest_xray.png", payload, "image/png")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["is_valid"] is False
    assert data["detected_modality"] in {"chest_xray", "unsupported_image"}


def test_scenario_7_corrupt_image_rejected(client):
    corrupt_payload = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRcorruptedcontent"
    res = client.post(
        "/api/validate-image",
        files={"file": ("broken.png", corrupt_payload, "image/png")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["is_valid"] is False
    assert data["rejection_reason"] == "image_decode_failed"


def test_scenario_8_oversized_file_rejected(client):
    huge_payload = b"0" * (10 * 1024 * 1024 + 100)
    res = client.post(
        "/api/analyze",
        files={"file": ("huge.png", huge_payload, "image/png")},
    )
    assert res.status_code == 413
    data = res.json()
    assert data["status"] == "unsupported_or_unsuitable"


def test_scenario_9_unclear_blurry_medical_image_abstention(client):
    blurry_payload = make_blurry_image_bytes()
    res = client.post(
        "/api/validate-image",
        files={"file": ("blurry.png", blurry_payload, "image/png")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["is_valid"] is False


def test_scenario_10_analyze_accepted_ultrasound_fail_closed(client):
    payload = make_ultrasound_bytes()
    res = client.post(
        "/api/analyze",
        files={"file": ("valid_us.png", payload, "image/png")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["detected_modality"] == "breast_ultrasound"
    assert data["status"] == "unable_to_assess"
    assert data["input_suitability"] == "ultrasound_like"
    assert data["abstention_reason"] == "classification_disabled_by_server"
    assert data["gradcam_overlay_url"] is not None
    assert "research prototype" in data["research_disclaimer"].lower()


def test_scenario_11_analyze_accepted_mammogram_fail_closed(client):
    payload = make_mammogram_bytes()
    res = client.post(
        "/api/analyze",
        files={"file": ("valid_mammo.jpg", payload, "image/jpeg")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["detected_modality"] == "mammography"
    assert data["status"] == "unable_to_assess"
    assert data["input_suitability"] == "mammogram_like"


def test_scenario_12_concurrent_requests(client):
    def send_req():
        return client.get("/health").status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(send_req) for _ in range(10)]
        results = [f.result() for f in futures]
    assert all(code == 200 for code in results)
