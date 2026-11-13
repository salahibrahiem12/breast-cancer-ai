"""Backend behavior tests; passing them does not measure OOD rejection quality."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFilter

from backend.app.config import PROJECT_ROOT
from backend.app.inference import UNSUPPORTED_MESSAGE, inspect_image, model_availability
from backend.app.main import app


def image_bytes(image_format: str = "PNG", kind: str = "book") -> bytes:
    image = Image.new("RGB", (192, 160), "white")
    draw = ImageDraw.Draw(image)
    if kind == "book":
        draw.rectangle((30, 12, 155, 145), fill="#a85d36", outline="black", width=4)
        draw.line((93, 15, 93, 140), fill="white", width=3)
    elif kind == "natural":
        draw.ellipse((18, 12, 112, 110), fill="#63a55b")
        draw.rectangle((0, 108, 192, 160), fill="#71a4d5")
    elif kind == "blurry":
        for offset in range(0, 192, 8):
            draw.line((offset, 0, offset, 160), fill="#333333" if offset % 16 else "#eeeeee", width=4)
        image = image.filter(ImageFilter.GaussianBlur(radius=7))
    else:
        draw.ellipse((48, 10, 142, 145), fill="#d0d0d0", outline="#222222", width=4)
        draw.line((95, 14, 95, 141), fill="#555555", width=3)
    output = io.BytesIO()
    image.save(output, format=image_format)
    return output.getvalue()


class BackendApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_and_model_info_report_inference_disabled(self):
        health = self.client.get("/health")
        model = self.client.get("/model-info")
        self.assertEqual(health.status_code, 200)
        self.assertFalse(health.json()["inference_enabled"])
        self.assertFalse(model.json()["inference_enabled"])
        self.assertEqual(model.json()["model_version"], "unavailable")

    def test_valid_image_is_abstained_without_modality_prediction(self):
        response = self.client.post(
            "/analyze",
            files={"file": ("sample.png", image_bytes(), "image/png")},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["input_suitability"], "unable_to_assess")
        self.assertEqual(body["status"], "research_classification_disabled")
        self.assertIn("disabled", body["message"])
        self.assertIsNone(body["classification"])
        self.assertIsNone(body["probabilities"])

    def test_unrelated_sample_fixtures_never_receive_cancer_predictions(self):
        examples = (
            ("book.jpg", image_bytes("JPEG", "book"), "image/jpeg"),
            ("natural.png", image_bytes("PNG", "natural"), "image/png"),
            ("other-medical.png", image_bytes("PNG", "medical"), "image/png"),
            ("other-ultrasound.png", image_bytes("PNG", "medical"), "image/png"),
            ("blurry.png", image_bytes("PNG", "blurry"), "image/png"),
        )
        for filename, payload, content_type in examples:
            response = self.client.post(
                "/analyze",
                files={"file": (filename, payload, content_type)},
            )
            with self.subTest(filename=filename):
                self.assertEqual(response.status_code, 200)
                self.assertIsNone(response.json()["classification"])
                self.assertIsNone(response.json()["probabilities"])
                self.assertIn(
                    response.json()["status"],
                    {"research_classification_disabled", "model_unavailable"},
                )

    def test_corrupt_content_is_rejected_even_with_image_extension(self):
        response = self.client.post(
            "/analyze",
            files={"file": ("scan.png", b"not a png", "image/png")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["input_suitability"], "rejected")
        self.assertEqual(response.json()["abstention_reason"], "image_decode_failed")
        self.assertIsNone(response.json()["classification"])

    def test_extension_and_signature_must_match_supported_format(self):
        response = self.client.post(
            "/analyze",
            files={"file": ("scan.png", image_bytes("JPEG"), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["input_suitability"], "rejected")
        self.assertEqual(response.json()["abstention_reason"], "unsupported_image_format")
        self.assertEqual(response.json()["message"], UNSUPPORTED_MESSAGE)

    def test_oversized_upload_is_rejected_before_decode(self):
        with patch("backend.app.main.MAX_UPLOAD_BYTES", 64):
            response = self.client.post(
                "/analyze",
                files={"file": ("large.png", b"x" * 65, "image/png")},
            )
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["abstention_reason"], "upload_size_limit_exceeded")
        self.assertIsNone(response.json()["classification"])

    def test_spooled_oversized_upload_is_closed_and_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch("tempfile.tempdir", temporary):
                with patch("backend.app.main.MAX_UPLOAD_BYTES", 64):
                    response = self.client.post(
                        "/analyze",
                        files={"file": ("large.png", b"x" * (2 * 1024 * 1024), "image/png")},
                    )
            self.assertEqual(response.status_code, 413)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_no_upload_filename_is_echoed_or_logged_in_response(self):
        response = self.client.post(
            "/analyze",
            files={"file": ("private-patient-name.png", image_bytes(), "image/png")},
        )
        self.assertNotIn("private-patient-name", response.text)

    def test_local_busi_class_images_are_never_given_cancer_predictions(self):
        test_root = PROJECT_ROOT / "data" / "processed_busi" / "test"
        image_paths = {
            label: list((test_root / label).glob("*.png"))
            if (test_root / label).is_dir() else []
            for label in ("benign", "malignant", "normal")
        }
        if not all(image_paths.values()):
            self.skipTest("Local BUSI test images are not present.")
        for label, paths in image_paths.items():
            path = paths[0]
            response = self.client.post(
                "/analyze",
                files={"file": (path.name, path.read_bytes(), "image/png")},
            )
            with self.subTest(label=label):
                self.assertEqual(response.status_code, 200)
                self.assertIsNone(response.json()["classification"])
                self.assertEqual(response.json()["input_suitability"], "unable_to_assess")

    def test_presence_of_weight_files_never_enables_unvalidated_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "model.pth"
            detector_path = Path(temporary) / "eligibility.pth"
            model_path.touch()
            detector_path.touch()
            classifier_report_path = Path(temporary) / "model.evaluation.json"
            with (
                patch("backend.app.config.MODEL_PATH", model_path),
                patch("backend.app.config.CLASSIFIER_EVALUATION_PATH", classifier_report_path),
                patch("backend.app.config.ELIGIBILITY_MODEL_PATH", detector_path),
                patch(
                    "backend.app.config.ELIGIBILITY_EVALUATION_PATH",
                    Path(temporary) / "missing-evaluation.json",
                ),
                patch("backend.app.config.CLASSIFICATION_ENABLED", True),
            ):
                availability = model_availability()
        self.assertFalse(availability["model_available"])
        self.assertFalse(availability["classifier_evaluation_approved"])
        self.assertFalse(availability["eligibility_gate_available"])
        self.assertTrue(availability["model_artifact_present"])
        self.assertTrue(availability["eligibility_artifact_present"])
        self.assertFalse(availability["inference_enabled"])

    def test_hash_mismatch_or_missing_review_approval_blocks_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_path = root / "model.pth"
            gate_path = root / "eligibility.pth"
            classifier_report_path = root / "model.evaluation.json"
            gate_report_path = root / "eligibility.evaluation.json"
            model_path.write_bytes(b"classifier")
            gate_path.write_bytes(b"eligibility")
            classifier_report_path.write_text(
                json.dumps(
                    {
                        "research_inference_approved": True,
                        "checkpoint_sha256": "wrong-classifier-hash",
                    }
                ),
                encoding="utf-8",
            )
            gate_report_path.write_text(
                json.dumps(
                    {
                        "research_inference_approved": True,
                        "checkpoint_sha256": "wrong-gate-hash",
                        "selected_threshold": 0.9,
                        "threshold_selection_split": "threshold",
                        "final_test_split": "test",
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch("backend.app.config.MODEL_PATH", model_path),
                patch("backend.app.config.ELIGIBILITY_MODEL_PATH", gate_path),
                patch("backend.app.config.CLASSIFIER_EVALUATION_PATH", classifier_report_path),
                patch("backend.app.config.ELIGIBILITY_EVALUATION_PATH", gate_report_path),
                patch("backend.app.config.CLASSIFICATION_ENABLED", True),
            ):
                availability = model_availability()
        self.assertFalse(availability["classifier_evaluation_approved"])
        self.assertFalse(availability["eligibility_gate_available"])
        self.assertFalse(availability["inference_enabled"])

    def test_disabled_server_flag_prevents_both_gate_and_classifier_calls(self):
        gate = Mock(side_effect=AssertionError("eligibility gate must not run"))
        classifier = Mock(side_effect=AssertionError("classifier must not run"))
        with (
            patch("backend.app.config.CLASSIFICATION_ENABLED", False),
            patch("backend.app.main.run_eligibility_gate", gate),
            patch("backend.app.main.run_cancer_classifier", classifier),
        ):
            response = self.client.post(
                "/analyze",
                files={"file": ("sample.png", image_bytes(), "image/png")},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "research_classification_disabled")
        self.assertIsNone(response.json()["classification"])
        self.assertIsNone(response.json()["probabilities"])
        gate.assert_not_called()
        classifier.assert_not_called()

    def test_missing_gate_weights_block_classification(self):
        with (
            patch("backend.app.config.CLASSIFICATION_ENABLED", True),
            patch(
                "backend.app.main.model_availability",
                return_value={
                    "inference_enabled": False,
                    "eligibility_gate_available": False,
                },
            ),
            patch(
                "backend.app.main.run_cancer_classifier",
                side_effect=AssertionError("classifier must not run without the gate"),
            ) as classifier,
        ):
            response = self.client.post(
                "/analyze",
                files={"file": ("sample.png", image_bytes(), "image/png")},
            )
        self.assertEqual(response.json()["status"], "model_unavailable")
        self.assertEqual(
            response.json()["abstention_reason"],
            "ultrasound_eligibility_detector_unavailable",
        )
        self.assertIsNone(response.json()["classification"])
        classifier.assert_not_called()

    def test_gate_rejection_never_calls_cancer_classifier(self):
        with (
            patch("backend.app.config.CLASSIFICATION_ENABLED", True),
            patch(
                "backend.app.main.model_availability",
                return_value={
                    "inference_enabled": True,
                    "eligibility_gate_available": True,
                },
            ),
            patch("backend.app.main.run_eligibility_gate", return_value=(False, 0.1)),
            patch(
                "backend.app.main.run_cancer_classifier",
                side_effect=AssertionError("classifier must not run after gate rejection"),
            ) as classifier,
        ):
            response = self.client.post(
                "/analyze",
                files={"file": ("sample.png", image_bytes(), "image/png")},
            )
        self.assertEqual(response.json()["status"], "unsupported_or_unsuitable")
        self.assertEqual(response.json()["input_suitability"], "rejected")
        self.assertIsNone(response.json()["classification"])
        self.assertIsNone(response.json()["probabilities"])
        classifier.assert_not_called()

    def test_server_enabled_gate_pass_is_the_only_classification_path(self):
        with (
            patch("backend.app.config.CLASSIFICATION_ENABLED", True),
            patch(
                "backend.app.main.model_availability",
                return_value={
                    "inference_enabled": True,
                    "eligibility_gate_available": True,
                },
            ),
            patch("backend.app.main.run_eligibility_gate", return_value=(True, 0.98)),
            patch(
                "backend.app.main.run_cancer_classifier",
                return_value=("Benign", {"Benign": 0.7, "Malignant": 0.3}, "test-model"),
            ) as classifier,
        ):
            response = self.client.post(
                "/analyze",
                files={"file": ("sample.png", image_bytes(), "image/png")},
            )
        self.assertEqual(response.json()["status"], "classification_available")
        self.assertEqual(response.json()["input_suitability"], "ultrasound_like")
        self.assertEqual(response.json()["classification"], "Benign")
        self.assertEqual(response.json()["probabilities"], {"Benign": 0.7, "Malignant": 0.3})
        self.assertIn("does not verify breast anatomy", response.json()["message"])
        classifier.assert_called_once()

    def test_real_breast_ultrasound_fixture_is_still_not_classified_by_default(self):
        source = PROJECT_ROOT / "data" / "raw" / "mendeley_v1" / "official-v1" / "originals" / "benign"
        samples = sorted(source.glob("*.bmp")) if source.is_dir() else []
        if not samples:
            self.skipTest("No licensed local breast-ultrasound image fixture is available.")
        path = samples[0]
        converted = io.BytesIO()
        with Image.open(path) as image:
            image.convert("RGB").save(converted, format="PNG")
        with patch("backend.app.config.CLASSIFICATION_ENABLED", False):
            response = self.client.post(
                "/analyze",
                files={"file": (f"{path.stem}.png", converted.getvalue(), "image/png")},
            )
        self.assertEqual(response.json()["status"], "research_classification_disabled")

    def test_image_quality_flags_are_not_modality_proof(self):
        low_detail = Image.new("RGB", (32, 32), "black")
        output = io.BytesIO()
        low_detail.save(output, format="PNG")
        result = inspect_image(output.getvalue(), "low.png")
        self.assertIn("image_dimensions_below_minimum", result.quality_flags)

    def test_low_quality_upload_is_rejected_with_the_safe_message(self):
        image = Image.new("RGB", (32, 32), "black")
        output = io.BytesIO()
        image.save(output, format="PNG")
        response = self.client.post(
            "/analyze",
            files={"file": ("low.png", output.getvalue(), "image/png")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["input_suitability"], "rejected")
        self.assertEqual(response.json()["message"], UNSUPPORTED_MESSAGE)
        self.assertEqual(response.json()["abstention_reason"], "image_dimensions_below_minimum")


if __name__ == "__main__":
    unittest.main()
