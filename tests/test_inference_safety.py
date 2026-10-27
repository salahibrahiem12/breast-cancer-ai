"""Regression tests for the fail-closed image eligibility gate."""

import io
from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageDraw
from streamlit.testing.v1 import AppTest
import torch

from src.eligibility import (
    UNSUPPORTED_IMAGE_MESSAGE,
    UnsupportedImageError,
    decode_rgb_image,
)
from src.explainability import GradCAM, explain_prediction
from src.predict import predict_single_image


class ModelThatMustNotRun:
    def __call__(self, *_args, **_kwargs):
        raise AssertionError("The classifier must not run for unverified images.")


def image_bytes(kind: str) -> bytes:
    image = Image.new("RGB", (192, 128), "white")
    draw = ImageDraw.Draw(image)
    if kind == "book-like image":
        draw.rectangle((30, 12, 155, 118), fill="#a85d36", outline="black", width=4)
        draw.line((93, 15, 93, 115), fill="white", width=3)
        for y in range(35, 100, 12):
            draw.line((45, y, 80, y), fill="black", width=1)
            draw.line((105, y, 140, y), fill="black", width=1)
    elif kind == "natural-image-like drawing":
        draw.ellipse((20, 10, 110, 100), fill="#63a55b")
        draw.rectangle((0, 95, 192, 128), fill="#71a4d5")
    elif kind == "unrelated-medical-like drawing":
        draw.ellipse((48, 10, 142, 115), fill="#d0d0d0", outline="#222222", width=4)
        draw.line((95, 15, 95, 110), fill="#555555", width=3)
    elif kind == "blurry image":
        image = Image.new("RGB", (2, 2), "#777777")
    elif kind == "low quality image":
        image = Image.new("RGB", (1, 1), "#000000")

    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class InferenceSafetyTests(unittest.TestCase):
    def assert_prediction_refused(self, raw_bytes: bytes) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.png"
            path.write_bytes(raw_bytes)
            with self.assertRaisesRegex(
                UnsupportedImageError, f"^{UNSUPPORTED_IMAGE_MESSAGE}$"
            ):
                predict_single_image(path, ModelThatMustNotRun())

    def test_non_ultrasound_and_low_quality_images_are_refused(self):
        cases = (
            "book-like image",
            "natural-image-like drawing",
            "unrelated-medical-like drawing",
            "blurry image",
            "low quality image",
        )
        for case in cases:
            with self.subTest(case=case):
                self.assert_prediction_refused(image_bytes(case))

    def test_corrupted_image_is_refused(self):
        self.assert_prediction_refused(b"not an image")
        with self.assertRaisesRegex(
            UnsupportedImageError, f"^{UNSUPPORTED_IMAGE_MESSAGE}$"
        ):
            decode_rgb_image(b"not an image")

    def test_file_extension_does_not_establish_modality(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ultrasound.jpg"
            path.write_bytes(image_bytes("book photograph"))
            with self.assertRaisesRegex(
                UnsupportedImageError, f"^{UNSUPPORTED_IMAGE_MESSAGE}$"
            ):
                predict_single_image(path, ModelThatMustNotRun())

    def test_local_busi_classes_are_also_refused_until_detector_exists(self):
        dataset_root = Path(__file__).resolve().parents[1] / "data" / "processed_busi" / "test"
        class_samples = {
            label: list((dataset_root / label).glob("*.png"))
            if (dataset_root / label).is_dir()
            else []
            for label in ("benign", "malignant", "normal")
        }
        if not all(class_samples.values()):
            self.skipTest("Local BUSI test images are not included in this checkout.")

        for label in ("benign", "malignant", "normal"):
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    UnsupportedImageError, f"^{UNSUPPORTED_IMAGE_MESSAGE}$"
                ):
                    predict_single_image(class_samples[label][0], ModelThatMustNotRun())

    def test_explanation_entrypoint_is_also_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.png"
            path.write_bytes(image_bytes("book-like image"))
            with self.assertRaisesRegex(
                UnsupportedImageError, f"^{UNSUPPORTED_IMAGE_MESSAGE}$"
            ):
                explain_prediction(ModelThatMustNotRun(), path)

    def test_streamlit_displays_the_exact_unsupported_message(self):
        image_buffer = io.BytesIO(image_bytes("book-like image"))
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path)).run(timeout=45)
        self.assertEqual(len(app.exception), 0)
        app.file_uploader[0].set_value(
            ("book.png", image_buffer.getvalue(), "image/png")
        )
        app.run(timeout=45)
        app.button[0].click().run(timeout=45)
        self.assertIn(UNSUPPORTED_IMAGE_MESSAGE, [error.value for error in app.error])
        self.assertEqual(len(app.exception), 0)

    def test_corrupted_image_is_refused_by_explanation_entrypoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corrupt.png"
            path.write_bytes(b"not an image")
            with self.assertRaisesRegex(
                UnsupportedImageError, f"^{UNSUPPORTED_IMAGE_MESSAGE}$"
            ):
                explain_prediction(ModelThatMustNotRun(), path)

    def test_direct_gradcam_entrypoint_is_also_blocked(self):
        gradcam = GradCAM(ModelThatMustNotRun(), torch.nn.Identity())
        with self.assertRaisesRegex(
            UnsupportedImageError, f"^{UNSUPPORTED_IMAGE_MESSAGE}$"
        ):
            gradcam.generate_cam(torch.zeros((1, 3, 8, 8)))


if __name__ == "__main__":
    unittest.main()
