"""Fail-closed inference state and upload validation.

The file checks below protect the service from malformed or excessive uploads;
they are not a breast-ultrasound eligibility detector.
"""

from dataclasses import dataclass
from io import BytesIO
import json
import hashlib
import logging

from PIL import Image, ImageStat, UnidentifiedImageError

from backend.app import config
from backend.app.config import MAX_IMAGE_PIXELS, MIN_IMAGE_SIDE, SUPPORTED_FORMATS


UNSUPPORTED_MESSAGE = (
    "Unsupported image: Unable to verify that this is a breast ultrasound image."
)
CLASS_LABELS = ["Benign", "Malignant", "Normal"]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageValidation:
    image_format: str
    width: int
    height: int
    quality_flags: tuple[str, ...]


class InvalidImageError(ValueError):
    """Raised for unsupported, corrupt, or unreasonable image payloads."""


def inspect_image(payload: bytes, filename: str | None) -> ImageValidation:
    """Validate the actual image container, decoding, dimensions, and basic quality."""
    suffix = (filename or "").rsplit(".", maxsplit=1)
    if len(suffix) != 2 or suffix[-1].lower() not in {"png", "jpg", "jpeg"}:
        raise InvalidImageError("unsupported_file_extension")
    expected_format = "PNG" if suffix[-1].lower() == "png" else "JPEG"
    if not payload:
        raise InvalidImageError("empty_upload")

    try:
        with Image.open(BytesIO(payload)) as probe:
            image_format = (probe.format or "").upper()
            if image_format not in SUPPORTED_FORMATS or image_format != expected_format:
                raise InvalidImageError("unsupported_image_format")
            width, height = probe.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise InvalidImageError("invalid_image_dimensions")
            probe.verify()
        with Image.open(BytesIO(payload)) as decoded:
            decoded.load()
            grayscale = decoded.convert("L")
            pixel_spread = ImageStat.Stat(grayscale).stddev[0]
    except InvalidImageError:
        raise
    except (
        EOFError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        OSError,
        SyntaxError,
        UnidentifiedImageError,
        ValueError,
    ) as exc:
        raise InvalidImageError("image_decode_failed") from exc

    quality_flags: list[str] = []
    if min(width, height) < MIN_IMAGE_SIDE:
        quality_flags.append("image_dimensions_below_minimum")
    if pixel_spread < 2.0:
        quality_flags.append("near_uniform_image")
    return ImageValidation(
        image_format=image_format,
        width=width,
        height=height,
        quality_flags=tuple(quality_flags),
    )


def model_availability() -> dict[str, bool]:
    """Require server opt-in, both weights, and a reviewed gate-evaluation record."""
    classifier_present = config.MODEL_PATH.is_file()
    classifier_report_present = config.CLASSIFIER_EVALUATION_PATH.is_file()
    gate_present = config.ELIGIBILITY_MODEL_PATH.is_file()
    report_present = config.ELIGIBILITY_EVALUATION_PATH.is_file()
    classifier_approved = False
    report_approved = False
    if classifier_report_present:
        try:
            classifier_report = json.loads(
                config.CLASSIFIER_EVALUATION_PATH.read_text(encoding="utf-8")
            )
            classifier_digest = hashlib.sha256(config.MODEL_PATH.read_bytes()).hexdigest()
            classifier_approved = (
                isinstance(classifier_report, dict)
                and classifier_report.get("research_inference_approved") is True
                and classifier_report.get("checkpoint_sha256") == classifier_digest
            )
        except (OSError, ValueError, TypeError, AttributeError):
            logger.warning("Classifier evaluation record is unavailable or invalid.")
    if report_present:
        try:
            report = json.loads(config.ELIGIBILITY_EVALUATION_PATH.read_text(encoding="utf-8"))
            gate_digest = hashlib.sha256(config.ELIGIBILITY_MODEL_PATH.read_bytes()).hexdigest()
            report_approved = (
                isinstance(report, dict)
                and report.get("research_inference_approved") is True
                and report.get("checkpoint_sha256") == gate_digest
                and isinstance(report.get("selected_threshold"), (int, float))
                and 0.0 <= float(report["selected_threshold"]) <= 1.0
                and report.get("threshold_selection_split") == "threshold"
                and report.get("final_test_split") == "test"
            )
        except (OSError, ValueError, TypeError, AttributeError):
            logger.warning("Eligibility evaluation record is unavailable or invalid.")

    return {
        "model_available": classifier_present and classifier_approved,
        "classifier_evaluation_approved": classifier_approved,
        "eligibility_gate_available": gate_present and report_approved,
        "model_artifact_present": classifier_present,
        "eligibility_artifact_present": gate_present,
        "classification_enabled_by_server": config.CLASSIFICATION_ENABLED,
        "inference_enabled": (
            config.CLASSIFICATION_ENABLED
            and classifier_present
            and classifier_approved
            and gate_present
            and report_approved
        ),
    }


def run_eligibility_gate(payload: bytes) -> tuple[bool, float]:
    """Return whether an image passes the reviewed ultrasound-like gate threshold."""
    import torch
    from torchvision import transforms

    from ml.model import IMAGE_MEAN, IMAGE_SIZE, IMAGE_STD, build_model

    availability = model_availability()
    if not availability["eligibility_gate_available"]:
        raise RuntimeError("ultrasound_eligibility_detector_unavailable")
    report = json.loads(config.ELIGIBILITY_EVALUATION_PATH.read_text(encoding="utf-8"))
    threshold = float(report["selected_threshold"])
    checkpoint = torch.load(
        config.ELIGIBILITY_MODEL_PATH, map_location="cpu", weights_only=True
    )
    expected_order = ["not_breast_ultrasound", "breast_ultrasound"]
    if checkpoint.get("class_order") != expected_order:
        raise ValueError("eligibility_checkpoint_class_mapping_invalid")
    if checkpoint.get("image_size") != IMAGE_SIZE:
        raise ValueError("eligibility_checkpoint_preprocessing_invalid")
    model = build_model(num_classes=2, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
    )
    with Image.open(BytesIO(payload)) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        score = float(torch.softmax(model(tensor), dim=1)[0, 1].item())
    return score >= threshold, score


def run_cancer_classifier(payload: bytes) -> tuple[str, dict[str, float], str]:
    """Run the research classifier; caller must already have passed server gates."""
    import torch
    from torchvision import transforms

    from ml.model import IMAGE_MEAN, IMAGE_SIZE, IMAGE_STD, load_checkpoint

    model, checkpoint = load_checkpoint(str(config.MODEL_PATH), torch.device("cpu"))
    class_names = tuple(checkpoint["class_order"])
    if not set(class_names).issubset({"benign", "malignant", "normal"}):
        raise ValueError("classifier_checkpoint_class_mapping_invalid")
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
    )
    with Image.open(BytesIO(payload)) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        probabilities = torch.softmax(
            model(tensor) / float(checkpoint["temperature"]), dim=1
        )[0]
    index = int(probabilities.argmax().item())
    return (
        class_names[index].title(),
        {
            class_name.title(): float(probabilities[class_index].item())
            for class_index, class_name in enumerate(class_names)
        },
        str(checkpoint["model_version"]),
    )
