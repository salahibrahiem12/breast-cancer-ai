"""Fail-closed ultrasound eligibility gate for the experimental classifier."""

from io import BytesIO
from pathlib import Path

from PIL import Image


UNSUPPORTED_IMAGE_MESSAGE = (
    "Unsupported image: Unable to verify that this is a breast ultrasound image."
)


class UnsupportedImageError(ValueError):
    """Raised when an image cannot be verified as an eligible ultrasound."""


def decode_rgb_image(image_source: bytes | str | Path) -> Image.Image:
    """Decode an image or raise the same safe error used for ineligible inputs."""
    try:
        image_bytes = (
            image_source
            if isinstance(image_source, bytes)
            else Path(image_source).read_bytes()
        )
        with Image.open(BytesIO(image_bytes)) as opened_image:
            opened_image.verify()
        with Image.open(BytesIO(image_bytes)) as opened_image:
            return opened_image.convert("RGB")
    except (
        EOFError,
        Image.DecompressionBombError,
        OSError,
        SyntaxError,
        ValueError,
    ) as exc:
        raise UnsupportedImageError(UNSUPPORTED_IMAGE_MESSAGE) from exc


def require_verified_ultrasound(image: Image.Image) -> None:
    """Block classification until a separately validated modality detector exists.

    The current project has no trained or independently evaluated detector that
    can distinguish breast ultrasound images from general or medical images.
    File metadata and visual heuristics cannot establish image modality.
    """
    del image
    raise UnsupportedImageError(UNSUPPORTED_IMAGE_MESSAGE)
