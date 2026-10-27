"""Experimental image inference, currently blocked by a fail-closed safety gate."""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import BEST_MODEL_PATH, CLASS_NAMES, DEVICE, NUM_CLASSES
from src.dataset import get_eval_transforms
from src.eligibility import (
    UnsupportedImageError,
    decode_rgb_image,
    require_verified_ultrasound,
)
from src.models import build_model
from src.utils import load_checkpoint


def load_inference_model(
    model_name: str = "breastnet",
    checkpoint_path: Path = BEST_MODEL_PATH,
    device: torch.device = DEVICE,
) -> torch.nn.Module:
    """Instantiates and loads trained weights for evaluation/inference."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

    model = build_model(
        model_name=model_name,
        num_classes=NUM_CLASSES,
        pretrained=False,
    )
    load_checkpoint(checkpoint_path, model, device=device)
    model.to(device)
    model.eval()
    return model


def predict_single_image(
    image_path: Path,
    model: torch.nn.Module,
    device: torch.device = DEVICE,
) -> Tuple[str, float, Dict[str, float]]:
    """
    Performs inference only after ultrasound eligibility is verified.

    The current implementation intentionally rejects all images because no
    validated eligibility detector is available.
    Returns:
        predicted_class (str): E.g., 'malignant'
        confidence (float): Raw, uncalibrated softmax score
        class_probabilities (dict): Raw softmax scores, not calibrated estimates
    """
    image_path = Path(image_path)
    image = decode_rgb_image(image_path)
    require_verified_ultrasound(image)
    transform = get_eval_transforms()
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1).squeeze().cpu().numpy()

    top_idx = int(probabilities.argmax())
    predicted_class = CLASS_NAMES[top_idx]
    confidence = float(probabilities[top_idx])

    class_probabilities = {
        cls_name: float(probabilities[i]) for i, cls_name in enumerate(CLASS_NAMES)
    }

    return predicted_class, confidence, class_probabilities


def main():
    parser = argparse.ArgumentParser(
        description="Experimental image classifier (prediction currently disabled)"
    )
    parser.add_argument("--image", type=str, required=True, help="Path to ultrasound image file")
    args = parser.parse_args()

    try:
        image = decode_rgb_image(args.image)
        require_verified_ultrasound(image)
    except UnsupportedImageError as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
