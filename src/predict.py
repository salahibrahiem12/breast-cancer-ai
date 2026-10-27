"""
Inference & Prediction module for single images or directory batches.
Includes probability distribution and optional Grad-CAM explainability generation.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import BEST_MODEL_PATH, CLASS_NAMES, DEVICE, NUM_CLASSES
from src.dataset import get_eval_transforms
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
    Performs inference on a single ultrasound image.
    Returns:
        predicted_class (str): E.g., 'malignant'
        confidence (float): Probability score of top predicted class
        class_probabilities (dict): Full probability breakdown
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at: {image_path}")

    image = Image.open(image_path).convert("RGB")
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
    parser = argparse.ArgumentParser(description="Predict Breast Cancer Diagnosis from Ultrasound Image")
    parser.add_argument("--image", type=str, required=True, help="Path to ultrasound image file")
    parser.add_argument("--model", type=str, default="breastnet", help="Model architecture")
    parser.add_argument("--checkpoint", type=str, default=str(BEST_MODEL_PATH), help="Path to weights checkpoint")
    parser.add_argument("--explain", action="store_true", help="Generate and save Grad-CAM visualization")
    args = parser.parse_args()

    model = load_inference_model(model_name=args.model, checkpoint_path=Path(args.checkpoint))
    pred_class, conf, probs = predict_single_image(Path(args.image), model)

    print("\n" + "=" * 50)
    print("      BREAST CANCER AI DIAGNOSIS REPORT      ")
    print("=" * 50)
    print(f"Target Image:      {args.image}")
    print(f"Predicted Class:   {pred_class.upper()}")
    print(f"Confidence Score:  {conf * 100:.2f}%")
    print("-" * 50)
    print("Class Probability Breakdown:")
    for cls_name, p in probs.items():
        bar = "#" * int(p * 30)
        print(f"  {cls_name.capitalize():<10}: {p*100:5.2f}%  {bar}")
    print("=" * 50)

    if args.explain:
        from src.explainability import explain_prediction
        save_path = Path("reports/figures") / f"explain_{Path(args.image).stem}.png"
        explain_prediction(model, Path(args.image), save_path=save_path)
        print(f"Grad-CAM explainability figure saved to: {save_path}")


if __name__ == "__main__":
    main()
