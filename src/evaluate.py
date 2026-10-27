"""
Experimental model performance reporting module.
Computes research metrics on the existing image-level split.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    BEST_MODEL_PATH,
    CLASS_NAMES,
    DEVICE,
    FIGURES_DIR,
    NUM_CLASSES,
)
from src.dataset import get_dataloaders
from src.models import build_model
from src.utils import (
    calculate_medical_metrics,
    load_checkpoint,
    plot_confusion_matrix_figure,
    plot_multiclass_roc,
)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device = DEVICE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs forward inference across the available image-level test split.
    Returns:
        y_true: True ground truth class indices (N,)
        y_pred: Predicted class indices (N,)
        y_prob: Uncalibrated softmax score distributions (N, C)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def run_evaluation(
    model_name: str = "breastnet",
    checkpoint_path: Path = BEST_MODEL_PATH,
    device: torch.device = DEVICE,
) -> Dict[str, float]:
    """
    Loads checkpoint and reports research metrics; the split is not patient independent.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}. Train the model first.")

    print("\nEvaluating on the Existing Image-Level Test Split (Not Patient-Independent)")
    print(f"Loading weights from: {checkpoint_path}")
    print(f"Evaluation Device: {device}\n" + "=" * 65)

    _, _, test_loader, _ = get_dataloaders()

    model = build_model(model_name=model_name, num_classes=NUM_CLASSES)
    load_checkpoint(checkpoint_path, model, device=device)
    model.to(device)

    y_true, y_pred, y_prob = evaluate_model(model, test_loader, device=device)

    # 1. Classification Report
    print("\n--- EXPERIMENTAL CLASSIFICATION REPORT (NOT CLINICAL VALIDATION) ---")
    print(classification_report(y_true, y_pred, target_names=[c.capitalize() for c in CLASS_NAMES], digits=4, zero_division=0))

    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_path = FIGURES_DIR / f"confusion_matrix_{model_name}.png"
    plot_confusion_matrix_figure(cm, CLASS_NAMES, save_path=cm_path)
    print(f"Confusion Matrix saved to: {cm_path}")

    # 3. ROC Curves
    roc_path = FIGURES_DIR / f"roc_curves_{model_name}.png"
    plot_multiclass_roc(y_true, y_prob, CLASS_NAMES, save_path=roc_path)
    print(f"ROC Curves saved to: {roc_path}")

    # 4. Comprehensive Metrics
    metrics = calculate_medical_metrics(y_true, y_pred, y_prob, CLASS_NAMES)

    print("\n" + "=" * 65)
    print("         SUMMARY OF EXPERIMENTAL CLASSIFICATION METRICS         ")
    print("=" * 65)
    print(f"Overall Accuracy:            {metrics['accuracy']*100:6.2f}%")
    print(f"Macro F1-Score:              {metrics['f1_macro']:.4f}")
    print(f"Sensitivity (Recall Macro):  {metrics['recall_macro_sensitivity']*100:6.2f}%")
    print(f"Specificity (Macro):         {metrics['specificity_macro']*100:6.2f}%")
    print(f"Multi-class ROC-AUC:         {metrics['roc_auc']:.4f}")
    print("-" * 65)
    for cls in CLASS_NAMES:
        sens = metrics.get(f"{cls}_sensitivity", 0.0) * 100
        spec = metrics.get(f"{cls}_specificity", 0.0) * 100
        print(f"Class [{cls.capitalize():<9}] -> Sensitivity: {sens:5.1f}% | Specificity: {spec:5.1f}%")
    print("=" * 65 + "\n")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Breast Cancer Model on Test Set")
    parser.add_argument(
        "--model",
        type=str,
        default="breastnet",
        choices=["breastnet", "resnet18", "resnet50", "efficientnet_b0"],
    )
    parser.add_argument("--checkpoint", type=str, default=str(BEST_MODEL_PATH))
    args = parser.parse_args()

    run_evaluation(model_name=args.model, checkpoint_path=Path(args.checkpoint))


if __name__ == "__main__":
    main()
