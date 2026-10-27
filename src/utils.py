"""
Utility functions for PyTorch Breast Cancer Detection.
Includes seeding, model checkpointing, metric calculations, and medical visualization helpers.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set random seeds across Python, NumPy, and PyTorch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metric_val: float, path: Path):
    """Save model weights, optimizer state, and training metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "metric_val": metric_val,
    }
    torch.save(state, path)


def load_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None, device: torch.device = None):
    """Load model weights and optional optimizer state from a checkpoint file."""
    checkpoint = torch.save if not Path(path).exists() else torch.load(path, map_location=device or "cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("epoch", 0), checkpoint.get("metric_val", 0.0)


def calculate_medical_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, class_names: List[str]
) -> Dict[str, float]:
    """
    Compute comprehensive clinical and statistical performance metrics:
    - Overall Accuracy
    - Macro & Weighted F1-Scores
    - Sensitivity (Recall) per class & macro average
    - Specificity per class & macro average
    - Multi-class ROC-AUC (One-vs-Rest)
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # Calculate per-class Sensitivity and Specificity from Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    specificities = []
    sensitivities = []

    for i in range(len(class_names)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fp + fn)

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivities.append(sens)
        specificities.append(spec)

    # ROC-AUC calculation (One-vs-Rest)
    try:
        if y_prob.shape[1] == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:
        auc = 0.0

    metrics = {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "precision_macro": float(precision_macro),
        "recall_macro_sensitivity": float(recall_macro),
        "specificity_macro": float(np.mean(specificities)),
        "roc_auc": float(auc),
    }

    # Add per-class sensitivity & specificity
    for idx, name in enumerate(class_names):
        metrics[f"{name}_sensitivity"] = float(sensitivities[idx])
        metrics[f"{name}_specificity"] = float(specificities[idx])

    return metrics


def plot_training_curves(history: Dict[str, List[float]], save_path: Path = None):
    """Generate professional loss and accuracy learning curves."""
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(epochs, history["train_loss"], "o-", color="#1f77b4", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "s-", color="#ff7f0e", label="Validation Loss", linewidth=2)
    ax1.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Accuracy curve
    ax2.plot(epochs, history["train_acc"], "o-", color="#2ca02c", label="Train Accuracy", linewidth=2)
    ax2.plot(epochs, history["val_acc"], "s-", color="#d62728", label="Validation Accuracy", linewidth=2)
    ax2.set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix_figure(cm: np.ndarray, class_names: List[str], save_path: Path = None):
    """Plot an annotated, clinical-grade confusion matrix heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[cls.capitalize() for cls in class_names],
        yticklabels=[cls.capitalize() for cls in class_names],
        cbar=True,
        annot_kws={"size": 13, "weight": "bold"},
    )
    plt.title("Confusion Matrix - Diagnostic Classification", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Medical Diagnosis", fontsize=12, labelpad=10)
    plt.ylabel("True Pathological Diagnosis", fontsize=12, labelpad=10)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_multiclass_roc(y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str], save_path: Path = None):
    """Plot Receiver Operating Characteristic (ROC) curves for each breast cancer category."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve

    plt.figure(figsize=(8, 6))
    colors = ["#2b5c8f", "#d95f02", "#7570b3"]

    for i, class_name in enumerate(class_names):
        y_true_binary = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
        try:
            auc_score = roc_auc_score(y_true_binary, y_prob[:, i])
        except Exception:
            auc_score = 0.5
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, label=f"{class_name.capitalize()} (AUC = {auc_score:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7, label="Random Guess (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.title("Multi-Class ROC Curves", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
