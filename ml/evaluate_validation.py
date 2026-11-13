"""Evaluate classifier candidates on validation only; this command never reads test rows."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from ml.model import load_checkpoint
from ml.train import ManifestDataset, collect_logits


THRESHOLDS = (0.25, 0.4, 0.5, 0.6, 0.75)


def evaluate_validation(
    source: Path, manifest: Path, checkpoint_path: Path
) -> dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(str(checkpoint_path), device)
    class_names = tuple(checkpoint["class_order"])
    dataset = ManifestDataset(source, manifest, "validation", class_names)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    logits, labels, paths = collect_logits(model, loader, device)
    probabilities = torch.softmax(logits / float(checkpoint["temperature"]), dim=1).numpy()
    targets = labels.numpy()
    predictions = probabilities.argmax(axis=1)
    report = classification_report(
        targets,
        predictions,
        labels=np.arange(len(class_names)),
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )

    operating_points = None
    if "malignant" in class_names:
        malignant_index = class_names.index("malignant")
        actual_malignant = targets == malignant_index
        other_actual = ~actual_malignant
        operating_points = []
        for threshold in THRESHOLDS:
            calls_malignant = probabilities[:, malignant_index] >= threshold
            operating_points.append(
                {
                    "malignant_probability_threshold": threshold,
                    "sensitivity": (
                        float((calls_malignant & actual_malignant).sum() / actual_malignant.sum())
                        if actual_malignant.any()
                        else None
                    ),
                    "specificity": (
                        float((~calls_malignant & other_actual).sum() / other_actual.sum())
                        if other_actual.any()
                        else None
                    ),
                    "false_positive_count": int((calls_malignant & other_actual).sum()),
                    "false_negative_count": int((~calls_malignant & actual_malignant).sum()),
                }
            )

    with manifest.open(newline="", encoding="utf-8") as file:
        rows = {row["path"]: row for row in csv.DictReader(file)}
    validation_group_ids = sorted({rows[path]["group_id"] for path in paths})
    one_hot = np.eye(len(class_names), dtype=np.float64)[targets]
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    confidence = probabilities.max(axis=1)
    correct = predictions == targets
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        selected = (confidence >= lower) & (
            confidence < upper if upper < 1.0 else confidence <= upper
        )
        if selected.any():
            ece += float(selected.mean()) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )

    return {
        "model_version": checkpoint["model_version"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "class_order": list(class_names),
        "validation_sample_count": len(targets),
        "validation_group_count": len(validation_group_ids),
        "validation_is_used_for_model_selection": True,
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "per_class": {
            name: {
                "support": int(report[name]["support"]),
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
            }
            for name in class_names
        },
        "confusion_matrix_actual_rows_predicted_columns": confusion_matrix(
            targets, predictions, labels=np.arange(len(class_names))
        ).tolist(),
        "malignant_operating_points": operating_points,
        "calibration": {
            "multiclass_brier_score": brier,
            "top_label_ece_10_bins": ece,
            "temperature": float(checkpoint["temperature"]),
            "temperature_fit_split": "validation",
        },
        "limitations": [
            "Validation data were used for checkpoint selection and temperature fitting; these results are not an independent performance estimate.",
            "The already-inspected test partition is deliberately not loaded by this evaluator.",
            "Patient IDs are unavailable; reported groups are duplicate groups, not patients.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_validation(args.source, args.manifest, args.checkpoint)
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
