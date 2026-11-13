"""Select the eligibility threshold on a dedicated split and report locked-test results."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.model import build_model
from ml.prepare_eligibility import validate_manifest
from ml.train_eligibility import EligibilityDataset, _collect_scores


def select_threshold(
    scores: np.ndarray, labels: np.ndarray, max_invalid_acceptance: float
) -> float | None:
    if not 0 <= max_invalid_acceptance <= 1:
        raise ValueError("max-invalid-acceptance must be between 0 and 1.")
    negative_scores = scores[labels == 0]
    positive_scores = scores[labels == 1]
    if not len(negative_scores) or not len(positive_scores):
        raise ValueError("Threshold-selection split needs both positive and negative images.")
    candidate_values = np.unique(np.concatenate(([0.0], scores, [1.0])))
    candidates = np.unique(
        np.concatenate(
            (candidate_values, np.nextafter(candidate_values, np.inf))
        )
    )
    candidates = candidates[candidates <= 1.0]
    eligible = [
        float(threshold)
        for threshold in candidates
        if float((negative_scores >= threshold).mean()) <= max_invalid_acceptance
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda threshold: (
            float((positive_scores >= threshold).mean()),
            -float((negative_scores >= threshold).mean()),
            threshold,
        ),
    )


def _bootstrap_rate(
    selected: np.ndarray,
    groups: np.ndarray,
    iterations: int = 2000,
    seed: int = 42,
) -> dict[str, float] | None:
    unique = np.unique(groups)
    if len(unique) < 2:
        return None
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(iterations):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([members[group] for group in sampled])
        estimates.append(float(selected[indices].mean()))
    return {
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
        "iterations": iterations,
    }


def _score_split(
    source: Path,
    manifest: Path,
    model: torch.nn.Module,
    split: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    dataset = EligibilityDataset(source, manifest, split)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    scores, labels = _collect_scores(model, loader, torch.device("cpu"))
    return scores, labels, dataset.rows


def evaluate(
    source: Path,
    manifest: Path,
    checkpoint_path: Path,
    output: Path,
    max_invalid_acceptance: float,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    manifest_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if checkpoint.get("manifest_sha256") != manifest_digest:
        raise ValueError(
            "Eligibility manifest is unbound or changed since training; "
            "evaluation requires a checkpoint bound to this exact manifest."
        )
    validate_manifest(manifest)
    if checkpoint.get("class_order") != ["not_breast_ultrasound", "breast_ultrasound"]:
        raise ValueError("Eligibility checkpoint class mapping is not supported.")
    if Path(checkpoint["manifest"]).resolve() != manifest.resolve():
        raise ValueError("Evaluation must use the exact locked manifest used during training.")
    if Path(checkpoint["training_source"]).resolve() != source.resolve():
        raise ValueError("Evaluation source differs from the training source.")
    model = build_model(num_classes=2, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    threshold_scores, threshold_labels, _ = _score_split(
        source, manifest, model, "threshold"
    )
    threshold = select_threshold(
        threshold_scores, threshold_labels, max_invalid_acceptance
    )
    test_scores, test_labels, test_rows = _score_split(source, manifest, model, "test")
    if threshold is None:
        overall_valid_acceptance = None
        overall_invalid_acceptance = None
        accepted = np.zeros(len(test_scores), dtype=bool)
    else:
        accepted = test_scores >= threshold
        positive = test_labels == 1
        negative = test_labels == 0
        overall_valid_acceptance = (
            float(accepted[positive].mean()) if positive.any() else None
        )
        overall_invalid_acceptance = (
            float(accepted[negative].mean()) if negative.any() else None
        )
    groups = np.asarray([row["group_id"] for row in test_rows])
    category_results: dict[str, object] = {}
    for category in sorted({row["category"] for row in test_rows if row["label"] == "negative"}):
        indices = np.asarray(
            [
                index
                for index, row in enumerate(test_rows)
                if row["label"] == "negative" and row["category"] == category
            ]
        )
        category_accepted = accepted[indices]
        category_results[category] = {
            "sample_count": int(len(indices)),
            "group_count": int(len(np.unique(groups[indices]))),
            "false_acceptance_rate": (
                float(category_accepted.mean()) if threshold is not None else None
            ),
            "confidence_interval_95": (
                _bootstrap_rate(category_accepted, groups[indices])
                if threshold is not None else None
            ),
        }
    positive_indices = np.flatnonzero(test_labels == 1)
    negative_indices = np.flatnonzero(test_labels == 0)
    report: dict[str, object] = {
        "model_version": checkpoint.get("model_version", "unknown"),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "manifest_sha256": manifest_digest,
        "sample_count": int(len(test_rows)),
        "evaluation_scope": checkpoint["evaluation_scope"],
        "patient_independent": checkpoint["evaluation_scope"] == "patient_grouped",
        "threshold_selection_split": "threshold",
        "threshold_selection_samples": int(len(threshold_labels)),
        "max_invalid_acceptance_target": max_invalid_acceptance,
        "selected_threshold": threshold,
        "final_test_split": "test",
        "valid_image_acceptance": (
            overall_valid_acceptance if threshold is not None else None
        ),
        "valid_image_sample_count": int(len(positive_indices)),
        "valid_image_group_count": int(len(np.unique(groups[positive_indices]))),
        "valid_image_acceptance_ci_95": (
            _bootstrap_rate(accepted[positive_indices], groups[positive_indices])
            if threshold is not None else None
        ),
        "invalid_image_false_acceptance": overall_invalid_acceptance,
        "invalid_image_sample_count": int(len(negative_indices)),
        "invalid_image_group_count": int(len(np.unique(groups[negative_indices]))),
        "invalid_image_false_acceptance_ci_95": (
            _bootstrap_rate(accepted[negative_indices], groups[negative_indices])
            if threshold is not None else None
        ),
        "negative_categories": category_results,
        "deployment_gate_pass": False,
        "research_inference_approved": False,
        "limitations": [
            "Threshold is selected only on the dedicated threshold-selection partition; final test data do not tune it.",
            "Intervals bootstrap patient groups when supplied, otherwise exact/near-duplicate groups; they do not establish clinical reliability.",
            "Per-category false-acceptance estimates depend on the negative examples and do not prove general OOD rejection.",
            "An accepted image is only ultrasound-like under this dataset/model; the gate does not verify breast anatomy or diagnostic suitability.",
            "Public cancer classification remains disabled pending independent review and external validation.",
        ],
    }
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != manifest_digest:
        raise ValueError("Eligibility manifest changed during evaluation; no report was saved.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    predictions_path = output.with_name("eligibility_test_predictions.csv")
    with predictions_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("path", "label", "category", "score", "accepted", "group_id"))
        for row, label, score, accept in zip(
            test_rows, test_labels, test_scores, accepted, strict=True
        ):
            writer.writerow(
                (
                    row["path"],
                    row["label"],
                    row["category"],
                    f"{score:.8f}",
                    bool(accept),
                    row["group_id"],
                )
            )
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-invalid-acceptance", type=float, required=True)
    args = parser.parse_args()
    evaluate(
        args.source, args.manifest, args.checkpoint, args.output,
        args.max_invalid_acceptance,
    )


if __name__ == "__main__":
    main()
