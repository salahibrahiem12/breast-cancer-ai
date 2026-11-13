"""One-pass held-out test evaluation with calibration and sampling-unit intervals."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from ml.model import CLASS_NAMES
from ml.train import ManifestDataset, collect_logits
from ml.model import load_checkpoint


def prediction_indices(probabilities: np.ndarray, class_names: tuple[str, ...]) -> np.ndarray:
    """Apply the documented binary >=0.5 rule, including exact ties."""
    if len(class_names) == 2 and "malignant" in class_names:
        malignant = class_names.index("malignant")
        return np.where(probabilities[:, malignant] >= 0.5, malignant, 1 - malignant)
    return probabilities.argmax(axis=1)


def calibration_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, class_names: tuple[str, ...]
) -> dict[str, float]:
    one_hot = np.eye(len(class_names), dtype=np.float64)[y_true]
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    confidence = probabilities.max(axis=1)
    correctness = prediction_indices(probabilities, class_names) == y_true
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        selected = (confidence >= lower) & (
            confidence < upper if upper < 1.0 else confidence <= upper
        )
        if selected.any():
            ece += float(selected.mean()) * abs(float(correctness[selected].mean()) - float(confidence[selected].mean()))
    return {"multiclass_brier_score": brier, "top_label_ece_10_bins": ece}


def metrics_for_sample(
    y_true: np.ndarray, probabilities: np.ndarray, class_names: tuple[str, ...]
) -> dict[str, float]:
    predictions = prediction_indices(probabilities, class_names)
    values = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(
            classification_report(
                y_true, predictions, labels=np.arange(len(class_names)),
                output_dict=True, zero_division=0,
            )["macro avg"]["f1-score"]
        ),
    }
    if "malignant" in class_names:
        malignant_index = class_names.index("malignant")
        positive = y_true == malignant_index
        negative = ~positive
        malignant_calls = probabilities[:, malignant_index] >= 0.5
        values["malignant_sensitivity_at_probability_0_5"] = (
            float((malignant_calls & positive).sum() / positive.sum())
            if positive.any() else float("nan")
        )
        values["malignant_specificity_at_probability_0_5"] = (
            float((~malignant_calls & negative).sum() / negative.sum())
            if negative.any() else float("nan")
        )
    return values


def bootstrap_intervals(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    class_names: tuple[str, ...],
    sampling_unit: str,
    iterations: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float] | str]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return {"sampling_unit": "unknown", "iterations": 0}
    members = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    values = {"macro_f1": []}
    if "malignant" in class_names:
        values["malignant_sensitivity_at_probability_0_5"] = []
        values["malignant_specificity_at_probability_0_5"] = []
    for _ in range(iterations):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([members[group] for group in sampled_groups])
        metrics = metrics_for_sample(y_true[indices], probabilities[indices], class_names)
        for name in values:
            value = metrics[name]
            if np.isfinite(value):
                values[name].append(value)
    return {
        "sampling_unit": sampling_unit,
        "iterations": iterations,
        **{
            name: {
                "lower_95": float(np.quantile(samples, 0.025)),
                "upper_95": float(np.quantile(samples, 0.975)),
            }
            for name, samples in values.items()
            if samples
        },
    }


def ensure_no_external_overlap(training_manifest: Path, external_manifest: Path) -> dict[str, int]:
    with training_manifest.open(newline="", encoding="utf-8") as file:
        training_rows = list(csv.DictReader(file))
    with external_manifest.open(newline="", encoding="utf-8") as file:
        external_rows = list(csv.DictReader(file))
    required = {"sha256", "dhash"}
    if not training_rows or not external_rows or not required.issubset(training_rows[0]) or not required.issubset(external_rows[0]):
        raise ValueError("Both manifests must contain SHA-256 and perceptual hashes.")

    training_exact = {row["sha256"] for row in training_rows}
    external_exact = {row["sha256"] for row in external_rows}
    exact_overlap = len(training_exact & external_exact)
    if len(training_rows) * len(external_rows) > 10_000_000:
        raise ValueError(
            "External near-duplicate scan exceeds the 10-million pair safety limit; "
            "use a dedicated scalable duplicate audit before evaluation."
        )
    training_hashes = [int(row["dhash"]) for row in training_rows]
    near_pairs = sum(
        (int(row["dhash"]) ^ training_hash).bit_count() <= 3
        and row["sha256"] not in training_exact
        for row in external_rows
        for training_hash in training_hashes
    )
    overlap = {"exact_sha256_matches": exact_overlap, "near_duplicate_pairs": near_pairs}
    if exact_overlap or near_pairs:
        raise ValueError(
            "External evaluation blocked because the external source overlaps the training manifest: "
            f"{overlap}."
        )
    return overlap


def evaluate(
    source: Path,
    manifest: Path,
    checkpoint_path: Path,
    output_dir: Path,
    bootstrap_iterations: int = 1000,
    external: bool = False,
) -> dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(str(checkpoint_path), device)
    class_names = tuple(checkpoint["class_order"])
    sampling_scope = checkpoint["evaluation_scope"]
    if external:
        if source.resolve() == Path(checkpoint["training_source"]).resolve():
            raise ValueError("External evaluation requires a separate source directory.")
        if manifest.resolve() == Path(checkpoint["manifest"]).resolve():
            raise ValueError("External evaluation requires a separate locked manifest.")
        external_report_path = manifest.parent / "preparation_report.json"
        if not external_report_path.is_file():
            raise FileNotFoundError(
                f"External preparation report is missing: {external_report_path}"
            )
        external_report = json.loads(external_report_path.read_text(encoding="utf-8"))
        expected_manifest_hash = external_report.get("manifest_sha256")
        if expected_manifest_hash is not None and hashlib.sha256(manifest.read_bytes()).hexdigest() != expected_manifest_hash:
            raise ValueError("External manifest changed after preparation.")
        if tuple(external_report.get("class_order", ())) != class_names:
            raise ValueError("External dataset class_order must match the trained model exactly.")
        sampling_scope = str(external_report.get("evaluation_scope", "image_level_internal_holdout"))
        cross_dataset_overlap = ensure_no_external_overlap(
            Path(checkpoint["manifest"]), manifest
        )
    else:
        cross_dataset_overlap = None
    dataset = ManifestDataset(source, manifest, "test", class_names)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    logits, labels, paths = collect_logits(model, loader, device)
    y_true = labels.numpy()
    temperature = float(checkpoint["temperature"])
    probabilities = torch.softmax(logits / temperature, dim=1).numpy()
    predictions = prediction_indices(probabilities, class_names)
    report = classification_report(
        y_true,
        predictions,
        labels=np.arange(len(class_names)),
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    one_hot = np.eye(len(class_names), dtype=np.float64)[y_true]
    auc_by_class: dict[str, dict[str, float | None]] = {}
    for index, class_name in enumerate(class_names):
        if np.unique(one_hot[:, index]).size < 2:
            auc_by_class[class_name] = {"roc_auc_ovr": None, "average_precision_ovr": None}
        else:
            auc_by_class[class_name] = {
                "roc_auc_ovr": float(roc_auc_score(one_hot[:, index], probabilities[:, index])),
                "average_precision_ovr": float(
                    average_precision_score(one_hot[:, index], probabilities[:, index])
                ),
            }
    valid_roc = [
        value["roc_auc_ovr"] for value in auc_by_class.values()
        if value["roc_auc_ovr"] is not None
    ]
    valid_pr = [
        value["average_precision_ovr"] for value in auc_by_class.values()
        if value["average_precision_ovr"] is not None
    ]
    macro_roc_auc = float(np.mean(valid_roc)) if valid_roc else None
    macro_pr_auc = float(np.mean(valid_pr)) if valid_pr else None
    confusion = confusion_matrix(y_true, predictions, labels=np.arange(len(class_names)))
    if "malignant" in class_names:
        malignant_index = class_names.index("malignant")
        malignant_calls = probabilities[:, malignant_index] >= 0.5
        malignant_actual = y_true == malignant_index
        malignant_specificity = (
            float((~malignant_calls & ~malignant_actual).sum() / (~malignant_actual).sum())
            if (~malignant_actual).any() else None
        )
        malignant_sensitivity = (
            float((malignant_calls & malignant_actual).sum() / malignant_actual.sum())
            if malignant_actual.any() else None
        )
    else:
        malignant_index = None
        malignant_specificity = malignant_sensitivity = None
    with manifest.open(newline="", encoding="utf-8") as manifest_file:
        manifest_rows = {row["path"]: row for row in csv.DictReader(manifest_file)}
    groups = []
    for path in paths:
        row = manifest_rows[path]
        groups.append(str(row["group_id"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, object] = {
        "model_version": checkpoint["model_version"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "class_order": list(class_names),
        "test_sample_count": len(y_true),
        "evaluation_scope": "external_dataset" if external else checkpoint["evaluation_scope"],
        "cohort_grouping_scope": sampling_scope,
        "external_evaluation": external,
        "cross_dataset_overlap_check": cross_dataset_overlap,
        "threshold_policy": (
            "malignant probability >= 0.5, predeclared; not tuned on test"
            if malignant_index is not None else "not applicable: dataset has no malignant class"
        ),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class": {
            class_name: {
                "support": int(report[class_name]["support"]),
                "precision": float(report[class_name]["precision"]),
                "recall": float(report[class_name]["recall"]),
                "f1": float(report[class_name]["f1-score"]),
                **auc_by_class[class_name],
            }
            for class_name in class_names
        },
        "confusion_matrix_rows_actual_columns_predicted": confusion.tolist(),
        "macro_roc_auc_ovr": macro_roc_auc,
        "macro_pr_auc_ovr_average_precision": macro_pr_auc,
        "malignant_sensitivity_at_probability_0_5": malignant_sensitivity,
        "malignant_specificity_at_probability_0_5": malignant_specificity,
        "malignant_false_negative_count": (
            int(((y_true == malignant_index) & (predictions != malignant_index)).sum())
            if malignant_index is not None else None
        ),
        "malignant_false_positive_count": (
            int(((y_true != malignant_index) & (predictions == malignant_index)).sum())
            if malignant_index is not None else None
        ),
        "calibration": calibration_metrics(y_true, probabilities, class_names),
        "confidence_intervals": bootstrap_intervals(
            y_true,
            probabilities,
            np.asarray(groups),
            class_names,
            (
                "patient_and_duplicate_group"
                if sampling_scope in {"patient_grouped_internal_holdout", "patient_grouped_external_cohort"}
                else "duplicate_group_or_image"
            ),
            bootstrap_iterations,
        ),
        "limitations": [
            (
                "The external dataset is not a clinical validation cohort."
                if external else "Internal holdout metrics do not establish external or clinical validity."
            ),
            "Patient-level interpretation is possible only when the manifest contains patient IDs.",
            "Calibration is fitted on validation data from the same source, not a separate external cohort.",
            "Metrics do not evaluate breast-ultrasound eligibility or OOD rejection.",
        ],
        "inference_enabled": False,
        "research_inference_approved": False,
    }
    report_path = output_dir / "test_metrics.json"
    report_path.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    malignant_probability_rows = (
        probabilities[:, malignant_index]
        if malignant_index is not None
        else np.full(len(probabilities), np.nan)
    )
    probability_header = (
        "malignant_probability" if malignant_index is not None else "positive_class_probability"
    )
    with (output_dir / "test_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as predictions_file:
        writer = csv.writer(predictions_file)
        writer.writerow(("path", "actual", "predicted", probability_header))
        writer.writerows(
            (
                path,
                class_names[int(actual)],
                class_names[int(predicted)],
                f"{malignant_probability_rows[index]:.8f}",
            )
            for index, (path, actual, predicted) in enumerate(
                zip(paths, y_true, predictions, strict=True)
            )
        )
    with (output_dir / "error_analysis.csv").open(
        "w", newline="", encoding="utf-8"
    ) as errors_file:
        writer = csv.writer(errors_file)
        writer.writerow(("path", "actual", "predicted", "error_type"))
        for path, actual, predicted in zip(paths, y_true, predictions, strict=True):
            if actual == predicted:
                continue
            if malignant_index is not None and actual == malignant_index:
                error_type = "malignant_false_negative"
            elif malignant_index is not None and predicted == malignant_index:
                error_type = "malignant_false_positive"
            else:
                error_type = "other_class_misclassification"
            writer.writerow(
                (path, class_names[int(actual)], class_names[int(predicted)], error_type)
            )
    plt.figure(figsize=(7, 6))
    for index, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(one_hot[:, index], probabilities[:, index])
        plt.plot(fpr, tpr, label=class_name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="#6b7280")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(
        "External dataset ROC curves (research only)"
        if external else "Internal held-out ROC curves (research only)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "test_roc.png", dpi=150)
    plt.close()

    confidence = probabilities.max(axis=1)
    correctness = predictions == y_true
    bin_edges = np.linspace(0.0, 1.0, 11)
    mean_confidence, accuracy_bins = [], []
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        selected = (confidence >= lower) & (
            confidence < upper if upper < 1.0 else confidence <= upper
        )
        if selected.any():
            mean_confidence.append(float(confidence[selected].mean()))
            accuracy_bins.append(float(correctness[selected].mean()))
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="#6b7280", label="Ideal")
    plt.plot(mean_confidence, accuracy_bins, marker="o", label="Model")
    plt.xlabel("Mean confidence")
    plt.ylabel("Observed accuracy")
    plt.title(
        "Calibration (10 bins, external dataset)"
        if external else "Calibration (10 bins, internal test)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "test_reliability.png", dpi=150)
    plt.close()
    print(json.dumps(metrics, indent=2, allow_nan=False))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/research_v1"))
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument(
        "--external",
        action="store_true",
        help="Evaluate a separate source directory and independently prepared test manifest.",
    )
    args = parser.parse_args()
    evaluate(
        args.source, args.manifest, args.checkpoint, args.output_dir,
        args.bootstrap_iterations, args.external,
    )


if __name__ == "__main__":
    main()
