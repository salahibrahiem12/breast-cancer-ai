"""Audit a fixed, already inspected TCIA evaluation without tuning its model."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_auc_score
from scipy.stats import spearmanr

from ml.inspect_tcia_breast import read_case_metadata


def binary_metrics(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if (labels.ndim != 1 or scores.shape != labels.shape or not len(labels)
            or not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all()
            or ((scores < 0) | (scores > 1)).any()):
        raise ValueError("Expected paired binary labels and finite probabilities in [0, 1].")
    actual, called = labels == 1, scores >= 0.5
    tp, tn = int((actual & called).sum()), int((~actual & ~called).sum())
    fp, fn = int((~actual & called).sum()), int((actual & ~called).sum())
    ratio = lambda numerator, denominator: numerator / denominator if denominator else None
    sensitivity, specificity = ratio(tp, tp + fn), ratio(tn, tn + fp)
    confidence = np.maximum(scores, 1 - scores)
    correct = actual == called
    ece = 0.0
    for index in range(10):
        selected = (confidence >= index / 10) & (
            confidence <= 1 if index == 9 else confidence < (index + 1) / 10
        )
        if selected.any():
            ece += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    two_classes = len(np.unique(labels)) == 2
    precision, recall, _ = precision_recall_curve(labels, scores) if two_classes else ([], [], [])
    clipped = np.clip(scores, 1e-15, 1 - 1e-15)
    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "sensitivity": sensitivity, "specificity": specificity,
        "precision_ppv": ratio(tp, tp + fp), "npv": ratio(tn, tn + fn),
        "accuracy": float(correct.mean()),
        "balanced_accuracy": (sensitivity + specificity) / 2 if two_classes else None,
        "roc_auc": float(roc_auc_score(labels, scores)) if two_classes else None,
        "pr_auc_average_precision": float(average_precision_score(labels, scores)) if two_classes else None,
        "pr_auc_trapezoid": float(auc(recall, precision)) if two_classes else None,
        "malignant_prevalence": float(actual.mean()),
        "binary_brier": float(np.mean((scores - labels) ** 2)),
        "two_class_brier": float(2 * np.mean((scores - labels) ** 2)),
        "log_loss": float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))),
        "top_label_ece_10_bins": float(ece),
    }


def group_intervals(labels, scores, groups, iterations=2000, seed=42):
    if not isinstance(iterations, int) or iterations < 1:
        raise ValueError("Bootstrap iterations must be a positive integer.")
    labels, scores, groups = np.asarray(labels), np.asarray(scores), np.asarray(groups)
    if len(groups) != len(labels) or any(not str(group).strip() for group in groups):
        raise ValueError("Every observation requires a nonempty sampling group.")
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("At least two independent sampling groups are required.")
    members = [np.flatnonzero(groups == group) for group in unique]
    keys = [key for key in binary_metrics(labels, scores) if key not in {"tn", "fp", "fn", "tp"}]
    draws = {key: [] for key in keys}
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        indices = np.concatenate([members[index] for index in rng.integers(len(unique), size=len(unique))])
        values = binary_metrics(labels[indices], scores[indices])
        for key in keys:
            if values[key] is not None:
                draws[key].append(values[key])
    return {key: {
        "lower_95": float(np.quantile(values, 0.025)) if values else None,
        "upper_95": float(np.quantile(values, 0.975)) if values else None,
        "valid_resamples": len(values),
    } for key, values in draws.items()}


def image_features(path):
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
        gray = np.asarray(image.convert("L"))
        height, width = gray.shape
        border = np.ones(gray.shape, dtype=bool)
        border[height // 10:height - height // 10, width // 10:width - width // 10] = False
        return {
            "width": width, "height": height, "aspect_ratio": width / height,
            "black_fraction": float((gray < 10).mean()),
            "black_border_fraction": float((gray[border] < 10).mean()),
            "bright_border_fraction": float((gray[border] > 230).mean()),
            "color_fraction": float((rgb.max(axis=2).astype(int) - rgb.min(axis=2) > 25).mean()),
            "mean_intensity": float(gray.mean()), "intensity_std": float(gray.std()),
        }


def artifact_correlations(labels, scores, feature_rows):
    result = {}
    for key in ("width", "height", "aspect_ratio", "black_fraction", "black_border_fraction", "color_fraction", "mean_intensity", "intensity_std"):
        values = np.asarray([row[key] for row in feature_rows])
        result[key] = {
            "spearman_with_malignant_score": float(spearmanr(values, scores).statistic) if np.ptp(values) else None,
            "spearman_with_ground_truth": float(spearmanr(values, labels).statistic) if np.ptp(values) else None,
        }
    return result


def contacts(rows, source, output, prefix):
    """Local review only; hide outer metadata bands and all source identifiers."""
    paths = []
    for page, start in enumerate(range(0, len(rows), 16), 1):
        subset = rows[start:start + 16]
        sheet = Image.new("RGB", (1120, 1000), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(subset):
            with Image.open(source / row["path"]) as original:
                width, height = original.size
                # Privacy review crop, not an inference transform or lesion ROI.
                image = original.convert("RGB").crop((int(width * .15), int(height * .20),
                                                       int(width * .85), int(height * .85)))
                image = ImageOps.contain(image, (270, 205))
            x, y = index % 4 * 280, index // 4 * 250
            sheet.paste(image, (x + (280 - image.width) // 2, y + 28))
            label = f"{prefix}-{start + index + 1:03d}"
            if "malignant_probability" in row:
                label += f" p={float(row['malignant_probability']):.3f}"
            draw.text((x + 8, y + 6), label, fill="black")
        path = output / f"{prefix}_{page:02d}.png"
        sheet.save(path)
        paths.append(str(path.resolve()))
    return paths


def audit(source, manifest, predictions, checkpoint_path, metadata, development_source, development_manifest, output):
    import torch
    import torchvision
    import PIL
    from torch.utils.data import DataLoader
    from ml.model import IMAGE_MEAN, IMAGE_SIZE, IMAGE_STD, load_checkpoint
    from ml.train import ManifestDataset, collect_logits

    if output.exists():
        raise FileExistsError("Use a new audit output directory.")
    read_csv = lambda path: list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    manifest_rows, rows = read_csv(manifest), read_csv(predictions)
    cases = {case["Image_filename"]: case for case in read_case_metadata(metadata)}
    indexed = {row["path"]: row for row in manifest_rows}
    if len(indexed) != len(manifest_rows) or len({row["path"] for row in rows}) != len(rows) or set(indexed) != {row["path"] for row in rows}:
        raise ValueError("Predictions and manifest must match one-to-one.")
    for row in rows:
        entry, case = indexed[row["path"]], cases[row["path"]]
        if (row["actual"] != case["Classification"] or entry["label"] != case["Classification"]
                or case["Verification"] != "confirmed by biopsy" or entry["split"] != "test"
                or entry["patient_id"] != f"tcia-breast-lesions-usg:{case['CaseID']}"
                or entry["group_id"] != entry["patient_id"]
                or hashlib.sha256((source / row["path"]).read_bytes()).hexdigest() != entry["sha256"]):
            raise ValueError("Source, ground truth, grouping, split or image integrity mismatch.")
    groups = [indexed[row["path"]]["patient_id"] for row in rows]
    if len(set(groups)) != len(rows):
        raise ValueError("Repeated patient: one-image-per-patient assumption needs revision.")
    torch.set_num_threads(2)
    model, checkpoint = load_checkpoint(str(checkpoint_path), torch.device("cpu"))
    if checkpoint["class_order"] != ["benign", "malignant"]:
        raise ValueError("Unexpected class order.")
    temperature = float(checkpoint["temperature"])
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid checkpoint temperature.")
    dataset = ManifestDataset(source, manifest, "test", tuple(checkpoint["class_order"]))
    logits, targets, replay_paths = collect_logits(model, DataLoader(dataset, batch_size=32, shuffle=False), torch.device("cpu"))
    replay = dict(zip(replay_paths, torch.softmax(logits / temperature, dim=1)[:, 1].numpy().tolist(), strict=True))
    scores = np.asarray([float(row["malignant_probability"]) for row in rows])
    labels = np.asarray([int(row["actual"] == "malignant") for row in rows])
    maximum_difference = max(abs(replay[row["path"]] - score) for row, score in zip(rows, scores, strict=True))
    if maximum_difference > 1e-6:
        raise ValueError("Checkpoint replay differs from saved predictions.")
    expected_calls = np.where(scores >= .5, "malignant", "benign")
    if any(row["predicted"] != called for row, called in zip(rows, expected_calls, strict=True)):
        raise ValueError("Saved classifications disagree with the fixed 0.5 threshold.")
    output.mkdir(parents=True)
    features, buckets = [], {"FP": [], "FN": [], "TN": [], "TP": []}
    for row, actual, score in zip(rows, labels, scores, strict=True):
        bucket = ("T" if (score >= .5) == bool(actual) else "F") + ("P" if score >= .5 else "N")
        buckets[bucket].append(row)
        features.append({"outcome": bucket, **image_features(source / row["path"])})
    development = [row for row in read_csv(development_manifest) if row["split"] in {"train", "validation"}]
    dev_features = [image_features(development_source / row["path"]) for row in development]
    summaries = {}
    for bucket in ["FP", "FN", "TN", "TP", "external_all", "development"]:
        subset = dev_features if bucket == "development" else features if bucket == "external_all" else [row for row in features if row["outcome"] == bucket]
        summaries[bucket] = {key: {"median": float(np.median([row[key] for row in subset])),
                                  "min": float(min(row[key] for row in subset)),
                                  "max": float(max(row[key] for row in subset))}
                             for key in dev_features[0]} if subset else None
    sheets = {name: contacts(bucket, source, output, name) for name, bucket in buckets.items() if name in {"FP", "FN"}}
    result = {
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        "observations": len(rows), "reported_patient_groups": len(set(groups)),
        "unit": "one exported case image per source-reported patient, not lesion or independently verified examination",
        "ground_truth_matches": len(rows), "class_counts": dict(Counter(row["actual"] for row in rows)),
        "checkpoint": {key: checkpoint.get(key) for key in ["architecture", "class_order", "image_size", "normalization", "temperature", "best_epoch", "training_source", "manifest"]},
        "preprocessing": {"convert": "RGB", "resize": [IMAGE_SIZE, IMAGE_SIZE], "interpolation": "PIL bilinear via torchvision Resize, antialias=True", "mean": IMAGE_MEAN, "std": IMAGE_STD, "crop": None, "augmentation": False, "aspect_ratio_preserved": False},
        "actual_transform": repr(dataset.transform),
        "software": {"torch": torch.__version__, "torchvision": torchvision.__version__, "Pillow": PIL.__version__, "numpy": np.__version__},
        "threshold": .5, "threshold_ties": int((scores == .5).sum()),
        "replay_max_absolute_probability_difference": maximum_difference,
        "confusion_matrix_rows_actual_columns_predicted_benign_malignant": [[int(((labels == 0) & (scores < .5)).sum()), int(((labels == 0) & (scores >= .5)).sum())], [int(((labels == 1) & (scores < .5)).sum()), int(((labels == 1) & (scores >= .5)).sum())]],
        "metrics": binary_metrics(labels, scores),
        "confidence_intervals": {"method": "nonparametric patient-group percentile bootstrap", "iterations": 2000, "seed": 42, "unit": "source-reported patient", "values": group_intervals(labels, scores, groups)},
        "image_feature_summaries": summaries, "local_privacy_cropped_review_sheets": sheets,
        "artifact_correlations": artifact_correlations(labels, scores, features),
        "inspected_external_cohort": True, "fresh_final_cohort_available": False,
        "inference_enabled": False, "eligibility_gate_performance": None,
        "limitations": ["No radiologist error adjudication or causal artifact attribution.", "Publisher-backed patient grouping; cross-source patient identity cannot be verified.", "Biopsy-selected cohort; PPV/NPV reflect this cohort's prevalence.", "No threshold or model selection on TCIA; replay verifies the fixed pipeline only.", "Review sheets omit outer bands for privacy and cannot assess obscured metadata text."],
    }
    (output / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ["observations", "reported_patient_groups", "replay_max_absolute_probability_difference", "metrics", "confidence_intervals"]}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["source", "manifest", "predictions", "checkpoint", "metadata", "development-source", "development-manifest", "output"]:
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    audit(args.source, args.manifest, args.predictions, args.checkpoint, args.metadata,
          args.development_source, args.development_manifest, args.output)


if __name__ == "__main__":
    main()
