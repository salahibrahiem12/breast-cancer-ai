"""Fit a fixed baseline and preprocessing candidate on development splits only."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ml.audit_tcia_evaluation import binary_metrics, group_intervals
from ml.prepare_dataset import dhash


def development_rows(source, manifest):
    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    owners, paths = {}, set()
    for row in rows:
        if row["split"] not in {"train", "validation", "excluded_previous_test"}:
            raise ValueError("Only development and explicitly excluded old-test rows are supported.")
        if not row["group_id"] or row["label"] not in {"benign", "malignant"}:
            raise ValueError("Missing group or unsupported label.")
        previous = owners.setdefault(row["group_id"], row["split"])
        if previous != row["split"]:
            raise ValueError("Duplicate/patient groups cross partitions.")
        if row["path"] in paths:
            raise ValueError("Repeated image path.")
        paths.add(row["path"])
    for index, row in enumerate(rows):
        for other in rows[:index]:
            if row["split"] != other["split"] and (
                row["sha256"] == other["sha256"]
                or (int(row["dhash"]) ^ int(other["dhash"])).bit_count() <= 5
            ):
                raise ValueError("Exact or near duplicates cross development/excluded partitions.")
    selected = [row for row in rows if row["split"] in {"train", "validation"}]
    # Excluded images are read only for integrity/duplicate checks, never features.
    for row in rows:
        path = (source / row["path"]).resolve()
        if not path.is_relative_to(source.resolve()):
            raise ValueError("Image path escapes development source.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Development image content changed.")
        if str(dhash(path)) != row["dhash"]:
            raise ValueError("Development perceptual hash changed.")
    for split in ("train", "validation"):
        if {row["label"] for row in selected if row["split"] == split} != {"benign", "malignant"}:
            raise ValueError("Both classes are required in each development split.")
    return rows, selected


def features(path, candidate):
    with Image.open(path) as image:
        gray = image.convert("L")
        if candidate == "aspect_contrast":
            gray = ImageOps.autocontrast(gray, cutoff=1)
            fill = int(np.median(np.asarray(gray)))
            gray = ImageOps.pad(gray, (32, 32), method=Image.Resampling.BILINEAR, color=fill)
        elif candidate == "baseline":
            gray = gray.resize((32, 32), resample=Image.Resampling.BILINEAR)
        else:
            raise ValueError("Unknown development candidate.")
        return np.asarray(gray, dtype=np.float64).reshape(-1) / 255


def compare(source, manifest, output):
    if output.exists():
        raise FileExistsError("Use a new experiment output directory.")
    all_rows, selected = development_rows(source, manifest)
    manifest_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    train_rows = [row for row in selected if row["split"] == "train"]
    validation_rows = [row for row in selected if row["split"] == "validation"]
    y_train = np.asarray([row["label"] == "malignant" for row in train_rows], dtype=int)
    y_validation = np.asarray([row["label"] == "malignant" for row in validation_rows], dtype=int)
    parameters, results, predictions = {}, {}, []
    for candidate in ("baseline", "aspect_contrast"):
        x_train = np.asarray([features(source / row["path"], candidate) for row in train_rows])
        x_validation = np.asarray([features(source / row["path"], candidate) for row in validation_rows])
        scaler = StandardScaler().fit(x_train)
        model = LogisticRegression(C=1, class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=42)
        model.fit(scaler.transform(x_train), y_train)
        scores = model.predict_proba(scaler.transform(x_validation))[:, list(model.classes_).index(1)]
        parameters.update({candidate + "_" + key: value for key, value in {
            "mean": scaler.mean_, "scale": scaler.scale_, "coef": model.coef_, "intercept": model.intercept_,
        }.items()})
        results[candidate] = {
            "metrics": binary_metrics(y_validation, scores),
            "confidence_intervals": group_intervals(y_validation, scores, [row["group_id"] for row in validation_rows], iterations=1000),
            "optimizer_iterations": int(model.n_iter_[0]),
        }
        predictions.extend({"candidate": candidate, "path": row["path"], "actual": row["label"],
                            "malignant_probability": float(score), "split": "validation"}
                           for row, score in zip(validation_rows, scores, strict=True))
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != manifest_digest:
        raise ValueError("Development manifest changed during fitting.")
    output.mkdir(parents=True)
    np.savez(output / "linear_models.npz", **parameters)
    # Reconstruct both models from saved non-executable numerical parameters.
    replay_error = {}
    with np.load(output / "linear_models.npz", allow_pickle=False) as saved:
        for candidate in results:
            x = np.asarray([features(source / row["path"], candidate) for row in validation_rows])
            logits = ((x - saved[candidate + "_mean"]) / saved[candidate + "_scale"]) @ saved[candidate + "_coef"].T + saved[candidate + "_intercept"]
            replay = 1 / (1 + np.exp(-logits.ravel()))
            expected = [row["malignant_probability"] for row in predictions if row["candidate"] == candidate]
            replay_error[candidate] = float(np.max(np.abs(replay - expected)))
            if replay_error[candidate] > 1e-10:
                raise ValueError("Saved model replay mismatch.")
    with (output / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    report = {
        "source": str(source.resolve()), "manifest_sha256": manifest_digest,
        "split_counts": {split: dict(Counter(row["label"] for row in all_rows if row["split"] == split))
                         for split in ("train", "validation", "excluded_previous_test")},
        "split_group_counts": {split: len({row["group_id"] for row in all_rows if row["split"] == split})
                               for split in ("train", "validation", "excluded_previous_test")},
        "patient_ids_available": all(bool(row["patient_id"]) for row in selected),
        "integrity_checked_source_images": len(all_rows),
        "near_duplicate_distance": 5, "threshold": .5, "seed": 42,
        "classifier": "StandardScaler fit on train only; L2 logistic regression C=1, balanced, lbfgs, max_iter=2000",
        "software": {"numpy": np.__version__, "sklearn": sklearn.__version__},
        "results": results, "saved_parameter_replay_max_error": replay_error,
        "tcia_evaluated": False, "fresh_final_cohort_available": False, "inference_enabled": False,
        "limitations": ["Exploratory reused validation data, not independent testing.",
                        "Duplicate groups are not patients; no patient mapping exists for this source.",
                        "No clinical or eligibility-gate reliability is established.",
                        "Intervals resample validation duplicate groups, not independent patients."],
    }
    (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"split_counts": report["split_counts"], "results": {
        key: value["metrics"] for key, value in results.items()}, "saved_parameter_replay_max_error": replay_error}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["source", "manifest", "output"]:
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    compare(args.source, args.manifest, args.output)


if __name__ == "__main__":
    main()
