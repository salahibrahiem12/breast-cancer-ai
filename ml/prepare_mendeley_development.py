"""Rebuild a development split while excluding the previously inspected test set."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import StratifiedGroupKFold

from ml.prepare_dataset import assign_groups, dhash


OLD_SPLITS = {"train", "validation", "test"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_development(
    source: Path,
    previous_manifest: Path,
    output: Path,
    seed: int = 42,
    near_duplicate_distance: int = 5,
) -> dict[str, object]:
    source = source.resolve()
    previous_manifest = previous_manifest.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing development split: {output}")
    with previous_manifest.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows or not {"path", "label", "patient_id", "sha256", "dhash", "split"}.issubset(
        rows[0]
    ):
        raise ValueError("The previous manifest is missing required dataset fields.")
    if any(row["split"] not in OLD_SPLITS for row in rows):
        raise ValueError("The previous manifest has unexpected partition labels.")

    labels = sorted({row["label"] for row in rows})
    if len(labels) < 2:
        raise ValueError("A binary or multiclass labeled source is required.")
    for row in rows:
        path = (source / row["path"]).resolve()
        try:
            path.relative_to(source)
        except ValueError as exc:
            raise ValueError(f"Manifest image path escapes the source: {row['path']}") from exc
        if not path.is_file() or _sha256(path) != row["sha256"]:
            raise ValueError(f"Source image changed since its prior manifest: {row['path']}")
        try:
            with Image.open(path) as image:
                image.verify()
            if str(dhash(path)) != row["dhash"]:
                raise ValueError(f"Perceptual hash changed for {row['path']}")
        except (OSError, SyntaxError, UnidentifiedImageError, ValueError) as exc:
            raise ValueError(f"Invalid source image {row['path']}: {exc}") from exc

    group_ids = assign_groups(rows, near_duplicate_distance)
    members_by_group: dict[str, list[int]] = defaultdict(list)
    labels_by_group: dict[str, set[str]] = defaultdict(set)
    for index, (row, group_id) in enumerate(zip(rows, group_ids, strict=True)):
        members_by_group[group_id].append(index)
        labels_by_group[group_id].add(row["label"])
    conflicting = [group for group, values in labels_by_group.items() if len(values) != 1]
    if conflicting:
        raise ValueError(
            "Near-duplicate grouping joins conflicting labels; manual review is required."
        )

    excluded_groups = {
        group_id for group_id, members in members_by_group.items()
        if any(rows[index]["split"] == "test" for index in members)
    }
    eligible_indices = [
        index for group_id, members in members_by_group.items()
        if group_id not in excluded_groups for index in members
    ]
    if len(eligible_indices) < 8:
        raise ValueError("Too few development images remain after excluding old-test groups.")
    eligible_labels = np.asarray([labels.index(rows[index]["label"]) for index in eligible_indices])
    eligible_groups = np.asarray([group_ids[index] for index in eligible_indices])
    if any(
        len(set(eligible_groups[eligible_labels == class_index])) < 4
        for class_index in range(len(labels))
    ):
        raise ValueError("Each class needs four independent duplicate groups for validation.")

    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    train_local, validation_local = next(
        splitter.split(np.zeros(len(eligible_indices)), eligible_labels, eligible_groups)
    )
    train_indices = {eligible_indices[int(index)] for index in train_local}
    validation_indices = {eligible_indices[int(index)] for index in validation_local}
    if train_indices & validation_indices:
        raise RuntimeError("Internal split error: train and validation rows overlap.")
    train_groups = {group_ids[index] for index in train_indices}
    validation_groups = {group_ids[index] for index in validation_indices}
    if train_groups & validation_groups:
        raise RuntimeError("Internal split error: duplicate groups cross development partitions.")
    if train_groups & excluded_groups or validation_groups & excluded_groups:
        raise RuntimeError("Internal split error: old-test groups entered development.")

    output.mkdir(parents=True)
    manifest_path = output / "split_manifest.csv"
    columns = ["path", "label", "patient_id", "group_id", "sha256", "dhash", "split"]
    split_counts: Counter[tuple[str, str]] = Counter()
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for index, row in enumerate(rows):
            if group_ids[index] in excluded_groups:
                split = "excluded_previous_test"
            elif index in train_indices:
                split = "train"
            else:
                split = "validation"
            writer.writerow(
                {
                    "path": row["path"],
                    "label": row["label"],
                    "patient_id": row["patient_id"],
                    "group_id": group_ids[index],
                    "sha256": row["sha256"],
                    "dhash": row["dhash"],
                    "split": split,
                }
            )
            split_counts[(split, row["label"])] += 1

    excluded_image_count = sum(len(members_by_group[group]) for group in excluded_groups)
    report = {
        "source": str(source),
        "previous_manifest": str(previous_manifest),
        "seed": seed,
        "class_order": labels,
        "near_duplicate_distance": near_duplicate_distance,
        "image_count": len(rows),
        "duplicate_group_count": len(members_by_group),
        "patient_ids_present": all(bool(row["patient_id"]) for row in rows),
        "evaluation_scope": "duplicate_grouped_image_level_development_only",
        "patient_independent": False,
        "previous_test_images_excluded": sum(row["split"] == "test" for row in rows),
        "old_test_linked_images_excluded_from_development": excluded_image_count,
        "split_counts": {
            split: {
                label: split_counts[(split, label)]
                for label in labels
                if split_counts[(split, label)]
            }
            for split in ("train", "validation", "excluded_previous_test")
        },
        "test_partition": None,
        "limitations": [
            "The original test split was previously inspected and is completely excluded, along with every duplicate group connected to it at the configured perceptual-hash distance.",
            "Validation estimates are development-only and no final test set is created by this command.",
            "Patient IDs are absent; perceptual duplicate grouping cannot establish patient independence.",
            "A new independent, patient-level external test cohort is still required.",
        ],
    }
    (output / "preparation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--previous-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-duplicate-distance", type=int, default=5)
    args = parser.parse_args()
    if not 0 <= args.near_duplicate_distance <= 64:
        parser.error("near-duplicate-distance must be between 0 and 64.")
    print(
        json.dumps(
            prepare_development(
                args.source,
                args.previous_manifest,
                args.output,
                args.seed,
                args.near_duplicate_distance,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
