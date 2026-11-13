"""Validate, group, and split a licensed breast-ultrasound eligibility dataset."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import StratifiedGroupKFold

from ml.prepare_dataset import UnionFind, dhash


IMAGE_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg"}
POSITIVE_CATEGORY = "breast_ultrasound"
LABELS = {"breast_ultrasound", "negative"}
REQUIRED_NEGATIVE_CATEGORIES = {
    "other_ultrasound",
    "other_medical",
    "natural_image",
    "object",
    "screenshot",
    "low_quality",
}
SPLITS = {"train", "validation", "threshold", "test"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_manifest(source: Path, input_csv: Path) -> list[dict[str, str]]:
    source = source.resolve()
    rows: list[dict[str, str]] = []
    with input_csv.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {
            "image_path",
            "label",
            "category",
            "patient_id",
            "license",
            "source_url",
            "rights_verified",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Input CSV requires columns: {', '.join(sorted(required))}.")
        for row_number, raw in enumerate(reader, start=2):
            relative = Path(raw["image_path"].strip())
            image_path = (source / relative).resolve()
            try:
                image_path.relative_to(source)
            except ValueError as exc:
                raise ValueError(f"Image path escapes source at CSV row {row_number}.") from exc
            label = raw["label"].strip().lower()
            category = raw["category"].strip().lower()
            license_name = raw["license"].strip()
            source_url = raw["source_url"].strip()
            if label not in LABELS:
                raise ValueError(f"Unsupported label at CSV row {row_number}: {label}.")
            if (label == "breast_ultrasound") != (category == POSITIVE_CATEGORY):
                raise ValueError(
                    f"Label/category mismatch at CSV row {row_number}; positive rows must use "
                    f"category '{POSITIVE_CATEGORY}', negatives must name their source category."
                )
            if not license_name or not source_url.startswith(("https://", "http://")):
                raise ValueError(f"License and source_url are required at CSV row {row_number}.")
            if raw["rights_verified"].strip().lower() not in {"true", "yes"}:
                raise ValueError(
                    f"Rights must be explicitly reviewed and marked true at CSV row {row_number}."
                )
            if image_path.suffix.lower() not in IMAGE_SUFFIXES or not image_path.is_file():
                raise ValueError(f"Missing or unsupported image at CSV row {row_number}.")
            try:
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    image.convert("RGB").load()
            except (OSError, SyntaxError, UnidentifiedImageError, ValueError) as exc:
                raise ValueError(f"Invalid image at CSV row {row_number}: {exc}") from exc
            rows.append(
                {
                    "path": image_path.relative_to(source).as_posix(),
                    "label": label,
                    "category": category,
                    "patient_id": raw["patient_id"].strip(),
                    "license": license_name,
                    "source_url": source_url,
                    "sha256": _sha256(image_path),
                    "dhash": str(dhash(image_path)),
                }
            )
    if not rows:
        raise ValueError("No licensed eligibility images were listed.")
    if not any(row["label"] == "breast_ultrasound" for row in rows):
        raise ValueError("The dataset has no breast_ultrasound positive samples.")
    if not any(row["label"] == "negative" for row in rows):
        raise ValueError("The dataset has no negative samples.")
    return rows


def group_rows(rows: list[dict[str, str]], distance: int) -> list[str]:
    union = UnionFind(len(rows))
    exact: dict[str, int] = {}
    exact_labels: dict[str, str] = {}
    patients: dict[str, int] = {}
    hashes = [int(row["dhash"]) for row in rows]
    for index, row in enumerate(rows):
        if row["sha256"] in exact:
            if exact_labels[row["sha256"]] != row["label"]:
                raise ValueError("An exact duplicate image has conflicting eligibility labels.")
            union.union(index, exact[row["sha256"]])
        else:
            exact[row["sha256"]] = index
            exact_labels[row["sha256"]] = row["label"]
        if row["patient_id"]:
            if row["patient_id"] in patients:
                union.union(index, patients[row["patient_id"]])
            else:
                patients[row["patient_id"]] = index
    for index, value in enumerate(hashes):
        for other in range(index):
            if (value ^ hashes[other]).bit_count() <= distance:
                union.union(index, other)
    roots = [union.find(index) for index in range(len(rows))]
    compact = {root: f"group_{index:05d}" for index, root in enumerate(sorted(set(roots)))}
    groups = [compact[root] for root in roots]
    return groups


def validate_manifest(manifest: Path) -> list[dict[str, str]]:
    with manifest.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    required = {
        "path",
        "label",
        "category",
        "group_id",
        "patient_id",
        "split",
        "sha256",
        "license",
        "source_url",
        "rights_verified",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Eligibility manifest must contain {', '.join(sorted(required))}.")
    if any(row["split"] not in SPLITS for row in rows):
        raise ValueError("Eligibility manifest contains an unsupported split.")
    if any(not row["group_id"] for row in rows):
        raise ValueError("Every eligibility image requires a non-empty duplicate/patient group.")
    group_owner: dict[str, str] = {}
    seen_paths: set[str] = set()
    exact_owner: dict[str, tuple[str, str]] = {}
    patient_owner: dict[str, str] = {}
    categories = {row["category"] for row in rows}
    missing_categories = REQUIRED_NEGATIVE_CATEGORIES - categories
    if missing_categories:
        raise ValueError(
            "Eligibility data must include these challenging negative categories: "
            f"{', '.join(sorted(missing_categories))}."
        )
    for row in rows:
        if row["path"] in seen_paths:
            raise ValueError(f"Duplicate path in eligibility manifest: {row['path']}.")
        seen_paths.add(row["path"])
        if not row["license"] or not row["source_url"].startswith(("https://", "http://")):
            raise ValueError(f"Missing license/source URL in manifest for {row['path']}.")
        if row["rights_verified"].lower() != "true":
            raise ValueError(f"Rights are not attested in manifest for {row['path']}.")
        if row["label"] not in LABELS:
            raise ValueError(f"Unsupported eligibility label for {row['path']}.")
        if (row["label"] == "breast_ultrasound") != (row["category"] == POSITIVE_CATEGORY):
            raise ValueError(f"Eligibility label/category mismatch for {row['path']}.")
        previous = group_owner.setdefault(row["group_id"], row["split"])
        if previous != row["split"]:
            raise ValueError("Patient or duplicate groups cross eligibility partitions.")
        hash_owner = exact_owner.setdefault(row["sha256"], (row["group_id"], row["split"]))
        if hash_owner != (row["group_id"], row["split"]):
            raise ValueError("Exact duplicate images cross groups or partitions.")
        if row["patient_id"]:
            previous_split = patient_owner.setdefault(row["patient_id"], row["split"])
            if previous_split != row["split"]:
                raise ValueError("Patient IDs cross eligibility partitions.")
    for split in SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        if not {"breast_ultrasound", "negative"}.issubset(
            {row["label"] for row in split_rows}
        ):
            raise ValueError(f"The {split} partition must include positive and negative images.")
        absent = categories - {row["category"] for row in split_rows}
        if absent:
            raise ValueError(
                f"The {split} partition is missing image categories: {', '.join(sorted(absent))}."
            )
    return rows


def prepare(
    source: Path,
    input_csv: Path,
    output: Path,
    seed: int = 42,
    near_duplicate_distance: int = 3,
    overwrite: bool = False,
) -> dict[str, object]:
    rows = scan_manifest(source, input_csv)
    groups = group_rows(rows, near_duplicate_distance)
    for row, group in zip(rows, groups, strict=True):
        row["group_id"] = group
    categories = np.asarray([row["category"] for row in rows])
    missing_categories = REQUIRED_NEGATIVE_CATEGORIES - set(categories)
    if missing_categories:
        raise ValueError(
            "Eligibility data must include these challenging negative categories: "
            f"{', '.join(sorted(missing_categories))}."
        )
    groups_array = np.asarray(groups)
    if any(len(set(groups_array[categories == category])) < 5 for category in set(categories)):
        raise ValueError(
            "Each image category needs at least five independent patient/duplicate groups "
            "to populate train, validation, threshold-selection, and test partitions."
        )
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(rows)), categories, groups_array))
    assignments = [""] * len(rows)
    for split_name, indices in (
        ("test", folds[0][1]),
        ("threshold", folds[1][1]),
        ("validation", folds[2][1]),
        ("train", np.concatenate([folds[3][1], folds[4][1]])),
    ):
        for index in indices:
            assignments[int(index)] = split_name
    if any(not split for split in assignments):
        raise RuntimeError("Eligibility split did not assign every image.")
    split_groups = {
        split: {groups[index] for index, value in enumerate(assignments) if value == split}
        for split in ("train", "validation", "threshold", "test")
    }
    if any(
        split_groups[left] & split_groups[right]
        for index, left in enumerate(split_groups)
        for right in list(split_groups)[index + 1 :]
    ):
        raise RuntimeError("Patient or duplicate groups cross eligibility partitions.")
    for split in ("train", "validation", "threshold", "test"):
        if not {"breast_ultrasound", "negative"}.issubset(
            {rows[index]["label"] for index, value in enumerate(assignments) if value == split}
        ):
            raise ValueError(f"The {split} partition must include positive and negative images.")
    categories_by_split = {
        split: {
            category: sum(
                assignments[index] == split and row["category"] == category
                for index, row in enumerate(rows)
            )
            for category in sorted(set(categories))
        }
        for split in ("train", "validation", "threshold", "test")
    }
    for split in ("threshold", "test"):
        missing = [
            category
            for category in set(categories)
            if categories_by_split[split][category] == 0
        ]
        if missing:
            raise ValueError(f"The {split} partition is missing categories: {sorted(missing)}.")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "eligibility_manifest.csv"
    report_path = output / "eligibility_preparation.json"
    if not overwrite and (manifest_path.exists() or report_path.exists()):
        raise FileExistsError(f"Eligibility manifest already exists in {output}.")
    columns = (
        "path",
        "label",
        "category",
        "patient_id",
        "group_id",
        "sha256",
        "dhash",
        "license",
        "source_url",
        "rights_verified",
        "split",
    )
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row, split in zip(rows, assignments, strict=True):
            writer.writerow({**row, "rights_verified": "true", "split": split})
    report: dict[str, object] = {
        "source": str(source.resolve()),
        "manifest": str(manifest_path.resolve()),
        "seed": seed,
        "image_count": len(rows),
        "group_count": len(set(groups)),
        "patient_ids_complete": all(row["patient_id"] for row in rows),
        "evaluation_scope": (
            "patient_grouped" if all(row["patient_id"] for row in rows) else "duplicate_grouped_image_level"
        ),
        "category_counts_by_split": categories_by_split,
        "limitations": [
            "Dataset rights and license entries require independent curator verification.",
            "Perceptual hashes cannot detect all near duplicates or patient relationships.",
            "A validation split, a threshold-selection split, and final test split are group-separated.",
            "This eligibility target is breast-ultrasound-like versus the documented negative categories; acceptance does not prove breast anatomy or suitability.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    validate_manifest(manifest_path)
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-duplicate-distance", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    prepare(
        args.source,
        args.input_csv,
        args.output,
        args.seed,
        args.near_duplicate_distance,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
