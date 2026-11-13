"""Validate, deduplicate, group, and split a class-folder image dataset."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import StratifiedGroupKFold


CLASS_NAMES = ("benign", "malignant", "normal")
IMAGE_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg"}


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def dhash(path: Path) -> int:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((9, 8)), dtype=np.int16)
    return sum(
        int(pixels[row, col] > pixels[row, col + 1]) << (row * 8 + col)
        for row in range(8)
        for col in range(8)
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_patient_map(metadata_csv: Path | None, source: Path) -> dict[str, str]:
    if metadata_csv is None:
        return {}
    mapping: dict[str, str] = {}
    with metadata_csv.open(newline="", encoding="utf-8-sig") as metadata:
        reader = csv.DictReader(metadata)
        if not {"image_path", "patient_id"}.issubset(reader.fieldnames or []):
            raise ValueError("Metadata CSV requires image_path and patient_id columns.")
        for row in reader:
            image_key = row["image_path"].strip()
            patient_id = row["patient_id"].strip()
            if not image_key or not patient_id:
                raise ValueError("Patient metadata contains an empty image_path or patient_id.")
            image_path = Path(image_key)
            resolved_key = str(image_path.resolve() if image_path.is_absolute() else (source / image_path).resolve())
            if resolved_key in mapping:
                raise ValueError(f"Duplicate metadata entry for {image_key}.")
            mapping[resolved_key] = patient_id
    return mapping


def scan_images(source: Path, patient_map: dict[str, str]) -> list[dict[str, object]]:
    source = source.resolve()
    rows = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if "_mask" in path.stem.lower():
            continue
        label = path.parent.name.lower()
        if label not in CLASS_NAMES:
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
                image.convert("RGB").load()
            if width < 2 or height < 2:
                raise ValueError("image dimensions are too small")
        except (OSError, SyntaxError, UnidentifiedImageError, ValueError) as exc:
            raise ValueError(f"Invalid image {path}: {exc}") from exc

        patient_id = patient_map.get(str(path.resolve()))
        rows.append(
            {
                "path": path.relative_to(source).as_posix(),
                "label": label,
                "patient_id": patient_id or "",
                "sha256": sha256(path),
                "dhash": dhash(path),
            }
        )
    if not rows:
        raise ValueError(f"No supported class-folder images found under {source}.")
    missing_metadata = [row["path"] for row in rows if str((source / str(row["path"])).resolve()) not in patient_map]
    if patient_map and missing_metadata:
        raise ValueError(
            f"Patient metadata is incomplete: {len(missing_metadata)} dataset images are unmapped."
        )
    return rows


def assign_groups(rows: list[dict[str, object]], near_duplicate_distance: int) -> list[str]:
    union = UnionFind(len(rows))
    exact_hashes: dict[str, int] = {}
    patient_rows: dict[str, int] = {}
    hashes: list[int] = []
    for index, row in enumerate(rows):
        exact_hash = str(row["sha256"])
        if exact_hash in exact_hashes:
            union.union(index, exact_hashes[exact_hash])
        else:
            exact_hashes[exact_hash] = index
        patient_id = str(row["patient_id"])
        if patient_id:
            if patient_id in patient_rows:
                union.union(index, patient_rows[patient_id])
            else:
                patient_rows[patient_id] = index
        hashes.append(int(row["dhash"]))

    for index, image_hash in enumerate(hashes):
        for other in range(index):
            if (image_hash ^ hashes[other]).bit_count() <= near_duplicate_distance:
                union.union(index, other)

    roots = [union.find(index) for index in range(len(rows))]
    compact = {root: f"group_{position:05d}" for position, root in enumerate(sorted(set(roots)))}
    return [compact[root] for root in roots]


def make_splits(
    rows: list[dict[str, object]],
    class_names: tuple[str, ...],
    allow_image_level_split: bool,
    seed: int,
) -> tuple[list[str], str]:
    has_patient_ids = all(str(row["patient_id"]) for row in rows)
    if not has_patient_ids and not allow_image_level_split:
        raise ValueError(
            "No complete patient mapping was provided. Pass --allow-image-level-split "
            "to explicitly accept non-patient-independent evaluation."
        )
    groups = np.asarray([str(row["group_id"]) for row in rows])
    labels = np.asarray([class_names.index(str(row["label"])) for row in rows])
    if any(len(set(groups[labels == class_index])) < 3 for class_index in range(len(class_names))):
        raise ValueError(
            "At least three independent groups per class are needed for train/validation/test."
        )
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(rows)), labels, groups))
    train_validation_indices, test_indices = folds[0]
    validation_indices = folds[1][1]
    validation_set = set(validation_indices.tolist())
    train_indices = np.asarray(
        [index for index in train_validation_indices if int(index) not in validation_set],
        dtype=int,
    )
    if set(train_indices).intersection(set(test_indices)):
        raise RuntimeError("Internal split error: train and test overlap.")
    if set(validation_indices).intersection(set(test_indices)):
        raise RuntimeError("Internal split error: validation and test overlap.")
    if set(train_indices).intersection(set(validation_indices)):
        raise RuntimeError("Internal split error: train and validation overlap.")
    split_groups = {
        split_name: {groups[index] for index in indices}
        for split_name, indices in (
            ("train", train_indices),
            ("validation", validation_indices),
            ("test", test_indices),
        )
    }
    if (
        split_groups["train"] & split_groups["validation"]
        or split_groups["train"] & split_groups["test"]
        or split_groups["validation"] & split_groups["test"]
    ):
        raise RuntimeError("Internal split error: a patient or duplicate group crosses partitions.")
    assignments = [""] * len(rows)
    for split_name, indices in (
        ("train", train_indices),
        ("validation", validation_indices),
        ("test", test_indices),
    ):
        for index in indices:
            assignments[int(index)] = split_name
    if any(not value for value in assignments):
        raise RuntimeError("Internal split error: one or more images were not assigned.")
    for split_name in ("train", "validation", "test"):
        observed_classes = {
            str(rows[index]["label"])
            for index, value in enumerate(assignments)
            if value == split_name
        }
        missing_classes = set(class_names) - observed_classes
        if missing_classes:
            raise ValueError(
                f"The {split_name} partition lacks classes {sorted(missing_classes)}; "
                "use more independent groups or a different seed."
            )
    scope = "patient_grouped_internal_holdout" if has_patient_ids else "image_level_internal_holdout"
    return assignments, scope


def prepare_dataset(
    source: Path,
    output: Path,
    metadata_csv: Path | None = None,
    allow_image_level_split: bool = False,
    seed: int = 42,
    near_duplicate_distance: int = 3,
    overwrite: bool = False,
) -> dict[str, object]:
    if not 0 <= near_duplicate_distance <= 64:
        raise ValueError("near_duplicate_distance must be between 0 and 64 bits.")
    np.random.seed(seed)
    source = source.resolve()
    rows = scan_images(source, load_patient_map(metadata_csv, source))
    class_names = tuple(
        class_name for class_name in CLASS_NAMES
        if any(row["label"] == class_name for row in rows)
    )
    if len(class_names) < 2:
        raise ValueError("At least two documented classes are required for classification research.")
    group_ids = assign_groups(rows, near_duplicate_distance)
    for row, group_id in zip(rows, group_ids, strict=True):
        row["group_id"] = group_id
    labels_per_group: dict[str, set[str]] = {}
    for row in rows:
        labels_per_group.setdefault(str(row["group_id"]), set()).add(str(row["label"]))
    conflicting_groups = [group for group, labels in labels_per_group.items() if len(labels) > 1]
    if conflicting_groups:
        raise ValueError(
            "Near-duplicate or patient groups contain conflicting class labels; "
            f"review these groups before splitting: {conflicting_groups[:10]}"
        )
    assignments, evaluation_scope = make_splits(
        rows, class_names, allow_image_level_split, seed
    )
    for row, split_name in zip(rows, assignments, strict=True):
        row["split"] = split_name

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "split_manifest.csv"
    report_path = output / "preparation_report.json"
    if not overwrite and (manifest_path.exists() or report_path.exists()):
        raise FileExistsError(
            f"A prepared manifest already exists in {output}. "
            "Choose a new output directory or pass --overwrite intentionally."
        )
    columns = ("path", "label", "patient_id", "group_id", "sha256", "dhash", "split")
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest:
        writer = csv.DictWriter(manifest, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    counts = {
        split_name: {
            class_name: sum(
                row["split"] == split_name and row["label"] == class_name for row in rows
            )
            for class_name in class_names
        }
        for split_name in ("train", "validation", "test")
    }
    report = {
        "source": str(source),
        "manifest": str(manifest_path),
        "seed": seed,
        "class_order": list(class_names),
        "image_count": len(rows),
        "group_count": len(set(group_ids)),
        "evaluation_scope": evaluation_scope,
        "patient_independent": evaluation_scope == "patient_grouped_internal_holdout",
        "near_duplicate_distance": near_duplicate_distance,
        "split_counts": counts,
        "limitations": [
            "A grouped internal holdout is not external validation.",
            "Perceptual hashes cannot detect every duplicate or same-patient relationship.",
            "The dataset source license must be verified separately before training or sharing.",
        ],
    }
    report_path.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument("--allow-image-level-split", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-duplicate-distance", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    prepare_dataset(
        args.source,
        args.output,
        args.metadata_csv,
        args.allow_image_level_split,
        args.seed,
        args.near_duplicate_distance,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
