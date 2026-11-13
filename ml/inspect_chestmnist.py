"""Verify and inventory the locally downloaded ChestMNIST 64 archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

EXPECTED_MD5 = "9de6cd0b934ebb5b7426cfba5efbae16"
SPLITS = ("train", "val", "test")


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_archive(path: Path, expected_md5: str = EXPECTED_MD5) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"ChestMNIST archive not found: {path}")
    actual_md5 = md5_file(path)
    if actual_md5 != expected_md5:
        raise ValueError(
            f"ChestMNIST archive MD5 mismatch: expected {expected_md5}, "
            f"received {actual_md5}"
        )

    splits = {}
    with np.load(path, allow_pickle=False) as archive:
        expected_keys = {
            f"{split}_{suffix}"
            for split in SPLITS
            for suffix in ("images", "labels")
        }
        if set(archive.files) != expected_keys:
            raise ValueError(
                f"Unexpected ChestMNIST archive members: {sorted(archive.files)}"
            )
        for split in SPLITS:
            images = archive[f"{split}_images"]
            labels = archive[f"{split}_labels"]
            if (
                images.ndim != 3
                or images.shape[1:] != (64, 64)
                or images.shape[0] == 0
                or images.dtype != np.uint8
                or images.shape[0] != labels.shape[0]
            ):
                raise ValueError(f"Invalid {split} image/label shapes.")
            if labels.ndim != 2 or labels.shape[1] != 14:
                raise ValueError(f"Invalid {split} multi-label target shape.")
            if not np.isin(labels, (0, 1)).all():
                raise ValueError(f"Non-binary targets found in {split} labels.")
            splits[split] = {
                "image_count": int(images.shape[0]),
                "image_shape": list(images.shape[1:]),
                "image_dtype": str(images.dtype),
                "pixel_min": int(images.min()),
                "pixel_max": int(images.max()),
                "label_column_count": int(labels.shape[1]),
                "positive_counts_by_unmapped_column": [
                    int(count) for count in labels.sum(axis=0)
                ],
            }

    return {
        "dataset": "ChestMNIST 64",
        "source_url": "https://zenodo.org/records/10519652",
        "license_record": "CC BY 4.0; source provenance and attribution review pending",
        "archive": str(path.resolve()),
        "archive_size_bytes": path.stat().st_size,
        "archive_md5": actual_md5,
        "split_assignments": "source-provided train/val/test partitions preserved",
        "patient_ids_available": False,
        "rights_verified_for_training": False,
        "splits": splits,
        "limitations": [
            "This is a low-resolution chest-radiograph source, not breast ultrasound.",
            "The archive does not expose patient identifiers.",
            "Label columns are counted but not mapped to names by this inventory.",
            "Source rights and attribution need curator review before gate training.",
            "This source alone cannot establish general ultrasound eligibility.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/raw/medmnist/chestmnist_64.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/prepared/medmnist_candidate_v1/inventory.json"),
    )
    args = parser.parse_args()
    report = inspect_archive(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
