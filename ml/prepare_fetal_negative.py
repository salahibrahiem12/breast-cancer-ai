"""Safely inventory the CC BY 4.0 fetal-ultrasound negative-source archive."""

import argparse
import csv
import hashlib
import json
import stat
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image, UnidentifiedImageError

from ml.prepare_eligibility import group_rows
from ml.prepare_dataset import dhash


SOURCE_URL = "https://zenodo.org/records/8265464"
LICENSE = "CC BY 4.0"
EXPECTED_ARCHIVE_SIZE = 129_809_023
EXPECTED_ARCHIVE_MD5 = "52f24da73f61cab9f3b1cfafa6c69cf2"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_archive(archive_path: Path, output_dir: Path) -> dict[str, object]:
    archive_path = archive_path.resolve()
    output_dir = output_dir.resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {archive_path}")
    if archive_path.stat().st_size != EXPECTED_ARCHIVE_SIZE:
        raise ValueError("Dataset archive size does not match the Zenodo record.")
    archive_md5 = md5_file(archive_path)
    if archive_md5 != EXPECTED_ARCHIVE_MD5:
        raise ValueError("Dataset archive MD5 does not match the Zenodo record.")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing inventory: {output_dir}")

    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True)
    rows: list[dict[str, str]] = []
    dimensions: Counter[str] = Counter()
    with zipfile.ZipFile(archive_path) as bundle:
        invalid_member = bundle.testzip()
        if invalid_member is not None:
            raise ValueError(f"ZIP CRC check failed for {invalid_member}.")
        for member in bundle.infolist():
            path = PurePosixPath(member.filename)
            if member.is_dir() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if path.is_absolute() or ".." in path.parts or "\\" in member.filename:
                raise ValueError(f"Unsafe archive path: {member.filename}")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Symbolic links are not allowed: {member.filename}")
            target = image_dir / path.name
            if target.exists():
                raise ValueError(f"Duplicate archive basename: {path.name}")
            with bundle.open(member) as source, target.open("wb") as destination:
                while chunk := source.read(1024 * 1024):
                    destination.write(chunk)
            try:
                with Image.open(target) as image:
                    image.verify()
                with Image.open(target) as image:
                    image.convert("RGB").load()
                    dimensions[f"{image.width}x{image.height}"] += 1
            except (OSError, SyntaxError, UnidentifiedImageError, ValueError) as exc:
                target.unlink()
                raise ValueError(f"Invalid source image {member.filename}: {exc}") from exc
            rows.append(
                {
                    "path": target.relative_to(output_dir).as_posix(),
                    "label": "negative",
                    "category": "other_ultrasound",
                    "patient_id": "",
                    "license": LICENSE,
                    "source_url": SOURCE_URL,
                    "rights_verified": "false",
                    "rights_review_status": "curator_review_required",
                    "split": "unassigned",
                    "sha256": sha256_file(target),
                    "dhash": str(dhash(target)),
                }
            )

    if not rows:
        raise ValueError("The verified source archive contains no supported images.")
    groups = group_rows(rows, distance=3)
    for row, group_id in zip(rows, groups, strict=True):
        row["duplicate_group_id"] = group_id

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "source_inventory.csv"
    columns = [
        "path",
        "label",
        "category",
        "patient_id",
        "license",
        "source_url",
        "rights_verified",
        "rights_review_status",
        "split",
        "sha256",
        "dhash",
        "duplicate_group_id",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "source": "Alzubaidi et al., Large-Scale Annotation Dataset for Fetal Head Biometry in Ultrasound Images",
        "source_url": SOURCE_URL,
        "archive": str(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_md5": archive_md5,
        "license": LICENSE,
        "image_count": len(rows),
        "decoded_image_count": sum(dimensions.values()),
        "dimensions": dict(sorted(dimensions.items())),
        "exact_and_near_duplicate_group_count_dhash_le_3": len(set(groups)),
        "patient_id_count": 0,
        "split_assignments": "unassigned; no patient metadata and no combined gate dataset",
        "rights_verified_for_training": False,
        "limitations": [
            "The dataset record states CC BY 4.0; curator review of underlying-image provenance and requested original-author attribution is still required.",
            "Fetal ultrasound is one non-breast-ultrasound source, not a representative sample of other anatomies or devices.",
            "Do not train or evaluate an eligibility gate from this source alone.",
            "Image-level duplicate groups do not establish patient independence.",
        ],
    }
    (output_dir / "inventory_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = inventory_archive(args.archive, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
