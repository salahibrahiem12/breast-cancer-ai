"""Inventory the official TCIA BrEaST release without making train/test splits."""

import argparse
from collections import Counter
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
import stat
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np
from PIL import Image

from ml.audit_acquired_duplicates import phash_from_32x32

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
SOURCE = "https://www.cancerimagingarchive.net/collection/breast-lesions-usg/"


def safe_members(archive: ZipFile) -> dict[str, object]:
    members = {}
    total = 0
    for item in archive.infolist():
        name = item.filename
        path = PurePosixPath(name)
        if (path.is_absolute() or ".." in path.parts or "\\" in name
                or ":" in name or stat.S_ISLNK(item.external_attr >> 16)):
            raise ValueError("Unsafe archive member.")
        if name in members:
            raise ValueError("Duplicate archive member.")
        total += item.file_size
        if total > 1024**3 or item.file_size > 64 * 1024**2:
            raise ValueError("Archive expansion limit exceeded.")
        members[name] = item
    if archive.testzip() is not None:
        raise ValueError("Archive CRC validation failed.")
    return members


def read_case_metadata(path: Path) -> list[dict[str, str]]:
    """Read only case/image/mask identifiers, label and verification method."""
    with ZipFile(path) as archive:
        safe_members(archive)
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(node.itertext()) for node in
                       ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("s:si", NS)]
        rows = []
        for row in ET.fromstring(archive.read("xl/worksheets/sheet1.xml")).findall("s:sheetData/s:row", NS):
            values = {}
            for cell in row.findall("s:c", NS):
                column = re.sub(r"\d", "", cell.attrib["r"])
                node = cell.find("s:v", NS)
                value = node.text or "" if node is not None else ""
                if cell.get("t") == "s":
                    value = strings[int(value)]
                elif cell.get("t") == "inlineStr":
                    value = "".join(cell.find("s:is", NS).itertext())
                values[column] = value
            rows.append(values)
    headers, *records = rows
    wanted = {"CaseID", "Image_filename", "Mask_tumor_filename", "Mask_other_filename",
              "Classification", "Verification"}
    if not wanted.issubset(headers.values()):
        raise ValueError("TCIA workbook is missing required metadata columns.")
    result = [{headers[col]: row.get(col, "") for col in headers if headers[col] in wanted}
              for row in records if any(row.values())]
    if len({row["CaseID"] for row in result}) != len(result):
        raise ValueError("Duplicate case identifiers.")
    if len({row["Image_filename"] for row in result}) != len(result):
        raise ValueError("Duplicate case image filenames.")
    for row in result:
        if not row["CaseID"] or row["Classification"] not in {"benign", "malignant", "normal"}:
            raise ValueError("Missing case identifier or unsupported class label.")
        if not re.fullmatch(r"case\d+\.png", row["Image_filename"]):
            raise ValueError("Unexpected primary image filename.")
    return result


def fingerprint(payload: bytes) -> dict[str, str | int]:
    with Image.open(BytesIO(payload)) as image:
        image.load()
        rgb = image.convert("RGB")
        gray = image.convert("L")
        pixels = np.asarray(gray.resize((9, 8)), dtype=np.int16)
        dhash = sum(int(pixels[r, c] > pixels[r, c + 1]) << (r * 8 + c)
                    for r in range(8) for c in range(8))
        phash = phash_from_32x32(np.asarray(
            gray.resize((32, 32), resample=Image.Resampling.BOX), dtype=np.float64
        )[None])[0]
        return {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "pixel_sha256": hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest(),
            "width": image.width, "height": image.height,
            "dhash": str(dhash), "phash": str(phash),
        }


def inspect(archive_path: Path, metadata_path: Path, output: Path,
            comparison_source: Path | None = None) -> dict:
    if output.exists():
        raise FileExistsError("Inventory output already exists; choose a new directory.")
    if comparison_source is not None and not comparison_source.is_dir():
        raise FileNotFoundError("Comparison image directory is missing.")
    cases = read_case_metadata(metadata_path)
    rows = []
    with ZipFile(archive_path) as archive:
        members = safe_members(archive)
        image_members = {PurePosixPath(name).name: name for name in members if name.endswith(".png")}
        if len(image_members) != sum(name.endswith(".png") for name in members):
            raise ValueError("Ambiguous image basenames.")
        primary = {name for name in image_members if re.fullmatch(r"case\d+\.png", name)}
        if primary != {row["Image_filename"] for row in cases}:
            raise ValueError("Primary images and metadata do not match.")
        for row in cases:
            filename = row["Image_filename"]
            for mask_field in ("Mask_tumor_filename", "Mask_other_filename"):
                for mask in filter(None, re.split(r"[;,&]\s*", row[mask_field])):
                    if mask not in image_members:
                        raise ValueError("Referenced mask is missing.")
            rows.append({
                "path": image_members[filename], "label": row["Classification"],
                "patient_id": f"tcia-breast-lesions-usg:{row['CaseID']}",
                "verification": row["Verification"],
                **fingerprint(archive.read(image_members[filename])),
            })
        # Decode masks too; never include them as independent scans.
        for filename, member in image_members.items():
            if filename not in primary:
                with Image.open(BytesIO(archive.read(member))) as mask:
                    mask.load()
    comparison = []
    if comparison_source is not None:
        for path in sorted(comparison_source.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"} and "mask" not in path.stem.lower():
                comparison.append(fingerprint(path.read_bytes()))
    overlap = {}
    for key in ("sha256", "pixel_sha256"):
        known = {row[key] for row in comparison}
        overlap[key + "_matches"] = sum(row[key] in known for row in rows)
    for key in ("dhash", "phash"):
        overlap[key + "_pairs_distance_at_most_3"] = sum(
            (int(row[key]) ^ int(other[key])).bit_count() <= 3
            for row in rows for other in comparison
        )
    report = {
        "source": SOURCE, "dataset_doi": "10.7937/9WKK-Q141",
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "scan_count": len(rows), "case_count": len({row["patient_id"] for row in rows}),
        "class_counts": dict(Counter(row["label"] for row in rows)),
        "verification_counts": dict(Counter(row["verification"] for row in rows)),
        "archive_png_count": len(image_members), "mask_count": len(image_members) - len(primary),
        "within_cohort_repeated_pixel_occurrences": len(rows) - len({row["pixel_sha256"] for row in rows}),
        "comparison_image_count": len(comparison), "cross_source_overlap": overlap,
        "patient_mapping_basis": "TCIA and source paper state 256 scans from 256 patients; source CaseID is namespaced, not an independently verified identity.",
        "research_inference_approved": False,
        "limitations": [
            "Inventory only: no split, training, model evaluation, or threshold selection.",
            "Hash similarity is not proof of shared patients; absence of matches is not proof of independent patients across sources.",
            "Only four normal cases; labels use both biopsy and follow-up reference standards.",
            "No negative eligibility categories or documented quality labels are supplied by this breast-only cohort.",
        ],
    }
    output.mkdir(parents=True)
    with (output / "cohort_inventory.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "inventory.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparison-source", type=Path)
    args = parser.parse_args()
    inspect(args.archive, args.metadata, args.output, args.comparison_source)


if __name__ == "__main__":
    main()
