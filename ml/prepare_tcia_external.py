"""Prepare a biopsy-only external test cohort from the audited TCIA release."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from ml.inspect_tcia_breast import read_case_metadata, safe_members


def prepare(archive: Path, metadata: Path, inventory: Path, source: Path, output: Path) -> dict:
    if source.exists() or output.exists():
        raise FileExistsError("Use new source and output directories for the locked external cohort.")
    report = json.loads((inventory / "inventory.json").read_text(encoding="utf-8"))
    for path, key in ((archive, "archive_sha256"), (metadata, "metadata_sha256")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != report[key]:
            raise ValueError("TCIA source changed since inventory.")
    with (inventory / "cohort_inventory.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    cases = read_case_metadata(metadata)
    case_by_image = {row["Image_filename"]: row for row in cases}
    for row in rows:
        case = case_by_image.get(Path(row["path"]).name)
        if (case is None or row["label"] != case["Classification"]
                or row["verification"] != case["Verification"]
                or row["patient_id"] != f"tcia-breast-lesions-usg:{case['CaseID']}"):
            raise ValueError("Inventory metadata differs from the original clinical workbook.")
    expected = {row["Image_filename"] for row in cases
                if row["Verification"] == "confirmed by biopsy"
                and row["Classification"] in {"benign", "malignant"}}
    selected = [row for row in rows if row["verification"] == "confirmed by biopsy"
                and row["label"] in {"benign", "malignant"}]
    if {Path(row["path"]).name for row in selected} != expected or len(selected) != len(expected):
        raise ValueError("Inventory and biopsy-confirmed case selection disagree.")
    if {row["label"] for row in selected} != {"benign", "malignant"}:
        raise ValueError("External cohort must contain both binary classes.")
    payloads = []
    with ZipFile(archive) as zipfile:
        safe_members(zipfile)
        for row in selected:
            payload = zipfile.read(row["path"])
            if hashlib.sha256(payload).hexdigest() != row["sha256"]:
                raise ValueError("Inventory image hash mismatch.")
            payloads.append((row, payload))
    source.mkdir(parents=True)
    output.mkdir(parents=True)
    manifest_rows = []
    for row, payload in payloads:
        name = Path(row["path"]).name
        (source / name).write_bytes(payload)
        manifest_rows.append({
            "path": name, "label": row["label"], "patient_id": row["patient_id"],
            "group_id": row["patient_id"], "sha256": row["sha256"],
            "dhash": row["dhash"], "split": "test",
        })
    manifest = output / "split_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    prepared = {
        "source": str(source.resolve()), "source_doi": report["dataset_doi"],
        "class_order": ["benign", "malignant"],
        "evaluation_scope": "patient_grouped_external_cohort",
        "class_counts": dict(Counter(row["label"] for row in manifest_rows)),
        "image_count": len(manifest_rows), "excluded_case_count": len(cases) - len(manifest_rows),
        "inclusion_rule": "Biopsy-confirmed benign/malignant cases only, selected before predictions.",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "archive_sha256": report["archive_sha256"], "metadata_sha256": report["metadata_sha256"],
        "research_inference_approved": False,
        "limitations": [item for item in report["limitations"] if not item.startswith("Inventory only:")] + [
            "One source scan per patient is documented by the publisher; cross-source patient identity cannot be verified.",
            "Biopsy-selected population; excludes follow-up benign cases and normal images.",
            "Every included image is test-only and must not be used for training or tuning.",
        ],
    }
    (output / "preparation_report.json").write_text(json.dumps(prepared, indent=2), encoding="utf-8")
    print(json.dumps(prepared, indent=2))
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "metadata", "inventory", "source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.archive, args.metadata, args.inventory, args.source, args.output)


if __name__ == "__main__":
    main()
