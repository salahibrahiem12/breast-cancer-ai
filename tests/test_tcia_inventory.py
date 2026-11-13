"""Generated file fixtures verify TCIA inventory logic, not model performance."""

import csv
import hashlib
import json
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from PIL import Image

from ml.inspect_tcia_breast import inspect, read_case_metadata, safe_members
from ml.prepare_tcia_external import prepare
from ml.evaluate import evaluate


def metadata(path, duplicate=False, two_classes=False):
    columns = ["CaseID", "Image_filename", "Mask_tumor_filename",
               "Mask_other_filename", "Classification", "Verification"]
    values = ["1", "case001.png", "case001_tumor.png",
              "case001_other1.png&case001_other2.png", "benign", "confirmed by biopsy"]
    sheet = ET.Element("worksheet", xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main")
    data = ET.SubElement(sheet, "sheetData")
    extra = [["2", "case002.png", "case002_tumor.png", "", "malignant", "confirmed by biopsy"]] if two_classes else []
    for index, row in enumerate([columns, values] + ([values] if duplicate else extra), 1):
        element = ET.SubElement(data, "row", r=str(index))
        for col, value in enumerate(row):
            cell = ET.SubElement(element, "c", r=f"{chr(65 + col)}{index}", t="inlineStr")
            ET.SubElement(ET.SubElement(cell, "is"), "t").text = value
    with ZipFile(path, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", ET.tostring(sheet))


class TciaInventoryTests(unittest.TestCase):
    def test_external_evaluation_rejects_changed_locked_manifest_before_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            manifest.write_text("locked test cohort")
            digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            (root / "preparation_report.json").write_text(json.dumps({"manifest_sha256": digest}))
            manifest.write_text("edited test cohort")
            checkpoint = {"class_order": ["benign", "malignant"],
                          "evaluation_scope": "image_level_internal_holdout",
                          "training_source": str(root / "train"), "manifest": str(root / "train.csv")}
            with (
                patch("ml.evaluate.load_checkpoint", return_value=(None, checkpoint)),
                patch("ml.evaluate.collect_logits") as inference,
                self.assertRaisesRegex(ValueError, "manifest changed"),
            ):
                evaluate(root / "test", manifest, root / "model.pth", root / "report", external=True)
            inference.assert_not_called()

    def test_external_preparation_keeps_biopsy_cases_test_only_and_detects_metadata_edits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            book = root / "metadata.xlsx"
            metadata(book, two_classes=True)
            buffer = BytesIO()
            Image.new("RGB", (16, 16), "gray").save(buffer, format="PNG")
            archive_path = root / "images.zip"
            with ZipFile(archive_path, "w") as archive:
                for name in ("case001.png", "case001_tumor.png", "case001_other1.png",
                             "case001_other2.png", "case002.png", "case002_tumor.png"):
                    archive.writestr("images/" + name, buffer.getvalue())
            inventory = root / "inventory"
            inspect(archive_path, book, inventory)
            output = root / "prepared"
            source = root / "source"
            report = prepare(archive_path, book, inventory, source, output)
            self.assertEqual(report["class_counts"], {"benign": 1, "malignant": 1})
            self.assertFalse(report["research_inference_approved"])
            with (output / "split_manifest.csv").open(newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual({row["split"] for row in rows}, {"test"})
            self.assertEqual(len(list(source.glob("*.png"))), 2)
            self.assertEqual((source / "case001.png").read_bytes(), buffer.getvalue())
            inventory_csv = inventory / "cohort_inventory.csv"
            inventory_csv.write_text(inventory_csv.read_text().replace(
                "tcia-breast-lesions-usg:1", "modified-patient"
            ))
            with self.assertRaisesRegex(ValueError, "differs from the original"):
                prepare(archive_path, book, inventory, root / "source2", root / "prepared2")

    def test_archive_rejects_unsafe_paths(self):
        for name in ("../bad.png", "/bad.png", "C:/bad.png", "folder\\bad.png"):
            with self.subTest(name=name), BytesIO() as buffer:
                with ZipFile(buffer, "w") as archive:
                    archive.writestr(name, b"bad")
                buffer.seek(0)
                with ZipFile(buffer) as archive:
                    # Windows ZipInfo normalizes backslashes when writing; restore
                    # the untrusted central-directory name for this validation test.
                    archive.infolist()[0].filename = name
                    with self.assertRaisesRegex(ValueError, "Unsafe"):
                        safe_members(archive)

    def test_metadata_rejects_duplicate_case_identifiers(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metadata.xlsx"
            metadata(path, duplicate=True)
            with self.assertRaisesRegex(ValueError, "Duplicate case"):
                read_case_metadata(path)

    def test_inventory_separates_masks_and_detects_exact_cross_source_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            book = root / "metadata.xlsx"
            metadata(book)
            buffer = BytesIO()
            Image.new("RGB", (16, 16), "gray").save(buffer, format="PNG")
            payload = buffer.getvalue()
            archive_path = root / "images.zip"
            with ZipFile(archive_path, "w") as archive:
                for filename in ("case001.png", "case001_tumor.png",
                                 "case001_other1.png", "case001_other2.png"):
                    archive.writestr("images/" + filename, payload)
            comparison = root / "comparison"
            comparison.mkdir()
            (comparison / "image.png").write_bytes(payload)
            output = root / "inventory"
            report = inspect(archive_path, book, output, comparison)
            self.assertEqual(report["scan_count"], 1)
            self.assertEqual(report["mask_count"], 3)
            self.assertEqual(report["cross_source_overlap"]["sha256_matches"], 1)
            self.assertFalse(report["research_inference_approved"])
            with (output / "cohort_inventory.csv").open(newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            self.assertNotIn("Age", rows[0])
            self.assertEqual(rows[0]["patient_id"], "tcia-breast-lesions-usg:1")


if __name__ == "__main__":
    unittest.main()
