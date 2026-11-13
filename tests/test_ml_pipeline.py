"""Data-preparation and preprocessing regression tests using generated test fixtures."""

import csv
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from ml.model import CLASS_NAMES, IMAGE_SIZE, load_checkpoint
from ml.prepare_dataset import prepare_dataset
from ml.prepare_mendeley_development import prepare_development
from ml.train import ManifestDataset, fit_temperature, validation_selection_key
from ml.evaluate import ensure_no_external_overlap
from ml.inspect_chestmnist import inspect_archive
import torch


def write_fixture(path: Path, label_index: int, variation: int) -> None:
    random = np.random.default_rng(label_index * 1000 + variation)
    pixels = random.integers(0, 256, size=(128, 128), dtype=np.uint8)
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    x = 8 + variation * 11
    draw.ellipse((x, 18 + label_index * 7, min(x + 24, 126), 54 + label_index * 7), fill=230)
    image.save(path)


class MlPipelineTests(unittest.TestCase):
    def test_chestmnist_inventory_checks_archive_and_preserves_source_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "chestmnist.npz"
            arrays = {}
            for split in ("train", "val", "test"):
                arrays[f"{split}_images"] = np.zeros((2, 64, 64), dtype=np.uint8)
                arrays[f"{split}_labels"] = np.zeros((2, 14), dtype=np.uint8)
            np.savez(archive_path, **arrays)
            checksum = hashlib.md5(archive_path.read_bytes()).hexdigest()

            report = inspect_archive(archive_path, expected_md5=checksum)

            self.assertEqual(report["splits"]["train"]["image_count"], 2)
            self.assertEqual(report["splits"]["val"]["label_column_count"], 14)
            self.assertFalse(report["rights_verified_for_training"])
            with self.assertRaisesRegex(ValueError, "MD5 mismatch"):
                inspect_archive(archive_path, expected_md5="0" * 32)

    def create_dataset(self, root: Path, patients: bool = True) -> Path:
        mapping_rows = []
        for label_index, label in enumerate(CLASS_NAMES):
            folder = root / label
            folder.mkdir(parents=True, exist_ok=True)
            for index in range(10):
                name = f"{label}_{index:02}.png"
                write_fixture(folder / name, label_index, index)
                mapping_rows.append((f"{label}/{name}", f"patient-{label}-{index}"))
        metadata = root.parent / "patients.csv"
        if patients:
            with metadata.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(("image_path", "patient_id"))
                writer.writerows(mapping_rows)
        return metadata if patients else None

    def test_patient_groups_are_disjoint_across_deterministic_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            root.mkdir()
            metadata = self.create_dataset(root)
            output = Path(temporary) / "prepared"
            report = prepare_dataset(root, output, metadata_csv=metadata, seed=17)
            self.assertEqual(report["evaluation_scope"], "patient_grouped_internal_holdout")
            with (output / "split_manifest.csv").open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            groups_by_split = {
                split: {row["group_id"] for row in rows if row["split"] == split}
                for split in ("train", "validation", "test")
            }
            self.assertFalse(groups_by_split["train"] & groups_by_split["validation"])
            self.assertFalse(groups_by_split["train"] & groups_by_split["test"])
            self.assertFalse(groups_by_split["validation"] & groups_by_split["test"])
            self.assertEqual(len(rows), 30)
            with self.assertRaises(FileExistsError):
                prepare_dataset(root, output, metadata_csv=metadata, seed=17)

    def test_image_level_split_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            root.mkdir()
            self.create_dataset(root, patients=False)
            with self.assertRaisesRegex(ValueError, "allow-image-level-split"):
                prepare_dataset(root, Path(temporary) / "prepared")
            report = prepare_dataset(
                root,
                Path(temporary) / "prepared-ok",
                allow_image_level_split=True,
                seed=23,
            )
            self.assertEqual(report["evaluation_scope"], "image_level_internal_holdout")
            self.assertFalse(report["patient_independent"])

    def test_development_split_excludes_old_test_and_near_duplicate_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            root.mkdir()
            for label_index, label in enumerate(CLASS_NAMES[:2]):
                folder = root / label
                folder.mkdir()
                for index in range(8):
                    write_fixture(folder / f"{label}_{index:02}.png", label_index, index)
            previous = Path(temporary) / "previous"
            prepare_dataset(root, previous, allow_image_level_split=True, seed=3)
            with (previous / "split_manifest.csv").open(
                newline="", encoding="utf-8"
            ) as file:
                previous_rows = list(csv.DictReader(file))
            old_test_paths = {row["path"] for row in previous_rows if row["split"] == "test"}
            output = Path(temporary) / "development"
            report = prepare_development(
                root,
                previous / "split_manifest.csv",
                output,
                seed=3,
                near_duplicate_distance=0,
            )
            with (output / "split_manifest.csv").open(
                newline="", encoding="utf-8"
            ) as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(report["previous_test_images_excluded"], len(old_test_paths))
            self.assertTrue(
                all(
                    row["split"] == "excluded_previous_test"
                    for row in rows
                    if row["path"] in old_test_paths
                )
            )
            self.assertFalse(
                any(
                    row["split"] in {"train", "validation"} and row["path"] in old_test_paths
                    for row in rows
                )
            )

    def test_exact_duplicates_are_kept_in_the_same_group_and_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            root.mkdir()
            self.create_dataset(root, patients=False)
            duplicate = root / "benign" / "benign_duplicate.png"
            duplicate.write_bytes((root / "benign" / "benign_00.png").read_bytes())
            output = Path(temporary) / "prepared"
            prepare_dataset(
                root, output, allow_image_level_split=True, seed=42
            )
            with (output / "split_manifest.csv").open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            originals = {
                row["path"]: row
                for row in rows
                if row["path"] in {"benign/benign_00.png", "benign/benign_duplicate.png"}
            }
            self.assertEqual(len(originals), 2)
            self.assertEqual(
                originals["benign/benign_00.png"]["group_id"],
                originals["benign/benign_duplicate.png"]["group_id"],
            )
            self.assertEqual(
                originals["benign/benign_00.png"]["split"],
                originals["benign/benign_duplicate.png"]["split"],
            )

    def test_conflicting_labels_for_a_duplicate_group_are_blocked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            root.mkdir()
            self.create_dataset(root, patients=False)
            duplicate = root / "malignant" / "malignant_duplicate.png"
            duplicate.write_bytes((root / "benign" / "benign_00.png").read_bytes())
            with self.assertRaisesRegex(ValueError, "conflicting class labels"):
                prepare_dataset(
                    root,
                    Path(temporary) / "prepared",
                    allow_image_level_split=True,
                )

    def test_multiple_images_for_one_patient_remain_grouped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            root.mkdir()
            metadata = self.create_dataset(root)
            with metadata.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            rows[1]["patient_id"] = rows[0]["patient_id"]
            with metadata.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=("image_path", "patient_id"))
                writer.writeheader()
                writer.writerows(rows)
            output = Path(temporary) / "prepared"
            prepare_dataset(root, output, metadata_csv=metadata)
            with (output / "split_manifest.csv").open(newline="", encoding="utf-8") as file:
                manifest_rows = list(csv.DictReader(file))
            first = next(row for row in manifest_rows if row["path"] == "benign/benign_00.png")
            second = next(row for row in manifest_rows if row["path"] == "benign/benign_01.png")
            self.assertEqual(first["group_id"], second["group_id"])
            self.assertEqual(first["split"], second["split"])

    def test_two_class_source_preserves_its_supported_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            for label_index, label in enumerate(CLASS_NAMES[:2]):
                folder = root / label
                folder.mkdir(parents=True)
                for index in range(8):
                    write_fixture(folder / f"{label}_{index:02}.png", label_index, index)
            report = prepare_dataset(
                root,
                Path(temporary) / "prepared",
                allow_image_level_split=True,
                seed=12,
            )
            self.assertEqual(report["class_order"], ["benign", "malignant"])
            self.assertNotIn("normal", report["split_counts"]["train"])

    def test_bmp_images_are_included(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            root.mkdir()
            self.create_dataset(root, patients=False)
            source_image = root / "benign" / "benign_00.png"
            with Image.open(source_image) as image:
                image.convert("RGB").save(root / "benign" / "benign_bmp.bmp")
            output = Path(temporary) / "prepared"
            prepare_dataset(
                root,
                output,
                allow_image_level_split=True,
                seed=23,
            )
            with (output / "split_manifest.csv").open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertIn("benign/benign_bmp.bmp", {row["path"] for row in rows})

    def test_external_dataset_duplicate_is_blocked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training = root / "training.csv"
            external = root / "external.csv"
            columns = ("sha256", "dhash")
            for path, digest in ((training, "same"), (external, "same")):
                with path.open("w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=columns)
                    writer.writeheader()
                    writer.writerow({"sha256": digest, "dhash": "1"})
            with self.assertRaisesRegex(ValueError, "overlaps"):
                ensure_no_external_overlap(training, external)

    def test_train_augmentation_is_separate_from_evaluation_preprocessing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            root.mkdir()
            self.create_dataset(root, patients=False)
            output = Path(temporary) / "prepared"
            prepare_dataset(root, output, allow_image_level_split=True)
            training = ManifestDataset(root, output / "split_manifest.csv", "train")
            validation = ManifestDataset(root, output / "split_manifest.csv", "validation")
            self.assertEqual(training[0][0].shape[-2:], (IMAGE_SIZE, IMAGE_SIZE))
            self.assertEqual(validation[0][0].shape[-2:], (IMAGE_SIZE, IMAGE_SIZE))
            self.assertEqual(training.transform.transforms[1].__class__.__name__, "RandomHorizontalFlip")
            self.assertEqual(validation.transform.transforms[1].__class__.__name__, "ToTensor")
            self.assertEqual(training.transform.transforms[-1].mean, (0.485, 0.456, 0.406))
            self.assertEqual(validation.transform.transforms[-1].std, (0.229, 0.224, 0.225))
            self.assertEqual(CLASS_NAMES, ("benign", "malignant", "normal"))

    def test_training_ablation_can_disable_augmentation_without_changing_normalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            root.mkdir()
            self.create_dataset(root, patients=False)
            output = Path(temporary) / "prepared"
            prepare_dataset(root, output, allow_image_level_split=True)
            training = ManifestDataset(
                root, output / "split_manifest.csv", "train", augment=False
            )
            validation = ManifestDataset(root, output / "split_manifest.csv", "validation")
            self.assertEqual(
                [type(transform).__name__ for transform in training.transform.transforms],
                [type(transform).__name__ for transform in validation.transform.transforms],
            )

    def test_checkpoint_with_mismatched_preprocessing_is_rejected(self):
        checkpoint = {
            "class_order": ["benign", "malignant"],
            "image_size": IMAGE_SIZE,
            "normalization": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
        }
        with patch("torch.load", return_value=checkpoint):
            with self.assertRaisesRegex(ValueError, "normalization"):
                load_checkpoint("unused.pth", torch.device("cpu"))

    def test_validation_selection_policy_exposes_sensitivity_tradeoff(self):
        recall_first = validation_selection_key(1.0, 0.79, 0.36, "malignant_recall_then_macro_f1")
        macro_first = validation_selection_key(0.87, 0.92, 0.24, "macro_f1_then_malignant_recall")
        self.assertGreater(recall_first[0], recall_first[1])
        self.assertEqual(macro_first, (0.92, 0.87, -0.24))

    def test_temperature_is_positive_and_finite(self):
        logits = torch.tensor([[2.0, 0.1, -0.5], [0.2, 1.5, 0.1], [0.0, 0.2, 2.0]])
        labels = torch.tensor([0, 1, 2])
        temperature = fit_temperature(logits, labels)
        self.assertTrue(torch.isfinite(torch.tensor(temperature)).item())
        self.assertGreater(temperature, 0)


if __name__ == "__main__":
    unittest.main()
