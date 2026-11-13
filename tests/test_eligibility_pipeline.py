"""Eligibility-pipeline unit tests; these do not measure image rejection performance."""

import unittest
import csv
import tempfile
import hashlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from ml.evaluate_eligibility import evaluate, select_threshold
from ml.train_eligibility import train
from ml.prepare_eligibility import validate_manifest


NEGATIVE_CATEGORIES = (
    "other_ultrasound",
    "other_medical",
    "natural_image",
    "object",
    "screenshot",
    "low_quality",
)


class EligibilityPipelineTests(unittest.TestCase):
    def test_evaluation_binds_report_and_rejects_midrun_manifest_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            for mutate in (False, True):
                with self.subTest(manifest_changes=mutate):
                    manifest.write_text("locked manifest", encoding="utf-8")
                    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
                    checkpoint = root / "gate.pth"
                    torch.save({
                        "manifest_sha256": digest,
                        "manifest": str(manifest.resolve()),
                        "training_source": str(root.resolve()),
                        "class_order": ["not_breast_ultrasound", "breast_ultrasound"],
                        "model_state": {},
                        "evaluation_scope": "duplicate_grouped_image_level",
                    }, checkpoint)
                    output = root / f"evaluation-{mutate}.json"

                    def score_split(_source, _manifest, _model, split):
                        if mutate and split == "test":
                            manifest.write_text("changed split", encoding="utf-8")
                        return np.asarray([0.9, 0.1]), np.asarray([1, 0]), [
                            {"path": "positive.png", "label": "breast_ultrasound", "category": "breast_ultrasound", "group_id": "p1"},
                            {"path": "negative.png", "label": "negative", "category": "object", "group_id": "n1"},
                        ]

                    with (
                        patch("ml.evaluate_eligibility.validate_manifest"),
                        patch("ml.evaluate_eligibility.build_model"),
                        patch("ml.evaluate_eligibility._score_split", side_effect=score_split),
                        patch("ml.evaluate_eligibility._bootstrap_rate", return_value=None),
                    ):
                        if mutate:
                            with self.assertRaisesRegex(ValueError, "changed during evaluation"):
                                evaluate(root, manifest, checkpoint, output, 0.01)
                            self.assertFalse(output.exists())
                        else:
                            report = evaluate(root, manifest, checkpoint, output, 0.01)
                            self.assertEqual(report["manifest_sha256"], digest)
                            self.assertFalse(report["research_inference_approved"])

    def test_evaluation_rejects_edited_or_unbound_manifest_before_scoring(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            manifest.write_text("original locked split", encoding="utf-8")
            original_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            manifest.write_text("edited labels or split assignments", encoding="utf-8")
            for digest in (original_digest, None):
                with self.subTest(checkpoint_manifest_hash=digest):
                    checkpoint = root / "gate.pth"
                    torch.save({"manifest_sha256": digest}, checkpoint)
                    output = root / "evaluation.json"
                    with (
                        patch("ml.evaluate_eligibility._score_split") as score,
                        self.assertRaisesRegex(ValueError, "unbound or changed"),
                    ):
                        evaluate(root, manifest, checkpoint, output, 0.01)
                    score.assert_not_called()
                    self.assertFalse(output.exists())

    def test_training_records_manifest_hash_and_rejects_midrun_changes(self):
        class SmallDataset(torch.utils.data.Dataset):
            rows = [{"label": "negative"}, {"label": "breast_ultrasound"}]

            def __len__(self):
                return 2

            def __getitem__(self, index):
                return torch.ones(3, 2, 2) * index, index

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            for mutate in (False, True):
                with self.subTest(manifest_changes=mutate):
                    manifest.write_text("patient_id\np1\np2\n", encoding="utf-8")
                    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
                    output = root / f"gate-{mutate}.pth"

                    def collect_scores(*_args):
                        if mutate:
                            manifest.write_text("patient_id\nchanged\n", encoding="utf-8")
                        return np.asarray([0.2, 0.8]), np.asarray([0, 1])

                    with (
                        patch("ml.train_eligibility.EligibilityDataset", return_value=SmallDataset()),
                        patch("ml.train_eligibility.build_model", side_effect=lambda **_kw: torch.nn.Sequential(
                            torch.nn.Flatten(), torch.nn.Linear(12, 2)
                        )),
                        patch("ml.train_eligibility._collect_scores", side_effect=collect_scores),
                        patch("ml.train_eligibility.torch.cuda.is_available", return_value=False),
                    ):
                        if mutate:
                            with self.assertRaisesRegex(ValueError, "changed during training"):
                                train(root, manifest, output, epochs=1, pretrained=False)
                            self.assertFalse(output.exists())
                        else:
                            report = train(root, manifest, output, epochs=1, pretrained=False)
                            checkpoint = torch.load(output, weights_only=True)
                            self.assertEqual(checkpoint["manifest_sha256"], digest)
                            self.assertEqual(report["manifest_sha256"], digest)

    def test_threshold_uses_only_the_supplied_threshold_partition(self):
        scores = np.asarray([0.95, 0.82, 0.30, 0.70])
        labels = np.asarray([1, 1, 0, 0])
        selected = select_threshold(scores, labels, max_invalid_acceptance=0.5)
        self.assertGreater(selected, 0.7)
        self.assertLessEqual(selected, 0.82)

    def test_threshold_returns_none_when_target_cannot_be_met(self):
        scores = np.asarray([1.0, 1.0])
        labels = np.asarray([1, 0])
        selected = select_threshold(scores, labels, max_invalid_acceptance=0.0)
        self.assertIsNone(selected)

    def test_threshold_rejects_an_invalid_false_acceptance_target(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            select_threshold(
                np.asarray([0.8, 0.2]),
                np.asarray([1, 0]),
                max_invalid_acceptance=1.1,
            )

    def test_manifest_rejects_missing_challenging_negative_categories(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.csv"
            columns = (
                "path", "label", "category", "group_id", "patient_id", "split",
                "sha256", "license", "source_url", "rights_verified",
            )
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=columns)
                writer.writeheader()
                writer.writerow(
                    {
                        "path": "one.png",
                        "label": "breast_ultrasound",
                        "category": "breast_ultrasound",
                        "group_id": "group-1",
                        "patient_id": "",
                        "split": "train",
                        "sha256": "hash-1",
                        "license": "CC BY 4.0",
                        "source_url": "https://example.org/dataset",
                        "rights_verified": "true",
                    }
                )
            with self.assertRaisesRegex(ValueError, "challenging negative categories"):
                validate_manifest(path)

    def test_manifest_rejects_groups_crossing_partitions(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.csv"
            columns = (
                "path", "label", "category", "group_id", "patient_id", "split",
                "sha256", "license", "source_url", "rights_verified",
            )
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=columns)
                writer.writeheader()
                for split in ("train", "validation", "threshold", "test"):
                    writer.writerow(
                        {
                            "path": f"{split}/positive.png",
                            "label": "breast_ultrasound",
                            "category": "breast_ultrasound",
                            "group_id": f"{split}-positive",
                            "patient_id": "",
                            "split": split,
                            "sha256": f"{split}-positive",
                            "license": "CC BY 4.0",
                            "source_url": "https://example.org/dataset",
                            "rights_verified": "true",
                        }
                    )
                    for category in NEGATIVE_CATEGORIES:
                        writer.writerow(
                            {
                                "path": f"{split}/{category}.png",
                                "label": "negative",
                                "category": category,
                                "group_id": (
                                    "cross-split-group"
                                    if split in {"train", "test"} and category == "object"
                                    else f"{split}-{category}"
                                ),
                                "patient_id": "",
                                "split": split,
                                "sha256": f"{split}-{category}",
                                "license": "CC BY 4.0",
                                "source_url": "https://example.org/dataset",
                                "rights_verified": "true",
                            }
                        )
            with self.assertRaisesRegex(ValueError, "cross eligibility partitions"):
                validate_manifest(path)


if __name__ == "__main__":
    unittest.main()
