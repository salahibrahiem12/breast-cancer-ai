"""Statistical and partition safeguards; no medical performance is inferred."""

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ml.audit_tcia_evaluation import binary_metrics, group_intervals
from ml.development_baseline import development_rows, features
from ml.evaluate import calibration_metrics, prediction_indices


class AuditDevelopmentTests(unittest.TestCase):
    def test_binary_ties_follow_declared_threshold_in_either_class_order(self):
        probabilities = np.asarray([[.5, .5], [.50001, .49999], [.1, .9]])
        self.assertEqual(prediction_indices(probabilities, ("benign", "malignant")).tolist(), [1, 0, 1])
        self.assertEqual(prediction_indices(probabilities, ("malignant", "benign")).tolist(), [0, 0, 1])

    def test_calibration_uses_the_same_binary_tie_decision(self):
        result = calibration_metrics(np.asarray([1, 1]), np.asarray([[.5, .5], [.49, .51]]), ("benign", "malignant"))
        self.assertAlmostEqual(result["top_label_ece_10_bins"], .495)
    def test_metrics_match_known_complete_confusion_matrix(self):
        labels = np.asarray([0] * 99 + [1] * 98)
        scores = np.asarray([.2] * 6 + [.8] * 93 + [.3] * 9 + [.9] * 89)
        result = binary_metrics(labels, scores)
        self.assertEqual([result[key] for key in ("tn", "fp", "fn", "tp")], [6, 93, 9, 89])
        self.assertAlmostEqual(result["sensitivity"], 89 / 98)
        self.assertAlmostEqual(result["specificity"], 6 / 99)
        self.assertAlmostEqual(result["precision_ppv"], 89 / 182)
        self.assertAlmostEqual(result["npv"], .4)
        self.assertAlmostEqual(result["balanced_accuracy"], (89 / 98 + 6 / 99) / 2)

    def test_undefined_metrics_are_unavailable_not_zero(self):
        result = binary_metrics([0, 0], [.8, .9])
        self.assertIsNone(result["sensitivity"])
        self.assertIsNone(result["npv"])
        self.assertIsNone(result["roc_auc"])
        self.assertIsNone(result["pr_auc_average_precision"])

    def test_metrics_reject_invalid_inputs(self):
        for scores in ([float("nan"), .5], [1.01, .5], [.2]):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                binary_metrics([0, 1], scores)

    def test_cluster_bootstrap_does_not_treat_repeated_images_as_new_patients(self):
        labels, scores, groups = np.array([0, 0, 1, 1]), np.array([.1, .7, .8, .9]), np.array(["a", "b", "c", "d"])
        first = group_intervals(labels, scores, groups, iterations=50)
        repeated = group_intervals(np.repeat(labels, 2), np.repeat(scores, 2), np.repeat(groups, 2), iterations=50)
        for key in first:
            for bound in ("lower_95", "upper_95"):
                self.assertAlmostEqual(first[key][bound], repeated[key][bound])

    def test_development_rejects_cross_split_groups_before_loading_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "manifest.csv"
            with path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["path", "label", "split", "group_id", "sha256", "dhash"])
                writer.writeheader()
                for split in ("train", "validation"):
                    writer.writerow({"path": split + ".png", "label": "benign", "split": split,
                                     "group_id": "shared-patient", "sha256": split, "dhash": "0"})
            with self.assertRaisesRegex(ValueError, "groups cross"):
                development_rows(root, path)

    def test_feature_candidate_preserves_expected_size_and_finite_values(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.png"
            Image.fromarray(np.arange(48 * 96, dtype=np.uint8).reshape(48, 96)).save(path)
            for candidate in ("baseline", "aspect_contrast"):
                values = features(path, candidate)
                self.assertEqual(values.shape, (1024,))
                self.assertTrue(np.isfinite(values).all())
                self.assertGreaterEqual(values.min(), 0)
                self.assertLessEqual(values.max(), 1)


if __name__ == "__main__":
    unittest.main()
