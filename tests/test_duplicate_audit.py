import unittest

import numpy as np

from ml.audit_acquired_duplicates import (
    find_phash_neighbor_codes,
    phash_from_32x32,
    summarize_exact_groups,
)
from ml.validate_duplicate_review import evenly_spaced, validate_phash_accounting


class DuplicateAuditTests(unittest.TestCase):
    def test_phash_is_stable_for_identical_pixels(self):
        image = np.arange(32 * 32, dtype=np.float64).reshape(1, 32, 32)
        self.assertEqual(phash_from_32x32(image), phash_from_32x32(image.copy()))

    def test_index_finds_all_candidates_within_three_bits(self):
        hashes = [0, 0b111, 0b11110000]
        rows = [
            {
                "id": f"image-{index}",
                "split": split,
                "class": "benign",
                "sha256": f"sha-{index}",
                "phash": value,
            }
            for index, (value, split) in enumerate(
                zip(hashes, ("train", "test", "val"), strict=True)
            )
        ]

        neighbors, summary = find_phash_neighbor_codes(rows)

        self.assertEqual(summary["neighbor_code_pair_count"], 1)
        self.assertEqual(neighbors[0]["hamming_distance"], 3)
        self.assertEqual(
            summary["neighbor_image_pairs_by_split_pair_and_distance"]["3"],
            {"test|train": 1},
        )

    def test_exact_hashes_report_source_split_pairs(self):
        rows = [
            {
                "id": "train-0",
                "split": "train",
                "class": None,
                "sha256": "same",
            },
            {
                "id": "train-1",
                "split": "train",
                "class": None,
                "sha256": "same",
            },
            {
                "id": "test-0",
                "split": "test",
                "class": None,
                "sha256": "same",
            },
        ]

        result = summarize_exact_groups(rows)

        self.assertEqual(result["duplicate_group_count"], 1)
        self.assertEqual(result["duplicate_image_pair_count"], 3)
        self.assertEqual(
            result["duplicate_pairs_by_source_split_pair"],
            {"test|train": 2, "train|train": 1},
        )

    def test_phash_accounting_separates_code_pairs_from_image_pairs(self):
        rows = [
            {"id": "train-0", "split": "train", "sha256": "a", "phash": 0},
            {"id": "train-1", "split": "train", "sha256": "b", "phash": 0},
            {"id": "test-0", "split": "test", "sha256": "c", "phash": 3},
        ]
        dataset = {
            "rows": rows,
            "identical_phash_nonexact_collisions": {
                "nonexact_same_phash_image_pair_count": 1,
                "nonexact_same_phash_pairs_by_source_split_pair": {
                    "train|train": 1
                },
                "groups": [{"phash": "0"}],
            },
            "near_hash_code_pairs": [
                {
                    "first_phash": "0000000000000000",
                    "second_phash": "0000000000000003",
                    "hamming_distance": 3,
                    "first_member_count": 2,
                    "second_member_count": 1,
                }
            ],
            "perceptual_hash_search": {
                "neighbor_code_pair_count": 1,
                "neighbor_image_pair_counts_by_distance": {
                    "0": 0,
                    "1": 0,
                    "2": 2,
                    "3": 0,
                },
                "neighbor_image_pairs_by_split_pair_and_distance": {
                    "2": {"test|train": 2}
                },
                "unique_phash_count": 2,
                "hamming_distance_comparisons": 1,
            },
        }

        with self.assertRaisesRegex(ValueError, "Hamming distance"):
            validate_phash_accounting(dataset)

        dataset["near_hash_code_pairs"][0]["hamming_distance"] = 2
        summary = validate_phash_accounting(dataset)
        self.assertEqual(summary["same_phash_nonexact_image_pair_count"], 1)
        self.assertEqual(summary["nearby_unique_code_pair_count"], 1)
        self.assertEqual(summary["nearby_image_pair_count"], 2)

    def test_evenly_spaced_candidate_sampling_is_deterministic(self):
        candidates = [{"id": index} for index in range(10)]

        sample = evenly_spaced(candidates, 4)

        self.assertEqual(sample, [{"id": 0}, {"id": 3}, {"id": 6}, {"id": 9}])


if __name__ == "__main__":
    unittest.main()
