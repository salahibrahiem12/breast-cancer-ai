"""Audit exact and perceptual duplicates in the acquired research archives."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import PIL
from PIL import Image


EXPECTED_ARCHIVES = {
    "mendeley": {
        "filename": "us-dataset.zip",
        "bytes": 3_264_338,
        "sha256": "87dbd3aaa89e2d8886839983be3a985b7b190d5d8cb167d95a8dcd5d224b63f7",
    },
    "chestmnist": {
        "filename": "chestmnist_64.npz",
        "bytes": 401_604_127,
        "sha256": "e28fbed21400c50040d6a4e51f379dce89bcb24819d3878c252ca78d52d92c3e",
    },
}
PHASH_DISTANCE = 3
PHASH_SIZE = 32
PHASH_LOW_FREQUENCY_SIZE = 8
PILLOW_VERSION = PIL.__version__


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phash_basis() -> np.ndarray:
    positions = np.arange(PHASH_SIZE, dtype=np.float64)
    frequencies = np.arange(PHASH_LOW_FREQUENCY_SIZE, dtype=np.float64)[:, None]
    basis = np.cos(np.pi * (2 * positions + 1) * frequencies / (2 * PHASH_SIZE))
    basis *= np.sqrt(2 / PHASH_SIZE)
    basis[0, :] = np.sqrt(1 / PHASH_SIZE)
    return basis


DCT_BASIS = phash_basis()


def phash_from_32x32(images: np.ndarray) -> list[int]:
    """Calculate 64-bit DCT perceptual hashes from grayscale 32x32 images."""
    coefficients = np.matmul(np.matmul(DCT_BASIS, images), DCT_BASIS.T)
    flat = coefficients.reshape(len(images), -1)
    medians = np.median(flat, axis=1)
    bits = flat > medians[:, None]
    packed = np.packbits(bits, axis=1, bitorder="big")
    return [int.from_bytes(row.tobytes(), byteorder="big") for row in packed]


def phash_mendeley(path: Path) -> int:
    with Image.open(path) as image:
        gray = image.convert("L")
        resized = gray.resize(
            (PHASH_SIZE, PHASH_SIZE),
            resample=Image.Resampling.BOX,
        )
        values = np.asarray(resized, dtype=np.float64)[None, :, :]
    return phash_from_32x32(values)[0]


def phash_chest_batch(images: np.ndarray) -> list[int]:
    # Each source image is 64x64, so averaging each 2x2 cell is exact area
    # resampling to 32x32 and matches the BOX-resampling policy above.
    reduced = images.reshape(len(images), 32, 2, 32, 2).mean(axis=(2, 4))
    return phash_from_32x32(reduced)


def pair_key(first: str, second: str) -> str:
    return "|".join(sorted((first, second)))


def split_pair_counts(items: list[dict[str, object]]) -> Counter[str]:
    split_counts = Counter(str(item.get("split", "")) for item in items)
    result: Counter[str] = Counter()
    split_names = sorted(split_counts)
    for index, first in enumerate(split_names):
        for second in split_names[index:]:
            if first == second:
                count = split_counts[first] * (split_counts[first] - 1) // 2
            else:
                count = split_counts[first] * split_counts[second]
            if count:
                result[pair_key(first, second)] += count
    return result


def cross_histogram_pair_counts(
    first_counts: Counter[str],
    second_counts: Counter[str],
    *,
    within_one_group: bool,
) -> Counter[str]:
    result: Counter[str] = Counter()
    first_names = sorted(first_counts)
    second_names = sorted(second_counts)
    for first_index, first in enumerate(first_names):
        for second in second_names:
            if within_one_group and second < first:
                continue
            if within_one_group and first == second:
                count = first_counts[first] * (first_counts[first] - 1) // 2
            else:
                count = first_counts[first] * second_counts[second]
            if count:
                result[pair_key(first, second)] += count
    return result


def summarize_exact_groups(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["sha256"])].append(index)

    groups: list[dict[str, object]] = []
    split_pairs: Counter[str] = Counter()
    class_pairs: Counter[str] = Counter()
    for digest, indexes in grouped.items():
        if len(indexes) < 2:
            continue
        members = [rows[index] for index in indexes]
        split_pairs.update(split_pair_counts(members))
        classes = Counter(str(member.get("class", "")) for member in members)
        class_names = sorted(classes)
        for first_pos, first in enumerate(class_names):
            for second in class_names[first_pos:]:
                if first == second:
                    count = classes[first] * (classes[first] - 1) // 2
                else:
                    count = classes[first] * classes[second]
                class_pairs[pair_key(first, second)] += count
        groups.append(
            {
                "sha256": digest,
                "members": [
                    {
                        "id": member["id"],
                        "split": member.get("split"),
                        "class": member.get("class"),
                    }
                    for member in members
                ],
            }
        )

    groups.sort(key=lambda group: group["members"][0]["id"])
    return {
        "unique_sha256_count": len(grouped),
        "duplicate_group_count": len(groups),
        "duplicate_excess_images": sum(len(group["members"]) - 1 for group in groups),
        "duplicate_image_pair_count": sum(
            len(group["members"]) * (len(group["members"]) - 1) // 2
            for group in groups
        ),
        "duplicate_pairs_by_source_split_pair": dict(sorted(split_pairs.items())),
        "duplicate_pairs_by_class_pair": dict(sorted(class_pairs.items())),
        "groups": groups,
    }


def summarize_identical_phash_codes(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["phash"])].append(row)

    collision_groups: list[dict[str, object]] = []
    nonexact_pairs_by_split: Counter[str] = Counter()
    nonexact_pairs_by_class: Counter[str] = Counter()
    total_nonexact_pairs = 0

    for code, members in grouped.items():
        sha_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
        for member in members:
            sha_groups[str(member["sha256"])].append(member)
        if len(sha_groups) < 2:
            continue

        all_split_pairs = split_pair_counts(members)
        all_class_pairs = cross_histogram_pair_counts(
            Counter(str(member.get("class") or "") for member in members),
            Counter(str(member.get("class") or "") for member in members),
            within_one_group=True,
        )
        exact_split_pairs: Counter[str] = Counter()
        exact_class_pairs: Counter[str] = Counter()
        for exact_members in sha_groups.values():
            exact_split_pairs.update(split_pair_counts(exact_members))
            exact_class_pairs.update(
                cross_histogram_pair_counts(
                    Counter(str(member.get("class") or "") for member in exact_members),
                    Counter(str(member.get("class") or "") for member in exact_members),
                    within_one_group=True,
                )
            )

        split_counts = Counter(
            {
                pair: count - exact_split_pairs[pair]
                for pair, count in all_split_pairs.items()
                if count - exact_split_pairs[pair]
            }
        )
        class_counts = Counter(
            {
                pair: count - exact_class_pairs[pair]
                for pair, count in all_class_pairs.items()
                if count - exact_class_pairs[pair]
            }
        )
        group_pair_count = sum(split_counts.values()) if split_counts else (
            sum(class_counts.values()) if class_counts else 0
        )
        total_nonexact_pairs += group_pair_count
        nonexact_pairs_by_split.update(split_counts)
        nonexact_pairs_by_class.update(class_counts)
        if group_pair_count:
            collision_groups.append(
                {
                    "phash": f"{code:016x}",
                    "member_count": len(members),
                    "distinct_sha256_count": len(sha_groups),
                    "nonexact_image_pair_count": group_pair_count,
                    "nonexact_pairs_by_source_split_pair": dict(sorted(split_counts.items())),
                    "nonexact_pairs_by_class_pair": dict(sorted(class_counts.items())),
                    "sample_members": [
                        {
                            "id": member["id"],
                            "split": member.get("split"),
                            "class": member.get("class"),
                            "sha256": member["sha256"],
                        }
                        for member in members[:20]
                    ],
                    "sample_members_truncated": len(members) > 20,
                }
            )

    collision_groups.sort(
        key=lambda group: (-group["member_count"], group["phash"])
    )
    return {
        "nonexact_same_phash_image_pair_count": total_nonexact_pairs,
        "nonexact_same_phash_pairs_by_source_split_pair": dict(
            sorted(nonexact_pairs_by_split.items())
        ),
        "nonexact_same_phash_pairs_by_class_pair": dict(
            sorted(nonexact_pairs_by_class.items())
        ),
        "groups": collision_groups,
    }


def find_phash_neighbor_codes(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Find all unique pHash code pairs within Hamming distance 3.

    Five disjoint 12/13-bit bands are used. Any pair at distance <=3 must
    have at least two unchanged bands, so this is a complete candidate index
    for the stated threshold. Pairwise band signatures avoid the large
    candidate buckets created by single-band matching.
    """
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[int(row["phash"])].append(index)

    codes = sorted(grouped)
    widths = (13, 13, 13, 13, 12)
    starts = [0]
    for width in widths[:-1]:
        starts.append(starts[-1] + width)
    band_pairs = list(
        (first, second)
        for first in range(len(widths))
        for second in range(first + 1, len(widths))
    )
    buckets: list[dict[int, list[int]]] = [
        defaultdict(list) for _ in band_pairs
    ]
    neighbors: list[dict[str, object]] = []
    candidate_pairs = 0
    distance_checks = 0
    image_pairs_by_distance: Counter[int] = Counter()
    split_pair_counts_by_distance: dict[int, Counter[str]] = defaultdict(Counter)
    class_pair_counts_by_distance: dict[int, Counter[str]] = defaultdict(Counter)

    def band_signature(code: int, first_band: int, second_band: int) -> int:
        first_width = widths[first_band]
        second_width = widths[second_band]
        first_mask = (1 << first_width) - 1
        second_mask = (1 << second_width) - 1
        first_value = (code >> starts[first_band]) & first_mask
        second_value = (code >> starts[second_band]) & second_mask
        return (first_value << second_width) | second_value

    for current_index, code in enumerate(codes):
        candidates: set[int] = set()
        for bucket, (first_band, second_band) in zip(buckets, band_pairs, strict=True):
            signature = band_signature(code, first_band, second_band)
            candidates.update(bucket.get(signature, ()))
        candidate_pairs += len(candidates)

        for prior_index in sorted(candidates):
            prior_code = codes[prior_index]
            distance_checks += 1
            distance = (code ^ prior_code).bit_count()
            if distance > PHASH_DISTANCE:
                continue

            first_items = grouped[prior_code]
            second_items = grouped[code]
            first_splits = Counter(str(rows[index].get("split") or "") for index in first_items)
            second_splits = Counter(str(rows[index].get("split") or "") for index in second_items)
            first_classes = Counter(str(rows[index].get("class") or "") for index in first_items)
            second_classes = Counter(str(rows[index].get("class") or "") for index in second_items)
            split_pair_counts_by_distance[distance].update(
                cross_histogram_pair_counts(
                    first_splits,
                    second_splits,
                    within_one_group=False,
                )
            )
            class_pair_counts_by_distance[distance].update(
                cross_histogram_pair_counts(
                    first_classes,
                    second_classes,
                    within_one_group=False,
                )
            )
            image_pairs_by_distance[distance] += len(first_items) * len(second_items)
            neighbors.append(
                {
                    "hamming_distance": distance,
                    "first_phash": f"{prior_code:016x}",
                    "second_phash": f"{code:016x}",
                    "first_member_count": len(first_items),
                    "second_member_count": len(second_items),
                }
            )

        for bucket, (first_band, second_band) in zip(buckets, band_pairs, strict=True):
            signature = band_signature(code, first_band, second_band)
            bucket[signature].append(current_index)
        if current_index and current_index % 20_000 == 0:
            print(
                f"pHash index: {current_index}/{len(codes)} unique hashes",
                flush=True,
            )

    return neighbors, {
        "unique_phash_count": len(codes),
        "index_scheme": {
            "band_widths_low_to_high": list(widths),
            "indexed_band_pairs": len(band_pairs),
            "completeness_argument": (
                "At most three changed bits can affect at most three of five "
                "disjoint bands, leaving at least two identical bands."
            ),
        },
        "indexed_candidate_code_pair_count": candidate_pairs,
        "hamming_distance_comparisons": distance_checks,
        "neighbor_code_pair_count": len(neighbors),
        "neighbor_image_pair_counts_by_distance": {
            str(distance): image_pairs_by_distance[distance]
            for distance in range(PHASH_DISTANCE + 1)
        },
        "neighbor_image_pairs_by_split_pair_and_distance": {
            str(distance): dict(sorted(counts.items()))
            for distance, counts in split_pair_counts_by_distance.items()
        },
        "neighbor_image_pairs_by_class_pair_and_distance": {
            str(distance): dict(sorted(counts.items()))
            for distance, counts in class_pair_counts_by_distance.items()
        },
    }


def validate_mendeley(
    dataset_dir: Path, inventory: dict[str, object]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    archive_path = dataset_dir / "us-dataset.zip"
    archive_spec = EXPECTED_ARCHIVES["mendeley"]
    actual_archive_sha = sha256_file(archive_path)
    if archive_path.stat().st_size != archive_spec["bytes"]:
        raise ValueError("Mendeley archive size differs from the recorded manifest.")
    if actual_archive_sha != archive_spec["sha256"]:
        raise ValueError("Mendeley archive SHA-256 differs from the recorded manifest.")

    expected: dict[str, dict[str, object]] = {
        str(row["path"]).replace("\\", "/"): row
        for row in inventory["inventory"]["all_files"]
    }
    extracted_root = dataset_dir / "extracted"
    image_rows: list[dict[str, object]] = []
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        bad_member = archive.testzip()
        archive_files = {
            member.filename.replace("\\", "/"): member
            for member in members
            if not member.is_dir()
        }
        if len(members) != inventory["inventory"]["archive_entries"]:
            raise ValueError("Mendeley ZIP entry count differs from inventory.")
        expected_paths = set(expected)
        image_paths = {
            name for name in archive_files if name.lower().endswith(".bmp")
        }
        if image_paths != expected_paths:
            raise ValueError("Mendeley archive image paths differ from inventory.")
        if bad_member is not None:
            raise ValueError(f"Mendeley ZIP CRC validation failed: {bad_member}")

        for relative_path, record in expected.items():
            path = extracted_root / Path(relative_path)
            if not path.is_file():
                raise ValueError(f"Missing extracted Mendeley image: {relative_path}")
            raw = path.read_bytes()
            if raw != archive.read(archive_files[relative_path]):
                raise ValueError(f"Extracted bytes differ from archive: {relative_path}")
            if path.stat().st_size == 0:
                raise ValueError(f"Empty extracted Mendeley image: {relative_path}")
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
                if image.format != record["format"]:
                    raise ValueError(f"Image format mismatch: {relative_path}")
                if [image.width, image.height] != [
                    record["width"],
                    record["height"],
                ]:
                    raise ValueError(f"Image dimensions mismatch: {relative_path}")
                phash = phash_mendeley(path)
            image_rows.append(
                {
                    "id": relative_path,
                    "class": record["class_from_source_path"],
                    "split": None,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "phash": phash,
                }
            )

    disk_images = [
        path
        for path in extracted_root.rglob("*")
        if path.is_file()
    ]
    if len(disk_images) != len(expected):
        raise ValueError("Mendeley extracted file count differs from inventory.")
    return image_rows, {
        "archive_filename": archive_path.name,
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": actual_archive_sha,
        "zip_entries_including_directories": len(members),
        "zip_crc_check": "passed",
        "inventory_image_count": len(expected),
        "archive_image_count": len(image_paths),
        "extracted_file_count": len(disk_images),
        "archive_and_extraction_bytes_match": True,
        "decoded_format_and_dimensions_match_inventory": True,
    }


def validate_chestmnist(
    dataset_dir: Path, inventory: dict[str, object], batch_size: int
) -> tuple[list[dict[str, object]], dict[str, object]]:
    archive_path = dataset_dir / "chestmnist_64.npz"
    archive_spec = EXPECTED_ARCHIVES["chestmnist"]
    actual_archive_sha = sha256_file(archive_path)
    if archive_path.stat().st_size != archive_spec["bytes"]:
        raise ValueError("ChestMNIST archive size differs from the recorded manifest.")
    if actual_archive_sha != archive_spec["sha256"]:
        raise ValueError("ChestMNIST archive SHA-256 differs from the recorded manifest.")

    image_rows: list[dict[str, object]] = []
    split_shapes: dict[str, object] = {}
    with np.load(archive_path, allow_pickle=False) as archive:
        if sorted(archive.files) != sorted(inventory["inventory"]["internal_arrays"]):
            raise ValueError("ChestMNIST NPZ members differ from inventory.")
        for split in ("train", "val", "test"):
            images = archive[f"{split}_images"]
            labels = archive[f"{split}_labels"]
            spec = inventory["inventory"]["splits_as_published"][split]
            if list(images.shape) != spec["image_tensor_shape"]:
                raise ValueError(f"ChestMNIST {split} image shape differs from inventory.")
            if list(labels.shape) != spec["labels_shape"]:
                raise ValueError(f"ChestMNIST {split} label shape differs from inventory.")
            if str(images.dtype) != spec["image_dtype"]:
                raise ValueError(f"ChestMNIST {split} image dtype differs from inventory.")
            if str(labels.dtype) != spec["label_dtype"]:
                raise ValueError(f"ChestMNIST {split} label dtype differs from inventory.")
            if images.ndim != 3 or images.shape[1:] != (64, 64):
                raise ValueError(f"ChestMNIST {split} image tensor is not N x 64 x 64.")
            if labels.ndim != 2 or labels.shape[1] != 14:
                raise ValueError(f"ChestMNIST {split} labels are not N x 14.")

            split_shapes[split] = {
                "image_count": int(images.shape[0]),
                "image_shape": list(images.shape),
                "label_shape": list(labels.shape),
            }
            for start in range(0, len(images), batch_size):
                stop = min(start + batch_size, len(images))
                batch = np.ascontiguousarray(images[start:stop])
                hashes = phash_chest_batch(batch)
                for offset, image in enumerate(batch):
                    image_bytes = image.tobytes()
                    image_rows.append(
                        {
                            "id": f"{split}[{start + offset}]",
                            "class": None,
                            "split": split,
                            "sha256": hashlib.sha256(image_bytes).hexdigest(),
                            "phash": hashes[offset],
                        }
                    )

    return image_rows, {
        "archive_filename": archive_path.name,
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": actual_archive_sha,
        "npz_members": sorted(inventory["inventory"]["internal_arrays"]),
        "source_splits_preserved": split_shapes,
        "total_decoded_images": len(image_rows),
        "array_shapes_and_dtypes_match_inventory": True,
    }


def write_json_atomic(path: Path, content: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as temporary:
        json.dump(content, temporary, indent=2)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--research-data",
        type=Path,
        default=Path("data/research_data/limited_acquisition_2026-11-01"),
    )
    parser.add_argument(
        "--mendeley-inventory",
        type=Path,
        default=Path(
            "data/research_data/limited_acquisition_2026-11-01/"
            "Mendeley_Breast_Ultrasound_Image_v1/inventory.json"
        ),
    )
    parser.add_argument(
        "--chestmnist-inventory",
        type=Path,
        default=Path(
            "data/research_data/limited_acquisition_2026-11-01/"
            "MedMNIST_Plus_ChestMNIST_v3.0/inventory.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/research_data/limited_acquisition_2026-11-01/"
            "duplicate_audit.json"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")

    mendeley_inventory = json.loads(
        args.mendeley_inventory.read_text(encoding="utf-8")
    )
    chest_inventory = json.loads(
        args.chestmnist_inventory.read_text(encoding="utf-8")
    )
    research_root = args.research_data
    mendeley_rows, mendeley_validation = validate_mendeley(
        research_root / "Mendeley_Breast_Ultrasound_Image_v1",
        mendeley_inventory,
    )
    chest_rows, chest_validation = validate_chestmnist(
        research_root / "MedMNIST_Plus_ChestMNIST_v3.0",
        chest_inventory,
        args.batch_size,
    )

    mendeley_exact = summarize_exact_groups(mendeley_rows)
    chest_exact = summarize_exact_groups(chest_rows)
    mendeley_phash_collisions = summarize_identical_phash_codes(mendeley_rows)
    chest_phash_collisions = summarize_identical_phash_codes(chest_rows)
    print("Validated both datasets and computed exact duplicate groups.", flush=True)
    mendeley_neighbors, mendeley_search = find_phash_neighbor_codes(mendeley_rows)
    print("Completed Mendeley perceptual-hash index.", flush=True)
    chest_neighbors, chest_search = find_phash_neighbor_codes(chest_rows)
    print("Completed ChestMNIST perceptual-hash index.", flush=True)
    result: dict[str, object] = {
        "audit_date": "2026-11-01",
        "audit_scope": "Local acquired files only; no training, evaluation, threshold tuning, or partition creation.",
        "method": {
            "exact_duplicate": (
                "SHA-256 of original file bytes for Mendeley BMP files; SHA-256 "
                "of each contiguous uint8 64x64 pixel array for ChestMNIST."
            ),
            "perceptual_hash": (
                "64-bit DCT pHash. Convert to grayscale, resize to 32x32 using "
                "area/BOX resampling, take the top-left 8x8 coefficients of the "
                "orthonormal 2D DCT, and threshold each coefficient against the "
                "median of all 64 low-frequency coefficients. A near-match is "
                "a direct pHash Hamming distance <= 3; this is a screening "
                "threshold, not proof of duplicate images or patient identity."
            ),
            "candidate_index": (
                "For threshold 3, split the 64-bit hash into five disjoint "
                "12/13-bit bands and index all ten two-band signatures. "
                "Every pair within Hamming distance 3 shares at least two "
                "unchanged bands. Compare unique-code candidates only; no "
                "all-image-pairs scan or transitive component expansion."
            ),
            "pillow_version": PILLOW_VERSION,
            "numpy_version": np.__version__,
            "batch_size": args.batch_size,
            "patient_identity_limitation": (
                "Neither source provides usable patient-to-image identifiers. "
                "Hashes and similarity do not establish whether images belong "
                "to the same patient, nor can this audit assess all patient "
                "leakage or repeat-exam relationships."
            ),
        },
        "mendeley": {
            "inventory_validation": mendeley_validation,
            "exact_file_duplicates": mendeley_exact,
            "identical_phash_nonexact_collisions": mendeley_phash_collisions,
            "perceptual_hash_search": mendeley_search,
            "near_hash_code_pairs": mendeley_neighbors,
            "rows": mendeley_rows,
        },
        "chestmnist": {
            "inventory_validation": chest_validation,
            "exact_pixel_duplicates": chest_exact,
            "identical_phash_nonexact_collisions": chest_phash_collisions,
            "perceptual_hash_search": chest_search,
            "near_hash_code_pairs": chest_neighbors,
            "rows": chest_rows,
        },
    }
    write_json_atomic(args.output, result)
    print(json.dumps({
        "output": str(args.output),
        "mendeley": {
            **mendeley_validation,
            "exact_duplicate_groups": mendeley_exact["duplicate_group_count"],
            "exact_duplicate_pairs": mendeley_exact["duplicate_image_pair_count"],
            "nonexact_same_phash_pairs": mendeley_phash_collisions[
                "nonexact_same_phash_image_pair_count"
            ],
            "phash_neighbor_hash_code_pairs": mendeley_search["neighbor_code_pair_count"],
            "phash_neighbor_image_pairs_by_class_pair": mendeley_search[
                "neighbor_image_pairs_by_class_pair_and_distance"
            ],
        },
        "chestmnist": {
            **chest_validation,
            "exact_duplicate_groups": chest_exact["duplicate_group_count"],
            "exact_duplicate_pairs": chest_exact["duplicate_image_pair_count"],
            "exact_pairs_by_split_pair": chest_exact[
                "duplicate_pairs_by_source_split_pair"
            ],
            "nonexact_same_phash_pairs_by_split_pair": chest_phash_collisions[
                "nonexact_same_phash_pairs_by_source_split_pair"
            ],
            "phash_neighbor_hash_code_pairs": chest_search["neighbor_code_pair_count"],
            "phash_neighbor_image_pairs_by_split_pair": chest_search[
                "neighbor_image_pairs_by_split_pair_and_distance"
            ],
        },
    }, indent=2))


if __name__ == "__main__":
    main()
