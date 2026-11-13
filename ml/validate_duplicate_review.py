"""Reproduce duplicate accounting and create local visual-review contact sheets."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CHEST_LABEL_NAMES = (
    "atelectasis",
    "cardiomegaly",
    "effusion",
    "infiltration",
    "mass",
    "nodule",
    "pneumonia",
    "pneumothorax",
    "consolidation",
    "edema",
    "emphysema",
    "fibrosis",
    "pleural",
    "hernia",
)
SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}


def pair_key(first: str, second: str) -> str:
    return "|".join(sorted((first, second)))


def pair_count(size: int) -> int:
    return size * (size - 1) // 2


def parse_chest_id(image_id: str) -> tuple[str, int]:
    match = re.fullmatch(r"(train|val|test)\[(\d+)\]", image_id)
    if match is None:
        raise ValueError(f"Unexpected ChestMNIST row identifier: {image_id}")
    return match.group(1), int(match.group(2))


def image_digest(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def source_split_pair_counts(members: list[dict[str, object]]) -> Counter[str]:
    counts = Counter(str(member["split"]) for member in members)
    result: Counter[str] = Counter()
    splits = sorted(counts)
    for index, first in enumerate(splits):
        for second in splits[index:]:
            amount = (
                pair_count(counts[first])
                if first == second
                else counts[first] * counts[second]
            )
            if amount:
                result[pair_key(first, second)] += amount
    return result


def group_by_sha(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["sha256"])].append(row)
    return groups


def validate_phash_accounting(dataset: dict[str, object]) -> dict[str, object]:
    rows = dataset["rows"]
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate image identifiers in pHash audit rows.")

    by_code: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_code[int(row["phash"])].append(row)

    same_code_pairs = 0
    same_code_groups = 0
    same_code_pairs_by_split: Counter[str] = Counter()
    seen_image_pairs: set[tuple[str, str]] = set()
    for members in by_code.values():
        by_sha = group_by_sha(members)
        if len(by_sha) < 2:
            continue
        same_code_groups += 1
        same_code_pairs += pair_count(len(members)) - sum(
            pair_count(len(exact_members)) for exact_members in by_sha.values()
        )
        for first_index, first in enumerate(members):
            for second in members[first_index + 1 :]:
                if first["sha256"] != second["sha256"]:
                    image_pair = tuple(sorted((str(first["id"]), str(second["id"]))))
                    if image_pair in seen_image_pairs:
                        raise ValueError("Repeated image pair in same-pHash candidates.")
                    seen_image_pairs.add(image_pair)
                    same_code_pairs_by_split[
                        pair_key(str(first.get("split")), str(second.get("split")))
                    ] += 1

    recorded_collisions = dataset["identical_phash_nonexact_collisions"]
    if same_code_pairs != recorded_collisions["nonexact_same_phash_image_pair_count"]:
        raise ValueError("Recomputed same-pHash image-pair count differs from audit.")
    if same_code_groups != len(recorded_collisions["groups"]):
        raise ValueError("Recomputed same-pHash group count differs from audit.")
    recorded_split_pairs = recorded_collisions[
        "nonexact_same_phash_pairs_by_source_split_pair"
    ]
    if dict(sorted(same_code_pairs_by_split.items())) != recorded_split_pairs:
        raise ValueError("Same-pHash split-pair accounting differs from audit.")

    recorded_neighbors = dataset["near_hash_code_pairs"]
    seen_code_pairs: set[tuple[int, int]] = set()
    image_pairs_by_distance: Counter[int] = Counter()
    split_pairs_by_distance: dict[int, Counter[str]] = defaultdict(Counter)
    for candidate in recorded_neighbors:
        first_code = int(candidate["first_phash"], 16)
        second_code = int(candidate["second_phash"], 16)
        code_pair = tuple(sorted((first_code, second_code)))
        if code_pair[0] == code_pair[1] or code_pair in seen_code_pairs:
            raise ValueError("Near-code pairs contain an identical or repeated pair.")
        seen_code_pairs.add(code_pair)

        measured_distance = (first_code ^ second_code).bit_count()
        if measured_distance != candidate["hamming_distance"] or measured_distance > 3:
            raise ValueError("Recorded pHash Hamming distance is incorrect.")
        first_members = by_code[first_code]
        second_members = by_code[second_code]
        if len(first_members) != candidate["first_member_count"]:
            raise ValueError("First pHash member count differs from source rows.")
        if len(second_members) != candidate["second_member_count"]:
            raise ValueError("Second pHash member count differs from source rows.")
        image_pairs_by_distance[measured_distance] += (
            len(first_members) * len(second_members)
        )
        for first in first_members:
            for second in second_members:
                image_pair = tuple(sorted((str(first["id"]), str(second["id"]))))
                if image_pair in seen_image_pairs:
                    raise ValueError("Repeated image pair across pHash candidates.")
                seen_image_pairs.add(image_pair)
        first_splits = Counter(str(row.get("split") or "") for row in first_members)
        second_splits = Counter(str(row.get("split") or "") for row in second_members)
        for first_split, first_count in first_splits.items():
            for second_split, second_count in second_splits.items():
                split_pairs_by_distance[measured_distance][
                    pair_key(first_split, second_split)
                ] += first_count * second_count

    search = dataset["perceptual_hash_search"]
    if len(seen_code_pairs) != search["neighbor_code_pair_count"]:
        raise ValueError("Unique near-code pair count differs from audit.")
    if dict(sorted(image_pairs_by_distance.items())) != {
        int(distance): count
        for distance, count in search["neighbor_image_pair_counts_by_distance"].items()
        if count
    }:
        raise ValueError("Weighted image-pair counts differ from near-code records.")
    measured_split_summary = {
        str(distance): dict(sorted(counts.items()))
        for distance, counts in sorted(split_pairs_by_distance.items())
    }
    if measured_split_summary != search["neighbor_image_pairs_by_split_pair_and_distance"]:
        raise ValueError("Near-code split-pair counts differ from audit.")

    return {
        "same_phash_nonexact_group_count": same_code_groups,
        "same_phash_nonexact_image_pair_count": same_code_pairs,
        "same_phash_pairs_by_split_pair": dict(sorted(same_code_pairs_by_split.items())),
        "nearby_unique_code_pair_count": len(seen_code_pairs),
        "nearby_image_pair_count": sum(image_pairs_by_distance.values()),
        "nearby_image_pairs_by_hamming_distance": {
            str(distance): image_pairs_by_distance[distance]
            for distance in sorted(image_pairs_by_distance)
        },
        "nearby_image_pairs_by_split_pair_and_distance": measured_split_summary,
        "unique_phash_codes": len(by_code),
        "reported_index_candidate_checks": search["hamming_distance_comparisons"],
    }


def validate_chest_exact_groups(
    audit: dict[str, object], archive_path: Path
) -> tuple[dict[str, object], list[dict[str, object]]]:
    chest = audit["chestmnist"]
    audit_rows = {str(row["id"]): row for row in chest["rows"]}
    image_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    unique_by_split: dict[str, set[str]] = {split: set() for split in SPLIT_ORDER}
    split_counts: Counter[str] = Counter()

    with np.load(archive_path, allow_pickle=False) as archive:
        if not {"train", "val", "test"} <= {
            key.removesuffix("_images")
            for key in archive.files
            if key.endswith("_images")
        }:
            raise ValueError("ChestMNIST archive is missing a source split.")
        for split in ("train", "val", "test"):
            images = archive[f"{split}_images"]
            labels = archive[f"{split}_labels"]
            if labels.ndim != 2 or labels.shape[1] != len(CHEST_LABEL_NAMES):
                raise ValueError(f"Unexpected {split} ChestMNIST label shape.")
            if not np.isin(labels, (0, 1)).all():
                raise ValueError(f"Unexpected non-binary {split} ChestMNIST labels.")
            split_counts[split] = len(images)
            for index, image in enumerate(images):
                image_id = f"{split}[{index}]"
                row = audit_rows.get(image_id)
                if row is None:
                    raise ValueError(f"Missing audit row for source image {image_id}.")
                digest = image_digest(image)
                if digest != row["sha256"]:
                    raise ValueError(f"Source pixel hash differs for {image_id}.")
                unique_by_split[split].add(digest)
                image_groups[digest].append(
                    {
                        "id": image_id,
                        "split": split,
                        "index": index,
                        "labels": [int(value) for value in labels[index]],
                    }
                )

    if len(audit_rows) != sum(split_counts.values()):
        raise ValueError("ChestMNIST audit row count differs from source arrays.")
    duplicate_groups = [
        {"sha256": digest, "members": members}
        for digest, members in image_groups.items()
        if len(members) > 1
    ]
    duplicate_groups.sort(key=lambda group: group["members"][0]["id"])

    recorded_groups = chest["exact_pixel_duplicates"]["groups"]
    recorded_by_members = {
        tuple(sorted(str(member["id"]) for member in group["members"])): group
        for group in recorded_groups
    }
    actual_by_members = {
        tuple(sorted(str(member["id"]) for member in group["members"])): group
        for group in duplicate_groups
    }
    if actual_by_members.keys() != recorded_by_members.keys():
        raise ValueError("Recomputed exact duplicate members differ from audit.")
    for member_ids, group in actual_by_members.items():
        if group["sha256"] != recorded_by_members[member_ids]["sha256"]:
            raise ValueError("Recomputed exact duplicate SHA differs from audit.")

    details: list[dict[str, object]] = []
    identical_label_groups = 0
    conflicting_label_groups = 0
    for group_index, group in enumerate(duplicate_groups, start=1):
        members = group["members"]
        label_vectors = {tuple(member["labels"]) for member in members}
        labels_match = len(label_vectors) == 1
        identical_label_groups += int(labels_match)
        conflicting_label_groups += int(not labels_match)
        details.append(
            {
                "group": f"G{group_index:02d}",
                "sha256": group["sha256"],
                "members": [
                    {
                        "id": member["id"],
                        "split": member["split"],
                        "labels": member["labels"],
                        "positive_labels": [
                            name
                            for name, value in zip(
                                CHEST_LABEL_NAMES, member["labels"], strict=True
                            )
                            if value
                        ],
                    }
                    for member in members
                ],
                "label_vectors_identical": labels_match,
                "duplicate_pair_count": pair_count(len(members)),
                "source_split_pairs": dict(
                    sorted(source_split_pair_counts(members).items())
                ),
            }
        )

    exact_pairs_by_split: Counter[str] = Counter()
    exact_pair_count = 0
    exact_excess_images = 0
    for group in duplicate_groups:
        members = group["members"]
        exact_pairs_by_split.update(source_split_pair_counts(members))
        exact_pair_count += pair_count(len(members))
        exact_excess_images += len(members) - 1

    within_split_unique = {
        split: len(digests) for split, digests in unique_by_split.items()
    }
    all_unique = len(image_groups)
    expected_inventory_counts = {"train": 78_468, "val": 11_219, "test": 22_433}
    if dict(split_counts) != expected_inventory_counts:
        raise ValueError("ChestMNIST source split sizes differ from the inventory.")

    summary = {
        "source_image_count": int(sum(split_counts.values())),
        "source_split_image_counts": dict(split_counts),
        "exact_duplicate_group_count": len(duplicate_groups),
        "exact_duplicate_pair_count": exact_pair_count,
        "exact_duplicate_excess_images": exact_excess_images,
        "exact_duplicate_pairs_by_source_split_pair": dict(
            sorted(exact_pairs_by_split.items())
        ),
        "unique_images_after_deduplicating_within_each_split": within_split_unique,
        "within_split_unique_image_count_sum": int(sum(within_split_unique.values())),
        "globally_unique_pixel_image_count": all_unique,
        "unique_sample_reduction_after_global_exact_deduplication": int(
            sum(split_counts.values()) - all_unique
        ),
        "identical_label_group_count": identical_label_groups,
        "conflicting_label_group_count": conflicting_label_groups,
        "label_names_in_source_order": list(CHEST_LABEL_NAMES),
    }
    if len(duplicate_groups) != 25 or exact_pair_count != 27:
        raise ValueError("Recomputed ChestMNIST exact-duplicate counts differ.")
    return summary, details


def evenly_spaced(items: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    if len(items) <= limit:
        return items
    if limit <= 0:
        return []
    if limit == 1:
        return [items[len(items) // 2]]
    indices = {
        round(index * (len(items) - 1) / (limit - 1))
        for index in range(limit)
    }
    return [items[index] for index in sorted(indices)]


def expand_phash_pairs(
    dataset: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows_by_code: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in dataset["rows"]:
        rows_by_code[int(row["phash"])].append(row)
    same_code_pairs: list[dict[str, object]] = []
    for code, members in rows_by_code.items():
        for first, second in combinations(members, 2):
            if first["sha256"] == second["sha256"]:
                continue
            same_code_pairs.append(
                {
                    "kind": "same-pHash non-exact",
                    "hamming_distance": 0,
                    "first": first,
                    "second": second,
                    "first_phash": f"{code:016x}",
                    "second_phash": f"{code:016x}",
                }
            )

    nearby_pairs: list[dict[str, object]] = []
    for candidate in dataset["near_hash_code_pairs"]:
        first_code = int(candidate["first_phash"], 16)
        second_code = int(candidate["second_phash"], 16)
        for first in rows_by_code[first_code]:
            for second in rows_by_code[second_code]:
                nearby_pairs.append(
                    {
                        "kind": "nearby distinct pHash",
                        "hamming_distance": candidate["hamming_distance"],
                        "first": first,
                        "second": second,
                        "first_phash": candidate["first_phash"],
                        "second_phash": candidate["second_phash"],
                    }
                )
    return same_code_pairs, nearby_pairs


def sample_visual_pairs(
    dataset_name: str,
    same_code_pairs: list[dict[str, object]],
    nearby_pairs: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    same_code_pairs.sort(
        key=lambda pair: (
            str(pair["first"]["id"]),
            str(pair["second"]["id"]),
        )
    )
    nearby_pairs.sort(
        key=lambda pair: (
            pair_key(
                str(pair["first"].get("split") or ""),
                str(pair["second"].get("split") or ""),
            ),
            str(pair["first_phash"]),
            str(pair["second_phash"]),
            str(pair["first"]["id"]),
            str(pair["second"]["id"]),
        )
    )
    if dataset_name == "mendeley":
        selected_nearby = nearby_pairs
        method = "All same-pHash and all distance<=3 candidate image pairs."
    else:
        strata: dict[str, list[dict[str, object]]] = defaultdict(list)
        for pair in nearby_pairs:
            first_split = str(pair["first"]["split"])
            second_split = str(pair["second"]["split"])
            strata[pair_key(first_split, second_split)].append(pair)
        selected_nearby = []
        limits = {
            "test|train": 10,
            "test|val": 10,
            "train|val": 10,
            "test|test": 5,
            "train|train": 5,
            "val|val": 5,
        }
        for stratum, limit in limits.items():
            selected_nearby.extend(evenly_spaced(strata.get(stratum, []), limit))
        method = (
            "All same-pHash non-exact image pairs; for distinct-code candidates, "
            "evenly spaced sorted sample of up to 10 pairs from each cross-split "
            "stratum and up to 5 from each within-split stratum."
        )
    selected = same_code_pairs + selected_nearby
    selected.sort(
        key=lambda pair: (
            pair["kind"],
            pair_key(
                str(pair["first"].get("split") or ""),
                str(pair["second"].get("split") or ""),
            ),
            str(pair["first"]["id"]),
            str(pair["second"]["id"]),
        )
    )
    counts = Counter(
        f"{pair['kind']} / {pair_key(str(pair['first'].get('split') or ''), str(pair['second'].get('split') or ''))}"
        for pair in selected
    )
    return selected, {
        "method": method,
        "same_phash_nonexact_candidates_available": len(same_code_pairs),
        "nearby_distinct_hash_candidates_available": len(nearby_pairs),
        "same_phash_pairs_selected": len(same_code_pairs),
        "nearby_distinct_hash_pairs_selected": len(selected_nearby),
        "total_pairs_selected_for_visual_review": len(selected),
        "selected_counts_by_kind_and_split_pair": dict(sorted(counts.items())),
    }


def get_image_loader(
    dataset_name: str, research_data: Path
):
    if dataset_name == "mendeley":
        image_root = (
            research_data
            / "Mendeley_Breast_Ultrasound_Image_v1"
            / "extracted"
        )

        def load(image_id: str) -> Image.Image:
            with Image.open(image_root / Path(image_id)) as image:
                return image.convert("L").copy()

        return load

    archive_path = (
        research_data
        / "MedMNIST_Plus_ChestMNIST_v3.0"
        / "chestmnist_64.npz"
    )
    archive = np.load(archive_path, allow_pickle=False)
    split_images: dict[str, np.ndarray] = {}

    def load(image_id: str) -> Image.Image:
        split, index = parse_chest_id(image_id)
        if split not in split_images:
            split_images[split] = archive[f"{split}_images"]
        return Image.fromarray(split_images[split][index], mode="L")

    return load, archive


def draw_image_tile(
    canvas: Image.Image,
    image: Image.Image,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    thumbnail = image.copy()
    thumbnail.thumbnail((width, height), Image.Resampling.NEAREST)
    canvas.paste(
        thumbnail,
        (x + (width - thumbnail.width) // 2, y + (height - thumbnail.height) // 2),
    )


def write_exact_contact_sheets(
    details: list[dict[str, object]],
    research_data: Path,
    output_dir: Path,
) -> list[str]:
    if not details:
        return []
    loader, archive = get_image_loader("chestmnist", research_data)
    font = ImageFont.load_default()
    output_files: list[str] = []
    groups_per_sheet = 5
    for offset in range(0, len(details), groups_per_sheet):
        page = offset // groups_per_sheet + 1
        batch = details[offset : offset + groups_per_sheet]
        sheet = Image.new("RGB", (1100, 980), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text(
            (12, 10),
            f"ChestMNIST exact-pixel groups {offset + 1}-{offset + len(batch)} of {len(details)}",
            fill="black",
            font=font,
        )
        for row_index, group in enumerate(batch):
            top = 40 + row_index * 185
            label_status = (
                "labels identical" if group["label_vectors_identical"] else "LABEL CONFLICT"
            )
            draw.text(
                (12, top),
                f"{group['group']} | {label_status} | "
                f"positive targets: {', '.join(group['members'][0]['positive_labels']) or 'none'}",
                fill="black",
                font=font,
            )
            for member_index, member in enumerate(group["members"]):
                left = 12 + member_index * 270
                image = loader(str(member["id"]))
                draw_image_tile(
                    sheet,
                    image,
                    x=left,
                    y=top + 20,
                    width=170,
                    height=140,
                )
                draw.text(
                    (left, top + 164),
                    f"{member['id']} labels={member['labels']}",
                    fill="black",
                    font=font,
                )
        name = f"chestmnist_exact_groups_{page:02d}.png"
        sheet.save(output_dir / name, optimize=True)
        output_files.append(name)
    archive.close()
    return output_files


def write_phash_contact_sheets(
    dataset_name: str,
    pairs: list[dict[str, object]],
    research_data: Path,
    output_dir: Path,
) -> list[str]:
    if not pairs:
        return []
    loaded = get_image_loader(dataset_name, research_data)
    if dataset_name == "mendeley":
        loader = loaded
        archive = None
    else:
        loader, archive = loaded
    font = ImageFont.load_default()
    output_files: list[str] = []
    pairs_per_sheet = 24
    image_cache: dict[str, Image.Image] = {}
    for offset in range(0, len(pairs), pairs_per_sheet):
        page = offset // pairs_per_sheet + 1
        batch = pairs[offset : offset + pairs_per_sheet]
        sheet = Image.new("RGB", (1280, 990), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text(
            (12, 8),
            f"{dataset_name}: pHash visual candidates {offset + 1}-{offset + len(batch)} of {len(pairs)}",
            fill="black",
            font=font,
        )
        for pair_index, pair in enumerate(batch):
            col = pair_index % 4
            row = pair_index // 4
            left = 10 + col * 317
            top = 30 + row * 158
            first = pair["first"]
            second = pair["second"]
            split_pair = pair_key(
                str(first.get("split") or ""),
                str(second.get("split") or ""),
            )
            draw.text(
                (left, top),
                f"{pair['kind']} d={pair['hamming_distance']} {split_pair}",
                fill="black",
                font=font,
            )
            for member_index, member in enumerate((first, second)):
                image_id = str(member["id"])
                if image_id not in image_cache:
                    image_cache[image_id] = loader(image_id)
                tile_x = left + member_index * 150
                draw_image_tile(
                    sheet,
                    image_cache[image_id],
                    x=tile_x,
                    y=top + 16,
                    width=144,
                    height=105,
                )
                draw.text(
                    (tile_x, top + 123),
                    image_id[-20:],
                    fill="black",
                    font=font,
                )
        name = f"{dataset_name}_phash_candidates_{page:02d}.png"
        sheet.save(output_dir / name, optimize=True)
        output_files.append(name)
    if archive is not None:
        archive.close()
    return output_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("data") / "research_data" / "limited_acquisition_2026-11-01"
    parser.add_argument("--research-data", type=Path, default=root)
    parser.add_argument("--audit-json", type=Path, default=root / "duplicate_audit.json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "duplicate_review_2026-11-01",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing review output directory: {args.output_dir}"
        )

    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    for dataset_name in ("mendeley", "chestmnist"):
        audit[dataset_name]["phash_validation"] = validate_phash_accounting(
            audit[dataset_name]
        )
    chest_summary, exact_details = validate_chest_exact_groups(
        audit,
        args.research_data
        / "MedMNIST_Plus_ChestMNIST_v3.0"
        / "chestmnist_64.npz",
    )
    audit["chestmnist"]["exact_duplicate_label_validation"] = {
        "summary": chest_summary,
        "groups": exact_details,
    }
    audit["mendeley"]["exact_file_duplicate_reproduction"] = {
        "archive_sha256_verified": audit["mendeley"]["inventory_validation"][
            "archive_sha256"
        ],
        "exact_file_duplicate_group_count": audit["mendeley"][
            "exact_file_duplicates"
        ]["duplicate_group_count"],
        "exact_file_duplicate_pair_count": audit["mendeley"][
            "exact_file_duplicates"
        ]["duplicate_image_pair_count"],
        "image_count": audit["mendeley"]["inventory_validation"][
            "inventory_image_count"
        ],
    }

    nearby_candidates: dict[str, list[dict[str, object]]] = {}
    visual_samples: dict[str, dict[str, object]] = {}
    for dataset_name in ("mendeley", "chestmnist"):
        same, nearby = expand_phash_pairs(audit[dataset_name])
        selected, sample_summary = sample_visual_pairs(dataset_name, same, nearby)
        visual_samples[dataset_name] = sample_summary
        audit[dataset_name]["visual_review"] = sample_summary
        audit[dataset_name]["visual_review"]["nearby_candidate_pairs_available"] = len(
            nearby
        )
        nearby_candidates[dataset_name] = selected

    args.output_dir.mkdir(parents=True, exist_ok=False)
    exact_sheets = write_exact_contact_sheets(
        exact_details, args.research_data, args.output_dir
    )
    visual_samples["chestmnist"]["exact_duplicate_contact_sheets"] = exact_sheets
    for dataset_name in ("mendeley", "chestmnist"):
        sheets = write_phash_contact_sheets(
            dataset_name,
            nearby_candidates[dataset_name],
            args.research_data,
            args.output_dir,
        )
        visual_samples[dataset_name]["phash_contact_sheets"] = sheets
    audit["visual_review"] = visual_samples
    output_json = args.output_dir / "duplicate_validation_and_review.json"
    output_json.write_text(
        json.dumps(audit, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"output_dir": str(args.output_dir), "visual_review": visual_samples}, indent=2))


if __name__ == "__main__":
    main()
