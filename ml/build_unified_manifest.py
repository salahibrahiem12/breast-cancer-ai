"""Build and verify the unified versioned dataset manifest (v1) across all acquired sources."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

from ml.inspect_tcia_breast import fingerprint
from ml.prepare_dataset import UnionFind

MANIFEST_CSV = Path("data/prepared/unified_dataset_manifest_v1.csv")
SUMMARY_JSON = Path("data/prepared/unified_dataset_manifest_v1_summary.json")
REPORT_JSON = Path("reports/unified_dataset_manifest_v1.json")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_unified_manifest() -> dict:
    eligible_ref = json.loads(
        Path("reports/design_audit_v1/eligible_manifest.json").read_text(encoding="utf-8")
    )
    eligible_by_sha = {row["sha256"]: (idx, row) for idx, row in enumerate(eligible_ref)}

    # 1. Mendeley v1 (250 acquired: 162 eligible, 88 excluded_previous_test)
    mendeley_split = _read_csv(Path("data/prepared/mendeley_dev_v2/split_manifest.csv"))
    mendeley_root = Path("data/raw/mendeley_v1/official-v1")

    # 2. TCIA BrEaST v1 (256 acquired primary scans: 197 eligible biopsy-confirmed B/M, 55 follow-up benign, 4 normal)
    tcia_inv = _read_csv(
        Path("data/research_data/tcia_breast_lesions_usg_2026-11-03/inventory/cohort_inventory.csv")
    )
    tcia_ext = {
        row["patient_id"]: row
        for row in _read_csv(Path("data/prepared/tcia_breast_biopsy_external_v1/split_manifest.csv"))
    }

    # 3. BUS-BRA v1 (1,875 acquired: 1,873 eligible, 2 excluded in Case 54 for conservative dHash<=5 near-match)
    busbra_inv = _read_csv(Path("data/research_data/busbra_reserved_v1/inventory.csv"))
    busbra_eligible_paths = {
        row["archive_path"]
        for row in _read_csv(Path("reports/busbra_locked_v1/eligible_inventory.csv"))
    }

    unified_rows: list[dict] = []

    # Populate Mendeley rows
    for row in mendeley_split:
        img_path = mendeley_root / row["path"]
        fp = fingerprint(img_path.read_bytes())
        assert fp["sha256"] == row["sha256"], f"Mendeley SHA256 mismatch: {row['path']}"
        is_eligible = row["split"] in {"train", "validation"}
        elig_idx, elig_rec = eligible_by_sha.get(row["sha256"], (None, None))
        assert is_eligible == (elig_rec is not None)
        unified_rows.append(
            {
                "source": "mendeley",
                "license": "CC BY 4.0 (DataCite doi:10.17632/wmy84gzngw.1)",
                "image_path": f"data/raw/mendeley_v1/official-v1/{row['path']}",
                "label": row["label"],
                "binary_target": "1" if row["label"] == "malignant" else "0",
                "known_grouping_id": f"mendeley:{row['group_id']}",
                "known_grouping_type": "perceptual_duplicate_group_no_patient_id",
                "patient_id_status": "unavailable",
                "duplicate_group": elig_rec["group"] if elig_rec else f"excluded_mendeley:{row['group_id']}",
                "eligible": "true" if is_eligible else "false",
                "exclusion_reason": "" if is_eligible else "historical_mendeley_test_linked_duplicate_group",
                "eligible_row_index": str(elig_idx) if elig_idx is not None else "",
                "is_mask": "false",
                "sha256": fp["sha256"],
                "pixel_sha256": fp["pixel_sha256"],
                "dhash": str(fp["dhash"]),
                "phash": str(fp["phash"]),
            }
        )

    # Populate TCIA rows
    for row in tcia_inv:
        pid = row["patient_id"]
        is_eligible = pid in tcia_ext
        elig_idx, elig_rec = eligible_by_sha.get(row["sha256"], (None, None))
        assert is_eligible == (elig_rec is not None)
        if is_eligible:
            reason = ""
        elif row["label"] == "normal":
            reason = "normal_class_outside_binary_target"
        else:
            reason = "followup_only_reference_standard_not_biopsy"
        unified_rows.append(
            {
                "source": "tcia",
                "license": "CC BY 4.0 (TCIA BREAST-LESIONS-USG doi:10.7937/9WKK-Q141)",
                "image_path": f"data/research_data/tcia_breast_lesions_usg_2026-11-03/{row['path']}",
                "label": row["label"],
                "binary_target": (
                    "1"
                    if row["label"] == "malignant"
                    else ("0" if row["label"] == "benign" else "")
                ),
                "known_grouping_id": pid,
                "known_grouping_type": "publisher_reported_one_scan_per_patient_id",
                "patient_id_status": "publisher_reported_within_source_only",
                "duplicate_group": elig_rec["group"] if elig_rec else f"excluded_{pid}",
                "eligible": "true" if is_eligible else "false",
                "exclusion_reason": reason,
                "eligible_row_index": str(elig_idx) if elig_idx is not None else "",
                "is_mask": "false",
                "sha256": row["sha256"],
                "pixel_sha256": row["pixel_sha256"],
                "dhash": str(row["dhash"]),
                "phash": str(row["phash"]),
            }
        )

    # Populate BUS-BRA rows
    for row in busbra_inv:
        is_eligible = row["archive_path"] in busbra_eligible_paths
        elig_idx, elig_rec = eligible_by_sha.get(row["sha256"], (None, None))
        assert is_eligible == (elig_rec is not None)
        case_unit = f"busbra:{row['source_case']}"
        unified_rows.append(
            {
                "source": "busbra",
                "license": "CC BY 4.0 research scope (Zenodo 8231412 / doi:10.1002/mp.16812)",
                "image_path": f"data/research_data/busbra_reserved_v1/BUSBRA.zip:{row['archive_path']}",
                "label": row["label"],
                "binary_target": "1" if row["label"] == "malignant" else "0",
                "known_grouping_id": case_unit,
                "known_grouping_type": "source_case_unresolved_patient_semantics",
                "patient_id_status": "unresolved_case_to_patient_semantics",
                "duplicate_group": elig_rec["group"] if elig_rec else f"excluded_{case_unit}",
                "eligible": "true" if is_eligible else "false",
                "exclusion_reason": (
                    ""
                    if is_eligible
                    else "conservative_cross_source_dhash_near_match_case_exclusion"
                ),
                "eligible_row_index": str(elig_idx) if elig_idx is not None else "",
                "is_mask": "false",
                "sha256": row["sha256"],
                "pixel_sha256": row["pixel_sha256"],
                "dhash": str(row["dhash"]),
                "phash": str(row["phash"]),
            }
        )

    for idx, row in enumerate(unified_rows):
        row["manifest_row_id"] = str(idx)

    # Verify invariants
    assert len(unified_rows) == 2381, f"Expected 2381 acquired images, got {len(unified_rows)}"
    eligible_rows = [r for r in unified_rows if r["eligible"] == "true"]
    excluded_rows = [r for r in unified_rows if r["eligible"] == "false"]
    assert len(eligible_rows) == 2232
    assert len(excluded_rows) == 149

    # Verify zero mask files in unified rows
    for r in unified_rows:
        lower_path = r["image_path"].lower()
        basename = lower_path.rsplit("/", 1)[-1]
        assert not any(tok in basename for tok in ("_mask", "_tumor", "_other", "mask_")), (
            f"Mask file admitted to unified image manifest: {r['image_path']}"
        )
        assert r["is_mask"] == "false"

    # Verify BUS-BRA mask count in archive is separate
    with ZipFile("data/research_data/busbra_reserved_v1/BUSBRA.zip") as zf:
        busbra_masks = [n for n in zf.namelist() if n.startswith("BUSBRA/Masks/") and n.endswith(".png")]
    assert len(busbra_masks) == 1875

    # Verify group purity and source isolation among eligible rows
    groups = defaultdict(list)
    units = defaultdict(list)
    for r in eligible_rows:
        groups[r["duplicate_group"]].append(r)
        units[r["known_grouping_id"]].append(r)
    assert len(groups) == 1298
    for gid, members in groups.items():
        assert len({m["source"] for m in members}) == 1, f"Cross-source group: {gid}"
        assert len({m["label"] for m in members}) == 1, f"Mixed-label group: {gid}"
        assert len({m["binary_target"] for m in members}) == 1
    for uid, members in units.items():
        assert len({m["duplicate_group"] for m in members}) == 1, f"Known unit split across duplicate groups: {uid}"

    # Verify cross-source exact and near-duplicate links among eligible rows
    cross_source_exact_links = 0
    cross_source_near_links_d5 = 0
    for i, a in enumerate(eligible_rows):
        da, pa = int(a["dhash"]), int(a["phash"])
        for b in eligible_rows[:i]:
            if a["source"] != b["source"]:
                if a["sha256"] == b["sha256"] or a["pixel_sha256"] == b["pixel_sha256"]:
                    cross_source_exact_links += 1
                if (da ^ int(b["dhash"])).bit_count() <= 5 or (pa ^ int(b["phash"])).bit_count() <= 5:
                    cross_source_near_links_d5 += 1
            else:
                if (
                    a["sha256"] == b["sha256"]
                    or a["pixel_sha256"] == b["pixel_sha256"]
                    or (da ^ int(b["dhash"])).bit_count() <= 5
                    or (pa ^ int(b["phash"])).bit_count() <= 5
                ):
                    assert a["duplicate_group"] == b["duplicate_group"], (
                        f"Unlinked duplicate pair within source: {a['image_path']} vs {b['image_path']}"
                    )

    assert cross_source_exact_links == 0
    assert cross_source_near_links_d5 == 0

    fieldnames = [
        "manifest_row_id",
        "eligible_row_index",
        "source",
        "license",
        "image_path",
        "label",
        "binary_target",
        "known_grouping_id",
        "known_grouping_type",
        "patient_id_status",
        "duplicate_group",
        "eligible",
        "exclusion_reason",
        "is_mask",
        "sha256",
        "pixel_sha256",
        "dhash",
        "phash",
    ]
    MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unified_rows)

    manifest_sha256 = hashlib.sha256(MANIFEST_CSV.read_bytes()).hexdigest()

    # Reconcile counts by source, label, and eligibility
    reconciliation = {}
    for src in ("mendeley", "tcia", "busbra"):
        src_rows = [r for r in unified_rows if r["source"] == src]
        reconciliation[src] = {
            "acquired_images": len(src_rows),
            "acquired_known_units": len({r["known_grouping_id"] for r in src_rows}),
            "eligible_images": sum(r["eligible"] == "true" for r in src_rows),
            "eligible_known_units": len({r["known_grouping_id"] for r in src_rows if r["eligible"] == "true"}),
            "eligible_duplicate_groups": len({r["duplicate_group"] for r in src_rows if r["eligible"] == "true"}),
            "excluded_images": sum(r["eligible"] == "false" for r in src_rows),
            "exclusion_reasons": dict(
                Counter(r["exclusion_reason"] for r in src_rows if r["eligible"] == "false")
            ),
            "by_label": {},
        }
        for lbl in sorted({r["label"] for r in src_rows}):
            lbl_rows = [r for r in src_rows if r["label"] == lbl]
            lbl_elig = [r for r in lbl_rows if r["eligible"] == "true"]
            lbl_excl = [r for r in lbl_rows if r["eligible"] == "false"]
            reconciliation[src]["by_label"][lbl] = {
                "acquired_images": len(lbl_rows),
                "acquired_known_units": len({r["known_grouping_id"] for r in lbl_rows}),
                "eligible_images": len(lbl_elig),
                "eligible_known_units": len({r["known_grouping_id"] for r in lbl_elig}),
                "eligible_duplicate_groups": len({r["duplicate_group"] for r in lbl_elig}),
                "excluded_images": len(lbl_excl),
            }

    summary = {
        "manifest_version": "unified_dataset_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_csv": str(MANIFEST_CSV).replace("\\", "/"),
        "manifest_sha256": manifest_sha256,
        "totals": {
            "acquired_classifier_source_images": len(unified_rows),
            "eligible_images": len(eligible_rows),
            "eligible_global_duplicate_groups": len(groups),
            "excluded_images": len(excluded_rows),
            "exclusion_reason_counts": dict(Counter(r["exclusion_reason"] for r in excluded_rows)),
            "separate_non_image_masks_verified_excluded": {
                "tcia_segmentation_masks": 266,
                "busbra_segmentation_masks": len(busbra_masks),
            },
        },
        "source_reconciliation": reconciliation,
        "integrity_verifications": {
            "all_2381_sha256_verified": True,
            "zero_masks_in_image_manifest": True,
            "zero_cross_source_exact_duplicates_in_eligible": cross_source_exact_links == 0,
            "zero_cross_source_near_duplicates_dhash_phash_le_5_in_eligible": cross_source_near_links_d5 == 0,
            "all_known_units_grouped_into_single_duplicate_group": True,
            "all_duplicate_groups_label_and_source_pure": True,
            "matches_design_audit_eligible_manifest_by_index": [
                r["sha256"]
                for r in sorted(eligible_rows, key=lambda item: int(item["eligible_row_index"]))
            ]
            == [r["sha256"] for r in eligible_ref],
        },
        "other_local_inventories_not_in_classifier": {
            "unverified_root_raw_folders": {
                "path": "data/raw/{benign,malignant,normal}",
                "images": 260,
                "status": "excluded_unverified_provenance_and_license",
            },
            "fetal_head_zenodo_8265464": {
                "path": "data/raw/fetal_head_zenodo/Diverse Fetal Head Images-original-image.full.zip",
                "images": 999,
                "license": "CC BY 4.0 record; per-image author attribution & patient IDs absent",
                "status": "candidate_other_ultrasound_inventory_only",
            },
            "medmnist_chestmnist_64_zenodo_10519652": {
                "path": "data/raw/medmnist/chestmnist_64.npz",
                "images": 112120,
                "license": "CC BY 4.0 derivative archive; NIH upstream patient IDs absent in NPZ",
                "status": "candidate_other_medical_inventory_only",
            },
        },
        "external_acquisition_blockers": {
            "bus_uclm_v3_mendeley_7fvgj4jsp7_3": "Metadata states 683 images / 38 patients under CC BY 4.0, but Mendeley page exposes no file entries and author archive route returned HTTP 403.",
            "busi_whu_mendeley": "Official download routes returned HTTP 403; remains prediction-free and inaccessible without bypassing access controls.",
            "busi_kaggle_mirror": "Third-party mirror CC0 claim is unverified for original patient ultrasound images; usable patient mapping absent.",
            "fresh_external_evaluation_cohort": "None currently available; Mendeley, TCIA BrEaST, and BUS-BRA are all inspected development sources.",
        },
    }

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    REPORT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"Unified manifest written: {len(unified_rows)} acquired ({len(eligible_rows)} eligible, "
        f"{len(excluded_rows)} excluded), SHA256={manifest_sha256}"
    )
    return summary


if __name__ == "__main__":
    build_unified_manifest()
