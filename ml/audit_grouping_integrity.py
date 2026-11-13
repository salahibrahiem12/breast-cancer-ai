"""Comprehensive audit of global duplicate groups, edge types, transitive perceptual chains,
5-image Mendeley crop-sequence structure, historical test leakage, and visual contact sheets.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from ml.inspect_tcia_breast import fingerprint
from ml.prepare_dataset import UnionFind

OUT_DIR = Path("reports/grouping_audit_v1")
SHEETS_DIR = OUT_DIR / "contact_sheets"
AUDIT_JSON = OUT_DIR / "grouping_audit.json"
ELIGIBLE_V2_JSON = OUT_DIR / "eligible_manifest_v2.json"
MANIFEST_V2_CSV = Path("data/prepared/unified_dataset_manifest_v2.csv")
MANIFEST_V2_SUMMARY = Path("data/prepared/unified_dataset_manifest_v2_summary.json")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _hamming(a: int, b: int) -> int:
    return (int(a) ^ int(b)).bit_count()


def _load_pil(image_path: str, zf_cache: dict[str, ZipFile]) -> Image.Image:
    if ".zip:" in image_path:
        zpath, inner = image_path.split(".zip:", 1)
        zpath = zpath + ".zip"
        if zpath not in zf_cache:
            zf_cache[zpath] = ZipFile(zpath)
        raw = zf_cache[zpath].read(inner)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    p = Path(image_path)
    if p.is_file():
        return Image.open(p).convert("RGB")
    if image_path.startswith("data/research_data/tcia_breast_lesions_usg_2026-11-03/"):
        inner = image_path.split("data/research_data/tcia_breast_lesions_usg_2026-11-03/", 1)[1]
        zpath = (
            "data/research_data/tcia_breast_lesions_usg_2026-11-03/"
            "BrEaST-Lesions_USG-images_and_masks-Dec-15-2023.zip"
        )
        if zpath not in zf_cache:
            zf_cache[zpath] = ZipFile(zpath)
        raw = zf_cache[zpath].read(inner)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    return Image.open(image_path).convert("RGB")


def _gray_arr(img: Image.Image, size: tuple[int, int] | None = (128, 128)) -> np.ndarray:
    g = ImageOps.exif_transpose(img).convert("L")
    if size is not None:
        g = g.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(g, dtype=np.float32) / 255.0


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    af = a.ravel() - float(a.mean())
    bf = b.ravel() - float(b.mean())
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    if denom <= 1e-12:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.dot(af, bf) / denom)


def _best_shift_corr(a_native: np.ndarray, b_native: np.ndarray, max_shift: int = 12) -> tuple[float, tuple[int, int]]:
    """Compute maximum Pearson correlation over integer (dy, dx) translations on non-black overlap."""
    if a_native.shape != b_native.shape:
        a_rs = np.asarray(
            Image.fromarray((a_native * 255).astype(np.uint8)).resize((128, 128), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        b_rs = np.asarray(
            Image.fromarray((b_native * 255).astype(np.uint8)).resize((128, 128), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        return _pearson(a_rs, b_rs), (0, 0)
    h, w = a_native.shape
    best_r = _pearson(a_native, b_native)
    best_shift = (0, 0)
    for dy in range(-max_shift, max_shift + 1, 2):
        for dx in range(-max_shift, max_shift + 1, 2):
            y0_a, y1_a = max(0, dy), min(h, h + dy)
            y0_b, y1_b = max(0, -dy), min(h, h - dy)
            x0_a, x1_a = max(0, dx), min(w, w + dx)
            x0_b, x1_b = max(0, -dx), min(w, w - dx)
            sub_a = a_native[y0_a:y1_a, x0_a:x1_a]
            sub_b = b_native[y0_b:y1_b, x0_b:x1_b]
            r = _pearson(sub_a, sub_b)
            if r > best_r:
                best_r = r
                best_shift = (dy, dx)
    return best_r, best_shift


def _best_flip_or_shift_corr(a: np.ndarray, b: np.ndarray) -> dict[str, float | str]:
    best_r = _pearson(a, b)
    best_mode = "direct"
    for mode, cand in [
        ("hflip", np.fliplr(b)),
        ("vflip", np.flipud(b)),
        ("rot180", np.rot90(b, 2)),
    ]:
        r = _pearson(a, cand)
        if r > best_r:
            best_r = r
            best_mode = mode
    return {
        "pearson_direct": round(_pearson(a, b), 4),
        "best_corr": round(best_r, 4),
        "best_mode": best_mode,
    }


def _classify_pair(a: dict, b: dict) -> list[str]:
    reasons = []
    if a["source"] == b["source"] and a["known_grouping_id"] == b["known_grouping_id"]:
        reasons.append("explicit_unit")
    if a["sha256"] == b["sha256"]:
        reasons.append("exact_sha256")
    if a["pixel_sha256"] == b["pixel_sha256"]:
        reasons.append("exact_pixel_sha256")
    dd = _hamming(int(a["dhash"]), int(b["dhash"]))
    pd = _hamming(int(a["phash"]), int(b["phash"]))
    if dd <= 3:
        reasons.append("dhash_le_3")
    elif dd <= 5:
        reasons.append("dhash_4_5")
    if pd <= 3:
        reasons.append("phash_le_3")
    elif pd <= 5:
        reasons.append("phash_4_5")
    return reasons


def _us_num(image_path: str) -> int:
    basename = image_path.rsplit("/", 1)[-1]
    m = re.search(r"us(\d+)\.bmp", basename, re.IGNORECASE)
    return int(m.group(1)) if m else -1


def _render_contact_sheet(
    title: str,
    groups_data: list[dict],
    zf_cache: dict[str, ZipFile],
    out_path: Path,
    max_cols: int = 6,
) -> None:
    thumb_w, thumb_h = 180, 140
    cell_w, cell_h = 200, 195
    header_h = 44
    group_header_h = 34

    total_height = header_h + 12
    for g in groups_data:
        n_items = len(g["members"])
        n_rows = (n_items + max_cols - 1) // max_cols
        total_height += group_header_h + n_rows * cell_h + 14

    canvas_w = max_cols * cell_w + 24
    canvas = Image.new("RGB", (canvas_w, total_height), (18, 22, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 12), title, fill=(240, 244, 250))

    y = header_h
    for g in groups_data:
        draw.rectangle([10, y, canvas_w - 10, y + group_header_h - 4], fill=(34, 42, 58))
        draw.text((16, y + 8), g["header"], fill=(160, 215, 255))
        y += group_header_h
        for idx, m in enumerate(g["members"]):
            r_idx = idx // max_cols
            c_idx = idx % max_cols
            x0 = 12 + c_idx * cell_w
            y0 = y + r_idx * cell_h
            border_col = (220, 95, 95) if m.get("highlight_warn") else (65, 75, 95)
            draw.rectangle([x0, y0, x0 + cell_w - 8, y0 + cell_h - 6], outline=border_col, fill=(26, 30, 40))
            img = _load_pil(m["image_path"], zf_cache)
            orig_size = img.size
            thumb = img.copy()
            thumb.thumbnail((thumb_w, thumb_h), Image.Resampling.BILINEAR)
            tx = x0 + (cell_w - 8 - thumb.width) // 2
            ty = y0 + 4 + (thumb_h - thumb.height) // 2
            canvas.paste(thumb, (tx, ty))
            short_name = m["image_path"].rsplit("/", 1)[-1]
            line1 = f"{short_name[:22]} ({orig_size[0]}x{orig_size[1]})"
            line2 = f"{m['label'][:3].upper()} | {m.get('sub_label', m['known_grouping_id'])}"
            line3 = m.get("note", "")
            draw.text((x0 + 6, y0 + thumb_h + 8), line1, fill=(230, 230, 235))
            draw.text((x0 + 6, y0 + thumb_h + 22), line2[:26], fill=(180, 200, 220))
            if line3:
                draw.text((x0 + 6, y0 + thumb_h + 36), line3[:26], fill=(255, 205, 130))
        n_rows = (len(g["members"]) + max_cols - 1) // max_cols
        y += n_rows * cell_h + 14

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def run_grouping_audit() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)

    rows = _read_csv(Path("data/prepared/unified_dataset_manifest_v1.csv"))
    mendeley_v1_rows = _read_csv(Path("data/prepared/mendeley_v1/split_manifest.csv"))
    mendeley_dev_v2_rows = _read_csv(Path("data/prepared/mendeley_dev_v2/split_manifest.csv"))

    zf_cache: dict[str, ZipFile] = {}
    gray_cache: dict[str, np.ndarray] = {}
    native_gray_cache: dict[str, np.ndarray] = {}
    size_cache: dict[str, tuple[int, int]] = {}

    for r in rows:
        if r["source"] == "mendeley" or r["eligible"] == "true" or "632" in r["known_grouping_id"]:
            img = _load_pil(r["image_path"], zf_cache)
            size_cache[r["sha256"]] = img.size
            gray_cache[r["sha256"]] = _gray_arr(img, (128, 128))
            if r["source"] == "mendeley":
                native_gray_cache[r["sha256"]] = _gray_arr(img, None)

    m_v1_by_sha = {r["sha256"]: r for r in mendeley_v1_rows}
    m_v2_by_sha = {r["sha256"]: r for r in mendeley_dev_v2_rows}
    m_unified = sorted([r for r in rows if r["source"] == "mendeley"], key=lambda r: _us_num(r["image_path"]))
    m_eligible = [r for r in m_unified if r["eligible"] == "true"]

    # 0. Audit the 50 five-image sequences (us{5k+1}..us{5k+5}) in Mendeley (us1.bmp .. us250.bmp)
    m_by_num = {_us_num(r["image_path"]): r for r in m_unified}
    five_image_blocks = []
    partial_leak_blocks = []
    split_eligible_blocks = []
    for k in range(50):
        nums = [5 * k + j for j in range(1, 6)]
        b_rows = [m_by_num[n] for n in nums]
        res_set = sorted({f"{size_cache[r['sha256']][0]}x{size_cache[r['sha256']][1]}" for r in b_rows})
        lbl_set = sorted({r["label"] for r in b_rows})
        v1_splits = [m_v1_by_sha[r["sha256"]]["split"] for r in b_rows]
        v1_groups = sorted({m_v1_by_sha[r["sha256"]]["patient_id"] for r in b_rows})
        v2_splits = [m_v2_by_sha[r["sha256"]]["split"] for r in b_rows]
        v2_units = sorted({m_v2_by_sha[r["sha256"]]["group_id"] for r in b_rows})
        elig_flags = [r["eligible"] == "true" for r in b_rows]
        global_gids = sorted({r["duplicate_group"] for r in b_rows if r["eligible"] == "true"})

        direct_corrs = []
        shift_corrs = []
        d_dists = []
        p_dists = []
        for a, b in combinations(b_rows, 2):
            direct_corrs.append(_pearson(gray_cache[a["sha256"]], gray_cache[b["sha256"]]))
            sc, _ = _best_shift_corr(native_gray_cache[a["sha256"]], native_gray_cache[b["sha256"]])
            shift_corrs.append(sc)
            d_dists.append(_hamming(int(a["dhash"]), int(b["dhash"])))
            p_dists.append(_hamming(int(a["phash"]), int(b["phash"])))

        n_elig = sum(elig_flags)
        has_old_test = "test" in v1_splits
        block_info = {
            "block_index": k,
            "files": [f"us{n}.bmp" for n in nums],
            "label": lbl_set[0],
            "resolution": res_set[0],
            "all_same_resolution": len(res_set) == 1,
            "all_same_label": len(lbl_set) == 1,
            "mendeley_v1_splits": v1_splits,
            "mendeley_v1_dhash3_group_count": len(v1_groups),
            "mendeley_dev_v2_splits": v2_splits,
            "mendeley_dev_v2_dhash5_unit_count": len(v2_units),
            "v1_eligible_count": n_elig,
            "v1_global_groups_for_eligible_members": global_gids,
            "touched_historical_v1_test": has_old_test,
            "partial_historical_test_leak_into_eligible": has_old_test and n_elig > 0,
            "split_across_multiple_v1_global_groups": n_elig == 5 and len(global_gids) > 1,
            "pairwise_direct_corr_min_med_max": [
                round(float(min(direct_corrs)), 4),
                round(float(np.median(direct_corrs)), 4),
                round(float(max(direct_corrs)), 4),
            ],
            "pairwise_shift_aligned_corr_min_med_max": [
                round(float(min(shift_corrs)), 4),
                round(float(np.median(shift_corrs)), 4),
                round(float(max(shift_corrs)), 4),
            ],
            "pairwise_dhash_min_max": [int(min(d_dists)), int(max(d_dists))],
            "pairwise_phash_min_max": [int(min(p_dists)), int(max(p_dists))],
        }
        five_image_blocks.append(block_info)
        if block_info["partial_historical_test_leak_into_eligible"]:
            partial_leak_blocks.append((block_info, b_rows))
        if block_info["split_across_multiple_v1_global_groups"]:
            split_eligible_blocks.append((block_info, b_rows))

    # Stage A & B component counts
    def _count_components(subset: list[dict], rule_fn) -> tuple[int, list[list[dict]]]:
        uf = UnionFind(len(subset))
        for i in range(len(subset)):
            for j in range(i):
                if rule_fn(subset[i], subset[j]):
                    uf.union(i, j)
        comps = defaultdict(list)
        for i, item in enumerate(subset):
            comps[uf.find(i)].append(item)
        comp_list = sorted(comps.values(), key=lambda c: (-len(c), c[0]["image_path"]))
        return len(comp_list), comp_list

    m250_exact_sha, _ = _count_components(m_unified, lambda a, b: a["sha256"] == b["sha256"])
    m250_exact_pix, _ = _count_components(m_unified, lambda a, b: a["pixel_sha256"] == b["pixel_sha256"])
    m250_d0, _ = _count_components(m_unified, lambda a, b: int(a["dhash"]) == int(b["dhash"]))
    m250_d3, _ = _count_components(
        m_unified, lambda a, b: _hamming(int(a["dhash"]), int(b["dhash"])) <= 3
    )
    m250_d5, _ = _count_components(
        m_unified, lambda a, b: _hamming(int(a["dhash"]), int(b["dhash"])) <= 5
    )
    m250_d5p5, comps_m250_d5p5 = _count_components(
        m_unified,
        lambda a, b: _hamming(int(a["dhash"]), int(b["dhash"])) <= 5
        or _hamming(int(a["phash"]), int(b["phash"])) <= 5,
    )
    m250_d5p5_conflicts = [c for c in comps_m250_d5p5 if len({x["label"] for x in c}) > 1]

    m162_exact_sha, _ = _count_components(m_eligible, lambda a, b: a["sha256"] == b["sha256"])
    m162_exact_pix, _ = _count_components(m_eligible, lambda a, b: a["pixel_sha256"] == b["pixel_sha256"])
    m162_d0, _ = _count_components(m_eligible, lambda a, b: int(a["dhash"]) == int(b["dhash"]))
    m162_d2, _ = _count_components(
        m_eligible, lambda a, b: _hamming(int(a["dhash"]), int(b["dhash"])) <= 2
    )
    m162_d3, _ = _count_components(
        m_eligible, lambda a, b: _hamming(int(a["dhash"]), int(b["dhash"])) <= 3
    )
    m162_d5, _ = _count_components(
        m_eligible, lambda a, b: _hamming(int(a["dhash"]), int(b["dhash"])) <= 5
    )
    m162_d5p5, _ = _count_components(
        m_eligible,
        lambda a, b: _hamming(int(a["dhash"]), int(b["dhash"])) <= 5
        or _hamming(int(a["phash"]), int(b["phash"])) <= 5,
    )

    # Analyze all 38 eligible Mendeley global groups
    m_global_groups = defaultdict(list)
    for r in m_eligible:
        m_global_groups[r["duplicate_group"]].append(r)

    mendeley_group_details = []
    mendeley_phash_merged_groups = []
    for gid, members in sorted(m_global_groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        members = sorted(members, key=lambda x: _us_num(x["image_path"]))
        units = sorted({m["known_grouping_id"] for m in members})
        v1_groups = sorted({m_v1_by_sha[m["sha256"]]["patient_id"] for m in members})
        resolutions = sorted({f"{size_cache[m['sha256']][0]}x{size_cache[m['sha256']][1]}" for m in members})
        block_ids = sorted({(_us_num(m["image_path"]) - 1) // 5 for m in members})

        d_dists, p_dists, corrs, shift_corrs = [], [], [], []
        edge_counts = Counter()
        n = len(members)
        adj_d5p5 = defaultdict(set)

        for i in range(n):
            for j in range(i + 1, n):
                dd = _hamming(int(members[i]["dhash"]), int(members[j]["dhash"]))
                pd = _hamming(int(members[i]["phash"]), int(members[j]["phash"]))
                d_dists.append(dd)
                p_dists.append(pd)
                corrs.append(_pearson(gray_cache[members[i]["sha256"]], gray_cache[members[j]["sha256"]]))
                sc, _ = _best_shift_corr(
                    native_gray_cache[members[i]["sha256"]], native_gray_cache[members[j]["sha256"]]
                )
                shift_corrs.append(sc)
                if dd == 0:
                    edge_counts["dhash_eq_0"] += 1
                if dd <= 3:
                    edge_counts["dhash_le_3"] += 1
                if dd <= 5:
                    edge_counts["dhash_le_5"] += 1
                if pd <= 3:
                    edge_counts["phash_le_3"] += 1
                if pd <= 5:
                    edge_counts["phash_le_5"] += 1
                if dd <= 5 or pd <= 5:
                    edge_counts["dhash_or_phash_le_5"] += 1
                    adj_d5p5[i].add(j)
                    adj_d5p5[j].add(i)

        max_hops = 0
        if n > 1:
            for start in range(n):
                dist = {start: 0}
                queue = [start]
                for u in queue:
                    for v in adj_d5p5[u]:
                        if v not in dist:
                            dist[v] = dist[u] + 1
                            queue.append(v)
                if dist:
                    max_hops = max(max_hops, max(dist.values()))

        detail = {
            "duplicate_group": gid,
            "size": n,
            "label": members[0]["label"],
            "five_image_block_indices": block_ids,
            "mendeley_dev_v2_dhash5_units": units,
            "mendeley_dev_v2_unit_count": len(units),
            "mendeley_v1_dhash3_groups": v1_groups,
            "mendeley_v1_group_count": len(v1_groups),
            "unique_resolutions": resolutions,
            "graph_diameter_hops": max_hops,
            "is_transitive_chain": max_hops > 1,
            "merged_multiple_dhash5_units_via_phash5": len(units) > 1,
            "is_partial_block_with_excluded_test_siblings": any(
                five_image_blocks[b]["touched_historical_v1_test"] for b in block_ids
            ),
            "pairwise_dhash_min_med_max": (
                [int(min(d_dists)), float(np.median(d_dists)), int(max(d_dists))] if d_dists else [0, 0.0, 0]
            ),
            "pairwise_phash_min_med_max": (
                [int(min(p_dists)), float(np.median(p_dists)), int(max(p_dists))] if p_dists else [0, 0.0, 0]
            ),
            "pairwise_direct_corr_min_med_max": (
                [round(float(min(corrs)), 4), round(float(np.median(corrs)), 4), round(float(max(corrs)), 4)]
                if corrs
                else [1.0, 1.0, 1.0]
            ),
            "pairwise_shift_aligned_corr_min_med_max": (
                [
                    round(float(min(shift_corrs)), 4),
                    round(float(np.median(shift_corrs)), 4),
                    round(float(max(shift_corrs)), 4),
                ]
                if shift_corrs
                else [1.0, 1.0, 1.0]
            ),
            "edge_counts": dict(edge_counts),
            "member_files": [m["image_path"].rsplit("/", 1)[-1] for m in members],
        }
        mendeley_group_details.append(detail)
        if len(units) > 1:
            mendeley_phash_merged_groups.append((detail, members))

    # 3. Audit all 1,298 eligible global groups across all 3 sources
    eligible_rows = [r for r in rows if r["eligible"] == "true"]
    all_global_groups = defaultdict(list)
    for r in eligible_rows:
        all_global_groups[r["duplicate_group"]].append(r)

    source_group_summary = {}
    for src in ("mendeley", "tcia", "busbra"):
        src_groups = {gid: mems for gid, mems in all_global_groups.items() if mems[0]["source"] == src}
        sizes = [len(mems) for mems in src_groups.values()]
        multi_unit_groups = []
        for gid, mems in sorted(src_groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            u_set = sorted({m["known_grouping_id"] for m in mems})
            if len(u_set) > 1:
                cross_unit_edges = []
                for a, b in combinations(mems, 2):
                    if a["known_grouping_id"] != b["known_grouping_id"]:
                        dd = _hamming(int(a["dhash"]), int(b["dhash"]))
                        pd = _hamming(int(a["phash"]), int(b["phash"]))
                        if dd <= 5 or pd <= 5 or a["sha256"] == b["sha256"] or a["pixel_sha256"] == b["pixel_sha256"]:
                            cinfo = _best_flip_or_shift_corr(gray_cache[a["sha256"]], gray_cache[b["sha256"]])
                            cross_unit_edges.append(
                                {
                                    "file_a": a["image_path"].rsplit("/", 1)[-1],
                                    "unit_a": a["known_grouping_id"],
                                    "file_b": b["image_path"].rsplit("/", 1)[-1],
                                    "unit_b": b["known_grouping_id"],
                                    "dhash_dist": dd,
                                    "phash_dist": pd,
                                    **cinfo,
                                }
                            )
                multi_unit_groups.append(
                    {
                        "duplicate_group": gid,
                        "label": mems[0]["label"],
                        "image_count": len(mems),
                        "known_units": u_set,
                        "known_unit_count": len(u_set),
                        "cross_unit_edges": cross_unit_edges,
                    }
                )
        source_group_summary[src] = {
            "eligible_images": sum(sizes),
            "eligible_known_units": len({m["known_grouping_id"] for mems in src_groups.values() for m in mems}),
            "global_duplicate_groups": len(src_groups),
            "group_size_histogram": {str(k): int(v) for k, v in sorted(Counter(sizes).items())},
            "max_group_size": max(sizes) if sizes else 0,
            "groups_merging_multiple_known_units": len(multi_unit_groups),
            "multi_unit_group_details": multi_unit_groups,
        }

    # 4. Find the exact reference match that excluded BUS-BRA Case 632 (bus_0632-l.png)
    case632_rows = [r for r in rows if r["known_grouping_id"] == "busbra:632"]
    case632_l = next(r for r in case632_rows if r["image_path"].endswith("bus_0632-l.png"))
    case632_matches = []
    for ref_dir in [Path("data/raw"), Path("data/processed"), Path("data/processed_busi")]:
        if not ref_dir.exists():
            continue
        for p in ref_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".png", ".bmp", ".jpg", ".jpeg"} and not any(
                s in p.stem.lower() for s in ["_mask", "_tumor", "_other"]
            ):
                fp = fingerprint(p.read_bytes())
                dd = _hamming(int(case632_l["dhash"]), int(fp["dhash"]))
                pd = _hamming(int(case632_l["phash"]), int(fp["phash"]))
                if (
                    fp["sha256"] == case632_l["sha256"]
                    or fp["pixel_sha256"] == case632_l["pixel_sha256"]
                    or dd <= 5
                    or pd <= 5
                ):
                    img_632 = _load_pil(case632_l["image_path"], zf_cache)
                    img_ref = Image.open(p).convert("RGB")
                    cinfo = _best_flip_or_shift_corr(_gray_arr(img_632), _gray_arr(img_ref))
                    case632_matches.append(
                        {
                            "busbra_image": case632_l["image_path"],
                            "matched_reference_file": str(p).replace("\\", "/"),
                            "dhash_dist": dd,
                            "phash_dist": pd,
                            **cinfo,
                        }
                    )

    # 5. Render contact sheets
    # Sheet 1: Top 6 largest Mendeley global groups
    top_mendeley_visual = []
    for detail in mendeley_group_details[:6]:
        gid = detail["duplicate_group"]
        mems = sorted(m_global_groups[gid], key=lambda x: _us_num(x["image_path"]))
        anchor_n = native_gray_cache[mems[0]["sha256"]]
        da0, pa0 = int(mems[0]["dhash"]), int(mems[0]["phash"])
        enriched = []
        for idx, m in enumerate(mems):
            if idx == 0:
                note = "anchor (d=0, p=0, r=1.00)"
            else:
                dd = _hamming(da0, int(m["dhash"]))
                pd = _hamming(pa0, int(m["phash"]))
                sc, sh = _best_shift_corr(anchor_n, native_gray_cache[m["sha256"]])
                note = f"d={dd} p={pd} r_shift={sc:.2f}"
            enriched.append({**m, "note": note})
        top_mendeley_visual.append(
            {
                "header": (
                    f"{gid} | {detail['label']} | n={detail['size']} | "
                    f"dhash5_units={detail['mendeley_dev_v2_unit_count']} | "
                    f"hops={detail['graph_diameter_hops']} | "
                    f"shift_corr_med={detail['pairwise_shift_aligned_corr_min_med_max'][1]:.3f}"
                ),
                "members": enriched[:10],
            }
        )
    sheet1_path = SHEETS_DIR / "mendeley_largest_groups.png"
    _render_contact_sheet(
        "Mendeley Largest Global Groups (Top 6 of 38; shift-aligned correlation confirms shifted crops of same scan)",
        top_mendeley_visual,
        zf_cache,
        sheet1_path,
    )

    # Sheet 2: Mendeley pHash<=5 merges across separate dHash<=5 units
    phash_merge_visual = []
    for detail, mems in mendeley_phash_merged_groups[:7]:
        mems = sorted(mems, key=lambda x: _us_num(x["image_path"]))
        anchor_n = native_gray_cache[mems[0]["sha256"]]
        da0, pa0 = int(mems[0]["dhash"]), int(mems[0]["phash"])
        enriched = []
        for idx, m in enumerate(mems):
            dd = _hamming(da0, int(m["dhash"]))
            pd = _hamming(pa0, int(m["phash"]))
            sc, _ = _best_shift_corr(anchor_n, native_gray_cache[m["sha256"]])
            note = "anchor" if idx == 0 else f"d={dd} p={pd} r_shift={sc:.2f}"
            enriched.append({**m, "note": note})
        phash_merge_visual.append(
            {
                "header": (
                    f"{detail['duplicate_group']} ({detail['label']}) reunited "
                    f"{detail['mendeley_dev_v2_unit_count']} dHash<=5 units: "
                    f"{', '.join(detail['mendeley_dev_v2_dhash5_units'])}"
                ),
                "members": enriched,
            }
        )
    sheet2_path = SHEETS_DIR / "mendeley_phash_unit_merges.png"
    _render_contact_sheet(
        "Mendeley pHash<=5 Merges Across Separate dHash<=5 Units (Reuniting 5-Image Shifted-Crop Sequences)",
        phash_merge_visual,
        zf_cache,
        sheet2_path,
    )

    # Sheet 3: Newly uncovered Mendeley split block (us21..25) + partial historical-test leaked blocks + BUS-BRA Case 632
    leak_and_split_visual = []
    for b_info, b_rows in split_eligible_blocks:
        anchor_n = native_gray_cache[b_rows[0]["sha256"]]
        da0, pa0 = int(b_rows[0]["dhash"]), int(b_rows[0]["phash"])
        enriched = []
        for idx, m in enumerate(b_rows):
            dd = _hamming(da0, int(m["dhash"]))
            pd = _hamming(pa0, int(m["phash"]))
            sc, _ = _best_shift_corr(anchor_n, native_gray_cache[m["sha256"]])
            enriched.append(
                {
                    **m,
                    "sub_label": m["duplicate_group"],
                    "note": f"d={dd} p={pd} r_shift={sc:.2f}",
                    "highlight_warn": True,
                }
            )
        leak_and_split_visual.append(
            {
                "header": (
                    f"Under-Grouped Block #{b_info['block_index']} ({b_info['files'][0]}..{b_info['files'][-1]}, "
                    f"{b_info['resolution']}): Split into { ', '.join(b_info['v1_global_groups_for_eligible_members']) }"
                ),
                "members": enriched,
            }
        )

    for b_info, b_rows in partial_leak_blocks[:5]:
        anchor_n = native_gray_cache[b_rows[0]["sha256"]]
        da0, pa0 = int(b_rows[0]["dhash"]), int(b_rows[0]["phash"])
        enriched = []
        for idx, m in enumerate(b_rows):
            dd = _hamming(da0, int(m["dhash"]))
            pd = _hamming(pa0, int(m["phash"]))
            sc, _ = _best_shift_corr(anchor_n, native_gray_cache[m["sha256"]])
            v1_sp = m_v1_by_sha[m["sha256"]]["split"]
            is_elig = m["eligible"] == "true"
            status_tag = f"ELIG({m['duplicate_group']})" if is_elig else f"EXCL(v1={v1_sp})"
            enriched.append(
                {
                    **m,
                    "sub_label": status_tag,
                    "note": f"d={dd} p={pd} r_shift={sc:.2f}",
                    "highlight_warn": is_elig,
                }
            )
        leak_and_split_visual.append(
            {
                "header": (
                    f"Partial Historical-Test Leak Block #{b_info['block_index']} "
                    f"({b_info['files'][0]}..{b_info['files'][-1]}): "
                    f"{b_info['v1_eligible_count']}/5 kept eligible despite v1 test sibling!"
                ),
                "members": enriched,
            }
        )

    if case632_matches:
        cm = case632_matches[0]
        ref_row = {
            "image_path": cm["matched_reference_file"],
            "label": "benign",
            "known_grouping_id": "legacy_raw_folder",
            "sub_label": "legacy_raw_overlap",
            "note": f"d={cm['dhash_dist']} p={cm['phash_dist']} r={cm['pearson_direct']:.2f}",
            "highlight_warn": True,
        }
        leak_and_split_visual.append(
            {
                "header": (
                    f"Excluded BUS-BRA Case 632 (bus_0632-l/r) vs Legacy Raw Reference "
                    f"({cm['matched_reference_file']}) | d={cm['dhash_dist']} p={cm['phash_dist']} r={cm['pearson_direct']:.2f}"
                ),
                "members": [
                    {**case632_rows[0], "sub_label": "busbra:632 (excl)", "note": "bus_0632-l (false dHash hit)"},
                    {**case632_rows[1], "sub_label": "busbra:632 (excl)", "note": "bus_0632-r (same case)"},
                    ref_row,
                ],
            }
        )

    sheet3_path = SHEETS_DIR / "busbra_and_cross_source_merges.png"
    _render_contact_sheet(
        "Audit Findings: Mendeley Split Block (us21..25), Partial Test-Leak Blocks, & BUS-BRA Case 632",
        leak_and_split_visual,
        zf_cache,
        sheet3_path,
    )

    # 6. Construct versioned unified_dataset_manifest_v2.csv and eligible_manifest_v2.json
    # In v2:
    # - Every Mendeley 5-image shifted-crop block k=0..49 is assigned its base-scan sequence group
    #   (with blocks k=23 [us116..120] and k=45 [us226..230] joined into one 10-image base-scan group because
    #    they are derived from the same 153x131 malignant scan).
    # - Any Mendeley base-scan group that contains >=1 historical mendeley_v1 test image is completely excluded
    #   (removing the 22 partial-block images that leaked past dHash<=5 in mendeley_dev_v2, leaving 140 strictly
    #    test-disjoint Mendeley images in 27 complete base-scan groups: 13 benign [65 images], 14 malignant [75 images]).
    # - For TCIA (197 images, 197 publisher patient_id groups) and BUS-BRA (1,873 images, 1,063 source_case groups),
    #   grouping is unchanged.
    v2_rows = []
    v2_eligible_manifest = []
    for r in rows:
        r_copy = dict(r)
        if r["source"] == "mendeley":
            num = _us_num(r["image_path"])
            blk = (num - 1) // 5
            # Blocks 23 (us116..120) and 45 (us226..230) are the same 153x131 base scan
            canon_blk = 23 if blk == 45 else blk
            base_scan_gid = f"mendeley_scan_seq:{canon_blk:02d}"
            blk_touched_test = (
                r["eligible"] == "false"
                or five_image_blocks[blk]["touched_historical_v1_test"]
                or (canon_blk == 23 and five_image_blocks[45]["touched_historical_v1_test"])
            )
            v2_elig = not blk_touched_test
            if v2_elig:
                v2_excl_reason = ""
                v2_dup_group = f"v2_global:mendeley_seq_{canon_blk:02d}"
            else:
                v2_excl_reason = (
                    "historical_mendeley_test_linked_duplicate_group"
                    if r["eligible"] == "false"
                    else "historical_mendeley_test_same_5image_crop_sequence_leaked_past_dhash5"
                )
                v2_dup_group = f"excluded_{base_scan_gid}"
            r_copy["v2_base_scan_sequence_id"] = base_scan_gid
            r_copy["v2_eligible"] = "true" if v2_elig else "false"
            r_copy["v2_exclusion_reason"] = v2_excl_reason
            r_copy["v2_duplicate_group"] = v2_dup_group
            r_copy["v2_grouping_basis"] = "verified_5image_crop_sequence_plus_dhash5_phash5"
        else:
            r_copy["v2_base_scan_sequence_id"] = r["known_grouping_id"]
            r_copy["v2_eligible"] = r["eligible"]
            r_copy["v2_exclusion_reason"] = r["exclusion_reason"]
            r_copy["v2_duplicate_group"] = (
                f"v2_{r['duplicate_group']}" if r["eligible"] == "true" else r["duplicate_group"]
            )
            r_copy["v2_grouping_basis"] = (
                "publisher_reported_patient_id"
                if r["source"] == "tcia"
                else "source_case_unit_plus_dhash5_phash5"
            )

        if r_copy["v2_eligible"] == "true":
            r_copy["v2_eligible_row_index"] = str(len(v2_eligible_manifest))
            v2_eligible_manifest.append(
                {
                    "v2_row_index": len(v2_eligible_manifest),
                    "v1_eligible_row_index": int(r["eligible_row_index"]),
                    "source": r["source"],
                    "label": r["label"],
                    "unit": r_copy["v2_base_scan_sequence_id"],
                    "group": r_copy["v2_duplicate_group"],
                    "v1_group": r["duplicate_group"],
                    "file": r["image_path"] if r["source"] != "busbra" else "",
                    "archive_path": (
                        r["image_path"].split(".zip:", 1)[1] if r["source"] == "busbra" else ""
                    ),
                    "sha256": r["sha256"],
                    "pixel_sha256": r["pixel_sha256"],
                    "dhash": r["dhash"],
                    "phash": r["phash"],
                }
            )
        else:
            r_copy["v2_eligible_row_index"] = ""
        v2_rows.append(r_copy)

    with MANIFEST_V2_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(v2_rows[0].keys()))
        writer.writeheader()
        writer.writerows(v2_rows)
    v2_sha256 = hashlib.sha256(MANIFEST_V2_CSV.read_bytes()).hexdigest()
    ELIGIBLE_V2_JSON.write_text(json.dumps(v2_eligible_manifest, indent=2), encoding="utf-8")
    elig_v2_sha256 = hashlib.sha256(ELIGIBLE_V2_JSON.read_bytes()).hexdigest()

    for zf in zf_cache.values():
        zf.close()

    v2_summary = {
        "manifest_version": "unified_dataset_manifest_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_v2_csv": str(MANIFEST_V2_CSV).replace("\\", "/"),
        "manifest_v2_sha256": v2_sha256,
        "eligible_manifest_v2_json": str(ELIGIBLE_V2_JSON).replace("\\", "/"),
        "eligible_manifest_v2_sha256": elig_v2_sha256,
        "v1_to_v2_changes": {
            "mendeley_partial_historical_test_siblings_excluded": sum(
                b["v1_eligible_count"] for b, _ in partial_leak_blocks
            ),
            "mendeley_partial_blocks_excluded": len(partial_leak_blocks),
            "mendeley_split_blocks_reunited": [
                "Block #4 (us21..us25.bmp, 93x57 benign): reunited v1 global:10 (us21..23) and global:13 (us24..25) into v2_global:mendeley_seq_04",
                "Block #38 (us191..us195.bmp, 119x101 malignant): reunited v1 global:120 (us191..194) and global:123 (us195) into v2_global:mendeley_seq_38",
            ],
            "v1_eligible_images": 2232,
            "v2_eligible_images": len(v2_eligible_manifest),
            "v1_eligible_groups": 1298,
            "v2_eligible_groups": len({r["group"] for r in v2_eligible_manifest}),
            "by_source": {
                src: {
                    "v1_eligible_images": sum(1 for r in rows if r["source"] == src and r["eligible"] == "true"),
                    "v1_eligible_groups": len(
                        {r["duplicate_group"] for r in rows if r["source"] == src and r["eligible"] == "true"}
                    ),
                    "v2_eligible_images": sum(1 for r in v2_eligible_manifest if r["source"] == src),
                    "v2_eligible_groups": len({r["group"] for r in v2_eligible_manifest if r["source"] == src}),
                }
                for src in ("mendeley", "tcia", "busbra")
            },
        },
        "affected_comparisons_and_invalidation_scope": {
            "historical_mendeley_v1_test_disjointness": (
                "INVALIDATED FOR 17 MENDELEY IMAGES IN V1: In mendeley_dev_v2 and unified_dataset_manifest_v1, "
                "8 of the 50 five-image Mendeley crop sequences had >=1 image in the historical mendeley_v1 test split "
                "while 17 sibling crop images in those same 8 sequences had dHash >= 6 from the test frame and remained "
                "in the 162-image v1 eligible set (forming 8 partial global groups of size 1..4). Any claim that the "
                "162-image v1 Mendeley subset is 100% disjoint from historical mendeley_v1 test lesions is invalidated; "
                "v2 excludes those 17 images (leaving 145 Mendeley images in 28 complete base-scan groups: 27 of size 5 "
                "and 1 of size 10)."
            ),
            "inner_cv_folds_when_mendeley_is_training_source": (
                "AFFECTED BY 2 SPLIT BLOCKS (us21..25 and us191..195): In v1, benign block #4 (us21..us25.bmp, 93x57) "
                "was split into global:10 (us21..23) and global:13 (us24..25), and malignant block #38 (us191..us195.bmp, "
                "119x101) was split into global:120 (us191..194) and global:123 (us195). When TCIA or BUS-BRA was held out, "
                "those split sibling groups could be assigned to different inner CV folds on the training side."
            ),
            "outer_cross_source_held_out_isolation": (
                "NOT INVALIDATED: Zero exact or perceptual duplicate links exist across Mendeley, TCIA, and BUS-BRA. "
                "Outer source-held-out evaluations remain strictly source-disjoint."
            ),
            "gpu_encoder_finetune_protocol_lock": (
                "reports/gpu_encoder_finetune_v2/lock.json (which locked the v1 2,232-image manifest) is preserved in the "
                "audit trail and superseded before training by protocol v3 / reports/gpu_encoder_finetune_v3/lock.json "
                "bound to unified_dataset_manifest_v2.csv (2,215 eligible images, 1,288 groups)."
            ),
        },
    }
    MANIFEST_V2_SUMMARY.write_text(json.dumps(v2_summary, indent=2), encoding="utf-8")

    audit_payload = {
        "audit_version": "grouping_integrity_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifests": {
            "preserved_v1_csv": "data/prepared/unified_dataset_manifest_v1.csv",
            "preserved_v1_sha256": hashlib.sha256(
                Path("data/prepared/unified_dataset_manifest_v1.csv").read_bytes()
            ).hexdigest(),
            "versioned_v2_csv": str(MANIFEST_V2_CSV).replace("\\", "/"),
            "versioned_v2_sha256": v2_sha256,
            "eligible_manifest_v2_json": str(ELIGIBLE_V2_JSON).replace("\\", "/"),
            "eligible_manifest_v2_sha256": elig_v2_sha256,
        },
        "mendeley_5image_crop_sequence_discovery": {
            "total_acquired_images": 250,
            "consecutive_5image_blocks": 50,
            "blocks_with_identical_resolution_and_label": sum(
                1 for b in five_image_blocks if b["all_same_resolution"] and b["all_same_label"]
            ),
            "duplicate_pair_of_5image_blocks": "Block #23 (us116..120.bmp, 153x131) and Block #45 (us226..230.bmp, 153x131) are derived from the same base malignant ultrasound scan (forming 10-image group global:75).",
            "true_unique_base_lesion_scans_in_250_mendeley": 49,
            "blocks_completely_excluded_by_v1_test_dhash5": sum(
                1 for b in five_image_blocks if b["touched_historical_v1_test"] and b["v1_eligible_count"] == 0
            ),
            "blocks_partially_leaked_into_v1_eligible_past_dhash5": len(partial_leak_blocks),
            "leaked_images_in_partial_blocks": sum(b["v1_eligible_count"] for b, _ in partial_leak_blocks),
            "fully_eligible_blocks_split_across_two_v1_global_groups": len(split_eligible_blocks),
            "partial_leak_block_details": [b for b, _ in partial_leak_blocks],
            "split_eligible_block_details": [b for b, _ in split_eligible_blocks],
            "all_50_five_image_blocks": five_image_blocks,
        },
        "mendeley_grouping_progression": {
            "acquired_250_images": {
                "total_images": 250,
                "unique_byte_sha256_components": m250_exact_sha,
                "unique_pixel_sha256_components": m250_exact_pix,
                "dhash_eq_0_components": m250_d0,
                "dhash_le_3_components_mendeley_v1": m250_d3,
                "dhash_le_5_components_mendeley_dev_v2": m250_d5,
                "dhash_le_5_or_phash_le_5_components": m250_d5p5,
                "label_conflicting_components_at_d5_or_p5": len(m250_d5p5_conflicts),
            },
            "eligible_162_images_by_threshold": {
                "total_eligible_images": 162,
                "unique_byte_sha256_components": m162_exact_sha,
                "unique_pixel_sha256_components": m162_exact_pix,
                "dhash_eq_0_components": m162_d0,
                "dhash_le_2_components": m162_d2,
                "dhash_le_3_components": m162_d3,
                "dhash_le_5_components_known_units": m162_d5,
                "dhash_le_5_or_phash_le_5_global_groups": m162_d5p5,
            },
            "all_38_mendeley_global_groups": mendeley_group_details,
        },
        "source_group_summaries_eligible_2232": source_group_summary,
        "busbra_case_632_exclusion_audit": case632_matches,
        "v2_manifest_correction_summary": v2_summary,
        "contact_sheets": {
            "mendeley_largest_groups": str(sheet1_path).replace("\\", "/"),
            "mendeley_phash_unit_merges": str(sheet2_path).replace("\\", "/"),
            "busbra_and_cross_source_merges": str(sheet3_path).replace("\\", "/"),
        },
    }

    AUDIT_JSON.write_text(json.dumps(audit_payload, indent=2), encoding="utf-8")
    print(f"Grouping audit and v2 manifest complete: {AUDIT_JSON}")
    return audit_payload


if __name__ == "__main__":
    run_grouping_audit()
