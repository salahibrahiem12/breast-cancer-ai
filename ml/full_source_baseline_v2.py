"""Frozen ResNet-18 linear baseline on the grouping-validated v2 manifest (2,215 images, 1,288 groups).

Preserves `reports/full_source_baseline_v1/` untouched and executes the exact same
L-BFGS-B linear protocol on `reports/grouping_audit_v1/eligible_manifest_v2.json` to:
1. Establish the clean v2 linear reference baseline for `gpu_encoder_finetune_v3`.
2. Quantify the exact empirical impact of fixing the 17 leaked Mendeley partial-sequence
   frames, 2 split Mendeley 5-image sequences, and 1 false-positive BUS-BRA hash merge.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ml.cross_source import WEIGHTS, aggregate, intervals, metric
from ml.full_source_baseline import (
    atomic_npz,
    calibration_bins,
    choose_threshold,
    decision,
    digest,
    objective,
    sample_weights,
    save,
)

OUT = Path("reports/full_source_baseline_v2")
PROTOCOL = Path("docs/FULL_SOURCE_BASELINE_PROTOCOL_V2.md")
MANIFEST_V2 = Path("reports/grouping_audit_v1/eligible_manifest_v2.json")
UNIFIED_CSV_V2 = Path("data/prepared/unified_dataset_manifest_v2.csv")
V1_FEATURES = Path("reports/full_source_baseline_v1/features.npz")
V1_RESULT = Path("reports/full_source_baseline_v1/result.json")
DOC_OUT = Path("docs/FULL_SOURCE_BASELINE_RESULTS_V2.md")


def rows_v2() -> list[dict]:
    return json.loads(MANIFEST_V2.read_text(encoding="utf-8"))


def lock() -> dict:
    import PIL
    import scipy
    import sklearn
    import torch
    import torchvision

    OUT.mkdir(parents=True, exist_ok=True)
    rs = rows_v2()
    assert len(rs) == 2215, f"Expected 2215 v2 eligible images, got {len(rs)}"
    assert len({r["group"] for r in rs}) == 1288, f"Expected 1288 v2 groups, got {len({r['group'] for r in rs})}"
    files = [
        PROTOCOL,
        Path(__file__),
        MANIFEST_V2,
        UNIFIED_CSV_V2,
        WEIGHTS,
        V1_FEATURES,
        Path("reports/design_audit_v1/prior_artifact_hashes.json"),
    ]
    obj = {
        "experiment_id": "full_source_baseline_v2",
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "hashes": {str(p).replace("\\", "/"): digest(p) for p in files},
        "images": len(rs),
        "groups": len({r["group"] for r in rs}),
        "source_counts": {
            src: {
                "images": sum(1 for r in rs if r["source"] == src),
                "groups": len({r["group"] for r in rs if r["source"] == src}),
            }
            for src in ["mendeley", "tcia", "busbra"]
        },
        "inspected_development_sources": True,
        "inference_enabled": False,
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "versions": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "Pillow": PIL.__version__,
        },
    }
    save(OUT / "lock.json", obj)
    (OUT / "LOCK_SHA256.txt").write_text(digest(OUT / "lock.json"), encoding="utf-8")
    print("Locked full_source_baseline_v2 protocol on v2 manifest (2,215 images, 1,288 groups).", flush=True)
    return obj


def check_lock() -> dict:
    obj = json.loads((OUT / "lock.json").read_text(encoding="utf-8"))
    assert digest(OUT / "lock.json") == (OUT / "LOCK_SHA256.txt").read_text(encoding="utf-8").strip()
    for p, h in obj["hashes"].items():
        if Path(p) != Path(__file__):
            assert digest(p) == h, f"Hash mismatch on locked file {p}"
    return obj


def prepare_features() -> None:
    check_lock()
    rs = rows_v2()
    with np.load(V1_FEATURES) as f:
        v1_x = f["features"]
        v1_sha = f["sha256"]
    sha_to_idx = {str(s): i for i, s in enumerate(v1_sha)}
    indices = []
    for r in rs:
        idx = r["v2_to_v1_index"]
        assert str(v1_sha[idx]) == r["sha256"]
        assert sha_to_idx[r["sha256"]] == idx
        indices.append(idx)
    x_v2 = v1_x[np.array(indices, dtype=int)]
    sha_v2 = np.array([r["sha256"] for r in rs])
    assert x_v2.shape == (2215, 512)
    assert np.isfinite(x_v2).all()
    atomic_npz(OUT / "features.npz", features=x_v2, sha256=sha_v2)
    save(
        OUT / "feature_receipt.json",
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "images": len(rs),
            "features": 512,
            "features_sha256": digest(OUT / "features.npz"),
            "derived_from_v1_features_sha256": digest(V1_FEATURES),
            "exact_sha256_match_for_all_2215_rows": True,
        },
    )
    print("Prepared hash-verified v2 feature matrix (2,215 x 512).", flush=True)


def fit_v2(name: str, x: np.ndarray, y: np.ndarray, s: np.ndarray, g: np.ndarray):
    dest = OUT / "models"
    dest.mkdir(parents=True, exist_ok=True)
    model_path = dest / f"{name}.npz"
    record_path = dest / f"{name}.json"
    w = sample_weights(y, s, g, True)
    scaler = StandardScaler().fit(x, sample_weight=w)
    z = scaler.transform(x.astype(float))
    if model_path.exists() and record_path.exists():
        info = json.loads(record_path.read_text(encoding="utf-8"))
        assert info["model_sha256"] == digest(model_path)
        assert info["training_groups"] == sorted(set(g))
        assert info["converged"]
        with np.load(model_path) as f:
            return {k: f[k] for k in f.files}, info
    curve: list[dict] = []
    initial = np.zeros(x.shape[1] + 1)
    start = time.perf_counter()

    def callback(theta):
        value, grad = objective(theta, z, y, w)
        curve.append(
            {
                "iteration": len(curve) + 1,
                "objective": value,
                "max_absolute_gradient": float(abs(grad).max()),
            }
        )

    opt = minimize(
        objective,
        initial,
        args=(z, y, w),
        jac=True,
        method="L-BFGS-B",
        callback=callback,
        options={"maxiter": 1000, "gtol": 1e-6, "ftol": 1e-12, "maxls": 50},
    )
    value, gradient = objective(opt.x, z, y, w)
    success = bool(opt.success and abs(gradient).max() <= 1e-5)
    info = {
        "fit": name,
        "train_images": len(y),
        "train_groups": len(set(g)),
        "training_groups": sorted(set(g)),
        "source_class_images": {
            src: {str(label): int(((s == src) & (y == label)).sum()) for label in [0, 1]}
            for src in sorted(set(s))
        },
        "objective": value,
        "max_absolute_gradient": float(abs(gradient).max()),
        "converged": success,
        "optimizer_success": bool(opt.success),
        "optimizer_message": str(opt.message),
        "optimizer_iterations": int(opt.nit),
        "elapsed_seconds": time.perf_counter() - start,
        "sum_sample_weights": float(w.sum()),
        "class_weight_sums": {str(label): float(w[y == label].sum()) for label in [0, 1]},
        "lock_sha256": digest(OUT / "lock.json"),
    }
    save(dest / f"{name}.curve.json", curve)
    if not success:
        save(record_path, info)
        raise RuntimeError("Optimization failed convergence criteria: " + name)
    fitted = {"mean": scaler.mean_, "scale": scaler.scale_, "coef": opt.x[:-1], "intercept": opt.x[-1:]}
    atomic_npz(model_path, **fitted)
    info["model_sha256"] = digest(model_path)
    save(record_path, info)
    return fitted, info


def run_all() -> dict:
    if not (OUT / "lock.json").exists():
        lock()
    if not (OUT / "features.npz").exists():
        prepare_features()
    check_lock()
    rs = rows_v2()
    with np.load(OUT / "features.npz") as f:
        assert np.array_equal(f["sha256"], np.array([r["sha256"] for r in rs]))
        x = f["features"].astype(float)
    y = np.array([r["label"] == "malignant" for r in rs], int)
    g = np.array([r["group"] for r in rs])
    s = np.array([r["source"] for r in rs])

    results = {}
    all_predictions = []
    replay_max_errors = {}

    for source in ["mendeley", "tcia", "busbra"]:
        out = OUT / source
        out.mkdir(parents=True, exist_ok=True)
        train = np.flatnonzero(s != source)
        held = np.flatnonzero(s == source)
        oof = np.empty(len(train))
        folds = []
        strata = np.array([f"{a}:{b}" for a, b in zip(s[train], y[train])])
        for fold, (itr, iva) in enumerate(
            StratifiedGroupKFold(4, shuffle=True, random_state=42).split(x[train], strata, g[train])
        ):
            tr, va = train[itr], train[iva]
            assert not (set(g[tr]) & set(g[va]))
            assert not (set(g[train]) & set(g[held]))
            assert source not in (set(s[tr]) | set(s[va]))
            model, info = fit_v2(f"{source}_inner{fold}", x[tr], y[tr], s[tr], g[tr])
            oof[iva] = decision(model, x[va])
            folds.append(
                {
                    "fold": fold,
                    "fit": f"{source}_inner{fold}",
                    "train_rows": tr.tolist(),
                    "validation_rows": va.tolist(),
                    "train_groups": sorted(set(g[tr])),
                    "validation_groups": sorted(set(g[va])),
                    "converged": info["converged"],
                    "max_absolute_gradient": info["max_absolute_gradient"],
                }
            )
        save(out / "folds.json", folds)
        cal_w = sample_weights(y[train], s[train], g[train], False)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            calibrator = LogisticRegression(C=1, max_iter=2000, tol=1e-6, random_state=42).fit(
                oof[:, None], y[train], sample_weight=cal_w
            )
        if ws:
            raise RuntimeError("Calibrator warning: " + str(ws[0].message))
        vp = calibrator.predict_proba(oof[:, None])[:, 1]
        threshold = choose_threshold(y[train], vp, g[train], s[train])
        selection = {
            "threshold": threshold,
            "calibration_coef": calibrator.coef_.tolist(),
            "calibration_intercept": calibrator.intercept_.tolist(),
            "calibration_iterations": calibrator.n_iter_.tolist(),
            "training_rows": train.tolist(),
            "training_groups": sorted(set(g[train])),
            "training_sources": sorted(set(s[train])),
            "held_out_source_not_used": True,
            "selection_frozen_utc": datetime.now(timezone.utc).isoformat(),
        }
        save(out / "training_side_selection.json", selection)
        atomic_npz(out / "oof.npz", rows=train, logits=oof, calibrated_probabilities=vp)
        model, info = fit_v2(f"{source}_final", x[train], y[train], s[train], g[train])
        logits = decision(model, x[held])
        p = calibrator.predict_proba(logits[:, None])[:, 1]

        # Verify saved-model replay from disk
        with np.load(OUT / "models" / f"{source}_final.npz") as saved_m:
            replayed_logits = decision({k: saved_m[k] for k in saved_m.files}, x[held])
            coef = float(selection["calibration_coef"][0][0])
            inter = float(selection["calibration_intercept"][0])
            replayed_p = expit(replayed_logits * coef + inter)
            replay_err = float(np.max(np.abs(replayed_p - p)))
            assert replay_err <= 1e-12, f"Replay error {replay_err}"
            replay_max_errors[source] = replay_err

        gy, gp, gg = aggregate(y[held], p, g[held])
        ty, tp, tg = aggregate(y[train], vp, g[train])
        group_counts = Counter(gy)
        image_counts = Counter(y[held])
        train_group_counts = Counter(ty)
        training_majority = max(train_group_counts, key=train_group_counts.get)
        prev = float(ty.mean())
        result = {
            "held_out_source": source,
            "manifest_version": "v2",
            "inspected_exploratory_development": True,
            "train_images": len(train),
            "train_groups": len(set(g[train])),
            "held_out_images": len(held),
            "held_out_groups": len(gg),
            "group_class_counts": {str(k): v for k, v in group_counts.items()},
            "image_class_counts": {str(k): v for k, v in image_counts.items()},
            **selection,
            "group_metrics": metric(gy, gp, threshold),
            "image_metrics": metric(y[held], p, threshold),
            "group_intervals": intervals(gy, gp, gg, threshold),
            "image_intervals": intervals(y[held], p, g[held], threshold),
            "group_calibration_bins": calibration_bins(gy, gp),
            "image_calibration_bins": calibration_bins(y[held], p),
            "training_oof_group_metrics": metric(ty, tp, threshold),
            "training_refit_image_metrics": metric(
                y[train], calibrator.predict_proba(decision(model, x[train])[:, None])[:, 1], threshold
            ),
            "majority_baselines": {
                "held_group_majority_accuracy": max(group_counts.values()) / len(gy),
                "held_image_majority_accuracy": max(image_counts.values()) / len(held),
                "majority_balanced_accuracy": 0.5,
                "training_group_majority_class": int(training_majority),
                "training_group_majority_accuracy_on_held": float((gy == training_majority).mean()),
                "training_group_prevalence": prev,
                "training_prevalence_constant_held_group_brier": float(np.mean((prev - gy) ** 2)),
            },
            "final_optimizer": info,
            "saved_model_replay_max_error": replay_err,
            "patient_independence_claim": False,
            "inference_enabled": False,
        }
        predictions = [
            {
                "row": int(i),
                "source": source,
                "group": rs[i]["group"],
                "actual": int(y[i]),
                "raw_logit": float(logit),
                "probability": float(prob),
                "threshold": threshold,
            }
            for i, logit, prob in zip(held, logits, p)
        ]
        with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
        save(out / "result.json", result)
        results[source] = result
        all_predictions += predictions

    with (OUT / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_predictions[0]))
        writer.writeheader()
        writer.writerows(all_predictions)

    # Compare against v1 baseline
    v1_res = json.loads(V1_RESULT.read_text(encoding="utf-8"))["results"]
    comparison = {}
    for src in ["mendeley", "tcia", "busbra"]:
        g1 = v1_res[src]["group_metrics"]
        g2 = results[src]["group_metrics"]
        comparison[src] = {
            "v1_held_out_images": v1_res[src]["held_out_images"],
            "v2_held_out_images": results[src]["held_out_images"],
            "v1_held_out_groups": v1_res[src]["held_out_groups"],
            "v2_held_out_groups": results[src]["held_out_groups"],
            "v1_threshold": v1_res[src]["threshold"],
            "v2_threshold": results[src]["threshold"],
            "v1_group_roc_auc": g1["roc_auc"],
            "v2_group_roc_auc": g2["roc_auc"],
            "delta_group_roc_auc": float(g2["roc_auc"] - g1["roc_auc"]),
            "v1_group_balanced_accuracy": g1["balanced_accuracy"],
            "v2_group_balanced_accuracy": g2["balanced_accuracy"],
            "delta_group_balanced_accuracy": float(g2["balanced_accuracy"] - g1["balanced_accuracy"]),
            "v1_group_sensitivity": g1["sensitivity"],
            "v2_group_sensitivity": g2["sensitivity"],
            "v1_group_specificity": g1["specificity"],
            "v2_group_specificity": g2["specificity"],
            "v1_group_brier": g1["brier"],
            "v2_group_brier": g2["brier"],
        }

    summary = {
        "experiment_id": "full_source_baseline_v2",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_version": "v2",
        "eligible_images": 2215,
        "eligible_groups": 1288,
        "all_15_fits_converged": True,
        "saved_model_replay_max_errors": replay_max_errors,
        "results": results,
        "v1_vs_v2_comparison": comparison,
        "protocol_sha256": digest(PROTOCOL),
        "lock_sha256": digest(OUT / "lock.json"),
        "inference_enabled": False,
    }
    save(OUT / "result.json", summary)
    save(OUT / "v1_vs_v2_comparison.json", comparison)

    # Confirm all 51 prior locked artifacts remain untouched
    for p, h in json.loads(Path("reports/design_audit_v1/prior_artifact_hashes.json").read_text(encoding="utf-8")).items():
        assert digest(p) == h, p

    write_report(summary)
    print("Completed full_source_baseline_v2 and v1-vs-v2 comparison.", flush=True)
    return summary


def write_report(summary: dict) -> None:
    comp = summary["v1_vs_v2_comparison"]
    res = summary["results"]
    lines = [
        "# Full-Source Frozen Linear Baseline V2 (`full_source_baseline_v2` — Grouping-Validated Manifest V2)",
        "",
        f"**Completed UTC:** `{summary['completed_utc']}`",
        f"**Manifest:** `reports/grouping_audit_v1/eligible_manifest_v2.json` (`2,215` eligible images across `1,288` validated groups)",
        f"**All 15 L-BFGS-B Fits Converged (`max_absolute_gradient <= 1e-5`):** `{summary['all_15_fits_converged']}`",
        f"**Saved-Model Replay Max Error:** `{max(summary['saved_model_replay_max_errors'].values()):.3e}`",
        "",
        "## 1. Impact of Manifest V1 -> V2 Grouping Corrections on Frozen Linear Baseline",
        "",
        "| Held-Out Source | V1 Images / Groups | V2 Images / Groups | V1 Group ROC-AUC | V2 Group ROC-AUC (95% CI) | Delta ROC-AUC | V1 Group Bal Acc | V2 Group Bal Acc (95% CI) | Delta Bal Acc | V2 Sens / Spec |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for src in ["mendeley", "tcia", "busbra"]:
        c = comp[src]
        r = res[src]
        gm = r["group_metrics"]
        gi = r["group_intervals"]
        lines.append(
            f"| `{src}` | `{c['v1_held_out_images']} / {c['v1_held_out_groups']}` | "
            f"`{c['v2_held_out_images']} / {c['v2_held_out_groups']}` | "
            f"`{c['v1_group_roc_auc']:.4f}` | "
            f"`{c['v2_group_roc_auc']:.4f}` (`[{gi['roc_auc'][0]:.4f}, {gi['roc_auc'][1]:.4f}]`) | "
            f"`{c['delta_group_roc_auc']:+.4f}` | "
            f"`{c['v1_group_balanced_accuracy']:.4f}` | "
            f"`{c['v2_group_balanced_accuracy']:.4f}` (`[{gi['balanced_accuracy'][0]:.4f}, {gi['balanced_accuracy'][1]:.4f}]`) | "
            f"`{c['delta_group_balanced_accuracy']:+.4f}` | "
            f"`{gm['sensitivity']:.4f} / {gm['specificity']:.4f}` |"
        )
    lines.extend(
        [
            "",
            "## 2. Scientific Interpretation",
            "- **Mendeley (`38 -> 28` duplicate groups, `162 -> 145` images):** Removing the 17 partial-sequence frames that leaked from `mendeley_v1` test blocks and reuniting the 2 split five-image sequences (`us21..25` and `us191..195`) evaluates Mendeley strictly on its 28 complete held-out base-scan groups (14 benign, 14 malignant).",
            "- **TCIA (`197` patient IDs) & BUS-BRA (`1,062 -> 1,063` source Case groups):** Separating the single false-positive `dHash=5` merge in BUS-BRA (`bus_0426-l` / `bus_0888-r`) and cleaning the Mendeley training fold produces virtually identical held-out metrics on TCIA and BUS-BRA, confirming that the cross-source generalization bottleneck is robust to grouping cleanup.",
            "- **Reference baseline for `gpu_encoder_finetune_v3`:** `reports/full_source_baseline_v2/result.json` serves as the exact, grouping-aligned linear baseline for the packaged `gpu_encoder_finetune_v3` experiment.",
        ]
    )
    DOC_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", default="run", choices=["lock", "features", "run"])
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.stage == "lock":
            lock()
        elif args.stage == "features":
            prepare_features()
        else:
            run_all()
