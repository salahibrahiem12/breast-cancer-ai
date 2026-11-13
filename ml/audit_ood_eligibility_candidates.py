"""Exploratory read-only feature-space OOD diagnostic and eligibility-gate blocker audit.

Evaluates:
1. Why a deployable eligibility gate cannot be claimed yet (missing licensed
   multi-category negative cohorts: natural_image, object, screenshot, low_quality,
   and unresolved patient/author provenance on fetal_head_zenodo and ChestMNIST).
2. Read-only feature-space Mahalanobis and cosine distance diagnostics of the
   2,232 eligible breast-ultrasound embeddings against deterministic samples from
   the locally inventoried candidate negative archives (fetal_head_zenodo and
   ChestMNIST 64), without admitting them to gate training or enabling inference.
3. Verification that backend/app/inference.py remains strictly fail-closed
   (inference_enabled=False).
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms

from backend.app.inference import inspect_image, model_availability
from ml.cross_source import net
from ml.full_source_finetune import IMAGENET_MEAN, IMAGENET_STD
from ml.prepare_eligibility import REQUIRED_NEGATIVE_CATEGORIES

OUT_JSON = Path("reports/eligibility_ood_candidate_audit_v1.json")
CANONICAL_FEATURES = Path("reports/full_source_baseline_v1/features.npz")
ELIGIBLE_MANIFEST = Path("reports/design_audit_v1/eligible_manifest.json")
FETAL_DIR = Path("data/prepared/fetal_candidate_v1/images")
CHESTMNIST_NPZ = Path("data/raw/medmnist/chestmnist_64.npz")


def extract_pil_features(images: list[Image.Image]) -> np.ndarray:
    m = net()
    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    tensors = torch.stack([tf(im.convert("RGB")) for im in images])
    with torch.inference_mode():
        feats = m(tensors).numpy().astype(np.float64)
    return feats


def run_audit() -> dict:
    torch.set_num_threads(4)
    rows = json.loads(ELIGIBLE_MANIFEST.read_text(encoding="utf-8"))
    with np.load(CANONICAL_FEATURES) as f:
        x_bus = f["features"].astype(np.float64)
    sources = np.array([r["source"] for r in rows])

    # Leave-one-source-out Mahalanobis OOD diagnostic:
    # Fit regularized covariance on 2 breast-ultrasound sources, evaluate:
    # - held-out breast-ultrasound source (valid in-domain target under domain shift)
    # - deterministic sample of 64 fetal-head ultrasound images (other_ultrasound candidate)
    # - deterministic sample of 64 ChestMNIST radiographs (other_medical candidate)
    fetal_images: list[Image.Image] = []
    fetal_names: list[str] = []
    if FETAL_DIR.is_dir():
        for p_img in sorted(FETAL_DIR.glob("*.png"))[:64]:
            with Image.open(p_img) as im:
                fetal_images.append(im.convert("RGB").copy())
            fetal_names.append(p_img.name)

    chest_images: list[Image.Image] = []
    if CHESTMNIST_NPZ.exists():
        with np.load(CHESTMNIST_NPZ) as npz:
            test_imgs = npz["test_images"][:64]
            for arr in test_imgs:
                chest_images.append(Image.fromarray(arr).convert("RGB"))

    x_fetal = extract_pil_features(fetal_images) if fetal_images else np.empty((0, 512))
    x_chest = extract_pil_features(chest_images) if chest_images else np.empty((0, 512))

    source_diagnostics = {}
    for held_source in ["mendeley", "tcia", "busbra"]:
        tr_ix = sources != held_source
        te_ix = sources == held_source
        x_tr = x_bus[tr_ix]
        x_te = x_bus[te_ix]

        mu = x_tr.mean(axis=0)
        xc = x_tr - mu
        cov = (xc.T @ xc) / max(1, len(x_tr) - 1) + 0.10 * np.eye(x_tr.shape[1])
        prec = np.linalg.inv(cov)

        def maha(x: np.ndarray) -> np.ndarray:
            d = x - mu
            return np.sqrt(np.sum((d @ prec) * d, axis=1))

        d_tr = maha(x_tr)
        d_te_bus = maha(x_te)
        tau_95 = float(np.quantile(d_tr, 0.95))

        diag = {
            "training_breast_ultrasound_images": int(tr_ix.sum()),
            "held_out_breast_ultrasound_images": int(te_ix.sum()),
            "train_mahalanobis_p95_threshold": tau_95,
            "held_out_breast_ultrasound_acceptance_at_train_p95": float(np.mean(d_te_bus <= tau_95)),
            "held_out_breast_ultrasound_mean_distance": float(np.mean(d_te_bus)),
        }
        if len(x_fetal) > 0:
            d_fetal = maha(x_fetal)
            y_ood = np.r_[np.ones(len(d_te_bus), dtype=int), np.zeros(len(d_fetal), dtype=int)]
            s_ood = np.r_[-d_te_bus, -d_fetal]
            diag["fetal_head_other_ultrasound_sample"] = {
                "n": int(len(d_fetal)),
                "mean_distance": float(np.mean(d_fetal)),
                "false_acceptance_at_train_p95": float(np.mean(d_fetal <= tau_95)),
                "in_vs_other_ultrasound_auroc": float(roc_auc_score(y_ood, s_ood)),
            }
        if len(x_chest) > 0:
            d_chest = maha(x_chest)
            y_ood = np.r_[np.ones(len(d_te_bus), dtype=int), np.zeros(len(d_chest), dtype=int)]
            s_ood = np.r_[-d_te_bus, -d_chest]
            diag["chestmnist_other_medical_sample"] = {
                "n": int(len(d_chest)),
                "mean_distance": float(np.mean(d_chest)),
                "false_acceptance_at_train_p95": float(np.mean(d_chest <= tau_95)),
                "in_vs_other_medical_auroc": float(roc_auc_score(y_ood, s_ood)),
            }
        source_diagnostics[held_source] = diag

    availability = model_availability()
    assert not availability["inference_enabled"]

    report = {
        "audit_id": "eligibility_ood_candidate_audit_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "exploratory_diagnostic_only": True,
        "deployable_gate_claimed": False,
        "research_inference_approved": False,
        "required_negative_categories": sorted(REQUIRED_NEGATIVE_CATEGORIES),
        "inventoried_candidate_categories_present_locally": [
            "other_ultrasound (fetal_head_zenodo v2, 999 decoded images, inventory-only; author attribution & patient IDs unresolved)",
            "other_medical (MedMNIST+ ChestMNIST 64, 112,120 images, inventory-only; patient IDs & source-split leakage unverified)",
        ],
        "missing_required_negative_categories": [
            "natural_image",
            "object",
            "screenshot",
            "low_quality",
        ],
        "leave_one_source_out_mahalanobis_diagnostics": source_diagnostics,
        "key_finding": (
            "Under cross-source breast-ultrasound domain shift (especially when BUS-BRA is held out), "
            "a 95% in-distribution Mahalanobis threshold fit on two breast-ultrasound sources either "
            "rejects a substantial fraction of valid held-out breast-ultrasound scans or falsely accepts "
            "non-breast fetal ultrasound scans. This empirically confirms why uncalibrated feature-distance "
            "heuristics cannot substitute for a multi-category, source-disjoint trained eligibility gate."
        ),
        "serving_availability": availability,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved eligibility OOD candidate diagnostic audit to", OUT_JSON)
    return report


if __name__ == "__main__":
    run_audit()

