"""Predeclared full-eligible source-held-out fine-tuning experiment (full_source_finetune_v1).

Executes the controlled nonlinear adaptation experiment defined in
docs/FULL_SOURCE_FINETUNE_PROTOCOL_V1.md across all 2,232 eligible images
(1,298 global duplicate/case groups) with:
- Anatomy-preserving ultrasound augmentations (horizontal flip, mild rotation
  within [-6, +6] degrees, 94% central crop resized to 224x224, mild B-mode
  brightness/contrast gain adjustment +/-8%, plus epoch-level stochastic view
  interpolation and multiplicative speckle/gain jitter on training splits only)
- Deliberate source, diagnosis, and duplicate-group sample weighting
- Epoch-level training and validation loss/AUC/BA tracking
- Training-side validation early stopping and checkpoint selection
- Training-side OOF Platt calibration and threshold selection
- Separate per-source evaluation with 1,000-resample group bootstrap 95% CIs
- Optional --mode full_layer4_gpu for raw-image layer4+head fine-tuning on CUDA GPUs
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from ml.cross_source import WEIGHTS, aggregate, intervals, metric, net, payload
from ml.full_source_baseline import calibration_bins, choose_threshold, sample_weights

OUT = Path("reports/full_source_finetune_v1")
PROTOCOL = Path("docs/FULL_SOURCE_FINETUNE_PROTOCOL_V1.md")
MANIFEST = Path("reports/design_audit_v1/eligible_manifest.json")
UNIFIED_MANIFEST = Path("data/prepared/unified_dataset_manifest_v1.csv")
CANONICAL_FEATURES = Path("reports/full_source_baseline_v1/features.npz")
PRIOR_HASHES = Path("reports/design_audit_v1/prior_artifact_hashes.json")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def digest(p: Path | str) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p: Path | str, obj: object) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, p)


def atomic_npz(path: Path | str, **items: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez(f, **items)
    os.replace(tmp, path)


def rows() -> list[dict]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def ultrasound_augment_pil(img: Image.Image, row_index: int, base_seed: int = 101) -> tuple[Image.Image, dict]:
    """Apply deterministic anatomy-preserving B-mode ultrasound augmentations to a PIL image."""
    rng = np.random.default_rng(base_seed * 100_003 + int(row_index))
    img = img.convert("RGB")
    # 1. Horizontal reflection (preserves skin-to-chest-wall vertical depth axis)
    flipped = True
    img = TF.hflip(img)
    # 2. Mild in-plane probe tilt rotation within [-6, +6] degrees
    angle = float(rng.uniform(-6.0, 6.0))
    img = TF.rotate(img, angle=angle, interpolation=InterpolationMode.BILINEAR, fill=0)
    # 3. Aspect-preserving 94% central crop resized to 224x224
    w, h = img.size
    crop_h = max(16, int(round(h * 0.94)))
    crop_w = max(16, int(round(w * 0.94)))
    img = TF.center_crop(img, [crop_h, crop_w])
    img = TF.resize(img, [224, 224], interpolation=InterpolationMode.BILINEAR, antialias=True)
    # 4. Mild B-mode brightness and contrast gain adjustment (+/-8%)
    brightness = float(rng.uniform(0.92, 1.08))
    contrast = float(rng.uniform(0.92, 1.08))
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    meta = {
        "horizontal_flip": flipped,
        "angle_deg": round(angle, 4),
        "center_crop_fraction": 0.94,
        "brightness_factor": round(brightness, 4),
        "contrast_factor": round(contrast, 4),
    }
    return img, meta


def augmented_tensors(batch_rows: list[dict], row_indices: list[int], base_seed: int = 101) -> torch.Tensor:
    to_tensor_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    out = []
    for r, idx in zip(batch_rows, row_indices):
        raw = payload(r)
        assert hashlib.sha256(raw).hexdigest() == r["sha256"], f"SHA mismatch on row {idx}"
        with Image.open(io.BytesIO(raw)) as im:
            aug_im, _ = ultrasound_augment_pil(im, row_index=idx, base_seed=base_seed)
            out.append(to_tensor_norm(aug_im))
    return torch.stack(out)


class ResidualAdapterHead(nn.Module):
    """Residual bottleneck adapter + classification head on 512-d ResNet-18 embeddings."""

    def __init__(self, dim: int = 512, bottleneck: int = 256, drop_adapter: float = 0.25, drop_head: float = 0.20):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop_adapter)
        self.fc2 = nn.Linear(bottleneck, dim)
        self.gate = nn.Parameter(torch.zeros(1))
        self.drop_head = nn.Dropout(drop_head)
        self.classifier = nn.Linear(dim, 2)

    def forward_features(self, z: torch.Tensor) -> torch.Tensor:
        delta = self.fc2(self.drop1(self.act(self.fc1(self.norm(z)))))
        return z + torch.tanh(self.gate) * delta

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.forward_features(z)
        logits2 = self.classifier(self.drop_head(h))
        # Binary malignant logit = logit(class 1) - logit(class 0)
        return logits2[:, 1] - logits2[:, 0]

    def forward_two_class(self, z: torch.Tensor) -> torch.Tensor:
        h = self.forward_features(z)
        return self.classifier(self.drop_head(h))


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def lock() -> None:
    import PIL
    import scipy
    import sklearn
    import sys
    import torchvision

    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "models").exists() and any((OUT / "models").iterdir()):
        raise FileExistsError("Models already exist; do not relock after fitting starts.")
    assert PROTOCOL.exists(), f"Missing protocol: {PROTOCOL}"
    assert MANIFEST.exists(), f"Missing manifest: {MANIFEST}"
    assert UNIFIED_MANIFEST.exists(), f"Missing unified manifest: {UNIFIED_MANIFEST}"
    assert CANONICAL_FEATURES.exists(), f"Missing canonical baseline features: {CANONICAL_FEATURES}"

    rs = rows()
    assert len(rs) == 2232 and len({r["group"] for r in rs}) == 1298
    files = [
        PROTOCOL,
        Path("ml/full_source_finetune.py"),
        MANIFEST,
        UNIFIED_MANIFEST,
        WEIGHTS,
        CANONICAL_FEATURES,
        Path("ml/cross_source.py"),
        Path("ml/full_source_baseline.py"),
        PRIOR_HASHES,
    ]
    obj = {
        "experiment_id": "full_source_finetune_v1",
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "hashes": {str(p).replace("\\", "/"): digest(p) for p in files},
        "images": len(rs),
        "groups": len({r["group"] for r in rs}),
        "class_order": ["benign", "malignant"],
        "architecture": "ResNet18-ResidualAdapter",
        "trainable_parameters": count_trainable_parameters(ResidualAdapterHead()),
        "hyperparameters": {
            "optimizer": "AdamW",
            "learning_rate": 3e-4,
            "eta_min": 1e-5,
            "weight_decay": 1e-2,
            "batch_size": 64,
            "max_epochs": 35,
            "warmup_epochs": 8,
            "early_stopping_patience": 10,
            "max_grad_norm": 2.0,
            "augmentation_seed": 101,
            "mixview_alpha_max": 0.35,
            "speckle_jitter_std": 0.03,
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
    print("Locked full_source_finetune_v1 protocol before model fitting.", flush=True)


def check_lock() -> dict:
    obj = json.loads((OUT / "lock.json").read_text(encoding="utf-8"))
    assert digest(OUT / "lock.json") == (OUT / "LOCK_SHA256.txt").read_text(encoding="utf-8").strip()
    for p, h in obj["hashes"].items():
        assert digest(p) == h, f"Locked file hash changed: {p}"
    return obj


def aug_features() -> None:
    check_lock()
    torch.set_num_threads(4)
    rs = rows()
    directory = OUT / "aug_chunks"
    directory.mkdir(parents=True, exist_ok=True)
    m = net()
    output = []
    start = time.perf_counter()
    chunk_size = 64
    for begin in range(0, len(rs), chunk_size):
        batch = rs[begin : begin + chunk_size]
        indices = list(range(begin, begin + len(batch)))
        file = directory / f"{begin:06d}.npz"
        sha = np.array([r["sha256"] for r in batch])
        if file.exists():
            with np.load(file) as f:
                assert np.array_equal(f["sha256"], sha)
                x = f["features"]
        else:
            with torch.inference_mode():
                xb = augmented_tensors(batch, indices, base_seed=101)
                x = m(xb).numpy().astype(np.float32)
            assert np.isfinite(x).all()
            atomic_npz(file, features=x, sha256=sha)
        output.append(x)
        if begin % 256 == 0 or begin + len(batch) == len(rs):
            print(
                f"aug_features {begin + len(batch)} / {len(rs)} "
                f"({time.perf_counter() - start:.1f}s)",
                flush=True,
            )
    full_aug = np.concatenate(output, axis=0)
    assert full_aug.shape == (len(rs), 512)
    atomic_npz(
        OUT / "aug_features.npz",
        features=full_aug,
        sha256=np.array([r["sha256"] for r in rs]),
    )
    save(
        OUT / "aug_feature_receipt.json",
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "images": len(rs),
            "features": 512,
            "augmentation_seed": 101,
            "augmentation_policy": {
                "horizontal_flip": True,
                "rotation_degrees_range": [-6.0, 6.0],
                "center_crop_fraction": 0.94,
                "brightness_range": [0.92, 1.08],
                "contrast_range": [0.92, 1.08],
                "vertical_flip": False,
            },
            "elapsed_seconds": time.perf_counter() - start,
            "aug_features_sha256": digest(OUT / "aug_features.npz"),
            "input_sha256_checked_on_new_chunk_creation": True,
        },
    )


def weighted_bce_with_logits(logits: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    total = float(w.sum())
    loss_vec = np.logaddexp(0.0, logits) - y * logits
    return float(np.dot(w, loss_vec) / total)


def equal_source_group_metrics(y: np.ndarray, logits: np.ndarray, g: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """Compute equal-source-weighted group ROC-AUC and group balanced accuracy at p=0.5 (logit=0)."""
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    gy, gp, gg = aggregate(y, probs, g)
    gs = np.array([s[np.flatnonzero(g == v)[0]] for v in gg])
    aucs = []
    bas = []
    for src in sorted(set(gs)):
        ix = gs == src
        ys = gy[ix]
        ps = gp[ix]
        if len(set(ys)) == 2:
            aucs.append(float(roc_auc_score(ys, ps)))
            sens = float(np.mean(ps[ys == 1] >= 0.5))
            spec = float(np.mean(ps[ys == 0] < 0.5))
            bas.append(0.5 * (sens + spec))
    return (float(np.mean(aucs)) if aucs else 0.5, float(np.mean(bas)) if bas else 0.5)


def predict_adapter_logits(state: dict, x: np.ndarray) -> np.ndarray:
    z = (x.astype(np.float32) - state["mean"].astype(np.float32)) / state["scale"].astype(np.float32)
    model = ResidualAdapterHead()
    torch_state = {
        k.replace("module.", ""): torch.from_numpy(v)
        for k, v in state.items()
        if k not in ("mean", "scale")
    }
    model.load_state_dict(torch_state, strict=True)
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(z)).numpy().astype(float)
    return logits


def fit_adapter(
    name: str,
    x_tr: np.ndarray,
    x_aug_tr: np.ndarray,
    y_tr: np.ndarray,
    s_tr: np.ndarray,
    g_tr: np.ndarray,
    x_va: np.ndarray | None = None,
    y_va: np.ndarray | None = None,
    s_va: np.ndarray | None = None,
    g_va: np.ndarray | None = None,
    fixed_epochs: int | None = None,
    seed: int = 42,
) -> tuple[dict, dict]:
    dest = OUT / "models"
    dest.mkdir(parents=True, exist_ok=True)
    model_npz = dest / f"{name}.npz"
    model_pth = dest / f"{name}.pth"
    record_file = dest / f"{name}.json"
    curve_file = dest / f"{name}.curve.json"

    if model_npz.exists() and record_file.exists() and curve_file.exists():
        info = json.loads(record_file.read_text(encoding="utf-8"))
        assert info["model_sha256"] == digest(model_npz)
        assert info["training_groups"] == sorted(set(g_tr))
        with np.load(model_npz) as f:
            return {k: f[k] for k in f.files}, info

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    w_tr = sample_weights(y_tr, s_tr, g_tr, True)
    scaler = StandardScaler().fit(x_tr.astype(float), sample_weight=w_tr)
    mean32 = scaler.mean_.astype(np.float32)
    scale32 = scaler.scale_.astype(np.float32)

    z_tr_can = ((x_tr - mean32) / scale32).astype(np.float32)
    z_tr_aug = ((x_aug_tr - mean32) / scale32).astype(np.float32)

    has_val = x_va is not None and y_va is not None and s_va is not None and g_va is not None
    if has_val:
        w_va = sample_weights(y_va, s_va, g_va, True)
        z_va_can = ((x_va - mean32) / scale32).astype(np.float32)

    model = ResidualAdapterHead()
    max_epochs = int(fixed_epochs) if fixed_epochs is not None else 35
    warmup_epochs = 8
    patience = 10
    batch_size = 64

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)

    curve = []
    best_epoch = 1
    best_key = (float("inf"), -float("inf"), -float("inf"))
    best_state_tensors = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    epochs_since_improvement = 0
    start = time.perf_counter()

    n_train = len(y_tr)
    y_tr_t = torch.from_numpy(y_tr.astype(np.float32))
    w_tr_t = torch.from_numpy(w_tr.astype(np.float32))

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = rng.permutation(n_train)
        # Stochastic ultrasound view mixing + multiplicative speckle/gain jitter
        use_aug = rng.random(n_train) < 0.65
        alpha = rng.uniform(0.0, 0.35, size=(n_train, 1)).astype(np.float32)
        z_base = np.where(use_aug[:, None], (1.0 - alpha) * z_tr_aug + alpha * z_tr_can, z_tr_can)
        speckle = rng.normal(1.0, 0.03, size=z_base.shape).astype(np.float32)
        z_epoch = torch.from_numpy(z_base * speckle)

        for b0 in range(0, n_train, batch_size):
            ix = perm[b0 : b0 + batch_size]
            zb = z_epoch[ix]
            yb = y_tr_t[ix]
            wb = w_tr_t[ix]
            optimizer.zero_grad()
            logits = model(zb)
            loss_vec = nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            loss = torch.sum(loss_vec * wb) / torch.clamp(torch.sum(wb), min=1e-8)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()

        # Evaluate epoch on canonical training and validation splits
        model.eval()
        with torch.inference_mode():
            tr_logits = model(torch.from_numpy(z_tr_can)).numpy().astype(float)
            tr_loss = weighted_bce_with_logits(tr_logits, y_tr, w_tr)
            tr_auc, tr_ba = equal_source_group_metrics(y_tr, tr_logits, g_tr, s_tr)

            row_entry = {
                "epoch": epoch,
                "learning_rate": current_lr,
                "train_loss": tr_loss,
                "train_group_roc_auc": tr_auc,
                "train_group_ba": tr_ba,
            }

            if has_val:
                va_logits = model(torch.from_numpy(z_va_can)).numpy().astype(float)
                va_loss = weighted_bce_with_logits(va_logits, y_va, w_va)
                va_auc, va_ba = equal_source_group_metrics(y_va, va_logits, g_va, s_va)
                row_entry.update(
                    {
                        "val_loss": va_loss,
                        "val_group_roc_auc": va_auc,
                        "val_group_ba": va_ba,
                    }
                )
                candidate_key = (round(va_loss, 8), -round(va_auc, 8), -round(va_ba, 8))
                if candidate_key < best_key:
                    best_key = candidate_key
                    best_epoch = epoch
                    best_state_tensors = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1
            else:
                best_epoch = epoch
                best_state_tensors = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            curve.append(row_entry)

        if has_val and epoch >= warmup_epochs and epochs_since_improvement >= patience:
            break

    save(curve_file, curve)

    npz_dict = {
        "mean": scaler.mean_.astype(np.float64),
        "scale": scaler.scale_.astype(np.float64),
    }
    for k, v in best_state_tensors.items():
        npz_dict[f"module.{k}"] = v.numpy()

    atomic_npz(model_npz, **npz_dict)
    torch.save(
        {
            "architecture": "ResNet18-ResidualAdapter",
            "class_order": ["benign", "malignant"],
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "state_dict": best_state_tensors,
            "selected_epoch": best_epoch,
            "epochs_ran": len(curve),
        },
        model_pth,
    )

    best_curve_row = curve[best_epoch - 1]
    info = {
        "fit": name,
        "architecture": "ResNet18-ResidualAdapter",
        "trainable_parameters": count_trainable_parameters(model),
        "train_images": int(len(y_tr)),
        "train_groups": int(len(set(g_tr))),
        "training_groups": sorted(set(g_tr)),
        "validation_images": int(len(y_va)) if has_val else 0,
        "validation_groups": int(len(set(g_va))) if has_val else 0,
        "source_class_images": {
            src: {str(label): int(((s_tr == src) & (y_tr == label)).sum()) for label in [0, 1]}
            for src in sorted(set(s_tr))
        },
        "max_epochs_budget": max_epochs,
        "epochs_ran": len(curve),
        "selected_epoch": int(best_epoch),
        "early_stopped": bool(has_val and len(curve) < max_epochs),
        "selected_metrics": best_curve_row,
        "elapsed_seconds": time.perf_counter() - start,
        "sum_sample_weights": float(w_tr.sum()),
        "class_weight_sums": {str(label): float(w_tr[y_tr == label].sum()) for label in [0, 1]},
        "model_sha256": digest(model_npz),
        "pth_sha256": digest(model_pth),
        "curve_sha256": digest(curve_file),
        "lock_sha256": digest(OUT / "lock.json"),
    }
    save(record_file, info)
    print(
        f"fitted {name}: epochs_ran={len(curve)}, selected_epoch={best_epoch}, "
        f"train_loss={best_curve_row['train_loss']:.4f}"
        + (f", val_loss={best_curve_row['val_loss']:.4f}, val_auc={best_curve_row['val_group_roc_auc']:.4f}" if has_val else ""),
        flush=True,
    )
    return npz_dict, info


def run() -> None:
    check_lock()
    torch.set_num_threads(2)
    rs = rows()
    sha_expected = np.array([r["sha256"] for r in rs])
    with np.load(CANONICAL_FEATURES) as f:
        assert np.array_equal(f["sha256"], sha_expected)
        x_can = f["features"].astype(np.float32)
    with np.load(OUT / "aug_features.npz") as f:
        assert np.array_equal(f["sha256"], sha_expected)
        x_aug = f["features"].astype(np.float32)

    y = np.array([r["label"] == "malignant" for r in rs], int)
    g = np.array([r["group"] for r in rs])
    s = np.array([r["source"] for r in rs])

    results = {}
    all_predictions = []
    for source_idx, source in enumerate(["mendeley", "tcia", "busbra"]):
        out = OUT / source
        out.mkdir(parents=True, exist_ok=True)
        if (out / "result.json").exists():
            results[source] = json.loads((out / "result.json").read_text(encoding="utf-8"))
            with (out / "predictions.csv").open("r", encoding="utf-8") as f:
                all_predictions += list(csv.DictReader(f))
            continue

        train = np.flatnonzero(s != source)
        held = np.flatnonzero(s == source)
        oof = np.empty(len(train), dtype=float)
        folds = []
        inner_selected_epochs = []
        strata = np.array([f"{a}:{b}" for a, b in zip(s[train], y[train])])

        for fold, (itr, iva) in enumerate(
            StratifiedGroupKFold(4, shuffle=True, random_state=42).split(x_can[train], strata, g[train])
        ):
            tr, va = train[itr], train[iva]
            assert not (set(g[tr]) & set(g[va]))
            assert not (set(g[train]) & set(g[held]))
            assert source not in (set(s[tr]) | set(s[va]))

            model_state, info = fit_adapter(
                name=f"{source}_inner{fold}",
                x_tr=x_can[tr],
                x_aug_tr=x_aug[tr],
                y_tr=y[tr],
                s_tr=s[tr],
                g_tr=g[tr],
                x_va=x_can[va],
                y_va=y[va],
                s_va=s[va],
                g_va=g[va],
                fixed_epochs=None,
                seed=42 + source_idx * 10 + fold,
            )
            oof[iva] = predict_adapter_logits(model_state, x_can[va])
            inner_selected_epochs.append(int(info["selected_epoch"]))
            folds.append(
                {
                    "fold": fold,
                    "fit": f"{source}_inner{fold}",
                    "train_rows": tr.tolist(),
                    "validation_rows": va.tolist(),
                    "train_groups": sorted(set(g[tr])),
                    "validation_groups": sorted(set(g[va])),
                    "selected_epoch": int(info["selected_epoch"]),
                    "epochs_ran": int(info["epochs_ran"]),
                    "early_stopped": bool(info["early_stopped"]),
                    "selected_metrics": info["selected_metrics"],
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
        refit_epochs = max(8, int(round(float(np.mean(inner_selected_epochs)))))

        selection = {
            "threshold": threshold,
            "calibration_coef": calibrator.coef_.tolist(),
            "calibration_intercept": calibrator.intercept_.tolist(),
            "calibration_iterations": calibrator.n_iter_.tolist(),
            "inner_selected_epochs": inner_selected_epochs,
            "refit_epochs": refit_epochs,
            "training_rows": train.tolist(),
            "training_groups": sorted(set(g[train])),
            "training_sources": sorted(set(s[train])),
            "held_out_source_not_used": True,
            "selection_frozen_utc": datetime.now(timezone.utc).isoformat(),
        }
        save(out / "training_side_selection.json", selection)
        atomic_npz(out / "oof.npz", rows=train, logits=oof, calibrated_probabilities=vp)

        final_state, final_info = fit_adapter(
            name=f"{source}_final",
            x_tr=x_can[train],
            x_aug_tr=x_aug[train],
            y_tr=y[train],
            s_tr=s[train],
            g_tr=g[train],
            fixed_epochs=refit_epochs,
            seed=42 + source_idx * 10 + 9,
        )
        save(
            out / "EVALUATION_STARTED.json",
            {
                "utc": datetime.now(timezone.utc).isoformat(),
                "selection_sha256": digest(out / "training_side_selection.json"),
                "model_sha256": final_info["model_sha256"],
            },
        )

        logits = predict_adapter_logits(final_state, x_can[held])
        p = calibrator.predict_proba(logits[:, None])[:, 1]
        gy, gp, gg = aggregate(y[held], p, g[held])
        ty, tp, tg = aggregate(y[train], vp, g[train])

        group_counts = Counter(gy)
        image_counts = Counter(y[held])
        train_group_counts = Counter(ty)
        training_majority = max(train_group_counts, key=train_group_counts.get)
        prev = float(ty.mean())

        train_refit_logits = predict_adapter_logits(final_state, x_can[train])
        train_refit_p = calibrator.predict_proba(train_refit_logits[:, None])[:, 1]

        result = {
            "held_out_source": source,
            "inspected_exploratory_development": True,
            "train_images": int(len(train)),
            "train_groups": int(len(set(g[train]))),
            "held_out_images": int(len(held)),
            "held_out_groups": int(len(gg)),
            "group_class_counts": {str(k): int(v) for k, v in group_counts.items()},
            "image_class_counts": {str(k): int(v) for k, v in image_counts.items()},
            **selection,
            "group_metrics": metric(gy, gp, threshold),
            "image_metrics": metric(y[held], p, threshold),
            "group_intervals": intervals(gy, gp, gg, threshold),
            "image_intervals": intervals(y[held], p, g[held], threshold),
            "group_calibration_bins": calibration_bins(gy, gp),
            "image_calibration_bins": calibration_bins(y[held], p),
            "training_oof_group_metrics": metric(ty, tp, threshold),
            "training_refit_image_metrics": metric(y[train], train_refit_p, threshold),
            "majority_baselines": {
                "held_group_majority_accuracy": max(group_counts.values()) / len(gy),
                "held_image_majority_accuracy": max(image_counts.values()) / len(held),
                "majority_balanced_accuracy": 0.5,
                "training_group_majority_class": int(training_majority),
                "training_group_majority_accuracy_on_held": float((gy == training_majority).mean()),
                "training_group_prevalence": prev,
                "training_prevalence_constant_held_group_brier": float(np.mean((prev - gy) ** 2)),
            },
            "final_optimizer": final_info,
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
        print(
            f"completed held-out source {source}: "
            f"group_BA={result['group_metrics']['balanced_accuracy']:.3f}, "
            f"group_AUC={result['group_metrics']['roc_auc']:.3f}",
            flush=True,
        )

    with (OUT / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_predictions[0]))
        writer.writeheader()
        writer.writerows(all_predictions)

    save(
        OUT / "result.json",
        {
            "experiment_id": "full_source_finetune_v1",
            "results": results,
            "protocol_sha256": digest(PROTOCOL),
            "lock_sha256": digest(OUT / "lock.json"),
            "architecture_search": False,
            "inference_enabled": False,
        },
    )
    for p, h in json.loads(PRIOR_HASHES.read_text(encoding="utf-8")).items():
        assert digest(p) == h, f"Prior locked artifact changed: {p}"
    print("Completed all three full-eligible fine-tuned source-held-out folds; prior artifacts unchanged.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["lock", "aug_features", "run"])
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        globals()[args.stage]()

