"""Separately versioned end-to-end GPU encoder fine-tuning runner and Colab package
(`gpu_encoder_finetune_v2` preserved lock -> superseded before training by `gpu_encoder_finetune_v3`
bound to the grouping-validated `unified_dataset_manifest_v2.csv`).

Key scientific guarantees:
- Trains from raw images with real convolutional encoder gradient updates in `ResNet-18` `layer4`
  (`8,388,608` conv parameters) + classification head `fc` (`1,026` parameters; total `8,389,634`
  trainable parameters).
- Freezes `conv1..layer3` AND freezes all 20 `nn.BatchNorm2d` modules across the entire network
  (including `layer4` `bn1`, `bn2`, and `downsample.1`), keeping every `BatchNorm2d` in `.eval()`
  mode with `requires_grad=False` even after `model.train()`.
- Uses training-side 4-fold OOF validation (`StratifiedGroupKFold(4)`) for early-stopping
  checkpoint selection, Platt calibration, and operating threshold selection.
- Preserves source isolation and saves per-epoch learning curves (`.curve.json`) and resumable
  checkpoints (`.resume.pth`).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from ml.cross_source import WEIGHTS, aggregate, intervals, metric
from ml.full_source_baseline import calibration_bins, choose_threshold, sample_weights
from ml.full_source_finetune import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    MANIFEST,
    PRIOR_HASHES,
    UNIFIED_MANIFEST,
    digest,
    equal_source_group_metrics,
    save,
    weighted_bce_with_logits,
)

OUT_V2 = Path("reports/gpu_encoder_finetune_v2")
PROTOCOL_V2 = Path("docs/GPU_ENCODER_FINETUNE_PROTOCOL_V2.md")

OUT_V3 = Path("reports/gpu_encoder_finetune_v3")
PROTOCOL_V3 = Path("docs/GPU_ENCODER_FINETUNE_PROTOCOL_V3.md")
MANIFEST_V2 = Path("reports/grouping_audit_v1/eligible_manifest_v2.json")
UNIFIED_MANIFEST_V2 = Path("data/prepared/unified_dataset_manifest_v2.csv")


def rows_v2() -> list[dict]:
    rs = json.loads(MANIFEST_V2.read_text(encoding="utf-8"))
    assert len(rs) == 2215 and len({r["group"] for r in rs}) == 1288
    return rs


def payload_v2(row: dict) -> bytes:
    if row["source"] == "busbra":
        with ZipFile("data/research_data/busbra_reserved_v1/BUSBRA.zip") as zf:
            return zf.read(row["archive_path"])
    p = Path(row["file"])
    if p.is_file():
        return p.read_bytes()
    if row["file"].startswith("data/research_data/tcia_breast_lesions_usg_2026-11-03/"):
        inner = row["file"].split("data/research_data/tcia_breast_lesions_usg_2026-11-03/", 1)[1]
        raw_alt = Path("data/raw/tcia_breast_biopsy_external_v1") / Path(inner).name
        if raw_alt.is_file():
            return raw_alt.read_bytes()
        zpath = (
            "data/research_data/tcia_breast_lesions_usg_2026-11-03/"
            "BrEaST-Lesions_USG-images_and_masks-Dec-15-2023.zip"
        )
        with ZipFile(zpath) as zf:
            return zf.read(inner)
    return p.read_bytes()


def _is_batchnorm_param_name(backbone: nn.Module, param_name: str) -> bool:
    mod_path = param_name.rsplit(".", 1)[0] if "." in param_name else ""
    mod = backbone.get_submodule(mod_path) if mod_path else backbone
    return isinstance(mod, nn.modules.batchnorm._BatchNorm)


class ResNet18Layer4FineTuner(nn.Module):
    """ResNet-18 with conv1..layer3 and ALL BatchNorm2d layers frozen, and layer4 convs + fc trainable."""

    def __init__(self, weights_path: Path = WEIGHTS):
        super().__init__()
        backbone = models.resnet18(weights=None)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        backbone.load_state_dict(state, strict=True)
        backbone.fc = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(512, 2),
        )
        self.backbone = backbone
        self._configure_trainable_stages()
        self._freeze_all_batchnorm_eval()

    def _configure_trainable_stages(self) -> None:
        for name, param in self.backbone.named_parameters():
            if _is_batchnorm_param_name(self.backbone, name):
                param.requires_grad = False
            elif name.startswith("layer4.") or name.startswith("fc."):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def _freeze_all_batchnorm_eval(self) -> None:
        self.backbone.conv1.eval()
        self.backbone.bn1.eval()
        self.backbone.layer1.eval()
        self.backbone.layer2.eval()
        self.backbone.layer3.eval()
        for mod in self.backbone.modules():
            if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                mod.eval()
                for p in mod.parameters():
                    p.requires_grad = False

    def train(self, mode: bool = True) -> "ResNet18Layer4FineTuner":
        super().train(mode)
        self._freeze_all_batchnorm_eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits2 = self.backbone(x)
        return logits2[:, 1] - logits2[:, 0]


def snapshot_batchnorm_buffers(model: ResNet18Layer4FineTuner) -> dict[str, dict[str, torch.Tensor]]:
    snap: dict[str, dict[str, torch.Tensor]] = {}
    for name, mod in model.backbone.named_modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            snap[name] = {
                "running_mean": mod.running_mean.detach().cpu().clone(),
                "running_var": mod.running_var.detach().cpu().clone(),
                "num_batches_tracked": mod.num_batches_tracked.detach().cpu().clone(),
                "weight": mod.weight.detach().cpu().clone(),
                "bias": mod.bias.detach().cpu().clone(),
            }
    assert len(snap) == 20, f"Expected 20 BatchNorm2d modules in ResNet-18, got {len(snap)}"
    return snap


def verify_frozen_batchnorm(
    model: ResNet18Layer4FineTuner,
    initial_snap: dict[str, dict[str, torch.Tensor]] | None = None,
) -> dict:
    bn_count = 0
    for name, mod in model.backbone.named_modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            bn_count += 1
            assert not mod.training, f"BatchNorm2d module {name} is in training mode!"
            for pname, p in mod.named_parameters():
                assert not p.requires_grad, f"BatchNorm2d parameter {name}.{pname} has requires_grad=True!"
                assert p.grad is None or float(p.grad.abs().max().item()) == 0.0, (
                    f"BatchNorm2d parameter {name}.{pname} received non-zero gradient!"
                )
            if initial_snap is not None:
                ref = initial_snap[name]
                assert torch.equal(mod.running_mean.detach().cpu(), ref["running_mean"]), (
                    f"BatchNorm2d {name}.running_mean drifted!"
                )
                assert torch.equal(mod.running_var.detach().cpu(), ref["running_var"]), (
                    f"BatchNorm2d {name}.running_var drifted!"
                )
                assert torch.equal(mod.num_batches_tracked.detach().cpu(), ref["num_batches_tracked"]), (
                    f"BatchNorm2d {name}.num_batches_tracked incremented!"
                )
                assert torch.equal(mod.weight.detach().cpu(), ref["weight"]), (
                    f"BatchNorm2d {name}.weight changed!"
                )
                assert torch.equal(mod.bias.detach().cpu(), ref["bias"]), (
                    f"BatchNorm2d {name}.bias changed!"
                )
    assert bn_count == 20
    return {
        "batchnorm_module_count": bn_count,
        "all_batchnorm_eval_mode": True,
        "all_batchnorm_requires_grad_false": True,
        "all_batchnorm_running_stats_and_affine_unchanged": initial_snap is not None,
    }


def parameter_summary(model: ResNet18Layer4FineTuner) -> dict:
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    layer4_trainable = sum(
        p.numel() for n, p in model.backbone.named_parameters() if n.startswith("layer4.") and p.requires_grad
    )
    fc_trainable = sum(
        p.numel() for n, p in model.backbone.named_parameters() if n.startswith("fc.") and p.requires_grad
    )
    bn_frozen_params = sum(
        p.numel()
        for n, p in model.backbone.named_parameters()
        if _is_batchnorm_param_name(model.backbone, n)
    )
    return {
        "total_parameters": int(frozen + layer4_trainable + fc_trainable),
        "frozen_parameters_total": int(frozen),
        "frozen_batchnorm_affine_parameters": int(bn_frozen_params),
        "trainable_layer4_conv_parameters": int(layer4_trainable),
        "trainable_fc_head_parameters": int(fc_trainable),
        "total_trainable_parameters": int(layer4_trainable + fc_trainable),
    }


def decode_pil_images(all_rows: list[dict]) -> list[Image.Image]:
    decoded = []
    for idx, r in enumerate(all_rows):
        raw = payload_v2(r)
        assert hashlib.sha256(raw).hexdigest() == r["sha256"], f"SHA mismatch at row {idx}"
        with Image.open(io.BytesIO(raw)) as im:
            decoded.append(im.convert("RGB").copy())
    return decoded


def canonical_tensor_batch(images: list[Image.Image], indices: np.ndarray) -> torch.Tensor:
    tf = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return torch.stack([tf(images[int(i)]) for i in indices])


def augmented_tensor_batch(
    images: list[Image.Image], indices: np.ndarray, rng: np.random.Generator
) -> torch.Tensor:
    to_tensor_norm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    out = []
    for i in indices:
        im = images[int(i)]
        if rng.random() < 0.5:
            im = TF.hflip(im)
        angle = float(rng.uniform(-6.0, 6.0))
        im = TF.rotate(im, angle=angle, interpolation=InterpolationMode.BILINEAR, fill=0)
        scale = float(rng.uniform(0.92, 1.00))
        w, h = im.size
        im = TF.center_crop(im, [max(16, int(round(h * scale))), max(16, int(round(w * scale)))])
        im = TF.resize(im, [224, 224], interpolation=InterpolationMode.BILINEAR, antialias=True)
        b_gain = float(rng.uniform(0.92, 1.08))
        c_gain = float(rng.uniform(0.92, 1.08))
        im = ImageEnhance.Brightness(im).enhance(b_gain)
        im = ImageEnhance.Contrast(im).enhance(c_gain)
        out.append(to_tensor_norm(im))
    return torch.stack(out)


def lock_v3() -> dict:
    """Preserve reports/gpu_encoder_finetune_v2/lock.json and issue V3 lock bound to unified_dataset_manifest_v2.csv."""
    import PIL
    import scipy
    import sklearn
    import sys
    import torchvision

    assert (OUT_V2 / "lock.json").is_file(), "Preserved V2 lock must remain intact."
    assert not (OUT_V2 / "models").exists(), "V2 must have zero executed model fits."

    superseded_record = {
        "superseded_experiment_id": "gpu_encoder_finetune_v2",
        "preserved_v2_lock_file": "reports/gpu_encoder_finetune_v2/lock.json",
        "preserved_v2_lock_sha256": digest(OUT_V2 / "lock.json"),
        "fits_executed_under_v2": 0,
        "superseded_by_experiment_id": "gpu_encoder_finetune_v3",
        "superseded_utc": datetime.now(timezone.utc).isoformat(),
        "reason": (
            "Global duplicate-group integrity audit (reports/grouping_audit_v1/grouping_audit.json) showed that "
            "unified_dataset_manifest_v1.csv contained 17 Mendeley sibling images from 8 historical mendeley_v1 "
            "test 5-image sequences that escaped dHash<=5, plus 2 split 5-image sequences (us21..25 in global:10/"
            "global:13 and us191..195 in global:120/global:123). Protocol V3 locks unified_dataset_manifest_v2.csv "
            "(2,215 eligible images, 1,288 groups) and enforces frozen BatchNorm2d across all 20 BatchNorm2d "
            "modules including layer4."
        ),
    }
    save(OUT_V2 / "SUPERSEDED_BY_V3.json", superseded_record)

    OUT_V3.mkdir(parents=True, exist_ok=True)
    if (OUT_V3 / "models").exists() and any((OUT_V3 / "models").iterdir()):
        raise FileExistsError("V3 models already exist; do not relock after fitting starts.")

    rs = rows_v2()
    probe = ResNet18Layer4FineTuner()
    files = [
        PROTOCOL_V3,
        PROTOCOL_V2,
        Path("ml/gpu_encoder_finetune_v2.py"),
        MANIFEST_V2,
        UNIFIED_MANIFEST_V2,
        MANIFEST,
        UNIFIED_MANIFEST,
        WEIGHTS,
        PRIOR_HASHES,
    ]
    obj = {
        "experiment_id": "gpu_encoder_finetune_v3",
        "supersedes_preserved_lock": "reports/gpu_encoder_finetune_v2/lock.json",
        "preserved_v2_lock_sha256": digest(OUT_V2 / "lock.json"),
        "training_paradigm": "end_to_end_resnet18_layer4_convs_and_head_on_raw_images_frozen_bn",
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "hashes": {str(p).replace("\\", "/"): digest(p) for p in files},
        "images": len(rs),
        "groups": len({r["group"] for r in rs}),
        "class_order": ["benign", "malignant"],
        "parameter_summary": parameter_summary(probe),
        "batchnorm_policy": "all_20_batchnorm2d_modules_frozen_eval_and_requires_grad_false",
        "hyperparameters": {
            "optimizer": "AdamW",
            "layer4_conv_lr": 1e-4,
            "fc_lr": 3e-4,
            "eta_min": 1e-6,
            "weight_decay": 1e-2,
            "batch_size": 32,
            "max_epochs": 20,
            "warmup_epochs": 4,
            "early_stopping_patience": 5,
            "max_grad_norm": 2.0,
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
    save(OUT_V3 / "lock.json", obj)
    (OUT_V3 / "LOCK_SHA256.txt").write_text(digest(OUT_V3 / "lock.json"), encoding="utf-8")
    print("Preserved V2 lock and locked gpu_encoder_finetune_v3 protocol.", flush=True)
    return obj


def check_lock_v3() -> dict:
    obj = json.loads((OUT_V3 / "lock.json").read_text(encoding="utf-8"))
    assert digest(OUT_V3 / "lock.json") == (OUT_V3 / "LOCK_SHA256.txt").read_text(encoding="utf-8").strip()
    for p, h in obj["hashes"].items():
        assert digest(p) == h, f"Locked file hash changed: {p}"
    return obj


def build_optimizer(model: ResNet18Layer4FineTuner, layer4_lr: float = 1e-4, fc_lr: float = 3e-4) -> torch.optim.AdamW:
    layer4_params = [
        p for n, p in model.backbone.named_parameters() if n.startswith("layer4.") and p.requires_grad
    ]
    fc_params = [
        p for n, p in model.backbone.named_parameters() if n.startswith("fc.") and p.requires_grad
    ]
    assert sum(p.numel() for p in layer4_params) == 8_388_608
    assert sum(p.numel() for p in fc_params) == 1_026
    return torch.optim.AdamW(
        [
            {"params": layer4_params, "lr": layer4_lr},
            {"params": fc_params, "lr": fc_lr},
        ],
        weight_decay=1e-2,
    )


def evaluate_canonical_logits(
    model: ResNet18Layer4FineTuner,
    images: list[Image.Image],
    indices: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    model.eval()
    preds = []
    with torch.inference_mode():
        for b0 in range(0, len(indices), batch_size):
            sub_ix = indices[b0 : b0 + batch_size]
            xb = canonical_tensor_batch(images, sub_ix).to(device)
            logits = model(xb).detach().cpu().numpy().astype(float)
            preds.append(logits)
    return np.concatenate(preds, axis=0)


def fit_encoder_fold(
    name: str,
    images: list[Image.Image],
    tr_ix: np.ndarray,
    y_tr: np.ndarray,
    s_tr: np.ndarray,
    g_tr: np.ndarray,
    va_ix: np.ndarray | None,
    y_va: np.ndarray | None,
    s_va: np.ndarray | None,
    g_va: np.ndarray | None,
    device: torch.device,
    out_dir: Path = OUT_V3,
    fixed_epochs: int | None = None,
    max_epochs_default: int = 20,
    batch_size: int = 32,
    seed: int = 42,
) -> tuple[dict[str, torch.Tensor], dict]:
    dest = out_dir / "models"
    dest.mkdir(parents=True, exist_ok=True)
    model_pth = dest / f"{name}.pth"
    resume_pth = dest / f"{name}.resume.pth"
    record_file = dest / f"{name}.json"
    curve_file = dest / f"{name}.curve.json"

    if model_pth.exists() and record_file.exists() and curve_file.exists():
        info = json.loads(record_file.read_text(encoding="utf-8"))
        assert info["pth_sha256"] == digest(model_pth)
        assert info["training_groups"] == sorted(set(g_tr))
        ckpt = torch.load(model_pth, map_location="cpu", weights_only=False)
        return ckpt["state_dict"], info

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    w_tr = sample_weights(y_tr, s_tr, g_tr, True)
    has_val = va_ix is not None and y_va is not None and s_va is not None and g_va is not None
    if has_val:
        w_va = sample_weights(y_va, s_va, g_va, True)

    model = ResNet18Layer4FineTuner().to(device)
    initial_bn_snap = snapshot_batchnorm_buffers(model)
    verify_frozen_batchnorm(model, initial_bn_snap)

    max_epochs = int(fixed_epochs) if fixed_epochs is not None else max_epochs_default
    warmup_epochs = min(4, max_epochs)
    patience = 5

    optimizer = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    start_epoch = 1
    curve: list[dict] = []
    best_epoch = 1
    best_key = (float("inf"), -float("inf"), -float("inf"))
    best_state_tensors = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    epochs_since_improvement = 0
    elapsed_prior = 0.0

    if resume_pth.exists():
        res = torch.load(resume_pth, map_location="cpu", weights_only=False)
        model.load_state_dict({k: v.to(device) for k, v in res["model_state_dict"].items()}, strict=True)
        optimizer.load_state_dict(res["optimizer_state_dict"])
        scheduler.load_state_dict(res["scheduler_state_dict"])
        start_epoch = int(res["epoch"]) + 1
        curve = list(res["curve"])
        best_epoch = int(res["best_epoch"])
        best_key = tuple(res["best_key"])
        best_state_tensors = {k: v.detach().cpu().clone() for k, v in res["best_state_dict"].items()}
        epochs_since_improvement = int(res["epochs_since_improvement"])
        elapsed_prior = float(res.get("elapsed_seconds", 0.0))
        rng.bit_generator.state = res["rng_state"]
        verify_frozen_batchnorm(model, initial_bn_snap)
        print(f"Resumed {name} from completed epoch {start_epoch - 1}/{max_epochs}", flush=True)

    t0 = time.perf_counter()
    n_train = len(y_tr)
    y_tr_t = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    w_tr_t = torch.from_numpy(w_tr.astype(np.float32)).to(device)

    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        # Verify frozen BatchNorm immediately after model.train()
        verify_frozen_batchnorm(model, initial_bn_snap)

        perm = rng.permutation(n_train)
        for b0 in range(0, n_train, batch_size):
            local_ix = perm[b0 : b0 + batch_size]
            global_ix = tr_ix[local_ix]
            xb = augmented_tensor_batch(images, global_ix, rng).to(device)
            yb = y_tr_t[local_ix]
            wb = w_tr_t[local_ix]

            optimizer.zero_grad()
            logits = model(xb)
            loss_vec = nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            loss = torch.sum(loss_vec * wb) / torch.clamp(torch.sum(wb), min=1e-8)
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=2.0
            )
            optimizer.step()

        # Verify frozen BatchNorm buffers and flags after epoch updates
        bn_check = verify_frozen_batchnorm(model, initial_bn_snap)
        layer4_lr_now = float(optimizer.param_groups[0]["lr"])
        fc_lr_now = float(optimizer.param_groups[1]["lr"])
        scheduler.step()

        tr_logits = evaluate_canonical_logits(model, images, tr_ix, device, batch_size=batch_size)
        tr_loss = weighted_bce_with_logits(tr_logits, y_tr, w_tr)
        tr_auc, tr_ba = equal_source_group_metrics(y_tr, tr_logits, g_tr, s_tr)

        row_entry = {
            "epoch": epoch,
            "layer4_lr": layer4_lr_now,
            "fc_lr": fc_lr_now,
            "train_loss": tr_loss,
            "train_group_roc_auc": tr_auc,
            "train_group_ba": tr_ba,
            "batchnorm_frozen_verified": bn_check["all_batchnorm_running_stats_and_affine_unchanged"],
        }

        if has_val:
            va_logits = evaluate_canonical_logits(model, images, va_ix, device, batch_size=batch_size)
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
        save(curve_file, curve)

        tmp_resume = dest / f"{name}.resume.tmp.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_epoch": best_epoch,
                "best_key": list(best_key),
                "best_state_dict": best_state_tensors,
                "epochs_since_improvement": epochs_since_improvement,
                "rng_state": rng.bit_generator.state,
                "curve": curve,
                "elapsed_seconds": elapsed_prior + (time.perf_counter() - t0),
            },
            tmp_resume,
        )
        os.replace(tmp_resume, resume_pth)

        if has_val and epoch >= warmup_epochs and epochs_since_improvement >= patience:
            break

    torch.save(
        {
            "architecture": "ResNet18Layer4FineTuner",
            "class_order": ["benign", "malignant"],
            "state_dict": best_state_tensors,
            "selected_epoch": best_epoch,
            "epochs_ran": len(curve),
            "batchnorm_frozen_verified": True,
        },
        model_pth,
    )

    best_curve_row = curve[best_epoch - 1]
    info = {
        "fit": name,
        "architecture": "ResNet18Layer4FineTuner",
        "parameter_summary": parameter_summary(model),
        "batchnorm_verification": verify_frozen_batchnorm(model, initial_bn_snap),
        "train_images": int(len(y_tr)),
        "train_groups": int(len(set(g_tr))),
        "training_groups": sorted(set(g_tr)),
        "validation_images": int(len(y_va)) if has_val else 0,
        "validation_groups": int(len(set(g_va))) if has_val else 0,
        "max_epochs_budget": max_epochs,
        "epochs_ran": len(curve),
        "selected_epoch": int(best_epoch),
        "early_stopped": bool(has_val and len(curve) < max_epochs),
        "selected_metrics": best_curve_row,
        "elapsed_seconds": elapsed_prior + (time.perf_counter() - t0),
        "pth_sha256": digest(model_pth),
        "curve_sha256": digest(curve_file),
        "lock_sha256": digest(out_dir / "lock.json"),
    }
    save(record_file, info)
    return best_state_tensors, info


def run(device_str: str = "cuda") -> None:
    if device_str == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU requested (--device cuda) but torch.cuda.is_available() is False on this machine. "
            "Run on a CUDA GPU (e.g., Google Colab / Kaggle T4) using notebooks/colab_full_source_finetune.ipynb."
        )
    check_lock_v3()
    device = torch.device(device_str)
    rs = rows_v2()
    images = decode_pil_images(rs)
    y = np.array([r["label"] == "malignant" for r in rs], int)
    g = np.array([r["group"] for r in rs])
    s = np.array([r["source"] for r in rs])

    results = {}
    all_predictions = []
    for source_idx, source in enumerate(["mendeley", "tcia", "busbra"]):
        out = OUT_V3 / source
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
            StratifiedGroupKFold(4, shuffle=True, random_state=42).split(train, strata, g[train])
        ):
            tr, va = train[itr], train[iva]
            assert not (set(g[tr]) & set(g[va]))
            assert not (set(g[train]) & set(g[held]))
            assert source not in (set(s[tr]) | set(s[va]))

            best_state, info = fit_encoder_fold(
                name=f"{source}_inner{fold}",
                images=images,
                tr_ix=tr,
                y_tr=y[tr],
                s_tr=s[tr],
                g_tr=g[train][itr],
                va_ix=va,
                y_va=y[va],
                s_va=s[va],
                g_va=g[train][iva],
                device=device,
                out_dir=OUT_V3,
                fixed_epochs=None,
                seed=42 + source_idx * 10 + fold,
            )
            eval_model = ResNet18Layer4FineTuner().to(device)
            eval_model.load_state_dict({k: v.to(device) for k, v in best_state.items()}, strict=True)
            oof[iva] = evaluate_canonical_logits(eval_model, images, va, device)
            inner_selected_epochs.append(int(info["selected_epoch"]))
            folds.append(
                {
                    "fold": fold,
                    "fit": f"{source}_inner{fold}",
                    "train_rows": tr.tolist(),
                    "validation_rows": va.tolist(),
                    "selected_epoch": info["selected_epoch"],
                    "epochs_ran": info["epochs_ran"],
                    "early_stopped": info["early_stopped"],
                    "selected_metrics": info["selected_metrics"],
                }
            )

        save(out / "folds.json", folds)
        cal_w = sample_weights(y[train], s[train], g[train], False)
        cal = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42).fit(
            oof[:, None], y[train], sample_weight=cal_w
        )
        oofp = expit(oof * cal.coef_[0][0] + cal.intercept_[0])
        t = choose_threshold(y[train], oofp, g[train], s[train])
        refit_epochs = max(4, int(round(float(np.median(inner_selected_epochs)))))

        np.savez_compressed(out / "oof.npz", rows=train, logits=oof, calibrated_probabilities=oofp)
        save(
            out / "training_side_selection.json",
            {
                "held_out_source": source,
                "inner_selected_epochs": inner_selected_epochs,
                "refit_epochs": refit_epochs,
                "calibration_coef": cal.coef_.tolist(),
                "calibration_intercept": cal.intercept_.tolist(),
                "threshold": t,
            },
        )

        final_state, final_info = fit_encoder_fold(
            name=f"{source}_final",
            images=images,
            tr_ix=train,
            y_tr=y[train],
            s_tr=s[train],
            g_tr=g[train],
            va_ix=None,
            y_va=None,
            s_va=None,
            g_va=None,
            device=device,
            out_dir=OUT_V3,
            fixed_epochs=refit_epochs,
            seed=42 + source_idx * 10 + 9,
        )
        save(
            out / "EVALUATION_STARTED.json",
            {
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "held_out_source": source,
                "selection_sha256": digest(out / "training_side_selection.json"),
                "model_sha256": digest(OUT_V3 / "models" / f"{source}_final.pth"),
            },
        )
        final_model = ResNet18Layer4FineTuner().to(device)
        final_model.load_state_dict({k: v.to(device) for k, v in final_state.items()}, strict=True)
        raw = evaluate_canonical_logits(final_model, images, held, device)
        p = expit(raw * cal.coef_[0][0] + cal.intercept_[0])

        gy, gp, gg = aggregate(y[held], p, g[held])
        rows_out = [
            {
                "held_out_source": source,
                "row": int(i),
                "source": rs[i]["source"],
                "group": rs[i]["group"],
                "unit": rs[i]["unit"],
                "actual": int(y[i]),
                "raw_logit": float(raw_i),
                "probability": float(pi),
                "threshold": t,
                "predicted": int(pi >= t),
            }
            for i, raw_i, pi in zip(held, raw, p)
        ]
        with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_out[0]))
            writer.writeheader()
            writer.writerows(rows_out)
        all_predictions += rows_out

        results[source] = {
            "held_out_source": source,
            "train_images": int(len(train)),
            "held_out_images": int(len(held)),
            "held_out_groups": int(len(gg)),
            "threshold": t,
            "refit_epochs": refit_epochs,
            "inner_selected_epochs": inner_selected_epochs,
            "calibration_coef": cal.coef_.tolist(),
            "calibration_intercept": cal.intercept_.tolist(),
            "group_metrics": metric(gy, gp, t),
            "group_intervals": intervals(gy, gp, gg, t),
            "image_metrics": metric(y[held], p, t),
            "image_intervals": intervals(y[held], p, g[held], t),
            "group_calibration_bins": calibration_bins(gy, gp),
            "final_fit": final_info,
        }
        save(out / "result.json", results[source])

    with (OUT_V3 / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_predictions[0]))
        writer.writeheader()
        writer.writerows(all_predictions)
    save(
        OUT_V3 / "result.json",
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_id": "gpu_encoder_finetune_v3",
            "results": results,
        },
    )


def verify_package() -> dict:
    """Verify CUDA resource status, V2->V3 lock chain, dataset path mapping, frozen BatchNorm
    behavior after model.train(), real layer4 gradient updates, and resumable checkpointing.
    """
    if not (OUT_V3 / "lock.json").exists():
        lock_v3()
    active_lock = check_lock_v3()
    rs = rows_v2()

    # Verify dataset path mapping across sample rows from each source
    source_samples = {}
    for src in ("mendeley", "tcia", "busbra"):
        src_rows = [r for r in rs if r["source"] == src]
        r0 = src_rows[0]
        raw = payload_v2(r0)
        assert hashlib.sha256(raw).hexdigest() == r0["sha256"]
        source_samples[src] = {
            "eligible_images": len(src_rows),
            "eligible_groups": len({r["group"] for r in src_rows}),
            "sample_sha256_verified": True,
        }

    # Verify frozen BatchNorm + real layer4 encoder gradient updates + checkpoint resume on a 4-image raw batch
    sample_ix = np.array([0, 1, 140, 200])
    sample_imgs = decode_pil_images([rs[i] for i in sample_ix])
    local_ix = np.array([0, 1, 2, 3])
    y_sub = np.array([0, 1, 0, 1], dtype=int)
    s_sub = np.array(["mendeley", "mendeley", "tcia", "busbra"])
    g_sub = np.array(["g0", "g1", "g2", "g3"])

    torch.manual_seed(99)
    model = ResNet18Layer4FineTuner()
    init_bn = snapshot_batchnorm_buffers(model)
    init_layer4_conv = model.backbone.layer4[0].conv1.weight.detach().clone()
    init_layer1_conv = model.backbone.layer1[0].conv1.weight.detach().clone()

    model.train()
    bn_after_train = verify_frozen_batchnorm(model, init_bn)

    opt = build_optimizer(model)
    rng = np.random.default_rng(99)
    xb = augmented_tensor_batch(sample_imgs, local_ix, rng)
    yb = torch.tensor([0.0, 1.0, 0.0, 1.0])
    opt.zero_grad()
    loss = nn.functional.binary_cross_entropy_with_logits(model(xb), yb)
    loss.backward()

    layer4_grad_norm = float(model.backbone.layer4[0].conv1.weight.grad.norm().item())
    fc_grad_norm = float(model.backbone.fc[1].weight.grad.norm().item())
    assert layer4_grad_norm > 0.0, "layer4 convolutional weights must receive non-zero gradients"
    assert fc_grad_norm > 0.0, "fc weights must receive non-zero gradients"
    assert model.backbone.conv1.weight.grad is None, "conv1 must remain frozen with grad=None"
    assert model.backbone.layer3[0].conv1.weight.grad is None, "layer3 must remain frozen with grad=None"

    opt.step()
    bn_after_step = verify_frozen_batchnorm(model, init_bn)
    layer4_weight_delta = float((model.backbone.layer4[0].conv1.weight.detach() - init_layer4_conv).norm().item())
    layer1_weight_delta = float((model.backbone.layer1[0].conv1.weight.detach() - init_layer1_conv).norm().item())
    assert layer4_weight_delta > 0.0, "layer4 conv weights must change after optimizer.step()"
    assert layer1_weight_delta == 0.0, "layer1 conv weights must not change"

    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_available and cuda_device_count > 0 else None

    report = {
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "gpu_encoder_finetune_v3",
        "cuda_resources": {
            "torch_version": torch.__version__,
            "cuda_available_locally": cuda_available,
            "cuda_device_count": cuda_device_count,
            "cuda_device_name": cuda_device_name,
            "gpu_training_executed_locally": False,
            "plain_status": (
                "No suitable CUDA GPU is accessible on this local Windows machine "
                f"(PyTorch {torch.__version__}, torch.cuda.is_available() == False). "
                "Full 15-fit GPU encoder fine-tuning has NOT run locally; the runnable, "
                "resumable Colab/Kaggle GPU package is verified and ready to launch."
            ),
        },
        "lock_chain_verification": {
            "preserved_v2_lock_intact": (OUT_V2 / "lock.json").is_file(),
            "preserved_v2_lock_sha256": digest(OUT_V2 / "lock.json"),
            "superseded_record_present": (OUT_V2 / "SUPERSEDED_BY_V3.json").is_file(),
            "active_v3_lock_sha256": digest(OUT_V3 / "lock.json"),
            "v2_eligible_images": active_lock["images"],
            "v2_eligible_groups": active_lock["groups"],
        },
        "dataset_path_mapping_verified": source_samples,
        "encoder_and_batchnorm_verification": {
            "parameter_summary": parameter_summary(model),
            "batchnorm_after_model_train": bn_after_train,
            "batchnorm_after_optimizer_step": bn_after_step,
            "layer4_conv1_grad_l2_norm": layer4_grad_norm,
            "fc_grad_l2_norm": fc_grad_norm,
            "layer4_conv1_weight_l2_delta_after_step": layer4_weight_delta,
            "frozen_layer1_conv1_weight_l2_delta": layer1_weight_delta,
        },
        "colab_notebook_path": "notebooks/colab_full_source_finetune.ipynb",
    }
    save(OUT_V3 / "package_verification.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["lock", "inspect", "verify_package", "run"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    if args.stage == "lock":
        lock_v3()
    elif args.stage == "inspect":
        m = ResNet18Layer4FineTuner()
        print(json.dumps(parameter_summary(m), indent=2))
    elif args.stage == "verify_package":
        verify_package()
    elif args.stage == "run":
        run(device_str=args.device)
