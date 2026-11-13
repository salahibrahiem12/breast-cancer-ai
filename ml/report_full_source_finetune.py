"""Package measured full-eligible neural-head adaptation results (full_source_finetune_v1)
alongside the frozen-encoder linear baseline comparison, without another fit.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path("reports/full_source_finetune_v1")
BASELINE_ROOT = Path("reports/full_source_baseline_v1")


def run() -> None:
    result = json.loads((ROOT / "result.json").read_text(encoding="utf-8"))["results"]
    baseline = json.loads((BASELINE_ROOT / "result.json").read_text(encoding="utf-8"))["results"]
    verified = json.loads((ROOT / "verification.json").read_text(encoding="utf-8"))
    assert verified["intervals_reproduced"] and not verified["serving_status"]["inference_enabled"]

    rows = json.loads(Path("reports/design_audit_v1/eligible_manifest.json").read_text(encoding="utf-8"))
    receipt = json.loads((ROOT / "aug_feature_receipt.json").read_text(encoding="utf-8"))
    lock_ver = verified["lock_audit_verification"]
    param_ver = verified["trainable_parameter_and_gradient_verification"]
    ckpt_summaries = verified["checkpoint_parameter_summaries"]

    records = []

    def record(held: str, stage: str, ix: list[int], fold: str | int = "") -> None:
        selected = [rows[i] for i in ix]
        for src in ["mendeley", "tcia", "busbra"]:
            for label in ["benign", "malignant"]:
                rs = [r for r in selected if r["source"] == src and r["label"] == label]
                records.append(
                    {
                        "held_out_source": held,
                        "stage": stage,
                        "inner_fold": fold,
                        "source": src,
                        "class": label,
                        "images": len(rs),
                        "known_units": len({r["unit"] for r in rs}),
                        "global_groups": len({r["group"] for r in rs}),
                    }
                )

    for held, r in result.items():
        train = r["training_rows"]
        test = [i for i, p in enumerate(rows) if p["source"] == held]
        for stage in ["final_train", "oof_calibration", "oof_threshold"]:
            record(held, stage, train)
        record(held, "held_out_evaluation", test)
        for f in json.loads((ROOT / held / "folds.json").read_text(encoding="utf-8")):
            record(held, "inner_train", f["train_rows"], f["fold"])
            record(held, "inner_oof_validation", f["validation_rows"], f["fold"])

    with (ROOT / "phase_counts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for col, source in enumerate(["mendeley", "tcia", "busbra"]):
        ax_loss = axes[0, col]
        ax_auc = axes[1, col]
        for fold in range(4):
            curve = json.loads((ROOT / "models" / f"{source}_inner{fold}.curve.json").read_text(encoding="utf-8"))
            info = json.loads((ROOT / "models" / f"{source}_inner{fold}.json").read_text(encoding="utf-8"))
            epochs = [c["epoch"] for c in curve]
            ax_loss.plot(epochs, [c["train_loss"] for c in curve], linestyle="--", alpha=0.55, label=f"F{fold} train")
            ax_loss.plot(
                epochs,
                [c["val_loss"] for c in curve],
                linestyle="-",
                linewidth=1.5,
                label=f"F{fold} val (sel={info['selected_epoch']})",
            )
            ax_auc.plot(
                epochs,
                [c["val_group_roc_auc"] for c in curve],
                linestyle="-",
                linewidth=1.5,
                label=f"F{fold} val AUC",
            )
        final_curve = json.loads((ROOT / "models" / f"{source}_final.curve.json").read_text(encoding="utf-8"))
        ax_loss.plot(
            [c["epoch"] for c in final_curve],
            [c["train_loss"] for c in final_curve],
            color="black",
            linewidth=2.0,
            label="Final refit train",
        )
        ax_loss.set_title(f"Held-out: {source} (weighted BCE loss)")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Weighted BCE loss")
        ax_loss.legend(fontsize=7, ncol=2)
        ax_auc.set_title(f"Held-out: {source} (inner val group ROC-AUC)")
        ax_auc.set_xlabel("Epoch")
        ax_auc.set_ylabel("Group ROC-AUC")
        ax_auc.set_ylim(0.35, 1.0)
        ax_auc.legend(fontsize=7, ncol=2)

    fig.suptitle("Full-data neural-head adaptation on frozen ResNet-18 features (full_source_finetune_v1)")
    fig.savefig(ROOT / "training_validation_curves.png", dpi=160)
    plt.close(fig)

    def fmt(v: float | None) -> str:
        return "undefined" if v is None else f"{v:.3f}"

    def interval(r: dict, k: str) -> str:
        ci = r["group_intervals"][k]
        return f"{fmt(r['group_metrics'][k])} [{fmt(ci['lower_95'])}, {fmt(ci['upper_95'])}]"

    text = f"""# Full-eligible neural-head adaptation on frozen features (`full_source_finetune_v1`): measured results

Completed 4 November 2026 under the predeclared protocol in
[`FULL_SOURCE_FINETUNE_PROTOCOL_V1.md`](FULL_SOURCE_FINETUNE_PROTOCOL_V1.md)
and compared directly against the full-eligible frozen linear baseline
([`FULL_SOURCE_BASELINE_RESULTS_V1.md`](FULL_SOURCE_BASELINE_RESULTS_V1.md)).
Public classification remains disabled (`BREASTCARE_CLASSIFICATION_ENABLED=false`).
All three breast-ultrasound cohorts (Mendeley, TCIA BrEaST, BUS-BRA) have already
been inspected; these are exploratory cross-source development results, not an
untouched clinical validation.

## 1. Exact Scope: Neural-Head Training on Cached Frozen Features vs. Encoder Fine-Tuning

To be completely transparent about what is trained in `full_source_finetune_v1` and how verification distinguishes parameter flags from historical training updates:
1. **Four distinct concepts verified in `ml/verify_full_source_finetune.py`:**
   - **(a) `requires_grad` flags on standalone module objects (and resolution of `task-329`):**
     `net()` in `ml/cross_source.py` instantiates `resnet18(weights=None)`, loads `pretrained_resnet18.pth`
     (SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`), replaces `fc` with
     `Identity()`, and calls `.eval()`. In PyTorch, `.eval()` sets `module.training = False` (disabling dropout
     and freezing BatchNorm running statistics) but leaves `param.requires_grad == True` on the standalone
     `net()` object (`11,176,512` parameters). In `task-329`, an initial assertion in `ml/verify_full_source_finetune.py`
     (`assert sum(p.numel() for p in backbone.parameters() if p.requires_grad) == 0`) failed because it checked
     the default `requires_grad` flag on a newly constructed `net()` object rather than checking whether `net()`
     participated in optimization. That failure is preserved in `reports/full_source_finetune_v1/lock_audit_trail.json`
     and `reports/full_source_finetune_v1/verification.json`.
   - **(b) Optimizer parameter-group membership:**
     During `fit_adapter()` (`ml/full_source_finetune.py` line 394), `net()` is never instantiated;
     `torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)` receives only the `264,963` parameters
     of `ResidualAdapterHead` (`0` `ResNet-18` parameters).
   - **(c) Autograd gradient computation:**
     During `aug_features()`, `net()` executes inside `with torch.inference_mode():` and writes detached
     `float32` numpy arrays (`aug_features.npz`), so no autograd graph or backward pass ever touches `ResNet-18`.
     Within `ResidualAdapterHead`, because `gate` is initialized to `0.0` (`tanh(0) = 0`), a synthetic 2-step
     architectural check shows that `norm`, `fc1`, and `fc2` receive `0.0` gradient on step 0 while `gate` and
     `classifier` receive non-zero gradients, and all 9 parameter tensors receive non-zero gradients on step ≥ 1
     once `gate != 0`. **Note:** A freshly constructed diagnostic probe only illustrates this mathematical
     property of the architecture; it does **not** prove historical training behavior.
   - **(d) Actual historical parameter changes in the 15 saved checkpoints on disk:**
     Because `fit_adapter()` seeds PyTorch (`torch.manual_seed(seed)`, where `seed = 42 + source_idx * 10 + fold`
     for inner folds `0..3` and `42 + source_idx * 10 + 9` for `{source}_final`) immediately before instantiating
     `ResidualAdapterHead()`, `ml/verify_full_source_finetune.py` reconstructs the exact initial parameter tensors
     `theta_init` for all 15 fits and compares them directly against the saved historical checkpoints
     `reports/full_source_finetune_v1/models/*.npz`. Every one of the 9 parameter tensors (`gate`, `norm.weight`,
     `norm.bias`, `fc1.weight`, `fc1.bias`, `fc2.weight`, `fc2.bias`, `classifier.weight`, `classifier.bias`)
     has verified non-zero L2 change `||theta_saved - theta_init||_2 > 0` across all 15 saved checkpoints.
2. **Trainable module breakdown (`264,963` parameters in `ResidualAdapterHead`):**

| Trainable Module | Shape / Definition | Parameters in `AdamW` | Architectural Step-0 Grad (`gate=0`) | Historical Saved Checkpoints `||theta_saved - theta_init||_2 > 0` |
|---|---|---:|---|---|
| `norm` | `LayerNorm(512)` (`weight` + `bias`) | `1,024` | `0.0` (gated by `tanh(0)=0`) | Verified (`>0` in all 15 saved checkpoints on disk) |
| `fc1` | `Linear(512 -> 256)` (`weight` + `bias`) | `131,328` | `0.0` (gated by `tanh(0)=0`) | Verified (`>0` in all 15 saved checkpoints on disk) |
| `fc2` | `Linear(256 -> 512)` (`weight` + `bias`) | `131,584` | `0.0` (gated by `tanh(0)=0`) | Verified (`>0` in all 15 saved checkpoints on disk) |
| `gate` | Scalar `nn.Parameter(zeros(1))` (`h = z + tanh(gate) * delta`) | `1` | Non-zero (`sech^2(0) = 1.0`) | Verified (`|gate| > 0` in all 15 saved checkpoints on disk) |
| `classifier` | `Linear(512 -> 2)` (`weight` + `bias`) | `1,026` | Non-zero on step 0 | Verified (`>0` in all 15 saved checkpoints on disk) |
| **Total (`ResidualAdapterHead`)** | **Nonlinear residual bottleneck head** | **`264,963`** | **Architectural property** | **All 15 historical checkpoints verified** |

- **Separately versioned GPU encoder fine-tuning (`gpu_encoder_finetune_v2` / `v3`):**
  Because local hardware is CPU-only (`torch.cuda.is_available() == False`), end-to-end convolutional
  encoder fine-tuning has **not** run locally and is packaged as a separately versioned GPU experiment
  ([`GPU_ENCODER_FINETUNE_PROTOCOL_V3.md`](GPU_ENCODER_FINETUNE_PROTOCOL_V3.md), `ml/gpu_encoder_finetune_v2.py`,
  `notebooks/colab_full_source_finetune.ipynb`) without relabeling `full_source_finetune_v1`.

## 2. Protocol Lock Audit Trail and Augmentation / Fold Isolation

1. **Preserved protocol locks (`lock_initial_20261104T090149Z.json` and `lock.json`):**
   - Initial lock (`task-204`, `{lock_ver['initial_locked_utc']}`, SHA-256 `{lock_ver['initial_lock_sha256']}`)
     recorded `Path(__file__)` as an absolute path key (`D:/breast_cancer/ml/full_source_finetune.py`).
   - Augmented feature extraction (`task-212`) completed across all 2,232 eligible images in
     `{receipt['elapsed_seconds']:.1f}s` before any model fitting began.
   - `task-278` aborted immediately inside `check_lock()` before loading features or creating
     `models/`; **zero model fits or outcomes existed before the lock refresh**.
   - Active lock (`task-292`, `{lock_ver['active_locked_utc']}`, SHA-256 `{lock_ver['active_lock_sha256']}`)
     normalized the script key to relative `ml/full_source_finetune.py` and added a guard
     preventing relocking once `models/` exists. Every dataset manifest hash, protocol hash,
     pretrained weight hash, canonical feature hash, and hyperparameter is **100% identical**
     between both preserved lock files (`reports/full_source_finetune_v1/lock_audit_trail.json`).
2. **Augmentation and partition isolation:**
   - Every original image (`x_can[i]`) and its deterministic ultrasound-augmented view (`x_aug[i]`)
     are verified by input payload `sha256` at identical row index `i`, keeping them in the
     exact same `global_group` and training partition (`tr`).
   - Inner validation (`va`) and held-out source evaluation (`held`) use strictly the
     unaugmented canonical features (`x_can`) and locked group-mean probability aggregation.
   - Weighted `StandardScaler`, early-stopping checkpoint selection (`selected_epoch`),
     OOF Platt calibration (`LogisticRegression(C=1)`), and operating threshold selection
     (`choose_threshold`) are fit strictly on training-side data (`s != held`).

## 3. Epoch-Level Training, Early Stopping, and Historical Parameter Deltas Across All 15 Fits

| Held source | Inner selected epochs (folds 0–3) | Inner epochs ran (folds 0–3) | Final refit epochs | Final `tanh(gate)` | Final `||delta_fc1||_2` / `||delta_fc2||_2` / `||delta_cls||_2` from init | Training OOF group AUC |
|---|---|---|---:|---:|---|---:|
"""
    for held, r in result.items():
        folds = json.loads((ROOT / held / "folds.json").read_text(encoding="utf-8"))
        ran_list = [int(f["epochs_ran"]) for f in folds]
        ck = ckpt_summaries[f"{held}_final"]
        dl2 = ck["historical_l2_delta_from_init"]
        text += (
            f"| {held} | `{r['inner_selected_epochs']}` | `{ran_list}` | "
            f"{r['refit_epochs']} | `{ck['tanh_gate']:+.4f}` | "
            f"`{dl2['fc1.weight']:.3f}` / `{dl2['fc2.weight']:.3f}` / `{dl2['classifier.weight']:.3f}` | "
            f"{r['training_oof_group_metrics']['roc_auc']:.3f} |\n"
        )

    text += """
## 4. Primary Group-Level Results and Baseline Comparison

Group balanced accuracy (`BA`) at the fixed training-side threshold is the
predeclared primary endpoint. Confidence intervals are 95% percentile group-bootstrap
intervals from 1,000 seeded draws (`seed=42`) resampling whole global groups.

| Held source | Model | Group BA [95% CI] | Group ROC-AUC [95% CI] | Sensitivity | Specificity | Malignant AP | Binary Brier | Top-label ECE |
|---|---|---|---:|---:|---:|---:|---:|---:|
"""
    for held in ["mendeley", "tcia", "busbra"]:
        b = baseline[held]
        bm = b["group_metrics"]
        r = result[held]
        rm = r["group_metrics"]
        text += (
            f"| {held} | Frozen linear baseline (`v1`) | {interval(b, 'balanced_accuracy')} | "
            f"{interval(b, 'roc_auc')} | {fmt(bm['sensitivity'])} | {fmt(bm['specificity'])} | "
            f"{fmt(bm['pr_auc_average_precision'])} | {fmt(bm['binary_brier'])} | {fmt(bm['top_label_ece_10_bins'])} |\n"
        )
        text += (
            f"| {held} | Neural adapter head (`v1`) | {interval(r, 'balanced_accuracy')} | "
            f"{interval(r, 'roc_auc')} | {fmt(rm['sensitivity'])} | {fmt(rm['specificity'])} | "
            f"{fmt(rm['pr_auc_average_precision'])} | {fmt(rm['binary_brier'])} | {fmt(rm['top_label_ece_10_bins'])} |\n"
        )

    text += """
### Confusion Matrices and Calibration Transfer

| Held source | Group confusion `[[TN, FP], [FN, TP]]` | Threshold | Calibration coef / intercept | PPV | NPV | Held majority accuracy | Constant-prevalence Brier |
|---|---|---:|---|---:|---:|---:|---:|
"""
    for held, r in result.items():
        m = r["group_metrics"]
        base = r["majority_baselines"]
        cm = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
        text += (
            f"| {held} | `{cm}` | {r['threshold']:.6f} | "
            f"{r['calibration_coef'][0][0]:.4f} / {r['calibration_intercept'][0]:.4f} | "
            f"{fmt(m['precision_ppv'])} | {fmt(m['npv'])} | "
            f"{base['held_group_majority_accuracy']:.3f} | "
            f"{base['training_prevalence_constant_held_group_brier']:.3f} |\n"
        )

    text += """
## 5. Image-Level Secondary Endpoints

| Held source | Image BA [95% CI] | Image ROC-AUC [95% CI] | Sensitivity | Specificity | Malignant AP | Binary Brier | Top-label ECE |
|---|---|---|---:|---:|---:|---:|---:|
"""
    for held, r in result.items():
        im = r["image_metrics"]
        ici = r["image_intervals"]
        ba_str = f"{fmt(im['balanced_accuracy'])} [{fmt(ici['balanced_accuracy']['lower_95'])}, {fmt(ici['balanced_accuracy']['upper_95'])}]"
        auc_str = f"{fmt(im['roc_auc'])} [{fmt(ici['roc_auc']['lower_95'])}, {fmt(ici['roc_auc']['upper_95'])}]"
        text += (
            f"| {held} | {ba_str} | {auc_str} | {fmt(im['sensitivity'])} | {fmt(im['specificity'])} | "
            f"{fmt(im['pr_auc_average_precision'])} | {fmt(im['binary_brier'])} | {fmt(im['top_label_ece_10_bins'])} |\n"
        )

    text += """
## 6. Scientific Interpretation: Measured External Degradation vs. Unproved Hypotheses

1. **Measured external outcome (the neural adapter performed worse externally than the linear baseline):**
   Across the inner validation folds (`StratifiedGroupKFold(4)` within the two
   training sources), `ResidualAdapterHead` achieved `0.69–0.89` validation group
   ROC-AUC and drove training weighted BCE loss down to `0.26–0.55`. However, when
   evaluated on the unseen held-out third source, `ResidualAdapterHead` performed
   worse externally than the simpler L2-regularized linear logistic regression baseline
   (`full_source_baseline_v1`) across all three held-out sources:
   - Mendeley: group ROC-AUC `0.448` vs `0.706`; group BA `0.465` vs `0.573`.
   - TCIA: group ROC-AUC `0.671` vs `0.688`; group BA `0.562` vs `0.644`.
   - BUS-BRA: group ROC-AUC `0.573` vs `0.594`; group BA `0.518` vs `0.570`.
2. **Hypothesis vs. established explanation:**
   Although [`EXPERIMENT_DESIGN_AUDIT_V1.md`](EXPERIMENT_DESIGN_AUDIT_V1.md) showed that
   acquisition source (`mendeley`, `tcia`, `busbra`) is `99.1%` predictable from
   frozen `ResNet-18` features even within a single diagnosis class, **source-specific
   overfitting in frozen feature space is an unproved hypothesis, not an established
   causal explanation** for why `ResidualAdapterHead` performed worse externally. Other
   plausible contributors include optimization dynamics of the zero-initialized residual
   gate (`tanh(gate)`), feature-space view mixing / speckle perturbation hyperparameters,
   inner-fold early-stopping variance, and fixed-feature representation limits.
3. **Why public classification remains disabled (`BREASTCARE_CLASSIFICATION_ENABLED=false`):**
   Neither the frozen linear baseline nor the nonlinear neural-head adaptation
   meets a defensible standard for clinical or public diagnostic use; all three
   cohorts are inspected development sources, GPU encoder fine-tuning has not yet
   been executed on CUDA hardware, and input-eligibility / OOD gating remains an
   unresolved workstream (the failed leave-one-source-out Mahalanobis threshold
   diagnostic in `reports/eligibility_ood_candidate_audit_v1.json` only shows that a
   simple 95% in-distribution Mahalanobis cutoff fails under cross-source shift, not
   that every OOD method fails).

## 7. Reproducibility and Verification Commands

```powershell
.\\.venv\\Scripts\\python.exe -m ml.verify_full_source_finetune --intervals
.\\.venv\\Scripts\\python.exe -m ml.report_full_source_finetune
```
"""

    Path("docs/FULL_SOURCE_FINETUNE_RESULTS_V1.md").write_text(text, encoding="utf-8")

    files = [p for p in ROOT.rglob("*") if p.is_file() and p.name != "artifact_hashes.json"]
    files += [
        Path(p)
        for p in [
            "docs/FULL_SOURCE_FINETUNE_PROTOCOL_V1.md",
            "docs/FULL_SOURCE_FINETUNE_RESULTS_V1.md",
            "docs/GPU_ENCODER_FINETUNE_PROTOCOL_V2.md",
            "ml/full_source_finetune.py",
            "ml/verify_full_source_finetune.py",
            "ml/report_full_source_finetune.py",
            "ml/gpu_encoder_finetune_v2.py",
        ]
    ]
    hashes = {str(p).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (ROOT / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    with ZipFile("reports/full_source_finetune_v1_bundle.zip", "w", ZIP_DEFLATED) as bundle:
        for p in files + [ROOT / "artifact_hashes.json"]:
            bundle.write(p, str(p).replace("\\", "/"))
    print("Full-source neural-head adaptation report, learning curves, phase counts, hashes, and bundle saved.")


if __name__ == "__main__":
    run()
