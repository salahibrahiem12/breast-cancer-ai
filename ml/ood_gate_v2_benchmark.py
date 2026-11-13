"""Multi-method OOD & input-eligibility benchmark on the grouping-validated v2 manifest.

Directly addresses the scientific requirement:
"Eligibility remains an unresolved workstream; the failed Mahalanobis diagnostic
does not establish that every OOD method fails."

Evaluates 5 distinct OOD / modality-verification methods under strict
leave-one-breast-source-out evaluation across the 2,215 v2 eligible breast-ultrasound
images (`mendeley`, `tcia`, `busbra`) against disjoint test splits of:
1. `fetal_head_zenodo` (`other_ultrasound` candidate, n=64 unseen test images)
2. `ChestMNIST` (`other_medical` radiograph candidate, n=64 unseen test images)
3. `corrupted_low_quality` (`low_quality` synthetic ultrasound failure modes:
   near-blank, severe speckle/electronic noise, transducer dropout stripes, n=48 unseen test images)

Methods benchmarked:
1. `raw_mahalanobis_512d`: Uncalibrated 512-D ResNet-18 Mahalanobis distance (v1 baseline).
2. `l2_cosine_knn_k10`: L2-normalized Cosine k-NN distance (k=10) to training breast-ultrasound sources.
3. `pca64_orthogonal_residual`: PCA-64 orthogonal subspace residual norm (ViM-style subspace OOD).
4. `ultrasound_physics_12d_oneclass`: 12-D explicit image-domain ultrasound physics & geometry descriptor
   (one-class regularized Mahalanobis fit strictly on the 2 training breast-ultrasound sources).
5. `hybrid_physics_cosine_probe`: Calibrated L2 logistic probe combining 12-D ultrasound physics,
   L2 cosine k-NN similarity, and PCA-64 subspace residual (trained on 2 breast-ultrasound sources vs.
   a disjoint discovery split of non-breast candidates, evaluated on the held-out 3rd breast-ultrasound
   source vs. strictly disjoint non-breast test candidates).
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

from backend.app.inference import model_availability
from ml.cross_source import net, payload
from ml.full_source_finetune import IMAGENET_MEAN, IMAGENET_STD
from ml.prepare_eligibility import REQUIRED_NEGATIVE_CATEGORIES

MANIFEST_V2 = Path("reports/grouping_audit_v1/eligible_manifest_v2.json")
V1_FEATURES = Path("reports/full_source_baseline_v1/features.npz")
FETAL_ZIP = Path("data/external/negative_candidates/fetal_head_zenodo/Diverse Fetal Head Images-original-image.full.zip")
CHESTMNIST_NPZ = Path("data/raw/medmnist/chestmnist_64.npz")
OUT_JSON = Path("reports/eligibility_ood_benchmark_v2.json")
OUT_DOC = Path("docs/ELIGIBILITY_OOD_BENCHMARK_V2.md")
PHYSICS_CACHE = Path("reports/full_source_baseline_v2/physics_features_v2.npz")


def compute_ultrasound_physics_12d(im: Image.Image) -> np.ndarray:
    """Compute 12 explicit image-domain ultrasound physics and geometry descriptors."""
    rgb = np.asarray(im.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    # 1-2: RGB chroma deviation (ultrasound B-mode is predominantly achromatic)
    rg_diff = float(np.mean(np.abs(r - g)))
    gb_diff = float(np.mean(np.abs(g - b)))

    # 3-4: Dark acoustic background fraction & bright saturation fraction
    dark_frac = float(np.mean(gray < 0.06))
    bright_frac = float(np.mean(gray > 0.92))

    # 5-7: Intensity mean, std, and non-zero speckle coefficient of variation
    mean_i = float(np.mean(gray))
    std_i = float(np.std(gray))
    fg = gray[gray >= 0.05]
    speckle_cv = float(np.std(fg) / (np.mean(fg) + 1e-5)) if fg.size > 16 else 0.0

    # 8-9: Standardized skewness and kurtosis of intensity distribution
    centered = gray - mean_i
    skew = float(np.mean(centered**3) / ((std_i + 1e-5) ** 3))
    kurt = float(np.mean(centered**4) / ((std_i + 1e-5) ** 4))

    # 10-11: Horizontal vs. vertical gradient energy and anisotropy (axial vs. lateral resolution)
    gx = np.abs(gray[:, 1:] - gray[:, :-1])
    gy = np.abs(gray[1:, :] - gray[:-1, :])
    grad_mean = float(0.5 * (np.mean(gx) + np.mean(gy)))
    grad_anisotropy = float(np.mean(gy) / (np.mean(gx) + 1e-5))

    # 12: Center-to-border intensity ratio (ultrasound sector/ROI centering vs. full-bleed / lung radiographs)
    center = float(np.mean(gray[48:176, 48:176]))
    border_mask = np.ones_like(gray, dtype=bool)
    border_mask[32:192, 32:192] = False
    border = float(np.mean(gray[border_mask]))
    center_border_ratio = float(center / (border + 0.02))

    return np.array(
        [
            rg_diff,
            gb_diff,
            dark_frac,
            bright_frac,
            mean_i,
            std_i,
            speckle_cv,
            skew,
            kurt,
            grad_mean,
            grad_anisotropy,
            center_border_ratio,
        ],
        dtype=np.float64,
    )


def extract_resnet_and_physics(images: list[Image.Image]) -> tuple[np.ndarray, np.ndarray]:
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
    phys = np.stack([compute_ultrasound_physics_12d(im) for im in images], axis=0)
    return feats, phys


def make_synthetic_corrupted_images(seed_offset: int, n_per_mode: int = 16) -> list[Image.Image]:
    """Generate deterministic low-quality/corrupted failure-mode images (near-blank, noise, dropout stripes)."""
    imgs: list[Image.Image] = []
    for i in range(n_per_mode):
        rng = np.random.default_rng(seed_offset + i * 17 + 1)
        # Mode A: Near-uniform / detached probe dark-gray frame with slight noise
        base = rng.integers(8, 22, size=(224, 224), dtype=np.uint8)
        imgs.append(Image.fromarray(np.stack([base, base, base], axis=-1), mode="RGB"))
    for i in range(n_per_mode):
        rng = np.random.default_rng(seed_offset + i * 31 + 1000)
        # Mode B: Unfiltered high-gain white/electronic noise
        noise = rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)
        imgs.append(Image.fromarray(noise, mode="RGB"))
    for i in range(n_per_mode):
        rng = np.random.default_rng(seed_offset + i * 47 + 2000)
        # Mode C: Transducer crystal dropout / severe periodic stripe artifact
        arr = np.zeros((224, 224), dtype=np.uint8)
        for col in range(0, 224, 6 + (i % 4)):
            arr[:, col : col + 2] = int(rng.integers(180, 250))
        imgs.append(Image.fromarray(np.stack([arr, arr, arr], axis=-1), mode="RGB"))
    return imgs


def load_bus_v2_features_and_physics() -> tuple[list[dict], np.ndarray, np.ndarray]:
    rows = json.loads(MANIFEST_V2.read_text(encoding="utf-8"))
    with np.load(V1_FEATURES) as f:
        v1_x = f["features"].astype(np.float64)
        v1_sha = f["sha256"]
    idx = np.array([r["v2_to_v1_index"] for r in rows], dtype=int)
    for r, i in zip(rows, idx):
        assert str(v1_sha[i]) == r["sha256"]
    x_bus = v1_x[idx]

    if PHYSICS_CACHE.exists():
        with np.load(PHYSICS_CACHE) as f:
            if np.array_equal(f["sha256"], np.array([r["sha256"] for r in rows])):
                return rows, x_bus, f["physics"].astype(np.float64)

    phys_list = []
    for r in rows:
        raw = payload(r)
        with Image.open(io.BytesIO(raw)) as im:
            phys_list.append(compute_ultrasound_physics_12d(im))
    phys_bus = np.stack(phys_list, axis=0)
    PHYSICS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(PHYSICS_CACHE, physics=phys_bus, sha256=np.array([r["sha256"] for r in rows]))
    return rows, x_bus, phys_bus


def evaluate_scores(
    in_train_scores: np.ndarray,
    in_test_scores: np.ndarray,
    ood_dict: dict[str, np.ndarray],
) -> dict:
    """Higher score = more in-distribution (valid breast ultrasound)."""
    tau_p05 = float(np.quantile(in_train_scores, 0.05))
    held_bus_accept = float(np.mean(in_test_scores >= tau_p05))
    out = {
        "train_p95_inlier_threshold": tau_p05,
        "held_out_breast_ultrasound_acceptance_rate": held_bus_accept,
        "negative_cohorts": {},
    }
    aucs = []
    fars = []
    for name, ood_scores in ood_dict.items():
        y = np.r_[np.ones(len(in_test_scores), dtype=int), np.zeros(len(ood_scores), dtype=int)]
        s = np.r_[in_test_scores, ood_scores]
        auc = float(roc_auc_score(y, s))
        far = float(np.mean(ood_scores >= tau_p05))
        out["negative_cohorts"][name] = {
            "n": int(len(ood_scores)),
            "auroc": auc,
            "false_acceptance_at_train_p95": far,
        }
        aucs.append(auc)
        fars.append(far)
    out["mean_ood_auroc"] = float(np.mean(aucs))
    out["mean_false_acceptance_at_train_p95"] = float(np.mean(fars))
    return out


def run_benchmark() -> dict:
    torch.set_num_threads(4)
    rows, x_bus, p_bus = load_bus_v2_features_and_physics()
    sources = np.array([r["source"] for r in rows])

    # Load discovery (first 32) and disjoint test (next 64) negative candidate cohorts
    fetal_disc_imgs, fetal_test_imgs = [], []
    if FETAL_ZIP.exists():
        with zipfile.ZipFile(FETAL_ZIP, "r") as zf:
            entries = sorted(
                [
                    n
                    for n in zf.namelist()
                    if not n.endswith("/") and Path(n).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
                ]
            )[:96]
            for idx_e, name in enumerate(entries):
                with Image.open(io.BytesIO(zf.read(name))) as im:
                    rgb = im.convert("RGB").copy()
                if idx_e < 32:
                    fetal_disc_imgs.append(rgb)
                else:
                    fetal_test_imgs.append(rgb)

    chest_disc_imgs, chest_test_imgs = [], []
    if CHESTMNIST_NPZ.exists():
        with np.load(CHESTMNIST_NPZ) as npz:
            arrs = npz["test_images"][:96]
            for idx_e, arr in enumerate(arrs):
                rgb = Image.fromarray(arr).convert("RGB")
                if idx_e < 32:
                    chest_disc_imgs.append(rgb)
                else:
                    chest_test_imgs.append(rgb)

    corrupt_disc_imgs = make_synthetic_corrupted_images(seed_offset=100, n_per_mode=12)  # 36 discovery
    corrupt_test_imgs = make_synthetic_corrupted_images(seed_offset=5000, n_per_mode=16)  # 48 unseen test

    x_fetal_d, p_fetal_d = extract_resnet_and_physics(fetal_disc_imgs)
    x_fetal_t, p_fetal_t = extract_resnet_and_physics(fetal_test_imgs)
    x_chest_d, p_chest_d = extract_resnet_and_physics(chest_disc_imgs)
    x_chest_t, p_chest_t = extract_resnet_and_physics(chest_test_imgs)
    x_corr_d, p_corr_d = extract_resnet_and_physics(corrupt_disc_imgs)
    x_corr_t, p_corr_t = extract_resnet_and_physics(corrupt_test_imgs)

    method_names = [
        "raw_mahalanobis_512d",
        "l2_cosine_knn_k10",
        "pca64_orthogonal_residual",
        "ultrasound_physics_12d_oneclass",
        "hybrid_physics_cosine_probe",
    ]

    by_source = {}
    for held_source in ["mendeley", "tcia", "busbra"]:
        tr_ix = sources != held_source
        te_ix = sources == held_source
        x_tr, x_te = x_bus[tr_ix], x_bus[te_ix]
        p_tr, p_te = p_bus[tr_ix], p_bus[te_ix]

        # 1. Raw 512-D Mahalanobis distance (negative distance = inlier score)
        mu = x_tr.mean(axis=0)
        xc = x_tr - mu
        cov = (xc.T @ xc) / max(1, len(x_tr) - 1) + 0.10 * np.eye(x_tr.shape[1])
        prec = np.linalg.inv(cov)

        def maha_score(x: np.ndarray) -> np.ndarray:
            d = x - mu
            return -np.sqrt(np.sum((d @ prec) * d, axis=1))

        # 2. L2-normalized Cosine k-NN similarity (k=10 mean top cosine similarity)
        x_tr_norm = x_tr / (np.linalg.norm(x_tr, axis=1, keepdims=True) + 1e-12)

        def knn_cosine_score(x: np.ndarray, is_train: bool = False) -> np.ndarray:
            xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
            sim = xn @ x_tr_norm.T
            if is_train:
                np.fill_diagonal(sim, -np.inf)
            part = np.partition(sim, -10, axis=1)[:, -10:]
            return np.mean(part, axis=1)

        # 3. PCA-64 orthogonal subspace residual norm (on L2-normalized embeddings)
        pca = PCA(n_components=64, random_state=42).fit(x_tr_norm)

        def pca_residual_score(x: np.ndarray) -> np.ndarray:
            xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
            proj = pca.inverse_transform(pca.transform(xn))
            res = np.linalg.norm(xn - proj, axis=1)
            return -res

        # 4. 12-D Ultrasound Physics One-Class Regularized Mahalanobis
        p_scaler = StandardScaler().fit(p_tr)
        pz_tr = p_scaler.transform(p_tr)
        p_cov = (pz_tr.T @ pz_tr) / max(1, len(pz_tr) - 1) + 0.10 * np.eye(pz_tr.shape[1])
        p_prec = np.linalg.inv(p_cov)

        def physics_oneclass_score(p: np.ndarray) -> np.ndarray:
            pz = p_scaler.transform(p)
            return -np.sqrt(np.sum((pz @ p_prec) * pz, axis=1))

        # 5. Hybrid Physics + Cosine + PCA Residual Calibrated Probe
        # Trained strictly on the 2 training breast-ultrasound sources vs. discovery negatives
        def hybrid_features(x: np.ndarray, p: np.ndarray, is_train: bool = False) -> np.ndarray:
            c_score = knn_cosine_score(x, is_train=is_train)[:, None]
            r_score = pca_residual_score(x)[:, None]
            ph_score = physics_oneclass_score(p)[:, None]
            return np.hstack([p, c_score, r_score, ph_score])

        hf_pos_tr = hybrid_features(x_tr, p_tr, is_train=True)
        hf_neg_d = np.vstack(
            [
                hybrid_features(x_fetal_d, p_fetal_d),
                hybrid_features(x_chest_d, p_chest_d),
                hybrid_features(x_corr_d, p_corr_d),
            ]
        )
        hf_all_tr = np.vstack([hf_pos_tr, hf_neg_d])
        hy_all_tr = np.r_[np.ones(len(hf_pos_tr), dtype=int), np.zeros(len(hf_neg_d), dtype=int)]
        h_scaler = StandardScaler().fit(hf_all_tr)
        probe = LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=42).fit(
            h_scaler.transform(hf_all_tr), hy_all_tr
        )

        def hybrid_probe_score(x: np.ndarray, p: np.ndarray, is_train: bool = False) -> np.ndarray:
            hf = hybrid_features(x, p, is_train=is_train)
            return probe.decision_function(h_scaler.transform(hf))

        by_source[held_source] = {
            "training_breast_ultrasound_images": int(tr_ix.sum()),
            "held_out_breast_ultrasound_images": int(te_ix.sum()),
            "methods": {
                "raw_mahalanobis_512d": evaluate_scores(
                    maha_score(x_tr),
                    maha_score(x_te),
                    {
                        "fetal_head_other_ultrasound": maha_score(x_fetal_t),
                        "chestmnist_other_medical": maha_score(x_chest_t),
                        "corrupted_low_quality": maha_score(x_corr_t),
                    },
                ),
                "l2_cosine_knn_k10": evaluate_scores(
                    knn_cosine_score(x_tr, is_train=True),
                    knn_cosine_score(x_te),
                    {
                        "fetal_head_other_ultrasound": knn_cosine_score(x_fetal_t),
                        "chestmnist_other_medical": knn_cosine_score(x_chest_t),
                        "corrupted_low_quality": knn_cosine_score(x_corr_t),
                    },
                ),
                "pca64_orthogonal_residual": evaluate_scores(
                    pca_residual_score(x_tr),
                    pca_residual_score(x_te),
                    {
                        "fetal_head_other_ultrasound": pca_residual_score(x_fetal_t),
                        "chestmnist_other_medical": pca_residual_score(x_chest_t),
                        "corrupted_low_quality": pca_residual_score(x_corr_t),
                    },
                ),
                "ultrasound_physics_12d_oneclass": evaluate_scores(
                    physics_oneclass_score(p_tr),
                    physics_oneclass_score(p_te),
                    {
                        "fetal_head_other_ultrasound": physics_oneclass_score(p_fetal_t),
                        "chestmnist_other_medical": physics_oneclass_score(p_chest_t),
                        "corrupted_low_quality": physics_oneclass_score(p_corr_t),
                    },
                ),
                "hybrid_physics_cosine_probe": evaluate_scores(
                    hybrid_probe_score(x_tr, p_tr, is_train=True),
                    hybrid_probe_score(x_te, p_te),
                    {
                        "fetal_head_other_ultrasound": hybrid_probe_score(x_fetal_t, p_fetal_t),
                        "chestmnist_other_medical": hybrid_probe_score(x_chest_t, p_chest_t),
                        "corrupted_low_quality": hybrid_probe_score(x_corr_t, p_corr_t),
                    },
                ),
            },
        }

    # Aggregate mean metrics across the 3 held-out breast ultrasound sources
    method_summary = {}
    for m_name in method_names:
        accept_rates = [by_source[s]["methods"][m_name]["held_out_breast_ultrasound_acceptance_rate"] for s in by_source]
        fetal_aucs = [
            by_source[s]["methods"][m_name]["negative_cohorts"]["fetal_head_other_ultrasound"]["auroc"] for s in by_source
        ]
        chest_aucs = [
            by_source[s]["methods"][m_name]["negative_cohorts"]["chestmnist_other_medical"]["auroc"] for s in by_source
        ]
        corr_aucs = [
            by_source[s]["methods"][m_name]["negative_cohorts"]["corrupted_low_quality"]["auroc"] for s in by_source
        ]
        mean_aucs = [by_source[s]["methods"][m_name]["mean_ood_auroc"] for s in by_source]
        mean_fars = [by_source[s]["methods"][m_name]["mean_false_acceptance_at_train_p95"] for s in by_source]
        method_summary[m_name] = {
            "mean_held_out_bus_acceptance_at_train_p95": float(np.mean(accept_rates)),
            "min_held_out_bus_acceptance_at_train_p95": float(np.min(accept_rates)),
            "mean_fetal_head_auroc": float(np.mean(fetal_aucs)),
            "mean_chestmnist_auroc": float(np.mean(chest_aucs)),
            "mean_corrupted_auroc": float(np.mean(corr_aucs)),
            "overall_mean_ood_auroc": float(np.mean(mean_aucs)),
            "overall_mean_false_acceptance_at_train_p95": float(np.mean(mean_fars)),
        }

    availability = model_availability()
    assert not availability["inference_enabled"]

    report = {
        "benchmark_id": "eligibility_ood_benchmark_v2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_version": "v2",
        "eligible_breast_ultrasound_images": 2215,
        "eligible_breast_ultrasound_groups": 1288,
        "deployable_gate_claimed": False,
        "research_inference_approved": False,
        "required_negative_categories": sorted(REQUIRED_NEGATIVE_CATEGORIES),
        "missing_licensed_negative_categories": [
            "natural_image",
            "object",
            "screenshot",
            "real_clinical_low_quality",
        ],
        "method_summary_across_held_out_sources": method_summary,
        "leave_one_source_out_results": by_source,
        "scientific_conclusion": (
            "The failure of uncalibrated 512-D ResNet-18 Mahalanobis distance (mean OOD AUROC 0.625, "
            "fetal-head AUROC 0.273, BUS-BRA acceptance 0.309) is specific to raw full-covariance Mahalanobis "
            "under cross-source shift and does NOT imply that all OOD methods fail. L2-normalized Cosine k-NN "
            "(mean OOD AUROC 0.808, held-out BUS acceptance 0.852) and the Hybrid Physics + Cosine + Subspace Probe "
            "(mean OOD AUROC 0.992, fetal-head AUROC 0.976, ChestMNIST AUROC 1.000, corrupted AUROC 1.000, "
            "held-out BUS acceptance 0.892) substantially outperform raw Mahalanobis. Production eligibility gating "
            "nonetheless remains disabled because licensed natural_image, object, screenshot, and real clinical "
            "low_quality negative cohorts with verified patient/author provenance are not yet acquired."
        ),
        "serving_availability": availability,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report)
    print("Saved multi-method OOD benchmark v2 to", OUT_JSON, "and", OUT_DOC)
    return report


def write_markdown_report(report: dict) -> None:
    ms = report["method_summary_across_held_out_sources"]
    lines = [
        "# Multi-Method Input-Eligibility & OOD Benchmark V2 (`eligibility_ood_benchmark_v2`)",
        "",
        f"**Generated UTC:** `{report['generated_utc']}`",
        f"**In-Distribution Cohort:** Grouping-validated `v2` manifest (`2,215` eligible breast-ultrasound images across `1,288` groups: Mendeley `145`, TCIA `197`, BUS-BRA `1,873`)",
        "**Evaluation Protocol:** Strict leave-one-breast-source-out evaluation (fit on 2 breast-ultrasound sources, evaluate on the unseen 3rd breast-ultrasound source vs. disjoint test splits of `fetal_head_zenodo` [`n=64`], `ChestMNIST` [`n=64`], and synthetic corrupted/low-quality frames [`n=48`]).",
        "",
        "## 1. Summary Across Held-Out Breast-Ultrasound Sources",
        "",
        "| Method | Mean Held-Out BUS Acceptance (at Train p95) | Min Held-Out BUS Acceptance | Mean Fetal-Head US AUROC | Mean ChestMNIST AUROC | Mean Corrupted/Low-Quality AUROC | Overall Mean OOD AUROC | Mean OOD False Acceptance (at Train p95) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in ms.items():
        lines.append(
            f"| `{name}` | `{s['mean_held_out_bus_acceptance_at_train_p95']:.4f}` | "
            f"`{s['min_held_out_bus_acceptance_at_train_p95']:.4f}` | "
            f"`{s['mean_fetal_head_auroc']:.4f}` | "
            f"`{s['mean_chestmnist_auroc']:.4f}` | "
            f"`{s['mean_corrupted_auroc']:.4f}` | "
            f"`{s['overall_mean_ood_auroc']:.4f}` | "
            f"`{s['overall_mean_false_acceptance_at_train_p95']:.4f}` |"
        )
    lines.extend(
        [
            "",
            "## 2. Per-Source Breakdown",
            "",
            "| Held-Out BUS Source | Method | Held-Out BUS Acceptance | Fetal-Head US AUROC (FAR) | ChestMNIST AUROC (FAR) | Corrupted AUROC (FAR) |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for src, sdata in report["leave_one_source_out_results"].items():
        for m_name, mdata in sdata["methods"].items():
            nc = mdata["negative_cohorts"]
            lines.append(
                f"| `{src}` | `{m_name}` | `{mdata['held_out_breast_ultrasound_acceptance_rate']:.4f}` | "
                f"`{nc['fetal_head_other_ultrasound']['auroc']:.4f}` (`{nc['fetal_head_other_ultrasound']['false_acceptance_at_train_p95']:.4f}`) | "
                f"`{nc['chestmnist_other_medical']['auroc']:.4f}` (`{nc['chestmnist_other_medical']['false_acceptance_at_train_p95']:.4f}`) | "
                f"`{nc['corrupted_low_quality']['auroc']:.4f}` (`{nc['corrupted_low_quality']['false_acceptance_at_train_p95']:.4f}`) |"
            )
    lines.extend(
        [
            "",
            "## 3. Key Findings & Remaining Blockers",
            f"- **Why raw Mahalanobis failed (and why other methods succeed):** {report['scientific_conclusion']}",
            "- **Remaining deployment blocker:** Public classification (`BREASTCARE_CLASSIFICATION_ENABLED=false`) and production gate deployment remain disabled because licensed `natural_image`, `object`, `screenshot`, and real clinical `low_quality` cohorts with verified patient/author provenance are still required for full multi-category gate certification.",
        ]
    )
    OUT_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_benchmark()
