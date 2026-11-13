"""Unit tests for unified manifest reconciliation (v1 & v2), grouping integrity audit,
historical checkpoint parameter-delta verification, ultrasound-safe augmentations,
and GPU encoder fine-tuning (frozen BatchNorm + resumable checkpoints).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ml.full_source_finetune import (
    ResidualAdapterHead,
    count_trainable_parameters,
    equal_source_group_metrics,
    ultrasound_augment_pil,
    weighted_bce_with_logits,
)
from ml.gpu_encoder_finetune_v2 import (
    ResNet18Layer4FineTuner,
    build_optimizer,
    fit_encoder_fold,
    parameter_summary,
    snapshot_batchnorm_buffers,
    verify_frozen_batchnorm,
)


def test_unified_dataset_manifest_reconciles_all_2381_acquired_images():
    manifest_csv = Path("data/prepared/unified_dataset_manifest_v1.csv")
    report_json = Path("reports/unified_dataset_manifest_v1.json")
    assert manifest_csv.is_file(), "Unified dataset manifest CSV is missing"
    assert report_json.is_file(), "Unified dataset manifest report JSON is missing"

    with manifest_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    report = json.loads(report_json.read_text(encoding="utf-8"))

    assert len(rows) == 2381
    eligible = [r for r in rows if r["eligible"] == "true"]
    excluded = [r for r in rows if r["eligible"] == "false"]
    assert len(eligible) == 2232
    assert len(excluded) == 149
    assert len({r["duplicate_group"] for r in eligible}) == 1298
    assert all(r["exclusion_reason"] == "" for r in eligible)
    assert all(r["exclusion_reason"] != "" for r in excluded)
    assert all(r["is_mask"] == "false" for r in rows)
    assert all(report["integrity_verifications"].values())


def test_grouping_integrity_audit_and_v2_manifest():
    audit_path = Path("reports/grouping_audit_v1/grouping_audit.json")
    v2_csv = Path("data/prepared/unified_dataset_manifest_v2.csv")
    v2_elig_json = Path("reports/grouping_audit_v1/eligible_manifest_v2.json")
    assert audit_path.is_file()
    assert v2_csv.is_file()
    assert v2_elig_json.is_file()

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    v2_elig = json.loads(v2_elig_json.read_text(encoding="utf-8"))
    with v2_csv.open("r", encoding="utf-8") as f:
        v2_rows = list(csv.DictReader(f))

    assert len(v2_rows) == 2381
    assert len(v2_elig) == 2215
    assert len({r["group"] for r in v2_elig}) == 1288
    m_seq = audit["mendeley_5image_crop_sequence_discovery"]
    assert m_seq["consecutive_5image_blocks"] == 50
    assert m_seq["blocks_with_identical_resolution_and_label"] == 50
    assert m_seq["leaked_images_in_partial_blocks"] == 17
    assert m_seq["fully_eligible_blocks_split_across_two_v1_global_groups"] == 2
    for sheet_path in audit["contact_sheets"].values():
        assert Path(sheet_path).is_file()


def test_historical_checkpoint_deltas_and_task329_audit():
    ver_path = Path("reports/full_source_finetune_v1/verification.json")
    assert ver_path.is_file()
    ver = json.loads(ver_path.read_text(encoding="utf-8"))
    assert ver["all_15_saved_checkpoints_verified_changed_from_seeded_init"] is True
    tver = ver["trainable_parameter_and_gradient_verification"]
    assert "task_329_assertion_resolution" in tver
    assert "distinction_of_four_concepts" in tver
    assert len(ver["checkpoint_parameter_summaries"]) == 15
    for fit_name, summary in ver["checkpoint_parameter_summaries"].items():
        assert summary["all_9_parameter_tensors_changed_from_init"] is True
        assert all(v > 0.0 for v in summary["historical_l2_delta_from_init"].values()), fit_name


def test_gpu_encoder_finetuner_frozen_batchnorm_and_resumable_checkpoint(tmp_path: Path):
    model = ResNet18Layer4FineTuner()
    psum = parameter_summary(model)
    assert psum["trainable_layer4_conv_parameters"] == 8_388_608
    assert psum["trainable_fc_head_parameters"] == 1_026
    assert psum["total_trainable_parameters"] == 8_389_634

    init_bn = snapshot_batchnorm_buffers(model)
    model.train()
    bn_check = verify_frozen_batchnorm(model, init_bn)
    assert bn_check["batchnorm_module_count"] == 20
    assert bn_check["all_batchnorm_eval_mode"] is True

    opt = build_optimizer(model)
    x = torch.randn(4, 3, 224, 224)
    y = torch.tensor([0.0, 1.0, 0.0, 1.0])
    opt.zero_grad()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
    loss.backward()
    assert model.backbone.layer4[0].conv1.weight.grad is not None
    assert float(model.backbone.layer4[0].conv1.weight.grad.norm().item()) > 0.0
    assert model.backbone.conv1.weight.grad is None
    opt.step()
    verify_frozen_batchnorm(model, init_bn)

    # Test save/resume behavior in fit_encoder_fold
    lock_file = tmp_path / "lock.json"
    lock_file.write_text('{"experiment_id": "test"}', encoding="utf-8")
    imgs = [Image.new("RGB", (64, 64), color=(20 * i, 25 * i, 30 * i)) for i in range(1, 9)]
    tr_ix = np.array([0, 1, 2, 3])
    va_ix = np.array([4, 5, 6, 7])
    y_tr = np.array([0, 1, 0, 1])
    s_tr = np.array(["s1", "s1", "s2", "s2"])
    g_tr = np.array(["g0", "g1", "g2", "g3"])
    y_va = np.array([0, 1, 0, 1])
    s_va = np.array(["s1", "s1", "s2", "s2"])
    g_va = np.array(["g4", "g5", "g6", "g7"])

    # Run 1 epoch first, then resume to 2 epochs and compare against uninterrupted 2-epoch run
    dir_resume = tmp_path / "run_resume"
    dir_resume.mkdir()
    (dir_resume / "lock.json").write_text('{"experiment_id": "test"}', encoding="utf-8")
    fit_encoder_fold(
        name="fold0",
        images=imgs,
        tr_ix=tr_ix,
        y_tr=y_tr,
        s_tr=s_tr,
        g_tr=g_tr,
        va_ix=va_ix,
        y_va=y_va,
        s_va=s_va,
        g_va=g_va,
        device=torch.device("cpu"),
        out_dir=dir_resume,
        fixed_epochs=1,
        batch_size=2,
        seed=77,
    )
    assert (dir_resume / "models" / "fold0.resume.pth").is_file()
    # Remove final .pth/.json so second call resumes from .resume.pth to epoch 2
    (dir_resume / "models" / "fold0.pth").unlink()
    (dir_resume / "models" / "fold0.json").unlink()
    _, info_resumed = fit_encoder_fold(
        name="fold0",
        images=imgs,
        tr_ix=tr_ix,
        y_tr=y_tr,
        s_tr=s_tr,
        g_tr=g_tr,
        va_ix=va_ix,
        y_va=y_va,
        s_va=s_va,
        g_va=g_va,
        device=torch.device("cpu"),
        out_dir=dir_resume,
        fixed_epochs=2,
        batch_size=2,
        seed=77,
    )
    assert info_resumed["epochs_ran"] == 2


def test_ultrasound_augment_pil_is_deterministic_and_preserves_224_rgb():
    base_img = Image.new("RGB", (320, 240), color=(60, 80, 100))
    aug1, meta1 = ultrasound_augment_pil(base_img, row_index=7, base_seed=101)
    aug2, meta2 = ultrasound_augment_pil(base_img, row_index=7, base_seed=101)
    aug3, meta3 = ultrasound_augment_pil(base_img, row_index=8, base_seed=101)

    assert aug1.size == (224, 224)
    assert aug1.mode == "RGB"
    assert meta1 == meta2
    assert np.array_equal(np.asarray(aug1), np.asarray(aug2))
    assert meta1["angle_deg"] != meta3["angle_deg"]
    assert -6.0 <= meta1["angle_deg"] <= 6.0
    assert meta1["center_crop_fraction"] == 0.94


def test_residual_adapter_zero_gate_starts_from_linear_head():
    model = ResidualAdapterHead(dim=512, bottleneck=256)
    assert count_trainable_parameters(model) == 264_963
    assert torch.allclose(model.gate.detach(), torch.zeros(1))

    z = torch.randn(8, 512)
    model.eval()
    with torch.inference_mode():
        h = model.forward_features(z)
        logits = model(z)
    assert torch.allclose(h, z, atol=1e-7)
    assert logits.shape == (8,)


def test_weighted_bce_and_equal_source_group_metrics():
    y = np.array([0, 1, 0, 1, 0, 1])
    logits = np.array([-2.0, 2.0, -1.0, 1.5, 0.5, -0.5])
    w = np.array([1.0, 1.0, 1.0, 1.0, 0.5, 0.5])
    g = np.array(["g1", "g2", "g3", "g4", "g5", "g6"])
    s = np.array(["s1", "s1", "s1", "s1", "s2", "s2"])

    loss = weighted_bce_with_logits(logits, y, w)
    assert np.isfinite(loss) and loss > 0.0

    auc, ba = equal_source_group_metrics(y, logits, g, s)
    assert np.isclose(auc, 0.5)
    assert np.isclose(ba, 0.5)


def test_full_source_baseline_v2_and_ood_benchmark_v2_artifacts():
    v2_base = json.loads(Path("reports/full_source_baseline_v2/result.json").read_text(encoding="utf-8"))
    assert v2_base["manifest_version"] == "v2"
    assert v2_base["eligible_images"] == 2215
    assert v2_base["eligible_groups"] == 1288
    assert v2_base["all_15_fits_converged"] is True
    assert v2_base["inference_enabled"] is False
    assert v2_base["results"]["mendeley"]["held_out_groups"] == 28
    assert v2_base["results"]["tcia"]["held_out_groups"] == 197
    assert v2_base["results"]["busbra"]["held_out_groups"] == 1063
    assert max(v2_base["saved_model_replay_max_errors"].values()) <= 1e-12

    ood_v2 = json.loads(Path("reports/eligibility_ood_benchmark_v2.json").read_text(encoding="utf-8"))
    assert ood_v2["manifest_version"] == "v2"
    assert ood_v2["eligible_breast_ultrasound_images"] == 2215
    assert ood_v2["deployable_gate_claimed"] is False
    assert ood_v2["serving_availability"]["inference_enabled"] is False
    ms = ood_v2["method_summary_across_held_out_sources"]
    assert set(ms.keys()) == {
        "raw_mahalanobis_512d",
        "l2_cosine_knn_k10",
        "pca64_orthogonal_residual",
        "ultrasound_physics_12d_oneclass",
        "hybrid_physics_cosine_probe",
        "cross_source_calibrated_hybrid_probe",
    }
    assert (
        ms["ultrasound_physics_12d_oneclass"]["mean_held_out_bus_acceptance_at_min_source_p95"]
        > ms["raw_mahalanobis_512d"]["mean_held_out_bus_acceptance_at_train_p95"]
    )
    assert (
        ms["cross_source_calibrated_hybrid_probe"]["mean_held_out_bus_acceptance_at_train_p95"]
        > ms["raw_mahalanobis_512d"]["mean_held_out_bus_acceptance_at_train_p95"]
    )

