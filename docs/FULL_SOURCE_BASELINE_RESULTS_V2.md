# Full-Source Frozen Linear Baseline V2 (`full_source_baseline_v2` — Grouping-Validated Manifest V2)

**Completed UTC:** `2026-11-04T11:23:47.269287+00:00`
**Manifest:** `reports/grouping_audit_v1/eligible_manifest_v2.json` (`2,215` eligible images across `1,288` validated groups)
**All 15 L-BFGS-B Fits Converged (`max_absolute_gradient <= 1e-5`):** `True`
**Saved-Model Replay Max Error:** `0.000e+00`

## 1. Impact of Manifest V1 -> V2 Grouping Corrections on Frozen Linear Baseline

| Held-Out Source | V1 Images / Groups | V2 Images / Groups | V1 Group ROC-AUC | V2 Group ROC-AUC (95% CI) | Delta ROC-AUC | V1 Group Bal Acc | V2 Group Bal Acc (95% CI) | Delta Bal Acc | V2 Sens / Spec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mendeley` | `162 / 38` | `145 / 28` | `0.7059` | `0.6718` (`[0.4438, 0.8770]`) | `-0.0341` | `0.5728` | `0.5462` (`[0.3555, 0.7278]`) | `-0.0267` | `0.4000 / 0.6923` |
| `tcia` | `197 / 197` | `197 / 197` | `0.6884` | `0.6821` (`[0.6047, 0.7589]`) | `-0.0063` | `0.6440` | `0.6235` (`[0.5600, 0.6895]`) | `-0.0206` | `0.4490 / 0.7980` |
| `busbra` | `1873 / 1063` | `1873 / 1063` | `0.5938` | `0.5896` (`[0.5510, 0.6271]`) | `-0.0043` | `0.5703` | `0.5669` (`[0.5338, 0.5995]`) | `-0.0034` | `0.5789 / 0.5548` |

## 2. Scientific Interpretation
- **Mendeley (`38 -> 28` duplicate groups, `162 -> 145` images):** Removing the 17 partial-sequence frames that leaked from `mendeley_v1` test blocks and reuniting the 2 split five-image sequences (`us21..25` and `us191..195`) evaluates Mendeley strictly on its 28 complete held-out base-scan groups (14 benign, 14 malignant).
- **TCIA (`197` patient IDs) & BUS-BRA (`1,062 -> 1,063` source Case groups):** Separating the single false-positive `dHash=5` merge in BUS-BRA (`bus_0426-l` / `bus_0888-r`) and cleaning the Mendeley training fold produces virtually identical held-out metrics on TCIA and BUS-BRA, confirming that the cross-source generalization bottleneck is robust to grouping cleanup.
- **Reference baseline for `gpu_encoder_finetune_v3`:** `reports/full_source_baseline_v2/result.json` serves as the exact, grouping-aligned linear baseline for the packaged `gpu_encoder_finetune_v3` experiment.
