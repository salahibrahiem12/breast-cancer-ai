# Cross-source exploratory development protocol v1

Recorded before fitting. The user explicitly authorized relabeling TCIA and
BUS-BRA as inspected development sources. Local research reuse fits the
recorded CC BY 4.0 terms and citation obligations; no clinical deployment or
broader release is authorized. Prior locked artifacts are preserved separately.

Sources: 162 eligible Mendeley development images (old-test-linked exclusions
retained), 197 biopsy-confirmed TCIA images, and 1,873 eligible BUS-BRA images.
No additional normal or follow-up cases are added. TCIA identifiers have
publisher-backed one-image-per-patient semantics; BUS-BRA Case semantics remain
unresolved; Mendeley identifiers are duplicate groups, not patients.

Merge known groups plus exact pixel/byte and dHash/pHash distance <=5 links
over the complete eligible inventory. Drop conflicting-label components and
components spanning sources from this source-held-out experiment rather than
placing a component on both sides. Do not infer patient identity from hashes.
For bounded compute choose up to 16 pure groups per class/source, seed42, before
predictions. Include all images within selected groups. This is a balanced,
small convenience development sample and cannot establish population accuracy.

Three outer source-held-out folds, two arms each: frozen ImageNet ResNet18
512 pooled features plus StandardScaler/L2 balanced logistic C=1; and ResNet18
with layer4/head fine-tuned for exactly two epochs, AdamW lr1e-4,
weight_decay1e-4, batch16, weighted cross-entropy, frozen BatchNorm statistics.
Use RGB, full-frame bilinear224/ImageNet normalization, no augmentation.
CPU threads2, seed42, cached official weights. No search or outer-result
selection. Six fitted models and six fine-tuning epochs total.

Each outer training side gets one inner source-and-label-stratified grouped
four-fold split (first fold validation). Fit scaler/classifier/weights on inner
train only; fit regularized Platt logistic calibration C=1 on inner validation
logits only. Select threshold there by maximum balanced accuracy, ties favor
specificity then higher threshold. Do not refit after calibration or inspect
outer outcomes until this procedure is complete. Record thresholds and inner
sizes. Primary endpoint: held-out source grouped mean-score balanced accuracy
at the training-side selected threshold. Also report image metrics, fixed .5
results, ROC-AUC, malignant AP, Brier/log-loss/ECE and 1,000 group-bootstrap
intervals; the group unit is explicit and patient independence is conditional
on available metadata. Means aggregate each pure global group; source Case
aggregation is supported only where the selected group equals a source Case.

Encoder audit before model comparison: strict pretrained tensor equality,
eval/normalization/RGB/512-layer checks, sample/batch repeat consistency,
per-feature variance/effective rank, independent input transform comparison,
and replay the frozen-encoder coefficients for original development OOF rows.
Audit numerical saved-scaler versus StandardScaler dtype differences explicitly.

Source shortcut diagnostic: one grouped three-fold source classifier from
frozen embeddings, training-only scaler, logistic C=1; report accuracy versus
majority and balanced accuracy. This is association, not proof of causation.
For every outer frozen model, test held-out development images with outer10%
replaced by median, and center80% replaced by median, with the exact original
calibrator and threshold. Do not refit. These interventions also remove anatomy.

Seek a fourth licensed source before any model-based inspection. A metadata
lead without accessible files or verified grouping is not a reserved qualified
external cohort. Final independent validation remains blocked if none qualifies.
No UI changes, gate claims, production checkpoint changes or enabling serving.
