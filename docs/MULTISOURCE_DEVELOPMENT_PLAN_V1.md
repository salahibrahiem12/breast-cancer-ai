# Multi-source research development plan v1

Recorded 4 November 2026. Public classification stays disabled. This plan does
not authorize new deployment or establish eligibility-gate performance.

## Source roles and labels

Mendeley, TCIA and BUS-BRA are inspected development sources. Historical failed
results and locked protocols remain immutable. No later analysis of these
sources may be described as untouched testing. The current bounded experiment
is defined in `CROSS_SOURCE_PROTOCOL_V1.md`, with measured outcomes in
`CROSS_SOURCE_RESULTS_V1.md`; its outer outcomes will not be used to tune a
threshold, reverse a score, or pick a crop for an independent-test claim.

Use only eligible benign/malignant labels with traceable reference standards.
Keep original Mendeley old-test-linked exclusions and BUS-BRA conservative
overlap exclusions. Follow-up/normal TCIA exclusions stay explicit. Retain the
original metadata and input hashes rather than silently correcting labels.
Any future label adjudication needs a versioned record and a new protocol.

## Grouping and overlap

Namespace source identifiers, preserve known group membership, and union
cross-source exact pixel/byte and close dHash/pHash links before partitioning.
Conflicting labels and cross-source components are quarantined from a
source-held-out fold unless an explicit future protocol defines their handling.
Review additional embedding-independent image similarity candidates locally;
image hash distance alone is neither proof of identity nor clinical equivalence.
Keep scans, metadata and pseudonymous IDs local; publish aggregate summaries.

TCIA has publisher-reported one image per patient; Mendeley has no patient map.
BUS-BRA Case is unresolved despite agreement with the published patient count.
Seek authoritative dictionaries for patient/breast/lesion/exam relationships;
do not infer independence from file numbering, matching counts or supplied folds.
Bootstrap and aggregate at the strongest supported available grouping, naming
the unit and its limits. Shared patients between sources remain unverified.

## Training-side evidence and future comparisons

The completed budget was six fits on 32 sampled groups/source, fixed before
prediction. There was no outcome-driven retry. A larger future comparison
requires a separately recorded budget before fitting, with more groups and
additional acquisition sources if rights permit. A two-epoch candidate on this
small sample does not establish the potential of all fine-tuning methods.

Continue source-held-out evaluation, with all scaler fitting, calibration,
threshold selection and any model selection confined to each training side.
Use identical eligible partitions between arms, source/label-stratified inner
group splits, and retain original held-out predictions. Report every source,
including failures, rather than a favorable pooled score. Quantify both group
and image endpoints and grouped intervals; distinguish training seed variability
from evaluation-sample uncertainty. Avoid repeated search on these small folds.

The audit identifies consistent full-frame preprocessing that differs from the
pretrained reference crop. A future predefined inner-training comparison can
test reference versus full-frame geometry. Border/center interventions also
remove anatomy and change area differently; their outer improvements are not
a basis to adopt a crop. Source predictability is an association. Better
controls require matched acquisition/annotation conditions or training-side
interventions evaluated across multiple sources, without claiming causation
from embedding clustering alone.

## New independent cohort

BUSI_WHU v3 is a CC BY 4.0 metadata lead with no model-based inspection. Checked
download routes returned403. Before qualification, acquire the official archive
and dictionary, verify license/attribution, benign/malignant reference standards,
patient/exam mapping and overlap with every available source. Record hashes,
exclusions and reservation in the ledger before predictions. Mere access does
not establish qualification. Keep the eventual cohort prediction-free through
model selection and protocol lock. No qualified fresh final cohort is currently
reserved, so final independent validation remains blocked.

## Next evidence-producing step

Obtain a traceable accessible archive and grouping dictionary for a new source,
then perform model-free rights, grouping, label and duplicate checks. If this
succeeds, reserve it and expand development diversity separately. If it fails,
report the access/grouping blocker without another favorable-score search on
TCIA or BUS-BRA. No outcome automatically enables public classification; the
eligibility gate remains a separately unmeasured workstream.
