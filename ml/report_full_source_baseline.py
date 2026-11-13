"""Package measured full-eligible baseline results, without another fit."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT=Path('reports/full_source_baseline_v1')


def run():
    result=json.loads((ROOT/'result.json').read_text())['results']
    verified=json.loads((ROOT/'verification.json').read_text())
    assert verified['intervals_reproduced'] and not verified['serving_status']['inference_enabled']
    rows=json.loads(Path('reports/design_audit_v1/eligible_manifest.json').read_text())
    receipt=json.loads((ROOT/'feature_receipt.json').read_text())
    records=[]
    def record(held,stage,ix,fold=''):
        selected=[rows[i] for i in ix]
        for src in ['mendeley','tcia','busbra']:
            for label in ['benign','malignant']:
                rs=[r for r in selected if r['source']==src and r['label']==label]
                records.append({'held_out_source':held,'stage':stage,'inner_fold':fold,'source':src,'class':label,
                                'images':len(rs),'known_units':len({r['unit'] for r in rs}),
                                'global_groups':len({r['group'] for r in rs})})
    for held,r in result.items():
        train=r['training_rows'];test=[i for i,p in enumerate(rows) if p['source']==held]
        for stage in ['final_train','oof_calibration','oof_threshold']:record(held,stage,train)
        record(held,'held_out_evaluation',test)
        for f in json.loads((ROOT/held/'folds.json').read_text()):
            record(held,'inner_train',f['train_rows'],f['fold'])
            record(held,'inner_oof_validation',f['validation_rows'],f['fold'])
    with (ROOT/'phase_counts.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for source in result:
        curve=json.loads((ROOT/'models'/f'{source}_final.curve.json').read_text())
        axes[0].plot([c['iteration'] for c in curve],[c['objective'] for c in curve],label=source)
        axes[1].semilogy([c['iteration'] for c in curve],[max(c['max_absolute_gradient'],1e-12) for c in curve],label=source)
    axes[0].set(ylabel='Weighted regularized objective',xlabel='Optimizer iteration')
    axes[1].set(ylabel='Maximum absolute gradient',xlabel='Optimizer iteration')
    axes[1].axhline(1e-5,color='gray',linestyle='--',label='Required final bound')
    axes[0].legend(title='Held-out source');axes[1].legend()
    fig.suptitle('Full-data frozen baseline: final-refit optimization curves')
    fig.savefig(ROOT/'optimizer_learning_curves.png',dpi=160);plt.close(fig)

    def fmt(v):return 'undefined' if v is None else f'{v:.3f}'
    def interval(r,k):
        ci=r['group_intervals'][k]
        return f"{fmt(r['group_metrics'][k])} [{fmt(ci['lower_95'])}, {fmt(ci['upper_95'])}]"
    text='''# Full-eligible frozen baseline v1: measured results

Completed 4 November 2026 after the count reconciliation and training diagnosis
in [EXPERIMENT_DESIGN_AUDIT_V1.md](EXPERIMENT_DESIGN_AUDIT_V1.md).
Public classification remains disabled. No architecture search, deployment,
eligibility-gate measurement or raw patient-data upload was performed.

## Design and completion

The earlier220-image experiment is a compute-limited pilot, not conclusive
model development. Its missing epoch learning curves prevent a convergence
claim; saved endpoints also show nearly perfect training ranking in some folds,
so lack of fitting is not a sufficient explanation for poor transfer.

One fixed baseline was subsequently locked and run on **all2,232 eligible
images /1,298 global groups**, without a compute sampling cap. It uses the same
fixed ImageNet ResNet18 encoder, input transform and512 pooled features. Each
source-held-out side has four grouped inner fits for OOF scores, then one fit
on all images of the other two sources: **15 target linear fits total**.
All15 optimize the same predefined weighted regularized logistic objective.
No alternate hyperparameters, architecture, seed or held-out threshold was tried.

Source/diagnosis/group weighting prevents repeated images or the larger source
from automatically dominating the classifier objective. Calibration preserves
natural group diagnosis proportions with equal source/group weights. All
weights, StandardScaler fits, calibration and thresholds use only training-side
rows. Four-fold OOF scores supply calibration and threshold selection; every
training-side image then enters the final refit. No held-out source supplies
fitted preprocessing, labels or predictions for selection. The calibrated
OOF-to-final logit transfer is an approximation, not validated calibration.

This differs from the pilot in data volume, weighting and OOF/refit procedure;
any result difference cannot be attributed solely to more images or epochs.
All existing sources have already been inspected. These are exploratory
source-held-out development results, not an untouched final test.

| Held source | Final training images/groups | OOF calibration images/groups | Evaluated images/groups | Held images B/M | Held groups B/M |
|---|---|---|---|---|---|
'''
    for held,r in result.items():
        text+=f"| {held} | {r['train_images']}/{r['train_groups']} | {r['train_images']}/{r['train_groups']} | {r['held_out_images']}/{r['held_out_groups']} | {r['image_class_counts']['0']}/{r['image_class_counts']['1']} | {r['group_class_counts']['0']}/{r['group_class_counts']['1']} |\n"
    text+='''
`phase_counts.csv` reconciles every source/class for final training, all four
inner training/validation folds, OOF calibration, OOF threshold and evaluation,
with both known-unit and global-group counts. Phase totals are not distinct
acquisitions: OOF rows recur in three inner fits and in the final training fit.

## Primary mean-group endpoints

BA is the predefined primary endpoint at each training-side fixed threshold.
Intervals are95% percentile group bootstrap intervals from1,000 seeded draws.
Mendeley units are duplicate components, not clinical cases; TCIA units are
publisher-reported patient IDs, one image each; BUS-BRA uses source Cases whose
patient/breast/lesion/exam semantics remain unresolved. No general patient-
independent claim is made. Bootstrap intervals do not capture unresolved shared
patients, training-seed uncertainty or new-source variability.

| Held source | BA [95% CI] | ROC-AUC [95% CI] | Sensitivity | Specificity | Malignant AP | Binary Brier | Top-label ECE |
|---|---|---|---:|---:|---:|---:|---:|
'''
    for held,r in result.items():
        m=r['group_metrics']
        text+=f"| {held} | {interval(r,'balanced_accuracy')} | {interval(r,'roc_auc')} | "+' | '.join(fmt(m[k]) for k in
             ['sensitivity','specificity','pr_auc_average_precision','binary_brier','top_label_ece_10_bins'])+' |\n'
    text+='''
Cross-source performance remains modest: the fixed operating points detect
only38%–58% of malignant source groups. None of these outcomes establishes
clinical reliability or promotes a checkpoint. Additional data, weighting and
OOF calibration changed together relative to the pilot, so this is not a causal
estimate of the effect of longer training or increased sample size.
As an additional descriptive reference, a constant probability of.5 has binary
Brier.25 on any binary sample. All three measured Brier scores are larger;
training-side calibration has not demonstrated reliable probability transfer.

Complete confusion matrices and PPV/NPV, accuracy, log loss, all metric intervals,
and calibration bins with counts/mean probability/observed frequency are in
`result.json`. AP means average precision, not trapezoidal PR area; binary Brier
is mean(p-y)^2, not its doubled two-class form. Undefined denominators are null.
Calibration bin summaries are descriptive and receive no held-out refit.

| Held source | Group confusion [[TN,FP],[FN,TP]] | Threshold | Calibration coefficient/intercept | Majority-group accuracy | Training-majority accuracy | Constant training-prevalence group Brier |
|---|---|---:|---|---:|---:|---:|
'''
    for held,r in result.items():
        m=r['group_metrics'];base=r['majority_baselines']
        cm=[[m['tn'],m['fp']],[m['fn'],m['tp']]]
        text+=f"| {held} | {cm} | {r['threshold']:.6f} | {r['calibration_coef'][0][0]:.6f}/{r['calibration_intercept'][0]:.6f} | {base['held_group_majority_accuracy']:.3f} | {base['training_group_majority_accuracy_on_held']:.3f} | {base['training_prevalence_constant_held_group_brier']:.3f} |\n"
    text+='''
The held-source majority accuracy is an outcome-distribution reference, not a
deployable prediction rule. Training-majority and training-prevalence baselines
use training-side group labels only. Constant-class BA=.5. Any degenerate
bootstrap interval reflects the empirical observed prediction behavior, not
zero population uncertainty. No clinical acceptance threshold was invented.

## Image-level secondary results

Image CIs in the machine report resample whole global groups; repeated images
are not treated as independent. Group scores mean calibrated probabilities
before thresholding. Label purity and source-unit consistency were checked.

| Held source | BA | ROC-AUC | Sensitivity | Specificity | AP | Binary Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
'''
    for held,r in result.items():
        text+=f'| {held} | '+' | '.join(fmt(r['image_metrics'][k]) for k in
           ['balanced_accuracy','roc_auc','sensitivity','specificity','pr_auc_average_precision','binary_brier','top_label_ece_10_bins'])+' |\n'
    text+='''
## Optimization adequacy and limitations

All15 target fits passed the predefined optimizer-success and maximum absolute
gradient<=1e-5 checks. Actual optimizer curves are saved for every fit; the plot
shows the three final refits. This certifies optimization of the frozen-feature
regularized objective, not adequacy of the encoder, data diversity or clinical
use. A frozen encoder has no neural training epochs. No fully trained neural
fine-tuning candidate is claimed in this continuation.

| Held source | Final optimizer iterations | Final objective | Maximum absolute gradient | Final train image AUC | Training-side OOF group AUC |
|---|---:|---:|---:|---:|---:|
'''
    for held,r in result.items():
        f=r['final_optimizer']
        text+=f"| {held} | {f['optimizer_iterations']} | {f['objective']:.6f} | {f['max_absolute_gradient']:.3g} | {r['training_refit_image_metrics']['roc_auc']:.3f} | {r['training_oof_group_metrics']['roc_auc']:.3f} |\n"
    text+=f"\nFeature extraction took {receipt['new_or_resumed_seconds']:.1f} seconds on the local CPU, reusing {receipt['cached_pilot_rows_reused_this_invocation']} identical pilot rows and hashing every input payload before new chunk creation. All2232 rows were retained.\n"
    text+='''
Training-side OOF metrics share the scores used for calibration and threshold
selection, so they are diagnostic and optimistic for those fitted steps.
They are not additional independent validation. No outcome in this report
automatically promotes a model or justifies another architecture search.

## Reproducibility, preservation and blockers

The lock records the pre-fit UTC timestamp, protocol/runner/input hashes and
runtime versions. `LOCK_SHA256.txt` binds it. New chunks/checkpoints are
resumable; previous pilot/external reports and models were not overwritten.
Verification recomputes all15 training-only weighted scalers, all15 objectives
and gradients, OOF/final logits, calibration/threshold replay, metric tables and
both group/image intervals. All2,232 eligible rows have exactly one held-out
prediction. Prior artifact hashes and the disabled serving state also pass.

From the project root, existing run:

```powershell
.\\.venv\\Scripts\\python.exe -m ml.verify_full_source_baseline --intervals
.\\.venv\\Scripts\\python.exe -m ml.report_full_source_baseline
```

To reproduce training, use an isolated copy of the project with identical
inputs and the recorded environment, an empty `reports/full_source_baseline_v1`
directory, and the completed design audit present. Then:

```powershell
.\\.venv\\Scripts\\python.exe -m ml.full_source_baseline lock
.\\.venv\\Scripts\\python.exe -m ml.full_source_baseline features
.\\.venv\\Scripts\\python.exe -m ml.full_source_baseline run
```

Resume features/run after interruption with the same commands and lock.
Completed fits are reused; partial coefficient checkpoints reset quasi-Newton
history, as recorded, without changing objective or hyperparameters. Do not
delete or overwrite the delivered run. Model NPZs include scaler and linear
parameters; training-side selection JSON holds calibration and threshold.
`artifact_hashes.json` binds delivered outputs/code, and the results ZIP omits
raw ultrasound images. The saved feature matrix supports fitting reproduction
without neural re-extraction when input receipts remain available.

No fresh external dataset was acquired and no access restrictions were bypassed.
BUSI_WHU remains an inaccessible, prediction-free licensed metadata lead with
unverified grouping. Final independent validation remains blocked pending an
official accessible archive and verified labels/grouping/overlap. Any new cohort
stays prediction-free until its protocol is locked. Existing sources remain
inspected development data. Eligibility-gate performance and the upload-to-
result system were not evaluated. Software tests do not establish clinical
reliability.
Six focused software tests passed in31.47s, covering the numerical objective
gradient, source/diagnosis/group weighting and existing aggregation/threshold
behavior. The earlier77-test regression run is historical, not a rerun here.
'''
    Path('docs/FULL_SOURCE_BASELINE_RESULTS_V1.md').write_text(text,encoding='utf-8')
    files=[p for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json']
    files += [p for p in Path('reports/design_audit_v1').rglob('*') if p.is_file()]
    files += [Path(p) for p in ['docs/EXPERIMENT_DESIGN_AUDIT_V1.md','docs/FULL_SOURCE_BASELINE_PROTOCOL_V1.md',
       'docs/FULL_SOURCE_BASELINE_RESULTS_V1.md','ml/design_audit.py','ml/pilot_training_diagnosis.py',
       'ml/report_design_audit.py','ml/full_source_baseline.py','ml/verify_full_source_baseline.py',
       'ml/report_full_source_baseline.py','tests/test_full_source_baseline.py']]
    hashes={str(p).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (ROOT/'artifact_hashes.json').write_text(json.dumps(hashes,indent=2))
    with ZipFile('reports/design_audit_full_baseline_v1.zip','w',ZIP_DEFLATED) as bundle:
        for p in files+[ROOT/'artifact_hashes.json']:bundle.write(p,str(p).replace('\\','/'))
    print('Full-data results report, learning curves, phase counts, hashes and bundle saved.')


if __name__=='__main__':run()
