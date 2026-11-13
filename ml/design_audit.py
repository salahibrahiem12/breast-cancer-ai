"""Audit the preserved pilot design. Writes only a new audit directory."""
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from ml.cross_source import WEIGHTS, payload, tensors, net, extract
from ml.development_baseline import development_rows
from ml.inspect_tcia_breast import fingerprint
from ml.prepare_dataset import UnionFind

OUT = Path('reports/design_audit_v1')
PILOT = Path('reports/cross_source_v1')


def read(path):
    return list(csv.DictReader(Path(path).open(encoding='utf-8', newline='')))


def save(name, obj):
    (OUT/name).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def preserve():
    files = list(PILOT.rglob('*')) + [Path('reports/cross_source_v1_results.zip'),
        Path('docs/CROSS_SOURCE_PROTOCOL_V1.md'), Path('docs/CROSS_SOURCE_RESULTS_V1.md'),
        Path('ml/cross_source.py')]
    hashes = {str(p).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in files if p.is_file()}
    save('prior_artifact_hashes.json', hashes)


def eligible_inventory():
    all_md, eligible_md = development_rows(Path('data/raw/mendeley_v1/official-v1'),
                                          Path('data/prepared/mendeley_dev_v2/split_manifest.csv'))
    rows = []
    for r in eligible_md:
        p = Path('data/raw/mendeley_v1/official-v1')/r['path']
        rows.append({'source': 'mendeley', 'label': r['label'], 'unit': 'mendeley:'+r['group_id'],
                     'file': str(p), **fingerprint(p.read_bytes())})
    ti = {Path(r['path']).name: r for r in read(
        'data/research_data/tcia_breast_lesions_usg_2026-11-03/inventory/cohort_inventory.csv')}
    for r in read('data/prepared/tcia_breast_biopsy_external_v1/split_manifest.csv'):
        rows.append({'source': 'tcia', 'label': r['label'], 'unit': r['patient_id'],
                     'file': str(Path('data/raw/tcia_breast_biopsy_external_v1')/r['path']),
                     **{k: ti[r['path']][k] for k in ['sha256', 'pixel_sha256', 'dhash', 'phash']}})
    for r in read('reports/busbra_locked_v1/eligible_inventory.csv'):
        rows.append({'source': 'busbra', 'label': r['label'], 'unit': 'busbra:'+r['source_case'],
                     'archive_path': r['archive_path'],
                     **{k: r[k] for k in ['sha256', 'pixel_sha256', 'dhash', 'phash']}})
    union = UnionFind(len(rows)); owners = {}; links = []
    for i, r in enumerate(rows):
        if r['unit'] in owners:
            union.union(i, owners[r['unit']])
        owners[r['unit']] = i
        for j, b in enumerate(rows[:i]):
            if r['sha256'] == b['sha256'] or r['pixel_sha256'] == b['pixel_sha256'] or any(
                    (int(r[k]) ^ int(b[k])).bit_count() <= 5 for k in ['dhash', 'phash']):
                union.union(i, j); links.append([i, j])
    components = defaultdict(list)
    for i, r in enumerate(rows):
        r['group'] = 'global:'+str(union.find(i))
        components[r['group']].append(r)
    excluded = []; clean = []
    for rs in components.values():
        reason = ('cross_source' if len({r['source'] for r in rs}) > 1 else
                  'contradictory_label' if len({r['label'] for r in rs}) > 1 else None)
        if reason:
            excluded.append({'reason': reason, 'rows': rs})
        else:
            clean.extend(rs)
    return clean, excluded, links, all_md, list(ti.values())


def count(rows):
    return {s: {label: {'images': len(v := [r for r in rows if r['source'] == s and r['label'] == label]),
                       'known_units': len({r['unit'] for r in v}),
                       'global_groups': len({r['group'] for r in v}) if all('group' in r for r in v) else None}
                for label in sorted({r['label'] for r in rows if r['source'] == s})}
            for s in ['mendeley', 'tcia', 'busbra']}


def source_probe(rows, x):
    predictions = []; answer = {}
    for diagnosis in ['all', 'benign', 'malignant']:
        ix = np.array([i for i, r in enumerate(rows) if diagnosis == 'all' or r['label'] == diagnosis])
        a = x[ix]; y = np.array([rows[i]['source'] for i in ix]); g = np.array([rows[i]['group'] for i in ix])
        calls = np.empty(len(ix), object); base = calls.copy(); conditional = calls.copy(); folds = []
        for fold, (tr, va) in enumerate(StratifiedGroupKFold(3, shuffle=True, random_state=42).split(a, y, g)):
            assert not set(g[tr]) & set(g[va])
            scaler = StandardScaler().fit(a[tr])
            c = LogisticRegression(C=1, max_iter=2000).fit(scaler.transform(a[tr]), y[tr])
            calls[va] = c.predict(scaler.transform(a[va]))
            majority = Counter(y[tr]).most_common(1)[0][0]; base[va] = majority
            for j in va:
                label = rows[ix[j]]['label']
                conditional[j] = Counter(y[k] for k in tr if rows[ix[k]]['label'] == label).most_common(1)[0][0]
                predictions.append({'diagnosis': diagnosis, 'fold': fold, 'row': int(ix[j]),
                                    'actual_source': y[j], 'called_source': calls[j],
                                    'group': g[j], 'majority_baseline': base[j],
                                    'diagnosis_only_baseline': conditional[j]})
            folds.append({'fold': fold, 'train_images': len(tr), 'validation_images': len(va),
                          'train_groups': sorted(set(g[tr])), 'validation_groups': sorted(set(g[va])),
                          'train_source_counts': dict(Counter(y[tr])), 'validation_source_counts': dict(Counter(y[va])),
                          'accuracy': float(accuracy_score(y[va], calls[va])),
                          'balanced_accuracy': float(balanced_accuracy_score(y[va], calls[va])),
                          'optimizer_iterations': c.n_iter_.tolist()})
        answer[diagnosis] = {'images': len(ix), 'groups': len(set(g)),
            'source_counts': dict(Counter(y)), 'accuracy': float(accuracy_score(y, calls)),
            'balanced_accuracy': float(balanced_accuracy_score(y, calls)),
            'training_majority_accuracy': float(accuracy_score(y, base)),
            'training_diagnosis_only_accuracy': float(accuracy_score(y, conditional)),
            'training_diagnosis_only_balanced_accuracy': float(balanced_accuracy_score(y, conditional)),
            'oracle_source_distribution_majority_accuracy': max(Counter(y).values())/len(y),
            'confusion_order': ['mendeley','tcia','busbra'],
            'confusion': confusion_matrix(y, calls, labels=['mendeley','tcia','busbra']).tolist(), 'folds': folds}
    save('source_probe.json', answer)
    with (OUT/'source_predictions.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(predictions[0])); w.writeheader(); w.writerows(predictions)


def tcia_explanation():
    ps = [r for r in read(PILOT/'predictions.csv') if r['source'] == 'tcia' and r['arm'] == 'fine_tuned']
    b = np.array([float(r['probability']) for r in ps if r['actual'] == '0'])
    m = np.array([float(r['probability']) for r in ps if r['actual'] == '1'])
    t = float(ps[0]['threshold']); wins = int((m[:,None] > b).sum()); ties = int((m[:,None] == b).sum())
    save('tcia_ba_auc_explanation.json', {'threshold': t, 'benign_images': len(b), 'malignant_images': len(m),
        'confusion': [[int((b<t).sum()), int((b>=t).sum())], [int((m<t).sum()), int((m>=t).sum())]],
        'pairwise_malignant_wins': wins, 'pairwise_ties': ties, 'all_pairs': len(b)*len(m),
        'roc_auc': (wins+.5*ties)/(len(b)*len(m)),
        'benign_score_quantiles': np.quantile(b,[0,.25,.5,.75,1]).tolist(),
        'malignant_score_quantiles': np.quantile(m,[0,.25,.5,.75,1]).tolist(),
        'scores_changed': False, 'threshold_changed': False})


def run():
    OUT.mkdir(exist_ok=True)
    if not (OUT/'prior_artifact_hashes.json').exists(): preserve()
    rows, excluded, links, all_md, all_ti = eligible_inventory()
    save('eligible_manifest.json', rows)
    pilot = json.loads((PILOT/'selection.json').read_text())['rows']
    lookup = {r['sha256']:r for r in rows}
    assert all(lookup[r['sha256']]['group'] == r['group'] for r in pilot)
    acquired = [{'source':'mendeley','label':r['label'],'unit':'mendeley:'+r['group_id']} for r in all_md]
    acquired += [{'source':'tcia','label':r['label'],'unit':r['patient_id']} for r in all_ti]
    with ZipFile('data/research_data/busbra_reserved_v1/BUSBRA.zip') as z:
        bus = list(csv.DictReader(StringIO(z.read('BUSBRA/bus_data.csv').decode('utf-8-sig'))))
    acquired += [{'source':'busbra','label':r['Pathology'].lower(),'unit':'busbra:'+r['Case']} for r in bus]
    selected_sha = {r['sha256'] for r in pilot}
    eligible_units = {(r['source'], r['unit']) for r in rows}
    integrity = [r for r in acquired if (r['source'],r['unit']) not in eligible_units]
    unsampled = [r for r in rows if r['sha256'] not in selected_sha]
    result = json.loads((PILOT/'result.json').read_text())
    folds = {}
    for source, arms in result['results'].items():
        r = arms['frozen']; phases = {}
        for phase, key in [('trained','inner_train_groups'),('validated_calibrated_threshold','inner_validation_groups'),('evaluated','held_out_groups')]:
            ids = set(r[key]); v = [p for p in pilot if p['group'] in ids]; phases[phase] = count(v)
            if phase != 'evaluated': assert all(p['source'] != source for p in v)
            else: assert all(p['source'] == source for p in v)
        assert all(r[k] == arms['fine_tuned'][k] for k in ['inner_train_groups','inner_validation_groups','held_out_groups'])
        train = [p for p in pilot if p['group'] in r['inner_train_groups']]
        counts = Counter(p['label'] for p in train)
        folds[source] = {'phases':phases, 'training_class_images':dict(counts),
                         'class_weights': {l:len(train)/(2*n) for l,n in counts.items()},
                         'fine_tuning_epochs':2, 'optimizer_steps':2*math.ceil(len(train)/16),
                         'source_absent_from_all_training_side_phases':True}
    save('count_reconciliation.json', {'created_utc':datetime.now(timezone.utc).isoformat(),
        'acquired':count(acquired), 'eligible':count(rows), 'integrity_excluded':count(integrity),
        'pilot_sampled':count(pilot), 'compute_only_not_sampled':count(unsampled),
        'hash_links':len(links), 'new_component_exclusions':excluded, 'folds':folds,
        'note':'Group counts across classes may overlap in acquired data; aggregate counts use unique units, not sums.'})
    tcia_explanation()
    with np.load(PILOT/'features.npz') as f: x = f['selected']
    source_probe(pilot, x)
    print('Count reconciliation, source-isolation checks, class-conditioned source probes and TCIA explanation saved.', flush=True)


if __name__ == '__main__': run()
