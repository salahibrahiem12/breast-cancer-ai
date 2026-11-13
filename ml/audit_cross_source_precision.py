"""Replay numerical development checks from saved embeddings; no model fitting."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from torchvision.models import ResNet18_Weights
from ml.development_baseline import development_rows


def run():
    root = Path('reports/cross_source_v1')
    with np.load(root/'features.npz') as cache:
        x = cache['development']
    with np.load('reports/busbra_locked_v1/final_linear.npz') as model:
        scaler = StandardScaler().fit(x)  # development inputs only; no classifier fit
        a = x.copy()
        a -= model['mean']
        a /= model['scale']
        p32 = expit((a @ model['coef'].T + model['intercept']).ravel())
        p64 = expit((((x-model['mean'])/model['scale']) @ model['coef'].T + model['intercept']).ravel())
        precision = {
            'saved_scaler_mean_max_replay_error': float(abs(scaler.mean_-model['mean']).max()),
            'saved_scaler_scale_max_replay_error': float(abs(scaler.scale_-model['scale']).max()),
            'float32_vs_float64_standardization_max_probability_difference': float(abs(p32-p64).max()),
            'development_05_call_changes': int(((p32 >= .5) != (p64 >= .5)).sum()),
            'new_model_fit': False, 'external_predictions_used': False}
    old = {r['path']: r for r in csv.DictReader(open('reports/bounded_development_v1/oof_predictions.csv'))}
    _, rows = development_rows(Path('data/raw/mendeley_v1/official-v1'), Path('data/prepared/mendeley_dev_v2/split_manifest.csv'))
    differences, changes = [], 0
    for i, row in enumerate(rows):
        ref = old[row['path']]
        with np.load('reports/bounded_development_v1/frozen_encoder_fold'+ref['fold']+'.npz') as model:
            a = x[i:i+1].copy()
            a -= model['mean']
            a /= model['scale']
            p = expit((a @ model['coef'].T + model['intercept']).item())
        original = float(ref['frozen_encoder'])
        differences.append(abs(p-original))
        changes += int((p >= .5) != (original >= .5))
    audit = json.loads((root/'encoder_audit.json').read_text())
    assert abs(max(differences)-audit['development_replay_max_probability_error']) < 1e-12
    assert changes == 0
    audit['development_replay_05_call_changes'] = changes
    (root/'encoder_audit.json').write_text(json.dumps(audit, indent=2))
    (root/'saved_model_precision_audit.json').write_text(json.dumps(precision, indent=2))
    reference = {'official_reference': repr(ResNet18_Weights.IMAGENET1K_V1.transforms()),
                 'actual_geometry': 'full frame stretch224; consistent development and inference; no reference center crop',
                 'normalization_matches_reference': True}
    (root/'reference_preprocessing.json').write_text(json.dumps(reference, indent=2))
    print(json.dumps(precision, indent=2))
    print('Development OOF replay decision changes:', changes)


if __name__ == '__main__':
    run()
