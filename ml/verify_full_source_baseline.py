"""Verify full-data baseline isolation, saved replay and reported metrics."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from ml.full_source_baseline import (OUT, check_lock, rows, sample_weights,
                                    objective, decision, choose_threshold, digest, save)
from ml.cross_source import metric, aggregate, intervals


def verify(repeat_intervals=False):
    check_lock();rs=rows();result=json.loads((OUT/'result.json').read_text())
    receipt=json.loads((OUT/'feature_receipt.json').read_text())
    assert receipt['features_sha256']==digest(OUT/'features.npz')
    all_pred=list(csv.DictReader((OUT/'predictions.csv').open()))
    assert len(all_pred)==len(rs)==len({p['row'] for p in all_pred})
    with np.load(OUT/'features.npz') as f:x=f['features'].astype(float)
    assert np.isfinite(x).all()
    y=np.array([r['label']=='malignant' for r in rs],int);g=np.array([r['group'] for r in rs]);s=np.array([r['source'] for r in rs])
    fits=0;max_error=0.
    for held,r in result['results'].items():
        train=np.flatnonzero(s!=held);test=np.flatnonzero(s==held);coverage=Counter()
        assert r['training_rows']==train.tolist() and r['held_out_images']==len(test)
        assert r['train_images']==len(train) and r['held_out_groups']==len(set(g[test]))
        expected_oof=np.empty(len(train));map_train={v:i for i,v in enumerate(train)}
        folds=json.loads((OUT/held/'folds.json').read_text())
        specs=[]
        for fold in folds:
            tr=np.array(fold['train_rows']);va=np.array(fold['validation_rows'])
            assert set(tr)|set(va)==set(train) and not set(tr)&set(va)
            assert not set(g[tr])&set(g[va]) and not set(g[train])&set(g[test])
            assert held not in set(s[tr])|set(s[va]);coverage.update(va.tolist())
            specs.append((fold['fit'],tr,va))
        assert coverage==Counter(train.tolist())
        specs.append((held+'_final',train,test))
        final=None
        for name,tr,va in specs:
            info=json.loads((OUT/'models'/f'{name}.json').read_text())
            assert info['lock_sha256']==digest(OUT/'lock.json')
            assert info['model_sha256']==digest(OUT/'models'/f'{name}.npz')
            assert info['training_groups']==sorted(set(g[tr])) and info['converged']
            assert info['max_absolute_gradient']<=1e-5
            with np.load(OUT/'models'/f'{name}.npz') as f:model={k:f[k] for k in f.files}
            w=sample_weights(y[tr],s[tr],g[tr],True);scaler=StandardScaler().fit(x[tr],sample_weight=w)
            np.testing.assert_allclose(scaler.mean_,model['mean'],rtol=0,atol=1e-12)
            np.testing.assert_allclose(scaler.scale_,model['scale'],rtol=0,atol=1e-12)
            theta=np.r_[model['coef'],model['intercept']]
            value,grad=objective(theta,scaler.transform(x[tr]),y[tr],w)
            assert abs(value-info['objective'])<1e-12 and abs(grad).max()<=1e-5
            if name.endswith('_final'):final=model
            else:
                for i,score in zip(va,decision(model,x[va])):expected_oof[map_train[i]]=score
            fits+=1
        with np.load(OUT/held/'oof.npz') as f:
            assert np.array_equal(f['rows'],train)
            np.testing.assert_allclose(expected_oof,f['logits'],rtol=0,atol=1e-12)
            oofp=expit(expected_oof*r['calibration_coef'][0][0]+r['calibration_intercept'][0])
            np.testing.assert_allclose(oofp,f['calibrated_probabilities'],rtol=0,atol=1e-12)
            original_oofp=f['calibrated_probabilities'].copy()
        assert choose_threshold(y[train],original_oofp,g[train],s[train])==r['threshold']
        pred=[p for p in all_pred if p['source']==held]
        marker=json.loads((OUT/held/'EVALUATION_STARTED.json').read_text())
        assert marker['selection_sha256']==digest(OUT/held/'training_side_selection.json')
        assert marker['model_sha256']==digest(OUT/'models'/f'{held}_final.npz')
        ix=np.array([int(p['row']) for p in pred]);assert ix.tolist()==test.tolist()
        for p in pred:
            row=rs[int(p['row'])];assert row['source']==held and row['group']==p['group']
            assert int(p['actual'])==int(row['label']=='malignant') and float(p['threshold'])==r['threshold']
        p=np.array([float(v['probability']) for v in pred]);raw=decision(final,x[ix])
        expected=expit(raw*r['calibration_coef'][0][0]+r['calibration_intercept'][0])
        max_error=max(max_error,float(abs(p-expected).max()));assert abs(p-expected).max()<1e-12
        gy,gp,gg=aggregate(y[ix],p,g[ix])
        for observed,stored in [(metric(gy,gp,r['threshold']),r['group_metrics']),
                                (metric(y[ix],p,r['threshold']),r['image_metrics'])]:
            for k,v in observed.items():
                if v is None:assert stored[k] is None
                else:assert abs(v-stored[k])<1e-12,(held,k,v,stored[k])
        if repeat_intervals:
            assert intervals(gy,gp,gg,r['threshold'])==r['group_intervals']
            assert intervals(y[ix],p,g[ix],r['threshold'])==r['image_intervals']
    assert fits==15
    prior=json.loads(Path('reports/design_audit_v1/prior_artifact_hashes.json').read_text())
    for p,h in prior.items():assert digest(p)==h,p
    from backend.app.inference import model_availability
    availability=model_availability();assert not availability['inference_enabled']
    save(OUT/'verification.json',{'verified_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        'all_eligible_images_scored_once':len(all_pred),'target_fits_verified':fits,
        'all_training_side_sources_isolated':True,'all_inner_validation_groups_isolated':True,
        'all_weighted_scalers_reproduced':True,'all_solver_objectives_and_gradients_verified':True,
        'oof_and_final_saved_model_replay':True,'maximum_final_probability_replay_error':max_error,
        'all_metrics_reproduced':True,'intervals_reproduced':repeat_intervals,
        'prior_artifacts_unchanged':len(prior),'new_target_fit':False,'serving_status':availability})
    print('Verified15 target fits, source/group isolation, convergence, OOF/calibration/threshold replay and2232 predictions.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--intervals',action='store_true');args=parser.parse_args();verify(args.intervals)
