"""One predeclared full-eligible frozen baseline, resumable without model search."""
import argparse
import csv
import hashlib
import json
import os
import platform
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ml.cross_source import net, tensors, payload, metric, aggregate, intervals, WEIGHTS

OUT=Path('reports/full_source_baseline_v1')
PROTOCOL=Path('docs/FULL_SOURCE_BASELINE_PROTOCOL_V1.md')
MANIFEST=Path('reports/design_audit_v1/eligible_manifest.json')


def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
    p=Path(p); tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8'); os.replace(tmp,p)
def atomic_npz(path,**items):
    path=Path(path);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f: np.savez(f,**items)
    os.replace(tmp,path)
def rows(): return json.loads(MANIFEST.read_text())


def lock():
    import torch,torchvision,sklearn,scipy,PIL,sys
    OUT.mkdir(exist_ok=True)
    if (OUT/'lock.json').exists(): raise FileExistsError('Protocol already locked; use features/run to resume.')
    assert Path('docs/EXPERIMENT_DESIGN_AUDIT_V1.md').exists(), 'Deliver the audit before the next experiment.'
    assert Path('reports/design_audit_v1/training_diagnosis.json').exists()
    rs=rows(); assert len(rs)==2232 and len({r['group'] for r in rs})==1298
    files=[PROTOCOL,Path(__file__),MANIFEST,WEIGHTS,Path('ml/cross_source.py'),
           Path('ml/audit_tcia_evaluation.py'),Path('reports/design_audit_v1/prior_artifact_hashes.json')]
    obj={'locked_utc':datetime.now(timezone.utc).isoformat(),
         'hashes':{str(p).replace('\\','/'):digest(p) for p in files},
         'images':len(rs),'groups':len({r['group'] for r in rs}),
         'inspected_development_sources':True,'inference_enabled':False,
         'python':platform.python_version(),'python_executable':sys.executable,
         'versions':{'torch':torch.__version__,'torchvision':torchvision.__version__,
                     'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'Pillow':PIL.__version__}}
    save(OUT/'lock.json',obj); (OUT/'LOCK_SHA256.txt').write_text(digest(OUT/'lock.json'))
    print('Locked full eligible baseline before model fitting.',flush=True)


def check_lock():
    obj=json.loads((OUT/'lock.json').read_text());assert digest(OUT/'lock.json')==(OUT/'LOCK_SHA256.txt').read_text()
    for p,h in obj['hashes'].items():assert digest(p)==h,p
    return obj


def features():
    import torch
    check_lock();torch.set_num_threads(2);rs=rows(); directory=OUT/'feature_chunks';directory.mkdir(exist_ok=True)
    pilot=json.loads(Path('reports/cross_source_v1/selection.json').read_text())['rows']
    cache={}
    with np.load('reports/cross_source_v1/features.npz') as f:
        for r,x in zip(pilot,f['selected']):cache[r['sha256']]=x
    m=net(); output=[]; reused=0; start=time.perf_counter()
    for begin in range(0,len(rs),32):
        batch=rs[begin:begin+32];file=directory/f'{begin:06d}.npz';sha=np.array([r['sha256'] for r in batch])
        if file.exists():
            with np.load(file) as f:assert np.array_equal(f['sha256'],sha);x=f['features']
        else:
            for r in batch:assert hashlib.sha256(payload(r)).hexdigest()==r['sha256']
            missing=[i for i,r in enumerate(batch) if r['sha256'] not in cache]
            x=np.empty((len(batch),512),np.float32)
            for i,r in enumerate(batch):
                if r['sha256'] in cache:x[i]=cache[r['sha256']];reused+=1
            if missing:
                with torch.inference_mode():a=m(tensors([batch[i] for i in missing])).numpy()
                x[missing]=a
            assert np.isfinite(x).all();atomic_npz(file,features=x,sha256=sha)
        output.append(x)
        if begin%128==0:print('features',begin+len(batch),'/',len(rs),flush=True)
    atomic_npz(OUT/'features.npz',features=np.concatenate(output),sha256=np.array([r['sha256'] for r in rs]))
    save(OUT/'feature_receipt.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),'images':len(rs),
        'features':512,'new_or_resumed_seconds':time.perf_counter()-start,'cached_pilot_rows_reused_this_invocation':reused,
        'features_sha256':digest(OUT/'features.npz'),'input_sha256_checked_on_new_chunk_creation':True})


def sample_weights(y,s,g,balance_diagnosis):
    units={v:np.flatnonzero(g==v) for v in np.unique(g)}
    counts=Counter(); nsource=len(set(s))
    for ix in units.values():
        assert len(set(s[ix]))==len(set(y[ix]))==1
        counts[(s[ix[0]],int(y[ix[0]]) if balance_diagnosis else -1)]+=1
    w=np.empty(len(y),float)
    for ix in units.values():
        key=(s[ix[0]],int(y[ix[0]]) if balance_diagnosis else -1)
        w[ix]=len(units)/(nsource*(2 if balance_diagnosis else 1)*counts[key]*len(ix))
    assert abs(w.sum()-len(units))<1e-8
    return w


def objective(theta,z,y,w,c=1):
    logits=z@theta[:-1]+theta[-1];total=w.sum()
    value=np.dot(w,np.logaddexp(0,logits)-y*logits)/total+np.dot(theta[:-1],theta[:-1])/(2*c*total)
    error=w*(expit(logits)-y)/total
    grad=np.r_[z.T@error+theta[:-1]/(c*total),error.sum()]
    return float(value),grad


def fit(name,x,y,s,g):
    dest=OUT/'models';dest.mkdir(exist_ok=True);model=dest/f'{name}.npz';record=dest/f'{name}.json'
    w=sample_weights(y,s,g,True);scaler=StandardScaler().fit(x,sample_weight=w);z=scaler.transform(x.astype(float))
    if model.exists() and record.exists():
        info=json.loads(record.read_text());assert info['model_sha256']==digest(model)
        assert info['training_groups']==sorted(set(g));assert info['converged']
        with np.load(model) as f:return {k:f[k] for k in f.files},info
    resume=dest/f'{name}.resume.npz';curvefile=dest/f'{name}.curve.json';curve=[];initial=np.zeros(x.shape[1]+1)
    if resume.exists():
        with np.load(resume) as f: initial=f['theta']
        if curvefile.exists():curve=json.loads(curvefile.read_text())
    resumed=bool(resume.exists());start=time.perf_counter()
    def callback(theta):
        value,grad=objective(theta,z,y,w)
        curve.append({'iteration':len(curve)+1,'objective':value,'max_absolute_gradient':float(abs(grad).max())})
        if len(curve)%25==0:
            atomic_npz(resume,theta=theta);save(curvefile,curve)
    opt=minimize(objective,initial,args=(z,y,w),jac=True,method='L-BFGS-B',callback=callback,
        options={'maxiter':1000,'gtol':1e-6,'ftol':1e-12,'maxls':50})
    value,gradient=objective(opt.x,z,y,w);success=bool(opt.success and abs(gradient).max()<=1e-5)
    info={'fit':name,'train_images':len(y),'train_groups':len(set(g)), 'training_groups':sorted(set(g)),
          'source_class_images':{src:{str(label):int(((s==src)&(y==label)).sum()) for label in [0,1]} for src in sorted(set(s))},
          'objective':value,'max_absolute_gradient':float(abs(gradient).max()),'converged':success,
          'optimizer_success':bool(opt.success),'optimizer_message':str(opt.message),'optimizer_iterations':int(opt.nit),
          'elapsed_seconds':time.perf_counter()-start,'resume_reset_quasi_newton_history':resumed,
          'sum_sample_weights':float(w.sum()),'class_weight_sums':{str(label):float(w[y==label].sum()) for label in [0,1]},
          'lock_sha256':digest(OUT/'lock.json')}
    save(curvefile,curve)
    if not success:
        save(record,info);raise RuntimeError('Optimization failed convergence criteria: '+name)
    fitted={'mean':scaler.mean_,'scale':scaler.scale_,'coef':opt.x[:-1],'intercept':opt.x[-1:]}
    atomic_npz(model,**fitted);info['model_sha256']=digest(model);save(record,info)
    print('converged',name,'iterations',opt.nit,'gradient',info['max_absolute_gradient'],flush=True)
    return fitted,info


def decision(model,x):return ((x-model['mean'])/model['scale'])@model['coef']+model['intercept'][0]


def choose_threshold(y,p,g,s):
    gy,gp,gg=aggregate(y,p,g);gs=np.array([s[np.flatnonzero(g==v)[0]] for v in gg])
    counts=Counter(gs);w=np.array([1/counts[v] for v in gs]);choices=[]
    for t in np.unique(np.r_[gp,0,.5,1]):
        sensitivity=np.average(gp[gy==1]>=t,weights=w[gy==1]);specificity=np.average(gp[gy==0]<t,weights=w[gy==0])
        choices.append(((sensitivity+specificity)/2,specificity,float(t)))
    return max(choices)[2]


def calibration_bins(y,p):
    result=[]
    for i in range(10):
        ix=(p>=i/10)&(p<((i+1)/10) if i<9 else p<=1)
        result.append({'lower':i/10,'upper':(i+1)/10,'n':int(ix.sum()),
                       'mean_probability':float(p[ix].mean()) if ix.any() else None,
                       'malignant_fraction':float(y[ix].mean()) if ix.any() else None})
    return result


def run():
    check_lock();rs=rows()
    with np.load(OUT/'features.npz') as f:
        assert np.array_equal(f['sha256'],np.array([r['sha256'] for r in rs]));x=f['features'].astype(float)
    y=np.array([r['label']=='malignant' for r in rs],int);g=np.array([r['group'] for r in rs]);s=np.array([r['source'] for r in rs])
    results={};all_predictions=[]
    for source in ['mendeley','tcia','busbra']:
        out=OUT/source;out.mkdir(exist_ok=True)
        if (out/'result.json').exists():
            results[source]=json.loads((out/'result.json').read_text())
            all_predictions+=list(csv.DictReader((out/'predictions.csv').open()));continue
        train=np.flatnonzero(s!=source);held=np.flatnonzero(s==source);oof=np.empty(len(train));folds=[]
        strata=np.array([str(a)+':'+str(b) for a,b in zip(s[train],y[train])])
        for fold,(itr,iva) in enumerate(StratifiedGroupKFold(4,shuffle=True,random_state=42).split(x[train],strata,g[train])):
            tr,va=train[itr],train[iva]
            assert not set(g[tr])&set(g[va]) and not set(g[train])&set(g[held])
            assert source not in set(s[tr])|set(s[va])
            model,info=fit(source+f'_inner{fold}',x[tr],y[tr],s[tr],g[tr]);oof[iva]=decision(model,x[va])
            folds.append({'fold':fold,'fit':source+f'_inner{fold}',
                'train_rows':tr.tolist(),'validation_rows':va.tolist(),
                'train_groups':sorted(set(g[tr])),'validation_groups':sorted(set(g[va])),
                'converged':info['converged']})
        save(out/'folds.json',folds)
        cal_w=sample_weights(y[train],s[train],g[train],False)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            calibrator=LogisticRegression(C=1,max_iter=2000,tol=1e-6,random_state=42).fit(oof[:,None],y[train],sample_weight=cal_w)
        if ws:raise RuntimeError('Calibrator warning: '+str(ws[0].message))
        vp=calibrator.predict_proba(oof[:,None])[:,1];threshold=choose_threshold(y[train],vp,g[train],s[train])
        selection={'threshold':threshold,'calibration_coef':calibrator.coef_.tolist(),
            'calibration_intercept':calibrator.intercept_.tolist(),'calibration_iterations':calibrator.n_iter_.tolist(),
            'training_rows':train.tolist(),'training_groups':sorted(set(g[train])),
            'training_sources':sorted(set(s[train])),'held_out_source_not_used':True,
            'selection_frozen_utc':datetime.now(timezone.utc).isoformat()}
        save(out/'training_side_selection.json',selection)
        atomic_npz(out/'oof.npz',rows=train,logits=oof,calibrated_probabilities=vp)
        model,info=fit(source+'_final',x[train],y[train],s[train],g[train])
        save(out/'EVALUATION_STARTED.json',{'utc':datetime.now(timezone.utc).isoformat(),
            'selection_sha256':digest(out/'training_side_selection.json'),'model_sha256':info['model_sha256']})
        logits=decision(model,x[held]);p=calibrator.predict_proba(logits[:,None])[:,1]
        gy,gp,gg=aggregate(y[held],p,g[held]);ty,tp,tg=aggregate(y[train],vp,g[train])
        group_counts=Counter(gy);image_counts=Counter(y[held]);train_group_counts=Counter(ty)
        training_majority=max(train_group_counts,key=train_group_counts.get);prev=float(ty.mean())
        result={'held_out_source':source,'inspected_exploratory_development':True,
            'train_images':len(train),'train_groups':len(set(g[train])), 'held_out_images':len(held),'held_out_groups':len(gg),
            'group_class_counts':{str(k):v for k,v in group_counts.items()},'image_class_counts':{str(k):v for k,v in image_counts.items()},
            **selection,'group_metrics':metric(gy,gp,threshold),'image_metrics':metric(y[held],p,threshold),
            'group_intervals':intervals(gy,gp,gg,threshold),'image_intervals':intervals(y[held],p,g[held],threshold),
            'group_calibration_bins':calibration_bins(gy,gp),'image_calibration_bins':calibration_bins(y[held],p),
            'training_oof_group_metrics':metric(ty,tp,threshold),
            'training_refit_image_metrics':metric(y[train],calibrator.predict_proba(decision(model,x[train])[:,None])[:,1],threshold),
            'majority_baselines':{'held_group_majority_accuracy':max(group_counts.values())/len(gy),
                 'held_image_majority_accuracy':max(image_counts.values())/len(held),'majority_balanced_accuracy':.5,
                 'training_group_majority_class':int(training_majority),'training_group_majority_accuracy_on_held':float((gy==training_majority).mean()),
                 'training_group_prevalence':prev,'training_prevalence_constant_held_group_brier':float(np.mean((prev-gy)**2))},
            'final_optimizer':info,'patient_independence_claim':False,'inference_enabled':False}
        predictions=[{'row':int(i),'source':source,'group':rs[i]['group'],'actual':int(y[i]),
                      'raw_logit':float(logit),'probability':float(prob),'threshold':threshold} for i,logit,prob in zip(held,logits,p)]
        with (out/'predictions.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
        save(out/'result.json',result);results[source]=result;all_predictions+=predictions
        print('completed held-out source',source,flush=True)
    with (OUT/'predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_predictions[0]));writer.writeheader();writer.writerows(all_predictions)
    save(OUT/'result.json',{'results':results,'protocol_sha256':digest(PROTOCOL),'lock_sha256':digest(OUT/'lock.json'),
                          'architecture_search':False,'inference_enabled':False})
    for p,h in json.loads(Path('reports/design_audit_v1/prior_artifact_hashes.json').read_text()).items():assert digest(p)==h,p
    print('Completed all three full-eligible source-held-out folds; previous artifacts unchanged.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['lock','features','run']);args=parser.parse_args()
    with threadpool_limits(limits=2):globals()[args.stage]()
