"""Bounded exploratory source-held-out development; never a fresh-test claim."""
import csv,json,hashlib
from collections import defaultdict,Counter
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
from datetime import datetime,timezone
import numpy as np
import torch
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score
from scipy.special import expit,logit
from torchvision.models import resnet18
from torchvision import transforms
from ml.inspect_tcia_breast import fingerprint
from ml.prepare_dataset import UnionFind
from ml.development_baseline import development_rows
from ml.audit_tcia_evaluation import binary_metrics

OUT=Path('reports/cross_source_v1')
WEIGHTS=Path('reports/bounded_development_v1/preserved/pretrained_resnet18.pth')
def save(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
def read(path):return list(csv.DictReader(open(path,newline='',encoding='utf-8')))
def payload(row):
    if row['source']=='busbra':
        with ZipFile('data/research_data/busbra_reserved_v1/BUSBRA.zip') as z:return z.read(row['archive_path'])
    return Path(row['file']).read_bytes()
def transform():return transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
def net():
    m=resnet18(weights=None);m.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True));m.fc=torch.nn.Identity();return m.eval()
def extract(m,images):
    with torch.inference_mode():return torch.cat([m(x) for x in images.split(16)]).numpy()
def tensors(rows,mode=None):
    result=[];t=transform()
    for row in rows:
        with Image.open(BytesIO(payload(row))) as image:
            a=np.array(image.convert('RGB'))
        if mode:
            h,w=a.shape[:2];mask=np.ones((h,w),bool);mask[h//10:h-h//10,w//10:w-w//10]=False
            if mode=='center_removed':mask=~mask
            a[mask]=np.median(a,axis=(0,1)).astype(np.uint8)
        result.append(t(Image.fromarray(a)))
    return torch.stack(result)
def metric(y,p,threshold):
    r=binary_metrics(y,p);d=binary_metrics(y,np.where(np.asarray(p)>=threshold,.75,.25))
    for k in ['tn','fp','fn','tp','sensitivity','specificity','precision_ppv','npv','accuracy','balanced_accuracy']:r[k]=d[k]
    return r
def aggregate(y,p,g):
    unique=np.unique(g);return np.array([y[g==v][0] for v in unique]),np.array([np.mean(p[g==v]) for v in unique]),unique
def intervals(y,p,g,t,n=1000):
    rng=np.random.default_rng(42);unique=np.unique(g);members=[np.flatnonzero(g==v) for v in unique];draws=defaultdict(list)
    for _ in range(n):
        ix=np.concatenate([members[i] for i in rng.integers(len(unique),size=len(unique))])
        for k,v in metric(y[ix],p[ix],t).items():
            if k not in ['tn','fp','fn','tp'] and v is not None:draws[k].append(v)
    return {k:{'lower_95':float(np.quantile(v,.025)),'upper_95':float(np.quantile(v,.975)),'valid_resamples':len(v)} for k,v in draws.items()}
def threshold(y,p,g):
    y,p,g=aggregate(y,p,g);choices=[]
    for t in np.unique(np.r_[p,.5,0,1]):
        m=metric(y,p,float(t));choices.append((m['balanced_accuracy'],m['specificity'],float(t)))
    return max(choices)[2]

def inventory():
    _,mr=development_rows(Path('data/raw/mendeley_v1/official-v1'),Path('data/prepared/mendeley_dev_v2/split_manifest.csv'))
    rows=[]
    for r in mr:
        path=Path('data/raw/mendeley_v1/official-v1')/r['path'];rows.append({'source':'mendeley','label':r['label'],'unit':'mendeley:'+r['group_id'],'file':str(path),**fingerprint(path.read_bytes())})
    ti={Path(r['path']).name:r for r in read('data/research_data/tcia_breast_lesions_usg_2026-11-03/inventory/cohort_inventory.csv')}
    for r in read('data/prepared/tcia_breast_biopsy_external_v1/split_manifest.csv'):
        rows.append({'source':'tcia','label':r['label'],'unit':r['patient_id'],'file':str(Path('data/raw/tcia_breast_biopsy_external_v1')/r['path']),**{k:ti[r['path']][k] for k in ['sha256','pixel_sha256','dhash','phash']}})
    for r in read('reports/busbra_locked_v1/eligible_inventory.csv'):rows.append({'source':'busbra','label':r['label'],'unit':'busbra:'+r['source_case'],'archive_path':r['archive_path'],**{k:r[k] for k in ['sha256','pixel_sha256','dhash','phash']}})
    union=UnionFind(len(rows));owners={};links=0
    for i,r in enumerate(rows):
        if r['unit'] in owners:union.union(i,owners[r['unit']])
        owners[r['unit']]=i
        for j,b in enumerate(rows[:i]):
            if r['pixel_sha256']==b['pixel_sha256'] or r['sha256']==b['sha256'] or any((int(r[k])^int(b[k])).bit_count()<=5 for k in ['dhash','phash']):union.union(i,j);links+=1
    components=defaultdict(list)
    for i,r in enumerate(rows):components[union.find(i)].append(r)
    pure=[];excluded=Counter()
    for key,rs in components.items():
        if len({r['source'] for r in rs})>1:excluded['cross_source_component_images']+=len(rs);continue
        if len({r['label'] for r in rs})>1:excluded['conflicting_label_component_images']+=len(rs);continue
        for r in rs:r['group']='global:'+str(key)
        pure.append(rs)
    rng=np.random.default_rng(42);selected=[]
    for source in ['mendeley','tcia','busbra']:
        for label in ['benign','malignant']:
            candidates=[c for c in pure if c[0]['source']==source and c[0]['label']==label]
            ix=rng.permutation(len(candidates))[:16]
            for i in ix:selected.extend(candidates[int(i)])
    save('selection.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'inventory_images':len(rows),'hash_candidate_links':links,'exclusions':dict(excluded),'selected_images':dict(Counter(r['source'] for r in selected)),'selected_groups':{s:len({r['group'] for r in selected if r['source']==s}) for s in ['mendeley','tcia','busbra']},'rows':selected})
    return selected

def run():
    if (OUT/'result.json').exists():raise FileExistsError('Completed bounded run exists.')
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
    rows=inventory();images=tensors(rows);m=net();x=extract(m,images)
    state=torch.load(WEIGHTS,map_location='cpu',weights_only=True)
    default_counters=[key for key in m.state_dict() if key not in state]
    equality=all(torch.equal(value,state[key]) if key in state else key.endswith('.num_batches_tracked') and int(value)==0 for key,value in m.state_dict().items())
    differences={}
    for n in ['mendeley','tcia','busbra']:
        ix=[i for i,r in enumerate(rows) if r['source']==n];a=x[ix];s=np.linalg.svd(a-a.mean(0),compute_uv=False)
        differences[n]={'images':len(ix),'zero_variance_dimensions':int((a.var(0)<1e-12).sum()),'median_feature_variance':float(np.median(a.var(0))),'centered_numerical_rank':int((s>max(a.shape)*np.finfo(np.float32).eps*s[0]).sum()),'mean_feature_norm':float(np.linalg.norm(a,axis=1).mean())}
    # Replay the exact original frozen-fold development models, never external models.
    originals=read('reports/bounded_development_v1/oof_predictions.csv');lookup={r['path']:r for r in originals}
    _,dev=development_rows(Path('data/raw/mendeley_v1/official-v1'),Path('data/prepared/mendeley_dev_v2/split_manifest.csv'))
    devrows=[{'source':'mendeley','file':str(Path('data/raw/mendeley_v1/official-v1')/r['path'])} for r in dev]
    dx=extract(m,tensors(devrows));errors=[]
    for i,r in enumerate(dev):
        original=lookup[r['path']]
        with np.load('reports/bounded_development_v1/frozen_encoder_fold'+original['fold']+'.npz') as z:
            # StandardScaler on float32 inputs subtracts and divides in float32.
            a=dx[i:i+1].copy();a-=z['mean'];a/=z['scale'];p=expit((a@z['coef'].T+z['intercept']).item())
            errors.append(abs(p-float(original['frozen_encoder'])))
    repeat=extract(m,images[:3]);single=np.vstack([extract(m,images[i:i+1]) for i in range(3)])
    audit={'pretrained_tensors_equal':equality,'weights_sha256':hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),'eval_mode':not m.training,'feature_layer':'global average pool after layer4, fc Identity, 512 dimensions','actual_transform':repr(transform()),'rgb_channels':3,'feature_statistics':differences,'repeated_batch_max_error':float(np.max(abs(repeat-x[:3]))),'single_vs_batch_max_error':float(np.max(abs(single-repeat))),'development_replay_max_probability_error':float(max(errors)),'collapsed':bool(np.max(x.var(0))<1e-12)}
    # An independently assembled preprocessing formula on one source example.
    with Image.open(BytesIO(payload(rows[0]))) as image:
        native=np.asarray(image.convert('RGB').resize((224,224),Image.Resampling.BILINEAR),dtype=np.float32)/255
    independent=(native.transpose(2,0,1)-np.array([.485,.456,.406],dtype=np.float32)[:,None,None])/np.array([.229,.224,.225],dtype=np.float32)[:,None,None]
    audit['independent_transform_max_error']=float(np.max(abs(independent-images[0].numpy())))
    audit['legacy_default_zero_batchnorm_counters']=default_counters
    np.savez(OUT/'features.npz',selected=x,development=dx)
    save('encoder_audit.json',audit)
    if not equality or audit['collapsed'] or max(errors)>1e-5:raise ValueError('Encoder audit failed; stop before model comparison.')
    y=np.array([r['label']=='malignant' for r in rows],int);g=np.array([r['group'] for r in rows]);sources=np.array([r['source'] for r in rows]);results={};pred=[]
    alt={kind:extract(m,tensors(rows,kind)) for kind in ['border_removed','center_removed']}
    source_calls=np.empty(len(rows),dtype=object)
    for tr,va in StratifiedGroupKFold(3,shuffle=True,random_state=42).split(x,sources,g):
        scaler=StandardScaler().fit(x[tr]);c=LogisticRegression(C=1,max_iter=2000).fit(scaler.transform(x[tr]),sources[tr]);source_calls[va]=c.predict(scaler.transform(x[va]))
    save('source_probe.json',{'accuracy':float((source_calls==sources).mean()),'balanced_accuracy':float(balanced_accuracy_score(sources,source_calls)),'majority_image_accuracy':float(max(Counter(sources).values())/len(rows)),'causal_claim':False})
    for fold,source in enumerate(['mendeley','tcia','busbra']):
        outer=np.flatnonzero(sources!=source);test=np.flatnonzero(sources==source)
        strata=np.array([str(s)+':'+str(v) for s,v in zip(sources[outer],y[outer])])
        tr,va=next(StratifiedGroupKFold(4,shuffle=True,random_state=42).split(x[outer],strata,g[outer]));tr,va=outer[tr],outer[va]
        assert not set(g[tr])&set(g[va]) and not set(g[outer])&set(g[test])
        results[source]={}
        for arm in ['frozen','fine_tuned']:
            if arm=='frozen':
                scaler=StandardScaler().fit(x[tr]);c=LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=42).fit(scaler.transform(x[tr]),y[tr])
                val_logits=c.decision_function(scaler.transform(x[va]));test_logits=c.decision_function(scaler.transform(x[test]))
                np.savez(OUT/f'{source}_linear.npz',mean=scaler.mean_,scale=scaler.scale_,coef=c.coef_,intercept=c.intercept_)
            else:
                torch.manual_seed(42+fold);model=resnet18(weights=None);model.load_state_dict(state);model.fc=torch.nn.Linear(512,2)
                for key,p in model.named_parameters():p.requires_grad=key.startswith(('layer4.','fc.'))
                optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
                counts=np.bincount(y[tr],minlength=2);loss_fn=torch.nn.CrossEntropyLoss(weight=torch.tensor(len(tr)/(2*counts),dtype=torch.float32))
                generator=torch.Generator().manual_seed(42+fold)
                for epoch in range(2):
                    model.train()
                    for module in model.modules():
                        if isinstance(module,torch.nn.BatchNorm2d):module.eval()
                    order=tr[torch.randperm(len(tr),generator=generator).numpy()]
                    for batch in np.array_split(order,int(np.ceil(len(order)/16))):
                        optimizer.zero_grad();loss=loss_fn(model(images[batch]),torch.tensor(y[batch]));loss.backward();optimizer.step()
                    print(source,arm,'epoch',epoch+1,flush=True)
                model.eval()
                with torch.inference_mode():
                    vl=np.concatenate([model(b).numpy() for b in images[va].split(16)]);tl=np.concatenate([model(b).numpy() for b in images[test].split(16)])
                val_logits=vl[:,1]-vl[:,0];test_logits=tl[:,1]-tl[:,0]
                torch.save(model.state_dict(),OUT/f'{source}_fine_tuned.pth')
            calibrator=LogisticRegression(C=1,max_iter=2000,random_state=42).fit(val_logits[:,None],y[va]);vp=calibrator.predict_proba(val_logits[:,None])[:,1];p=calibrator.predict_proba(test_logits[:,None])[:,1];t=threshold(y[va],vp,g[va])
            gy,gp,gg=aggregate(y[test],p,g[test]);my=metric(gy,gp,t)
            result={'train_images':len(tr),'inner_validation_images':len(va),'outer_images':len(test),'outer_groups':len(gg),'threshold':t,'calibration_coef':calibrator.coef_.tolist(),'calibration_intercept':calibrator.intercept_.tolist(),'group_class_counts':dict(Counter(map(str,gy.tolist()))),'group_majority_accuracy':float(max(Counter(gy).values())/len(gy)),'image_majority_accuracy':float(max(Counter(y[test]).values())/len(test)),'majority_balanced_accuracy':.5,'group_metrics':my,'group_intervals':intervals(gy,gp,gg,t),'image_metrics':metric(y[test],p,t),'image_intervals':intervals(y[test],p,g[test],t),'fixed_05_group_metrics':metric(gy,gp,.5),'inner_train_groups':sorted(set(g[tr])),'inner_validation_groups':sorted(set(g[va])),'held_out_groups':sorted(set(g[test]))}
            if arm=='frozen':
                result['ablations']={}
                for kind,xa in alt.items():
                    ap=calibrator.predict_proba(c.decision_function(scaler.transform(xa[test]))[:,None])[:,1];ay,ag,unused=aggregate(y[test],ap,g[test])
                    result['ablations'][kind]={'group_metrics':metric(ay,ag,t),'mean_absolute_image_score_change':float(np.mean(abs(ap-p))),'image_call_changes':int(((ap>=t)!=(p>=t)).sum())}
            results[source][arm]=result
            pred.extend({'row':int(i),'source':source,'group':rows[i]['group'],'actual':int(y[i]),'arm':arm,'probability':float(score),'threshold':t} for i,score in zip(test,p))
            save('partial_results.json',results)
    save('result.json',{'exploratory_inspected_sources':True,'patient_independence_claim':False,'inference_enabled':False,'results':results,'protocol_sha256':hashlib.sha256(Path('docs/CROSS_SOURCE_PROTOCOL_V1.md').read_bytes()).hexdigest(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    with (OUT/'predictions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(pred[0]));w.writeheader();w.writerows(pred)
    print('completed all six models',flush=True)

if __name__=='__main__':run()
