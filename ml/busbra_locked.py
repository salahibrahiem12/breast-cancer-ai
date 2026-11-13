"""Prepare, lock, then once-only evaluate a fixed encoder on reserved BUS-BRA."""
import argparse
import csv
from collections import defaultdict,Counter
from datetime import datetime,timezone,timedelta
import hashlib
from io import BytesIO,StringIO
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from PIL import Image
from ml.inspect_tcia_breast import fingerprint,safe_members
from ml.development_baseline import development_rows
from ml.audit_tcia_evaluation import binary_metrics,group_intervals

ROOT=Path('reports/busbra_locked_v1')
DATA=Path('data/research_data/busbra_reserved_v1')
WEIGHTS=Path('reports/bounded_development_v1/preserved/pretrained_resnet18.pth')
MANIFEST=Path('data/prepared/mendeley_dev_v2/split_manifest.csv')
SOURCE=Path('data/raw/mendeley_v1/official-v1')

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stamp():return datetime.now(timezone.utc).isoformat()
def save(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')

def encoder():
    import torch,torchvision
    from torchvision import transforms
    torch.set_num_threads(2);torch.manual_seed(42)
    model=torchvision.models.resnet18(weights=None)
    model.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True))
    model.fc=torch.nn.Identity();model.eval()
    transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    return model,transform

def embeddings(payloads):
    import torch
    model,transform=encoder();features=[];batch=[]
    with torch.inference_mode():
        for payload in payloads:
            with Image.open(BytesIO(payload)) as image:batch.append(transform(image.convert('RGB')))
            if len(batch)==16:
                features.append(model(torch.stack(batch)).numpy());batch=[]
        if batch:features.append(model(torch.stack(batch)).numpy())
    return np.concatenate(features)

def prepare():
    if ROOT.exists():raise FileExistsError('Use the existing frozen lock; do not overwrite.')
    ROOT.mkdir(parents=True)
    # Every available classifier-source image, including legacy/unverified data,
    # is inventoried for overlap only, never used for fitting.
    reference={};counts={}
    for directory in [Path('data/raw'),Path('data/processed'),Path('data/processed_busi')]:
        count=0
        for path in directory.rglob('*'):
            if path.is_file() and path.suffix.lower() in ['.png','.bmp','.jpg','.jpeg'] and not any(s in path.stem.lower() for s in ['_mask','_tumor','_other']):
                f=fingerprint(path.read_bytes());reference[f['pixel_sha256']]=f;count+=1
        counts[str(directory)]=count
    for r in csv.DictReader(open('data/research_data/tcia_breast_lesions_usg_2026-11-03/inventory/cohort_inventory.csv')):reference[r['pixel_sha256']]=r
    reserved=list(csv.DictReader((DATA/'inventory.csv').open()))
    reservation=json.loads((DATA/'reservation.json').read_text())
    if digest(DATA/'inventory.csv')!=reservation['inventory_sha256'] or digest(DATA/'BUSBRA.zip')!=reservation['archive_sha256']:raise ValueError('Reservation changed.')
    hits=[];excluded=set()
    for r in reserved:
        for other in reference.values():
            if r['pixel_sha256']==other['pixel_sha256'] or r['sha256']==other['sha256'] or any((int(r[k])^int(other[k])).bit_count()<=5 for k in ['dhash','phash']):
                hits.append({'source_case':r['source_case'],'archive_path':r['archive_path']});excluded.add(r['source_case']);break
    with ZipFile(DATA/'BUSBRA.zip') as archive:
        safe_members(archive)
        metadata=list(csv.DictReader(StringIO(archive.read('BUSBRA/bus_data.csv').decode('utf-8-sig'))))
    by_id={r['ID']:r for r in metadata}
    for r in reserved:
        raw=by_id[Path(r['archive_path']).stem]
        if raw['Case']!=r['source_case'] or raw['Pathology']!=r['label']:raise ValueError('Ground-truth mismatch.')
    eligible=[r for r in reserved if r['source_case'] not in excluded]
    if not eligible:raise ValueError('No eligible cohort.')
    save(ROOT/'precheck.json',{'created_utc':stamp(),'reference_paths_scanned':counts,'unique_reference_pixel_arrays':len(reference),'candidate_overlap_images':len(hits),'excluded_source_cases':len(excluded),'excluded_images':len(reserved)-len(eligible),'eligible_images':len(eligible),'eligible_cases':len(set(r['source_case'] for r in eligible)),'patient_semantics':'unresolved; source-case grouping only','license':'official CC BY 4.0, research-use scope','no_reserved_predictions_yet':True})
    with (ROOT/'eligible_inventory.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(reserved[0]));w.writeheader();w.writerows(eligible)
    save(ROOT/'overlap_exclusions.json',hits)
    # Fit exactly one selected pipeline on eligible development data only.
    _,rows=development_rows(SOURCE,MANIFEST)
    x=embeddings((SOURCE/r['path']).read_bytes() for r in rows)
    y=np.array([r['label']=='malignant' for r in rows],dtype=int)
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    import sklearn,torch,torchvision,PIL
    scaler=StandardScaler().fit(x)
    classifier=LogisticRegression(C=1,class_weight='balanced',solver='lbfgs',max_iter=2000,random_state=42).fit(scaler.transform(x),y)
    np.savez(ROOT/'final_linear.npz',mean=scaler.mean_,scale=scaler.scale_,coef=classifier.coef_,intercept=classifier.intercept_,classes=classifier.classes_)
    lock={'created_utc':stamp(),'created_dubai':datetime.now(timezone(timedelta(hours=4))).isoformat(),
          'architecture':'ImageNet ResNet18 frozen, fc Identity, 512 pooled features + train-only StandardScaler + balanced L2 logistic C=1 lbfgs max_iter=2000',
          'class_order':['benign','malignant'],'preprocessing':'RGB; bilinear Resize224x224 antialias True; ToTensor; mean .485,.456,.406 / std .229,.224,.225; full frame; no augmentation',
          'development_images':len(rows),'final_training':'one refit on all 162 eligible development images; 88 old-test-linked exclusions retained; no reserved data used',
          'threshold':.5,'calibration':'none; raw logistic probabilities; no external calibration','seed':42,
          'primary_endpoint':'balanced accuracy at fixed .5 over source Case mean image probability',
          'unit':'source Case, not verified patient, breast, lesion or examination','aggregation':'arithmetic mean of all eligible image probabilities within Case; one consistent source binary label required',
          'intervals':'2000 percentile bootstrap resamples of source Case, seed42; conditional case-level uncertainty, not verified patient-independent CI',
          'secondary':'image metrics with source-Case cluster bootstrap; descriptive source-case Device subgroup, mixed devices labelled mixed; no multiplicity claims',
          'exclusions':'entire Case excluded for any byte/pixel match or dHash/pHash distance<=5 to any available classifier-source image; unknown/missing labels, missing/corrupt images or contradictory Case labels cause failure, no silent exclusion',
          'grouping_limit':'Official record and accessible paper abstract give patient counts but do not define the Case field. Full article was not accessible. No patient-independence claim; within-person correlation across distinct Case values cannot be ruled out.',
          'clinical_acceptance_threshold':None,'public_classification_enabled':False,
          'hashes':{'weights':digest(WEIGHTS),'linear':digest(ROOT/'final_linear.npz'),'development_manifest':digest(MANIFEST),'reserved_archive':digest(DATA/'BUSBRA.zip'),'eligible_inventory':digest(ROOT/'eligible_inventory.csv'),'precheck':digest(ROOT/'precheck.json'),'runner':digest(Path(__file__))},
          'software':{'torch':torch.__version__,'torchvision':torchvision.__version__,'sklearn':sklearn.__version__,'numpy':np.__version__,'Pillow':PIL.__version__}}
    save(ROOT/'lock.json',lock)
    (ROOT/'LOCK_SHA256.txt').write_text(digest(ROOT/'lock.json'))
    protocol='# Locked BUS-BRA evaluation v1\n\n'+json.dumps(lock,indent=2)+'\n\nLock SHA-256: '+digest(ROOT/'lock.json')+'\n'
    Path('docs/BUSBRA_LOCKED_PROTOCOL_V1.md').write_text(protocol,encoding='utf-8')
    print(json.dumps({'phase':'locked before inference','lock_sha256':digest(ROOT/'lock.json'),'precheck':json.loads((ROOT/'precheck.json').read_text())},indent=2),flush=True)

def evaluate():
    if (ROOT/'INFERENCE_STARTED.json').exists():raise FileExistsError('Inference already started; do not retry selection or overwrite.')
    lock=json.loads((ROOT/'lock.json').read_text())
    if digest(ROOT/'lock.json')!=(ROOT/'LOCK_SHA256.txt').read_text():raise ValueError('Lock modified.')
    for name,path in [('weights',WEIGHTS),('linear',ROOT/'final_linear.npz'),('development_manifest',MANIFEST),('reserved_archive',DATA/'BUSBRA.zip'),('eligible_inventory',ROOT/'eligible_inventory.csv'),('precheck',ROOT/'precheck.json'),('runner',Path(__file__))]:
        if digest(path)!=lock['hashes'][name]:raise ValueError('Locked artifact changed: '+name)
    save(ROOT/'INFERENCE_STARTED.json',{'utc':stamp(),'lock_sha256':digest(ROOT/'lock.json')})
    try:
        rows=list(csv.DictReader((ROOT/'eligible_inventory.csv').open()))
        with ZipFile(DATA/'BUSBRA.zip') as archive:
            x=embeddings(archive.read(r['archive_path']) for r in rows)
        with np.load(ROOT/'final_linear.npz',allow_pickle=False) as m:
            logits=((x-m['mean'])/m['scale'])@m['coef'].T+m['intercept']
            from scipy.special import expit
            scores=expit(logits.ravel())
        grouped=defaultdict(list)
        for row,score in zip(rows,scores,strict=True):grouped[row['source_case']].append((row,float(score)))
        cases=[]
        for key,items in grouped.items():
            labels={r['label'] for r,p in items}
            if len(labels)!=1 or not labels<={'benign','malignant'}:raise ValueError('Conflicting or unknown Case label.')
            devices={r['device'] for r,p in items}
            cases.append({'source_case':key,'actual':next(iter(labels)),'malignant_probability':float(np.mean([p for r,p in items])),'image_count':len(items),'device':next(iter(devices)) if len(devices)==1 else 'mixed'})
        y=np.array([r['actual']=='malignant' for r in cases],dtype=int);p=np.array([r['malignant_probability'] for r in cases]);g=np.array([r['source_case'] for r in cases])
        image_y=np.array([r['label']=='malignant' for r in rows],dtype=int);image_groups=[r['source_case'] for r in rows]
        result={'completed_utc':stamp(),'lock_sha256':digest(ROOT/'lock.json'),'case_count':len(cases),'image_count':len(rows),'class_counts_cases':dict(Counter(r['actual'] for r in cases)),
                'primary_case_metrics':binary_metrics(y,p),'case_bootstrap_intervals':group_intervals(y,p,g,iterations=2000),
                'secondary_image_metrics':binary_metrics(image_y,scores),'image_case_cluster_intervals':group_intervals(image_y,scores,image_groups,iterations=2000),
                'device_subgroups':{},'patient_independent':False,'inference_enabled':False,'eligibility_gate_evaluated':False,
                'interpretation':'Offline lesion-containing research classification only; not upload-to-result evaluation; source-case uncertainty may underestimate person-level dependence; no clinical acceptance threshold.'}
        for device in sorted(set(r['device'] for r in cases)):
            idx=[i for i,r in enumerate(cases) if r['device']==device]
            result['device_subgroups'][device]={'cases':len(idx),'class_counts':dict(Counter(cases[i]['actual'] for i in idx)),'metrics':binary_metrics(y[idx],p[idx])}
        for filename,data in [('image_predictions.csv',[{'archive_path':r['archive_path'],'source_case':r['source_case'],'actual':r['label'],'malignant_probability':float(s)} for r,s in zip(rows,scores)]),('case_predictions.csv',cases)]:
            with (ROOT/filename).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
        result['prediction_hashes']={name:digest(ROOT/name) for name in ['image_predictions.csv','case_predictions.csv']}
        save(ROOT/'evaluation.json',result)
        save(ROOT/'INFERENCE_COMPLETED.json',{'utc':stamp(),'evaluation_sha256':digest(ROOT/'evaluation.json')})
        print(json.dumps({'case_count':len(cases),'primary_case_metrics':result['primary_case_metrics'],'case_bootstrap_intervals':result['case_bootstrap_intervals']},indent=2),flush=True)
    except Exception as error:
        save(ROOT/'FAILURE.json',{'utc':stamp(),'error_type':type(error).__name__,'message':str(error)})
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','evaluate']);a=p.parse_args()
    prepare() if a.phase=='prepare' else evaluate()
