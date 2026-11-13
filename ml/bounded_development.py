"""Three fixed development arms, identical grouped folds, no TCIA scoring."""
import csv
import hashlib
import json
from pathlib import Path
import argparse
import time

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18
from torchvision import transforms

from ml.development_baseline import development_rows, features
from ml.audit_tcia_evaluation import binary_metrics, group_intervals


def ablate(path, kind):
    with Image.open(path) as image:
        image = image.convert('L')
        array = np.array(image)
    h, w = array.shape
    mask = np.ones(array.shape, dtype=bool)
    mask[h//10:h-h//10, w//10:w-w//10] = False
    if kind == 'center_removed':
        mask = ~mask
    array[mask] = int(np.median(array))
    return np.asarray(Image.fromarray(array).resize((32,32), Image.Resampling.BILINEAR), dtype=float).ravel()/255


def run(source, manifest, output, weights):
    if (output/'comparison.json').exists():
        raise FileExistsError('Completed experiment exists.')
    all_rows, rows = development_rows(source,manifest)
    output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    images=[]
    for row in rows:
        with Image.open(source/row['path']) as image:images.append(transform(image.convert('RGB')))
    images=torch.stack(images)
    labels=np.array([r['label']=='malignant' for r in rows],dtype=int)
    groups=np.array([r['group_id'] for r in rows])
    pixels=np.array([features(source/r['path'],'baseline') for r in rows])
    encoder=resnet18(weights=None)
    encoder.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
    encoder.fc=torch.nn.Identity();encoder.eval()
    with torch.inference_mode():
        embeddings=torch.cat([encoder(batch[0]) for batch in DataLoader(TensorDataset(images),batch_size=16)]).numpy()
    scores={name:np.zeros(len(rows)) for name in ['pixels','frozen_encoder','fine_tuned']}
    altered={kind:np.zeros(len(rows)) for kind in ['border_removed','center_removed']}
    folds=[];fold_ids=np.zeros(len(rows),dtype=int)
    split=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=42)
    for fold,(train,val) in enumerate(split.split(pixels,labels,groups)):
        fold_ids[val]=fold
        record={'fold':fold,'train_images':len(train),'validation_images':len(val),'train_groups':len(set(groups[train])),'validation_groups':len(set(groups[val])),'models':{}}
        assert not set(groups[train])&set(groups[val])
        for name,x in [('pixels',pixels),('frozen_encoder',embeddings)]:
            scaler=StandardScaler().fit(x[train])
            classifier=LogisticRegression(C=1,class_weight='balanced',solver='lbfgs',max_iter=2000,random_state=42).fit(scaler.transform(x[train]),labels[train])
            scores[name][val]=classifier.predict_proba(scaler.transform(x[val]))[:,1]
            np.savez(output/f'{name}_fold{fold}.npz',mean=scaler.mean_,scale=scaler.scale_,coef=classifier.coef_,intercept=classifier.intercept_)
            record['models'][name]=binary_metrics(labels[val],scores[name][val])
            if name=='pixels':
                for kind in altered:
                    x_alt=np.array([ablate(source/rows[i]['path'],kind) for i in val])
                    altered[kind][val]=classifier.predict_proba(scaler.transform(x_alt))[:,1]
        torch.manual_seed(42+fold)
        model=resnet18(weights=None);model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
        model.fc=torch.nn.Linear(512,2)
        for name,parameter in model.named_parameters():parameter.requires_grad=name.startswith(('layer4.','fc.'))
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
        counts=np.bincount(labels[train],minlength=2)
        criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(len(train)/(2*counts),dtype=torch.float32))
        generator=torch.Generator().manual_seed(42+fold)
        loader=DataLoader(TensorDataset(images[train],torch.tensor(labels[train])),batch_size=16,shuffle=True,generator=generator)
        for epoch in range(2):
            model.train()
            for module in model.modules():
                if isinstance(module,torch.nn.BatchNorm2d):module.eval()
            losses=[]
            for x,y in loader:
                optimizer.zero_grad();loss=criterion(model(x),y);loss.backward();optimizer.step();losses.append(float(loss.detach()))
            print(f'fold={fold} epoch={epoch+1} loss={np.mean(losses):.4f}',flush=True)
        model.eval()
        with torch.inference_mode():
            scores['fine_tuned'][val]=torch.cat([torch.softmax(model(batch[0]),1)[:,1] for batch in DataLoader(TensorDataset(images[val]),batch_size=16)]).numpy()
        torch.save({'model_state':model.state_dict(),'class_order':['benign','malignant'],'fold':fold,'epochs':2,'research_inference_approved':False},output/f'fine_tuned_fold{fold}.pth')
        record['models']['fine_tuned']=binary_metrics(labels[val],scores['fine_tuned'][val])
        folds.append(record)
        (output/'folds_partial.json').write_text(json.dumps(folds,indent=2))
    results={}
    for name,p in scores.items():
        values=[r['models'][name]['balanced_accuracy'] for r in folds]
        candidates=np.unique(np.r_[0,.5,p,1])
        ranked=[]
        for threshold in candidates:
            calls=p>=threshold;tp=(calls&(labels==1)).sum();tn=(~calls&(labels==0)).sum()
            sensitivity=tp/(labels==1).sum();specificity=tn/(labels==0).sum()
            ranked.append(((sensitivity+specificity)/2,specificity,float(threshold),sensitivity))
        best=max(ranked)
        results[name]={'mean_fold_balanced_accuracy':float(np.mean(values)),'fold_balanced_accuracy_sd':float(np.std(values,ddof=1)),
                       'pooled_oof_metrics':binary_metrics(labels,p),'group_bootstrap':group_intervals(labels,p,groups,iterations=1000),
                       'exploratory_oof_threshold':{'threshold':best[2],'sensitivity':float(best[3]),'specificity':float(best[1]),'balanced_accuracy':float(best[0]),'optimistic_validation_reuse':True}}
    winner=max(results,key=lambda name:results[name]['mean_fold_balanced_accuracy'])
    ablations={kind:{'metrics':binary_metrics(labels,p),'mean_absolute_score_change':float(np.mean(abs(p-scores['pixels']))),'call_changes':int(((p>=.5)!=(scores['pixels']>=.5)).sum())} for kind,p in altered.items()}
    result={'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'pretrained_weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
            'folds':folds,'results':results,'development_only_selection':winner,'selection_metric':'mean fold balanced accuracy at 0.5',
            'ablations':ablations,'elapsed_seconds':time.monotonic()-started,'patient_independent':False,'tcia_scored':False,'inference_enabled':False,
            'limitations':['All development images were previously eligible for development; grouped CV is exploratory.','Duplicate groups are not patients.','OOF-selected thresholds are optimistic and have no independent test.','Border interventions can remove anatomy; no causal claim.']}
    (output/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    with (output/'oof_predictions.csv').open('w',newline='') as file:
        writer=csv.writer(file);writer.writerow(['path','fold','actual',*scores,*altered])
        for i,row in enumerate(rows):writer.writerow([row['path'],fold_ids[i],row['label'],*[float(v[i]) for v in scores.values()],*[float(v[i]) for v in altered.values()]])
    print(json.dumps({'results':results,'development_only_selection':winner,'ablations':ablations},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['source','manifest','output','weights']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();run(a.source,a.manifest,a.output,a.weights)
