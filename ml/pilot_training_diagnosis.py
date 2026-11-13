"""Evaluate preserved pilot checkpoints; never retrain or alter pilot outputs."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, log_loss
from scipy.special import expit
from torchvision.models import resnet18
from ml.cross_source import tensors, net, extract, WEIGHTS
from ml.design_audit import OUT, PILOT, save


def summarize(logits, y, weights):
    p = torch.softmax(torch.tensor(logits), dim=1).numpy()[:,1]
    loss = torch.nn.functional.cross_entropy(torch.tensor(logits), torch.tensor(y), weight=weights)
    return {'weighted_cross_entropy':float(loss), 'unweighted_log_loss':float(log_loss(y,p)),
            'raw_argmax_accuracy':float(((p>=.5)==y).mean()),
            'raw_argmax_class_counts':{'benign':int((p<.5).sum()),'malignant':int((p>=.5).sum())},
            'raw_roc_auc':float(roc_auc_score(y,p))}


def run():
    torch.set_num_threads(2)
    rows=json.loads((PILOT/'selection.json').read_text())['rows']
    results=json.loads((PILOT/'result.json').read_text())['results']
    start=time.perf_counter(); images=tensors(rows); decode=time.perf_counter()-start
    encoder=net(); start=time.perf_counter(); extract(encoder,images[:16]); warm=time.perf_counter()-start
    start=time.perf_counter(); extract(encoder,images[:16]); seconds=time.perf_counter()-start
    save('compute.json',{'cuda_available':torch.cuda.is_available(),'cpu_threads':2,
         'selected_image_decode_seconds':decode,'encoder_first_batch_seconds':warm,
         'encoder_warm_16_images_seconds':seconds,
         'estimated_uncached_2012_image_encoder_seconds':seconds/16*2012,
         'note':'Measured forward-pass extrapolation excludes decode, IO, solver and variation; no GPU assumed.'})
    y=np.array([r['label']=='malignant' for r in rows],dtype=np.int64)
    state=torch.load(WEIGHTS,map_location='cpu',weights_only=True); answer={}
    for fold,(source,arms) in enumerate(results.items()):
        r=arms['fine_tuned']; tr=np.array([i for i,p in enumerate(rows) if p['group'] in r['inner_train_groups']])
        va=np.array([i for i,p in enumerate(rows) if p['group'] in r['inner_validation_groups']])
        te=np.array([i for i,p in enumerate(rows) if p['source']==source])
        counts=np.bincount(y[tr],minlength=2);weights=torch.tensor(len(tr)/(2*counts),dtype=torch.float32)
        torch.manual_seed(42+fold); model=resnet18(weights=None);model.load_state_dict(state);model.fc=torch.nn.Linear(512,2);model.eval()
        initial={};final={}
        for phase,ix in [('train',tr),('validation',va)]:
            with torch.inference_mode(): logits=np.concatenate([model(b).numpy() for b in images[ix].split(16)])
            initial[phase]=summarize(logits,y[ix],weights)
        model.load_state_dict(torch.load(PILOT/f'{source}_fine_tuned.pth',map_location='cpu',weights_only=True));model.eval()
        trainable=sum(p.numel() for name,p in model.named_parameters() if name.startswith(('layer4.','fc.')))
        for phase,ix in [('train',tr),('validation',va),('held_out',te)]:
            with torch.inference_mode(): logits=np.concatenate([model(b).numpy() for b in images[ix].split(16)])
            final[phase]=summarize(logits,y[ix],weights)
        answer[source]={'initial':initial,'final':final,'epochs':2,'trainable_parameters':trainable,
             'epoch_learning_curves_available':False,'optimizer_state_available':False,
             'checkpoint_selection':'fixed final epoch 2; no validation early stopping or best-checkpoint selection',
             'convergence_established':False,
             'note':'Initial and final endpoint evaluations are not recovered per-epoch learning curves.'}
        save('training_diagnosis.json',answer)
        print(source,'saved initial/final train/validation diagnostics',flush=True)


if __name__=='__main__':run()
