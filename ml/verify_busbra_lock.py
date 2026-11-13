"""Verify locked artifacts and reproduce statistics without new image inference."""
import argparse
import csv
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
from ml.audit_tcia_evaluation import binary_metrics,group_intervals


def verify(intervals=False):
    root=Path('reports/busbra_locked_v1')
    read=lambda p:json.loads(p.read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    lock=read(root/'lock.json');receipt=read(root/'protocol_receipt.json')
    start=read(root/'INFERENCE_STARTED.json');done=read(root/'INFERENCE_COMPLETED.json')
    result=read(root/'evaluation.json')
    assert sha(root/'lock.json')==receipt['lock_sha256']==start['lock_sha256']==result['lock_sha256']
    assert sha(Path('docs/BUSBRA_LOCKED_PROTOCOL_V1.md'))==receipt['protocol_sha256']
    assert datetime.fromisoformat(lock['created_utc'])<=datetime.fromisoformat(receipt['created_before_inference_utc'])<datetime.fromisoformat(start['utc'])<datetime.fromisoformat(done['utc'])
    assert sha(root/'evaluation.json')==done['evaluation_sha256']
    for path,digest in receipt['dependencies'].items():assert sha(Path(path))==digest
    for key,path in {'weights':Path('reports/bounded_development_v1/preserved/pretrained_resnet18.pth'),'linear':root/'final_linear.npz','development_manifest':Path('data/prepared/mendeley_dev_v2/split_manifest.csv'),'reserved_archive':Path('data/research_data/busbra_reserved_v1/BUSBRA.zip'),'eligible_inventory':root/'eligible_inventory.csv','precheck':root/'precheck.json','runner':Path('ml/busbra_locked.py')}.items():assert sha(path)==lock['hashes'][key]
    for name,digest in result['prediction_hashes'].items():assert sha(root/name)==digest
    images=list(csv.DictReader((root/'image_predictions.csv').open()))
    cases=list(csv.DictReader((root/'case_predictions.csv').open()))
    grouped=defaultdict(list)
    for row in images:grouped[row['source_case']].append(float(row['malignant_probability']))
    for row in cases:assert abs(float(row['malignant_probability'])-np.mean(grouped[row['source_case']]))<1e-12
    labels=np.array([row['actual']=='malignant' for row in cases],dtype=int)
    scores=np.array([float(row['malignant_probability']) for row in cases])
    metrics=binary_metrics(labels,scores)
    for key,value in metrics.items():
        if value is None:assert result['primary_case_metrics'][key] is None
        else:assert abs(value-result['primary_case_metrics'][key])<1e-10
    if intervals:
        ci=group_intervals(labels,scores,[row['source_case'] for row in cases],iterations=2000)
        for key,value in ci.items():
            for field in value:
                assert value[field]==result['case_bootstrap_intervals'][key][field]
    for item in read(Path('reports/bounded_development_v1/preservation.json')):
        assert sha(Path(item['original']))==item['sha256']==sha(Path(item['copy']))
    output={'hashes_and_timestamp_sequence_verified':True,'image_predictions':len(images),'case_predictions':len(cases),'case_aggregation_and_metrics_reproduced':True,'case_intervals_reproduced':intervals,'new_image_inference':False}
    (root/'verification.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--intervals',action='store_true');args=parser.parse_args();verify(args.intervals)
