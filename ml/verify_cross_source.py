"""Reproduce grouped metrics and verify source isolation without model fitting."""
import argparse,csv,json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
from ml.cross_source import metric,aggregate,intervals,payload


def verify(repeat_intervals=False):
    root=Path('reports/cross_source_v1')
    selection=json.loads((root/'selection.json').read_text());rows=selection['rows']
    result=json.loads((root/'result.json').read_text());pred=list(csv.DictReader((root/'predictions.csv').open()))
    assert len(pred)==2*len(rows) and len({(p['row'],p['arm']) for p in pred})==len(pred)
    for row in rows:assert hashlib.sha256(payload(row)).hexdigest()==row['sha256']
    group_source={r['group']:r['source'] for r in rows}
    for source,arms in result['results'].items():
        for arm,r in arms.items():
            tr,va,te=[set(r[k]) for k in ['inner_train_groups','inner_validation_groups','held_out_groups']]
            assert not tr&va and not (tr|va)&te
            assert all(group_source[g]!=source for g in tr|va)
            assert all(group_source[g]==source for g in te)
            ps=[p for p in pred if p['source']==source and p['arm']==arm]
            for p in ps:
                row=rows[int(p['row'])]
                assert row['source']==source and row['group']==p['group']
                assert int(p['actual'])==int(row['label']=='malignant')
                assert float(p['threshold'])==r['threshold']
            # sklearn 1.9 preserves float32 for these logits/probabilities.
            # Restore it before group means; CSV parsing defaults to float64.
            y=np.array([int(p['actual']) for p in ps]);prob=np.array([float(p['probability']) for p in ps],dtype=np.float32);groups=np.array([p['group'] for p in ps])
            gy,gp,gg=aggregate(y,prob,groups)
            for observed,expected in [(metric(gy,gp,r['threshold']),r['group_metrics']),(metric(y,prob,r['threshold']),r['image_metrics'])]:
                for k,v in observed.items():
                    if v is None:assert expected[k] is None
                    else:assert abs(v-expected[k])<1e-10
            if repeat_intervals:
                ci=intervals(gy,gp,gg,r['threshold'])
                assert ci==r['group_intervals']
    for record in json.loads((root/'preservation.json').read_text()):
        assert hashlib.sha256(Path(record['original']).read_bytes()).hexdigest()==record['sha256']==hashlib.sha256(Path(record['copy']).read_bytes()).hexdigest()
    report={'all_original_locked_artifacts_unchanged':True,'selected_image_hashes_verified':len(rows),'held_out_predictions':len(pred),'train_validation_source_isolation_verified':True,'image_and_group_metrics_reproduced':True,'group_intervals_reproduced':repeat_intervals,'new_model_fit':False}
    (root/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--intervals',action='store_true');a=p.parse_args();verify(a.intervals)
