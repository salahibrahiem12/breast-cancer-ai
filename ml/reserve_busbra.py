"""Inventory official BUS-BRA for reservation; never load a predictive model."""
import csv
from collections import Counter,defaultdict
import hashlib
from io import StringIO
import json
from pathlib import Path
from zipfile import ZipFile

from ml.inspect_tcia_breast import safe_members,fingerprint


def reserve(root):
    archive=root/'BUSBRA.zip'
    if hashlib.md5(archive.read_bytes()).hexdigest()!='1f8b2be6476d58fc97bfb5e5a1ea9bab':
        raise ValueError('Official archive checksum mismatch.')
    existing=[]
    for row in csv.DictReader(open('data/research_data/tcia_breast_lesions_usg_2026-11-03/inventory/cohort_inventory.csv')):
        existing.append(row)
    for path in Path('data/raw/mendeley_v1/official-v1/originals').rglob('*.bmp'):
        existing.append(fingerprint(path.read_bytes()))
    rows=[]
    with ZipFile(archive) as z:
        safe_members(z)
        metadata=list(csv.DictReader(StringIO(z.read('BUSBRA/bus_data.csv').decode('utf-8-sig'))))
        for row in metadata:
            name='BUSBRA/Images/'+row['ID']+'.png'
            rows.append({'archive_path':name,'source_case':row['Case'],'label':row['Pathology'],'device':row['Device'],**fingerprint(z.read(name))})
        folds=list(csv.DictReader(StringIO(z.read('BUSBRA/5-fold-cv.csv').decode('utf-8-sig'))))
    cases=defaultdict(set);case_labels=defaultdict(set)
    lookup={r['ID']:r['kFold'] for r in folds}
    for raw,row in zip(metadata,rows):
        cases[row['source_case']].add(lookup[raw['ID']]);case_labels[row['source_case']].add(row['label'])
    overlaps={}
    for key in ['sha256','pixel_sha256']:
        known={r[key] for r in existing};overlaps[key+'_matches']=sum(r[key] in known for r in rows)
    for key in ['dhash','phash']:
        overlaps[key+'_near_pairs_distance_3']=sum((int(r[key])^int(other[key])).bit_count()<=3 for r in rows for other in existing)
    internal={}
    for key in ['dhash','phash']:
        same=cross=0
        for index,row in enumerate(rows):
            for other in rows[:index]:
                if (int(row[key])^int(other[key])).bit_count()<=3:
                    if row['source_case']==other['source_case']:same+=1
                    else:cross+=1
        internal[key]={'within_case_candidate_pairs':same,'cross_case_candidate_pairs':cross}
    result={'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'official_md5_verified':True,
            'source':'https://zenodo.org/records/8231412','license':'CC BY 4.0; limited local research reservation; cite Gomez-Flores et al. DOI 10.1002/mp.16812',
            'images':len(rows),'source_cases':len(cases),'class_counts':dict(Counter(r['label'] for r in rows)),
            'devices':len(set(r['device'] for r in rows)),'source_cases_crossing_published_test_folds':sum(len(v)>1 for v in cases.values()),
            'cases_with_conflicting_binary_labels':sum(len(v)>1 for v in case_labels.values()),
            'comparison_image_count':len(existing),'overlap':overlaps,
            'within_source_repeated_pixel_occurrences':len(rows)-len(set(r['pixel_sha256'] for r in rows)),
            'within_source_near_duplicate_candidates_distance_3':internal,
            'status':'quarantined reserved external source; no predictions permitted until grouping/duplicate review and final protocol',
            'patient_independence_verified':False,'patient_identifier_limitation':'Case is an explicit source field; its equivalence to stable patient/exam identity needs an authoritative data dictionary. Matching counts alone are insufficient.',
            'model_predictions_generated':False,'inference_enabled':False}
    with (root/'inventory.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result['inventory_sha256']=hashlib.sha256((root/'inventory.csv').read_bytes()).hexdigest()
    result['authoritative_metadata_sha256']=hashlib.sha256((root/'zenodo_metadata.json').read_bytes()).hexdigest()
    result['reserved_for']='one future locked external evaluation; no predictions until grouping and protocol review'
    (root/'reservation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':reserve(Path('data/research_data/busbra_reserved_v1'))
