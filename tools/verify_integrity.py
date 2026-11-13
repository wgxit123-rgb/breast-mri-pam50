"""Offline file/aggregate consistency check; standard library only."""
import argparse,csv,hashlib,json
from pathlib import Path

def check(root):
    root=Path(root).resolve()
    manifest=json.loads((root/'RELEASE_MANIFEST.json').read_text(encoding='utf8'))
    for rel,record in manifest['files'].items():
        path=(root/rel).resolve()
        if root not in path.parents:raise ValueError('Unsafe manifest path')
        raw=path.read_bytes()
        if len(raw)!=record['bytes'] or hashlib.sha256(raw).hexdigest()!=record['sha256']:
            raise ValueError('Integrity mismatch: '+rel)
    x=json.loads((root/'data/aggregates/analysis_results.json').read_text(encoding='utf8'))
    assert x['patients']==743 and x['original_round']['successful']==5 and x['original_round']['failed']==1
    for arm in x['arms'].values():
        cm=arm['pooled_selected_development']['confusion']
        assert [sum(row) for row in cm]==[318,108,138,179]
        assert sum(map(sum,cm))==743
        assert [sum(row[j] for row in cm) for j in range(4)]==arm['pooled_selected_development']['prediction_counts']
        assert [f['patients'] for f in arm['folds']]==[247,248,248]
    with (root/'data/tables/Table_13.csv').open(encoding='utf8',newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==9 and all(int(r['n'])==744 and float(r['q (9 tests)'])>0.05 for r in rows)
    print(json.dumps({'status':'PASS','checked_files':len(manifest['files']),'patients':743,'molecular_tests':9,'new_patient_computation':False}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();check(a.root)
