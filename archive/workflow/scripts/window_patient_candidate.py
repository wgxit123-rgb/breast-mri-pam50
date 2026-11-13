"""Development-only all-window aggregation interface; no released classifier.

This version permits only labelled software probes, not trained-model exports.
All windows are visited. Every nonempty acquired-support window has equal weight;
empty padding-only windows get zero weight but cannot disappear from the receipt.
"""
import math
import numpy as np
from safe_crop_loader_v1 import visit_development_case


def aggregate(rows, expected_starts, classes=4):
    expected=[tuple(s) for s in expected_starts]
    starts=[tuple(r['start_RAS']) for r in rows]
    if not expected or len(set(expected))!=len(expected): raise ValueError('Invalid expected window set')
    if len(starts)!=len(expected) or len(set(starts))!=len(starts) or set(starts)!=set(expected):
        raise ValueError('Missing, duplicated or foreign window')
    ordered=sorted(rows,key=lambda r:tuple(r['start_RAS']))
    accepted=[]
    for row in ordered:
        p=np.asarray(row['probabilities'],dtype=np.float64)
        if p.shape!=(classes,) or not np.isfinite(p).all() or (p<0).any() or (p>1).any():
            raise ValueError('Invalid probability vector')
        if abs(float(p.sum())-1)>1e-6: raise ValueError('Probability vector must sum to one')
        n=row['valid_voxels']
        if not isinstance(n,(int,np.integer)) or isinstance(n,bool) or n<0: raise ValueError('Invalid support count')
        if n>0: accepted.append(p)
    if not accepted: raise ValueError('No acquired support; no patient prediction')
    result=np.array([math.fsum(float(p[k]) for p in accepted)/len(accepted) for k in range(classes)])
    return dict(probabilities=result.tolist(),windows_visited=len(rows),windows_with_support=len(accepted),
                zero_support_windows=len(rows)-len(accepted),
                rule='equal arithmetic mean of nonempty-window probabilities; all windows visited',
                order_invariant=True,final_model_input=False)


def probe_development_patient(pid,predictor,probe_kind):
    if probe_kind not in ('synthetic_software_probe','untrained_network_probe'):
        raise ValueError('Trained-model prediction requires a separate validated release contract')
    rows=[]
    def callback(image,valid,metadata):
        probabilities=predictor(image,valid,metadata)
        rows.append(dict(start_RAS=metadata['start_RAS'],valid_voxels=int(valid.sum()),
                         probabilities=np.asarray(probabilities).tolist()))
    receipt=visit_development_case(pid,callback)
    if receipt['completed']!=receipt['expected'] or len(rows)!=receipt['expected']:
        raise RuntimeError('Incomplete patient; no output')
    result=aggregate(rows,[r['start_RAS'] for r in rows])
    result.update(pid=pid,probe_kind=probe_kind,loader_receipt=receipt,window_probe_outputs=rows,
                  clinical_prediction=False,performance_evaluation=False,trained_checkpoint_used=False)
    return result
