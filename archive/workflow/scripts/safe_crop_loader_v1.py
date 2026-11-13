"""Development-only complete-case window visitor; no classifier or aggregation.

Use validity_SAR with every window. A callback exception aborts the case and
cannot produce a completed-case receipt. Callback output is not a prediction.
"""
import json
import re
import nibabel as nib
import numpy as np
from nominal_phase_contract import ROOT, sha, load_contract
from safe_crop_adapter_v1 import OUT, SPATIAL_OUT, contract, extract_window, window_plan


def visit_windows(tensor, valid, affine, plan, callback):
    # This check prevents a consumer from passing only the first/top-k proposals.
    required={tuple(r['start_RAS']) for r in window_plan(valid.shape)}
    provided=[tuple(r['start_RAS']) for r in plan]
    if len(provided)!=len(required) or set(provided)!=required:
        raise ValueError('Complete unique full-FOV window plan required; subset/duplicate rejected')
    completed=[]
    for index,row in enumerate(plan):
        image,support,window_affine=extract_window(tensor,valid,affine,row['start_RAS'])
        metadata=dict(index=index,total=len(plan),start_RAS=row['start_RAS'],
                      affine_RAS=window_affine.tolist(),proposal_intersection=row['proposal_intersection'])
        callback(image,support,metadata)
        completed.append(row['start_RAS'])
    return dict(status='all_windows_visited',expected=len(plan),completed=len(completed),
                reference_masks_used=False,prediction_or_accuracy_claimed=False)


def visit_development_case(pid, callback):
    if not re.fullmatch(r'(?:ISPY2|ACRIN-6698)-[0-9]+',pid):raise ValueError('Invalid development patient ID')
    _,table=load_contract()
    if pid not in set(table.loc[table.cohort.eq('ispy'),'pid']):raise ValueError('Original development cohort only')
    candidate=contract();case_path=OUT/'safe_crop_cases'/('ispy_'+pid+'.json')
    if not case_path.exists():raise ValueError('No verified input case; geometry and extraction checks required')
    record=json.loads(case_path.read_text())
    assert record['status']=='full_fov_window_extraction_and_affine_verified'
    assert record['candidate_contract_sha256']==sha(candidate)
    spatial_path=SPATIAL_OUT/'spatial_cases'/('ispy_'+pid+'.json')
    assert record['spatial_receipt_sha256']==sha(spatial_path)
    spatial=json.loads(spatial_path.read_text())
    for f in spatial['files']:assert sha(f['path'])==f['sha256']
    tensor=np.load(spatial['files'][0]['path'],mmap_mode='r')
    valid=np.asarray(nib.load(spatial['files'][1]['path']).dataobj).astype(bool)
    receipt=visit_windows(tensor,valid,np.array(spatial['affine_RAS']),record['windows'],callback)
    receipt.update(pid=pid,candidate_input_only=True,training_or_inference_implemented=False,
                   safe_crop_case_sha256=sha(case_path),loader_sha256=sha(__file__))
    return receipt
