"""Proposal-guided complete-FOV windows, not a validated final model input.

Automatic boxes affect window ordering only: they can never remove context.
Uses existing 2-mm candidate images without additional resampling/normalization.
"""
import itertools
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from nominal_phase_contract import ROOT, OUT as PHASE_OUT, sha, write_json, load_contract
from physical_fov_candidate import OUT as SPATIAL_OUT, WINDOW, windows, contract as spatial_contract

OUT=ROOT/'reports/localization_20261013'


def native_affine_from_verified_ras(geometry, native_shape):
    if geometry['status']!='nominal_and_legacy_selected_geometry_verified':
        raise ValueError('Complete patient-specific geometry evidence required')
    ras=next(r for r in geometry['outputs'] if r['role']=='nominal_pre')
    transform=nib.orientations.ornt_transform(nib.orientations.axcodes2ornt(tuple(geometry['native_axes'])),
                                             nib.orientations.axcodes2ornt(('R','A','S')))
    ras_to_native=nib.orientations.inv_ornt_aff(transform, native_shape)
    native=np.array(ras['affine_RAS'])@np.linalg.inv(ras_to_native)
    assert ''.join(nib.aff2axcodes(native))==geometry['native_axes']
    return native,transform


def map_box(lo, hi, source_affine, target_affine, target_shape):
    """Map source voxel-cell bounds to a conservative target voxel-cell box."""
    lo=np.asarray(lo);hi=np.asarray(hi)
    assert np.all(lo>=0) and np.all(hi>lo)
    corners=np.array([(*p,1.) for p in itertools.product(*zip(lo-.5,hi-.5))]).T
    points=(np.linalg.inv(target_affine)@source_affine@corners)[:3]
    # Conservative integer enclosure; no implicit center/edge-origin mixing.
    lower=np.floor(points.min(1)+.5+1e-9).astype(int)
    upper=np.ceil(points.max(1)+.5-1e-9).astype(int)
    lower=np.clip(lower,0,target_shape);upper=np.clip(upper,0,target_shape)
    assert np.all(upper>lower), 'Mapped automatic box outside target FOV'
    return lower.tolist(),upper.tolist()


def window_plan(shape, proposal=None):
    candidates=windows(shape)
    def priority(start):
        if proposal is None:return 1
        lo,hi=map(np.asarray,proposal);s=np.asarray(start);stop=np.minimum(s+WINDOW,shape)
        return 0 if np.all(np.minimum(stop,hi)>np.maximum(s,lo)) else 1
    ordered=sorted(candidates,key=lambda start:(priority(start),*start))
    assert len(ordered)==len(candidates) and set(map(tuple,ordered))==set(map(tuple,candidates))
    return [dict(start_RAS=start,proposal_intersection=priority(start)==0) for start in ordered]


def extract_window(tensor_SAR, valid_RAS, affine_RAS, start_RAS):
    """Return 3xSxAxR data, SxAxR support and exact window RAS affine."""
    shape=np.array(valid_RAS.shape);start=np.array(start_RAS,dtype=int)
    assert tensor_SAR.shape==(3,*valid_RAS.shape[::-1])
    assert start.shape==(3,) and np.all(start>=0) and np.all(start<shape)
    stop=np.minimum(start+WINDOW,shape);extent=stop-start
    src=tuple(slice(int(a),int(b)) for a,b in zip(start[::-1],stop[::-1]))
    dst=tuple(slice(0,int(n)) for n in extent[::-1])
    image=np.zeros((3,WINDOW,WINDOW,WINDOW),dtype=np.float32)
    support=np.zeros((WINDOW,WINDOW,WINDOW),dtype=bool)
    image[(slice(None),*dst)]=tensor_SAR[(slice(None),*src)]
    support[dst]=valid_RAS.transpose(2,1,0)[src]
    assert np.isfinite(image).all() and np.all(image[:,~support]==0)
    offset=np.eye(4);offset[:3,3]=start
    return image,support,np.asarray(affine_RAS)@offset


def contract():
    load_contract();spatial_contract();OUT.mkdir(exist_ok=True)
    path=OUT/'safe_crop_candidate_contract.json'
    record=dict(status='input_layer_candidate_not_final_model',
        rule='Automatic proposal controls ordering only; always retain every full-FOV window',
        window_RAS=[96,96,96],spacing_mm=[2,2,2],stride_RAS=[48,48,48],
        no_additional_resampling=True,no_additional_normalization=True,
        axes='data: phase,S,A,R; validity: S,A,R; affine indexes: R,A,S',
        padding='zero with explicit validity',empty_or_absent_proposal='canonical full-FOV windows',
        missing_geometry='hold; never infer physical mm from placeholder native affine',
        localization_threshold_unchanged=True,reference_masks_used_for_planning=False,
        aggregation_or_classifier_implemented=False,final_model_frozen=False,
        caveats=['Retained FOV does not guarantee correct lesion identification or model attention',
                 'Existing proposal model uses legacy phases/native-index input, not a final physical-grid localizer',
                 'Whole-FOV windows preserve unilateral/bilateral domain differences',
                 'Two-mm image resampling fidelity remains a separate unresolved candidate issue'],
        phase_contract_sha256=sha(PHASE_OUT/'phase_contract.json'),
        spatial_contract_sha256=sha(SPATIAL_OUT/'spatial_candidate_contract.json'),script_sha256=sha(Path(__file__)))
    if path.exists():assert json.loads(path.read_text())==record,'Version a changed candidate'
    else:write_json(path,record)
    return path


def run():
    contract_path=contract()
    folds=pd.read_csv(ROOT/'reports/pam50_v2/fold_manifest.csv',usecols=['pid','outer']).set_index('pid')
    cases=[]
    for path in sorted((SPATIAL_OUT/'spatial_cases').glob('*.json')):
        s=json.loads(path.read_text());pid=s['pid'];cohort=s['cohort']
        assert s['status']=='physical_grid_and_full_window_coverage_checked'
        output=OUT/'safe_crop_cases'/(cohort+'_'+pid+'.json')
        if output.exists():
            previous=json.loads(output.read_text())
            assert previous['spatial_receipt_sha256']==sha(path)
            assert previous['candidate_contract_sha256']==sha(contract_path)
            cases.append(previous);continue
        for f in s['files']:assert sha(f['path'])==f['sha256']
        tensor=np.load(s['files'][0]['path'],mmap_mode='r')
        valid=np.asarray(nib.load(s['files'][1]['path']).dataobj).astype(bool)
        affine=np.array(s['affine_RAS']);shape=valid.shape;proposal=None;extra={}
        if cohort=='ispy':
            assert pid in folds.index
            gp=PHASE_OUT/'geometry_cases'/(pid+'.json');g=json.loads(gp.read_text())
            cp=ROOT/('data/pam50_v2/outer_%d/automatic_roi'%int(folds.loc[pid,'outer']))/(pid+'.json')
            crop=json.loads(cp.read_text())
            assert crop['source']=='automatic' and crop['pid']==pid and crop['outer']==int(folds.loc[pid,'outer'])
            assert g['pid']==pid and sha(gp)==s['input_evidence_sha256']
            native,tr=native_affine_from_verified_ras(g,crop['native_shape'])
            # Independently recheck actual selected precontrast data under this recovered map.
            native_phase=next(r for r in g['verified_phases'] if r['phase']==0)
            canon=next(r for r in g['outputs'] if r['role']=='nominal_pre')
            assert sha(native_phase['local_file'])==native_phase['local_sha256']
            assert sha(canon['path'])==canon['sha256']
            original=np.asarray(nib.load(native_phase['local_file']).dataobj)
            assert tuple(crop['native_shape'])==original.shape
            assert np.array_equal(nib.orientations.apply_orientation(original,tr),nib.load(canon['path']).dataobj)
            if not crop['empty_auto']:
                proposal=map_box(crop['crop_start'],crop['crop_stop'],native,affine,shape)
            extra=dict(geometry_sha256=sha(gp),automatic_crop_metadata_sha256=sha(cp),
                       native_affine_reconstructed=native.tolist(),legacy_phase_proposal_only=True)
        plan=window_plan(shape,proposal)
        coverage=np.zeros(shape,dtype=np.uint16)
        max_error=0.
        for window in plan:
            start=np.array(window['start_RAS']);stop=np.minimum(start+WINDOW,shape);extent=stop-start
            image,support,waff=extract_window(tensor,valid,affine,start)
            src=tuple(slice(int(a),int(b)) for a,b in zip(start[::-1],stop[::-1]))
            dst=tuple(slice(0,int(n)) for n in extent[::-1])
            assert np.array_equal(image[(slice(None),*dst)],tensor[(slice(None),*src)])
            assert np.array_equal(support[dst],valid.transpose(2,1,0)[src])
            for point in itertools.product(*[(0,int(n)-1) for n in extent]):
                expected=affine@np.r_[np.array(point)+start,1]
                observed=waff@np.r_[point,1]
                max_error=max(max_error,float(np.max(np.abs(expected-observed))))
            coverage[tuple(slice(int(a),int(b)) for a,b in zip(start,stop))]+=1
        assert coverage.min()>=1 and max_error<1e-7
        record=dict(status='full_fov_window_extraction_and_affine_verified',cohort=cohort,pid=pid,
            spatial_receipt_sha256=sha(path),candidate_contract_sha256=sha(contract_path),
            proposal_RAS=proposal,windows=plan,n_windows=len(plan),covered_voxels=int((coverage>0).sum()),
            valid_voxels=int(valid.sum()),uncovered_valid_voxels=int((valid&(coverage==0)).sum()),
            max_window_corner_error_mm=max_error,exact_window_data_and_validity=True,
            no_window_dropped=True,reference_mask_used=False,external_endpoints_used=False,
            classifier_run=False,training_run=False,model_input_released=False,**extra)
        write_json(output,record);cases.append(record)
        print(cohort,pid,len(plan),'windows verified',flush=True)
    summary=dict(status='candidate_input_software_checks_passed',n=len(cases),
        by_cohort={c:sum(r['cohort']==c for r in cases) for c in ['ispy','tcga']},
        total_windows=sum(r['n_windows'] for r in cases),uncovered_valid_voxels=sum(r['uncovered_valid_voxels'] for r in cases),
        max_window_corner_error_mm=max(r['max_window_corner_error_mm'] for r in cases),
        reference_mask_used=False,classifier_run=False,final_model_frozen=False,
        script_sha256=sha(Path(__file__)),candidate_contract_sha256=sha(contract_path))
    write_json(OUT/'safe_crop_summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':run()
