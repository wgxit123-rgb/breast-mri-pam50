"""Outcome-blind spatial adapter candidate; never a model release.

Retain whole acquired FOV on an enclosing RAS 2-mm grid, with validity map.
Enumerate 192-mm overlapping windows without choosing a breast from a mask.
Interpolation is not registration or proof of anatomical domain equivalence.
"""
import argparse
import itertools
import json
import time
from pathlib import Path
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from nominal_phase_contract import ROOT, OUT as PHASE_OUT, sha, write_json, load_contract

OUT = ROOT / 'reports/comparability_20261012'
DATA = ROOT / 'data/comparability_20261012'
ROLES = ['nominal_pre', 'first_post', 'last_post']
SPACING = 2.0
WINDOW = 96
STRIDE = 48


def bounds(shape, affine):
    corners = np.array([(*p, 1) for p in itertools.product(*[(-.5, n-.5) for n in shape])]).T
    world = (affine @ corners)[:3]
    return world.min(axis=1), world.max(axis=1)


def grid(shape, affine):
    lo, hi = bounds(shape, affine)
    lower = np.floor(lo / SPACING) * SPACING
    size = np.ceil((hi - lower) / SPACING).astype(int)
    assert np.all(size > 0) and np.all(size <= 512)
    target = np.diag([SPACING, SPACING, SPACING, 1.])
    target[:3, 3] = lower + SPACING / 2
    low2, high2 = bounds(size, target)
    assert np.all(low2 <= lo + 1e-7) and np.all(high2 >= hi - 1e-7)
    return tuple(int(n) for n in size), target


def starts(n):
    if n <= WINDOW:
        return [0]
    return sorted(set(list(range(0, n-WINDOW+1, STRIDE)) + [n-WINDOW]))


def windows(shape):
    axes = [starts(n) for n in shape]
    for n, values in zip(shape, axes):
        covered = np.zeros(n, bool)
        for start in values:
            covered[start:min(start+WINDOW,n)] = True
        assert covered.all(), 'Window coverage gap'
    return [list(x) for x in itertools.product(*axes)]


def resample(images):
    reference = images[0]
    target = grid(reference.shape, reference.affine)
    corners = np.array([(*v,1) for v in itertools.product(*[(0,n-1) for n in reference.shape])]).T
    arrays = []
    for im in images:
        assert im.shape == reference.shape
        assert np.max(np.abs((im.affine-reference.affine) @ corners)) < .02
        original = np.asarray(im.dataobj, dtype=np.float32)
        assert np.isfinite(original).all()
        float_image = nib.Nifti1Image(original, im.affine)
        arrays.append(np.asarray(resample_from_to(float_image, target, order=1,
                                                 mode='constant', cval=0).dataobj, dtype=np.float32))
    support = nib.Nifti1Image(np.ones(reference.shape, dtype=np.uint8), reference.affine)
    valid = np.asarray(resample_from_to(support, target, order=0, mode='constant', cval=0).dataobj).astype(bool)
    assert valid.any()
    positive = arrays[0][valid & (arrays[0] > 0)]
    assert positive.size
    scale = max(float(np.percentile(positive, 99.5)), 1.)
    stack = np.stack(arrays)
    assert np.isfinite(stack).all()
    scaled = np.clip(stack / scale, 0, 5)
    scaled[:, ~valid] = 0
    return scaled.astype(np.float32), valid, target[1], scale


def contract():
    path = OUT / 'spatial_candidate_contract.json'
    obj = dict(status='implementation_candidate_not_final_preprocessing',
               purpose='Physical scale and window coverage harmonization without lesion or endpoint access',
               spacing_mm=[2.,2.,2.], grid_axes='RAS', tensor_axes='phase,S,A,R',
               image_interpolation='linear, zero outside', support_interpolation='nearest, zero outside',
               support_semantics='centers supported by source center interpolation domain, not a breast or lesion mask',
               window_voxels_RAS=[96,96,96], stride_voxels_RAS=[48,48,48],
               window_mm=[192,192,192], retain_acquired_fov=True,
               incomplete_windows='zero padding outside tensor; validity must accompany every window',
               intensity='per-patient precontrast positive supported voxel p99.5, min denominator 1, clip 0..5',
               reference_mask_accessed=False, external_endpoint_accessed=False,
               clinical_input_comparability_established=False, final_model_frozen=False,
               limitations=['Not motion registration', 'Not lossless intensity resampling',
                            'No dedicated anti-alias filter: resolution sensitivity must be checked in development',
                            'Whole-FOV tiling retains unilateral/bilateral background-domain differences',
                            'Requires a model trained and checked with this preprocessing; existing native-index weights not released',
                            'No tile prediction aggregation or lesion-selection rule implemented'],
               script_sha256=sha(Path(__file__)), phase_contract_sha256=sha(PHASE_OUT/'phase_contract.json'))
    OUT.mkdir(parents=True,exist_ok=True)
    if path.exists():
        assert json.loads(path.read_text()) == obj, 'Candidate changed; use a new version'
    else:
        write_json(path,obj)
    return path


def inputs():
    _, selection = load_contract()
    allowed = set(selection.loc[selection.cohort.eq('ispy'),'pid'])
    rows = []
    for path in sorted((PHASE_OUT/'geometry_cases').glob('*.json')):
        r = json.loads(path.read_text())
        if r['status'] != 'nominal_and_legacy_selected_geometry_verified':
            continue
        assert r['pid'] in allowed
        mapping = {x['role']:x for x in r['outputs']}
        rows.append(('ispy', r['pid'], [mapping[role] for role in ROLES], path))
    # Do not read reference masks, subtype labels or any external model predictions.
    path = PHASE_OUT/'tcga_selected_inputs.json'
    for r in json.loads(path.read_text()):
        mapping = {x['role']:x for x in r['selected']}
        rows.append(('tcga', r['pid'], [mapping[role] for role in ROLES], path))
    return rows


def case(cohort,pid,files,evidence):
    dest = DATA/cohort/pid
    receipt = OUT/'spatial_cases'/(cohort+'_'+pid+'.json')
    assert not dest.exists() and not receipt.exists(), 'Refuse overwrite/retry'
    rec = dict(cohort=cohort,pid=pid,status='hold',input_evidence_sha256=sha(evidence),
               reference_mask_accessed=False,endpoint_accessed=False,model_run=False,final_model_frozen=False)
    start = time.monotonic()
    try:
        for f in files:
            assert sha(f['path']) == f['sha256']
        images = [nib.load(f['path']) for f in files]
        assert all(nib.aff2axcodes(im.affine)==('R','A','S') for im in images)
        arrays, valid, affine, scale = resample(images)
        size = valid.shape
        positions = windows(size)
        dest.mkdir(parents=True)
        tensor = dest/'phases_SAR.npy'
        np.save(tensor, arrays.transpose(0,3,2,1))
        support = dest/'valid_RAS.nii.gz'
        nib.save(nib.Nifti1Image(valid.astype(np.uint8),affine), support)
        reread = np.load(tensor, mmap_mode='r')
        assert np.array_equal(reread, arrays.transpose(0,3,2,1))
        oldlo, oldhi = bounds(images[0].shape, images[0].affine)
        newlo, newhi = bounds(size, affine)
        rec.update(status='physical_grid_and_full_window_coverage_checked',
                   files=[dict(path=str(tensor),sha256=sha(tensor)), dict(path=str(support),sha256=sha(support))],
                   source_image_sha256=[f['sha256'] for f in files],
                   shape_RAS=list(size),tensor_shape=list(reread.shape),affine_RAS=affine.tolist(),
                   source_extent_mm=(oldhi-oldlo).tolist(),target_extent_mm=(newhi-newlo).tolist(),
                   bounding_box_enclosed=True,valid_fraction=float(valid.mean()),intensity_scale=scale,
                   window_starts_RAS=positions,windows=len(positions),coverage_gaps=0,
                   anatomy_or_lesion_alignment_claimed=False,phase_motion_corrected=False)
    except Exception as exc:
        rec['error'] = repr(exc)
    rec['seconds']=time.monotonic()-start
    write_json(receipt,rec)
    print(cohort,pid,rec['status'],rec.get('error',''),flush=True)
    return rec


def tests():
    base=np.arange(7*8*9,dtype=np.float32).reshape(7,8,9)+1
    affine=np.diag([2.,2.,2.,1.]); affine[:3,3]=[1.,-15.,3.]
    expected=None; count=0
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([-1,1],repeat=3):
            transform=np.array(list(zip(perm,signs)))
            a=nib.orientations.apply_orientation(base,transform)
            aff=affine@nib.orientations.inv_ornt_aff(transform,base.shape)
            result=resample([nib.Nifti1Image(a*k,aff) for k in [1,2,3]])
            if expected is None: expected=result
            assert np.array_equal(result[0],expected[0]) and np.array_equal(result[1],expected[1])
            assert np.array_equal(result[2],expected[2]); count+=1
    for n in [1,64,95,96,97,144,193,400]:
        windows((n,7,8))
    theta=.21; rotation=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
    oblique=affine.copy(); oblique[:3,:3]=rotation@affine[:3,:3]
    size, target=grid(base.shape,oblique)
    lo,hi=bounds(base.shape,oblique); lo2,hi2=bounds(size,target)
    assert np.all(lo2<=lo+1e-7) and np.all(hi2>=hi-1e-7)
    failed=False
    shifted=affine.copy(); shifted[0,3]+=1
    try: resample([nib.Nifti1Image(base,affine),nib.Nifti1Image(base,shifted)])
    except AssertionError: failed=True
    assert failed
    return dict(permutation_flip_cases_passed=count,tile_boundary_cases_passed=8,
                oblique_enclosure_passed=True,phase_grid_mismatch_rejected=True,clinical_validation=False)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['test','run']); args=parser.parse_args()
    contract_path=contract()
    test_result=tests(); write_json(OUT/'spatial_adapter_tests.json',test_result); print(test_result,flush=True)
    if args.mode=='test': return
    records=[case(*row) for row in inputs()]
    from collections import Counter
    write_json(OUT/'spatial_summary.json',dict(candidate_contract_sha256=sha(contract_path),n=len(records),
               statuses=dict(Counter(r['status'] for r in records)),
               by_cohort={c:dict(Counter(r['status'] for r in records if r['cohort']==c)) for c in ['ispy','tcga']},
               reference_mask_accessed=False,external_endpoint_accessed=False,model_input_released=False,
               anatomical_fov_equivalence_established=False,final_model_frozen=False))


if __name__=='__main__': main()
