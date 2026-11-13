"""Stage-gated development runner. Draft contracts fail before targets/GPU I/O.

Not an independent-assessment entry point. A complete adopted input manifest
and separately validated stage evidence are mandatory;synthetic tests cannot
manufacture those approvals. Existing protocols/code are never rewritten here.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Dict, List

import numpy as np
import torch
from physical_training_bridge_candidate_v1 import PatientWindows, patient_forward, optimize_patient_group, fit_class_weights

CLASSES = ['Basal', 'Her2', 'LumA', 'LumB']
FIXED = dict(seed=20260923, patients=748, folds=3, max_epochs=80, learning_rate=.0003,
    weight_decay=.0001, dropout=.2, patient_microbatch=4, accumulation=4,
    window_microbatch=4, precision='BF16', optimizer='AdamW', warmup_epochs=5,
    patience=15, count_stale_after_epoch=10, clip_norm=5.0,
    selection='macro_AUC_rounded4_then_macro_F1_then_earliest')


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def contained(root: Path, name: str) -> Path:
    path = Path(name)
    if not name or '..' in path.parts:
        raise ValueError('Contained path required')
    resolved = (path if path.is_absolute() else root/path).resolve()
    resolved.relative_to(root.resolve())
    return resolved


def bound(root: Path, item: dict) -> Path:
    path = contained(root, item['path'])
    if sha(path) != item['sha256']:
        raise ValueError('Bound file changed:' + str(path))
    return path


def preflight(root: Path, contract_path: str, expected_sha256: str) -> dict:
    path = contained(root, contract_path)
    if sha(path) != expected_sha256:
        raise ValueError('Candidate contract SHA mismatch')
    c = json.loads(path.read_text(encoding='utf-8'))
    if c.get('schema') != 'physical_development_candidate_v1' or c.get('scope') != '748_development_only':
        raise ValueError('Unrecognized contract scope')
    if json.dumps(c.get('fixed'), sort_keys=True) != json.dumps(FIXED, sort_keys=True):
        raise ValueError('Fixed training settings changed')
    gates = json.loads(contained(root, c['release_gate_path']).read_text(encoding='utf-8'))
    blocking = []
    if c.get('status') != 'ADOPTED_FOR_DEVELOPMENT_COMPARISON':
        blocking.append('execution_contract_not_adopted')
    required = ['all_development_selected_phase_geometry_verified',
                'shared_anatomical_fov_and_automatic_localization_validated',
                'localizer_probability_reproducibility_resolved']
    blocking += [key for key in required if gates['required_gates'].get(key) is not True]
    if not c.get('input_manifest'):
        blocking.append('complete_input_manifest_not_bound')
    if not c.get('independent_execution_review'):
        blocking.append('candidate_execution_review_missing')
    # Physical-development performance/final freeze/endpoint assessment are
    # downstream. Do not require their gate values before this experiment.
    if blocking:
        return dict(status='BLOCKED', blockers=blocking, targets_opened=False,
                    patient_inputs_opened=False, GPU_allocated=False,
                    final_freeze_required_at_this_stage=False)
    for rel, digest in c['source_sha256'].items():
        if sha(contained(root, rel)) != digest:
            raise ValueError('Candidate code changed:' + rel)
    review = json.loads(bound(root, c['independent_execution_review']).read_text(encoding='utf-8'))
    basis = {k:v for k,v in c.items() if k not in ['status', 'independent_execution_review', 'review_basis_sha256']}
    basis_sha = hashlib.sha256(json.dumps(basis, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
    if c['review_basis_sha256'] != basis_sha or review.get('execution_ready') is not True or review.get('candidate_contract_basis_sha256') != basis_sha:
        raise ValueError('No independent candidate execution approval')
    inputs = json.loads(bound(root, c['input_manifest']).read_text(encoding='utf-8'))
    if inputs.get('scope') != '748_development_only' or inputs.get('patients') != 748:
        raise ValueError('Input manifest eligibility changed')
    return dict(status='READY_FOR_DEVELOPMENT_ONLY', contract=c, input_manifest=inputs,
                targets_opened=False, patient_inputs_opened=False, GPU_allocated=False,
                final_freeze_required_at_this_stage=False)


def load_patient(root: Path, row: dict, arm: str) -> PatientWindows:
    """Load only hash-bound imagery,never a reference lesion or molecular field."""
    if arm == 'native_nominal_control':
        image = np.load(bound(root, row['image']), mmap_mode='r')
        metadata = json.loads(bound(root, row['native_cache_metadata']).read_text(encoding='utf-8'))
        if metadata['pid'] != row['pid'] or metadata['sha256'] != row['image']['sha256'] or metadata['outer'] != row['outer']:
            raise ValueError('Native nominal patient/cache identity mismatch')
        if image.shape != (3, 64, 96, 96) or image.dtype not in (np.float16, np.float32):
            raise ValueError('Unexpected fixed native nominal control shape')
        def get_native(key):
            if key != (0, 0, 0):
                raise ValueError('Foreign native coordinate')
            return torch.from_numpy(np.array(image, dtype=np.float32, copy=True)), torch.ones(image.shape[1:], dtype=torch.bool)
        return PatientWindows(row['pid'], [(0, 0, 0)], get_native)
    if arm != 'physical_full_fov':
        raise ValueError('Unrecognized arm')
    import nibabel as nib
    from safe_crop_adapter_v1 import extract_window, window_plan
    spatial_path, crop_path = bound(root, row['spatial_receipt']), bound(root, row['safe_crop_receipt'])
    s, r = [json.loads(p.read_text(encoding='utf-8')) for p in [spatial_path, crop_path]]
    if s['pid'] != row['pid'] or r['pid'] != row['pid'] or s['cohort'] != 'ispy' or r['cohort'] != 'ispy':
        raise ValueError('Foreign physical patient')
    if s['status'] != 'physical_grid_and_full_window_coverage_checked' or r['status'] != 'full_fov_window_extraction_and_affine_verified':
        raise ValueError('Unverified physical input')
    if r['spatial_receipt_sha256'] != sha(spatial_path) or r['uncovered_valid_voxels'] != 0 or not r['no_window_dropped']:
        raise ValueError('Physical/crop receipt mismatch')
    files = [bound(root, f) for f in s['files']]
    tensor = np.load(files[0], mmap_mode='r')
    valid = np.asarray(nib.load(str(files[1])).dataobj).astype(bool)
    keys = [tuple(w['start_RAS']) for w in r['windows']]
    expected = [tuple(w['start_RAS']) for w in window_plan(valid.shape)]
    if len(keys) != len(expected) or len(set(keys)) != len(keys) or set(keys) != set(expected):
        raise ValueError('Missing or duplicate full-FOV windows')
    def get_physical(key):
        image, support, _ = extract_window(tensor, valid, np.asarray(s['affine_RAS']), key)
        return torch.from_numpy(image), torch.from_numpy(support)
    return PatientWindows(row['pid'], keys, get_physical)


def run(root: Path, contract_path: str, expected_sha256: str, arm: str, fold: int, gpu: int) -> None:
    stage = preflight(root, contract_path, expected_sha256)
    if stage['status'] != 'READY_FOR_DEVELOPMENT_ONLY':
        raise RuntimeError('Patient training blocked:' + ','.join(stage['blockers']))
    if arm not in ['native_nominal_control', 'physical_full_fov'] or fold not in range(3):
        raise ValueError('Fixed paired arms/three folds only')
    c, inputs = stage['contract'], stage['input_manifest']
    from development_access_v1_20261020 import load_targets
    targets = load_targets(root, c['targets']['path'], c['targets']['sha256']).labels
    if len(targets) != 748 or len({r.pid for r in targets}) != 748:
        raise ValueError('Development membership changed')
    rows = inputs['arms'][arm][str(fold)]
    lookup = {r['pid']: r for r in rows}
    if len(rows) != 748 or len(lookup) != 748 or set(lookup) != {r.pid for r in targets}:
        raise ValueError('Full fixed development input membership required')
    if any(r.get('outer') != fold for r in rows):
        raise ValueError('Input manifest fold ownership mismatch')
    # Never auto-launch an occupied GPU or re-use a run directory.
    import subprocess
    observed = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
    memory = {int(line.split(',')[0]): int(line.split(',')[1]) for line in observed.strip().splitlines()}
    if memory.get(gpu, 999999) >= 500:
        raise RuntimeError('Requested GPU not free;no process interruption or automatic reassignment')
    from pam50_v2_core import network, metrics
    from pam50_v2_train import seed_everything
    seed_everything(gpu, FIXED['seed'])
    device = torch.device('cuda', gpu)
    net = network(3, dropout=.2).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0003, weight_decay=.0001)
    fit, selection = [r for r in targets if r.outer != fold], [r for r in targets if r.outer == fold]
    yfit = [CLASSES.index(r.label) for r in fit]
    weights = fit_class_weights(yfit).to(device)
    dest = contained(root, c['run_root']) / ('%s_outer%d' % (arm, fold))
    dest.mkdir(parents=True, exist_ok=False)
    def save(name, obj):
        (dest/name).write_text(json.dumps(obj, indent=2), encoding='utf-8')
    save('config.json', dict(contract_sha256=expected_sha256, classes=CLASSES, arm=arm,
        outer=fold, fixed=FIXED, fit_ids=[r.pid for r in fit], selection_ids=[r.pid for r in selection],
        initial_state_sha256=hashlib.sha256(b''.join(k.encode()+v.detach().cpu().numpy().tobytes() for k,v in net.state_dict().items())).hexdigest(),
        candidate_only=True, historical_probability_recovery_claimed=False))
    generator, augment_rng = torch.Generator().manual_seed(FIXED['seed']), random.Random(FIXED['seed'])
    history, best, stale, started = [], None, 0, time.monotonic()
    for epoch in range(1, 81):
        factor = epoch/5 if epoch <= 5 else .5*(1+math.cos(math.pi*(epoch-5)/75))
        for group in optimizer.param_groups:
            group['lr'] = .0003*factor
        order, losses = torch.randperm(len(fit), generator=generator).tolist(), []
        for offset in range(0, len(order), 16):
            index = order[offset:offset+16]
            aug = []
            for block in range(0, len(index), 4):
                value = (augment_rng.random() < .5, augment_rng.random() < .5, augment_rng.uniform(.95, 1.05))
                aug.extend([value]*len(index[block:block+4]))
            group = [load_patient(root, lookup[fit[i].pid], arm) for i in index]
            receipt = optimize_patient_group(net, optimizer, group, [yfit[i] for i in index], weights,
                                              device, aug, precision='bf16')
            losses.append(receipt['weighted_patient_loss'])
            if time.monotonic()-started > 3600:
                raise TimeoutError('Original1hworker bound;no automatic retry')
        probabilities = []
        net.eval()
        with torch.no_grad():
            for target in selection:
                logp, _ = patient_forward(net, load_patient(root, lookup[target.pid], arm), device, precision='bf16')
                probabilities.append(logp.exp().cpu().numpy())
        probability = np.stack(probabilities)
        m = metrics(np.array([CLASSES.index(r.label) for r in selection]), probability)
        history.append(dict(epoch=epoch, group_loss_mean=float(np.mean(losses)), **m))
        save('history.json', history)
        score = (round(m['macro_auc'], 4), m['macro_f1'])
        checkpoint = dict(model=net.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                          arm=arm, outer=fold, classes=CLASSES, contract_sha256=expected_sha256)
        torch.save(checkpoint, str(dest/'last.pt'))
        if best is None or score > best:
            best, stale = score, 0
            torch.save(checkpoint, str(dest/'best.pt'))
            save('selected_metrics.json', dict(epoch=epoch, **m))
            with (dest/'selection_predictions.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.writer(stream)
                writer.writerow(['pid', 'label']+['prob_'+c for c in CLASSES])
                writer.writerows([[r.pid, r.label]+p.tolist() for r,p in zip(selection, probability)])
        elif epoch >= 10:
            stale += 1
        print(json.dumps(dict(epoch=epoch, macro_auc=m['macro_auc'], arm=arm, outer=fold)), flush=True)
        if epoch >= 10 and stale >= 15:
            break
    save('COMPLETED.json', dict(epochs=epoch, seconds=time.monotonic()-started,
        candidate_development_only=True, internal_test_used=False, external_used=False,
        final_model_frozen=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['check', 'train'])
    parser.add_argument('--contract', required=True)
    parser.add_argument('--contract-sha256', required=True)
    parser.add_argument('--arm', choices=['native_nominal_control', 'physical_full_fov'])
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.mode == 'check':
        print(json.dumps(preflight(root, args.contract, args.contract_sha256), indent=2))
    else:
        run(root, args.contract, args.contract_sha256, args.arm, args.fold, args.gpu)
