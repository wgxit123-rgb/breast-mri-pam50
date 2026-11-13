"""Isolated supervision adapter, not evidence that patient training is eligible.

Check mode reads only bound code/contracts/gate metadata. The handshake worker
performs the same checks before importing the existing torch runner. No old
runner, protocol, patient inputs or scientific gates are rewritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
FIXED = dict(seed=20260923, patients=748, folds=3, max_epochs=80,
             learning_rate=.0003, weight_decay=.0001, dropout=.2,
             patient_microbatch=4, accumulation=4, window_microbatch=4,
             precision='BF16', optimizer='AdamW', warmup_epochs=5,
             patience=15, count_stale_after_epoch=10, clip_norm=5.0,
             selection='macro_AUC_rounded4_then_macro_F1_then_earliest')
REQUIRED = ('all_development_selected_phase_geometry_verified',
            'shared_anatomical_fov_and_automatic_localization_validated',
            'localizer_probability_reproducibility_resolved')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inside(root, name):
    path = Path(name)
    if not name or '..' in path.parts:
        raise ValueError('Contained nonempty path required')
    result = (path if path.is_absolute() else root / path).resolve()
    result.relative_to(root.resolve())
    return result


def read_bound(root, item):
    path = inside(root, item['path'])
    if sha(path) != item['sha256']:
        raise ValueError('Bound metadata changed: ' + str(path))
    return json.loads(path.read_text(encoding='utf-8'))


def assess(plan, contract, gate):
    """Metadata screen only; READY still requires the old runner's full checks."""
    if plan.get('schema') != 'supervised_physical_development_v2':
        raise ValueError('Unrecognized plan')
    if plan.get('fixed') != FIXED or contract.get('fixed') != FIXED:
        raise ValueError('Fixed training settings changed')
    if plan.get('scope') != '748_development_only' or contract.get('scope') != plan['scope']:
        raise ValueError('Foreign population or endpoint scope')
    if plan.get('hard_timeout_s') != 3700 or plan.get('automatic_retries') != 0:
        raise ValueError('Original timeout/no-retry boundary changed')
    if plan.get('human_training_authorized') is not True:
        raise ValueError('Training authority missing')
    blockers = []
    if gate.get('development_total') != 748:
        blockers.append('original748_membership_count_mismatch')
    if gate.get('development_geometry_verified') != 748 or gate.get('development_geometry_remaining') != 0:
        blockers.append('strict_source_coverage_incomplete')
    blockers.extend(key for key in REQUIRED if gate.get('required_gates', {}).get(key) is not True)
    if contract.get('status') != 'ADOPTED_FOR_DEVELOPMENT_COMPARISON':
        blockers.append('execution_contract_not_adopted')
    if not contract.get('input_manifest'):
        blockers.append('complete_input_manifest_not_bound')
    if not contract.get('independent_execution_review'):
        blockers.append('independent_execution_review_missing')
    if contract.get('run_root') != plan.get('training_run_root'):
        blockers.append('new_isolated_run_root_not_bound_in_execution_contract')
    return dict(status='BLOCKED_NOT_LAUNCHED' if blockers else 'READY_FOR_SEPARATE_RUNNER_PREFLIGHT',
                blockers=blockers, source_verified=gate.get('development_geometry_verified'),
                source_required=748, source_remaining=gate.get('development_geometry_remaining'),
                targets_opened=False, patient_inputs_opened=False, GPU_allocated=False,
                optimizer_steps=0, new_CNN_epochs=0,
                final_freeze_required_at_this_stage=False,
                physical_validation_output_required_as_input=False,
                clinical_review_text_used_as_gate=False)


def inspect(root, plan_path, plan_sha):
    plan = read_bound(root, dict(path=plan_path, sha256=plan_sha))
    for name, digest in plan['code_sha256'].items():
        if sha(inside(root, name)) != digest:
            raise ValueError('Code binding changed: ' + name)
    contract = read_bound(root, plan['execution_contract'])
    gate = read_bound(root, plan['gate_snapshot'])
    # A sealed snapshot must still equal the live gate; never use a stale pass.
    if sha(inside(root, plan['live_gate_path'])) != plan['gate_snapshot']['sha256']:
        raise ValueError('Live gate changed; new explicit stage binding required')
    return assess(plan, contract, gate), plan


def owned(command, payload, root, output, timeout_s):
    from owned_windows_job_v2 import supervise
    return supervise(command, payload, root, output, timeout_s)


def launch(root, plan_path, plan_sha, arm, fold, gpu, output):
    check, plan = inspect(root, plan_path, plan_sha)
    if check['status'] != 'READY_FOR_SEPARATE_RUNNER_PREFLIGHT':
        raise RuntimeError('Not launched: ' + ','.join(check['blockers']))
    if arm not in ('native_nominal_control', 'physical_full_fov') or fold not in (0, 1, 2) or gpu not in (0, 1):
        raise ValueError('Fixed arms/folds/local GPU index required')
    expected = inside(root, plan['supervision_root']) / ('%s_outer%d' % (arm, fold))
    if output.resolve() != expected:
        raise ValueError('Unexpected supervision output path')
    command = [sys.executable, '-X', 'utf8', '-B', str(SELF), 'worker']
    return owned(command, dict(scope='production_development', plan=plan_path,
                 plan_sha256=plan_sha, arm=arm, fold=fold, gpu=gpu),
                 root, output, plan['hard_timeout_s'])


def worker():
    # No torch/model/patient access before the supervisor's post-assignment line.
    raw = sys.stdin.readline(32769)
    if not raw or len(raw) > 32768:
        raise ValueError('Bounded supervisor handshake required')
    payload = json.loads(raw)
    if payload.get('scope') == 'synthetic_adapter_fixture':
        mode = payload.get('mode')
        if mode == 'echo':
            print('SYNTHETIC_ADAPTER_HANDSHAKE_OK', flush=True)
            return
        if mode == 'hang':
            print('SYNTHETIC_ADAPTER_TIMEOUT_EXPECTED', flush=True)
            while True:
                time.sleep(.05)
        raise ValueError('Unknown synthetic mode')
    if payload.get('scope') != 'production_development':
        raise ValueError('Foreign handshake scope')
    check, plan = inspect(ROOT, payload['plan'], payload['plan_sha256'])
    if check['status'] != 'READY_FOR_SEPARATE_RUNNER_PREFLIGHT':
        raise RuntimeError('Worker refused before torch import: ' + ','.join(check['blockers']))
    if payload['arm'] not in ('native_nominal_control', 'physical_full_fov') or payload['fold'] not in (0, 1, 2) or payload['gpu'] not in (0, 1):
        raise ValueError('Foreign training arguments')
    from physical_development_runner_candidate_v1 import run
    bound_contract = plan['execution_contract']
    # This runner repeats its own manifest/review-basis/GPU availability checks.
    run(ROOT, bound_contract['path'], bound_contract['sha256'],
        payload['arm'], payload['fold'], payload['gpu'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['check', 'train', 'worker'])
    parser.add_argument('--plan')
    parser.add_argument('--plan-sha256')
    parser.add_argument('--output')
    parser.add_argument('--arm')
    parser.add_argument('--fold', type=int)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    if args.mode == 'worker':
        worker()
        return
    if not args.plan or not args.plan_sha256 or not args.output:
        parser.error('Explicit plan, plan SHA and output required')
    output = inside(ROOT, args.output)
    if args.mode == 'check':
        if output.exists():
            raise FileExistsError('No receipt overwrite/implicit recheck')
        result, _ = inspect(ROOT, args.plan, args.plan_sha256)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2)
        print(json.dumps(result), flush=True)
        return
    print(json.dumps(launch(ROOT, args.plan, args.plan_sha256,
                           args.arm, args.fold, args.gpu, output)), flush=True)


if __name__ == '__main__':
    main()
