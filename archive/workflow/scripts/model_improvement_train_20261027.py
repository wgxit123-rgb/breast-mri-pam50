"""Two prespecified development variants, three existing folds, GPU1 only.

Reuse the sealed original loop with audited AST changes, not an edited baseline.
No module import reads imagery/targets or allocates a GPU.
"""
import argparse
import ast
import copy
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import time
import unittest

import model_improvement_20261027 as io
import exploratory743_20261026 as old
from owned_windows_job_v2 import supervise

ROOT, BASE, RUNS = io.ROOT, io.BASE, io.RUNS
SELF = 'scripts/model_improvement_train_20261027.py'
VARIANTS = ['enhancement_weighted', 'enhancement_balanced']
STAGES = ['%s_outer%d' % (v, f) for f in range(3) for v in VARIANTS]
PLAN = 'reports/article_completion_plan_20261026/'
PLAN_FILES = [PLAN + n for n in ['PLAN.md', 'STATE.json', 'HANDOFF.json', 'NOTES.md']]


def command(mode):
    return [io.PYTHON, '-X', 'utf8', '-B', SELF, mode]


def enhancement_hook(module, args):
    import torch
    x = args[0]
    # Signed differences are never clipped or written into the source cache.
    return (torch.stack((x[:, 0], x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), dim=1),)


def balanced_order(yfit, generator):
    """Exactly four fitting patients/class/group, sampled with replacement.

    An epoch has ceil(nfit/16) groups, matching baseline optimizer-step budget.
    These are sampler draws, not unique patients; no inverse weighting is added.
    """
    import torch
    pools = [[i for i, y in enumerate(yfit) if y == c] for c in range(4)]
    if not yfit or any(not p for p in pools) or any(y not in range(4) for y in yfit):
        raise ValueError('Four-class fitting-only sampling population required')
    order = []
    for _ in range(math.ceil(len(yfit) / 16)):
        group = []
        for pool in pools:
            index = torch.randint(len(pool), (4,), generator=generator).tolist()
            group.extend(pool[i] for i in index)
        order.extend(group[i] for i in torch.randperm(16, generator=generator).tolist())
    return order


def transformed_train(variant):
    if variant not in VARIANTS:
        raise ValueError('Undeclared variant')
    source = inspect.getsource(old.train)
    tree = ast.parse(source)
    changes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.dump(node.func) == ast.dump(ast.parse('torch.device', mode='eval').body):
            if len(node.args) != 2 or not isinstance(node.args[1], ast.Constant) or node.args[1].value != 0:
                raise ValueError('Unexpected original train device site')
            node.args[1].value = 1
            changes.append('device_0_to_1')
        if variant == 'enhancement_balanced' and isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'order':
                expected = ast.parse('torch.randperm(len(fit),generator=generator).tolist()', mode='eval').body
                if ast.dump(node.value) != ast.dump(expected):
                    raise ValueError('Original patient-order expression changed')
                node.value = ast.parse('balanced_order(yfit,generator)', mode='eval').body
                changes.append('fitting_only_balanced_patient_order')
    wanted = ['device_0_to_1'] + (['fitting_only_balanced_patient_order'] if variant == VARIANTS[1] else [])
    if sorted(changes) != sorted(wanted):
        raise ValueError('Unexpected number of AST changes')
    reverse = copy.deepcopy(tree)
    for node in ast.walk(reverse):
        if isinstance(node, ast.Call) and ast.dump(node.func) == ast.dump(ast.parse('torch.device', mode='eval').body):
            node.args[1].value = 0
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'order':
            node.value = ast.parse('torch.randperm(len(fit),generator=generator).tolist()', mode='eval').body
    if ast.dump(reverse) != ast.dump(ast.parse(source)):
        raise ValueError('Unallowlisted original-loop change')
    return ast.fix_missing_locations(tree), changes


def seal():
    previous = io.read('reports/citation_context_batch3_20261027/PUBLISH_PROTOCOL.json')
    protected_names = set(previous['protected_sha256']) | set(previous['input_sha256'])
    # The four article-plan trackers are allowed to change only by versioned publication.
    names = set(old.CODE) | {SELF, io.SELF,
        'scripts/exploratory743_continue_20261026.py',
        'reports/exploratory743_20261026/INPUT_MANIFEST.json',
        'reports/exploratory743_20261026/labels743.csv',
        'reports/exploratory743_20261026/INPUT_BINDINGS.json',
        'reports/model_improvement_20261027/DIAGNOSTIC_PROTOCOL.json',
        'reports/model_improvement_20261027/DIAGNOSTIC.json',
        'reports/model_improvement_20261027/AUTHORIZATION.json',
        'reports/model_improvement_20261027/EXPERIMENT_DESIGN.md'}
    for outer in range(3):
        for arm in old.ARMS:
            folder = ROOT / 'runs/exploratory743_20261026' / ('%s_outer%d' % (arm, outer))
            names.update(io.rel(p) for p in folder.iterdir() if p.is_file())
    protected_names.update(names)
    protected_names.update(PLAN_FILES)
    protected = {n: io.sha(n) for n in sorted(protected_names)}
    inputs = {n: protected[n] for n in sorted(names)}
    io.write(BASE / 'PREPARE_TRAIN_PROTOCOL.json', dict(created_utc=io.now(), cwd=str(ROOT),
        command=command('worker'), handshake_stage='prepare', scope=io.SCOPE,
        input_sha256=inputs, protected_sha256=protected, hard_timeout_s=180,
        success='Protected copies and immutable input bindings before any patient/GPU computation',
        automatic_retries=0, failure='Preserve first error, no retry',
        deterministic_scope='Exact file bindings, copies, source AST and fixed protocol; not clinical validation',
        outputs=[io.rel(BASE / 'PROTOCOL.json'), io.rel(BASE / 'prelaunch')]))


def prepare(expected):
    if io.sha(BASE / 'PREPARE_TRAIN_PROTOCOL.json') != expected:
        raise ValueError('Prepare protocol mismatch')
    p = io.read(BASE / 'PREPARE_TRAIN_PROTOCOL.json')
    for n, h in p['protected_sha256'].items():
        if io.sha(n) != h:
            raise ValueError('Prelaunch input changed: ' + n)
        target = BASE / 'prelaunch' / n
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(io.path(n)), str(target))
        if io.sha(target) != h:
            raise ValueError('Backup mismatch')
    for variant in VARIANTS:
        transformed_train(variant)
    stages = {stage: dict(arm='physical_full_fov', outer=int(stage[-1]),
        variant=stage.rsplit('_outer', 1)[0], hard_timeout_s=3700) for stage in STAGES}
    io.write(BASE / 'PROTOCOL.json', dict(created_utc=io.now(), cwd=str(ROOT),
        command=command('queue'), worker_command=command('worker'), scope=io.SCOPE,
        input_sha256=p['input_sha256'], protected_sha256=p['protected_sha256'],
        allowed_versioned_tracker_changes=PLAN_FILES, backup_root=io.rel(BASE / 'prelaunch'),
        manifest=io.ref(old.BASE / 'INPUT_MANIFEST.json'), labels=io.ref(old.BASE / 'labels743.csv'),
        fixed=old.FIXED, variants=VARIANTS, stage_order=STAGES, stages=stages,
        input_transform='after shared gain/flips/clamp, before first conv: [pre, early-pre, late-pre]; signed differences retained; caches unmodified',
        balanced_variant='4 draws/class/16-slot update group, fitting-only replacement, same ceil(nfit/16) optimizer updates per epoch, uniform loss weights; never combine sampler balancing with inverse weighting',
        counts='743 unique eligible patients remain; balanced draws repeat IDs and do not mean new patients; training group occurrence keys distinguish repeated draws',
        fixed_unchanged=['seed', 'fold ownership', 'AdamW/lr/wd', 'dropout', 'BF16',
            '80 epoch maximum', 'warmup/cosine', 'selection rule', 'patience',
            'complete coordinates', 'positive-support mean', 'argmax threshold'],
        gpu=1, max_workers=1, hard_timeout_s_per_worker=3700,
        prepare_timeout_s=180, tests_timeout_s=180, preflight_timeout_s=240,
        gpu_availability=dict(wait_s=180, interval_s=5, consecutive=2,
            min_free_MiB=18432, utilization_exclusive_max=10), automatic_retries=0,
        success='All six distinct jobs exit0 with complete predictions/checkpoints/stop receipts, then single in-controller collection',
        failure='Stop queue at first worker failure/timeout; preserve all partials; no implicit retry/resume',
        determinism='Fixed seed, original initial parameters, exact metadata/folds/AST checks; BF16 CUDA remains stochastic, no bitwise training claim',
        interpretation='Adaptive exploratory development after previous scores were known; all variants/folds retained; no superiority or independent validation from selection scores',
        independent_evaluation=False, main_study_gates_modified=False,
        outputs=[io.rel(RUNS), io.rel(BASE / 'jobs'), io.rel(BASE / 'RESULTS.json')]))
    io.event('MODEL_IMPROVEMENT_PREPARED', protected=len(p['protected_sha256']), stages=STAGES)


def verify(expected):
    if io.sha(BASE / 'PROTOCOL.json') != expected:
        raise ValueError('New training protocol changed')
    p = io.read(BASE / 'PROTOCOL.json')
    for n, h in p['input_sha256'].items():
        if io.sha(n) != h:
            raise ValueError('Bound training input changed: ' + n)
    m = io.read(io.checked(p['manifest']))
    rows = old.labels(io.checked(p['labels']), 743)
    if set(r['pid'] for r in rows) & old.EXCLUDED or p['fixed'] != old.FIXED:
        raise ValueError('Scope or original fixed settings changed')
    return p, m, rows


def runtime(variant):
    import numpy as np
    import torch
    from pam50_v2_core import network, metrics
    from physical_development_runner_candidate_v1 import load_patient
    from physical_training_bridge_candidate_v1 import patient_forward, optimize_patient_group, fit_class_weights, PatientWindows
    torch.set_num_threads(4)
    torch.cuda.set_device(1)
    torch.backends.cudnn.benchmark = False
    random_seed = old.FIXED['seed']
    old.random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Declared BF16 runtime unavailable')
    def build(channels, dropout):
        net = network(channels, dropout)
        net.register_forward_pre_hook(enhancement_hook)
        return net
    def weights(y):
        if variant == VARIANTS[0]:
            return fit_class_weights(y)
        # Validate fitting labels with the old routine but do not double-correct.
        fit_class_weights(y)
        return torch.ones(4)
    def optimize(net, optimizer, patients, targets, class_weights, device, augmentations, precision):
        # The old bridge forbids duplicate PID in one optimizer update. Distinguish
        # sampler occurrences without changing source lookup or cohort identity.
        occurrences = [PatientWindows('%s#draw%d' % (p.pid, i), p.expected_starts, p.get_window)
                       for i, p in enumerate(patients)]
        return optimize_patient_group(net, optimizer, occurrences, targets, class_weights,
            device, augmentations, precision=precision)
    return np, torch, build, metrics, load_patient, patient_forward, optimize, weights


def gpu_guard(stage):
    from exploratory743_continue_20261026 import wait_available, gpu_query
    observations = []
    def record(row):
        observations.append(row)
        io.write(BASE / 'jobs' / stage / 'GPU_AVAILABILITY.json', observations, replace=True)
    return wait_available(1, gpu_query, record)


def tests(expected):
    verify(expected)
    import torch
    from pam50_v2_core import network
    from physical_training_bridge_candidate_v1 import PatientWindows, optimize_patient_group
    torch.set_num_threads(4)
    class Tests(unittest.TestCase):
        def test_signed_not_clipped_and_no_input_mutation(self):
            x = torch.tensor([2., 1., 4.]).reshape(1, 3, 1, 1, 1)
            original = x.clone()
            out = enhancement_hook(None, (x,))[0]
            self.assertEqual(out.flatten().tolist(), [2., -1., 2.])
            self.assertTrue(torch.equal(x, original))
        def test_common_offset_cancels_only_difference(self):
            x = torch.arange(24.).reshape(1, 3, 2, 2, 2)
            a, b = enhancement_hook(None, (x,))[0], enhancement_hook(None, (x+7,))[0]
            self.assertTrue(torch.equal(a[:, 1:], b[:, 1:]))
        def test_differentiable(self):
            x = torch.ones(1, 3, 2, 2, 2, requires_grad=True)
            enhancement_hook(None, (x,))[0].sum().backward()
            self.assertEqual(x.grad[:, 0].flatten().tolist(), [-1.] * 8)
        def test_initial_parameters_identical(self):
            torch.manual_seed(old.FIXED['seed']); a = network(3, .2)
            torch.manual_seed(old.FIXED['seed']); b = network(3, .2)
            b.register_forward_pre_hook(enhancement_hook)
            self.assertTrue(all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict()))
            self.assertEqual(len(list(b.named_buffers())), 0)
        def test_balanced_fit_only_deterministic(self):
            labels = [0]*8 + [1]*4 + [2]*3 + [3]*2
            a = balanced_order(labels, torch.Generator().manual_seed(old.FIXED['seed']))
            b = balanced_order(labels, torch.Generator().manual_seed(old.FIXED['seed']))
            self.assertEqual(a, b); self.assertEqual(len(a), 32)
            for offset in range(0, len(a), 16):
                self.assertEqual([sum(labels[i] == c for i in a[offset:offset+16]) for c in range(4)], [4]*4)
        def test_missing_class_refused(self):
            with self.assertRaises(ValueError):
                balanced_order([0, 1, 2], torch.Generator())
        def test_ast_only_declared_sites(self):
            self.assertEqual(len(transformed_train(VARIANTS[0])[1]), 1)
            self.assertEqual(len(transformed_train(VARIANTS[1])[1]), 2)
        def test_checkpoint_forward_backward(self):
            # Tiny synthetic model verifies the hook/old checkpoint bridge contract.
            net = torch.nn.Sequential(torch.nn.Conv3d(3, 4, 1), torch.nn.AdaptiveAvgPool3d(1), torch.nn.Flatten())
            net.register_forward_pre_hook(enhancement_hook)
            def window(key):
                return torch.arange(24.).reshape(3,2,2,2).float()/24, torch.ones(2,2,2,dtype=torch.bool)
            patients = [PatientWindows('synthetic%d' % i, [(0,0,0)], window) for i in range(4)]
            opt = torch.optim.SGD(net.parameters(), lr=.001)
            r = optimize_patient_group(net, opt, patients, list(range(4)), torch.ones(4), torch.device('cpu'),
                [(False,False,1.)]*4, precision='fp32')
            self.assertTrue(math.isfinite(r['weighted_patient_loss']))
            self.assertGreater(r['preclip_gradient_norm'], 0)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    io.write(BASE / 'TESTS.json', dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        success=result.wasSuccessful(), protocol_sha256=expected, script_sha256=io.sha(SELF),
        patient_payload_reads=0, GPU_allocated=False, scope='finite_synthetic_checks_not_full_algorithm_verification'))
    if not result.wasSuccessful():
        raise RuntimeError('Synthetic tests failed; do not train')


def preflight(expected):
    p, m, rows = verify(expected)
    if not io.read(BASE / 'TESTS.json')['success']:
        raise ValueError('Synthetic tests missing')
    observation = gpu_guard('preflight')
    np, torch, build, metrics, load, forward, optimize, weights = runtime(VARIANTS[0])
    net = build(3, .2).to('cuda:1')
    chosen = sorted(m['arms']['physical_full_fov']['0'], key=lambda r: (-r['n_windows'], r['pid']))[0]
    patient = load(ROOT, chosen, 'physical_full_fov')
    net.train(); net.zero_grad(set_to_none=True)
    logp, receipt = forward(net, patient, torch.device('cuda', 1), precision='bf16')
    (-logp[0]).backward()  # Fixed dummy target; no selection or optimizer update.
    norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError('Invalid transformed real-input gradient')
    io.write(BASE / 'PREFLIGHT.json', dict(status='PASSED_RUNTIME_ONLY', training_protocol_sha256=expected,
        record=receipt, dummy_class=0, gradient_norm=norm, optimizer_steps=0,
        case_rule='largest technical number of windows, PID ascending; no outcomes used for case choice',
        gpu_observation=observation, torch_version=torch.__version__, cuda_version=torch.version.cuda,
        runtime='actual scientific environment; downloaded docs not a runtime version assertion',
        peak_allocated_bytes=torch.cuda.max_memory_allocated(1), clinical_validation=False))
    io.event('REAL_INPUT_PREFLIGHT_PASSED', gradient_norm=norm, **receipt)


def train(expected, stage):
    p, m, rows = verify(expected)
    if stage not in p['stage_order'] or (RUNS / stage).exists():
        raise ValueError('Unknown or already used stage; no overwrite/resume/retry')
    variant = p['stages'][stage]['variant']
    tree, changes = transformed_train(variant)
    environment = dict(vars(old))
    environment.update(BASE=BASE, RUNS=RUNS, FIXED=old.FIXED,
        verify_train=lambda h: verify(h), runtime=lambda: runtime(variant),
        gpu_guard=lambda: gpu_guard(stage), balanced_order=balanced_order)
    io.write(BASE / 'jobs' / stage / 'AMENDMENT.json', dict(variant=variant,
        original_train_source_sha256=io.sha('scripts/exploratory743_20261026.py'),
        AST_changes=changes, reverse_AST_exact=True,
        runtime_changes=['GPU1', 'stateless_signed_enhancement_pre_hook',
            'balanced fitting-only sampler and uniform weights' if variant == VARIANTS[1] else 'original inverse-frequency loss'],
        cohort_and_cache_identity_unchanged=True))
    exec(compile(tree, '<sealed_new_model_development_' + variant + '>', 'exec'), environment)
    environment['train'](expected, stage)
    config = io.read(RUNS / stage / 'config.json')
    original = io.read(ROOT / 'runs/exploratory743_20261026/physical_full_fov_outer0/config.json')
    if config['initial_state_sha256'] != original['initial_state_sha256']:
        raise ValueError('Original seeded initial parameters not matched')
    io.write(RUNS / stage / 'VARIANT.json', dict(variant=variant, protocol_sha256=expected,
        rebuild='Build original network(3,.2) and register enhancement_hook before loading state_dict',
        original_initial_state_match=True, fitting_sampler_replacement=(variant == VARIANTS[1]),
        original_input_manifest_unchanged=True, independent_validation=False))


def successful(result):
    return (result.get('status') == 'completed' and result.get('exit_code') == 0
        and not result.get('timed_out') and result.get('job_closed') is True
        and result.get('owned_active_processes_after_termination') == 0)


def launch(stage):
    name = 'PREPARE_TRAIN_PROTOCOL.json' if stage == 'prepare' else 'PROTOCOL.json'
    limit = {'prepare':180, 'tests':180, 'preflight':240}.get(stage, 3700)
    result = supervise(command('worker'), dict(scope=io.SCOPE, stage=stage,
        protocol_sha256=io.sha(BASE / name)), ROOT, BASE / 'jobs' / stage, limit)
    io.event('OWNED_STAGE_TERMINAL', stage=stage, **result)
    return 0 if successful(result) else 1


def collect(expected):
    p, m, rows = verify(expected)
    import numpy as np
    from pam50_v2_core import metrics
    summaries, combined = [], {}
    for variant in VARIANTS:
        all_rows = []
        for outer in range(3):
            stage = '%s_outer%d' % (variant, outer)
            if not successful(io.read(BASE / 'jobs' / stage / 'RESULT.json')):
                raise ValueError('Cannot collect incomplete/failed stage')
            c = io.read(RUNS / stage / 'config.json')
            sm = io.read(RUNS / stage / 'selected_metrics.json')
            done = io.read(RUNS / stage / 'COMPLETED.json')
            with (RUNS / stage / 'selection_predictions.csv').open(encoding='utf-8', newline='') as f:
                prediction = list(old.csv.DictReader(f))
            if [r['pid'] for r in prediction] != c['selection_ids']:
                raise ValueError('Changed prediction order/denominator')
            expected_rows = [r for r in rows if int(r['outer']) == outer]
            if [(r['pid'], r['label']) for r in prediction] != [(r['pid'], r['label']) for r in expected_rows]:
                raise ValueError('Development fold/label identity mismatch')
            all_rows.extend(prediction)
            summaries.append(dict(variant=variant, outer=outer, selected=sm, completed=done))
        if len(all_rows) != 743 or len({r['pid'] for r in all_rows}) != 743:
            raise ValueError('All743 unique selection predictions required')
        probs = np.asarray([[float(r['prob_' + c]) for c in CLASSES] for r in all_rows])
        if not np.isfinite(probs).all() or (probs < 0).any() or not np.allclose(probs.sum(1),1,rtol=0,atol=1e-6):
            raise ValueError('Invalid patient probabilities')
        scores = metrics(np.array([CLASSES.index(r['label']) for r in all_rows]), probs)
        folds = [r['selected'] for r in summaries if r['variant'] == variant]
        combined[variant] = dict(pooled_selected_development=scores,
            mean_fold_macro_auc=sum(r['macro_auc'] for r in folds)/3,
            independent_validation=False)
    changed = []
    for n, h in p['protected_sha256'].items():
        if n not in PLAN_FILES and io.sha(n) != h:
            changed.append(n)
    if changed:
        raise ValueError('Protected files changed: ' + repr(changed))
    io.write(BASE / 'RESULTS.json', dict(status='ALL_SIX_NEW_JOBS_COMPLETED_NOT_INDEPENDENT_VALIDATION',
        utc=io.now(), jobs=summaries, variants=combined, protocol_sha256=expected,
        baseline_report='reports/exploratory743_analysis_20261026/ANALYSIS_REPORT.md',
        report_all_variants_and_negative_results=True, automatic_promotion=False,
        new_hyperparameter_or_seed_search=False, protected_files_unchanged_except_allowed_trackers=True,
        original748_or_test_or_external_evaluation=False))
    io.event('NEW_DEVELOPMENT_RESULTS_COLLECTED', variants=combined)


def queue():
    expected = io.sha(BASE / 'PROTOCOL.json')
    verify(expected)
    pf = io.read(BASE / 'PREFLIGHT.json')
    if pf.get('training_protocol_sha256') != expected or pf.get('status') != 'PASSED_RUNTIME_ONLY':
        raise ValueError('Preflight missing/stale')
    if (BASE / 'QUEUE_STATUS.json').exists():
        raise FileExistsError('Do not start a second queue or silently retry')
    completed = []
    for stage in STAGES:
        io.write(BASE / 'QUEUE_STATUS.json', dict(status='RUNNING', stage=stage, gpu=1,
            controller_pid=os.getpid(), completed=completed, planned=STAGES,
            protocol_sha256=expected, utc=io.now()), replace=True)
        code = launch(stage)
        if code:
            io.write(BASE / 'QUEUE_STATUS.json', dict(status='FAILED_STOPPED_NO_RETRY', failed_stage=stage,
                completed=completed, not_launched=STAGES[len(completed)+1:], utc=io.now()), replace=True)
            return code
        completed.append(stage)
    io.write(BASE / 'QUEUE_STATUS.json', dict(status='SIX_JOBS_COMPLETE_COLLECTING', completed=completed,
        controller_pid=os.getpid(), utc=io.now()), replace=True)
    try:
        # Separate owned 180s collection; no second finalizer is created.
        result = supervise(command('worker'), dict(scope=io.SCOPE, stage='collect', protocol_sha256=expected),
            ROOT, BASE / 'jobs/collect', 180)
        if not successful(result):
            raise RuntimeError('Collection failed; preserve first error, no retry')
    except Exception as error:
        io.write(BASE / 'QUEUE_STATUS.json', dict(status='COLLECTION_FAILED_NO_RETRY', completed=completed,
            error=repr(error), utc=io.now()), replace=True)
        return 1
    io.write(BASE / 'QUEUE_STATUS.json', dict(status='SIX_JOBS_AND_COLLECTION_COMPLETE', completed=completed,
        result=io.rel(BASE / 'RESULTS.json'), utc=io.now(), independent_validation=False), replace=True)
    return 0


def worker():
    raw = sys.stdin.readline(32769)
    if not raw or len(raw) > 32768:
        raise ValueError('Owned handshake required')
    message = json.loads(raw)
    if message.get('scope') != io.SCOPE:
        raise ValueError('Wrong scope')
    stage, expected = message['stage'], message['protocol_sha256']
    if stage == 'prepare': prepare(expected)
    elif stage == 'tests': tests(expected)
    elif stage == 'preflight': preflight(expected)
    elif stage == 'collect': collect(expected)
    elif stage in STAGES: train(expected, stage)
    else: raise ValueError('Undeclared stage')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['seal','prepare','tests','preflight','queue','worker'])
    args = parser.parse_args()
    if args.mode == 'seal': seal()
    elif args.mode == 'worker': worker()
    elif args.mode == 'queue': sys.exit(queue())
    else: sys.exit(launch(args.mode))
