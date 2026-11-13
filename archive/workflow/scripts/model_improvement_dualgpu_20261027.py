"""Scheduling-only amendment with tested exact JobObject supervisory handoff."""
import ast
from datetime import datetime, timezone, timedelta
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import unittest

import model_improvement_20261027 as io
import model_improvement_train_20261027 as current
import exploratory743_20261026 as old
from owned_job_handoff_20261027 import Adoption, process_birth
from owned_windows_job_v2 import supervise

ROOT=io.ROOT
BASE=ROOT/'reports/model_improvement_dualgpu_20261027'
ORIGINAL=io.BASE
SELF='scripts/model_improvement_dualgpu_20261027.py'
SCOPE='scheduling_only_amendment_same_six_model_improvement_jobs'
TRACKERS=['reports/article_completion_plan_20261026/'+n for n in ['PLAN.md','STATE.json','HANDOFF.json','NOTES.md']]
CHAINS={0:['enhancement_balanced_outer0','enhancement_weighted_outer1','enhancement_balanced_outer2'],
        1:['enhancement_balanced_outer1','enhancement_weighted_outer2']}


def command(mode):return [io.PYTHON,'-X','utf8','-B',SELF,mode]


def process_snapshot():
    source="Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' -and $_.CommandLine -match 'model_improvement_train_20261027.py' } | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
    raw=subprocess.check_output(['powershell.exe','-NoProfile','-Command',source],timeout=15)
    rows=json.loads(raw.decode('utf-8-sig'))
    return rows if isinstance(rows,list) else [rows]


def device_runtime_tree(gpu):
    if gpu not in (0,1):raise ValueError('Only explicitly authorized devices')
    tree=ast.parse(inspect.getsource(current.runtime));changed=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and ast.dump(node.func)==ast.dump(ast.parse('torch.cuda.set_device',mode='eval').body):
            if len(node.args)!=1 or not isinstance(node.args[0],ast.Constant) or node.args[0].value!=1:
                raise ValueError('Unexpected source runtime device')
            node.args[0].value=gpu;changed+=1
    if changed!=1:raise ValueError('Exactly one runtime device site required')
    return ast.fix_missing_locations(tree)


def device_train_tree(variant,gpu):
    tree,changes=current.transformed_train(variant);changed=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and ast.dump(node.func)==ast.dump(ast.parse('torch.device',mode='eval').body):
            if node.args[1].value!=1:raise ValueError('Unexpected source train device')
            node.args[1].value=gpu;changed+=1
    if changed!=1:raise ValueError('Exactly one train device site required')
    return ast.fix_missing_locations(tree),changes


def init():
    q=io.read(ORIGINAL/'QUEUE_STATUS.json')
    if q['status']!='RUNNING' or q['stage']!='enhancement_weighted_outer0' or q['completed']:
        raise ValueError('This amendment binds the original first live job only; no stale takeover')
    processes=process_snapshot()
    parent=[r for r in processes if r['ProcessId']==q['controller_pid'] and r['CommandLine'].rstrip().endswith('queue')]
    children=[r for r in processes if r['ParentProcessId']==q['controller_pid'] and r['CommandLine'].rstrip().endswith('worker')]
    if len(parent)!=1 or len(children)!=1:raise ValueError('Exact single current queue/worker required')
    parent,child=parent[0],children[0]
    launch=io.read(ORIGINAL/'jobs'/q['stage']/'LAUNCH.json')
    deadline=datetime.fromisoformat(launch['created_utc'])+timedelta(seconds=launch['timeout_s']-2)
    previous=io.read(ORIGINAL/'PROTOCOL.json')
    protected=dict(previous['protected_sha256'])
    protected.update({n:io.sha(n) for n in TRACKERS})
    protected[io.rel(ORIGINAL/'HANDOFF.json')]=io.sha(ORIGINAL/'HANDOFF.json')
    names=[SELF,'scripts/owned_job_handoff_20261027.py',current.SELF,io.SELF,
        'scripts/owned_windows_job_v2.py','scripts/exploratory743_continue_20261026.py',
        io.rel(ORIGINAL/'PROTOCOL.json'),io.rel(ORIGINAL/'TESTS.json'),io.rel(ORIGINAL/'PREFLIGHT.json'),
        io.rel(BASE/'AUTHORIZATION.json'),io.rel(BASE/'task_plan.md'),
        io.rel(ORIGINAL/'jobs'/q['stage']/'LAUNCH.json'),
        'scripts/exploratory743_20261026.py']
    inputs={n:io.sha(n) for n in names}
    protected.update(inputs)
    io.write(BASE/'PREPARE_PROTOCOL.json',dict(created_utc=io.now(),scope=SCOPE,cwd=str(ROOT),
        command=command('worker'),handshake_stage='prepare',hard_timeout_s=180,
        input_sha256=inputs,protected_sha256=protected,
        original_training_protocol_sha256=io.sha(ORIGINAL/'PROTOCOL.json'),
        handoff=dict(stage=q['stage'],parent=parent,worker=child,
            parent_birth=process_birth(parent['ProcessId']),worker_birth=process_birth(child['ProcessId']),
            original_deadline_utc=deadline.isoformat(),deadline_not_extended=True),
        tests=dict(hard_timeout_s=90,normal_synthetic_worker_s=2,timeout_synthetic_worker_s=60,
            adopted_synthetic_timeout_s=15,synthetic_controller_timeout_s=45),
        chains={str(g):s for g,s in CHAINS.items()},maximum_workers=2,one_worker_per_gpu=True,
        automatic_retries=0,training_hard_timeout_s=3700,
        worker_command=command('worker'),queue_command=command('queue'),
        operational_changes_only=['GPU0 for three originally unstarted stages','same-kernel-job ownership transfer for originalGPU1worker','two disjoint chains'],
        success='Unchanged active worker/PID and originaldeadline, disjoint bounded new workers, all original scientific settings/files unchanged; one six-job collection',
        failure='Before retirement leave old scheduler intact; after handoff preserve first failure, stop new launches, let other already-running owned worker finish within its declared bound; no retry',
        determinism='Exact device-only AST reversal, PID plus creation-time and same-kernel-job checks; CUDA training remains stochastic across identicalGPU models',
        source_references=['https://learn.microsoft.com/en-us/windows/win32/api/handleapi/nf-handleapi-duplicatehandle',
            'https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects',
            'https://raw.githubusercontent.com/winsiderss/systeminformer/master/phnt/include/ntexapi.h'],
        outputs=[io.rel(BASE),io.rel(ORIGINAL/'QUEUE_STATUS.json'),io.rel(io.RUNS)],
        allowed_versioned_tracker_changes=TRACKERS+[io.rel(ORIGINAL/'HANDOFF.json')],
        scientific_training_code_or_settings_modified=False))


def verify(expected):
    if io.sha(BASE/'PROTOCOL.json')!=expected:raise ValueError('Amendment protocol mismatch')
    p=io.read(BASE/'PROTOCOL.json')
    for n,h in p['input_sha256'].items():
        if io.sha(n)!=h:raise ValueError('Amendment source/input changed: '+n)
    return p


def prepare(expected):
    if io.sha(BASE/'PREPARE_PROTOCOL.json')!=expected:raise ValueError('Preparation protocol changed')
    p=io.read(BASE/'PREPARE_PROTOCOL.json')
    for n,h in p['protected_sha256'].items():
        if io.sha(n)!=h:raise ValueError('Protected input changed: '+n)
        dest=BASE/'prelaunch'/n;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(str(io.path(n)),str(dest))
        if io.sha(dest)!=h:raise ValueError('Backup mismatch')
    for g in (0,1):
        device_runtime_tree(g)
        for v in current.VARIANTS:device_train_tree(v,g)
    io.write(BASE/'PROTOCOL.json',dict(p,prepared_utc=io.now(),
        action='Adopt existing exact JobObject before retiring old scheduler; no worker restart'))
    io.event('DUALGPU_PROTOCOL_PREPARED',protected=len(p['protected_sha256']))


def fake_worker(payload):
    folder=io.path(payload['folder'])
    io.write(folder/'READY.json',dict(pid=os.getpid(),parent_pid=os.getppid(),birth=process_birth(os.getpid())))
    time.sleep(payload['sleep_s'])
    io.write(folder/'NATURAL_COMPLETE.json',dict(pid=os.getpid()))


def fake_controller(folder,sleep_s):
    result=supervise(command('worker'),dict(scope=SCOPE,stage='fake',folder=folder,sleep_s=sleep_s),
        ROOT,io.path(folder)/'original_job',45)
    io.write(io.path(folder)/'old_scheduler_result.json',result)


def tests(expected):
    p=verify(expected);outcomes=[]
    # AST: reversing only the device sites recovers the original complete trees.
    for g in (0,1):
        tree=device_runtime_tree(g)
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and ast.dump(node.func)==ast.dump(ast.parse('torch.cuda.set_device',mode='eval').body):node.args[0].value=1
        if ast.dump(tree)!=ast.dump(ast.parse(inspect.getsource(current.runtime))):raise ValueError('Runtime non-device change')
        for v in current.VARIANTS:
            tree,_=device_train_tree(v,g)
            for node in ast.walk(tree):
                if isinstance(node,ast.Call) and ast.dump(node.func)==ast.dump(ast.parse('torch.device',mode='eval').body):node.args[1].value=1
            baseline,_=current.transformed_train(v)
            if ast.dump(tree)!=ast.dump(baseline):raise ValueError('Train non-device change')
    outcomes.append(dict(test='device_only_reverse_AST_for_both_devices_both_variants',passed=True))
    # Adopt only disposable synthetic processes, never the real scheduler in tests.
    for name,sleep_s,expect_timeout in [('normal',2,False),('hard_timeout',60,True)]:
        folder=BASE/'synthetic'/name;folder.mkdir(parents=True,exist_ok=False)
        process=subprocess.Popen(command('fake_controller')+[io.rel(folder),str(sleep_s)],
            cwd=str(ROOT),creationflags=subprocess.CREATE_NO_WINDOW,
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        began=time.monotonic();ready=folder/'READY.json'
        try:
            while not ready.exists():
                if process.poll() is not None or time.monotonic()-began>10:raise RuntimeError('Synthetic controller failed to start')
                time.sleep(.02)
            r=io.read(ready)
            deadline=datetime.now(timezone.utc)+timedelta(seconds=15)
            adoption=Adoption(process.pid,r['pid'],process_birth(process.pid),r['birth'],deadline.isoformat())
            # A read-only ownership probe must leave both processes alive.
            adoption.close_probe()
            if process.poll() is not None:raise RuntimeError('Read-only probe affected old scheduler')
            adoption=Adoption(process.pid,r['pid'],process_birth(process.pid),r['birth'],deadline.isoformat())
            adoption.commit();result=adoption.finish()
            process.wait(timeout=5)
            if result['timed_out']!=expect_timeout or result['owned_active_processes_after_termination']!=0:
                raise ValueError('Synthetic handoff/timeout contract failed')
            if not expect_timeout and (result['exit_code']!=0 or not (folder/'NATURAL_COMPLETE.json').exists()):
                raise ValueError('Synthetic worker did not naturally finish unchanged')
            if expect_timeout and result['exit_code']!=124:raise ValueError('Synthetic hard timeout not preserved')
            io.write(folder/'ADOPTED_RESULT.json',result)
            outcomes.append(dict(test='synthetic_'+name,passed=True,result=result))
        finally:
            # Only these disposable local test processes may be cleaned up here.
            # Normal paths have already retired the scheduler and closed its job.
            if process.poll() is None:
                process.terminate();process.wait(timeout=5)
    io.write(BASE/'TESTS.json',dict(success=True,tests=outcomes,protocol_sha256=expected,
        patient_reads=0,model_GPU_allocated=False,synthetic_processes_only=True))
    io.event('SCHEDULING_TESTS_PASSED',tests=len(outcomes))


def guard_gpu(gpu,stage):
    from exploratory743_continue_20261026 import wait_available,gpu_query
    observations=[]
    def record(r):
        observations.append(r);io.write(ORIGINAL/'jobs'/stage/'GPU_AVAILABILITY.json',observations,replace=True)
    return wait_available(gpu,gpu_query,record)


def train(expected,stage,gpu):
    p=verify(expected)
    if stage not in p['chains'][str(gpu)] or (io.RUNS/stage).exists():raise ValueError('Undeclared/already-started assignment')
    variant=stage.rsplit('_outer',1)[0]
    runtime_env=dict(vars(current))
    exec(compile(device_runtime_tree(gpu),'<sealed_runtime_gpu%d>'%gpu,'exec'),runtime_env)
    tree,changes=device_train_tree(variant,gpu)
    environment=dict(vars(old))
    environment.update(BASE=ORIGINAL,RUNS=io.RUNS,FIXED=old.FIXED,
        verify_train=lambda h:current.verify(h),runtime=lambda:runtime_env['runtime'](variant),
        gpu_guard=lambda:guard_gpu(gpu,stage),balanced_order=current.balanced_order)
    io.write(ORIGINAL/'jobs'/stage/'AMENDMENT.json',dict(variant=variant,gpu=gpu,
        operational_protocol_sha256=expected,scientific_protocol_sha256=p['original_training_protocol_sha256'],
        original_loop_AST_changes=changes,only_additional_runtime_and_train_device_sites_changed=True,
        no_new_seed_or_hyperparameters=True,original_source_files_unchanged=True))
    exec(compile(tree,'<sealed_training_gpu%d>'%gpu,'exec'),environment)
    environment['train'](p['original_training_protocol_sha256'],stage)
    config=io.read(io.RUNS/stage/'config.json')
    reference=io.read(ROOT/'runs/exploratory743_20261026/physical_full_fov_outer0/config.json')
    if config['initial_state_sha256']!=reference['initial_state_sha256']:raise ValueError('Seeded initial parameters changed')
    io.write(io.RUNS/stage/'VARIANT.json',dict(variant=variant,gpu=gpu,
        protocol_sha256=p['original_training_protocol_sha256'],operational_amendment_sha256=expected,
        rebuild='original network(3,.2) with signed enhancement_hook then load state_dict',
        original_initial_state_match=True,fitting_sampler_replacement=(variant==current.VARIANTS[1]),
        original_input_manifest_unchanged=True,independent_validation=False))


def queue():
    expected=io.sha(BASE/'PROTOCOL.json');p=verify(expected)
    if not io.read(BASE/'TESTS.json')['success']:raise ValueError('Synthetic scheduling tests required')
    if (BASE/'QUEUE_STATUS.json').exists():raise FileExistsError('No duplicate dual-GPU queue/retry')
    q=io.read(ORIGINAL/'QUEUE_STATUS.json');h=p['handoff'];stage=h['stage']
    if q['status']!='RUNNING' or q['stage']!=stage or q['controller_pid']!=h['parent']['ProcessId']:
        raise ValueError('Old scheduling state changed before handoff; do not touch it')
    actual=process_snapshot()
    for bound in [h['parent'],h['worker']]:
        if not any(r==bound for r in actual):raise ValueError('Real process command/parent identity changed')
    # Check every unstarted artifact before retiring the original scheduler.
    for assigned in sum(CHAINS.values(),[]):
        if (io.RUNS/assigned).exists() or (ORIGINAL/'jobs'/assigned).exists():raise FileExistsError('Future assignment already claimed')
    before=io.read(io.RUNS/stage/'progress.json')
    # GPU0 availability is checked again inside its actual owned training worker.
    from exploratory743_continue_20261026 import gpu_query
    a=gpu_query(0);time.sleep(5);b=gpu_query(0)
    if any(x['free_MiB']<18432 or x['utilization_percent']>=10 for x in [a,b]):
        raise RuntimeError('GPU0 no longer available; old scheduler left intact')
    adoption=Adoption(h['parent']['ProcessId'],h['worker']['ProcessId'],h['parent_birth'],h['worker_birth'],h['original_deadline_utc'])
    try:
        io.write(BASE/'HANDOFF_BARRIER.json',dict(utc=io.now(),before=before,original_stage=stage,
            worker_pid=h['worker']['ProcessId'],worker_birth=h['worker_birth'],same_job_duplicate_retained=True,
            original_deadline_utc=h['original_deadline_utc'],gpu0_observations=[a,b],
            only_old_scheduler_will_retire=True,training_worker_will_not_be_restarted=True))
        backup=BASE/'pre_transition';backup.mkdir(exist_ok=False)
        shutil.copy2(str(ORIGINAL/'QUEUE_STATUS.json'),str(backup/'ORIGINAL_QUEUE_STATUS.json'))
        adoption.commit()
    except Exception:
        adoption.close_probe();raise
    lock=threading.Lock();failed=threading.Event();completed=[];tasks={stage:dict(status='ADOPTED_RUNNING',gpu=1)}
    def update(name,status,**extra):
        with lock:
            tasks[name]=dict(status=status,**extra)
            snapshot=dict(status='RUNNING_PARALLEL_AMENDMENT',controller_pid=os.getpid(),utc=io.now(),
                completed=list(completed),tasks=dict(tasks),operational_protocol_sha256=expected,
                original_training_protocol_sha256=p['original_training_protocol_sha256'],maximum_workers=2)
            io.write(BASE/'QUEUE_STATUS.json',snapshot,replace=True)
            io.write(ORIGINAL/'QUEUE_STATUS.json',dict(snapshot,canonical_queue=io.rel(BASE/'QUEUE_STATUS.json')),replace=True)
    update(stage,'ADOPTED_RUNNING',gpu=1,same_worker_pid=h['worker']['ProcessId'])
    def run_stage(name,gpu):
        if failed.is_set():update(name,'NOT_LAUNCHED_AFTER_FAILURE',gpu=gpu);return False
        update(name,'RUNNING',gpu=gpu)
        result=supervise(command('worker'),dict(scope=SCOPE,stage=name,gpu=gpu,protocol_sha256=expected),
            ROOT,ORIGINAL/'jobs'/name,3700)
        if not current.successful(result):failed.set();update(name,'FAILED_NO_RETRY',gpu=gpu,result=result);return False
        with lock:completed.append(name)
        update(name,'COMPLETED',gpu=gpu,result=result);return True
    def chain0():
        try:
            for name in CHAINS[0]:
                if not run_stage(name,0):break
        except Exception as error:
            failed.set();update('gpu0_controller','FAILED_NO_RETRY',error=repr(error))
    gpu0=threading.Thread(target=chain0,name='one_owned_gpu0_chain');gpu0.start()
    try:
        result=adoption.finish()
        result.update(output=str(ORIGINAL/'jobs'/stage),handoff_protocol_sha256=expected,
            original_launch_receipt_preserved=True,finished_utc=io.now())
        io.write(ORIGINAL/'jobs'/stage/'RESULT.json',result)
        if not current.successful(result):failed.set();update(stage,'FAILED_NO_RETRY',gpu=1,result=result)
        else:
            with lock:completed.append(stage)
            update(stage,'COMPLETED',gpu=1,result=result)
            for name in CHAINS[1]:
                if not run_stage(name,1):break
    except Exception as error:
        failed.set();update('gpu1_controller','FAILED_NO_RETRY',error=repr(error))
    finally:
        gpu0.join()
        adoption.close_probe()
    if failed.is_set() or set(completed)!=set(current.STAGES):
        terminal=dict(status='FAILED_STOPPED_NO_RETRY',completed=completed,tasks=tasks,utc=io.now())
        io.write(BASE/'QUEUE_STATUS.json',terminal,replace=True)
        io.write(ORIGINAL/'QUEUE_STATUS.json',dict(terminal,canonical_queue=io.rel(BASE/'QUEUE_STATUS.json')),replace=True)
        return 1
    result=supervise(command('worker'),dict(scope=SCOPE,stage='collect',protocol_sha256=expected),
        ROOT,ORIGINAL/'jobs/collect',180)
    terminal=dict(status='SIX_JOBS_AND_COLLECTION_COMPLETE' if current.successful(result) else 'COLLECTION_FAILED_NO_RETRY',
        completed=completed,tasks=tasks,utc=io.now(),result=io.rel(ORIGINAL/'RESULTS.json'),
        operational_amendment_sha256=expected,independent_validation=False)
    io.write(BASE/'QUEUE_STATUS.json',terminal,replace=True)
    io.write(ORIGINAL/'QUEUE_STATUS.json',dict(terminal,canonical_queue=io.rel(BASE/'QUEUE_STATUS.json')),replace=True)
    return 0 if current.successful(result) else 1


def collect(expected):
    p=verify(expected)
    current.collect(p['original_training_protocol_sha256'])
    io.write(ORIGINAL/'DEVICE_SCHEDULE_REPORT.json',dict(operational_protocol_sha256=expected,
        assignments={'enhancement_weighted_outer0':1,**{s:g for g,ss in CHAINS.items() for s in ss}},
        same_original_worker_adopted=True,original_deadline_preserved=True,
        original_scientific_settings_unchanged=True,cross_device_bitwise_training_not_claimed=True,
        canonical_queue=io.rel(BASE/'QUEUE_STATUS.json')))


def worker():
    message=json.loads(sys.stdin.readline(32769))
    if message.get('scope')!=SCOPE:raise ValueError('Wrong scope')
    stage=message['stage']
    if stage=='fake':fake_worker(message);return
    expected=message['protocol_sha256']
    if stage=='prepare':prepare(expected)
    elif stage=='tests':tests(expected)
    elif stage=='collect':collect(expected)
    else:train(expected,stage,message['gpu'])


if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='init':init()
    elif mode=='worker':worker()
    elif mode=='fake_controller':fake_controller(sys.argv[2],int(sys.argv[3]))
    elif mode=='queue':sys.exit(queue())
    elif mode in ['prepare','tests']:
        name='PREPARE_PROTOCOL.json' if mode=='prepare' else 'PROTOCOL.json'
        result=supervise(command('worker'),dict(scope=SCOPE,stage=mode,protocol_sha256=io.sha(BASE/name)),
            ROOT,BASE/'jobs'/mode,180 if mode=='prepare' else 90)
        io.event('DUALGPU_STAGE_TERMINAL',stage=mode,**result)
        sys.exit(0 if current.successful(result) else 1)
    else:raise ValueError('Unknown mode')
