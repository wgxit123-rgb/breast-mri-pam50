"""Author-approved continuation; old training loop is reused, not rewritten.

Only two audited AST device constants and the operational GPU availability
guard differ. All original experiment files and completed models remain intact.
"""
import argparse
import ast
import concurrent.futures as cf
import copy
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
import exploratory743_20261026 as old
from owned_windows_job_v2 import supervise

ROOT=old.ROOT
BASE=ROOT/'reports/exploratory743_continue_20261026'
SELF='scripts/exploratory743_continue_20261026.py'
OLD_PROTOCOL_SHA='8178035e893ba3aa2c38cc636df7ce04e2f2b024a83d16ce8952bca0404c7eee'
CHAINS={0:['physical_full_fov_outer1','native_nominal_control_outer1'],
        1:['physical_full_fov_outer2','native_nominal_control_outer2']}
COMPLETED=['physical_full_fov_outer0','native_nominal_control_outer0']


def command(mode):
    return [old.PYTHON,'-X','utf8','-B',SELF,mode]


def source_tree(name,gpu):
    if gpu not in (0,1):raise ValueError('Only declared GPU0/1')
    tree=ast.parse(inspect.getsource(getattr(old,name)))
    before=copy.deepcopy(tree);changes=[]
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call):continue
        if name=='train' and ast.dump(node.func)==ast.dump(ast.parse('torch.device',mode='eval').body):
            if len(node.args)==2 and isinstance(node.args[0],ast.Constant) and node.args[0].value=='cuda':
                if not isinstance(node.args[1],ast.Constant) or node.args[1].value!=0:raise ValueError('Unexpected train device')
                node.args[1].value=gpu;changes.append('train torch.device cuda index')
        if name=='runtime' and ast.dump(node.func)==ast.dump(ast.parse('torch.cuda.set_device',mode='eval').body):
            if len(node.args)!=1 or not isinstance(node.args[0],ast.Constant) or node.args[0].value!=0:
                raise ValueError('Unexpected runtime device')
            node.args[0].value=gpu;changes.append('runtime set_device index')
    if len(changes)!=1:raise ValueError('Exactly one allowlisted device site per function required')
    # Reversing the one site must exactly recover every AST node.
    reverse=copy.deepcopy(tree)
    for node in ast.walk(reverse):
        if not isinstance(node,ast.Call):continue
        pattern='torch.device' if name=='train' else 'torch.cuda.set_device'
        if ast.dump(node.func)==ast.dump(ast.parse(pattern,mode='eval').body):
            node.args[-1].value=0
    if ast.dump(reverse)!=ast.dump(before):raise ValueError('Non-device training-source change')
    return tree,dict(function=name,device=gpu,allowlisted_sites=changes,other_AST_nodes_exact=True)


def wait_available(gpu,query,record,clock=time.monotonic,sleep=time.sleep,limit=180,interval=5):
    started=clock();consecutive=0
    while True:
        observation=query(gpu)
        free,util=observation['free_MiB'],observation['utilization_percent']
        if not isinstance(free,int) or not isinstance(util,int) or free<0 or not 0<=util<=100:
            raise ValueError('Malformed GPU observation; fail closed')
        good=free>=18*1024 and util<10
        consecutive=consecutive+1 if good else 0
        record(dict(utc=old.now(),gpu=gpu,elapsed_s=clock()-started,qualifying=good,
            consecutive_qualifying=consecutive,**observation))
        if consecutive>=2:return observation
        if clock()-started>=limit:raise TimeoutError('GPU availability wait expired; no worker retry')
        sleep(min(interval,max(0,limit-(clock()-started))))


def gpu_query(gpu):
    output=subprocess.check_output(['nvidia-smi','-i',str(gpu),
        '--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],
        text=True,timeout=10)
    parts=output.strip().split(',')
    if len(parts)!=2:raise ValueError('Invalid bounded GPU query')
    return dict(free_MiB=int(parts[0]),utilization_percent=int(parts[1]))


def check(protocol_sha):
    if old.sha(BASE/'PROTOCOL.json')!=protocol_sha:raise ValueError('Continuation protocol changed')
    p=old.read(BASE/'PROTOCOL.json')
    for name,digest in p['input_sha256'].items():
        if old.sha(name)!=digest:raise ValueError('Continuation input changed: '+name)
    if old.sha(old.BASE/'TRAINING_PROTOCOL.json')!=OLD_PROTOCOL_SHA:raise ValueError('Original protocol changed')
    return p


def seal():
    if (BASE/'PROTOCOL.json').exists():raise FileExistsError('No reseal/retry')
    if old.sha(old.BASE/'TRAINING_PROTOCOL.json')!=OLD_PROTOCOL_SHA:raise ValueError('Original protocol changed')
    names=old.CODE+old.PROTECTED+[SELF,old.rel(BASE/'AUTHORIZATION.json'),
        old.rel(old.BASE/'TRAINING_PROTOCOL.json'),old.rel(old.BASE/'INPUT_MANIFEST.json'),
        old.rel(old.BASE/'INPUT_BINDINGS.json'),old.rel(old.BASE/'PREFLIGHT.json'),
        old.rel(old.BASE/'QUEUE_STATUS.json'),
        old.rel(old.BASE/'jobs/physical_full_fov_outer1/worker.log'),
        old.rel(old.BASE/'jobs/physical_full_fov_outer1/RESULT.json')]
    for stage in COMPLETED:
        names += [old.rel(f) for f in (old.RUNS/stage).iterdir() if f.is_file()]
        receipt=old.read(old.BASE/'jobs'/stage/'RESULT.json')
        if old.exit_status(receipt)!=0:raise ValueError('Completed job lacks successful owned receipt')
        names.append(old.rel(old.BASE/'jobs'/stage/'RESULT.json'))
    for stage in sum(CHAINS.values(),[]):
        if (old.RUNS/stage).exists():raise FileExistsError('Do not overwrite a partial/finished patient run: '+stage)
    bindings={}
    for name in dict.fromkeys(names):
        digest=old.sha(name);backup=BASE/'prelaunch'/name
        backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(str(old.inside(name)),str(backup))
        if old.sha(backup)!=digest:raise ValueError('Protection mismatch')
        bindings[name]=digest
    old.write(BASE/'PROTOCOL.json',dict(created_utc=old.now(),scope='continue_four_unfinished_exploratory743',
        cwd=str(ROOT),command=command('queue'),tests_command=command('tests'),worker_command=command('worker'),
        input_sha256=bindings,original_training_protocol_sha256=OLD_PROTOCOL_SHA,
        original_patient_bindings='Reuse sealed INPUT_BINDINGS; original loader rechecks imagery before first use per worker',
        source_transform='Exactly runtime set_device(0)->assigned_gpu and train torch.device(cuda,0)->assigned_gpu; all other AST nodes exact',
        operational_amendment=dict(max_workers=2,one_per_gpu=True,chains={str(k):v for k,v in CHAINS.items()},
            wait_limit_s=180,interval_s=5,consecutive_qualifying=2,min_free_MiB=18432,max_utilization_exclusive=10),
        hard_timeout_s_per_worker=3700,tests_timeout_s=30,automatic_retries=0,
        on_failure='No new launches; already-running other GPU worker may finish, never killed except its hard timeout',
        success='Four new jobs exit0 and write COMPLETED/checkpoints; retain previous two completions unchanged',
        outputs=[old.rel(BASE),old.rel(old.RUNS)],
        scientific_settings_changed=False,original_748_gates_modified=False,
        determinism='Exact source/metadata/initial-state checks; seeded BF16 GPU execution is stochastic, no bitwise reproducibility claim',
        original_metadata_protection='Existing4483 prelaunch copies retained; this directory additionally protects bound originals/completed artifacts'))
    old.event('CONTINUATION_SEALED',protocol_sha256=old.sha(BASE/'PROTOCOL.json'),protected=len(bindings))


def worker():
    raw=sys.stdin.readline(32769)
    if not raw or len(raw)>32768:raise ValueError('Owned-job handshake required')
    payload=json.loads(raw);check(payload['protocol_sha256'])
    stage,gpu=payload['stage'],payload['gpu']
    if stage not in CHAINS.get(gpu,[]):raise ValueError('Foreign continuation assignment')
    if (old.RUNS/stage).exists():raise FileExistsError('Refuse overwrite/resume/retry of patient artifacts')
    samples=[]
    def record(row):
        samples.append(row);old.write(BASE/'jobs'/stage/'GPU_AVAILABILITY.json',samples,replace=True)
    def guard():return wait_available(gpu,gpu_query,record)
    environment=dict(vars(old));environment['gpu_guard']=guard
    transformations=[]
    for name in ['runtime','train']:
        tree,receipt=source_tree(name,gpu);transformations.append(receipt)
        exec(compile(ast.fix_missing_locations(tree),'<sealed_%s_gpu%d>'%(name,gpu),'exec'),environment)
    old.write(BASE/'jobs'/stage/'RUNTIME_AMENDMENT.json',dict(stage=stage,gpu=gpu,
        continuation_protocol_sha256=payload['protocol_sha256'],original_training_protocol_sha256=OLD_PROTOCOL_SHA,
        transformations=transformations,scientific_loop_other_AST_nodes_exact=True))
    environment['train'](OLD_PROTOCOL_SHA,stage)
    config=old.read(old.RUNS/stage/'config.json')
    expected=old.read(old.RUNS/COMPLETED[0]/'config.json')['initial_state_sha256']
    if config['initial_state_sha256']!=expected:raise ValueError('Seeded initial state differs from retained baseline')


def queue():
    expected=old.sha(BASE/'PROTOCOL.json');p=check(expected)
    test=old.read(BASE/'TESTS.json')
    if test['success'] is not True or test['script_sha256']!=old.sha(SELF):raise ValueError('New dispatch tests missing')
    lock=threading.Lock();failed=threading.Event();state={}
    if (BASE/'QUEUE_STATUS.json').exists():raise FileExistsError('Continuation queue already used')
    def update(stage,status,**values):
        with lock:
            state[stage]=dict(status=status,**values)
            old.write(BASE/'QUEUE_STATUS.json',dict(status='RUNNING',controller_pid=os.getpid(),utc=old.now(),
                original_completed=COMPLETED,tasks=state,protocol_sha256=expected),replace=True)
    def chain(gpu):
        codes=[]
        for stage in CHAINS[gpu]:
            if failed.is_set():update(stage,'NOT_LAUNCHED_AFTER_FAILURE',gpu=gpu);break
            update(stage,'RUNNING',gpu=gpu)
            r=supervise(command('worker'),dict(stage=stage,gpu=gpu,protocol_sha256=expected),
                ROOT,BASE/'jobs'/stage,3700)
            code=old.exit_status(r);codes.append(code)
            update(stage,'COMPLETED' if code==0 else 'FAILED',gpu=gpu,result=r)
            old.event('CONTINUATION_STAGE_TERMINAL',stage=stage,gpu=gpu,**r)
            if code:failed.set();break
        return codes
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(chain,gpu) for gpu in CHAINS]
        codes=[code for f in futures for code in f.result()]
    if any(codes) or len(codes)!=4:
        final='FAILED_STOPPED_NO_RETRY';code=next((c for c in codes if c),1)
    else:final='COMPLETED_ALL_SIX_EXPLORATORY_JOBS';code=0
    old.write(BASE/'QUEUE_STATUS.json',dict(status=final,controller_pid=os.getpid(),utc=old.now(),
        original_completed=COMPLETED,tasks=state,protocol_sha256=expected),replace=True)
    return code


def tests():
    class DispatchTests(unittest.TestCase):
        def test_two_AST_sites_only(self):
            for gpu in (0,1):
                for name in ('runtime','train'):
                    _,r=source_tree(name,gpu);self.assertTrue(r['other_AST_nodes_exact'])
        def test_foreign_device(self):
            with self.assertRaises(ValueError):source_tree('train',2)
        def test_gpu0_source_identical(self):
            for name in ('runtime','train'):
                tree,_=source_tree(name,0)
                self.assertEqual(ast.dump(tree),ast.dump(ast.parse(inspect.getsource(getattr(old,name)))))
        def test_preserved_thresholds(self):
            times=[0.];observed=[];seq=iter([(18431,0),(18432,10),(18432,9),(18432,9)])
            def query(_):
                m,u=next(seq);return dict(free_MiB=m,utilization_percent=u)
            wait_available(1,query,observed.append,lambda:times[0],lambda d:times.__setitem__(0,times[0]+d))
            self.assertEqual([r['qualifying'] for r in observed],[False,False,True,True])
        def test_busy_timeout(self):
            t=[0.]
            with self.assertRaises(TimeoutError):wait_available(0,lambda _:dict(free_MiB=23000,utilization_percent=80),
                lambda _:None,lambda:t[0],lambda d:t.__setitem__(0,t[0]+d),limit=10)
            self.assertEqual(t[0],10)
        def test_unknown_query_fail_closed(self):
            with self.assertRaises(ValueError):wait_available(0,lambda _:dict(free_MiB=-1,utilization_percent=0),lambda _:None)
        def test_no_completed_tasks_relaunched(self):
            self.assertFalse(set(sum(CHAINS.values(),[]))&set(COMPLETED))
            self.assertEqual(len(set(sum(CHAINS.values(),[]))),4)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DispatchTests))
    old.write(BASE/'TESTS.json',dict(tests=result.testsRun,success=result.wasSuccessful(),
        failures=len(result.failures),errors=len(result.errors),script_sha256=old.sha(SELF),
        protocol_sha256=old.sha(BASE/'PROTOCOL.json'),patient_reads=False,GPU_used=False))
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['seal','tests','queue','worker']);a=p.parse_args()
    if a.mode=='seal':seal()
    elif a.mode=='worker':worker()
    elif a.mode=='tests':sys.exit(tests())
    else:sys.exit(queue())
