"""Persistent bounded job runner for the user-authorized V2 training plan."""
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import pandas as pd
from pam50_v2_core import ROOT,OUT,RUN,SEED,manifest,save_json,emit,sha

LOG=ROOT/'logs/pam50_v2'

def job(name,script,args,timeout=14400):
    LOG.mkdir(parents=True,exist_ok=True)
    receipt=LOG/(name+'_receipt.json')
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if old['status']=='completed': return old
        raise RuntimeError('Prior unsuccessful stage requires explicit review: '+name)
    log=LOG/(name+'.log'); command=[sys.executable,'-u',str(ROOT/'scripts'/script)]+list(map(str,args))
    started=time.time(); status=dict(name=name,command=command,started=started,status='running',log=str(log))
    with log.open('w',encoding='utf-8') as stream:
        process=subprocess.Popen(command,cwd=str(ROOT),stdout=stream,stderr=subprocess.STDOUT)
        status['pid']=process.pid; save_json(LOG/(name+'_running.json'),status)
        emit('v2_job_start',name=name,pid=process.pid)
        last_size=0; unchanged=0
        while process.poll() is None:
            size=log.stat().st_size
            unchanged=unchanged+1 if size==last_size else 0; last_size=size
            status.update(last_check=time.time(),log_bytes=size,alive=True,stall_advisory=unchanged>=3)
            save_json(LOG/(name+'_running.json'),status)
            if time.time()-started>timeout:
                emit('HARD_TIMEOUT_killing_owned_process_tree',name=name,pid=process.pid)
                subprocess.run(['taskkill.exe','/PID',str(process.pid),'/T','/F'],stdout=stream,stderr=subprocess.STDOUT)
                process.wait(timeout=30)
                status.update(status='timeout',exit_code=process.returncode)
                save_json(receipt,status); raise TimeoutError(name)
            time.sleep(15)
        code=process.returncode
    status.update(status='completed' if code==0 else 'failed',exit_code=code,seconds=time.time()-started,alive=False)
    save_json(receipt,status); save_json(LOG/(name+'_running.json'),status)
    emit('v2_job_end',name=name,status=status['status'],seconds=status['seconds'])
    if code: raise RuntimeError('Stage failed, pipeline will not retry: '+name)
    return status

def seg_worker(gpu):
    for outer in range(3):
        job('seg_outer%d_inner%d'%(outer,gpu),'pam50_v2_train.py',
            ['segmentation','--gpu',gpu,'--outer',outer,'--inner',gpu])

def classification_worker(gpu,source,view,seed,outers=(0,1,2)):
    for outer in outers:
        name='classifier_%s_%s_outer%d_seed%d'%(source,view,outer,seed)
        job(name,'pam50_v2_train.py',['classify','--gpu',gpu,'--outer',outer,'--source',source,'--view',view,'--seed',seed])

def screen_gate():
    rows=[]
    for source,view in [('automatic','single'),('automatic','multi'),('reference','multi')]:
        for outer in range(3):
            folder=RUN/'classifiers'/('%s_%s_outer%d_seed%d'%(source,view,outer,SEED))
            m=json.loads((folder/'selected_metrics.json').read_text())
            completed=json.loads((folder/'COMPLETED.json').read_text())
            baseline=json.loads((RUN/('baseline_outer_%d'%outer)/'metrics.json').read_text())
            rows.append(dict(source=source,view=view,outer=outer,macro_auc=m['macro_auc'],
                baseline_auc=baseline['macro_auc'],delta_auc=m['macro_auc']-baseline['macro_auc'],
                macro_f1=m['macro_f1'],recalls=m['recalls'],best_epoch=m['epoch'],
                all_recalls_positive=min(m['recalls'])>0,max_single_streak=completed['max_single_class_streak']))
    frame=pd.DataFrame(rows); frame.to_csv(OUT/'screening_results.csv',index=False)
    candidates=[]
    for view in ['single','multi']:
        part=frame[frame.source.eq('automatic')&frame.view.eq(view)]
        eligible=part.delta_auc.mean()>=.02 and (part.delta_auc>0).sum()>=2 and \
            part.all_recalls_positive.all() and part.max_single_streak.max()<5
        candidates.append(dict(view=view,eligible=bool(eligible),mean_auc=float(part.macro_auc.mean()),
            mean_delta=float(part.delta_auc.mean()),positive_folds=int((part.delta_auc>0).sum())))
    eligible=[r for r in candidates if r['eligible']]
    winner=max(eligible,key=lambda r:(r['mean_auc'],r['view']=='single'))['view'] if eligible else None
    result=dict(candidates=candidates,winner=winner,gate_is_resource_policy_not_significance=True,
        test_used=False,original_validation_used=False)
    save_json(OUT/'screening_gate.json',result); emit('v2_screening_gate',**result)
    return winner

def run():
    manifest(); LOG.mkdir(parents=True,exist_ok=True)
    lock=LOG/'pipeline.lock'
    descriptor=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(descriptor,str(os.getpid()).encode()); os.close(descriptor)
    save_json(OUT/'PIPELINE_STATUS.json',dict(status='running',pid=os.getpid(),started=time.time(),stage='sanity',test_used=False))
    try:
        job('roi_sanity','pam50_v2_core.py',['roi','--source','reference','--sanity-only'],timeout=3600)
        job('sanity','pam50_v2_train.py',['sanity','--gpu',0],timeout=3600)
        save_json(OUT/'PIPELINE_STATUS.json',dict(status='running',pid=os.getpid(),stage='six_isolated_segmenters',test_used=False))
        with cf.ThreadPoolExecutor(max_workers=3) as pool:
            work=[pool.submit(seg_worker,gpu) for gpu in [0,1]]
            work.append(pool.submit(job,'roi_reference','pam50_v2_core.py',['roi','--source','reference']))
            for f in cf.as_completed(work): f.result()
        for outer in range(3):
            job('assemble_outer%d'%outer,'pam50_v2_train.py',['assemble','--outer',outer])
            job('roi_auto_outer%d'%outer,'pam50_v2_core.py',['roi','--source','automatic','--outer',outer])
            job('baseline_outer%d'%outer,'pam50_v2_train.py',['baseline','--outer',outer])
        save_json(OUT/'PIPELINE_STATUS.json',dict(status='running',pid=os.getpid(),stage='ROI_classification',test_used=False))
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(classification_worker,0,'automatic','single',SEED),
                     pool.submit(classification_worker,1,'automatic','multi',SEED)]
            for f in cf.as_completed(futures): f.result()
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(classification_worker,0,'reference','multi',SEED,(0,2)),
                     pool.submit(classification_worker,1,'reference','multi',SEED,(1,))]
            for f in cf.as_completed(futures): f.result()
        winner=screen_gate()
        if winner:
            with cf.ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(classification_worker,gpu,'automatic',winner,SEED+100*(gpu+1)) for gpu in [0,1]]
                for f in cf.as_completed(futures): f.result()
        save_json(OUT/'PIPELINE_STATUS.json',dict(status='completed',pid=os.getpid(),finished=time.time(),
            winner=winner,additional_seed_runs=6 if winner else 0,test_used=False,original_validation_used=False,
            external_used=False,final_model_frozen=False))
        emit('v2_pipeline_completed',winner=winner,test_used=False)
    except Exception as exc:
        save_json(OUT/'PIPELINE_STATUS.json',dict(status='failed_needs_review',pid=os.getpid(),time=time.time(),
            error=repr(exc),test_used=False)); raise
    finally:
        lock.unlink()  # only the lock created by this process

if __name__=='__main__': run()
