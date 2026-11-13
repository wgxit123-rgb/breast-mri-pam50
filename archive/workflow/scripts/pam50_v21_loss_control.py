"""Bounded paired loss-normalization ablation. Frozen V2 code is not edited."""
import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from pam50_v2_core import ROOT, OUT, RUN, SEED, CLASSES, manifest, network, load_roi, infer, metrics, sha, save_json, emit
from pam50_v2_train import seed_everything

REPORT=ROOT/'reports/pam50_v21_loss_control'
RUNS=ROOT/'runs/pam50_v21_loss_control'
LOG=ROOT/'logs/pam50_v21_loss_control'

def setup():
    REPORT.mkdir(parents=True,exist_ok=True); RUNS.mkdir(parents=True,exist_ok=True); LOG.mkdir(parents=True,exist_ok=True)
    if (REPORT/'protocol.json').exists(): raise RuntimeError('Refuse to replace frozen protocol')
    manifest()
    for shard in range(2):
        assert json.loads((ROOT/'reports/pam50_v2_audit'/('reinference_shard_%d.json'%shard)).read_text())['all_passed']
    assert (ROOT/'reports/pam50_v2_audit/input_summary.json').exists()
    save_json(REPORT/'protocol.json',dict(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        authorization='2026-09-23 user approved audit-triggered single-factor experiments',
        hypothesis='Group-weight normalization may reduce collapse; performance improvement is not assumed',
        intervention='Only gradient accumulation normalization: microbatch sample fraction versus class-weight-sum fraction',
        fixed='Original V2 automatic multi ROI, 3 folds, seed, network, initial weights, sample order, augmentation, optimizer, LR, max80, early stop and checkpoint selection',
        arms=['legacy','group_weight'],runs=6,seed=SEED,train_patients=748,
        primary='paired development macro-AUC delta across three folds, report all folds; no significance claim',
        secondary=['macro_F1','per_class_recall','single_class_prediction_frequency','fit_vs_selection_gap'],
        resource_gate='No further seeds or models automatically; report both arms even if negative',
        termination='original early stopping; 1h per worker hard timeout; fail rather than auto-retry',
        original_validation_used=False,test_used=False,external_used=False,final_model_frozen=False,
        v2_protocol_sha256=sha(OUT/'protocol.json'),script_sha256=sha(Path(__file__)),
        source='https://docs.pytorch.org/docs/2.3/generated/torch.nn.CrossEntropyLoss.html'))

def train(gpu,outer,arm):
    protocol=json.loads((REPORT/'protocol.json').read_text())
    assert protocol['script_sha256']==sha(Path(__file__))
    df=manifest(); seed_everything(gpu,SEED)
    fit=df[df.outer.ne(outer)]; selection=df[df.outer.eq(outer)]
    dest=RUNS/('%s_outer%d'%(arm,outer)); dest.mkdir(parents=True,exist_ok=False)
    x,y=load_roi(fit,'automatic',outer,'multi'); xv,yv=load_roi(selection,'automatic',outer,'multi')
    net=network(3).cuda(); counts=np.bincount(y.numpy(),minlength=4)
    weights=torch.tensor(len(y)/(4*counts),device='cuda',dtype=torch.float32)
    loss_fn=torch.nn.CrossEntropyLoss(weight=weights)
    optimizer=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.0001)
    generator=torch.Generator().manual_seed(SEED); history=[]; best=None; stale=0; streak=0; max_streak=0
    begin=time.time()
    save_json(dest/'config.json',dict(arm=arm,outer=outer,seed=SEED,gpu=gpu,classes=CLASSES,
        fit_ids=fit.pid.tolist(),selection_ids=selection.pid.tolist(),protocol_sha256=sha(REPORT/'protocol.json')))
    for epoch in range(1,81):
        tick=time.time(); factor=epoch/5 if epoch<=5 else .5*(1+math.cos(math.pi*(epoch-5)/75))
        for optgroup in optimizer.param_groups: optgroup['lr']=.0003*factor
        net.train(); order=torch.randperm(len(fit),generator=generator); blocks=list(order.split(4))
        losses=[]; gradients=[]
        for start in range(0,len(blocks),4):
            group=blocks[start:start+4]; total=sum(len(index) for index in group)
            group_denominator=weights[y[torch.cat(group)].cuda()].sum() if arm=='group_weight' else None
            optimizer.zero_grad(set_to_none=True)
            for index in group:
                xb=x[index].cuda().float(); yb=y[index].cuda()
                for axis in [3,4]:
                    if random.random()<.5: xb=torch.flip(xb,[axis])
                xb=(xb*random.uniform(.95,1.05)).clamp_(0,5)
                with torch.autocast('cuda',dtype=torch.bfloat16): loss=loss_fn(net(xb),yb)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
                contribution=len(index)/total if arm=='legacy' else weights[yb].sum()/group_denominator
                (loss*contribution).backward(); losses.append(float(loss.detach().cpu()))
            norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),5)); assert np.isfinite(norm)
            gradients.append(norm); optimizer.step()
        p=infer(net,xv); m=metrics(yv.numpy(),p)
        single=sum(v>0 for v in m['prediction_counts'])==1
        streak=streak+1 if single else 0; max_streak=max(max_streak,streak)
        row=dict(epoch=epoch,train_loss=float(np.mean(losses)),macro_auc=m['macro_auc'],macro_f1=m['macro_f1'],
            selection_log_loss=m['log_loss'],recalls=json.dumps(m['recalls']),prediction_counts=json.dumps(m['prediction_counts']),
            single_class_streak=streak,gradient_norm_mean=float(np.mean(gradients)),seconds=round(time.time()-tick,3))
        history.append(row); pd.DataFrame(history).to_csv(dest/'history.csv',index=False)
        score=(round(m['macro_auc'],4),m['macro_f1'])
        ckpt=dict(model=net.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,arm=arm,outer=outer,seed=SEED,classes=CLASSES)
        torch.save(ckpt,dest/'last.tmp'); os.replace(dest/'last.tmp',dest/'last.pt')
        if best is None or score>best:
            best=score; stale=0; torch.save(ckpt,dest/'best.pt')
            pred=selection[['pid','label']].copy()
            for i,c in enumerate(CLASSES): pred['prob_'+c]=p[:,i]
            pred.to_csv(dest/'selection_predictions.csv',index=False)
            save_json(dest/'selected_metrics.json',dict(epoch=epoch,**m))
        elif epoch>=10: stale+=1
        emit('loss_control_epoch',arm=arm,outer=outer,**row)
        if time.time()-begin>3600: raise TimeoutError('1h arm timeout')
        if epoch>=10 and stale>=15: break
    save_json(dest/'COMPLETED.json',dict(epochs=epoch,seconds=time.time()-begin,max_single_class_streak=max_streak,test_used=False))

def pipeline():
    assert (REPORT/'protocol.json').exists()
    status=dict(status='running',started=time.time(),pid=os.getpid(),test_used=False,external_used=False)
    save_json(REPORT/'PIPELINE_STATUS.json',status)
    try:
        for outer in range(3):
            jobs=[]
            for gpu,arm in enumerate(['legacy','group_weight']):
                path=LOG/('%s_outer%d.log'%(arm,outer)); stream=path.open('w',encoding='utf-8')
                command=[sys.executable,'-X','utf8',str(Path(__file__)),'train','--gpu',str(gpu),'--outer',str(outer),'--arm',arm]
                proc=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,cwd=str(ROOT))
                jobs.append((proc,stream,path,arm)); emit('loss_control_started',arm=arm,outer=outer,pid=proc.pid)
            begun=time.time()
            while any(p.poll() is None for p,_,_,_ in jobs):
                if time.time()-begun>3700:
                    for p,_,_,_ in jobs:
                        if p.poll() is None: p.kill()
                    raise TimeoutError('Worker hard timeout')
                time.sleep(10)
            for p,stream,path,arm in jobs:
                stream.close()
                if p.returncode: raise RuntimeError('%s failed, inspect %s'%(arm,path))
            status.update(completed_folds=outer+1); save_json(REPORT/'PIPELINE_STATUS.json',status)
        rows=[]
        for outer in range(3):
            original=json.loads((RUN/'classifiers'/('automatic_multi_outer%d_seed%d'%(outer,SEED))/'selected_metrics.json').read_text())
            for arm in ['legacy','group_weight']:
                folder=RUNS/('%s_outer%d'%(arm,outer)); m=json.loads((folder/'selected_metrics.json').read_text())
                done=json.loads((folder/'COMPLETED.json').read_text())
                rows.append(dict(outer=outer,arm=arm,epoch=m['epoch'],macro_auc=m['macro_auc'],macro_f1=m['macro_f1'],
                    recalls=json.dumps(m['recalls']),pred_counts=json.dumps(m['prediction_counts']),
                    max_single_streak=done['max_single_class_streak'],original_v2_auc=original['macro_auc']))
        pd.DataFrame(rows).to_csv(REPORT/'paired_results.csv',index=False)
        status.update(status='completed',finished=time.time(),final_model_frozen=False,extra_experiments_started=0)
        save_json(REPORT/'PIPELINE_STATUS.json',status)
    except Exception as e:
        status.update(status='failed',error=repr(e),finished=time.time()); save_json(REPORT/'PIPELINE_STATUS.json',status)
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['setup','run','train']); p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--outer',type=int,default=0); p.add_argument('--arm',choices=['legacy','group_weight'],default='legacy'); a=p.parse_args()
    if a.mode=='setup': setup()
    elif a.mode=='run': pipeline()
    else: train(a.gpu,a.outer,a.arm)
