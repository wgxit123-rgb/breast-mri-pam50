"""Train-only paired phase ablation with frozen legacy automatic crops.

This diagnostic keeps native-index geometry and legacy segmentation fixed.
It is NOT the final aligned pipeline and never accesses independent endpoints.
"""
import argparse,concurrent.futures as cf,json,math,os,random,subprocess,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import nibabel as nib
from pam50_v2_core import ROOT,CLASSES,SEED,SHAPE,manifest,network,infer,metrics
from pam50_v2_train import seed_everything
from nominal_phase_contract import load_contract,sha,write_json
from prepare_nominal_phase_inputs import path_for,FIELDS

REPORT=ROOT/'reports/nominal_phase_control_20261012'
DATA=ROOT/'data/nominal_phase_control_20261012'
RUN=ROOT/'runs/nominal_phase_control_20261012'
LOG=ROOT/'logs/nominal_phase_control_20261012'
LEGACY=ROOT/'data/pam50_v2'

def emit(event,**kw): print(json.dumps(dict(event=event,**kw),ensure_ascii=False),flush=True)

def setup():
    for p in [REPORT,DATA,RUN,LOG]: p.mkdir(parents=True,exist_ok=True)
    assert not (REPORT/'protocol.json').exists(),'Refuse to overwrite protocol'
    df=manifest(); phase,table=load_contract()
    assert set(table.loc[table.cohort.eq('ispy'),'pid'])==set(df.pid)
    assert torch.cuda.device_count()>=2
    inventory=[]
    for fold in range(3):
        for pid in df.pid:
            base=LEGACY/('outer_%d/automatic_roi'%fold)
            p=base/(pid+'.json'); r=json.loads(p.read_text())
            assert r['pid']==pid and r['outer']==fold and r['source']=='automatic'
            inventory.append(dict(pid=pid,outer=fold,metadata=str(p),metadata_sha256=sha(p),
                image=str(base/(pid+'.npy')),image_sha256=r['sha256']))
    write_json(REPORT/'fixed_crop_inventory.json',inventory)
    write_json(REPORT/'protocol.json',dict(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
      authorization='User requested autonomous dependency setup/training and training plan, 2026-10-12',
      experiment='paired_nominal_phase_only_diagnostic',arms=['legacy','nominal'],folds=3,seed=SEED,runs=6,
      primary='paired macro one-vs-rest AUC change on existing development selection folds; descriptive only',
      secondary=['macro_F1','balanced_accuracy','class_recalls','single_class_prediction_frequency'],
      intervention='Only source phase indices: previous selection versus frozen nominal 0,1,n-1',
      fixed='patient folds, existing automatic native-index crop, intensity scale, network, initial state, sample order, augmentation, group-weight loss normalization, optimizer and stopping',
      caveat='Legacy segmenters are held fixed. This is a classifier-input phase ablation, not end-to-end nominal-phase retraining or physically aligned external validation.',
      timing_uncertain=True,physical_geometry_released=False,final_model_frozen=False,
      original_validation_used=False,test_used=False,external_used=False,train_patients=748,
      max_epochs=80,early_stop_patience=15,early_stop_count_after_epoch=10,
      learning_rate=.0003,weight_decay=.0001,batch=4,accumulation=4,dropout=.2,
      optimizer='AdamW',precision='BF16',class_weight='fit-only inverse frequency',
      selection='macro-AUC rounded 4dp, then macro-F1, then earliest',
      automatic_extra_seeds=False,automatic_retries=False,per_worker_hard_timeout_s=3700,
      script_sha256=sha(Path(__file__)),phase_contract_sha256=sha(ROOT/'reports/nominal_phase_20261012/phase_contract.json'),
      crop_inventory_sha256=sha(REPORT/'fixed_crop_inventory.json'),
      source_sha256={n:sha(ROOT/'scripts'/n) for n in ['pam50_v2_core.py','pam50_v2_train.py','nominal_phase_contract.py','prepare_nominal_phase_inputs.py']},
      environment=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
                       gpus=[torch.cuda.get_device_name(i) for i in range(2)])))
    emit('protocol_frozen',runs=6)

def contract():
    p=json.loads((REPORT/'protocol.json').read_text())
    assert sha(Path(__file__))==p['script_sha256']
    assert sha(REPORT/'fixed_crop_inventory.json')==p['crop_inventory_sha256']
    assert sha(ROOT/'reports/nominal_phase_20261012/phase_contract.json')==p['phase_contract_sha256']
    for n,h in p['source_sha256'].items(): assert sha(ROOT/'scripts'/n)==h,n
    load_contract(); manifest(); return p

def prepare():
    contract(); torch.set_num_threads(1)
    _,table=load_contract(); table=table[table.cohort.eq('ispy')]
    inv=json.loads((REPORT/'fixed_crop_inventory.json').read_text())
    lookup={(r['pid'],r['outer']):r for r in inv}
    def one(row):
        pid=row.pid; arrays=[]; source=[]
        for f in FIELDS:
            path=path_for(pid,int(getattr(row,f)))
            arrays.append(nib.load(str(path)).get_fdata(dtype=np.float32))
            source.append(dict(phase=int(getattr(row,f)),path=str(path),sha256=sha(path)))
        result=[]
        for fold in range(3):
            info=lookup[(pid,fold)]; assert sha(info['metadata'])==info['metadata_sha256']
            assert sha(info['image'])==info['image_sha256']
            crop=json.loads(Path(info['metadata']).read_text()); scale=crop['scale']
            assert all(list(a.shape)==crop['native_shape'] for a in arrays)
            slices=tuple(slice(a,b) for a,b in zip(crop['crop_start'],crop['crop_stop']))
            volumes=[]
            for arr in arrays:
                arr=np.clip(arr[slices]/scale,0,5)
                assert arr.size and np.isfinite(arr).all()
                v=torch.nn.functional.interpolate(torch.from_numpy(arr.copy())[None,None],size=SHAPE,mode='trilinear',align_corners=False)[0,0]
                volumes.append(v.numpy().astype(np.float16))
            x=np.stack(volumes); old=np.load(info['image'])
            assert np.array_equal(x[0],old[0]),'Unchanged precontrast channel does not reproduce'
            unchanged=[]
            for j,k in enumerate(['old_pre','old_early','old_late']):
                if int(getattr(row,FIELDS[j]))==int(getattr(row,k)):
                    assert np.array_equal(x[j],old[j]),'Unchanged channel differs'
                    unchanged.append(j)
            folder=DATA/('outer_%d'%fold); folder.mkdir(parents=True,exist_ok=True)
            dest=folder/(pid+'.npy'); assert not dest.exists(),'Refuse to overwrite cache'
            np.save(dest,x)
            r=dict(pid=pid,outer=fold,path=str(dest),sha256=sha(dest),source=source,
                fixed_crop_metadata_sha256=info['metadata_sha256'],unchanged_channels_exact=unchanged,
                shape=list(x.shape),geometry='unchanged_native_index_diagnostic_only',test_used=False)
            write_json(folder/(pid+'.json'),r); result.append(r)
        return result
    records=[]; start=time.time()
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        for i,r in enumerate(pool.map(one,table.itertuples(index=False)),1):
            records.extend(r)
            if i%25==0: emit('cache_progress',patients=i,total=748,seconds=round(time.time()-start))
    write_json(REPORT/'cache_inventory.json',records)
    write_json(REPORT/'CACHE_READY.json',dict(n=len(records),patients=748,
      inventory_sha256=sha(REPORT/'cache_inventory.json'),precontrast_reproduction_all_exact=True,
      test_used=False,seconds=time.time()-start))
    emit('cache_ready',n=len(records))

def data(df,fold,arm):
    inventory=json.loads((REPORT/('cache_inventory.json' if arm=='nominal' else 'fixed_crop_inventory.json')).read_text())
    lookup={r['pid']:r for r in inventory if r['outer']==fold}; xs=[]
    for pid in df.pid:
        r=lookup[pid]; path=r['path'] if arm=='nominal' else r['image']; h=r['sha256'] if arm=='nominal' else r['image_sha256']
        assert sha(path)==h,'Input changed: '+pid
        xs.append(np.load(path))
    return torch.from_numpy(np.stack(xs)),torch.tensor([CLASSES.index(v) for v in df.label],dtype=torch.long)

def train(gpu,fold,arm):
    protocol=contract(); ready=json.loads((REPORT/'CACHE_READY.json').read_text())
    assert ready['n']==2244 and sha(REPORT/'cache_inventory.json')==ready['inventory_sha256']
    df=manifest(); seed_everything(gpu,SEED)
    fit=df[df.outer.ne(fold)]; val=df[df.outer.eq(fold)]
    dest=RUN/('%s_outer%d'%(arm,fold)); dest.mkdir(parents=True,exist_ok=False)
    x,y=data(fit,fold,arm); xv,yv=data(val,fold,arm)
    net=network(3).cuda(); counts=np.bincount(y.numpy(),minlength=4)
    weights=torch.tensor(len(y)/(4*counts),device='cuda',dtype=torch.float32)
    loss_fn=torch.nn.CrossEntropyLoss(weight=weights)
    opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.0001)
    generator=torch.Generator().manual_seed(SEED); history=[]; best=None; stale=0; streak=0; maximum=0
    # Hash tensors, not serialization metadata, to verify paired initial states.
    import hashlib
    h=hashlib.sha256()
    for n,v in net.state_dict().items(): h.update(n.encode()); h.update(v.cpu().numpy().tobytes())
    write_json(dest/'config.json',dict(arm=arm,outer=fold,seed=SEED,gpu=gpu,initial_state_sha256=h.hexdigest(),
      fit_ids=fit.pid.tolist(),selection_ids=val.pid.tolist(),classes=CLASSES,protocol_sha256=sha(REPORT/'protocol.json')))
    begin=time.time()
    for epoch in range(1,81):
        tick=time.time(); factor=epoch/5 if epoch<=5 else .5*(1+math.cos(math.pi*(epoch-5)/75))
        for group in opt.param_groups: group['lr']=.0003*factor
        net.train(); order=torch.randperm(len(fit),generator=generator); blocks=list(order.split(4)); losses=[]
        for start in range(0,len(blocks),4):
            group=blocks[start:start+4]; denominator=weights[y[torch.cat(group)].cuda()].sum(); opt.zero_grad(set_to_none=True)
            for index in group:
                xb=x[index].cuda().float(); yb=y[index].cuda()
                for axis in [3,4]:
                    if random.random()<.5: xb=torch.flip(xb,[axis])
                xb=(xb*random.uniform(.95,1.05)).clamp_(0,5)
                with torch.autocast('cuda',dtype=torch.bfloat16): loss=loss_fn(net(xb),yb)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
                (loss*(weights[yb].sum()/denominator)).backward(); losses.append(float(loss.detach().cpu()))
            norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),5)); assert np.isfinite(norm); opt.step()
        prob=infer(net,xv); m=metrics(yv.numpy(),prob)
        streak=streak+1 if sum(v>0 for v in m['prediction_counts'])==1 else 0; maximum=max(maximum,streak)
        row=dict(epoch=epoch,train_loss=float(np.mean(losses)),macro_auc=m['macro_auc'],macro_f1=m['macro_f1'],
          selection_log_loss=m['log_loss'],recalls=json.dumps(m['recalls']),prediction_counts=json.dumps(m['prediction_counts']),
          single_class_streak=streak,seconds=round(time.time()-tick,3))
        history.append(row); pd.DataFrame(history).to_csv(dest/'history.csv',index=False)
        score=(round(m['macro_auc'],4),m['macro_f1'])
        ckpt=dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=epoch,arm=arm,outer=fold,seed=SEED,classes=CLASSES)
        torch.save(ckpt,dest/'last.tmp'); os.replace(dest/'last.tmp',dest/'last.pt')
        if best is None or score>best:
            best=score; stale=0; torch.save(ckpt,dest/'best.pt')
            pred=val[['pid','label']].copy()
            for i,c in enumerate(CLASSES): pred['prob_'+c]=prob[:,i]
            pred.to_csv(dest/'selection_predictions.csv',index=False)
            write_json(dest/'selected_metrics.json',dict(epoch=epoch,**m))
        elif epoch>=10: stale+=1
        emit('phase_control_epoch',arm=arm,outer=fold,**row)
        if time.time()-begin>3600: raise TimeoutError('1h run bound')
        if epoch>=10 and stale>=15: break
    write_json(dest/'COMPLETED.json',dict(epochs=epoch,seconds=time.time()-begin,max_single_class_streak=maximum,
       test_used=False,external_used=False,final_model_frozen=False))

def pipeline():
    contract(); assert (REPORT/'CACHE_READY.json').exists()
    target=REPORT/'PIPELINE_STATUS.json'; assert not target.exists(),'No automatic retry'
    status=dict(status='running',pid=os.getpid(),started=time.time(),completed_runs=0,test_used=False,external_used=False)
    write_json(target,status)
    try:
        for fold in range(3):
            jobs=[]
            for gpu,arm in enumerate(['legacy','nominal']):
                stream=(LOG/('%s_outer%d.log'%(arm,fold))).open('x',encoding='utf-8')
                cmd=[sys.executable,'-X','utf8',str(Path(__file__)),'train','--gpu',str(gpu),'--fold',str(fold),'--arm',arm]
                p=subprocess.Popen(cmd,cwd=str(ROOT),stdout=stream,stderr=subprocess.STDOUT)
                jobs.append((p,stream)); emit('training_started',gpu=gpu,arm=arm,fold=fold,pid=p.pid)
            started=time.time()
            while any(p.poll() is None for p,_ in jobs):
                status.update(current_fold=fold,last_check=time.time(),workers=[dict(pid=p.pid,exit_code=p.poll()) for p,_ in jobs])
                write_json(target,status)
                if time.time()-started>3700:
                    emit('hard_timeout_stopping_owned_workers',fold=fold)
                    for p,_ in jobs:
                        if p.poll() is None: p.kill()
                    for p,s in jobs: p.wait(); s.close()
                    raise TimeoutError('paired worker timeout')
                time.sleep(10)
            for p,s in jobs: s.close()
            assert all(p.returncode==0 for p,_ in jobs),'Worker failed; no automatic retry'
            status['completed_runs']+=2; write_json(target,status)
        status.update(status='completed',finished=time.time(),final_model_frozen=False)
        write_json(target,status); emit('training_completed',runs=6)
    except Exception as e:
        status.update(status='failed_needs_review',error=repr(e),finished=time.time()); write_json(target,status); raise

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['setup','prepare','train','run'])
    p.add_argument('--gpu',type=int,default=0); p.add_argument('--fold',type=int,default=0)
    p.add_argument('--arm',choices=['legacy','nominal'],default='legacy'); a=p.parse_args()
    if a.mode=='setup': setup()
    elif a.mode=='prepare': prepare()
    elif a.mode=='train': train(a.gpu,a.fold,a.arm)
    else: pipeline()
