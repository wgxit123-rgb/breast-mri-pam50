"""V2 learning gate, isolated segmentation training, ROI classifiers and controls."""
import argparse
import json
import math
import random
import time
import numpy as np
import pandas as pd
import torch
from pam50_v2_core import OUT,RUN,DATA,ROOT,WORK,SEED,CLASSES,manifest,network,load_roi,infer,metrics,save_json,emit,sha

def seed_everything(gpu,seed):
    torch.cuda.set_device(gpu); torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def sanity(gpu):
    seed_everything(gpu,SEED); manifest()
    assert (OUT/'roi_reference_sanity_READY.json').exists()
    dest=RUN/'sanity'
    if dest.exists(): raise RuntimeError('Do not overwrite previous learning check')
    dest.mkdir()
    df=pd.read_csv(OUT/'sanity_patients.csv'); assert len(df)==32
    x,y=load_roi(df,'reference',0,'multi')
    net=network(3,0).cuda()
    optimizer=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=0)
    loss_fn=torch.nn.CrossEntropyLoss()
    history=[]; begin=time.time(); passed=False
    generator=torch.Generator().manual_seed(SEED)
    initial=[p.detach().cpu().clone() for p in net.parameters()]
    for step in range(1,301):
        net.train(); optimizer.zero_grad(set_to_none=True); losses=[]
        order=torch.randperm(32,generator=generator)
        for idx in order.split(4):
            with torch.autocast('cuda',dtype=torch.bfloat16): loss=loss_fn(net(x[idx].cuda().float()),y[idx].cuda())
            assert torch.isfinite(loss)
            (loss/8).backward(); losses.append(float(loss.detach().cpu()))
        gradient=float(torch.nn.utils.clip_grad_norm_(net.parameters(),5))
        assert np.isfinite(gradient) and gradient>0
        optimizer.step()
        if step==1 or step%10==0:
            prob=infer(net,x); m=metrics(y.numpy(),prob)
            row=dict(step=step,loss=float(np.mean(losses)),accuracy=m['accuracy'],gradient_norm=gradient,seconds=time.time()-begin)
            history.append(row); pd.DataFrame(history).to_csv(dest/'history.csv',index=False)
            emit('v2_sanity',**row)
            if m['accuracy']>=.95:
                passed=True; break
        if time.time()-begin>3600: raise TimeoutError('Learning check exceeded 1h')
    changed=max(float((p.detach().cpu()-old).abs().max()) for p,old in zip(net.parameters(),initial))
    assert changed>0
    torch.save({'model':net.state_dict(),'step':step},dest/'last.pt')
    save_json(dest/'result.json',dict(passed=passed,accuracy=m['accuracy'],steps=step,
        parameter_change=changed,patients=df.pid.tolist(),seconds=time.time()-begin,
        scope='training-only reference ROI memorization, not generalization',test_used=False))
    if not passed: raise RuntimeError('Learning gate failed: no full training authorized by pipeline gate')

def require_gate():
    assert json.loads((RUN/'sanity/result.json').read_text())['passed'], 'Learning gate not passed'

def segmentation(gpu,outer,inner):
    require_gate(); df=manifest(); seed_everything(gpu,SEED+outer*2+inner)
    import molecular_stage as previous
    from torch.utils.data import DataLoader
    from monai.losses import DiceCELoss
    part=df.copy(); part['split']=np.where(part.outer.eq(outer),'val','train')
    part['fold']=part['inner_%d'%outer]
    train=set(part.loc[part.split.eq('train')&part.fold.ne(inner),'pid'])
    inner_hold=set(part.loc[part.split.eq('train')&part.fold.eq(inner),'pid'])
    outer_hold=set(part.loc[part.split.eq('val'),'pid'])
    assert train.isdisjoint(inner_hold|outer_hold) and inner_hold.isdisjoint(outer_hold)
    dest=RUN/('outer_%d/seg_inner_%d'%(outer,inner))
    if dest.exists(): raise RuntimeError('Existing segmentation run requires explicit resume: '+str(dest))
    dest.mkdir(parents=True)
    save_json(dest/'isolation_contract.json',dict(outer=outer,inner=inner,train_ids=sorted(train),
        inner_holdout_ids=sorted(inner_hold),outer_holdout_ids=sorted(outer_hold),
        original_val_used=False,test_used=False,fold_manifest_sha256=sha(OUT/'fold_manifest.csv')))
    previous.EPOCHS=36
    # Reuse tested coarse UNet training code, but supply isolated V2 patients
    # and a new output directory. Nothing reads the old fold model weights.
    previous._one_fold(gpu,inner,dest,part,torch,DataLoader,DiceCELoss)

def assemble_masks(outer):
    require_gate(); df=manifest(); dest=DATA/('outer_%d'%outer)/'automatic_masks'; dest.mkdir(parents=True,exist_ok=True)
    models=[RUN/('outer_%d/seg_inner_%d'%(outer,i)) for i in range(2)]
    configs=[]
    for folder in models:
        assert (folder/'COMPLETED.json').exists()
        c=json.loads((folder/'config.json').read_text())
        assert set(c['train_ids']).isdisjoint(set(df.loc[df.outer.eq(outer),'pid']))
        configs.append(c)
    rows=[]
    for r in df.itertuples(index=False):
        if r.outer==outer:
            values=[]
            for folder,c in zip(models,configs):
                assert r.pid not in c['train_ids'] and r.pid in c['val_ids']
                with np.load(folder/'val'/(r.pid+'.npz')) as f: values.append(f['prob'].astype(np.float32))
            mask=(sum(values)/2>.5).astype(np.uint8); recipe='mean_of_two_outer_excluding_models'
        else:
            inner=int(getattr(r,'inner_%d'%outer)); c=configs[inner]
            assert r.pid not in c['train_ids'] and r.pid in c['oof_ids']
            with np.load(models[inner]/'oof'/(r.pid+'.npz')) as f: mask=f['mask']
            recipe='inner_%d_oof'%inner
        np.savez_compressed(dest/(r.pid+'.npz'),mask=mask)
        rows.append(dict(pid=r.pid,outer=outer,role='selection' if r.outer==outer else 'fit',
            mask_recipe=recipe,empty=not bool(mask.any()),own_patient_excluded_from_seg_fit=True,
            outer_selection_patients_excluded_from_all_seg_fit=True))
    pd.DataFrame(rows).to_csv(OUT/('mask_provenance_outer_%d.csv'%outer),index=False)
    emit('v2_masks_complete',outer=outer,patients=len(rows),empty=sum(r['empty'] for r in rows))

def classify(gpu,outer,source,view,seed):
    require_gate(); seed_everything(gpu,seed); df=manifest()
    fit=df[df.outer.ne(outer)]; selection=df[df.outer.eq(outer)]
    name='%s_%s_outer%d_seed%d'%(source,view,outer,seed)
    dest=RUN/'classifiers'/name
    if dest.exists(): raise RuntimeError('Do not overwrite classifier: '+name)
    dest.mkdir(parents=True)
    x,y=load_roi(fit,source,outer,view); xv,yv=load_roi(selection,source,outer,view)
    net=network(1 if view=='single' else 3).cuda()
    counts=np.bincount(y.numpy(),minlength=4)
    weights=torch.tensor(len(y)/(4*counts),device='cuda',dtype=torch.float32)
    loss_fn=torch.nn.CrossEntropyLoss(weight=weights)
    optimizer=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.0001)
    generator=torch.Generator().manual_seed(seed); history=[]; best=None; stale=0; consecutive_single=0; max_single=0
    begin=time.time()
    save_json(dest/'config.json',dict(source=source,view=view,outer=outer,seed=seed,gpu=gpu,
        fit_ids=fit.pid.tolist(),selection_ids=selection.pid.tolist(),classes=CLASSES,
        protocol_sha256=sha(OUT/'protocol.json'),test_used=False))
    for epoch in range(1,81):
        tick=time.time(); factor=epoch/5 if epoch<=5 else .5*(1+math.cos(math.pi*(epoch-5)/75))
        for group in optimizer.param_groups: group['lr']=.0003*factor
        net.train(); order=torch.randperm(len(fit),generator=generator); blocks=list(order.split(4))
        losses=[]; grad_norm=[]; optim_steps=0
        # Correctly normalize the last partial accumulation group.
        for start in range(0,len(blocks),4):
            group=blocks[start:start+4]; total=sum(len(index) for index in group)
            optimizer.zero_grad(set_to_none=True)
            for index in group:
                xb=x[index].cuda().float(); yb=y[index].cuda()
                for axis in [3,4]:
                    if random.random()<.5: xb=torch.flip(xb,[axis])
                xb=(xb*random.uniform(.95,1.05)).clamp_(0,5)
                with torch.autocast('cuda',dtype=torch.bfloat16): loss=loss_fn(net(xb),yb)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite classification loss')
                (loss*(len(index)/total)).backward(); losses.append(float(loss.detach().cpu()))
            norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),5)); assert np.isfinite(norm)
            grad_norm.append(norm); optimizer.step(); optim_steps+=1
        p=infer(net,xv); m=metrics(yv.numpy(),p)
        single=sum(v>0 for v in m['prediction_counts'])==1
        consecutive_single=consecutive_single+1 if single else 0; max_single=max(max_single,consecutive_single)
        row=dict(epoch=epoch,train_loss=float(np.mean(losses)),macro_auc=m['macro_auc'],macro_f1=m['macro_f1'],
            selection_log_loss=m['log_loss'],recalls=json.dumps(m['recalls']),prediction_counts=json.dumps(m['prediction_counts']),
            samples=len(fit),microbatches=len(blocks),optimizer_steps=optim_steps,gradient_norm_mean=float(np.mean(grad_norm)),
            single_class_streak=consecutive_single,seconds=round(time.time()-tick,3))
        history.append(row); pd.DataFrame(history).to_csv(dest/'history.csv',index=False)
        score=(round(m['macro_auc'],4),m['macro_f1'])
        checkpoint=dict(model=net.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,view=view,
            source=source,outer=outer,seed=seed,classes=CLASSES)
        torch.save(checkpoint,dest/'last.tmp'); os_replace(dest/'last.tmp',dest/'last.pt')
        if best is None or score>best:
            best=score; stale=0; torch.save(checkpoint,dest/'best.pt')
            pred=selection[['pid','label']].copy()
            for i,c in enumerate(CLASSES): pred['prob_'+c]=p[:,i]
            pred.to_csv(dest/'selection_predictions.csv',index=False)
            save_json(dest/'selected_metrics.json',dict(epoch=epoch,**m))
        elif epoch>=10: stale+=1
        emit('v2_classification_epoch',name=name,**row)
        if consecutive_single==5: emit('ADVISORY_single_class_predictions',name=name,epoch=epoch)
        if time.time()-begin>14400: raise TimeoutError('4h classifier limit; checkpoint retained')
        if epoch>=10 and stale>=15: break
    save_json(dest/'COMPLETED.json',dict(epochs=epoch,seconds=time.time()-begin,max_single_class_streak=max_single,
        original_validation_used=False,test_used=False,interpretation='development checkpoint selection scores'))

def os_replace(a,b):
    import os
    os.replace(a,b)

def baseline(outer):
    require_gate(); df=manifest()
    import concurrent.futures as cf
    import molecular_stage as previous
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    previous.BASE=DATA/('outer_%d'%outer)
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        feature=pd.DataFrame(pool.map(previous.patient_features,[(p,'automatic') for p in df.pid]))
    columns=[c for c in feature if c.startswith(('roi_','pre_','early_','late_','dyn_'))]
    assert len(columns)==76
    combined=df.merge(feature,on='pid',validate='one_to_one')
    train=combined[combined.outer.ne(outer)]; val=combined[combined.outer.eq(outer)]
    model=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('variance',VarianceThreshold()),
        ('scale',StandardScaler()),('model',LogisticRegression(C=1,solver='lbfgs',class_weight='balanced',max_iter=3000,random_state=SEED))])
    y=np.array([CLASSES.index(v) for v in train.label]); yv=np.array([CLASSES.index(v) for v in val.label])
    model.fit(train[columns],y); prob=model.predict_proba(val[columns])
    dest=RUN/('baseline_outer_%d'%outer); dest.mkdir(exist_ok=False)
    joblib.dump(dict(pipeline=model,features=columns,classes=CLASSES),dest/'model.joblib')
    feature.to_csv(dest/'features.csv',index=False)
    pred=val[['pid','label']].copy()
    for i,c in enumerate(CLASSES): pred['prob_'+c]=prob[:,i]
    pred.to_csv(dest/'selection_predictions.csv',index=False)
    save_json(dest/'metrics.json',metrics(yv,prob))
    save_json(dest/'COMPLETED.json',dict(test_used=False,C=1,train=len(train),selection=len(val)))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['sanity','segmentation','assemble','classify','baseline'])
    p.add_argument('--gpu',type=int,default=0); p.add_argument('--outer',type=int,default=0); p.add_argument('--inner',type=int,default=0)
    p.add_argument('--source',default='automatic',choices=['automatic','reference']); p.add_argument('--view',default='multi',choices=['single','multi'])
    p.add_argument('--seed',type=int,default=SEED); a=p.parse_args()
    if a.mode=='sanity': sanity(a.gpu)
    elif a.mode=='segmentation': segmentation(a.gpu,a.outer,a.inner)
    elif a.mode=='assemble': assemble_masks(a.outer)
    elif a.mode=='baseline': baseline(a.outer)
    else: classify(a.gpu,a.outer,a.source,a.view,a.seed)
