"""Stage 2: cross-fitted coarse segmentation and molecular prediction baselines.

All results are development/validation results. No test predictions are made.
Features are explicitly voxel-grid handcrafted descriptors, NOT IBSI radiomics.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import random
import time

# Bound CPU library threads across the independent GPU and CPU workers.
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import numpy as np
import pandas as pd
from fast_start import ROOT, WORK, CachedVolumes, save_json, emit

BASE = ROOT / 'runs' / 'molecular_v1'
OUT = ROOT / 'reports' / 'molecular_v1'
TARGETS = ('HER2', 'HR', 'MP')
EPOCHS = 36  # fixed from the prior global-validation pilot; no outer-fold stopping.


def manifest():
    df = pd.read_csv(WORK / 'patient_manifest.csv')
    df = df[df.split.isin(['train','val'])].copy()
    assert df.pid.is_unique and df.split.value_counts().to_dict() == {'train':784,'val':99}
    return df


def setup():
    from sklearn.model_selection import StratifiedKFold
    BASE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df = manifest()
    df['fold'] = -1
    train = df[df.split == 'train']
    strata = train.HR.astype(int).astype(str) + '_' + train.HER2.astype(int).astype(str)
    for fold, (_, idx) in enumerate(StratifiedKFold(5, shuffle=True, random_state=20260923).split(train, strata)):
        df.loc[train.index[idx], 'fold'] = fold
    destination = BASE / 'fold_manifest.csv'
    if destination.exists():
        old = pd.read_csv(destination)
        assert old[['pid','split','fold']].equals(df[['pid','split','fold']].reset_index(drop=True))
    else:
        df.to_csv(destination, index=False)
    save_json(OUT / 'protocol.json', {
        'stage':'development only', 'targets':TARGETS, 'primary_target':'HER2',
        'main_comparison':'multiphase versus early-phase descriptors with identical logistic classifier',
        'train':784,'validation':99,'test_used':False,'fold_seed':20260923,'seg_epochs_fixed':EPOCHS,
        'crossfit':'Each training subject is inferred by a model never trained on that subject. Outer-fold masks do not select checkpoints.',
        'validation_masks':'Mean probabilities across all five models, threshold 0.5.',
        'prior_validation_reuse':'Global validation was already used to choose segmentation pilot epoch; all stage 2 validation scores remain exploratory.',
        'feature_scope':'Handcrafted coarse-grid intensity/texture/relative-enhancement descriptors; not standard PyRadiomics or physical-space morphology.',
        'classifier':'Elastic-net logistic regression; C=[0.01,0.1,1,10], l1_ratio=0.5; training-only 5-fold AUC selection.',
        'uncertainty':'2000 paired stratified patient bootstrap draws on validation; conditional on selected models, not selection-adjusted.',
        'segmentation_failure_policy':'Retain empty predicted masks; zero volume and empty flag, missing image features imputed inside training folds.',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'fold_manifest_sha256':hashlib.sha256(destination.read_bytes()).hexdigest()})
    prev = pd.read_csv(ROOT/'runs/fast_seg_multiphase/best_validation_patients.csv')
    prev.sort_values('dice').to_csv(OUT/'segmentation_failure_audit.csv',index=False)
    geometry = json.loads((ROOT/'reports/fast_v1/geometry_audit.json').read_text())
    rows=[]
    indexed=df.set_index('pid')
    for a in geometry:
        r=indexed.loc[a['pid']]
        nominal=float(a['native_foreground_voxels']*r.slice_thick*r.xy_spacing**2)
        rows.append({'pid':a['pid'],'axis0_matches_n_z':a['shape_native'][0]==int(r.n_z),
                     'header_spacing':str(a['header_spacing']),'csv_nominal_voxel_volume':float(r.slice_thick*r.xy_spacing**2),
                     'nominal_mask_volume':nominal,'metadata_tum_vol':float(r.tum_vol),
                     'nominal_to_metadata_ratio':nominal/float(r.tum_vol) if r.tum_vol else np.nan})
    pd.DataFrame(rows).to_csv(OUT/'geometry_consistency.csv',index=False)
    emit('setup_complete',train=784,val=99,fold_counts=df[df.split=='train'].fold.value_counts().to_dict())


def net3d():
    from monai.networks.nets import UNet
    return UNet(spatial_dims=3,in_channels=3,out_channels=1,channels=(16,32,64,128),strides=(2,2,2),num_res_units=1,norm='INSTANCE')


def train_folds(gpu, folds):
    import torch
    from torch.utils.data import DataLoader
    from monai.losses import DiceCELoss
    torch.cuda.set_device(gpu)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    df=pd.read_csv(BASE/'fold_manifest.csv')
    for fold in folds:
        run=BASE/f'fold_{fold}'
        run.mkdir(parents=True,exist_ok=True)
        if (run/'COMPLETED.json').exists():
            emit('fold_already_complete',fold=fold)
            continue
        if (run/'last.pt').exists():
            raise RuntimeError(f'Partial checkpoint requires explicit resume, not overwrite: {run}')
        try:
            _one_fold(gpu,fold,run,df,torch,DataLoader,DiceCELoss)
        except Exception as exc:
            save_json(run/'FAILED.json',{'fold':fold,'error':repr(exc),'time':time.time()})
            raise


def _one_fold(gpu,fold,run,df,torch,DataLoader,DiceCELoss):
    seed=42+fold
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    train=df[(df.split=='train')&(df.fold!=fold)]
    holdout=df[(df.split=='train')&(df.fold==fold)]
    val=df[df.split=='val']
    assert set(train.pid).isdisjoint(holdout.pid) and set(train.pid).isdisjoint(val.pid)
    config={'fold':fold,'gpu':gpu,'seed':seed,'epochs':EPOCHS,'train_ids':train.pid.tolist(),
            'oof_ids':holdout.pid.tolist(),'val_ids':val.pid.tolist(),'cache_schema':2,
            'checkpoint_rule':'fixed final epoch; no outer-fold early stopping',
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save_json(run/'config.json',config)
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(CachedVolumes(train.pid,3,True),batch_size=2,shuffle=True,num_workers=4,
                      pin_memory=True,persistent_workers=True,generator=generator)
    net=net3d().cuda()
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS)
    loss_fn=DiceCELoss(sigmoid=True)
    history=[]
    started=time.time()
    emit('fold_start',gpu=gpu,fold=fold,n_train=len(train),n_oof=len(holdout))
    for epoch in range(1,EPOCHS+1):
        tick=time.time(); net.train(); losses=[]
        for step,(x,y,_) in enumerate(loader,1):
            x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=loss_fn(net(x),y)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),12); optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if step%150==0: emit('fold_step',fold=fold,epoch=epoch,step=step,loss=float(np.mean(losses)))
        scheduler.step()
        row={'epoch':epoch,'train_loss':float(np.mean(losses)),'seconds':round(time.time()-tick,2)}
        history.append(row); pd.DataFrame(history).to_csv(run/'history.csv',index=False)
        checkpoint={'model':net.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                    'epoch':epoch,'config':config,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(), 'generator':generator.get_state()}
        torch.save(checkpoint,run/'last.tmp'); os.replace(run/'last.tmp',run/'last.pt')
        emit('fold_epoch',gpu=gpu,fold=fold,**row)
        if time.time()-started>12*3600: raise TimeoutError('12h fold budget reached; checkpoint retained')
    del loader
    net.eval(); metrics=[]
    for split,part in [('oof',holdout),('val',val)]:
        outdir=run/split; outdir.mkdir(exist_ok=True)
        predloader=DataLoader(CachedVolumes(part.pid,3),batch_size=2,num_workers=4,pin_memory=True)
        with torch.inference_mode():
            for x,y,pids in predloader:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    prob=net(x.cuda(non_blocking=True)).float().sigmoid().cpu().numpy()[:,0]
                truth=y.numpy()[:,0]>0
                for j,pid in enumerate(pids):
                    mask=prob[j]>.5
                    dice=float(2*np.logical_and(mask,truth[j]).sum()/max(1,mask.sum()+truth[j].sum()))
                    payload={'prob':prob[j].astype(np.float16)} if split=='val' else {'mask':mask.astype(np.uint8)}
                    np.savez_compressed(outdir/(pid+'.npz'),**payload)
                    metrics.append({'pid':pid,'split':split,'fold':fold,'dice':dice,'empty':not bool(mask.any())})
        del predloader
    pd.DataFrame(metrics).to_csv(run/'segmentation_metrics.csv',index=False)
    save_json(run/'COMPLETED.json',{'status':'completed','fold':fold,'n_oof':len(holdout),'n_val':len(val),
                                  'epochs':EPOCHS,'seconds':time.time()-started,'test_used':False})
    emit('fold_complete',fold=fold,seconds=round(time.time()-started,1))
    del checkpoint,net,optimizer,scheduler,loss_fn
    torch.cuda.empty_cache()


def summarize_values(values, prefix):
    names=('mean','std','p10','p25','median','p75','p90','iqr','skew')
    if values.size==0:
        return {prefix+'_'+name:np.nan for name in names}
    q=np.percentile(values,[10,25,50,75,90])
    std=float(values.std()); mean=float(values.mean())
    vals=[mean,std,*q,float(q[3]-q[1]),float(np.mean(((values-mean)/max(std,1e-6))**3))]
    return dict(zip([prefix+'_'+name for name in names],map(float,vals)))


def texture(image,mask,prefix):
    names=('contrast','homogeneity','energy','entropy','correlation')
    if not mask.any(): return {prefix+'_tex_'+n:np.nan for n in names}
    bins=16
    # Fixed bins across subjects, on the precontrast-99.5%-scaled intensity grid.
    quant=np.minimum((np.clip(image,0,5)/5*bins).astype(np.int16),bins-1)
    counts=np.zeros((bins,bins),dtype=np.float64)
    for axis in range(3):
        a=[slice(None)]*3; b=[slice(None)]*3
        a[axis]=slice(None,-1); b[axis]=slice(1,None)
        a,b=tuple(a),tuple(b)
        good=mask[a]&mask[b]
        pairs=quant[a][good]*bins+quant[b][good]
        counts+=np.bincount(pairs,minlength=bins*bins).reshape(bins,bins)
    if counts.sum()==0: return {prefix+'_tex_'+n:np.nan for n in names}
    counts+=counts.T.copy()
    p=counts/counts.sum(); i,j=np.indices(p.shape)
    mi=float((p*i).sum()); mj=float((p*j).sum())
    vi=float((p*(i-mi)**2).sum()); vj=float((p*(j-mj)**2).sum())
    positive=p[p>0]
    vals=[(p*(i-j)**2).sum(),(p/(1+(i-j)**2)).sum(),(p*p).sum(),-(positive*np.log2(positive)).sum(),
          (p*(i-mi)*(j-mj)).sum()/max(np.sqrt(vi*vj),1e-12)]
    return dict(zip([prefix+'_tex_'+n for n in names],map(float,vals)))


def patient_features(args):
    pid,source=args
    with np.load(WORK/'cache'/(pid+'.npz')) as f:
        image=f['image'].astype(np.float32)
        reference=f['mask']>0
    if source=='reference': mask=reference
    else:
        with np.load(BASE/'automatic_masks'/(pid+'.npz')) as f: mask=f['mask']>0
    assert mask.shape==reference.shape
    features={'pid':pid,'source':source,'roi_empty':int(not mask.any()),'roi_fraction':float(mask.mean())}
    if mask.any():
        locations=np.where(mask)
        lengths=[(int(a.max())-int(a.min())+1)/mask.shape[i] for i,a in enumerate(locations)]
        features.update(dict(zip(('roi_extent0','roi_extent1','roi_extent2'),lengths)))
        features['roi_bbox_fill']=float(mask.sum()/np.prod([int(a.max())-int(a.min())+1 for a in locations]))
    else:
        features.update({n:0.0 for n in ['roi_extent0','roi_extent1','roi_extent2','roi_bbox_fill']})
    for idx,name in enumerate(('pre','early','late')):
        features.update(summarize_values(image[idx][mask],name))
        features.update(texture(image[idx],mask,name))
    # Suppress near-zero baseline division, use common per-subject scaling.
    stable=mask&(image[0]>.05)
    features['dyn_stable_fraction']=float(stable.sum()/max(mask.sum(),1))
    for name,values in [('dyn_early',(image[1][stable]-image[0][stable])/image[0][stable]),
                        ('dyn_late',(image[2][stable]-image[0][stable])/image[0][stable]),
                        ('dyn_change',image[2][mask]-image[1][mask])]:
        features.update(summarize_values(np.clip(values,-10,10),name))
    return features


def extract_features(source,workers):
    df=manifest(); rows=[]
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(patient_features,(pid,source)) for pid in df.pid]
        for f in cf.as_completed(futures):
            rows.append(f.result())
            if len(rows)%100==0: emit('features_progress',source=source,completed=len(rows),total=len(df))
    data=pd.DataFrame(rows).sort_values('pid')
    assert data.pid.nunique()==883
    data.to_csv(OUT/f'{source}_features.csv',index=False)
    emit('features_complete',source=source,n_features=len(data.columns)-2)


def fit_classifiers(source):
    import joblib
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,balanced_accuracy_score,confusion_matrix
    from threadpoolctl import threadpool_limits
    df=pd.read_csv(BASE/'fold_manifest.csv')
    features=pd.read_csv(OUT/f'{source}_features.csv')
    df=df.merge(features,on='pid',validate='one_to_one')
    train=df[df.split=='train'].reset_index(drop=True)
    val=df[df.split=='val'].reset_index(drop=True)
    feature_names=[n for n in features if n not in ['pid','source']]
    early=[n for n in feature_names if n.startswith(('roi_','early_'))]
    multi=feature_names
    assert not set(TARGETS).intersection(feature_names)
    folds=[(np.where(train.fold.to_numpy()!=f)[0],np.where(train.fold.to_numpy()==f)[0]) for f in range(5)]
    results=[]; predictions=pd.DataFrame({'pid':val.pid})
    modeldir=BASE/'classifiers'/source; modeldir.mkdir(parents=True,exist_ok=True)
    for target in TARGETS:
        predictions[target]=val[target].astype(int)
        for view,columns in [('single',early),('multi',multi)]:
            pipe=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('variance',VarianceThreshold()),
                           ('scale',StandardScaler()),('model',LogisticRegression(penalty='elasticnet',solver='saga',l1_ratio=.5,
                                  class_weight='balanced',max_iter=5000,tol=1e-3,random_state=42))])
            search=GridSearchCV(pipe,{'model__C':[.01,.1,1,10]},scoring='roc_auc',cv=folds,n_jobs=2,refit=True,error_score='raise',return_train_score=False)
            with threadpool_limits(limits=1): search.fit(train[columns],train[target].astype(int))
            model=search.best_estimator_
            probability=model.predict_proba(val[columns])[:,1]
            predictions[f'{target}_{view}']=probability
            metrics={'source':source,'view':view,'target':target,'validation_n':len(val),'n_features':len(columns),
                     'train_cv_auc':float(search.best_score_),'validation_auc':float(roc_auc_score(val[target],probability)),
                     'validation_pr_auc':float(average_precision_score(val[target],probability)),
                     'validation_brier':float(brier_score_loss(val[target],probability)),
                     'validation_balanced_accuracy_at_0_5':float(balanced_accuracy_score(val[target],probability>=.5)),
                     'C':float(search.best_params_['model__C']),'iterations':int(model.named_steps['model'].n_iter_.max()),
                     'interpretation':'exploratory validation; CV for automatic features is not fully nested end-to-end validation'}
            results.append(metrics)
            joblib.dump({'pipeline':model,'features':columns,'target':target,'view':view,'source':source},modeldir/f'{target}_{view}.joblib')
            pd.DataFrame(search.cv_results_).to_csv(modeldir/f'{target}_{view}_cv.csv',index=False)
            pd.DataFrame(results).to_csv(OUT/f'{source}_metrics.csv',index=False)
            predictions.to_csv(OUT/f'{source}_validation_predictions.csv',index=False)
            emit('classifier_complete',**metrics)
    # Paired differences, conditional validation uncertainty (no test access).
    rng=np.random.default_rng(20260923)
    cis=[]
    for target in TARGETS:
        y=predictions[target].to_numpy(); p0=predictions[target+'_single'].to_numpy(); p1=predictions[target+'_multi'].to_numpy()
        pos=np.where(y==1)[0]; neg=np.where(y==0)[0]; samples=[]
        for _ in range(2000):
            idx=np.r_[rng.choice(pos,len(pos)),rng.choice(neg,len(neg))]
            a=roc_auc_score(y[idx],p0[idx]); b=roc_auc_score(y[idx],p1[idx]); samples.append((a,b,b-a))
        arr=np.array(samples); lo,hi=np.percentile(arr,[2.5,97.5],axis=0)
        for i,name in enumerate(['single_auc','multi_auc','multi_minus_single']):
            estimate=[roc_auc_score(y,p0),roc_auc_score(y,p1),roc_auc_score(y,p1)-roc_auc_score(y,p0)][i]
            cis.append({'source':source,'target':target,'metric':name,'estimate':estimate,'ci_low':lo[i],'ci_high':hi[i],
                        'bootstrap_draws':2000,'scope':'validation; conditional on selected models'})
    pd.DataFrame(cis).to_csv(OUT/f'{source}_validation_bootstrap.csv',index=False)
    save_json(OUT/f'{source}_COMPLETED.json',{'status':'completed','targets':TARGETS,'n_models':6,'test_used':False})


def finalize_auto(wait):
    started=time.time()
    while True:
        for fold in range(5):
            error=BASE/f'fold_{fold}'/'FAILED.json'
            if error.exists(): raise RuntimeError(error.read_text())
        completed=[f for f in range(5) if (BASE/f'fold_{f}'/'COMPLETED.json').exists()]
        if len(completed)==5: break
        if not wait: raise RuntimeError('OOF folds are not all complete')
        if time.time()-started>6*3600: raise TimeoutError('Waiting for folds exceeded 6h')
        emit('awaiting_folds',completed=completed)
        time.sleep(30)
    df=pd.read_csv(BASE/'fold_manifest.csv')
    dest=BASE/'automatic_masks'; dest.mkdir(exist_ok=True)
    provenance=[]
    for r in df.itertuples():
        if r.split=='train':
            conf=json.loads((BASE/f'fold_{r.fold}'/'config.json').read_text())
            assert r.pid not in conf['train_ids'] and r.pid in conf['oof_ids']
            with np.load(BASE/f'fold_{r.fold}'/'oof'/(r.pid+'.npz')) as f: mask=f['mask']
            recipe=f'fold_{r.fold}, out of fold'
        else:
            average=None
            for fold in range(5):
                conf=json.loads((BASE/f'fold_{fold}'/'config.json').read_text())
                assert r.pid not in conf['train_ids']
                with np.load(BASE/f'fold_{fold}'/'val'/(r.pid+'.npz')) as f: p=f['prob'].astype(np.float32)
                average=p/5 if average is None else average+p/5
            mask=(average>.5).astype(np.uint8); recipe='five-model probability mean, threshold=0.5'
        np.savez_compressed(dest/(r.pid+'.npz'),mask=mask)
        provenance.append({'pid':r.pid,'split':r.split,'mask_origin':recipe,'empty':not bool(mask.any())})
    pd.DataFrame(provenance).to_csv(OUT/'automatic_mask_provenance.csv',index=False)
    extract_features('automatic',8)
    fit_classifiers('automatic')
    emit('automatic_molecular_stage_complete')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['setup','folds','reference','automatic'])
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--folds',type=int,nargs='+',default=[0])
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--wait',action='store_true')
    args=parser.parse_args()
    if args.mode=='setup': setup()
    elif args.mode=='folds': train_folds(args.gpu,args.folds)
    elif args.mode=='reference':
        extract_features('reference',args.workers)
        fit_classifiers('reference')
    else: finalize_auto(args.wait)
