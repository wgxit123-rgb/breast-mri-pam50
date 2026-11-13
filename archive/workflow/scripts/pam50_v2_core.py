"""V2 train-only fold contracts, native-index ROI construction and networks."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from fast_start import ROOT,WORK,save_json,emit

OUT=ROOT/'reports/pam50_v2'
RUN=ROOT/'runs/pam50_v2'
DATA=ROOT/'data/pam50_v2'
CLASSES=['Basal','Her2','LumA','LumB']
SEED=20260923
SHAPE=(64,96,96)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def setup():
    from sklearn.model_selection import StratifiedKFold
    if (OUT/'protocol.json').exists(): raise RuntimeError('V2 protocol already exists; refuse overwrite')
    for p in [OUT,RUN,DATA]: p.mkdir(parents=True,exist_ok=True)
    original=pd.read_csv(WORK/'patient_manifest.csv')
    labels=pd.read_csv(ROOT/'reports/pam50_v1/development_manifest.csv')
    df=labels[labels.split.eq('train')&labels.primary_eligible][['pid','label']].copy().reset_index(drop=True)
    assert len(df)==748 and df.pid.is_unique
    assert set(df.pid).issubset(set(original.loc[original.split.eq('train'),'pid']))
    assert set(df.label)==set(CLASSES)
    df['outer']=-1
    for fold,(_,idx) in enumerate(StratifiedKFold(3,shuffle=True,random_state=SEED).split(df,df.label)):
        df.loc[idx,'outer']=fold
    for outer in range(3):
        name='inner_%d'%outer; df[name]=-1
        fit=df[df.outer.ne(outer)]
        for inner,(_,idx) in enumerate(StratifiedKFold(2,shuffle=True,random_state=SEED+outer+1).split(fit,fit.label)):
            df.loc[fit.index[idx],name]=inner
    df.to_csv(OUT/'fold_manifest.csv',index=False)
    sanity=pd.concat([df[df.label.eq(c)].sample(n=8,random_state=SEED+i) for i,c in enumerate(CLASSES)])
    sanity.to_csv(OUT/'sanity_patients.csv',index=False)
    protocol=dict(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),user_authorization='自行安排训练',
        train_patients=748,original_validation_used=False,test_used=False,external_used=False,
        normal_policy='unchanged_v1_excluded_primary_retained_for_later_sensitivity',
        outer_folds=3,inner_segmentation_folds=2,extra_unlabelled_segmentation_patients_used=0,
        segmentation_epochs=36,segmentation_model='existing_coarse_UNet_from_scratch_per_inner_fold',
        segmentation_threshold=.5,segmentation_global_validation_reuse='36 epochs selected in prior pilot; no new validation access',
        roi_shape=SHAPE,roi_margin_fraction_per_side=.2,
        roi_space='native_index_space_not_verified_physical_mm',
        coarse_to_native='adaptive_pool_support_envelope_then_context; unit tests before use',
        empty_auto_policy='full_native_volume_resize_and_flag; never reference-mask fallback',
        sanity=dict(n=32,per_class=8,steps=300,threshold=.95,augmentation=False,dropout=0,reference_ROI=True),
        classifier=dict(architecture='Residual3D_16_32_64_128_GroupNorm',max_epochs=80,
            lr=.0003,weight_decay=.0001,batch=4,accumulation=4,dropout=.2,
            warmup_epochs=5,patience=15,early_stop_count_after_epoch=10,
            checkpoint='highest_macro_AUC_rounded_4dp_then_macro_F1_then_earliest',
            class_weights='fit_partition_only_inverse_frequency',mixed_precision='BF16',
            augmentation='paired_inplane_flips_and_shared_gain_0.95_to_1.05'),
        baseline='same fold/masks 76 descriptors multinomial L2 C=1 fixed; no inner C search',
        gate=dict(mean_delta_auc=.02,positive_folds_at_least=2,
            selected_checkpoint_requires_all_class_recalls_positive=True,
            max_consecutive_single_class_epochs=4,
            seeds=[SEED,SEED+100,SEED+200]),
        scores='development selection scores, not unbiased nested performance; raw probabilities, no test evaluation',
        task_timeout_seconds=14400,original_manifest_sha256=sha(WORK/'patient_manifest.csv'),
        fold_manifest_sha256=sha(OUT/'fold_manifest.csv'),
        source_code_sha256={p.name:sha(p) for p in (ROOT/'scripts').glob('pam50_v2_*.py')},
        reused_code_sha256={n:sha(ROOT/'scripts'/n) for n in ['fast_start.py','molecular_stage.py']})
    save_json(OUT/'protocol.json',protocol)
    emit('v2_setup',patients=len(df),outer_sizes=df.outer.value_counts().to_dict(),sanity=32)

def manifest():
    p=json.loads((OUT/'protocol.json').read_text())
    assert sha(WORK/'patient_manifest.csv')==p['original_manifest_sha256']
    assert sha(OUT/'fold_manifest.csv')==p['fold_manifest_sha256']
    for name,value in p['source_code_sha256'].items():
        assert sha(ROOT/'scripts'/name)==value, 'Code changed after protocol freeze: '+name
    for name,value in p['reused_code_sha256'].items():
        assert sha(ROOT/'scripts'/name)==value, 'Dependency changed: '+name
    df=pd.read_csv(OUT/'fold_manifest.csv'); assert len(df)==748 and df.pid.is_unique
    return df

def bbox(mask,native_shape,margin=.2):
    """Map support of occupied coarse bins to native indices, stop exclusive."""
    native=np.asarray(native_shape,dtype=int)
    if not mask.any(): return np.zeros(3,dtype=int),native.copy(),True
    positions=np.where(mask)
    low=np.array([a.min() for a in positions]); high=np.array([a.max()+1 for a in positions])
    scale=native/np.array(mask.shape)
    low=np.floor(low*scale).astype(int); high=np.ceil(high*scale).astype(int)
    padding=np.ceil((high-low)*margin).astype(int)
    return np.maximum(0,low-padding),np.minimum(native,high+padding),False

def roi_one(task):
    pid,source,outer=task
    import nibabel as nib
    from torch.nn.functional import interpolate
    metadata=pd.read_csv(WORK/'patient_manifest.csv')
    row=metadata[metadata.pid.eq(pid)].iloc[0]
    assert row.split=='train','Attempted access outside original training partition'
    cache_audit=json.loads((WORK/'cache'/(pid+'.json')).read_text())
    native_shape=tuple(cache_audit['shape_native']); scale=float(cache_audit['intensity_scale'])
    if source=='reference':
        mask_image=nib.load(str(WORK/'masks'/(pid+'.nii.gz')))
        mask=np.asarray(mask_image.dataobj)>0
        assert mask.shape==native_shape and mask.any()
        affine=mask_image.affine
    else:
        with np.load(DATA/('outer_%d'%outer)/'automatic_masks'/(pid+'.npz')) as archive: mask=archive['mask']>0
        assert mask.shape==(64,128,128)
        affine=None
    lo,hi,empty=bbox(mask,native_shape)
    slices=tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))
    volumes=[]; source_files=[]
    for field in ['pre','post_early','post_late']:
        path=WORK/'selected_nifti'/pid/('acq_%d.nii.gz'%int(row[field]))
        im=nib.load(str(path)); assert im.shape==native_shape
        if affine is None: affine=im.affine
        assert np.allclose(im.affine,affine)
        arr=im.get_fdata(dtype=np.float32)[slices]
        assert arr.size and np.isfinite(arr).all()
        arr=np.clip(arr/scale,0,5)
        resized=interpolate(torch.from_numpy(arr.copy())[None,None],size=SHAPE,mode='trilinear',align_corners=False)[0,0]
        volumes.append(resized.numpy().astype(np.float16)); source_files.append(str(path))
    image=np.stack(volumes); assert image.shape==(3,*SHAPE) and image.max()>image.min()
    folder=DATA/('reference_roi' if source=='reference' else 'outer_%d/automatic_roi'%outer)
    folder.mkdir(parents=True,exist_ok=True)
    np.save(folder/(pid+'.npy'),image)
    info=dict(pid=pid,source=source,outer=outer,native_shape=native_shape,
        crop_start=lo.tolist(),crop_stop=hi.tolist(),empty_auto=empty,
        output_shape=list(image.shape),scale=scale,phase_files=source_files,physical_mm_verified=False,
        sha256=sha(folder/(pid+'.npy')))
    save_json(folder/(pid+'.json'),info)
    return {k:v for k,v in info.items() if k not in ['phase_files']}

def prepare_roi(source,outer=0,sanity_only=False):
    import concurrent.futures as cf
    torch.set_num_threads(1)
    df=manifest()
    if sanity_only: df=pd.read_csv(OUT/'sanity_patients.csv')
    records=[]
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        for i,result in enumerate(pool.map(roi_one,[(p,source,outer) for p in df.pid]),1):
            records.append(result)
            if i%50==0: emit('v2_roi',source=source,outer=outer,completed=i,total=len(df))
    name='roi_%s_%s'%(source,'sanity' if sanity_only else str(outer))
    pd.DataFrame(records).to_csv(OUT/(name+'.csv'),index=False)
    save_json(OUT/(name+'_READY.json'),dict(patients=len(df),source=source,outer=outer,test_used=False))

class Residual(nn.Module):
    def __init__(self,cin,cout):
        super().__init__()
        self.body=nn.Sequential(nn.Conv3d(cin,cout,3,stride=2,padding=1,bias=False),nn.GroupNorm(4,cout),nn.SiLU(),
            nn.Conv3d(cout,cout,3,padding=1,bias=False),nn.GroupNorm(4,cout))
        self.skip=nn.Conv3d(cin,cout,1,stride=2,bias=False)
        self.activation=nn.SiLU()
    def forward(self,x): return self.activation(self.body(x)+self.skip(x))

def network(channels=3,dropout=.2):
    return nn.Sequential(Residual(channels,16),Residual(16,32),Residual(32,64),Residual(64,128),
        nn.AdaptiveAvgPool3d(1),nn.Flatten(),nn.Dropout(dropout),nn.Linear(128,4))

def metrics(y,prob):
    from sklearn.metrics import roc_auc_score,f1_score,recall_score,accuracy_score,confusion_matrix,log_loss
    pred=prob.argmax(1)
    recalls=recall_score(y,pred,labels=[0,1,2,3],average=None,zero_division=0)
    return dict(macro_auc=float(roc_auc_score(y,prob,multi_class='ovr',average='macro')),
        macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),
        accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(recalls.mean()),
        recalls=recalls.tolist(),prediction_counts=np.bincount(pred,minlength=4).tolist(),
        confusion=confusion_matrix(y,pred,labels=[0,1,2,3]).tolist(),
        log_loss=float(log_loss(y,prob)))

def load_roi(df,source,outer,view):
    folder=DATA/('reference_roi' if source=='reference' else 'outer_%d/automatic_roi'%outer)
    images=np.stack([np.load(folder/(pid+'.npy')) for pid in df.pid])
    if view=='single': images=images[:,1:2].copy()
    return torch.from_numpy(images),torch.tensor([CLASSES.index(v) for v in df.label],dtype=torch.long)

def infer(net,x):
    probabilities=[]; net.eval()
    with torch.inference_mode():
        for batch in x.split(4):
            with torch.autocast('cuda',dtype=torch.bfloat16): logits=net(batch.cuda().float())
            probabilities.append(logits.float().softmax(1).cpu().numpy())
    return np.concatenate(probabilities)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['setup','roi'])
    p.add_argument('--source',choices=['reference','automatic'],default='reference')
    p.add_argument('--outer',type=int,default=0); p.add_argument('--sanity-only',action='store_true')
    args=p.parse_args()
    setup() if args.mode=='setup' else prepare_roi(args.source,args.outer,args.sanity_only)
