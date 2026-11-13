"""Phase-only handcrafted control; legacy grids/masks fixed, development only.

This control does not release the physically harmonized pipeline. No CNN is
trained, no geometry hold is waived, and no held-out endpoint is evaluated.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time
import warnings
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from nominal_phase_contract import ROOT, load_contract, sha, write_json
from prepare_nominal_phase_inputs import path_for, FIELDS
from pam50_v2_core import manifest, CLASSES, SEED, metrics
from molecular_stage import summarize_values, texture

OUT=ROOT/'reports/phase_matched_baseline_20261019_run01'
RUN=ROOT/'runs/phase_matched_baseline_20261019_run01'
WORK=ROOT/'data/fast_v1'
SHAPE=(64,128,128)


def features(image,mask,pid):
    """Same 76 equations as frozen molecular_stage.patient_features.

    Separate array interface avoids touching the reference mask. Every legacy
    output is checked against all saved development features before fitting.
    """
    out=dict(pid=pid,source='automatic',roi_empty=int(not mask.any()),roi_fraction=float(mask.mean()))
    if mask.any():
        pos=np.where(mask)
        spans=[int(a.max())-int(a.min())+1 for a in pos]
        out.update(zip(('roi_extent0','roi_extent1','roi_extent2'),[v/SHAPE[i] for i,v in enumerate(spans)]))
        out['roi_bbox_fill']=float(mask.sum()/np.prod(spans))
    else:
        out.update({k:0. for k in ['roi_extent0','roi_extent1','roi_extent2','roi_bbox_fill']})
    for i,name in enumerate(('pre','early','late')):
        out.update(summarize_values(image[i][mask],name))
        out.update(texture(image[i],mask,name))
    stable=mask&(image[0]>.05)
    out['dyn_stable_fraction']=float(stable.sum()/max(mask.sum(),1))
    for name,values in [('dyn_early',(image[1][stable]-image[0][stable])/image[0][stable]),
        ('dyn_late',(image[2][stable]-image[0][stable])/image[0][stable]),
        ('dyn_change',image[2][mask]-image[1][mask])]:
        out.update(summarize_values(np.clip(values,-10,10),name))
    return out


def protocol():
    df=manifest(); _,phase=load_contract()
    table=phase[phase.cohort.eq('ispy')].copy()
    assert len(table)==748 and set(table.pid)==set(df.pid)
    OUT.mkdir(exist_ok=False); RUN.mkdir(exist_ok=False)
    sources={str(p.relative_to(ROOT)):sha(p) for p in [
        Path(__file__).resolve(),ROOT/'reports/pam50_v2/protocol.json',ROOT/'reports/pam50_v2/fold_manifest.csv',
        ROOT/'reports/nominal_phase_20261012/phase_contract.json',ROOT/'reports/nominal_phase_20261012/phase_selection.csv',
        ROOT/'scripts/molecular_stage.py',ROOT/'scripts/fast_start.py',ROOT/'scripts/pam50_v2_core.py',
        ROOT/'scripts/prepare_nominal_phase_inputs.py']}
    for fold in range(3):
        for name in ['features.csv','selection_predictions.csv','metrics.json','model.joblib']:
            p=ROOT/('runs/pam50_v2/baseline_outer_%d'%fold)/name
            sources[str(p.relative_to(ROOT))]=sha(p)
    p=dict(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),experiment='phase_only_handcrafted_legacy_grid_control',
        authorization='User requests substantive completion; autonomous in-scope development experiments authorized',
        n=748,folds=3,seed=SEED,classes=CLASSES,arms=['legacy','nominal'],
        phase_rule='0,1,n-1 versus historical phase choice',timing_uncertain=True,
        fixed='same outer-fold automatic masks, native index coarse 64x128x128 grid, precontrast scale, 76 equations, folds and classifier',
        pipeline='fit-only median imputation with indicators, variance filter, standard scaling, balanced multinomial logistic L2 C=1 lbfgs max_iter=3000',
        primary='paired development fold macro one-vs-rest AUC; descriptive only',
        secondary=['macro_F1','balanced_accuracy','class_recalls'],
        no_hyperparameter_search=True,no_extra_seeds=True,no_CNN_training=True,
        legacy_feature_reproduction_tolerance=dict(rtol=1e-11,atol=1e-11),
        legacy_prediction_reproduction_max_abs_tolerance=2e-9,
        comparison_to_CNN='spatial representation differs; not an isolated architecture comparison',
        interpretation='Supplementary phase diagnostic only; not final aligned pipeline, not unbiased nested performance, no significance from dependent folds',
        original_validation_used=False,test_used=False,external_used=False,
        physical_input_released=False,final_model_frozen=False,sources_sha256=sources,
        hard_timeout_s=1800,automatic_retries=0,workers=4)
    write_json(OUT/'protocol.json',p)
    return df,table,p


def main():
    begin=time.time(); df,table,p=protocol()
    write_json(OUT/'STATUS.json',dict(status='running',pid=os.getpid(),started=begin,stage='feature_control'))
    old={f:pd.read_csv(ROOT/('runs/pam50_v2/baseline_outer_%d/features.csv'%f)).set_index('pid') for f in range(3)}
    columns=[c for c in old[0] if c.startswith(('roi_','pre_','early_','late_','dyn_'))]
    assert len(columns)==76
    def one(row):
        pid=row.pid
        cp=WORK/'cache'/(pid+'.npz'); meta=WORK/'cache'/(pid+'.json')
        audit=json.loads(meta.read_text()); scale=audit['intensity_scale']
        with np.load(cp) as z: legacy=z['image'].astype(np.float32)
        assert legacy.shape==(3,*SHAPE) and np.isfinite(legacy).all()
        nominal=legacy.copy(); source=[]
        for j,role in enumerate(FIELDS):
            index=int(getattr(row,role)); path=path_for(pid,index)
            source.append(dict(role=role,index=index,path=str(path),sha256=sha(path)))
            if index==int(getattr(row,['old_pre','old_early','old_late'][j])): continue
            native=nib.load(str(path)).get_fdata(dtype=np.float32)
            assert list(native.shape)==audit['shape_native'] and np.isfinite(native).all()
            a=zoom(np.clip(native/scale,0,5),np.array(SHAPE)/np.array(native.shape),order=1,prefilter=False)
            nominal[j]=a.astype(np.float16).astype(np.float32)
        assert np.array_equal(legacy[0],nominal[0])
        records=[]
        for f in range(3):
            mp=ROOT/('data/pam50_v2/outer_%d/automatic_masks'%f)/(pid+'.npz')
            with np.load(mp) as z: mask=z['mask']>0
            assert mask.shape==SHAPE
            original=features(legacy,mask,pid)
            expected=old[f].loc[pid,columns].to_numpy(dtype=float)
            actual=np.array([original[c] for c in columns])
            assert np.allclose(actual,expected,equal_nan=True,rtol=1e-11,atol=1e-11),('legacy features differ',pid,f)
            new=features(nominal,mask,pid)
            stable_cols=[c for c in columns if c.startswith(('roi_','pre_'))]
            assert np.allclose([new[c] for c in stable_cols],[original[c] for c in stable_cols],equal_nan=True,rtol=0,atol=0)
            records.append(dict(outer=f,**new))
            # Mask provenance is recorded for both fitting and selection patients.
            source.append(dict(outer=f,mask=str(mp),mask_sha256=sha(mp)))
        return records,dict(pid=pid,cache_sha256=sha(cp),cache_metadata_sha256=sha(meta),sources=source,
            legacy_features_reproduced_all_folds=True,precontrast_and_shape_descriptors_exact=True)
    rows=[]; receipts=[]
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i,(values,receipt) in enumerate(pool.map(one,table.itertuples(index=False)),1):
                rows.extend(values); receipts.append(receipt)
                if i%25==0:
                    print('features',i,'/748',round(time.time()-begin,1),flush=True)
                    write_json(OUT/'STATUS.json',dict(status='running',pid=os.getpid(),stage='feature_control',patients=i,started=begin))
                if time.time()-begin>1500: raise TimeoutError('Feature phase bound; no retry')
        feature=pd.DataFrame(rows); feature.to_csv(OUT/'nominal_features.csv',index=False)
        write_json(OUT/'input_receipts.json',receipts)
        import joblib
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.feature_selection import VarianceThreshold
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from threadpoolctl import threadpool_limits
        summary=[]
        for f in range(3):
            for arm in ['legacy','nominal']:
                feats=old[f].reset_index() if arm=='legacy' else feature[feature.outer.eq(f)].drop(columns='outer')
                d=df.merge(feats[['pid']+columns],on='pid',validate='one_to_one')
                fit=d[d.outer.ne(f)]; selection=d[d.outer.eq(f)]
                y=np.array([CLASSES.index(v) for v in fit.label]); ys=np.array([CLASSES.index(v) for v in selection.label])
                model=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('variance',VarianceThreshold()),
                    ('scale',StandardScaler()),('model',LogisticRegression(C=1,solver='lbfgs',class_weight='balanced',max_iter=3000,random_state=SEED))])
                with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
                    warnings.simplefilter('always'); model.fit(fit[columns],y)
                assert not any('converg' in str(w.message).lower() for w in caught), 'Logistic convergence hold'
                prob=model.predict_proba(selection[columns]); assert np.isfinite(prob).all()
                delta=None
                if arm=='legacy':
                    prev=pd.read_csv(ROOT/('runs/pam50_v2/baseline_outer_%d/selection_predictions.csv'%f)).set_index('pid')
                    arr=prev.loc[selection.pid,['prob_'+c for c in CLASSES]].to_numpy()
                    delta=float(np.abs(prob-arr).max()); assert delta<2e-9,delta
                dest=RUN/('%s_outer%d'%(arm,f)); dest.mkdir()
                joblib.dump(dict(pipeline=model,features=columns,classes=CLASSES,protocol_sha256=sha(OUT/'protocol.json')),dest/'model.joblib')
                pred=selection[['pid','label']].copy()
                for j,c in enumerate(CLASSES): pred['prob_'+c]=prob[:,j]
                pred.to_csv(dest/'selection_predictions.csv',index=False)
                m=metrics(ys,prob)
                write_json(dest/'metrics.json',dict(**m,legacy_prediction_max_abs_diff=delta,fit_n=len(fit),selection_n=len(selection)))
                summary.append(dict(arm=arm,outer=f,**m))
                print('fit',arm,f,m['macro_auc'],flush=True)
        means={arm:float(np.mean([r['macro_auc'] for r in summary if r['arm']==arm])) for arm in ['legacy','nominal']}
        paired=[dict(outer=f,delta_macro_auc=next(r['macro_auc'] for r in summary if r['arm']=='nominal' and r['outer']==f)
            -next(r['macro_auc'] for r in summary if r['arm']=='legacy' and r['outer']==f)) for f in range(3)]
        for rel,digest in p['sources_sha256'].items(): assert sha(ROOT/rel)==digest,rel
        write_json(OUT/'results.json',dict(status='completed',fold_results=summary,mean_macro_auc=means,
            paired_deltas=paired,mean_delta_macro_auc=means['nominal']-means['legacy'],
            all_2244_legacy_feature_vectors_reproduced=True,all_precontrast_and_shape_features_identical=True,
            elapsed_s=time.time()-begin,test_used=False,external_used=False,final_model_frozen=False,
            physical_input_released=False,significance_test_performed=False,fold_CI_computed=False,
            interpretation=p['interpretation'],protocol_sha256=sha(OUT/'protocol.json')))
        write_json(OUT/'STATUS.json',dict(status='completed',pid=os.getpid(),finished=time.time(),mean_macro_auc=means))
    except Exception as exc:
        write_json(OUT/'STATUS.json',dict(status='failed_needs_review',pid=os.getpid(),error=repr(exc),finished=time.time()))
        raise


if __name__=='__main__': main()
