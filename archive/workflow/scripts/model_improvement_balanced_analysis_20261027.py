"""One bounded post-failure analysis; never the original six-job collector."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path('E:/\u4e73\u817a')
BASE = Path('reports/model_improvement_balanced_analysis_20261027')
SELF = 'scripts/model_improvement_balanced_analysis_20261027.py'
PYTHON = os.environ.get("BREAST_MRI_ANALYSIS_PYTHON", sys.executable)
CLASSES = ['Basal', 'Her2', 'LumA', 'LumB']
KEYS = ['macro_auc', 'macro_f1', 'accuracy', 'balanced_accuracy', 'log_loss']
ARMS = {'enhancement_balanced':'runs/model_improvement_20261027',
        'physical_full_fov':'runs/exploratory743_20261026',
        'native_nominal_control':'runs/exploratory743_20261026'}
sys.path.insert(0, str(ROOT/'scripts'))
from owned_windows_job_v2 import supervise

def sha(name):
    h=hashlib.sha256()
    with (ROOT/name).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))

def write(name,obj):
    path=ROOT/name
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(obj,f,ensure_ascii=True,indent=2,allow_nan=False)
        f.write('\n')

def now():return datetime.now(timezone.utc).isoformat()

def seal():
    names=[SELF,'scripts/owned_windows_job_v2.py',
        'reports/model_improvement_dualgpu_v3_20261027/QUEUE_STATUS.json',
        'reports/model_improvement_failure_20261027/DISPOSITION.json',
        'reports/exploratory743_final_20261026/RESULTS.json']
    for arm,folder in ARMS.items():
        for fold in range(3):
            stage='%s_outer%d'%(arm,fold)
            names += [folder+'/'+stage+'/'+s for s in
                      ['selection_predictions.csv','selected_metrics.json','COMPLETED.json']]
            if arm=='enhancement_balanced':
                names.append('reports/model_improvement_20261027/jobs/'+stage+'/RESULT.json')
    protected=set(read('reports/model_improvement_failure_20261027/PROTOCOL.json')['protected_sha256'])
    protected.update(names)
    names=sorted(set(names))
    protocol=dict(created_utc=now(),cwd=str(ROOT),
        human_authority='2026-10-27 comprehensive retrospective, outstanding scientific contribution, new plan and autonomous advance',
        scope='Three completed balanced folds, same743 fixed selected predictions versus original physical/native controls; retain weighted2success+1timeout',
        command=[PYTHON,'-X','utf8','-B',SELF,'worker'],hard_timeout_s=180,
        input_sha256={n:sha(n) for n in names},
        protected_sha256={n:sha(n) for n in sorted(protected)},allowed_changes=[],
        targets='Only labels embedded in the nine saved development prediction CSVs; no source/heldout/external labels',
        outcomes=['All5fixed metrics','per-class confusion/recall/F1/counts','paired correctness'],
        tolerance=1e-12,fold_sizes=[247,248,248],class_counts=[318,108,138,179],
        unit='patient, within original selected-development folds; fitting sets overlap',
        exclusions='None within743. The original5 source holds remain outside this separate study',
        inference='Descriptive point estimates only. No bootstrap, p-value, CI or post-hoc threshold/seed/model search',
        success='All inputs/protection copies match; exact same IDs/labels/folds; nine selected metric checks and two prior pooled checks pass; all743 retained; ownedexit0',
        failure='Any mismatch/missing receipt/nonfinite or illegal probabilities -> explicit failure, no retry or output promotion',
        deterministic_scope='CPU NumPy float64 formulas on fixed saved CSVs; not model/training or clinical reproducibility',
        outputs=[str(BASE/x) for x in ['RESULTS.json','COMPARISON.csv','REPORT.md','PRESERVATION_CHECK.json']],
        automatic_retries=0,new_training=False,independent_evaluation=False)
    write(BASE/'PROTOCOL.json',protocol)
    print('SEALED',len(names),'inputs',len(protected),'protections',flush=True)

def run(expected):
    if sha(BASE/'PROTOCOL.json')!=expected:raise ValueError('Protocol changed')
    p=read(BASE/'PROTOCOL.json')
    for name,h in p['input_sha256'].items():
        if sha(name)!=h:raise ValueError('Input changed: '+name)
    for name,h in p['protected_sha256'].items():
        if sha(name)!=h:raise ValueError('Protected changed before copy: '+name)
        dest=ROOT/BASE/'prelaunch'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():raise ValueError('No backup overwrite')
        shutil.copy2(ROOT/name,dest)
        if hashlib.sha256(dest.read_bytes()).hexdigest()!=h:raise ValueError('Backup failed')
    write(BASE/'BACKUP_RECEIPT.json',dict(files=len(p['protected_sha256']),all_copy_hashes_match=True,utc=now()))
    terminal=read('reports/model_improvement_dualgpu_v3_20261027/QUEUE_STATUS.json')
    if terminal['status']!='FAILED_STOPPED_NO_RETRY' or len(terminal['completed'])!=5:
        raise ValueError('Unexpected original terminal queue')
    if terminal['tasks']['enhancement_weighted_outer2']['result']['exit_code']!=124:
        raise ValueError('Weighted timeout missing')
    os.environ['OMP_NUM_THREADS']='1'
    os.environ['OPENBLAS_NUM_THREADS']='1'
    import numpy as np

    def metrics(y,probs):
        pred=probs.argmax(axis=1)
        c=np.zeros((4,4),dtype=np.int64)
        np.add.at(c,(y,pred),1)
        true=c.sum(axis=1);called=c.sum(axis=0);tp=c.diagonal()
        recalls=tp/true
        f1=np.divide(2*tp,true+called,out=np.zeros(4,dtype=float),where=(true+called)>0)
        auc=[]
        for k in range(4):
            pos=probs[y==k,k];neg=probs[y!=k,k]
            if not len(pos) or not len(neg):raise ValueError('Missing binary class')
            pairs=pos[:,None]-neg[None,:]
            auc.append(float((np.sum(pairs>0)+.5*np.sum(pairs==0))/pairs.size))
        clipped=np.clip(probs,np.finfo(float).eps,1-np.finfo(float).eps)
        clipped/=clipped.sum(axis=1,keepdims=True)
        return dict(patients=len(y),macro_auc=float(np.mean(auc)),
            macro_f1=float(np.mean(f1)),accuracy=float(np.mean(pred==y)),
            balanced_accuracy=float(np.mean(recalls)),log_loss=float(-np.log(clipped[np.arange(len(y)),y]).mean()),
            class_auc=auc,recalls=recalls.tolist(),class_f1=f1.tolist(),
            true_class_counts=true.tolist(),prediction_counts=called.tolist(),confusion=c.tolist())

    def compare(observed,saved):
        for key in KEYS:
            if abs(observed[key]-saved[key])>p['tolerance']:
                raise ValueError('Saved metric mismatch '+key)
        for key in ['prediction_counts','confusion']:
            if key in saved and observed[key]!=saved[key]:raise ValueError('Count mismatch '+key)

    arms={};owned={}
    for arm,folder in ARMS.items():
        rows={};fold_results=[]
        for fold in range(3):
            stage='%s_outer%d'%(arm,fold);sub=folder+'/'+stage+'/'
            done=read(sub+'COMPLETED.json')
            if done['stopping_reason']!='fixed_patience' or done.get('internal_test_used',False) or done.get('external_used',False):
                raise ValueError('Unexpected completion scope')
            if arm=='enhancement_balanced':
                receipt=read('reports/model_improvement_20261027/jobs/'+stage+'/RESULT.json')
                if receipt['exit_code']!=0 or receipt['timed_out'] or receipt['owned_active_processes_after_termination']!=0:
                    raise ValueError('Incomplete balanced job')
            part=[]
            with (ROOT/(sub+'selection_predictions.csv')).open(encoding='utf-8',newline='') as f:
                reader=csv.DictReader(f)
                if reader.fieldnames!=['pid','label']+['prob_'+c for c in CLASSES]:raise ValueError('Unexpected column order')
                for row in reader:
                    pid=row['pid']
                    lab=CLASSES.index(row['label']) if row['label'] in CLASSES else int(row['label'])
                    prob=np.array([float(row['prob_'+c]) for c in CLASSES],dtype=float)
                    if lab not in range(4) or not np.isfinite(prob).all() or np.any(prob<0) or np.any(prob>1) or abs(prob.sum()-1)>1e-5:
                        raise ValueError('Illegal probability or label')
                    if pid in rows:raise ValueError('Duplicate patient')
                    rows[pid]=(lab,fold,prob);part.append(pid)
            if len(part)!=p['fold_sizes'][fold]:raise ValueError('Wrong fold size')
            y=np.array([rows[pid][0] for pid in part]);probs=np.stack([rows[pid][2] for pid in part])
            values=metrics(y,probs);compare(values,read(sub+'selected_metrics.json'))
            fold_results.append(dict(fold=fold,**values))
        ids=sorted(rows)
        if len(ids)!=743:raise ValueError('Wrong patient denominator')
        y=np.array([rows[i][0] for i in ids]);probs=np.stack([rows[i][2] for i in ids])
        pooled=metrics(y,probs)
        if pooled['true_class_counts']!=p['class_counts']:raise ValueError('Class counts changed')
        arms[arm]=dict(folds=fold_results,pooled_selected_development=pooled)
        owned[arm]=rows
    first=owned['enhancement_balanced']
    for arm,rows in owned.items():
        if set(rows)!=set(first):raise ValueError('Patient sets differ')
        if any(rows[i][:2]!=first[i][:2] for i in rows):raise ValueError('Labels/folds differ')
    old=read('reports/exploratory743_final_20261026/RESULTS.json')['pooled_selection_assisted_metrics']
    for arm in ['physical_full_fov','native_nominal_control']:
        compare(arms[arm]['pooled_selected_development'],old[arm])
    contrasts={}
    for arm in ['physical_full_fov','native_nominal_control']:
        good_new=np.array([np.argmax(first[i][2])==first[i][0] for i in sorted(first)])
        good_old=np.array([np.argmax(owned[arm][i][2])==first[i][0] for i in sorted(first)])
        contrasts[arm]=dict(delta_balanced_minus_control={k:arms['enhancement_balanced']['pooled_selected_development'][k]-arms[arm]['pooled_selected_development'][k] for k in KEYS},
            paired_correctness=dict(both_correct=int(np.sum(good_new&good_old)),only_balanced=int(np.sum(good_new&~good_old)),
                only_control=int(np.sum(~good_new&good_old)),both_wrong=int(np.sum(~good_new&~good_old))))
    result=dict(status='COMPLETE_BALANCED_ARM_DESCRIPTIVE_ANALYSIS_ORIGINAL_ROUND_INCOMPLETE',utc=now(),
        protocol_sha256=expected,python=sys.version,numpy=np.__version__,arms=arms,contrasts=contrasts,
        original_round=dict(planned=6,successful=5,failed=1,failed_stage='enhancement_weighted_outer2',failure_preserved=True,all_six_complete=False),
        patients=743,missing_or_dropped_predictions=0,selection_assistance=True,
        statistical_test_performed=False,CI_computed=False,new_bootstrap=False,
        independent_validation=False,new_training=False,
        scope='Adaptive same-patient selected-development summaries; pooled AUC differs from mean fold AUC; no superiority inference',
        weighted_partial_probabilities_consumed=False)
    write(BASE/'RESULTS.json',result)
    with (ROOT/BASE/'COMPARISON.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['arm','scope','patients']+KEYS)
        w.writeheader()
        for arm,a in arms.items():
            for label,m in [('fold'+str(q['fold']),q) for q in a['folds']]+[('pooled_selected_development',a['pooled_selected_development'])]:
                w.writerow(dict(arm=arm,scope=label,patients=m['patients'],**{k:m[k] for k in KEYS}))
    report=['# Complete balanced-arm descriptive analysis','',
        'Original six-job round remains INCOMPLETE:5successes plus1weighted_outer2 hard timeout. This separate report does not invoke or replace the all-success collector.',
        '', '| Arm | Pooled selected macro-AUC | Macro-F1 | Balanced accuracy | Accuracy | Log loss |',
        '|---|---:|---:|---:|---:|---:|']
    for arm,a in arms.items():
        m=a['pooled_selected_development']
        report.append('| '+arm+' | '+' | '.join('%.6f'%m[k] for k in ['macro_auc','macro_f1','balanced_accuracy','accuracy','log_loss'])+' |')
    report += ['', 'All743same IDs/embedded development labels/folds align. No dropped predictions. All nine fold selected metrics and both prior pooled controls matched within1e-12; counts exact.',
        '', 'No confidence interval, resampling, p-value, significance, clinical utility or independent generalization claim is made. This analysis follows visibility of selected-fold results. Conditional point estimates do not remove adaptive model selection, historical access, source selection or overlapping fitting-set dependence.',
        '', 'Only saved development CSVs including embedded labels were consumed. No image/expression/checkpoint tensor/heldout endpoint or new model inference. All old failures and original748/source743/pending5/sixfalse remain.']
    with (ROOT/BASE/'REPORT.md').open('x',encoding='utf-8') as f:f.write('\n'.join(report)+'\n')
    changes=[name for name,h in p['protected_sha256'].items() if sha(name)!=h]
    if changes:raise ValueError('Protected artifacts changed: '+repr(changes))
    write(BASE/'PRESERVATION_CHECK.json',dict(status='PASS_NAMED_INPUT_PAIRING_METRIC_AND_PROTECTION_SCOPE',
        protected_unchanged=len(p['protected_sha256']),changes=[],fold_metric_sets_checked=9,
        prior_pooled_controls_checked=2,tolerance=1e-12,no_full_engineering_or_clinical_certification=True))
    print('ANALYSIS_COMPLETE',json.dumps(arms['enhancement_balanced']['pooled_selected_development']),flush=True)

if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='seal':seal()
    elif mode=='worker':run(json.loads(sys.stdin.readline(32769))['protocol_sha256'])
    elif mode=='run':
        result=supervise([PYTHON,'-X','utf8','-B',SELF,'worker'],
            dict(protocol_sha256=sha(BASE/'PROTOCOL.json')),ROOT,BASE/'jobs/analyze',180)
        print(json.dumps(result),flush=True)
        sys.exit(0 if result.get('exit_code')==0 and not result.get('timed_out') and result.get('owned_active_processes_after_termination')==0 else 1)
    else:raise ValueError('Unknown mode')
