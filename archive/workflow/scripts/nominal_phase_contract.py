"""Approved ordinal phase contract. No label or model-score access."""
import hashlib,json,time
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/nominal_phase_20261012'
DATA=ROOT/'data/nominal_phase_20261012'
DECISION='采用“增强前＋首个增强期＋最后增强期”的名义期相方案，并明确报告时间不确定'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def write_json(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
    temp.replace(p)

def select_phases(phase_ids):
    ids=[int(x) for x in phase_ids]
    if len(ids)<3 or len(set(ids))!=len(ids) or sorted(ids)!=list(range(len(ids))):
        raise ValueError('Complete unique zero-based chronological phase inventory with >=3 phases required')
    return (0,1,len(ids)-1)

def setup():
    if (OUT/'phase_contract.json').exists(): raise RuntimeError('Phase contract exists; refuse overwrite')
    OUT.mkdir(exist_ok=True); DATA.mkdir(exist_ok=True)
    source=ROOT/'reports/alignment_20261008/ispy_phase_details.json'
    cases=json.loads(source.read_text(encoding='utf8'))
    membership=pd.read_csv(ROOT/'data/fast_v1/patient_manifest.csv',usecols=['pid','split']).set_index('pid')
    allowed=set(pd.read_csv(ROOT/'reports/pam50_v2/fold_manifest.csv',usecols=['pid']).pid)
    assert len(cases)==748 and {c['pid'] for c in cases}==allowed
    rows=[]
    for c in cases:
        assert membership.loc[c['pid'],'split']=='train'
        assert c['status']=='source_phase_semantics_verified' and c['pre']==0
        n=len(c['estimated_midpoints_s']); pre,first,last=select_phases(range(n))
        rows.append(dict(cohort='ispy',pid=c['pid'],partition='original_train',n_phases=n,
          nominal_pre=pre,first_post=first,last_post=last,old_pre=c['pre'],old_early=c['early'],old_late=c['late'],
          timing_uncertain=True,source_clock='estimated_effective_midpoint_relative_to_assumed_injection',
          measured_injection_available=False,selection_status='USER_APPROVED_RULE_FROZEN',
          baseline_basis='publisher_pre_index_zero_and_official_single_precontrast_convention'))
    tcga_source=ROOT/'reports/alignment_20260924/tcga_alignment_sidecars.json'
    tcga=json.loads(tcga_source.read_text(encoding='utf8')); assert len(tcga)==73
    for c in tcga:
        phases=c['phases']; assert all(phases[i+1]['seconds_since_first']>phases[i]['seconds_since_first'] for i in range(len(phases)-1))
        pre,first,last=select_phases(p['phase_index'] for p in phases)
        rows.append(dict(cohort='tcga',pid=c['pid'],partition='external_technical_queue_not_endpoint_finalized',
          n_phases=len(phases),nominal_pre=pre,first_post=first,last_post=last,
          old_pre=0,old_early=None,old_late=None,timing_uncertain=True,
          source_clock='acquisition_start_order_not_effective_midpoint',measured_injection_available=False,
          selection_status='USER_APPROVED_RULE_FROZEN',baseline_basis='fixed_source_fingerprint_supported_phase_zero_not_measured_injection'))
    table=pd.DataFrame(rows); assert not table[['cohort','pid']].duplicated().any()
    prior=pd.read_csv(ROOT/'reports/alignment_20261008/nominal_phase_PROPOSAL.csv').set_index(['cohort','pid'])
    for r in table.itertuples(index=False):
        q=prior.loc[(r.cohort,r.pid)]
        assert (r.nominal_pre,r.first_post,r.last_post)==(int(q.pre),int(q.early),int(q.late))
    manifest=OUT/'phase_selection.csv'; table.to_csv(manifest,index=False)
    decision=dict(date='2026-10-12',source='explicit_user_message',verbatim=DECISION,
      supersedes='2026-10-08 awaiting nominal-versus-exact timing choice',scope='phase selection only, not model release')
    write_json(OUT/'author_decision.json',decision)
    contract=dict(version='nominal_pre_first_last_v1',created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
      phase_rule_frozen=True,user_approved=True,phase_names=['nominal_pre','first_post','last_post'],
      rule='complete source chronological inventory: indices 0,1,n-1; at least 3 distinct phases',
      missing_policy='hold; no adjacent-phase substitution, interpolation or outcome-driven selection',
      timing_uncertain=True,exact_injection_time_alignment_claimed=False,
      feature_policy='do not divide enhancement by assumed elapsed injection time; no inferred pharmacokinetic rate',
      baseline_caveat='metadata/source-fingerprint designated baseline; actual injection event not independently established',
      final_model_frozen=False,model_input_ready=False,old_weights_reusable_as_final=False,
      development_original_train_n=748,external_technical_n=73,external_endpoint_cohort_finalized=False,
      held_out_test_accessed=False,external_labels_or_predictions_accessed=False,
      selected_manifest_sha256=sha(manifest),author_decision_sha256=sha(OUT/'author_decision.json'),
      selector_code_sha256=sha(Path(__file__)),source_phase_details_sha256=sha(source),tcga_source_sidecars_sha256=sha(tcga_source),
      remaining_gates=['patient-specific selected-phase geometry','shared FOV and ROI contract',
        'controlled development comparisons','final model and evaluation hash freeze','one-shot independent evaluation'])
    write_json(OUT/'phase_contract.json',contract)
    print('FROZEN PHASE RULE ONLY: 748 development + 73 external technical rows; timing uncertainty retained',flush=True)

def load_contract():
    c=json.loads((OUT/'phase_contract.json').read_text(encoding='utf8'))
    assert c['phase_rule_frozen'] and c['user_approved']
    assert sha(Path(__file__))==c['selector_code_sha256'],'selector changed after freeze'
    assert sha(OUT/'phase_selection.csv')==c['selected_manifest_sha256'],'phase manifest changed after freeze'
    assert sha(OUT/'author_decision.json')==c['author_decision_sha256'],'user decision changed'
    return c,pd.read_csv(OUT/'phase_selection.csv')

if __name__=='__main__': setup()
