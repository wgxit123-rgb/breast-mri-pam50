"""Predeclared exploratory imaging--published-signature associations in dev only.

Does not use TCGA, validation/test, classifier probabilities, new gene-set search,
or a molecular score as an imaging classifier input. Not a mechanism experiment.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import argparse
import csv
import gzip
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata, t
from nominal_phase_contract import ROOT,sha,write_json
from pam50_v2_core import manifest

OUT=ROOT/'reports/molecular_association_v2_20261019'
BASE=ROOT/'reports/phase_matched_baseline_precision_v2_20261019'
IMAGE=['dyn_early_std','dyn_late_std','late_tex_entropy']
SCORES=['Module11_Prolif_score','Module5_TcellBcell_score','Mod10_ECM']
SEED=20261019


def duplicate_platform_join_id():
    value = os.environ.get("BREAST_MRI_DUPLICATE_PLATFORM_JOIN_ID", "").strip()
    if not value.isascii() or not value.isdecimal():
        raise ValueError("Set BREAST_MRI_DUPLICATE_PLATFORM_JOIN_ID from the approved analysis records.")
    return value


def prepare():
    df=manifest(); OUT.mkdir(exist_ok=False)
    sources=[Path(__file__).resolve(),ROOT/'reports/pam50_v2/fold_manifest.csv',
        ROOT/'reports/molecular_linkage/Wolf2022_TableS2_parsed.csv',
        ROOT/'reports/molecular_linkage/GSE194040_sample_metadata.csv',
        ROOT/'data/molecular_audit/Wolf2022_TableS2.xlsx',
        ROOT/'data/molecular_audit/GSE194040_geneLevel_n988.txt.gz',BASE/'protocol.json']
    for f in range(3):sources.append(ROOT/('runs/pam50_v2/baseline_outer_%d/features.csv'%f))
    p=dict(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),type='exploratory_not_prospectively_registered',
        images=IMAGE,molecular_scores=SCORES,primary_tests=9,
        primary_input='nominal-phase handcrafted descriptors from each patient own outer selection fold automatic mask',
        population='748 original training four-class PAM50 patients, no selection by result',
        duplicate_policy='Exclude the confirmed duplicate-platform case from association analysis only: two array platforms; no duplicate samples as patients. PAM50 prediction cohort unchanged.',
        missing_policy='Pairwise complete cases for image/score/covariates; no imputation or outlier trimming; count all exclusions',
        covariates=['PAM50 category','GEO platform category','outer fold category','rank automatic-mask ROI fraction'],
        estimand='partial rank correlation conditional on subtype, platform, fold and automatic support size; descriptive not causal',
        primary_p='OLS standardized ranked score on standardized ranked image + covariates; HC3 robust variance; two-sided t with n-rank(X) df, approximate',
        multiplicity='Benjamini-Hochberg across exactly nine adjusted nominal-phase primary p-values; q<0.05 exploratory only',
        interval='1000 patient bootstraps stratified by PAM50, re-rank and re-fit each draw; percentile 95% interval for partial rank correlation; conditional on fixed feature pipeline',
        descriptive='raw rank correlations and within-subtype correlations, no extra inferential p-values; legacy-phase partial-rho sensitivity without new significance claims',
        seed=SEED,bootstrap=1000,source_sha256={str(x.relative_to(ROOT)):sha(x) for x in sources},
        external_used=False,test_used=False,original_validation_used=False,classifier_probabilities_used=False,
        molecular_values_as_classifier_input=False,mechanistic_validation_claimed=False,
        caveats=['PAM50 and published scores share underlying expression source, not independent assays',
            'GEO expression is author-normalized and batch-adjusted across the original resource, not fold-refit',
            'Subtype adjustment is descriptive and may remove shared biology; not a causal adjustment set',
            'Automatic ROI size is not pathological tumor volume',
            'Source geometry and historical masks remain development-only holds',
            'No age/site/grade adjustment where reliable harmonized fields are unavailable',
            'Published signatures are expression scores, not IHC percentages or protein pathway activation'])
    write_json(OUT/'protocol.json',p)
    print('prepared nine exploratory tests before association calculation',flush=True)


def standardized_rank(a):
    a=rankdata(np.asarray(a,dtype=float),method='average')
    scale=a.std()
    assert scale>0
    return (a-a.mean())/scale


def design(d):
    cat=pd.get_dummies(d[['label','platform','outer']].astype(str),drop_first=True,dtype=float)
    z=np.column_stack([np.ones(len(d)),cat.to_numpy(),standardized_rank(d.roi_fraction)])
    assert np.linalg.matrix_rank(z)==z.shape[1]
    return z


def partial(d,feature,score,inference=True):
    z=design(d); x=standardized_rank(d[feature]); y=standardized_rank(d[score])
    xr=x-z@np.linalg.lstsq(z,x,rcond=None)[0]
    yr=y-z@np.linalg.lstsq(z,y,rcond=None)[0]
    rho=float(np.corrcoef(xr,yr)[0,1])
    if not inference:return rho
    X=np.column_stack([z,x]); pinv=np.linalg.pinv(X); beta=pinv@y
    e=y-X@beta; leverage=np.sum(X*pinv.T,axis=1)
    assert np.max(leverage)<.99
    cov=(pinv*((e/(1-leverage))**2))@pinv.T
    se=float(np.sqrt(cov[-1,-1])); dof=len(d)-np.linalg.matrix_rank(X)
    statistic=float(beta[-1]/se); p=float(2*t.sf(abs(statistic),dof))
    return dict(partial_rank_rho=rho,rank_beta=float(beta[-1]),rank_beta_HC3_SE=se,
        HC3_t=statistic,p_approx_HC3=p,degrees_freedom=int(dof),max_leverage=float(leverage.max()),
        raw_rank_rho=float(np.corrcoef(x,y)[0,1]))


def bh(p):
    p=np.asarray(p); order=np.argsort(p); vals=p[order]*len(p)/np.arange(1,len(p)+1)
    vals=np.minimum.accumulate(vals[::-1])[::-1].clip(0,1)
    ans=np.empty(len(p));ans[order]=vals;return ans


def self_test():
    rng=np.random.default_rng(12);n=500
    d=pd.DataFrame(dict(label=np.tile(['Basal','Her2','LumA','LumB'],125),
        platform=np.tile(['a','b'],250),outer=np.arange(n)%3,roi_fraction=rng.uniform(size=n),
        x=rng.normal(size=n),y=rng.normal(size=n)))
    # Platform above is collinear with subtype; verify fail-closed design first.
    try: design(d)
    except AssertionError: pass
    else: raise AssertionError('Collinear covariate design was accepted')
    d['platform']=rng.choice(['a','b'],n)
    d['y']=d.x*0.5+rng.normal(size=n)
    r=partial(d,'x','y');assert .25<r['partial_rank_rho']<.65 and r['p_approx_HC3']<.001
    assert np.allclose(bh([.01,.04,.03]),[.03,.04,.04])
    assert abs(partial(d,'x','y',False)-partial(d,'y','x',False))<1e-12
    return dict(collinearity_rejected=True,known_positive_detected=True,BH_example_exact=True,rho_symmetric=True)


def analyze():
    p=json.loads((OUT/'protocol.json').read_text()); assert not (OUT/'results.json').exists()
    for rel,h in p['source_sha256'].items():assert sha(ROOT/rel)==h,rel
    ready=json.loads((BASE/'results.json').read_text());assert ready['status']=='completed'
    started=time.time(); df=manifest(); df['join_id']=df.pid.str.extract(r'(\d+)$')[0]
    geo=pd.read_csv(ROOT/'reports/molecular_linkage/GSE194040_sample_metadata.csv',dtype={'patient id':str})
    assert geo.tissue.eq('Breast cancer biopsy (pre-treatment)').all()
    counts=geo.groupby('patient id').size()
    unique=geo[geo['patient id'].map(counts).eq(1)][['patient id','platform_id']].rename(columns={'patient id':'join_id','platform_id':'platform'})
    scores=pd.read_csv(ROOT/'reports/molecular_linkage/Wolf2022_TableS2_parsed.csv',dtype={'Patient Identifier':str})
    d=df.merge(unique,on='join_id',how='left',validate='one_to_one').merge(
        scores[['Patient Identifier']+SCORES],left_on='join_id',right_on='Patient Identifier',how='left',validate='one_to_one')
    duplicate_id = duplicate_platform_join_id()
    assert d.loc[d.platform.isna(),'join_id'].tolist()==[duplicate_id]
    nominal=pd.read_csv(BASE/'nominal_features.csv',float_precision='round_trip')
    own=nominal.merge(df[['pid','outer']],on=['pid','outer'],validate='one_to_one')
    assert len(own)==748
    d=d.merge(own[['pid','roi_fraction']+IMAGE],on='pid',validate='one_to_one')
    d.to_csv(OUT/'development_association_input.csv',index=False)
    # Expression availability audit: parse header and gene names; no gene values
    # or held-out patients' molecular outcomes are persisted or analyzed here.
    expression=ROOT/'data/molecular_audit/GSE194040_geneLevel_n988.txt.gz'
    genes=set(); width=0
    with gzip.open(expression,'rt') as stream:
        reader=csv.reader(stream,delimiter='\t'); header=next(reader); width=len(header)
        ids=[str(x).split('_')[0].split('-')[0].split('.')[0] for x in header]
        for row in reader:
            assert len(row)==width+1 and row[0] not in genes
            genes.add(row[0])
    assert set(df.join_id).issubset(set(ids))
    availability=dict(development_n=748,expression_matched_development_n=748,
        expression_gene_n=len(genes),resource_sample_columns=width,full_gzip_CRC_pass=True,
        clinical_Ki67_assay_available=False,MKI67_present='MKI67' in genes,
        duplicate_platform_development_patients=[duplicate_id],
        nominal_feature_sha256=sha(BASE/'nominal_features.csv'),
        published_score_missing={c:int(d[c].isna().sum()) for c in SCORES},
        external_expression_accessed=False)
    write_json(OUT/'availability.json',availability)
    checks=self_test(); results=[]
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        for i,feature in enumerate(IMAGE):
            for j,score in enumerate(SCORES):
                keep=d[['pid','label','platform','outer','roi_fraction',feature,score]].replace([np.inf,-np.inf],np.nan).dropna()
                assert keep.pid.is_unique
                stats=partial(keep,feature,score)
                groups=[np.flatnonzero(keep.label.to_numpy()==v) for v in sorted(keep.label.unique())]
                rng=np.random.default_rng(SEED+i*3+j); draws=[]
                for b in range(1000):
                    ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
                    draws.append(partial(keep.iloc[ix],feature,score,False))
                row=dict(image_feature=feature,molecular_score=score,n=len(keep),
                    excluded_n=748-len(keep),**stats,
                    rho_CI95_low=float(np.percentile(draws,2.5)),rho_CI95_high=float(np.percentile(draws,97.5)))
                row['by_subtype']={str(k):dict(n=len(v),raw_rank_rho=float(np.corrcoef(standardized_rank(v[feature]),standardized_rank(v[score]))[0,1]))
                    for k,v in keep.groupby('label')}
                legacy_parts=[]
                for f in range(3):
                    old=pd.read_csv(ROOT/('runs/pam50_v2/baseline_outer_%d/features.csv'%f))
                    old=old[old.pid.isin(df.loc[df.outer.eq(f),'pid'])]
                    legacy_parts.append(old[['pid',feature]])
                legacy=keep.drop(columns=feature).merge(pd.concat(legacy_parts),on='pid',validate='one_to_one').dropna()
                row['legacy_phase_partial_rank_rho']=partial(legacy,feature,score,False)
                results.append(row)
                print('association',feature,score,'n',len(keep),'rho',row['partial_rank_rho'],flush=True)
                if time.time()-started>600:raise TimeoutError('Analysis budget exceeded')
    for row,q in zip(results,bh([r['p_approx_HC3'] for r in results])):row['q_BH_9']=float(q)
    scalar=[{k:v for k,v in r.items() if k!='by_subtype'} for r in results]
    pd.DataFrame(scalar).to_csv(OUT/'association_results.csv',index=False)
    for rel,h in p['source_sha256'].items():assert sha(ROOT/rel)==h,rel
    write_json(OUT/'results.json',dict(status='completed',protocol_sha256=sha(OUT/'protocol.json'),
        availability=availability,tests=results,primary_test_count=9,
        q_below_005=sum(r['q_BH_9']<.05 for r in results),elapsed_s=time.time()-started,
        input_sha256=sha(OUT/'development_association_input.csv'),self_tests=checks,
        test_used=False,external_used=False,original_validation_used=False,
        independent_biological_validation=False,mechanism_proven=False,caveats=p['caveats']))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','analyze']);a=ap.parse_args()
    prepare() if a.mode=='prepare' else analyze()


