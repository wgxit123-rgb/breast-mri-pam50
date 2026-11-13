"""Extract approved development phases locally without changing legacy inputs.

Tar entries never become arbitrary filesystem paths. Only whitelisted original
training IDs and approved phase indices are extracted. No validation/test pixel
data is decoded or persisted. Traversing compressed archive streams is necessary
but is not an evaluation or a test-image inspection.
"""
import argparse,gzip,os,re,shutil,tarfile,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from nominal_phase_contract import ROOT,OUT,DATA,sha,write_json,load_contract

FIELDS=['nominal_pre','first_post','last_post']

def path_for(pid,phase):
    old=ROOT/'data/fast_v1/selected_nifti'/pid/('acq_%d.nii.gz'%phase)
    return old if old.exists() else DATA/'native_phases'/pid/('acq_%d.nii.gz'%phase)

def extract_one(path,wanted):
    start=time.time(); clock_start=time.monotonic(); copied=[]; matched=[]; count=0; last=clock_start; before=path.stat()
    with tarfile.open(str(path),'r|gz') as archive:
        for member in archive:
            count+=1
            if time.monotonic()-clock_start>1800: raise TimeoutError('Per-archive 30 minute budget exceeded')
            if time.monotonic()-last>30:
                print(path.name,'scanned members',count,'extracted',len(copied),flush=True); last=time.monotonic()
            if not member.isfile(): continue
            match=re.fullmatch(r'((?:ACRIN-6698|ISPY2)-\d+)_spy2_vis1_dce_aqc_(\d+)\.nii\.gz',Path(member.name).name)
            if not match: continue
            pid,phase=match[1],int(match[2])
            if pid not in wanted or phase not in wanted[pid]: continue
            dest=DATA/'native_phases'/pid/('acq_%d.nii.gz'%phase)
            assert dest.resolve().is_relative_to(DATA.resolve()) if hasattr(dest,'is_relative_to') else str(dest.resolve()).startswith(str(DATA.resolve())+os.sep)
            matched.append((pid,phase,member.name))
            if dest.exists(): raise RuntimeError('Unexpected duplicate/existing selected member '+str(dest))
            dest.parent.mkdir(parents=True,exist_ok=True); temp=dest.with_suffix('.partial')
            h=__import__('hashlib').sha256()
            with archive.extractfile(member) as inp,temp.open('wb') as out:
                for block in iter(lambda:inp.read(1024*1024),b''):
                    h.update(block); out.write(block)
            assert temp.stat().st_size==member.size,'truncated tar member'
            # Read gzip stream to EOF to verify the member's own gzip CRC.
            with gzip.open(temp,'rb') as f:
                for _ in iter(lambda:f.read(1024*1024),b''): pass
            temp.replace(dest)
            rec=dict(pid=pid,phase=phase,archive=str(path),archive_size=before.st_size,
              archive_mtime_ns=before.st_mtime_ns,member=member.name,member_size=member.size,
              output=str(dest),sha256=h.hexdigest(),gzip_crc_checked=True,partition='original_train')
            write_json(dest.with_suffix('.source.json'),rec); copied.append(rec)
    after=path.stat(); assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'archive changed'
    result=dict(archive=path.name,matched_members=len(matched),extracted=len(copied),elapsed_s=round(time.time()-start,2))
    write_json(OUT/'extraction_archives'/(path.name+'.json'),result)
    print('COMPLETE',result,flush=True); return result

def extract():
    _,table=load_contract(); table=table[table.cohort.eq('ispy')]
    wanted={r.pid:{int(getattr(r,k)) for k in FIELDS if not path_for(r.pid,int(getattr(r,k))).exists()} for r in table.itertuples(index=False)}
    wanted={p:s for p,s in wanted.items() if s}
    missing=sum(len(v) for v in wanted.values()); print('Missing approved local phases:',missing,'patients:',len(wanted),flush=True)
    write_json(OUT/'extraction_initial.json',dict(pid=os.getpid(),started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),missing_volumes=missing,missing_patients=len(wanted),original_test_used=False))
    if not missing: return
    archives=sorted((ROOT/'data/BreastDCEDL_ISPY2').glob('*_dce_part*.tar.gz'))
    assert len(archives)==14
    # Threads avoid duplicating interpreter state; 3 streams bound disk pressure.
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(extract_one,p,wanted) for p in archives]
        for f in as_completed(jobs): f.result()
    remaining=[(pid,i) for pid,ids in wanted.items() for i in ids if not path_for(pid,i).exists()]
    write_json(OUT/'extraction_completion.json',dict(expected=missing,remaining=remaining,all_expected_present=not remaining,original_test_used=False))
    assert not remaining,'Approved source files absent in local archives'

def inspect_case(row):
    pid=row.pid; rec=dict(pid=pid,status='hold',geometry_verified=False,model_input_ready=False)
    files=[]; shape=None
    try:
        for role in FIELDS:
            phase=int(getattr(row,role)); p=path_for(pid,phase); assert p.exists(),'missing '+str(p)
            im=nib.load(str(p)); a=np.asarray(im.dataobj)
            assert a.ndim==3 and np.isfinite(a).all() and a.max()>a.min()
            if shape is None: shape=a.shape
            assert a.shape==shape,'within-patient phase shapes differ'
            files.append(dict(role=role,phase=phase,path=str(p),sha256=sha(p),shape=list(a.shape),
              minimum=float(a.min()),maximum=float(a.max()),header_spacing=list(map(float,im.header.get_zooms()))))
        mask=ROOT/'data/fast_v1/masks'/(pid+'.nii.gz'); m=np.asarray(nib.load(str(mask)).dataobj)
        assert m.shape==shape and np.isfinite(m).all() and set(np.unique(m)).issubset({0,1})
        rec.update(status='nominal_three_phase_native_files_verified',files=files,mask_path=str(mask),mask_sha256=sha(mask),
          header_affine_not_used_as_physical_truth=True,timing_uncertain=True)
    except Exception as e: rec['error']=repr(e)
    write_json(OUT/'native_cases'/(pid+'.json'),rec); return rec

def inspect():
    _,table=load_contract(); table=table[table.cohort.eq('ispy')]; records=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,r in enumerate(pool.map(inspect_case,table.itertuples(index=False)),1):
            records.append(r)
            if i%25==0: print('Native phase QC',i,'/',len(table),flush=True)
    from collections import Counter
    summary=dict(n=len(records),status=dict(Counter(r['status'] for r in records)),
      physical_geometry_released=False,model_frozen=False,external_predictions_run=False)
    write_json(OUT/'native_phase_summary.json',summary); print(summary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['extract','inspect']); a=p.parse_args()
    extract() if a.mode=='extract' else inspect()
