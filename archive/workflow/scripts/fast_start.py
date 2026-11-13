"""Reproducible first-pass data preparation and dual-GPU segmentation experiments.

This is a voxel-grid MONAI pilot, not the final nnU-Net/radiomics experiment.
Never evaluates the held-out test set. Raw archives are read-only.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'BreastDCEDL_ISPY2'
WORK = ROOT / 'data' / 'fast_v1'
REPORT = ROOT / 'reports' / 'fast_v1'
SHAPE = (64, 128, 128)


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)


def emit(event, **values):
    print(json.dumps({'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'event': event, **values}, ensure_ascii=False), flush=True)


def metadata():
    import pandas as pd
    df = pd.read_csv(RAW / 'BreastDCEDL_ISPY2_metadata.csv')
    clinical = pd.read_excel(ROOT / 'data' / 'TCIA' / 'ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx')
    df['join_id'] = df.pid.str.extract(r'(\d+)$')[0]
    clinical['join_id'] = clinical.Patient_ID.astype(str)
    df = df.merge(clinical, on='join_id', how='left', validate='one_to_one', suffixes=('', '_official'), indicator=True)
    assert len(df) == 982 and df.pid.is_unique and df['_merge'].eq('both').all()
    for label in ['HR', 'HER2', 'pCR']:
        assert df[label].eq(df[label + '_official']).all(), label
    df['split'] = df.test.map({0: 'train', 2: 'val', 1: 'test'})
    assert df.split.notna().all()
    return df


def extract_archive(path):
    """Copy only selected DCE phases; no tar paths used as filesystem targets."""
    import pandas as pd
    df = pd.read_csv(RAW / 'BreastDCEDL_ISPY2_metadata.csv')
    wanted = {r.pid: {int(r.pre), int(r.post_early), int(r.post_late)} for r in df.itertuples()}
    copied = 0
    with tarfile.open(path, 'r|gz') as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = Path(member.name).name
            match = re.fullmatch(r'(.+)_spy2_vis1_dce_aqc_(\d+)\.nii\.gz', name)
            if not match:
                continue
            pid, index = match[1], int(match[2])
            if pid not in wanted or index not in wanted[pid]:
                continue
            dest = WORK / 'selected_nifti' / pid / f'acq_{index}.nii.gz'
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                temporary = dest.with_suffix('.partial')
                with archive.extractfile(member) as inp, temporary.open('wb') as out:
                    shutil.copyfileobj(inp, out, length=1024 * 1024)
                if temporary.stat().st_size != member.size:
                    raise ValueError(f'Truncated archive member: {name}')
                os.replace(temporary, dest)
            elif dest.stat().st_size != member.size:
                raise ValueError(f'Existing file size mismatch: {dest}')
            copied += 1
    return {'archive': Path(path).name, 'selected_volumes': copied}


def make_cache(record):
    import nibabel as nib
    import numpy as np
    from scipy.ndimage import zoom
    pid = record['pid']
    out = WORK / 'cache' / (pid + '.npz')
    report_path = WORK / 'cache' / (pid + '.json')
    old_audit = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else None
    if out.exists() and old_audit and old_audit.get('cache_schema') == 2:
        return old_audit
    mask_img = nib.load(WORK / 'masks' / (pid + '.nii.gz'))
    mask_raw = np.asarray(mask_img.dataobj)
    assert mask_raw.ndim == 3 and np.isfinite(mask_raw).all(), pid
    assert set(np.unique(mask_raw)).issubset({0, 1}), pid
    assert mask_raw.any(), 'Empty reference mask: ' + pid
    if out.exists() and old_audit:
        with np.load(out) as previous:
            x = previous['image']
        headers, scale = old_audit['phase_headers'], old_audit['intensity_scale']
    else:
        image_list = []
        headers = []
        for phase in ['pre', 'post_early', 'post_late']:
            index = int(record[phase])
            img = nib.load(WORK / 'selected_nifti' / pid / f'acq_{index}.nii.gz')
            arr = img.get_fdata(dtype=np.float32)
            assert arr.shape == mask_raw.shape and np.isfinite(arr).all(), pid
            assert np.allclose(img.affine, mask_img.affine), 'Grid mismatch: ' + pid
            headers.append({'phase': phase, 'index': index, 'spacing': list(map(float, img.header.get_zooms())), 'shape': list(arr.shape)})
            image_list.append(arr)
        # Patient-wise shared intensity scaling, no label- or cohort-fitted statistics.
        positive = image_list[0][image_list[0] > 0]
        assert positive.size > 0, pid
        scale = max(float(np.percentile(positive, 99.5)), 1.0)
        factors = np.array(SHAPE) / np.array(mask_raw.shape)
        x = np.stack([zoom(np.clip(a / scale, 0, 5), factors, order=1, prefilter=False) for a in image_list]).astype(np.float16)
    # Uniform occupancy-preserving coarse targets. This expands coarse boundaries;
    # pilot Dice is explicitly NOT native-resolution anatomical segmentation Dice.
    import torch
    torch.set_num_threads(1)
    y = torch.nn.functional.adaptive_max_pool3d(torch.from_numpy(mask_raw.astype(np.float32))[None,None], SHAPE)[0,0].numpy().astype(np.uint8)
    assert y.any(), 'Empty coarse target: ' + pid
    assert x.shape == (3, *SHAPE) and y.shape == SHAPE
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix('.partial')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, image=x, mask=y.astype(np.uint8))
    os.replace(temporary, out)
    audit = {'pid': pid, 'cache_schema': 2, 'coarse_mask_method': 'adaptive_max_pool3d occupancy', 'shape_native': list(mask_raw.shape), 'header_spacing': list(map(float, mask_img.header.get_zooms())),
             'csv_slice_thick': float(record['slice_thick']), 'csv_xy_spacing': float(record['xy_spacing']),
             'native_foreground_voxels': int(mask_raw.sum()), 'pilot_foreground_voxels': int(y.sum()),
             'intensity_scale': scale, 'phase_headers': headers, 'physical_space_verified': False}
    save_json(report_path, audit)
    return audit


def baseline(df):
    import numpy as np
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    train, val = df[df.split == 'train'], df[df.split == 'val']
    results = []
    predictions = pd.DataFrame({'pid': val.pid})
    for target in ['HER2', 'HR', 'MP']:
        for name in ['prevalence', 'age_menopause']:
            if name == 'prevalence':
                probability = np.full(len(val), train[target].mean())
            else:
                model = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(), LogisticRegression(C=1, max_iter=1000, random_state=42))
                model.fit(train[['age', 'menopause']], train[target])
                probability = model.predict_proba(val[['age', 'menopause']])[:, 1]
            results.append({'target': target, 'model': name, 'split': 'val', 'n': len(val),
                            'auc': roc_auc_score(val[target], probability), 'pr_auc': average_precision_score(val[target], probability),
                            'brier': brier_score_loss(val[target], probability)})
            predictions[target + '_' + name] = probability
    REPORT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(REPORT / 'clinical_baseline_validation.csv', index=False)
    predictions.to_csv(REPORT / 'clinical_baseline_predictions.csv', index=False)
    emit('clinical_baselines_complete', results=results)


def prepare(workers):
    started = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    try:
        df = metadata()
        df.to_csv(WORK / 'patient_manifest.csv', index=False)
        save_json(REPORT / 'data_identity.json', {'n': len(df), 'split_counts': df.split.value_counts().to_dict(),
                  'metadata_sha256': hashlib.sha256((RAW / 'BreastDCEDL_ISPY2_metadata.csv').read_bytes()).hexdigest(),
                  'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'shape': SHAPE,
                  'geometry_mode': 'voxel-grid pilot only; no mm-based metrics', 'physical_space_verified': False})
        baseline(df)
        (WORK / 'masks').mkdir(exist_ok=True)
        ids = set(df.pid)
        with tarfile.open(RAW / 'BreastDCEDL_ISPY2_mask.tar.gz', 'r:gz') as arc:
            for member in arc:
                if not member.isfile() or not member.name.endswith('.nii.gz'):
                    continue
                match = re.match(r'((?:ACRIN-6698|ISPY2)-\d+)_', Path(member.name).name)
                if not match or match[1] not in ids:
                    raise ValueError('Unrecognized mask: ' + member.name)
                dest = WORK / 'masks' / (match[1] + '.nii.gz')
                if not dest.exists():
                    with arc.extractfile(member) as inp, dest.open('wb') as out:
                        shutil.copyfileobj(inp, out)
        emit('masks_ready', count=len(list((WORK / 'masks').glob('*.nii.gz'))))
        archives = sorted(RAW.glob('*_dce_part*.tar.gz'), key=lambda p: p.stat().st_size)
        expected_paths = [WORK/'selected_nifti'/r.pid/f'acq_{i}.nii.gz' for r in df.itertuples() for i in {int(r.pre),int(r.post_early),int(r.post_late)}]
        if all(p.exists() and p.stat().st_size > 0 for p in expected_paths):
            emit('reuse_extracted_volumes', count=len(expected_paths))
        else:
            with cf.ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(extract_archive, str(p)) for p in archives]
                for f in cf.as_completed(futures):
                    emit('archive_selected', **f.result(), elapsed_seconds=round(time.time() - started, 1))
        # Prepare train and validation only. Test imagery remains outside the pilot loop.
        eligible = df[df.split != 'test'].to_dict('records')
        audits = []
        with cf.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(make_cache, r) for r in eligible]
            for f in cf.as_completed(futures):
                audits.append(f.result())
                if len(audits) % 25 == 0:
                    emit('cache_progress', completed=len(audits), total=len(eligible))
        save_json(REPORT / 'geometry_audit.json', audits)
        save_json(WORK / 'READY.json', {'status': 'ready', 'cache_schema': 2, 'train': 784, 'val': 99, 'cache_count': len(audits),
                                      'test_evaluated': False, 'elapsed_seconds': round(time.time() - started, 1)})
        emit('prepare_complete', elapsed_seconds=round(time.time() - started, 1))
    except Exception as e:
        save_json(WORK / 'FAILED.json', {'error': repr(e), 'time': time.time()})
        raise


def gpu_check():
    import torch
    import monai
    from monai.networks.nets import UNet
    results = []
    torch.set_num_threads(4)
    for gpu in range(2):
        torch.cuda.set_device(gpu)
        net = UNet(spatial_dims=3, in_channels=3, out_channels=1, channels=(16,32,64,128), strides=(2,2,2), num_res_units=1, norm='INSTANCE').cuda()
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
        data = torch.randn(2, 3, *SHAPE, device='cuda')
        start = time.time()
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = net(data).square().mean()
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        result = {'gpu': gpu, 'name': torch.cuda.get_device_name(gpu), 'seconds_3_steps': time.time()-start,
                  'peak_MiB': torch.cuda.max_memory_allocated(gpu)/2**20, 'loss_finite': bool(torch.isfinite(loss).item())}
        results.append(result)
        emit('gpu_forward_backward_ok', **result)
        del opt, net, data, loss
        torch.cuda.empty_cache()
    save_json(REPORT / 'gpu_check.json', {'python': sys.executable, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'monai': monai.__version__, 'results': results})


class CachedVolumes:
    def __init__(self, pids, channels, augment=False):
        self.pids, self.channels, self.augment = list(pids), channels, augment

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, i):
        import numpy as np
        import torch
        with np.load(WORK / 'cache' / (self.pids[i] + '.npz')) as f:
            x = f['image'].astype(np.float32)
            y = f['mask'].astype(np.float32)[None]
        if self.channels == 1:
            x = x[1:2]
        if self.augment:
            for axis in (2, 3):
                if random.random() < .5:
                    x, y = np.flip(x, axis), np.flip(y, axis)
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy()), self.pids[i]


def train(gpu, channels, epochs, batch_size, workers):
    import numpy as np
    import pandas as pd
    import torch
    from monai.networks.nets import UNet
    from monai.losses import DiceCELoss
    from torch.utils.data import DataLoader
    wait_start = time.time()
    while not (WORK / 'READY.json').exists():
        if (WORK / 'FAILED.json').exists():
            raise RuntimeError((WORK / 'FAILED.json').read_text(encoding='utf-8'))
        if time.time() - wait_start > 6*3600:
            raise TimeoutError('Data preparation exceeded 6 hours')
        emit('waiting_for_prepared_data', gpu=gpu)
        time.sleep(30)
    torch.cuda.set_device(gpu)
    torch.set_num_threads(4)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    df = pd.read_csv(WORK / 'patient_manifest.csv')
    train_rows, val_rows = df[df.split == 'train'], df[df.split == 'val']
    assert set(train_rows.pid).isdisjoint(val_rows.pid)
    run = ROOT / 'runs' / ('fast_seg_single' if channels == 1 else 'fast_seg_multiphase')
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'last.pt').exists():
        raise RuntimeError(f'Existing checkpoint in {run}; explicit resume implementation required')
    config = {'gpu':gpu, 'channels':channels, 'max_epochs':epochs, 'batch_size':batch_size, 'seed':42,
              'shape':SHAPE, 'train_n':len(train_rows), 'val_n':len(val_rows), 'test_evaluated':False,
              'model':'MONAI UNet channels16,32,64,128 strides2,2,2 residual_units1',
              'optimizer':'AdamW lr=0.001 weight_decay=0.0001', 'loss':'DiceCELoss(sigmoid=True)',
              'purpose':'coarse whole-volume voxel-grid pilot; not nnU-Net or final publication result',
              'geometry':'index-space resized; physical spacing unresolved; occupancy-pooled coarse targets', 'cache_schema':2, 'early_stop_patience':12,
              'metadata_sha256':hashlib.sha256((RAW/'BreastDCEDL_ISPY2_metadata.csv').read_bytes()).hexdigest(),
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save_json(run / 'config.json', config)
    generator = torch.Generator().manual_seed(42)
    loaders = [DataLoader(CachedVolumes(part.pid, channels, aug), batch_size=batch_size, shuffle=aug,
                         num_workers=workers, pin_memory=True, persistent_workers=workers>0,
                         generator=generator if aug else None) for part,aug in [(train_rows,True),(val_rows,False)]]
    net = UNet(spatial_dims=3, in_channels=channels, out_channels=1, channels=(16,32,64,128), strides=(2,2,2), num_res_units=1, norm='INSTANCE').cuda()
    criterion = DiceCELoss(sigmoid=True)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = -1.0
    no_improvement = 0
    history = []
    started = time.time()
    emit('training_start', **config)
    for epoch in range(1, epochs+1):
        epoch_start = time.time()
        net.train()
        losses = []
        for step, (x,y,_) in enumerate(loaders[0],1):
            x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=criterion(net(x),y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),12)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            if step % 100 == 0:
                emit('train_step',epoch=epoch,step=step,total=len(loaders[0]),mean_loss=float(np.mean(losses)))
        scheduler.step()
        net.eval()
        predictions=[]
        with torch.inference_mode():
            for x,y,pids in loaders[1]:
                x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=net(x)
                p=logits>0
                dims=(1,2,3,4)
                intersection=(p*y).sum(dims)
                pv=p.sum(dims)
                truth=y.sum(dims)
                dice=(2*intersection/(pv+truth).clamp_min(1)).cpu().tolist()
                for pid,d,v in zip(pids,dice,pv.cpu().tolist()):
                    predictions.append({'pid':pid,'dice':d,'empty_prediction':v==0,'predicted_voxels':v})
        score=float(np.mean([p['dice'] for p in predictions]))
        entry={'epoch':epoch,'loss':float(np.mean(losses)),'val_dice':score,
               'val_empty_rate':float(np.mean([p['empty_prediction'] for p in predictions])),
               'seconds':round(time.time()-epoch_start,2),'elapsed_seconds':round(time.time()-started,2)}
        history.append(entry)
        pd.DataFrame(history).to_csv(run/'history.csv',index=False)
        improved=score>best+1e-5
        no_improvement=0 if improved else no_improvement+1
        if improved:
            best=score
        state={'epoch':epoch,'model':net.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
               'config':config,'best_val_dice':best,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
               'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(), 'loader_generator':generator.get_state()}
        temp=run/'last.tmp'
        torch.save(state,temp)
        os.replace(temp,run/'last.pt')
        if improved:
            shutil.copy2(run/'last.pt',run/'best.pt')
            pd.DataFrame(predictions).to_csv(run/'best_validation_patients.csv',index=False)
        emit('epoch_complete',gpu=gpu,**entry,best_val_dice=best)
        if no_improvement>=12:
            emit('early_stop',epoch=epoch)
            break
        if time.time()-started>24*3600:
            raise TimeoutError('24h hard budget reached; last checkpoint retained')
    save_json(run/'COMPLETED.json',{'status':'completed','epochs':epoch,'best_val_dice':best,
                                   'elapsed_seconds':round(time.time()-started,2),'test_evaluated':False})
    emit('training_complete',gpu=gpu,best_val_dice=best)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','gpu-check','train'])
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--channels',type=int,choices=[1,3],default=3)
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--batch-size',type=int,default=2)
    args=parser.parse_args()
    if args.mode=='prepare': prepare(args.workers)
    elif args.mode=='gpu-check': gpu_check()
    else: train(args.gpu,args.channels,args.epochs,args.batch_size,args.workers)
