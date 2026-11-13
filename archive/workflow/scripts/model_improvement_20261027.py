"""Finite, separately authorized 743-case model improvement; never a test gate."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'reports/model_improvement_20261027'
RUNS = ROOT / 'runs/model_improvement_20261027'
SELF = 'scripts/model_improvement_20261027.py'
PYTHON = 'C:/ProgramData/anaconda3/envs/pytorch/python.exe'
OLD = ROOT / 'reports/exploratory743_20261026'
CLASSES = ['Basal', 'Her2', 'LumA', 'LumB']
SCOPE = 'new_authorized_743_exploratory_model_improvement_only'


def now():
    return datetime.now(timezone.utc).isoformat()


def path(name):
    p = Path(name)
    p = (p if p.is_absolute() else ROOT / p).resolve()
    p.relative_to(ROOT.resolve())
    return p


def rel(p):
    return path(p).relative_to(ROOT).as_posix()


def sha(p):
    h = hashlib.sha256()
    with path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(path(p).read_text(encoding='utf-8'))


def write(p, obj, replace=False):
    p = path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)
    if replace:
        tmp = p.with_suffix(p.suffix + '.tmp')
        tmp.write_text(text, encoding='utf-8')
        os.replace(str(tmp), str(p))
    else:
        with p.open('x', encoding='utf-8') as f:
            f.write(text)


def event(name, **kwargs):
    print(json.dumps(dict(event=name, utc=now(), **kwargs)), flush=True)


def ref(p):
    return dict(path=rel(p), sha256=sha(p), bytes=path(p).stat().st_size)


def checked(item):
    p = path(item['path'])
    if sha(p) != item['sha256']:
        raise ValueError('Bound input changed: ' + item['path'])
    return p


def cmd(mode):
    return [PYTHON, '-X', 'utf8', '-B', SELF, mode]


def seal_diagnostic():
    names = [SELF, 'scripts/owned_windows_job_v2.py',
        'scripts/exploratory743_20261026.py', 'scripts/pam50_v2_core.py',
        'scripts/physical_training_bridge_candidate_v1.py',
        'scripts/physical_patient_loss_candidate_v1.py',
        'reports/exploratory743_20261026/INPUT_MANIFEST.json',
        'reports/exploratory743_20261026/labels743.csv',
        'reports/exploratory743_analysis_20261026/ANALYSIS_REPORT.md']
    for outer in range(3):
        for arm in ['physical_full_fov', 'native_nominal_control']:
            prefix = 'runs/exploratory743_20261026/%s_outer%d/' % (arm, outer)
            names += [prefix + n for n in ['config.json', 'history.json',
                                          'selected_metrics.json', 'COMPLETED.json']]
    write(BASE / 'DIAGNOSTIC_PROTOCOL.json', dict(created_utc=now(), cwd=str(ROOT),
        scope=SCOPE, command=cmd('worker'), handshake_stage='diagnostic',
        input_sha256={n: sha(n) for n in names}, hard_timeout_s=120,
        success='Read six existing histories/configurations and record optimization ranges; no new patient values or model prediction',
        failure='Preserve first error and stop; no retry', automatic_retries=0,
        determinism='Exact source bindings and descriptive history summaries; timing excluded',
        outputs=[rel(BASE / 'DIAGNOSTIC.json')], patient_payload_reads=False,
        retained_or_external_endpoint_reads=False))


def diagnostic(expected):
    if sha(BASE / 'DIAGNOSTIC_PROTOCOL.json') != expected:
        raise ValueError('Diagnostic protocol mismatch')
    p = read(BASE / 'DIAGNOSTIC_PROTOCOL.json')
    for n, h in p['input_sha256'].items():
        if sha(n) != h:
            raise ValueError('Diagnostic binding changed: ' + n)
    summaries = []
    for outer in range(3):
        for arm in ['physical_full_fov', 'native_nominal_control']:
            d = ROOT / 'runs/exploratory743_20261026' / ('%s_outer%d' % (arm, outer))
            hist, selected = read(d / 'history.json'), read(d / 'selected_metrics.json')
            config = read(d / 'config.json')
            summaries.append(dict(arm=arm, outer=outer, epochs=len(hist),
                first_group_loss=hist[0]['group_loss_mean'],
                last_group_loss=hist[-1]['group_loss_mean'],
                min_group_loss=min(r['group_loss_mean'] for r in hist),
                last10_group_loss_mean=sum(r['group_loss_mean'] for r in hist[-10:]) / len(hist[-10:]),
                first_auc=hist[0]['macro_auc'], selected_auc=selected['macro_auc'],
                selected_epoch=selected['epoch'], last_auc=hist[-1]['macro_auc'],
                selected_prediction_counts=selected['prediction_counts'],
                initial_state_sha256=config['initial_state_sha256']))
    write(BASE / 'DIAGNOSTIC.json', dict(status='DESCRIPTIVE_ONLY', rows=summaries,
        conclusions_not_established=['underfitting cause', 'lesion localization correctness',
                                    'benefit of any proposed change'], patient_payload_reads=0,
        historical_selection_outputs_already_seen=True))
    event('DIAGNOSTIC_COMPLETED', rows=summaries)


def worker():
    raw = sys.stdin.readline(32769)
    if not raw or len(raw) > 32768:
        raise ValueError('Owned-job handshake required')
    payload = json.loads(raw)
    if payload.get('scope') != SCOPE:
        raise ValueError('Wrong scope')
    if payload['stage'] == 'diagnostic':
        diagnostic(payload['protocol_sha256'])
    else:
        raise ValueError('Stage not yet sealed')


def launch_diagnostic():
    from owned_windows_job_v2 import supervise
    result = supervise(cmd('worker'), dict(scope=SCOPE, stage='diagnostic',
        protocol_sha256=sha(BASE / 'DIAGNOSTIC_PROTOCOL.json')), ROOT,
        BASE / 'jobs/diagnostic', 120)
    event('DIAGNOSTIC_TERMINAL', **result)
    return 0 if (result.get('exit_code') == 0 and not result.get('timed_out') and
        result.get('owned_active_processes_after_termination') == 0) else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['seal_diagnostic', 'diagnostic', 'worker'])
    args = parser.parse_args()
    if args.mode == 'seal_diagnostic':
        seal_diagnostic()
    elif args.mode == 'diagnostic':
        sys.exit(launch_diagnostic())
    else:
        worker()
