"""Explicit v3: verified scoped console helper; preserve all scientific code."""
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'scripts/model_improvement_dualgpu_20261027.py'
raw=SOURCE.read_bytes()
if hashlib.sha256(raw).hexdigest()!='408d805e30a5886d005766f91a790d733ed506321fb4d32af6cd4309f7a75bec':
    raise ValueError('Original scheduling source changed')
source=raw.decode('utf-8')
aliases=[('from owned_job_handoff_20261027 import','from owned_job_handoff_v3_20261027 import'),
    ("'reports/model_improvement_dualgpu_20261027'","'reports/model_improvement_dualgpu_v3_20261027'"),
    ("'scripts/model_improvement_dualgpu_20261027.py'","'scripts/model_improvement_dualgpu_v3_20261027.py'"),
    ("'scripts/owned_job_handoff_20261027.py'","'scripts/owned_job_handoff_v3_20261027.py'"),
    ("[('normal',2,False),('hard_timeout',60,True)]","[('normal',6,False),('hard_timeout',60,True)]")]
for before,after in aliases:
    if source.count(before)!=1:raise ValueError('Exactly one operational source substitution required')
    source=source.replace(before,after,1)
exec(compile(source,str(SOURCE),'exec'),dict(__name__='__main__',__file__=str(SOURCE)))
