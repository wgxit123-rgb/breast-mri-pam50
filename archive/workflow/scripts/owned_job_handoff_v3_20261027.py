"""Exact worker job plus at most its verified Windows console helper."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import owned_job_handoff_20261027 as original

if hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest()!='5e86f00fafa485b6a41aa8b96fb22a005cd9ccb092042fe5ace1a3ff60fab1d3':
    raise ValueError('Preserved native helper changed')
c,W=original.ctypes,original.W


def members(k,job,worker):
    k.GetProcessId.argtypes=[W.HANDLE];k.GetProcessId.restype=W.DWORD
    target=int(k.GetProcessId(worker))
    buf=c.create_string_buffer(4096)
    if not k.QueryInformationJobObject(job,3,buf,len(buf),None):raise c.WinError(c.get_last_error())
    assigned=W.DWORD.from_buffer(buf,0).value;count=W.DWORD.from_buffer(buf,4).value
    if not 1<=count<=2 or assigned!=count:raise ValueError('Only exact model worker and at most one console helper may be owned')
    ids=list((c.c_size_t*count).from_buffer(buf,8))
    if target not in ids:raise ValueError('Bound worker missing from job member list')
    extra=[int(pid) for pid in ids if pid!=target]
    records=[dict(pid=target,role='exact_bound_worker')]
    if extra:
        pid=extra[0]
        # Scope the live metadata lookup to the single kernel-proven auxiliary PID.
        source='Get-CimInstance Win32_Process -Filter "ProcessId = %d" | Select-Object ProcessId,ParentProcessId,ExecutablePath | ConvertTo-Json -Compress'%pid
        raw=subprocess.check_output(['powershell.exe','-NoProfile','-Command',source],timeout=10)
        row=json.loads(raw.decode('utf-8-sig'))
        allowed={(Path(os.environ['SystemRoot'])/'System32/conhost.exe').as_posix().casefold(),
                 (Path(os.environ['SystemRoot'])/'SysWOW64/conhost.exe').as_posix().casefold()}
        observed=str(row.get('ExecutablePath','')).replace('\\','/').casefold()
        if row.get('ProcessId')!=pid or row.get('ParentProcessId')!=target or observed not in allowed:
            raise ValueError('Unknown/non-descendant auxiliary process in worker job')
        h=k.OpenProcess(0x1000,False,pid)
        if not h:raise c.WinError(c.get_last_error())
        try:created=original.birth(k,h)
        finally:k.CloseHandle(h)
        records.append(dict(pid=pid,role='verified_windows_console_helper',parent_pid=target,
            path=row['ExecutablePath'],creation_time=created))
    return records


def locate_job(k,n,parent,worker,pid):
    candidates=[];observations=[]
    try:
        for value in original.handle_values(n,pid):
            duplicate=W.HANDLE()
            if not k.DuplicateHandle(parent,W.HANDLE(value),k.GetCurrentProcess(),c.byref(duplicate),0,False,2):continue
            keep=False
            try:
                buf=c.create_string_buffer(4096);needed=W.ULONG()
                if n.NtQueryObject(duplicate,2,buf,len(buf),c.byref(needed))!=0:continue
                name=original.UnicodeString.from_buffer(buf)
                if not name.Buffer or name.Length>512 or name.Length%2:continue
                if c.wstring_at(name.Buffer,name.Length//2)!='Job':continue
                is_member=W.BOOL()
                if not k.IsProcessInJob(worker,duplicate,c.byref(is_member)) or not is_member.value:continue
                limits=original.EXTENDED_LIMIT()
                if not k.QueryInformationJobObject(duplicate,9,c.byref(limits),c.sizeof(limits),None):raise c.WinError(c.get_last_error())
                flags=int(limits.BasicLimitInformation.LimitFlags)
                if not flags & 0x2000:continue
                try:record=members(k,duplicate,worker)
                except ValueError as error:observations.append(str(error));continue
                keep=True;candidates.append((duplicate,value))
                print(json.dumps(dict(event='EXACT_WORKER_JOB_IDENTIFIED',members=record)),flush=True)
            finally:
                if not keep:k.CloseHandle(duplicate)
        if len(candidates)!=1:raise ValueError('Exactly one exact worker-scoped job required: '+repr(observations))
        return candidates.pop()
    finally:
        for h,_ in candidates:k.CloseHandle(h)


original.locate_job=locate_job
process_birth=original.process_birth


class Adoption(original.Adoption):
    def commit(self):
        if self.deadline-time.monotonic()<10:raise ValueError('Insufficient original timeout remains')
        status=self.n.NtSuspendProcess(self.parent)
        if status!=0:raise OSError('Cannot suspend exact old scheduler: 0x%x'%(status&0xffffffff))
        self.suspended=True
        try:
            if self.k.WaitForSingleObject(self.worker,0)!=258:raise ValueError('Worker terminal at handoff barrier')
            record=members(self.k,self.job,self.worker)
            is_member=W.BOOL()
            if not self.k.IsProcessInJob(self.worker,self.job,c.byref(is_member)) or not is_member.value:
                raise ValueError('Worker ownership changed')
            if not self.k.TerminateProcess(self.parent,0):raise c.WinError(c.get_last_error())
            self.retired=True;self.suspended=False
            if self.k.WaitForSingleObject(self.parent,10000)!=0:raise RuntimeError('Old scheduling process did not retire')
            print(json.dumps(dict(event='SCHEDULER_RETIRED_WORKER_RETAINED',worker_pid=self.worker_pid,
                worker_members=record,worker_restart=False)),flush=True)
        except Exception:
            if not self.retired and self.suspended:
                self.n.NtResumeProcess(self.parent);self.suspended=False
            raise
