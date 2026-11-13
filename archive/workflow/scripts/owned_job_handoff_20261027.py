"""Retain one exact Windows JobObject before retiring its old scheduler.

Only explicitly identified caller-owned processes. No kernel query or process
control at import. A duplicate handle is not closed during a successful adoption
until the existing worker completes or its original wall-clock bound expires.
"""
import ctypes
from ctypes import wintypes as W
from datetime import datetime, timezone
import time

from owned_windows_job_v2 import OwnedJob, ACCOUNTING, EXTENDED_LIMIT


class HandleEntry(ctypes.Structure):
    _fields_ = [('Object', ctypes.c_void_p), ('UniqueProcessId', ctypes.c_size_t),
        ('HandleValue', ctypes.c_size_t), ('GrantedAccess', W.ULONG),
        ('CreatorBackTraceIndex', W.USHORT), ('ObjectTypeIndex', W.USHORT),
        ('HandleAttributes', W.ULONG), ('Reserved', W.ULONG)]


class UnicodeString(ctypes.Structure):
    _fields_ = [('Length', W.USHORT), ('MaximumLength', W.USHORT), ('Buffer', ctypes.c_void_p)]


def api():
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    n = ctypes.WinDLL('ntdll', use_last_error=True)
    specs = {
        'OpenProcess': ([W.DWORD,W.BOOL,W.DWORD],W.HANDLE),
        'GetCurrentProcess': ([],W.HANDLE),
        'CloseHandle': ([W.HANDLE],W.BOOL),
        'DuplicateHandle': ([W.HANDLE,W.HANDLE,W.HANDLE,ctypes.POINTER(W.HANDLE),W.DWORD,W.BOOL,W.DWORD],W.BOOL),
        'GetProcessTimes': ([W.HANDLE,ctypes.POINTER(W.FILETIME),ctypes.POINTER(W.FILETIME),ctypes.POINTER(W.FILETIME),ctypes.POINTER(W.FILETIME)],W.BOOL),
        'IsProcessInJob': ([W.HANDLE,W.HANDLE,ctypes.POINTER(W.BOOL)],W.BOOL),
        'QueryInformationJobObject': ([W.HANDLE,ctypes.c_int,ctypes.c_void_p,W.DWORD,ctypes.c_void_p],W.BOOL),
        'GetExitCodeProcess': ([W.HANDLE,ctypes.POINTER(W.DWORD)],W.BOOL),
        'WaitForSingleObject': ([W.HANDLE,W.DWORD],W.DWORD),
        'TerminateProcess': ([W.HANDLE,W.UINT],W.BOOL),
        'TerminateJobObject': ([W.HANDLE,W.UINT],W.BOOL)}
    for name,(args,result) in specs.items():
        fn=getattr(k,name);fn.argtypes=args;fn.restype=result
    n.NtQuerySystemInformation.argtypes=[W.ULONG,ctypes.c_void_p,W.ULONG,ctypes.POINTER(W.ULONG)]
    n.NtQuerySystemInformation.restype=ctypes.c_long
    n.NtQueryObject.argtypes=[W.HANDLE,W.ULONG,ctypes.c_void_p,W.ULONG,ctypes.POINTER(W.ULONG)]
    n.NtQueryObject.restype=ctypes.c_long
    for name in ['NtSuspendProcess','NtResumeProcess']:
        fn=getattr(n,name);fn.argtypes=[W.HANDLE];fn.restype=ctypes.c_long
    return k,n


def birth(k, handle):
    values=[W.FILETIME() for _ in range(4)]
    if not k.GetProcessTimes(handle,*[ctypes.byref(x) for x in values]):
        raise ctypes.WinError(ctypes.get_last_error())
    return (int(values[0].dwHighDateTime)<<32)|int(values[0].dwLowDateTime)


def process_birth(pid):
    k,n=api();h=k.OpenProcess(0x1000,False,pid)
    if not h:raise ctypes.WinError(ctypes.get_last_error())
    try:return birth(k,h)
    finally:k.CloseHandle(h)


def handle_values(n, pid):
    if ctypes.sizeof(ctypes.c_void_p)!=8 or ctypes.sizeof(HandleEntry)!=40:
        raise RuntimeError('Only tested 64-bit Windows handle layout is accepted')
    size=1024*1024
    while size<=32*1024*1024:
        buf=ctypes.create_string_buffer(size);required=W.ULONG()
        status=n.NtQuerySystemInformation(64,buf,size,ctypes.byref(required))
        if status==0:break
        if status & 0xffffffff != 0xc0000004:
            raise OSError('Extended handle query failed: 0x%x'%(status&0xffffffff))
        size=max(size*2,int(required.value)+65536)
    else:raise RuntimeError('Extended handle table exceeds fixed32MiB bound')
    count=ctypes.c_size_t.from_buffer(buf).value
    start=2*ctypes.sizeof(ctypes.c_size_t)
    if count>(size-start)//ctypes.sizeof(HandleEntry):
        raise ValueError('Malformed handle-table count')
    entries=(HandleEntry*count).from_buffer(buf,start)
    return [int(e.HandleValue) for e in entries if e.UniqueProcessId==pid]


def active(k, handle):
    a=ACCOUNTING()
    if not k.QueryInformationJobObject(handle,1,ctypes.byref(a),ctypes.sizeof(a),None):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(a.ActiveProcesses)


def locate_job(k,n,parent,worker,pid):
    candidates=[]
    try:
        for value in handle_values(n,pid):
            duplicate=W.HANDLE()
            if not k.DuplicateHandle(parent,W.HANDLE(value),k.GetCurrentProcess(),ctypes.byref(duplicate),0,False,2):
                continue
            keep=False
            try:
                buf=ctypes.create_string_buffer(4096);needed=W.ULONG()
                if n.NtQueryObject(duplicate,2,buf,len(buf),ctypes.byref(needed))!=0:
                    continue
                name=UnicodeString.from_buffer(buf)
                if not name.Buffer or name.Length>512 or name.Length%2:
                    continue
                if ctypes.wstring_at(name.Buffer,name.Length//2)!='Job':
                    continue
                member=W.BOOL()
                if not k.IsProcessInJob(worker,duplicate,ctypes.byref(member)) or not member.value:
                    continue
                limits=EXTENDED_LIMIT()
                if not k.QueryInformationJobObject(duplicate,9,ctypes.byref(limits),ctypes.sizeof(limits),None):
                    raise ctypes.WinError(ctypes.get_last_error())
                if not limits.BasicLimitInformation.LimitFlags & 0x2000 or active(k,duplicate)!=1:
                    raise ValueError('Exact single-worker kill-on-close job required')
                keep=True;candidates.append((duplicate,value))
            finally:
                if not keep:k.CloseHandle(duplicate)
        if len(candidates)!=1:
            raise ValueError('Exactly one matching owned JobObject required, got%d'%len(candidates))
        return candidates.pop()
    finally:
        for h,_ in candidates:k.CloseHandle(h)


class Adoption:
    def __init__(self, parent_pid, worker_pid, parent_birth, worker_birth, deadline_utc):
        self.k,self.n=api();self.parent=None;self.worker=None;self.job=None
        self.parent_pid=parent_pid;self.worker_pid=worker_pid;self.retired=False;self.suspended=False
        deadline=datetime.fromisoformat(deadline_utc).timestamp()
        self.deadline=time.monotonic()+max(0,deadline-time.time())
        try:
            self.parent=self.k.OpenProcess(0x1000|0x100000|0x40|0x800|1,False,parent_pid)
            self.worker=self.k.OpenProcess(0x1000|0x100000,False,worker_pid)
            if not self.parent or not self.worker:raise ctypes.WinError(ctypes.get_last_error())
            if birth(self.k,self.parent)!=parent_birth or birth(self.k,self.worker)!=worker_birth:
                raise ValueError('Process creation-time identity changed')
            if self.k.WaitForSingleObject(self.worker,0)!=258:
                raise ValueError('Worker already terminal; do not adopt a stale stage')
            self.job,self.original_job_handle_value=locate_job(self.k,self.n,self.parent,self.worker,parent_pid)
        except Exception:
            self.close_probe();raise

    def close_probe(self):
        # Before retirement the old scheduler still owns the same job handle.
        if self.suspended and not self.retired:
            self.n.NtResumeProcess(self.parent);self.suspended=False
        for name in ['job','parent','worker']:
            h=getattr(self,name,None)
            if h:self.k.CloseHandle(h);setattr(self,name,None)

    def commit(self):
        if self.deadline-time.monotonic()<10:
            raise ValueError('Insufficient original timeout remaining for safe handoff')
        status=self.n.NtSuspendProcess(self.parent)
        if status!=0:raise OSError('Cannot suspend exact old scheduler: 0x%x'%(status&0xffffffff))
        self.suspended=True
        try:
            member=W.BOOL()
            if self.k.WaitForSingleObject(self.worker,0)!=258 or active(self.k,self.job)!=1:
                raise ValueError('Worker/job changed at handoff barrier')
            if not self.k.IsProcessInJob(self.worker,self.job,ctypes.byref(member)) or not member.value:
                raise ValueError('Worker ownership changed')
            if not self.k.TerminateProcess(self.parent,0):raise ctypes.WinError(ctypes.get_last_error())
            self.retired=True;self.suspended=False
            if self.k.WaitForSingleObject(self.parent,10000)!=0:
                raise RuntimeError('Retired scheduler did not exit')
            if self.k.WaitForSingleObject(self.worker,0)!=258:
                # Natural completion at this boundary is not a restarted worker.
                pass
        except Exception:
            if not self.retired and self.suspended:
                self.n.NtResumeProcess(self.parent);self.suspended=False
            raise

    def finish(self):
        if not self.retired:raise ValueError('Cannot finish a read-only probe')
        remaining=max(0,self.deadline-time.monotonic())
        wait=self.k.WaitForSingleObject(self.worker,int(remaining*1000))
        timed_out=(wait==258)
        if wait not in (0,258):raise ctypes.WinError(ctypes.get_last_error())
        if timed_out and not self.k.TerminateJobObject(self.job,124):
            raise ctypes.WinError(ctypes.get_last_error())
        if self.k.WaitForSingleObject(self.worker,10000)!=0:
            raise RuntimeError('Owned worker did not exit at completion/timeout')
        code=W.DWORD()
        if not self.k.GetExitCodeProcess(self.worker,ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        before=active(self.k,self.job)
        if before:
            if not self.k.TerminateJobObject(self.job,124):raise ctypes.WinError(ctypes.get_last_error())
            limit=time.monotonic()+10
            while active(self.k,self.job):
                if time.monotonic()>limit:raise TimeoutError('Owned descendants did not exit')
                time.sleep(.01)
        after=active(self.k,self.job)
        result=dict(status='hard_timeout' if timed_out else ('completed' if code.value==0 else 'worker_failed'),
            exit_code=int(code.value),timed_out=timed_out,pid=self.worker_pid,
            original_scheduler_pid=self.parent_pid,original_scheduler_retired_not_training_worker=True,
            same_worker_process_not_restarted=True,original_deadline_preserved=True,
            job_assigned=True,handshake_sent=True,supervision_transferred=True,
            owned_active_processes_before_job_close=before,
            owned_active_processes_after_termination=after,job_closed=True,
            automatic_retries=0,no_other_processes_targeted=True)
        self.close_probe()
        return result
