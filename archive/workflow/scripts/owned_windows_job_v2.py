"""Version2:explicitUTF8childstdio;all process ownership/scientific rules unchanged.

The audited worker must wait on stdin before work. The supervisor assigns its
process to a KILL_ON_JOB_CLOSE object before sending that handshake. Descendants
are killed by the kernel on timeout or owner exit,without PID-tree guessing.
"""
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Dict, List, Sequence


class BASIC_LIMIT(ctypes.Structure):
    _fields_ = [('PerProcessUserTimeLimit',ctypes.c_longlong),('PerJobUserTimeLimit',ctypes.c_longlong),
        ('LimitFlags',wintypes.DWORD),('MinimumWorkingSetSize',ctypes.c_size_t),
        ('MaximumWorkingSetSize',ctypes.c_size_t),('ActiveProcessLimit',wintypes.DWORD),
        ('Affinity',ctypes.c_size_t),('PriorityClass',wintypes.DWORD),('SchedulingClass',wintypes.DWORD)]


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(name,ctypes.c_ulonglong) for name in ['ReadOperationCount','WriteOperationCount',
        'OtherOperationCount','ReadTransferCount','WriteTransferCount','OtherTransferCount']]


class EXTENDED_LIMIT(ctypes.Structure):
    _fields_ = [('BasicLimitInformation',BASIC_LIMIT),('IoInfo',IO_COUNTERS),
        ('ProcessMemoryLimit',ctypes.c_size_t),('JobMemoryLimit',ctypes.c_size_t),
        ('PeakProcessMemoryUsed',ctypes.c_size_t),('PeakJobMemoryUsed',ctypes.c_size_t)]


class ACCOUNTING(ctypes.Structure):
    _fields_ = [(n,ctypes.c_longlong) for n in ['TotalUserTime','TotalKernelTime',
        'ThisPeriodTotalUserTime','ThisPeriodTotalKernelTime']]+[(n,wintypes.DWORD) for n in
        ['TotalPageFaultCount','TotalProcesses','ActiveProcesses','TotalTerminatedProcesses']]


class OwnedJob:
    def __init__(self):
        if os.name != 'nt':
            raise RuntimeError('This declared supervisor is Windows-only;no silent fallback')
        self.kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes=[ctypes.c_void_p,wintypes.LPCWSTR]
        self.kernel.CreateJobObjectW.restype=wintypes.HANDLE
        self.kernel.SetInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]
        self.kernel.SetInformationJobObject.restype=wintypes.BOOL
        self.kernel.AssignProcessToJobObject.argtypes=[wintypes.HANDLE,wintypes.HANDLE]
        self.kernel.AssignProcessToJobObject.restype=wintypes.BOOL
        self.kernel.QueryInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD,ctypes.c_void_p]
        self.kernel.QueryInformationJobObject.restype=wintypes.BOOL
        self.kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        self.kernel.CloseHandle.restype=wintypes.BOOL
        self.kernel.TerminateJobObject.argtypes=[wintypes.HANDLE,wintypes.UINT]
        self.kernel.TerminateJobObject.restype=wintypes.BOOL
        self.handle=self.kernel.CreateJobObjectW(None,None)
        if not self.handle:raise ctypes.WinError(ctypes.get_last_error())
        settings=EXTENDED_LIMIT()
        settings.BasicLimitInformation.LimitFlags=0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(self.handle,9,ctypes.byref(settings),ctypes.sizeof(settings)):
            error=ctypes.get_last_error();self.close();raise ctypes.WinError(error)

    def assign(self,process: subprocess.Popen) -> None:
        if not self.kernel.AssignProcessToJobObject(self.handle,wintypes.HANDLE(int(process._handle))):
            raise ctypes.WinError(ctypes.get_last_error())

    def active(self) -> int:
        value=ACCOUNTING()
        if not self.kernel.QueryInformationJobObject(self.handle,1,ctypes.byref(value),ctypes.sizeof(value),None):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(value.ActiveProcesses)

    def close(self) -> None:
        if self.handle:
            if not self.kernel.CloseHandle(self.handle):raise ctypes.WinError(ctypes.get_last_error())
            self.handle=None

    def terminate_remaining(self) -> None:
        if self.active():
            if not self.kernel.TerminateJobObject(self.handle,124):
                raise ctypes.WinError(ctypes.get_last_error())
            limit=time.monotonic()+10
            while self.active():
                if time.monotonic()>limit:raise TimeoutError('Owned job did not terminate within cleanup verification bound')
                time.sleep(.01)


def supervise(command: Sequence[str], payload: dict, cwd: Path, output: Path,
              timeout_s: float) -> dict:
    """One owned process tree;preserve startup/failure/timeout and never retry.

    Caller supplies the audited handshake worker. No scientific preflight is
    bypassed. After a launch,output remains reserved even on failure.
    """
    if not command or not all(isinstance(s,str) for s in command) or not 0 < timeout_s <= 3700:
        raise ValueError('Explicit command and at-most3700shardbound required')
    output.mkdir(parents=True,exist_ok=False)
    launch=dict(command=list(command),cwd=str(cwd.resolve()),timeout_s=timeout_s,
                created_utc=datetime.now(timezone.utc).isoformat(),automatic_retries=0)
    (output/'LAUNCH.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    process=None;job=None;handshake=False;began=time.monotonic()
    result=dict(status='startup_failed',handshake_sent=False,job_assigned=False,timed_out=False)
    try:
        job=OwnedJob()
        job_env=os.environ.copy()
        job_env['PYTHONIOENCODING']='utf-8'
        result['child_stdio_encoding']='utf-8'
        with (output/'worker.log').open('x',encoding='utf-8') as log:
            process=subprocess.Popen(list(command),cwd=str(cwd),stdin=subprocess.PIPE,
                stdout=log,stderr=subprocess.STDOUT,text=True,encoding='utf-8',
                creationflags=subprocess.CREATE_NO_WINDOW,env=job_env)
            job.assign(process)
            result.update(pid=process.pid,job_assigned=True)
            remaining=timeout_s-(time.monotonic()-began)
            if remaining<=0:raise subprocess.TimeoutExpired(command,timeout_s)
            process.stdin.write(json.dumps(payload)+'\n');process.stdin.flush();process.stdin.close()
            handshake=True;result['handshake_sent']=True
            try:
                code=process.wait(timeout=max(.001,timeout_s-(time.monotonic()-began)))
                result.update(exit_code=code,status='completed' if code==0 else 'worker_failed')
            except subprocess.TimeoutExpired:
                result.update(status='hard_timeout',timed_out=True)
            result['owned_active_processes_before_job_close']=job.active()
            job.terminate_remaining()
            result['owned_active_processes_after_termination']=job.active()
            job.close()  # kills only processes assigned to this job and their descendants
            result['job_closed']=True
            if process.poll() is None:process.wait(timeout=10)
            result['exit_code']=process.returncode
    except Exception as error:
        result.update(error=repr(error),handshake_sent=handshake,
                      status='supervision_failed' if handshake else 'startup_failed')
        if job is not None:
            job.close();result['job_closed']=True
        # Before a handshake the worker cannot have started scientific work or
        # descendants. If assignment failed,kill only this retained process handle.
        if process is not None and process.poll() is None:
            process.kill();process.wait(timeout=10)
        if process is not None:result['exit_code']=process.returncode
    finally:
        if job is not None:job.close()
        result.update(elapsed_s=time.monotonic()-began,automatic_retries=0,
                      no_other_processes_targeted=True,output=str(output.resolve()))
        (output/'RESULT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
