"""Dependency and NVML polling without importing Torch or opening CUDA contexts."""
import hashlib
from pathlib import Path
import subprocess
import time


def process_snapshot(pid,proc_root=Path('/proc')):
    root=Path(proc_root);directory=root/str(pid)
    try:
        fields=(directory/'stat').read_text().rsplit(') ',1)[1].split()
        command=(directory/'cmdline').read_bytes()
        return {'pid':int(pid),'boot_id':(root/'sys/kernel/random/boot_id').read_text().strip(),
                'start_ticks':fields[19],'state':fields[0],
                'command':command.replace(b'\0',b' ').decode(errors='replace'),
                'command_sha256':hashlib.sha256(command).hexdigest()}
    except (FileNotFoundError,ProcessLookupError):return None


def dependency_active(expected,current):
    return current is not None and current['state'] not in ('Z','X') and all(
        current[k]==expected[k] for k in ('pid','boot_id','start_ticks'))


def wait_for_dependency(expected,event,*,snapshot=process_snapshot,sleep=time.sleep,interval=30):
    event('waiting_for_dependency',dependency=expected)
    while True:
        try:current=snapshot(expected['pid'])
        except (OSError,ValueError,IndexError) as error:
            event('dependency_probe_retry',error=str(error));sleep(interval);continue
        if not dependency_active(expected,current):
            event('dependency_finished',dependency=expected,observed=current);return
        sleep(interval)


def query_gpu_memory(env,*,run=subprocess.run):
    command=['nvidia-smi','--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits']
    try:
        result=run(command,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20)
        if result.returncode:return {'error':result.stderr.strip() or 'NVML query failed'}
        lines=result.stdout.strip().splitlines()
        if len(lines)!=1:return {'error':'Expected the one configured GPU'}
        free,total=map(float,lines[0].split(','))
        return {'free_bytes':int(free*2**20),'total_bytes':int(total*2**20)}
    except (OSError,ValueError,subprocess.TimeoutExpired) as error:return {'error':str(error)}


def wait_for_gpu(minimum_gib,env,event,*,sleep=time.sleep,interval=30):
    previous=None
    while True:
        value=query_gpu_memory(env)
        if 'error' in value:
            event('gpu_probe_retry',error=value['error']);sleep(interval);continue
        free=value['free_bytes']/2**30
        if free>=minimum_gib:
            event('gpu_available',free_gib=free,total_gib=value['total_bytes']/2**30);return
        if previous is None or abs(free-previous)>=4:
            event('waiting_for_free_gpu',free_gib=free,required_gib=minimum_gib);previous=free
        sleep(interval)
