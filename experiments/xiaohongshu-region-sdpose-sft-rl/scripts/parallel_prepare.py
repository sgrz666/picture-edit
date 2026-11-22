"""Run Native assets and independent frozen Qwen encoding concurrently."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--overlay',type=Path,required=True)
    p.add_argument('--adopt-cache',type=Path);a=p.parse_args()
    private=Path(__file__).resolve().parent;sys.path.insert(0,str(private))
    from prepare import read,ref,sha,write
    from waiting import wait_for_dependency,wait_for_gpu,process_snapshot,dependency_active
    import fcntl
    lock=(a.root/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    request=read(a.root/'launch_request.json')
    overlay=read(a.overlay/'manifest.json')
    for row in overlay['files']:
        if sha(a.overlay/row['path'])!=row['sha256']:raise ValueError('Dependency snapshot changed')
    py='/home/shangguanrz/nihaochen/envs/qwen21/bin/python'
    env=dict(os.environ,PICTUREEDIT_REPO=str(a.repo),PYTHONPATH=str(a.repo)+os.pathsep+str(a.overlay),
        CUDA_HOME='/home/shangguanrz/nihaochen/envs/qwen21',LD_LIBRARY_PATH='/data/shangguanrz/仪表盘/compat',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',
        PYTORCH_CUDA_ALLOC_CONF='backend:cudaMallocAsync',YOLO_CONFIG_DIR=str(a.root/'logs/yolo-settings'))
    def event(kind,**fields):
        row={'event':kind,'pid':os.getpid(),'time_utc':datetime.now(timezone.utc).isoformat(),**fields}
        with (a.root/'logs/events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    def start(label,command,fraction):
        child_env=dict(env,REGION_PILOT_MEMORY_FRACTION=str(fraction))
        log=a.root/('logs/'+label+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.log')
        with log.open('x') as f:process=subprocess.Popen(command,cwd=a.repo,env=child_env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        event('parallel_child',label=label,child_pid=process.pid,command=command,log=str(log),memory_fraction=fraction)
        return process,log
    def one(label,command,fraction=.35):
        process,log=start(label,command,fraction)
        if process.wait():raise RuntimeError(f'{label} failed; inspect {log}')
        event('parallel_finished',label=label)
    event('parallel_preparation_started',source_snapshot=ref(a.repo/'source_manifest.json'),overlay=ref(a.overlay/'manifest.json'))
    wait_for_dependency(request['dependency'],event)
    for name in ('initialization','source_checkpoint'):
        if ref(request[name]['path'])!=request[name]:raise ValueError('Pinned model changed')
    one('canonical_qwen_inputs',[py,str(private/'cache_preparation.py'),'prepare','--root',str(a.root),'--repo',str(a.repo)])
    wait_for_gpu(20 if a.adopt_cache else 50,env,event)
    jobs={}
    if not (a.root/'prepared.json').exists():
        jobs['native_assets']=start('native_assets_parallel',[py,str(private/'assets.py'),'--root',str(a.root),'--repo',str(a.repo)],.20)
    if a.adopt_cache and not (a.root/'report/independent-cache-complete.json').exists():
        expected=read(a.adopt_cache)
        class ExistingCache:
            def wait(self):
                while dependency_active(expected,process_snapshot(expected['pid'])):time.sleep(5)
                return 0 if (a.root/'report/independent-cache-complete.json').exists() else 1
        jobs['qwen_cache']=(ExistingCache(),expected['log'])
        event('adopted_cache_worker',identity=expected)
    elif not (a.root/'report/independent-cache-complete.json').exists():
        jobs['qwen_cache']=start('qwen_cache_parallel',[py,str(private/'cache_preparation.py'),'encode','--root',str(a.root),'--repo',str(a.repo)],.35)
    # Both are awaited; a failure in one does not discard valid work from the other.
    failures=[]
    for label,(process,log) in jobs.items():
        code=process.wait();event('parallel_finished',label=label,exit_code=code)
        if code:failures.append({'label':label,'log':str(log),'exit_code':code})
    if failures:
        write(a.root/('report/parallel-failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'),failures)
        raise RuntimeError('Parallel preparation failed; successful artifacts retained: '+json.dumps(failures))
    one('finalize_cache_binding',[py,str(private/'cache_preparation.py'),'finalize','--root',str(a.root),'--repo',str(a.repo)])
    write(a.root/'report/parallel-preparation-complete.json',{'prepared':ref(a.root/'prepared.json'),
        'cache_ready':ref(a.root/'cache_ready.json'),'canonical_inputs':ref(a.root/'data/qwen_condition_index.json'),
        'overlay':ref(a.overlay/'manifest.json'),'native_and_cache_bindings_verified':True})
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    one('resume_training_after_parallel',[py,str(private/'pipeline.py'),'--root',str(a.root),'--repo',str(a.repo)],request['memory_fraction'])
    event('parallel_pipeline_complete',report=str(a.root/'pipeline-complete.json'))


if __name__=='__main__':main()
