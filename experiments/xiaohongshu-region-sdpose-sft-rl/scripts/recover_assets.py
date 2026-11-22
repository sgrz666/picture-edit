"""Recover SAM asset preparation with a pinned pure-Python dependency overlay."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--overlay',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.repo/'experiments/xiaohongshu-region-sdpose-sft-rl/scripts'))
    from prepare import read,ref,write,sha
    from waiting import wait_for_dependency,wait_for_gpu
    import fcntl
    lock=(a.root/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    overlay=read(a.overlay/'manifest.json')
    for row in overlay['files']:
        if sha(a.overlay/row['path'])!=row['sha256']:raise ValueError('Dependency snapshot changed')
    if overlay['environment_modified'] or overlay['training_frameworks_copied']:
        raise ValueError('Only an isolated pure-Python overlay is allowed')
    request=read(a.root/'launch_request.json')
    py='/home/shangguanrz/nihaochen/envs/qwen21/bin/python'
    common=a.repo/'experiments/xiaohongshu-region-sdpose-sft-rl/scripts'
    env=dict(os.environ,PICTUREEDIT_REPO=str(a.repo),PYTHONPATH=str(a.repo)+os.pathsep+str(a.overlay),
        CUDA_HOME='/home/shangguanrz/nihaochen/envs/qwen21',LD_LIBRARY_PATH='/data/shangguanrz/仪表盘/compat',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',
        PYTORCH_CUDA_ALLOC_CONF='backend:cudaMallocAsync',REGION_PILOT_MEMORY_FRACTION=str(request['memory_fraction']),
        YOLO_CONFIG_DIR=str(a.root/'logs/yolo-settings'))
    def event(kind,**fields):
        row={'event':kind,'pid':os.getpid(),'time_utc':datetime.now(timezone.utc).isoformat(),**fields}
        with (a.root/'logs/events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    def child(label,command):
        log=a.root/('logs/'+label+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.log')
        with log.open('x') as f:
            process=subprocess.Popen(command,cwd=a.repo,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
            event('recovery_child',label=label,child_pid=process.pid,command=command,log=str(log))
            code=process.wait()
        if code:raise RuntimeError(f'{label} failed ({code}); {log}')
        event('recovery_finished',label=label)
    event('recover_sam_dependencies',overlay=ref(a.overlay/'manifest.json'),source_snapshot=ref(a.repo/'source_manifest.json'),
          asset_memory_fraction=.20,minimum_asset_free_gib=20,training_memory_fraction=request['memory_fraction'])
    wait_for_dependency(request['dependency'],event)
    for key in ('initialization','source_checkpoint'):
        if ref(request[key]['path'])!=request[key]:raise ValueError('Pinned model changed: '+key)
    if not (a.root/'prepared.json').exists():
        # SAM smoke used 1.59 GiB. DINO FP32 and rasterization get an isolated
        # 20%-of-device cap; the original training memory gate remains unchanged.
        wait_for_gpu(20,env,event)
        env['REGION_PILOT_MEMORY_FRACTION']='.20'
        child('prepare_native_assets_overlay',[py,str(common/'assets.py'),'--root',str(a.root),'--repo',str(a.repo)])
    write(a.root/'report/sam-recovery-complete.json',{'prepared':ref(a.root/'prepared.json'),
        'dependency_manifest':ref(a.overlay/'manifest.json'),'source_snapshot':ref(a.repo/'source_manifest.json'),
        'qwen_environment_modified':False,'sam_gpu_smoke':ref(a.root/'report/sam-smoke-v1.json')})
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    env['REGION_PILOT_MEMORY_FRACTION']=str(request['memory_fraction'])
    child('resume_training_pipeline',[py,str(common/'pipeline.py'),'--root',str(a.root),'--repo',str(a.repo)])
    event('recovered_pipeline_complete',report=str(a.root/'pipeline-complete.json'))


if __name__=='__main__':main()
