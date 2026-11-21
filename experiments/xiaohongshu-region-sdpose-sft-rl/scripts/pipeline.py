"""Finish the new dataset interface, then execute the existing paired pilot."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from prepare import read,ref,write

PY='/home/shangguanrz/nihaochen/envs/qwen21/bin/python'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--repo',type=Path,required=True);a=p.parse_args();root=a.root.resolve();repo=a.repo.resolve()
    scripts=Path(__file__).resolve().parent;root.joinpath('logs').mkdir(exist_ok=True)
    lock=(root/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,PICTUREEDIT_REPO=str(repo),PYTHONPATH=str(repo),CUDA_HOME='/home/shangguanrz/nihaochen/envs/qwen21',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',
        PYTORCH_CUDA_ALLOC_CONF='backend:cudaMallocAsync',LD_LIBRARY_PATH='/data/shangguanrz/仪表盘/compat')
    def event(kind,**fields):
        row={'event':kind,'pid':os.getpid(),'time_utc':datetime.now(timezone.utc).isoformat(),**fields}
        with (root/'logs/events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    def child(label,command):
        log=root/('logs/'+label+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.log')
        with log.open('x') as f:
            process=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
            event('pipeline_child',label=label,child_pid=process.pid,command=command,log=str(log))
            code=process.wait()
        if code:raise RuntimeError(f'{label} failed with exit {code}; {log}')
        event('pipeline_finished',label=label)
    def free_gpu():
        # The short-lived probe releases its CUDA context before the real child.
        command=[PY,'-c','import json,torch; f,t=torch.cuda.mem_get_info(); print(json.dumps([f,t]))']
        waited=False
        while True:
            output=subprocess.check_output(command,env=env,text=True);free,total=json.loads(output.strip().splitlines()[-1])
            if free>=70*2**30:break
            if not waited:event('waiting_for_free_gpu',free_gib=free/2**30,total_gib=total/2**30);waited=True
            time.sleep(30)
        event('gpu_available',free_gib=free/2**30,total_gib=total/2**30)
    event('xiaohongshu_pipeline',repo=str(repo),selection=ref(root/'data/selection.json'))
    if not (root/'data/initialization.json').exists():
        child('migrate_bilibili20k',[PY,str(scripts/'initialization.py'),'--root',str(root),'--repo',str(repo)])
    if not (root/'prepared.json').exists():
        free_gpu();child('prepare_native_assets',[PY,str(scripts/'assets.py'),'--root',str(root),'--repo',str(repo)])
    common=repo/'experiments/bilibili-single-spatial-train-20261107/region-sdpose-sft-rl/scripts'
    report=root/'report/data-regional-S3.json'
    if not report.exists():
        child('preflight_data',[PY,str(common/'stage.py'),'data','--root',str(root),
              '--config',str(root/'configs/templates/regional/S3.json'),'--report',str(report)])
    if read(report).get('passed') is not True:raise RuntimeError('Actual dataset preflight did not pass')
    free_gpu();child('paired_sft_rl',[PY,str(common/'run.py'),'--root',str(root),'--repo',str(repo),'--until','all'])
    comparison=read(root/'report/comparison-ready.json')
    write(root/'pipeline-complete.json',{'comparison':comparison,'prepared':ref(root/'prepared.json'),
          'scope':'one-seed 78/10 Xiaohongshu pilot initialized from Bilibili20k; extra RL rollout cost is separate'})
    event('xiaohongshu_complete',report=str(root/'pipeline-complete.json'))


if __name__=='__main__':main()
