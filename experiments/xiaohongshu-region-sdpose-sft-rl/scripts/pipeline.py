"""Finish the new dataset interface, then execute the existing paired pilot."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from prepare import read,ref,write
from waiting import wait_for_dependency,wait_for_gpu

PY='/home/shangguanrz/nihaochen/envs/qwen21/bin/python'


def main():
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
    from src.pose_control.qwen21_full.native_rl.regional_cli import add_arguments, configure_arguments

    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--repo',type=Path,required=True)
    add_arguments(p,launcher=True)
    a=p.parse_args();root=a.root.resolve();repo=a.repo.resolve()
    reward_flags=configure_arguments(p,a)
    if a.rl_python is not None:
        reward_flags+=['--rl-python',str(a.rl_python.absolute())]
    import fcntl
    scripts=Path(__file__).resolve().parent;root.joinpath('logs').mkdir(exist_ok=True)
    request=read(root/'launch_request.json') if (root/'launch_request.json').exists() else {}
    lock=(root/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,PICTUREEDIT_REPO=str(repo),PYTHONPATH=str(repo),CUDA_HOME='/home/shangguanrz/nihaochen/envs/qwen21',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',
        PYTORCH_CUDA_ALLOC_CONF='backend:cudaMallocAsync',LD_LIBRARY_PATH='/data/shangguanrz/仪表盘/compat',
        REGION_PILOT_MEMORY_FRACTION=str(request.get('memory_fraction',.9)),YOLO_CONFIG_DIR=str(root/'logs/yolo-settings'))
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
        wait_for_gpu(request.get('minimum_free_gpu_gib',70),env,event)
    event('xiaohongshu_pipeline',repo=str(repo),selection=ref(root/'data/selection.json'))
    if request.get('dependency'):wait_for_dependency(request['dependency'],event)
    for name in ('initialization','source_checkpoint','reuse_geometry'):
        bound=request.get(name)
        if bound is not None and ref(bound['path'])!=bound:raise ValueError('Pinned launch input changed: '+name)
    if not (root/'data/initialization.json').exists():
        child('migrate_bilibili20k',[PY,str(scripts/'initialization.py'),'--root',str(root),'--repo',str(repo)])
    if not (root/'data/geometry.json').exists():
        free_gpu();command=['/home/shangguanrz/project/pic-edit/add/envs/smplestx_bw/bin/python',str(scripts/'geometry.py'),
            '--root',str(root),'--repo',str(repo),'--smplest-root','/home/shangguanrz/project/pic-edit/add/tools/SMPLest-X']
        if request.get('reuse_geometry'):command.extend(('--reuse-index',request['reuse_geometry']['path']))
        child('new_frame_geometry',command)
    if not (root/'prepared.json').exists():
        free_gpu();child('prepare_native_assets',[PY,str(scripts/'assets.py'),'--root',str(root),'--repo',str(repo)])
    common=repo/'experiments/bilibili-single-spatial-train-20261107/region-sdpose-sft-rl/scripts'
    report=root/'report/data-regional-S3.json'
    if not report.exists():
        child('preflight_data',[PY,str(common/'stage.py'),'data','--root',str(root),
              '--config',str(root/'configs/templates/regional/S3.json'),'--report',str(report)])
    if read(report).get('passed') is not True:raise RuntimeError('Actual dataset preflight did not pass')
    free_gpu();child('paired_sft_rl',[PY,str(common/'run.py'),'--root',str(root),'--repo',str(repo),'--until','all',
        '--rl-updates',str(request.get('followup_updates',10)),*reward_flags])
    comparison=read(root/'report/comparison-ready.json')
    write(root/'pipeline-complete.json',{'comparison':comparison,'prepared':ref(root/'prepared.json'),
          'scope':'fixed-budget one-seed Xiaohongshu experiment from Bilibili20k; test excluded from optimization; RL rollout cost separate'})
    event('xiaohongshu_complete',report=str(root/'pipeline-complete.json'))


if __name__=='__main__':main()
