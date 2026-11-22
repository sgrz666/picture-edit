"""Keep the frozen trainer, adopt its active child, and defer bulk image work."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PY='/home/shangguanrz/nihaochen/envs/qwen21/bin/python'
SDPY='/data/shangguanrz/pic-edit-experiments/姿态估计/four-skeleton-compair/.venv/bin/python'


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2)


def live(pid):
    p=Path('/proc')/str(pid)
    if not p.exists():return False
    try:return (p/'stat').read_text().split(') ',1)[1][0]!='Z'
    except FileNotFoundError:return False


def adopt_training_report(report,label,adoption,event,wait):
    if report.exists() or adoption is None or adoption['label']!=label:return False
    event('adopting_active_training',label=label,process=adoption['process'])
    wait(adoption['process'],event)
    if not report.exists():raise RuntimeError('Adopted stage exited without its committed training report')
    return True


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--wait-pid',type=int)
    ap.add_argument('--common-dir',type=Path,required=True);ap.add_argument('--adopt-training',type=Path)
    ap.add_argument('--until',choices=['cache','sft','all'],default='all');ap.add_argument('--rl-updates',type=int)
    args=ap.parse_args();root=args.root.resolve();repo=args.repo.resolve();private=Path(__file__).resolve().parent;scripts=args.common_dir.resolve()
    adoption=None if args.adopt_training is None else read(args.adopt_training)
    prepared=read(root/'prepared.json')
    if args.rl_updates is None:args.rl_updates=prepared.get('followup_updates',10)
    os.environ['PICTUREEDIT_REPO']=str(repo);sys.path.insert(0,str(repo))
    from src.pose_control.qwen21_full.native_v62.training.io import file_sha
    from src.pose_control.qwen21_full.native_v7.training import TrainingConfigV7
    env=dict(os.environ,PICTUREEDIT_REPO=str(repo),PYTHONPATH=str(repo),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',CUDA_HOME='/home/shangguanrz/nihaochen/envs/qwen21',
        PYTORCH_CUDA_ALLOC_CONF='backend:cudaMallocAsync')
    import fcntl
    lock=(root/'orchestrator.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('An orchestrator already owns this pilot')
    def ref(path):return {'path':str(Path(path).resolve()),'sha256':file_sha(path)}
    def event(kind,**fields):
        row={'time_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'event':kind,**fields}
        with (root/'logs/events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    def child(label,command):
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');log=root/f'logs/{label}_{stamp}.log'
        event('started',label=label,command=command,log=str(log))
        with log.open('x') as f:
            process=subprocess.Popen(command,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT)
            event('child_pid',label=label,child_pid=process.pid)
            code=process.wait()
        if code:raise RuntimeError(f'{label} failed with exit {code}; inspect {log}')
        event('finished',label=label)
    def stage(action,config,*,report=None,output=None,parent=None,resume=None,label=None,split=None):
        cmd=[PY,str(scripts/'stage.py'),action,'--root',str(root),'--config',str(config)]
        for key,value in (('--report',report),('--output',output),('--parent',parent),('--resume',resume),('--label',label),('--split',split)):
            if value is not None:cmd.extend((key,str(value)))
        child(action+'_'+(label or Path(config).stem),cmd)
    def evaluate(manifest,label):
        pose=root/f'report/evaluation/{label}/sdpose.json';full=root/f'report/evaluation/{label}/metrics.json'
        if not pose.exists():child('sdpose_'+label,[SDPY,str(scripts/'evaluate.py'),'pose','--root',str(root),
            '--manifest',str(manifest),'--output',str(pose)])
        if not full.exists():child('identity_'+label,[PY,str(scripts/'evaluate.py'),'identity','--root',str(root),
            '--input',str(pose),'--output',str(full)])
        return full
    def generate(config,label,split='val'):
        if label!='initial':
            event('bulk_render_deferred',label=label,split=split,config=str(config));return None
        folder=root/f'output/generation/{label}' if split=='val' else root/f'output/generation/test/{label}'
        metric_label=label if split=='val' else 'test_'+label;manifest=folder/'manifest.json'
        if not manifest.exists():stage('generate',config,output=folder,label=metric_label,split=split)
        saved=read(manifest)
        if any(file_sha(x['path'])!=x['sha256'] for x in saved['records']):raise ValueError('Saved generated image changed')
        expected={r['pair_id'] for r in read(root/'data/evaluation.json') if r['split']==split}
        if len(saved['records'])!=len(expected) or {x['pair_id'] for x in saved['records']}!=expected:
            raise ValueError('Generation does not cover the exact locked '+split+' set')
        evaluate(manifest,metric_label)
        return manifest
    def ready_and_train(raw,label,parent=None):
        config_path=root/f'configs/run/{label}.json';gradient=root/f'report/{label}-gradient.json'
        if not config_path.exists():write(config_path,TrainingConfigV7.from_dict(raw).record())
        elif read(config_path)!=json.loads(json.dumps(TrainingConfigV7.from_dict(raw).record())):raise ValueError('Existing config differs: '+label)
        if not gradient.exists():stage('grad',config_path,report=gradient,label=label)
        accepted=read(gradient)
        if accepted.get('passed') is not True or accepted.get('production_passed') is not True:raise RuntimeError('Gradient acceptance failed')
        ready=root/f'configs/run/{label}-ready.json';execution={**read(config_path),'gradient_report':ref(gradient)}
        if not ready.exists():write(ready,execution)
        result_path=root/f'report/{label}-training.json'
        from waiting import wait_for_dependency
        adopt_training_report(result_path,label,adoption,event,wait_for_dependency)
        if not result_path.exists():
            commits=list((Path(raw['output_root'])/'checkpoints').glob('*.commit.json'))
            resume=None
            if commits:
                latest=max((read(p) for p in commits),key=lambda x:x['stage_update'])
                resume=latest['checkpoint']['path']
            stage('train',ready,report=result_path,parent=parent,resume=resume,label=label)
        result=read(result_path)
        from src.pose_control.qwen21_full.native_v62.training.checkpointing import load_committed
        saved,checkpoint,_=load_committed(result['checkpoint'],schema='native_v7_training_v1')
        if saved['training_state']['stage_update']!=raw['stage']['budget_updates']:raise RuntimeError('Stage budget incomplete')
        if saved['training_state']['config_sha256']!=TrainingConfigV7.from_dict(execution).signature():raise ValueError('Stage config signature differs')
        generation=root/f'configs/generation/{label}.json'
        if not generation.exists():write(generation,{**execution,'initialization':checkpoint,'gradient_report':None})
        del saved
        generate(generation,label)
        return checkpoint,ref(gradient),ready
    (root/'logs').mkdir(parents=True,exist_ok=True)
    event('orchestrator',until=args.until,repo=str(repo),wait_pid=args.wait_pid)
    if args.wait_pid and live(args.wait_pid):
        command_path=Path('/proc')/str(args.wait_pid)/'cmdline'
        expected_command=command_path.read_bytes()
        event('waiting_for_existing_gpu_job',wait_pid=args.wait_pid,command=expected_command.decode(errors='replace').replace('\0',' '))
        while live(args.wait_pid):
            try:
                if command_path.read_bytes()!=expected_command:break
            except FileNotFoundError:break
            time.sleep(30)
        event('existing_gpu_job_finished',wait_pid=args.wait_pid)
    cache=root/'cache_ready.json'
    if not cache.exists():stage('cache',root/'configs/templates/uniform/S0.json',label='sdpose_new_cache')
    receipt=read(cache)
    if file_sha(receipt['cache']['path'])!=receipt['cache']['sha256']:raise ValueError('Frozen cache index changed')
    if args.until=='cache':event('cache_complete');return
    generate(root/'configs/cache-ready/uniform/S0.json','initial')
    final={}
    for arm in ('uniform','regional'):
        parent=gate=None
        for phase in ('S0','S1','S2','S3'):
            raw=read(root/f'configs/cache-ready/{arm}/{phase}.json')
            if parent is not None:raw['initialization']=parent
            raw['body_baseline_report']=gate;label=arm+'_'+phase
            parent,gradient,ready=ready_and_train(raw,label,None if parent is None else parent['path'])
            if phase=='S0':
                gate_path=root/f'report/{arm}-body-technical-gate.json'
                if not gate_path.exists():write(gate_path,{'schema':'native_v7_body_pilot_gate_v1','body_accepted':True,
                    'manual_acknowledgment':False,'acceptance_kind':'technical_pilot_gate',
                    'checkpoint_sha256':parent['sha256'],'s0_gradient_report':gradient,
                    'continuous_stage_authorization':raw['continuous_stage_authorization'],
                    'scope':'completed bounded S0 with real gradients/frozen checks; not a human image-quality approval'})
                gate=ref(gate_path)
        final[arm]=parent
    if not (root/'sft_complete.json').exists():write(root/'sft_complete.json',{'checkpoints':final,'updates_per_arm':prepared['updates_per_sft_arm']})
    if args.until=='sft':event('sft_complete');return
    raw=read(root/'configs/cache-ready/regional/S3.json');raw['initialization']=final['regional']
    raw['stage']['stage_id']='S3_region_followup';raw['stage']['budget_updates']=args.rl_updates
    raw['save_every_updates']=min(5,args.rl_updates);raw['probe_every_updates']=args.rl_updates
    raw['output_root']=str(root/'output/continue_sft');raw['body_baseline_report']=None
    continued,_,_=ready_and_train(raw,'continue_sft',final['regional']['path'])
    rl_raw={**raw,'output_root':str(root/'output/grpo')}
    rl_path=root/'configs/run/grpo.json';gradient=root/'report/grpo-gradient.json'
    if not rl_path.exists():write(rl_path,TrainingConfigV7.from_dict(rl_raw).record())
    if not gradient.exists():stage('grad',rl_path,report=gradient,label='grpo')
    ready=root/'configs/run/grpo-ready.json'
    if not ready.exists():write(ready,{**read(rl_path),'gradient_report':ref(gradient)})
    rl_report=root/'report/grpo-training.json'
    if not rl_report.exists():
        command=[PY,str(scripts/'rl.py'),'--root',str(root),'--config',str(ready),
            '--parent',final['regional']['path'],'--report',str(rl_report),'--updates',str(args.rl_updates)]
        commits=sorted((root/'output/grpo/checkpoints').glob('*.commit.json'))
        if commits:
            latest=max((read(p) for p in commits),key=lambda x:x['completed_updates'])
            command.extend(('--resume',latest['checkpoint']['path']))
        child('grpo',command)
    rl_result=read(rl_report);generation=root/'configs/generation/grpo.json'
    if not generation.exists():write(generation,{**read(ready),'initialization':rl_result['checkpoint'],'gradient_report':None})
    generate(generation,'grpo')
    test_manifests={}
    result={'sft':final,'continue_sft':continued,'grpo':rl_result,'test_manifests':test_manifests,
            'scope':'single-seed fixed-budget experiment; test excluded from optimization and selection'}
    if not (root/'optimization-complete.json').exists():write(root/'optimization-complete.json',result)
    child('deferred_image_rendering',[PY,str(private/'deferred_render.py'),'run','--root',str(root),'--repo',str(repo),
          '--common-dir',str(scripts)])
    comparison=root/'report/comparison-ready.json'
    if not comparison.exists():
        output=root/('report/comparison_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        child('comparison',[PY,str(scripts/'summarize.py'),'--root',str(root),'--output',str(output)])
        write(comparison,{'output':str(output),'metrics':ref(output/'comparison.json'),
            'overview':ref(output/'overview.png'),'readme':ref(output/'README.md')})
    if prepared.get('test_pairs',0) and not (root/'report/test-comparison-ready.json').exists():
        output=root/('report/test-comparison_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        child('test_comparison',[PY,str(scripts/'summarize.py'),'--root',str(root),'--output',str(output),'--split','test'])
        write(root/'report/test-comparison-ready.json',{'output':str(output),'metrics':ref(output/'comparison.json'),
            'overview':ref(output/'overview.png'),'readme':ref(output/'README.md')})
    for name in ('metrics','overview','readme'):
        asset=read(comparison)[name]
        if file_sha(asset['path'])!=asset['sha256']:raise ValueError('Final comparison artifact changed: '+name)
    event('complete',report=str(root/'complete.json'))


if __name__=='__main__':main()
