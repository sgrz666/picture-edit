"""Serial, resumable experiment controller. A completed run is immutable."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .common import MODEL, PROJECT, TASKS, quality_manifest, training_manifest, sha256, write_json

def preflight(root):
    from .common import environment
    expected={
      'transformer/diffusion_pytorch_model.safetensors':'54078ee51ab477275e94b610249ea912d6f80d24256c4197fe48945eef706883',
      'connector/model.safetensors':'425ce59f596eeb573f9cd52138b775df47ce69c60d2782fb32cac5498ed208f3',
      'vae/diffusion_pytorch_model.safetensors':'5a170129752b4ad4ab98996d2e59097c8c693afc31a5964bd2ad60896686c1e2',
      'vlm/model-00001-of-00002.safetensors':'d6f11cc9c30184ce92605f25ac1e6b6644ee217e7b86a28282b3c9b17de3e609',
      'vlm/model-00002-of-00002.safetensors':'c675c19a231b0ead141d8d3a05fe930892ddf17f1324d75bb94d2a42a79f7ebc'}
    actual={name:sha256(MODEL/name) for name in expected}
    result=dict(environment=environment(),weights=actual,weights_valid=actual==expected,
                source_hashes={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')})
    write_json(root/'preflight.json',result)
    assert result['weights_valid'],'Weight hash mismatch'
    print(json.dumps(result),flush=True)

def idle():
    # nvidia-smi subprocess cannot create a CUDA context in the controller.
    p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    return not p.stdout.strip()

def used_budget(root,quality=False):
    offset=root/'budget_carryover.json'
    total=json.loads(offset.read_text()).get('quality_seconds' if quality else 'total_seconds',0) if offset.exists() else 0
    for path in (root/'runs').glob('*/summary.json'):
        config=json.loads((path.parent/'config.json').read_text())
        if not quality or config['kind']=='quality':
            total+=json.loads(path.read_text()).get('process_wall_seconds',0)
    return total

def execute(root,config,timeout=2700):
    root.mkdir(parents=True,exist_ok=True)
    run=root/'runs'/config['id']
    if (run/'summary.json').exists():
        previous=json.loads((run/'summary.json').read_text())
        if previous['status']=='NOT_RUN':
            archive=root/'attempts';archive.mkdir(exist_ok=True)
            run.rename(archive/(run.name+'_'+str(time.time_ns())))
        else:
            saved=json.loads((run/'config.json').read_text())
            if any(saved.get(k)!=v for k,v in config.items()):
                raise ValueError(f'Run ID {config["id"]} already exists with different configuration')
            return previous
    if run.exists():
        raise RuntimeError(f'Incomplete previous run: {run}; inspect before retry under a new ID')
    if used_budget(root)>72*3600 or (config['kind']=='quality' and used_budget(root,True)>6*3600):
        run.mkdir(parents=True)
        write_json(run/'config.json',config)
        write_json(run/'summary.json',dict(status='NOT_RUN',reason='GPU budget exhausted',process_wall_seconds=0))
        return json.loads((run/'summary.json').read_text())
    started_wait=time.time()
    while not idle():
        if time.time()-started_wait>6*3600:
            raise RuntimeError('GPU remained busy for 6 hours; no other processes were stopped')
        print('Waiting for existing GPU task to finish',flush=True)
        time.sleep(30)
    cfg=root/'configs'/(config['id']+'.json')
    write_json(cfg,config)
    logs=root/'logs';logs.mkdir(exist_ok=True)
    log_out=logs/(config['id']+'.stdout.log');log_err=logs/(config['id']+'.stderr.log')
    print('RUN',config['id'],flush=True)
    start=time.perf_counter()
    remaining=72*3600-used_budget(root)
    if config['kind']=='quality':
        remaining=min(remaining,6*3600-used_budget(root,True))
    timeout=max(1,min(timeout,remaining))
    with log_out.open('w') as fout,log_err.open('w') as ferr:
        env=dict(os.environ,PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        proc=subprocess.Popen([sys.executable,'-m','vram_lab.worker','--root',str(root),'--config',str(cfg)],
                              stdout=fout,stderr=ferr,env=env,start_new_session=True)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try: proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            run.mkdir(parents=True,exist_ok=True)
            write_json(run/'config.json',config)
            write_json(run/'summary.json',dict(status='TIMEOUT',process_wall_seconds=time.perf_counter()-start))
        except KeyboardInterrupt:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            run.mkdir(parents=True,exist_ok=True)
            write_json(run/'config.json',config)
            write_json(run/'summary.json',dict(status='CANCELLED',process_wall_seconds=time.perf_counter()-start))
            shutil.copy2(log_out,run/'stdout.log');shutil.copy2(log_err,run/'stderr.log')
            raise
    run.mkdir(parents=True,exist_ok=True)
    shutil.copy2(log_out,run/'stdout.log');shutil.copy2(log_err,run/'stderr.log')
    if not (run/'summary.json').exists():
        write_json(run/'config.json',config)
        write_json(run/'summary.json',dict(status='INVALID',error=f'worker exit {proc.returncode}',process_wall_seconds=time.perf_counter()-start))
    result=json.loads((run/'summary.json').read_text())
    print('RESULT',config['id'],result['status'],result.get('peak_nvml_gib'),flush=True)
    return result

def caches(root,resolutions):
    from .worker import cache_key
    for res in resolutions:
        for task in ('S0','D0'):
            config=dict(id=f'CACHE_{res}_{task}',kind='cache',group='cache',task=task,resolution=res,seed=1701,
                        preprocess='letterbox',prompt=TASKS[task],control_injected=False)
            config['id']+='_'+cache_key(config)[0][:10]
            result=execute(root,config)
            if result['status']!='PASS':
                raise RuntimeError(f'Cache failed: {config["id"]}')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['preflight','quality','cache','smoke','main','stable','sensitivity','q4','infer','reproduce'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--groups',default='Q0,Q1,Q2,Q3')
    parser.add_argument('--resolution',type=int,nargs='+',default=[512,768,1024])
    parser.add_argument('--limit',type=int)
    args=parser.parse_args();root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (PROJECT/'vram_lab_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.command=='preflight':
            preflight(root);return
        if not (root/'preflight.json').exists():
            raise RuntimeError('Run preflight first')
        if not json.loads((root/'preflight.json').read_text()).get('weights_valid'):
            raise RuntimeError('Preflight weight validation failed')
        if args.command=='cache':
            caches(root,args.resolution);return
        if args.command=='quality':
            configs=[c for c in quality_manifest() if c['group'] in args.groups.split(',')]
        elif args.command=='reproduce':
            configs=[dict(quality_manifest()[0],id='REPRO_Q0_S0_42',group='reproducibility')]
        elif args.command=='infer':
            configs=[]
            for res in args.resolution:
                checkpoint=root/'runs'/f'STABLE_M_{res}_0'/'adapter.pt'
                for task in ('S0','D0'):
                    if not checkpoint.exists():
                        missing=root/'runs'/f'ADAPTER_INFER_{res}_{task}'
                        if not (missing/'summary.json').exists():
                            write_json(missing/'config.json',dict(id=missing.name,kind='adapter_infer',group='adapter_infer',task=task,resolution=res))
                            write_json(missing/'summary.json',dict(status='NOT_RUN',reason='No stable adapter checkpoint',process_wall_seconds=0))
                        continue
                    configs.append(dict(id=f'ADAPTER_INFER_{res}_{task}',kind='adapter_infer',group='adapter_infer',task=task,
                                        seed=42,resolution=res,preprocess='letterbox',prompt=TASKS[task],negative_prompt='',
                                        cfg=4.5,steps=35,adapter_checkpoint=str(checkpoint),control_injected=True))
        elif args.command=='q4':
            selected=json.loads((root/'quality_selection.json').read_text())
            configs=[]
            for task in ('S2','D1'):
                for res in (768,1024):
                    for seed in (42,1024):
                        params=selected['parameters']
                        configs.append(dict(id=f'Q4_{task}_{res}_{seed}',kind='quality',group='Q4',task=task,seed=seed,
                                            resolution=res,preprocess='letterbox',prompt=TASKS[task],control_injected=False,**params))
        elif args.command=='stable':
            configs=[]
            for res in args.resolution:
                candidates=[]
                for enc in ('cache','online'):
                    for ckpt in (True,False):
                        pair=[]
                        for task in ('S0','D0'):
                            path=root/'runs'/f'T_M_{res}_{enc}_{int(ckpt)}_{task}'/'summary.json'
                            if path.exists():
                                s=json.loads(path.read_text())
                                if s['status']=='PASS' and s.get('validation_passed') and not s.get('nvml_error') and (s.get('peak_nvml_gib') or 0)>0: pair.append(s['peak_nvml_gib'])
                        if len(pair)==2:candidates.append((max(pair),enc,ckpt))
                if not candidates:
                    for repeat in range(3):
                        missing=root/'runs'/f'STABLE_M_{res}_{repeat}'
                        if (missing/'summary.json').exists():continue
                        write_json(missing/'config.json',dict(id=missing.name,kind='train',group='stable',resolution=res,adapter_size='M'))
                        write_json(missing/'summary.json',dict(status='NOT_RUN',reason='No eligible main configuration for both photos',process_wall_seconds=0))
                    print(f'No valid configuration at {res}',flush=True);continue
                _,enc,ckpt=min(candidates)
                for repeat in range(3):
                    configs.append(dict(id=f'STABLE_M_{res}_{repeat}',kind='train',group='stable',task='S0',seed=42+repeat,
                                        resolution=res,encoding=enc,checkpointing=ckpt,adapter_size='M',alternate=True,
                                        preprocess='letterbox',prompt=TASKS['S0'],warmup=5,updates=95,control_injected=True))
        else:
            configs=[c for c in training_manifest(args.command=='sensitivity') if c['resolution'] in args.resolution]
            if args.command=='smoke':
                configs=[dict(configs[0],id='SMOKE_M_512',resolution=512,warmup=0,updates=3,group='smoke')]
        if args.limit:configs=configs[:args.limit]
        write_json(root/(args.command+'_manifest.json'),configs)
        for config in configs:
            result=execute(root,config,timeout=7200 if config.get('alternate') else 2700)
            if result['status']=='INVALID':
                raise RuntimeError(f'Invalid job {config["id"]}; repair before continuing matrix')

if __name__=='__main__':main()
