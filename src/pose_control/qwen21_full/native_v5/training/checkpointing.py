"""Weights-only-safe RNG/state and immutable committed training snapshots."""
from pathlib import Path
import random,uuid,sys,platform
import numpy as np
import torch
from .config import SCHEMA,digest
from .io import verify,write_json,file_sha,verify_prefix,sync_directory

def capture_rng():
    n=np.random.get_state()
    return {'python':random.getstate(),'numpy':{'algorithm':n[0],'keys':torch.tensor(n[1].astype(np.int64)),'position':n[2],'has_gauss':n[3],'cached_gaussian':n[4]},
        'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}

def restore_rng(state):
    random.setstate(state['python']);n=state['numpy']
    np.random.set_state((n['algorithm'],n['keys'].cpu().numpy().astype(np.uint32),n['position'],n['has_gauss'],n['cached_gaussian']))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        if torch.cuda.device_count()!=len(state['cuda']):raise ValueError('CUDA RNG topology changed')
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])

def environment():
    import diffusers,transformers
    return {'python':sys.version,'platform':platform.system(),'torch':str(torch.__version__),'cuda':torch.version.cuda,
        'diffusers':diffusers.__version__,'transformers':transformers.__version__,
        'devices':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        'capabilities':[list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())],
        'deterministic':torch.are_deterministic_algorithms_enabled(),'cudnn_benchmark':torch.backends.cudnn.benchmark,
        'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32}

def save_snapshot(root,runtime,optimizer,scheduler,sampler,noise_generator,*,config,identity,catalog,active,
                  global_update,stage_update,attempts,journal,micro_position,gradient_report,stage_origin,parent=None,frozen_state=None):
    if micro_position!=0:raise ValueError('Only optimizer-update boundary snapshots are supported')
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    payload=runtime.checkpoint_payload(metadata={'training_interface':SCHEMA})
    payload['training_schema']=SCHEMA
    payload['training_state']={'config':config.record(),'config_sha256':config.signature(),'identity':identity,
        'identity_sha256':digest(identity),'catalog':catalog.record(),'catalog_sha256':catalog.signature(),
        'active_names':sorted(active),'optimizer_state':optimizer.state_dict(),'scheduler_state':scheduler.state_dict(),
        'sampler_state':sampler.state_dict(),'noise_generator_state':noise_generator.get_state(),
        'rng':capture_rng(),'environment':environment(),'global_update':global_update,'stage_update':stage_update,
        'attempts':attempts,'micro_position':0,'journal':journal.state_dict(),'gradient_report':gradient_report,
        'stage_origin':stage_origin,'parent':parent}
    payload['training_state']['frozen_state']=frozen_state
    path=root/(f'update_{global_update:08d}_'+uuid.uuid4().hex+'.pt')
    with path.open('xb') as f:
        import os
        torch.save(payload,f);f.flush();os.fsync(f.fileno())
    sync_directory(root)
    ref={'path':str(path.resolve()),'sha256':file_sha(path)}
    commit={'schema':'native_v5_training_commit_v1','checkpoint':ref,'global_update':global_update,'stage_update':stage_update,
        'training_state_sha256':digest({'config':config.signature(),'identity':digest(identity),'catalog':catalog.signature(),'journal':journal.state_dict()})}
    marker=write_json(path.with_suffix('.commit.json'),commit)
    return {'checkpoint':ref,'commit':marker}

def load_committed(reference):
    ref=verify(reference);marker=Path(ref['path']).with_suffix('.commit.json')
    if not marker.is_file():raise ValueError('Checkpoint has no completed commit record; use explicit initialization for model-only files')
    import json
    commit=json.loads(marker.read_text())
    if commit.get('schema')!='native_v5_training_commit_v1' or commit['checkpoint']!=ref:raise ValueError('Checkpoint/commit SHA mismatch')
    saved=torch.load(ref['path'],map_location='cpu',weights_only=True,mmap=True)
    if saved.get('training_schema')!=SCHEMA or 'training_state' not in saved:raise ValueError('Old V5 optimizer-only snapshots are initialization-only')
    state=saved['training_state']
    if state.get('micro_position')!=0:raise ValueError('Mid-accumulation resume is not supported')
    expected=digest({'config':state['config_sha256'],'identity':state['identity_sha256'],'catalog':state['catalog_sha256'],'journal':state['journal']})
    if commit.get('training_state_sha256')!=expected:raise ValueError('Training commit signature differs')
    tail=verify_prefix(state['journal'])
    return saved,ref,tail

def validate_resume(saved,config,identity,catalog):
    state=saved['training_state']
    from .config import TrainingConfig
    if state['config_sha256']!=config.signature() or TrainingConfig.from_dict(state['config']).signature()!=config.signature():raise ValueError('Resume config differs; use explicit advance-stage/revision')
    if state['identity_sha256']!=digest(identity) or digest(state['identity'])!=digest(identity):raise ValueError('Resume data/model/code/cache/evaluator identity differs')
    if state['catalog_sha256']!=catalog.signature():raise ValueError('Resume optimizer parameter catalog differs')
    if digest(state['environment'])!=digest(environment()):raise ValueError('Strict resume environment/backend/topology differs')
    if state['active_names']!=sorted(catalog.active_names(config.stage)):raise ValueError('Resume precise frozen mask differs')
    return state
