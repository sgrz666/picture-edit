#!/usr/bin/env python3
"""Continue the interrupted structured pilot from its committed step200.

This operational source stays outside the frozen implementation. It never edits
that implementation or opens a test manifest. All writes to RUN are deferred
until replay, 300-step proofs, and the unchanged evaluator contract pass.
Launch with a detached process and stdout/stderr connected to a durable file.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import gc
import hashlib
import importlib
from importlib.metadata import version
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace

import torch


ROOT=Path('/home/shangguanrz/project/pic-edit')
WORK=ROOT/'research/qwen21_dual_prompt_pilot_20261013'
RUN=ROOT/'experiments/4.two man /qwen21_dual_prompt_pilot/20261013_v1'
RECOVERY_SCHEMA='qwen21_dual_prompt_operational_recovery_v1'
TRAINING_SCHEMA='qwen21_dual_prompt_training_complete_v1'


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def tensor_digest(value):
    def describe(item):
        if torch.is_tensor(item):
            raw=item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            return dict(dtype=str(item.dtype),shape=list(item.shape),sha256=hashlib.sha256(raw).hexdigest())
        if isinstance(item,dict):return {str(key):describe(child) for key,child in item.items()}
        if isinstance(item,(tuple,list)):return [describe(child) for child in item]
        if item is None or type(item) in (str,int,float,bool):return item
        raise ValueError(f'unsupported proof value: {type(item).__name__}')
    return hashlib.sha256(json.dumps(describe(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path,value):
    atomic_bytes(path,(json.dumps(value,indent=2,allow_nan=False)+'\n').encode())


def atomic_bytes(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as stream:
        pending=Path(stream.name)
        stream.write(value);stream.flush();os.fsync(stream.fileno())
    try:os.replace(pending,path)
    finally:pending.unlink(missing_ok=True)


def atomic_copy(source,target):
    source=Path(source);target=Path(target)
    with tempfile.NamedTemporaryFile(dir=target.parent,prefix=target.name+'.',suffix='.tmp',delete=False) as stream:
        pending=Path(stream.name)
        with source.open('rb') as original:shutil.copyfileobj(original,stream)
        stream.flush();os.fsync(stream.fileno())
    try:
        if file_sha(pending)!=file_sha(source):raise ValueError('atomic artifact copy SHA mismatch')
        os.replace(pending,target)
    finally:pending.unlink(missing_ok=True)


def withdraw_own_completion_markers(published):
    """Withdraw only exact bytes from this operation, never another writer's files."""
    for name,expected in published.items():
        path=Path(name)
        if path.exists() and file_sha(path)==expected:path.unlink()


def publish_completion_fail_closed(markers,validate):
    if any(Path(path).exists() for path in markers):
        raise ValueError('existing completion marker cannot be overwritten by recovery')
    planned={str(path):hashlib.sha256((json.dumps(value,indent=2,allow_nan=False)+'\n').encode()).hexdigest()
             for path,value in markers.items()}
    try:
        for path,value in markers.items():atomic_json(path,value)
        validate()
        return planned
    except BaseException:
        # Include planned markers even when an atomic writer raises immediately
        # after replacement, before it can return control to this function.
        withdraw_own_completion_markers(planned)
        raise


def require_equal_contract(expected,current):
    for key in sorted(set(expected)|set(current)):
        if key not in expected or key not in current or expected[key]!=current[key]:
            raise ValueError(f'recovery contract mismatch: {key}')


def verify_files(expected):
    for name,digest in expected.items():
        if file_sha(name)!=digest:raise ValueError(f'input SHA changed: {name}')


def optimizer_proof(adapter,optimizer,names,*,expected_step,check_moments=True,hash_content=True):
    actual=[name for name,param in adapter.named_parameters() if param.requires_grad]
    params=[param for param in adapter.parameters() if param.requires_grad]
    if list(names)!=actual or len(optimizer.param_groups)!=1 or any(a is not b for a,b in zip(optimizer.param_groups[0]['params'],params)) or len(optimizer.param_groups[0]['params'])!=len(params):
        raise ValueError('optimizer parameter names/order mismatch')
    if expected_step==0 and not optimizer.state:
        return dict(step=0,state_count=0,parameter_count=len(params),parameter_names=actual)
    if set(optimizer.state)!=set(params):raise ValueError('optimizer states are incomplete or unexpected')
    for name,param in zip(actual,params):
        record=optimizer.state[param]
        clock=record.get('step')
        if not torch.is_tensor(clock) or clock.numel()!=1 or not bool(torch.isfinite(clock).all()) or float(clock)!=expected_step:
            raise ValueError(f'optimizer step mismatch: {name}')
        for moment in (('exp_avg','exp_avg_sq') if check_moments else ()):
            value=record.get(moment)
            if not torch.is_tensor(value) or value.shape!=param.shape or not bool(torch.isfinite(value).all()):
                raise ValueError(f'optimizer moment invalid: {name}.{moment}')
            if moment=='exp_avg_sq' and bool((value<0).any()):raise ValueError('optimizer squared moment is negative')
    proof=dict(step=expected_step,state_count=len(optimizer.state),parameter_count=len(params),parameter_names=actual)
    if hash_content:proof['optimizer_content_sha256']=tensor_digest(optimizer.state_dict())
    return proof


def _finite(value,name):
    if type(value) not in (int,float) or not math.isfinite(value):raise ValueError(f'nonfinite/invalid {name}')
    return float(value)


def validation_history(records):
    return [dict(step=row['step'],loss=row['validation_loss']) for row in records if 'validation_loss' in row]


def verify_history(records,schedule,*,expected_tail,save_every):
    if [row.get('step') for row in records]!=list(range(1,expected_tail+1)):
        raise ValueError('history has missing, duplicate or extra optimizer steps')
    for row in records:
        step=row['step'];_finite(row.get('loss'),'training loss');_finite(row.get('gradient_norm'),'gradient norm')
        if row.get('finite_gradients') is not True:raise ValueError('history lacks finite gradient evidence')
        micros=row.get('microbatches',[])
        if len(micros)!=4:raise ValueError('history effective batch is not four')
        for micro,item in zip(micros,schedule[(step-1)*4:step*4]):
            if (micro.get('sample_id'),micro.get('noise_seed'))!=(item.sample_id,item.noise_seed):
                raise ValueError('history sample/noise schedule mismatch')
            for field in ('flow','background','budget','sigma'):_finite(micro.get(field),field)
        if step%save_every==0:
            _finite(row.get('validation_loss'),'true val loss')
            evidence=row.get('validation',{})
            if (evidence.get('selection')!='true_val_split_only' or evidence.get('gradients_none') is not True
                or evidence.get('optimizer_steps')!=0 or evidence.get('inference_mode') is not True or evidence.get('autograd_enabled') is not False):
                raise ValueError('history lacks inference-only true-val evidence')
    expected=list(range(save_every,expected_tail+1,save_every))
    if [row['step'] for row in validation_history(records)]!=expected:
        raise ValueError('history has missing or extra true-val selections')
    return validation_history(records)


def _micro_total(row):
    if 'total_loss' in row:return _finite(row['total_loss'],'micro total loss')
    # The old archive omitted individual total_loss. Its three components were
    # actual float32 scalar tensors; preserve the eager loss expression/rounding.
    value=torch.tensor(row['flow'],dtype=torch.float32)+.1*torch.tensor(row['background'],dtype=torch.float32)+.01*torch.tensor(row['budget'],dtype=torch.float32)
    return float(value)


def compare_replay(archived,current):
    if archived.get('step')!=current.get('step'):raise ValueError('replay global step mismatch')
    comparisons=[]
    if len(archived.get('microbatches',[]))!=4 or len(current.get('microbatches',[]))!=4:
        raise ValueError('replay microbatch count mismatch')
    for index,(old,new) in enumerate(zip(archived['microbatches'],current['microbatches'])):
        for field in ('sample_id','noise_seed','sigma','flow','background','budget'):
            if old.get(field)!=new.get(field):raise ValueError(f'replay micro{index} {field} divergence: {old.get(field)} != {new.get(field)}')
        if _micro_total(old)!=_micro_total(new):raise ValueError(f'replay micro{index} total_loss divergence')
        comparisons.append(dict(micro=index,sample_id=new['sample_id'],noise_seed=new['noise_seed'],
            sigma=new['sigma'],flow=new['flow'],background=new['background'],budget=new['budget'],total_loss=_micro_total(new),exact=True))
    for field in ('loss','gradient_norm'):
        if archived.get(field)!=current.get(field):raise ValueError(f'replay {field} divergence')
    return dict(step=current['step'],passed=True,comparison='exact equality; no tolerance relaxation',
        micro_total_source='archived actual total_loss' if all('total_loss' in row for row in archived['microbatches']) else 'fp32 reconstruction from archived components',
        microbatch_comparisons=comparisons,mean_loss=current['loss'],gradient_norm=current['gradient_norm'])


def rng_digest():
    value={'cpu':torch.random.get_rng_state()}
    if torch.cuda.is_initialized():value['cuda']=torch.cuda.get_rng_state()
    return tensor_digest(value)


def resume_updates(adapter,optimizer,schedule,*,start_step,target_step,history,archived,train_forward,
                   validate,save_every,on_save,metrics_path=None):
    """Direct global-step loop. A replay mismatch cannot commit an optimizer step."""
    if not 0<=start_step<target_step<=300:raise ValueError('invalid recovery step range')
    verify_history(history,schedule,expected_tail=start_step,save_every=save_every)
    if len(schedule)<target_step*4 or any((item.step,item.micro)!=(index//4+1,index%4) for index,item in enumerate(schedule)):
        raise ValueError('recovery global schedule is invalid')
    names=[name for name,param in adapter.named_parameters() if param.requires_grad]
    optimizer_proof(adapter,optimizer,names,expected_step=start_step)
    params=[param for param in adapter.parameters() if param.requires_grad]
    validations=validation_history(history)
    best=min(validations,key=lambda row:(row['loss'],row['step'])) if validations else dict(step=None,loss=float('inf'))
    records=[];audits=[];began=time.monotonic()
    stream=Path(metrics_path).open('x',encoding='utf-8',buffering=1) if metrics_path is not None else None
    try:
        for step in range(start_step+1,target_step+1):
            optimizer.zero_grad(set_to_none=True);losses=[];stats=[]
            for item in schedule[(step-1)*4:step*4]:
                before_rng=rng_digest()
                loss,micro=train_forward(item,4)
                if rng_digest()!=before_rng:raise ValueError('ambient RNG changed during training; exact continuation cannot be proven')
                scalar=float(loss.detach());_finite(scalar,'recovery training loss')
                losses.append(scalar);stats.append(dict(sample_id=item.sample_id,noise_seed=item.noise_seed,**micro,total_loss=scalar))
                del loss
            grads=[param.grad for param in params if param.grad is not None]
            if not grads or not any(bool((grad!=0).any()) for grad in grads) or any(not bool(torch.isfinite(grad).all()) for grad in grads):
                raise ValueError('recovery gradients are missing or nonfinite')
            norm=float(torch.nn.utils.clip_grad_norm_(params,1.));_finite(norm,'gradient norm')
            record=dict(step=step,loss=sum(losses)/4,microbatches=stats,gradient_norm=norm,
                        finite_gradients=True,elapsed_seconds=time.monotonic()-began,global_rng_unchanged=True)
            if step in archived:
                proof=compare_replay(archived[step],record);proof['rng_unchanged']=True;audits.append(proof)
            optimizer.step()
            optimizer_proof(adapter,optimizer,names,expected_step=step,check_moments=False,hash_content=False)
            if step%save_every==0 or step==target_step:
                optimizer.zero_grad(set_to_none=True)
                with torch.inference_mode():score,evidence=validate(step)
                _finite(score,'validation loss')
                if any(param.grad is not None for param in adapter.parameters()):raise ValueError('validation created gradients')
                optimizer_proof(adapter,optimizer,names,expected_step=step,check_moments=False,hash_content=False)
                is_best=score<best['loss']
                if is_best:best=dict(step=step,loss=score)
                validations.append(dict(step=step,loss=score))
                record.update(validation_loss=score,validation=evidence,best_step=best['step'],best_validation=best['loss'])
                on_save(step,score,is_best,record)
            records.append(record)
            if stream is not None:stream.write(json.dumps(record,allow_nan=False)+'\n')
            print('RECOVERY_TRAIN',step,record['loss'],'elapsed',round(record['elapsed_seconds'],1),flush=True)
    finally:
        if stream is not None:stream.close()
    return dict(status='updates_finished_pending_integrity_proofs',records=records,replay_audits=audits,
        start_step=start_step,completed_step=target_step,committed_updates=target_step-start_step,
        best_step=best['step'],best_validation=best['loss'],validation_records=validations,
        elapsed_seconds=time.monotonic()-began,optimizer_proof=optimizer_proof(adapter,optimizer,names,expected_step=target_step))


def native_integrity_report(original_before,recovery_before,recovery_after):
    if original_before!=recovery_before or recovery_before!=recovery_after:
        raise ValueError('native backbone SHA differs from original/recovery boundary')
    return dict(original_process=dict(initial_memory_sha256=original_before,final_memory_sha256=None,
        final_memory_hash_observed=False,reason='SSH interruption before original final native hash; it cannot be reconstructed'),
        recovery_process=dict(sha256_before=recovery_before,sha256_after=recovery_after,unchanged=True,
        scope='actual freshly loaded recovery process; initial SHA agrees with persisted original initial SHA'))


def _imports(mirror):
    sys.path.insert(0,str(mirror))
    names={'pilot':'src.pose_control.qwen21_full.dual.prompt_pilot','checkpoint':'src.pose_control.qwen21_full.dual.checkpoint',
        'experiment':'src.pose_control.qwen21_full.dual.experiment','runtime':'src.pose_control.qwen21_full.dual.runtime',
        'precision':'src.pose_control.qwen21_full.dual.precision','cache':'src.pose_control.qwen21_full.qwen_runtime',
        'trainer':'scripts.train.train_qwen21_dual_prompt_pilot','evaluation':'src.evaluation.qwen21_dual_prompt_pilot'}
    result={key:importlib.import_module(name) for key,name in names.items()}
    if any(not Path(module.__file__).resolve().is_relative_to(mirror.resolve()) for module in result.values()):
        raise ValueError('recovery imported source outside the frozen implementation mirror')
    return SimpleNamespace(**result)


def _read_metrics(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def _restore(pkg,path,manifest,step):
    adapter,metadata,payload=pkg.checkpoint.load_checkpoint(path)
    if payload['step']!=step:raise ValueError('checkpoint global step differs from persisted boundary')
    require_equal_contract(manifest['run_contract'],metadata['run_contract'])
    for key,value in metadata['run_contract'].items():
        if metadata.get(key)!=value:raise ValueError(f'checkpoint top-level contract differs: {key}')
    adapter.freeze_warm_geometry().train()
    names=[name for name,param in adapter.named_parameters() if param.requires_grad]
    if names!=metadata['optimizer_parameter_names']:raise ValueError('optimizer names differ after restoring frozen scopes')
    optimizer=torch.optim.AdamW([param for param in adapter.parameters() if param.requires_grad],lr=1e-4,weight_decay=.01)
    pkg.experiment.validate_optimizer_contract(payload.get('optimizer_state_dict'),metadata['optimizer'],
        parameters=optimizer.param_groups[0]['params'],checkpoint_step=step)
    optimizer.load_state_dict(payload['optimizer_state_dict'])
    proof=optimizer_proof(adapter,optimizer,names,expected_step=step)
    del payload
    return adapter,optimizer,metadata,proof


def _checkpoint_selection(pkg,path,metadata,history):
    chosen=min(validation_history(history),key=lambda row:(row['loss'],row['step']))
    adapter,info,payload=pkg.checkpoint.load_checkpoint(path)
    require_equal_contract(metadata['run_contract'],info['run_contract'])
    if (payload['step'],info.get('best_step'),info.get('best_validation'),info.get('validation_loss'))!=(chosen['step'],chosen['step'],chosen['loss'],chosen['loss']):
        raise ValueError('actual best checkpoint does not match historical true-val minimum')
    proof=dict(path=str(Path(path).resolve()),sha256=file_sha(path),step=chosen['step'],
               adapter_sha256=pkg.runtime.tensor_model_sha256(adapter))
    del adapter,payload
    return proof


def _read_caches(pkg,data,run,manifests):
    from torchvision.transforms.functional import to_pil_image
    values={variant:{name:[] for name in ('train','val')} for variant in ('short','structured')}
    files={}
    root=run/'native_cache'/'shared_targets'
    if not (root/'cache_policy.json').is_file():raise ValueError('missing shared target cache policy')
    target=pkg.cache.Qwen21IPERCache(root,prompt='shared_target_latents_for_matched_prompt_pilot_v1',
                                   preprocessing='affine_letterbox_v8_bf16_full_reduction_v1')
    for name,split in data.splits.items():
        for index,sid in enumerate(split.sample_ids):
            truth=split.supervision(index)
            target_path=target._path(to_pil_image(truth.rgb[0]),'target')
            clean=torch.load(target_path,map_location='cpu',weights_only=True)
            files[str(target_path)]=file_sha(target_path)
            for variant in values:
                expected=manifests[variant]['cache_content'][name][index]
                generation=split.generation(index,variant)
                directory=Path(expected['cache_directory']).resolve()
                expected_dir=(run/'native_cache'/variant/name/pkg.pilot.source_prompt_cache_key(split.rows[variant][index])).resolve()
                if directory!=expected_dir or not (directory/'cache_policy.json').is_file():raise ValueError('source prompt cache directory/policy mismatch')
                cache=pkg.cache.Qwen21IPERCache(directory,prompt=generation.prompts[0],preprocessing='affine_letterbox_v8_bf16_full_reduction_v1')
                source_path=cache._path(to_pil_image(generation.source.rgb[0]),'source',prompt=generation.prompts[0])
                source=torch.load(source_path,map_location='cpu',weights_only=True)
                sample=pkg.pilot.CachedSample(sid,generation,source,clean)
                actual=dict(sample_id=sid,prompt=generation.prompts[0],source_prompt_cache_key=pkg.pilot.source_prompt_cache_key(split.rows[variant][index]),
                    cache_directory=str(directory),source_content_sha256=pkg.experiment.tensor_cache_sha256(sample.source),
                    target_latent_content_sha256=pkg.experiment.tensor_cache_sha256(sample.clean),
                    combined_content_sha256=sample.content_sha256,geometry_sha256=generation.geometry.geometry_hash)
                require_equal_contract(expected,actual)
                files[str(source_path)]=file_sha(source_path);files[str(directory/'cache_policy.json')]=file_sha(directory/'cache_policy.json')
                values[variant][name].append(sample)
            del truth
    files[str(root/'cache_policy.json')]=file_sha(root/'cache_policy.json')
    return values,files


def _preflight(args,pkg):
    if any((args.run_dir/path).exists() for path in ('training_complete.json','short/training_complete.json','structured/training_complete.json')):
        raise ValueError('existing completion requires investigation; recovery cannot overwrite it')
    contracts=read_json(args.evidence_dir/'recovery_contracts.json')
    root=read_json(args.run_dir/'run_manifest.json')
    require_equal_contract(contracts['root_manifest'],root)
    if (root.get('production_training'),root.get('requested_steps'),root.get('learning_rate'),root.get('save_every'),set(root.get('variants',[])))!=(True,300,1e-4,50,{'short','structured'}):
        raise ValueError('original run is not the authorized matched production experiment')
    manifests={variant:read_json(args.run_dir/variant/'run_manifest.json') for variant in ('short','structured')}
    for variant,manifest in manifests.items():
        require_equal_contract(contracts[f'{variant}_manifest'],manifest)
        if manifest['code_sha256']!=pkg.pilot.prompt_pilot_code_sha256(args.mirror):raise ValueError('frozen code closure mismatch')
        if manifest['runtime_versions']!={name:version(name) for name in manifest['runtime_versions']}:raise ValueError('runtime package versions mismatch')
        if manifest['initial_adapter_sha256']!=manifests['short']['initial_adapter_sha256']:raise ValueError('original initial adapter SHAs differ')
        gate=read_json(args.run_dir/variant/'real_model_gates.json')
        require_equal_contract(contracts['gates'][variant],gate)
        if not gate['zero_init_exact'] or not gate['frozen_coverage']['complete'] or gate['source_step']!=19500:
            raise ValueError('original real model initialization gates are incomplete')
    data=pkg.pilot.load_training_data(args.data_dir)
    require_equal_contract(root['data_content_sha256'],data.content_sha256)
    if root['dataset_sha256']!=data.dataset_sha256 or root['split_manifest']!=data.split_manifest:
        raise ValueError('dataset/split contract mismatch')
    schedule=pkg.pilot.shared_schedule(data.splits['train'].sample_ids,steps=300,seed=root['seed'])
    if read_json(args.run_dir/'shared_schedule.json')!=[asdict(item) for item in schedule] or root['schedule_sha256']!=pkg.pilot.schedule_sha256(schedule):
        raise ValueError('global shared sample/noise schedule mismatch')
    if file_sha(root['warm_start_path'])!=root['warm_start_sha256']:raise ValueError('warm start SHA changed')
    initial,_,initial_sha=pkg.pilot.fresh_adapter(root['seed'],root['warm_start_path'])
    if initial_sha!=manifests['short']['initial_adapter_sha256']:raise ValueError('regenerated original initialization SHA differs')
    frozen_digest=tensor_digest({name:param for name,param in initial.named_parameters() if not param.requires_grad})
    del initial
    short_rows=_read_metrics(args.evidence_dir/'short_original_train_metrics.jsonl')
    structured_rows=_read_metrics(args.evidence_dir/'structured_original_train_metrics.jsonl')
    verify_history(short_rows,schedule,expected_tail=300,save_every=50)
    if not 201<=len(structured_rows)<250:raise ValueError('archived structured tail is outside committed200 plus replay range')
    verify_history(structured_rows,schedule,expected_tail=len(structured_rows),save_every=50)
    if contracts['structured_archived201']!=structured_rows[200] or contracts['structured_archived201']['step']!=201:
        raise ValueError('true archived201 contract is mislabeled or inconsistent')
    if contracts.get('structured_last_archived_step')!=len(structured_rows):raise ValueError('archived structured tail contract differs')
    if contracts.get('structured_discarded_after_checkpoint')!=structured_rows[200:]:raise ValueError('discarded archive rows differ')
    anchor=args.evidence_dir/'structured_step200.pt'
    if file_sha(anchor)!=file_sha(args.run_dir/'structured/checkpoint_last.pt'):raise ValueError('live structured checkpoint is no longer the preserved step200 boundary')
    adapter,optimizer,metadata,structured_proof=_restore(pkg,anchor,manifests['structured'],200)
    if tensor_digest({name:param for name,param in adapter.named_parameters() if not param.requires_grad})!=frozen_digest:
        raise ValueError('structured frozen warm geometry differs from the actual original warm start')
    require_equal_contract(contracts['structured_checkpoint_metadata'],metadata)
    if contracts['structured_checkpoint_step']!=200:raise ValueError('CPU audit did not authenticate checkpoint200')
    short_adapter,short_optimizer,short_meta,short_proof=_restore(pkg,args.run_dir/'short/checkpoint_last.pt',manifests['short'],300)
    if tensor_digest({name:param for name,param in short_adapter.named_parameters() if not param.requires_grad})!=frozen_digest:
        raise ValueError('short frozen warm geometry differs from the actual original warm start')
    short_last_sha=pkg.runtime.tensor_model_sha256(short_adapter)
    del short_adapter,short_optimizer
    bests={variant:_checkpoint_selection(pkg,args.run_dir/variant/'checkpoint_best.pt',manifests[variant],
             short_rows if variant=='short' else structured_rows[:200]) for variant in manifests}
    if file_sha(args.evidence_dir/'structured_best50_before_recovery.pt')!=bests['structured']['sha256']:
        raise ValueError('preserved structured best50 differs from live best')
    caches,cache_files=_read_caches(pkg,data,args.run_dir,manifests)
    for variant in manifests:require_equal_contract(manifests[variant]['cache_content'],read_json(args.run_dir/'cache_manifest.json')[variant])
    protected={str(path):file_sha(path) for path in [args.evidence_dir/'recovery_contracts.json',args.evidence_dir/'interruption_audit.json',
        args.evidence_dir/'original_training.log',args.evidence_dir/'structured_step200.pt',args.evidence_dir/'structured_best50_before_recovery.pt',
        args.evidence_dir/'short_original_train_metrics.jsonl',args.evidence_dir/'structured_original_train_metrics.jsonl',
        args.run_dir/'run_manifest.json',args.run_dir/'cache_manifest.json',args.run_dir/'shared_schedule.json',
        *[args.run_dir/variant/name for variant in manifests for name in ('run_manifest.json','real_model_gates.json','checkpoint_last.pt','checkpoint_best.pt','train_metrics.jsonl')]]}
    for variant,rows in [('short',short_rows),('structured',structured_rows)]:
        if _read_metrics(args.run_dir/variant/'train_metrics.jsonl')!=rows:raise ValueError('live metrics differ from immutable interrupted backups')
    return dict(root=root,manifests=manifests,data=data,schedule=schedule,short_rows=short_rows,structured_rows=structured_rows,
        adapter=adapter,optimizer=optimizer,metadata=metadata,bests=bests,caches=caches,cache_files=cache_files,protected=protected,
        optimizer_proofs=dict(short=short_proof,structured=structured_proof),short_last_sha=short_last_sha)


def _branch_completion(args,plan,variant,rows,last_path,best_path,native_scope,recovery,pkg):
    verify_history(rows,plan['schedule'],expected_tail=300,save_every=50)
    adapter,optimizer,metadata,clock=_restore(pkg,last_path,plan['manifests'][variant],300)
    best=_checkpoint_selection(pkg,best_path,metadata,rows)
    chosen=min(validation_history(rows),key=lambda row:(row['loss'],row['step']))
    if (metadata['best_step'],metadata['best_validation'])!=(chosen['step'],chosen['loss']):raise ValueError('last checkpoint global best fields differ')
    result=dict(variant=variant,status='success',schema_version=TRAINING_SCHEMA,production_training=True,
        requested_steps=300,completed_steps=300,dataset_sha256=plan['data'].dataset_sha256,split_manifest=plan['data'].split_manifest,
        model_id=metadata['model_id'],initial_adapter_sha256=metadata['initial_adapter_sha256'],
        validation_selection='true_val_split_only',train_pair_ids=metadata['train_pair_ids'],val_pair_ids=metadata['val_pair_ids'],
        train_sample_ids=metadata['train_sample_ids'],val_sample_ids=metadata['val_sample_ids'],
        best_step=chosen['step'],best_validation=chosen['loss'],validation_records=validation_history(rows),
        best_checkpoint=dict(path=str((args.run_dir/variant/'checkpoint_best.pt').resolve()),sha256=file_sha(best_path),step=chosen['step']),
        last_checkpoint=dict(path=str((args.run_dir/variant/'checkpoint_last.pt').resolve()),sha256=file_sha(last_path),step=300),
        best_adapter_sha256=best['adapter_sha256'],trained_adapter_sha256=pkg.runtime.tensor_model_sha256(adapter),
        warm_start_sha256=metadata['warm_start_sha256'],caches_unchanged=True,native_gradients_none=True,
        backbone_sha256_before=metadata['backbone_sha256'],backbone_sha256_after=native_scope['recovery_process']['sha256_after'],
        backbone_unchanged=True,initialization_sha256_equal=True,native_integrity_scope=native_scope,
        recovery_provenance=recovery,optimizer_completion_proof=clock,
        elapsed_seconds=rows[-1]['elapsed_seconds'],
        elapsed_seconds_scope='persisted metric lower bound; original final validation timing was not preserved')
    del adapter,optimizer
    return result


def _stage_and_commit(args,plan,branches,root,structured_last,structured_best,merged,pkg):
    stage=args.attempt_dir/'verified_completion'
    stage.mkdir()
    staged_branches=copy.deepcopy(branches)
    for variant in branches:
        directory=stage/variant;directory.mkdir()
        source=structured_best if variant=='structured' else args.run_dir/variant/'checkpoint_best.pt'
        shutil.copyfile(source,directory/'checkpoint_best.pt')
        staged_branches[variant]['best_checkpoint']['path']=str((directory/'checkpoint_best.pt').resolve())
        atomic_json(directory/'training_complete.json',staged_branches[variant])
    staged_root=dict(root,variants=staged_branches)
    atomic_json(stage/'training_complete.json',staged_root)
    raw_dataset=read_json(plan['data'].dataset_path)
    bundle=dict(dataset_path=plan['data'].dataset_path,dataset_sha256=plan['data'].dataset_sha256,
        split_manifest=plan['data'].split_manifest,
        pair_ids={name:list(split.pair_ids) for name,split in plan['data'].splits.items()},
        sample_ids={name:list(split.sample_ids) for name,split in plan['data'].splits.items()},
        manifests={name:dict(short=split.manifest_paths['short_generation'],structured=split.manifest_paths['structured_generation'],
                             supervision=split.manifest_paths['supervision']) for name,split in plan['data'].splits.items()})
    # Only the test path is used to reject forbidden training content. No test
    # generation, supervision, tensors or metrics are opened during recovery.
    bundle['manifests']['test']={'supervision':Path(raw_dataset['splits']['test']['supervision'])}
    pkg.evaluation.validate_training_contracts(stage,bundle)
    verify_files(plan['protected']);verify_files(plan['cache_files']);plan['data'].assert_unchanged()
    if pkg.pilot.prompt_pilot_code_sha256(args.mirror)!=plan['root']['code_sha256']:raise ValueError('frozen code changed before commit')
    atomic_copy(structured_last,args.run_dir/'structured/checkpoint_last.pt')
    if file_sha(structured_best)!=plan['bests']['structured']['sha256']:
        atomic_copy(structured_best,args.run_dir/'structured/checkpoint_best.pt')
    atomic_bytes(args.run_dir/'structured/train_metrics.jsonl',(''.join(json.dumps(row,allow_nan=False)+'\n' for row in merged)).encode())
    # The same frozen evaluator is the final consumer; keep its full contract.
    markers={args.run_dir/variant/'training_complete.json':record for variant,record in branches.items()}
    markers[args.run_dir/'training_complete.json']=root
    return publish_completion_fail_closed(markers,lambda:pkg.evaluation.validate_training_contracts(args.run_dir,bundle))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir',type=Path,default=WORK)
    p.add_argument('--run-dir',type=Path,default=RUN)
    p.add_argument('--mirror',type=Path)
    p.add_argument('--data-dir',type=Path)
    p.add_argument('--evidence-dir',type=Path)
    p.add_argument('--attempt-dir',type=Path)
    p.add_argument('--preflight-only',action='store_true',help='CPU audits only; no native model load or RUN mutation')
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    args.work_dir=args.work_dir.resolve();args.run_dir=args.run_dir.resolve()
    args.mirror=(args.mirror or args.work_dir/'implementation').resolve()
    args.data_dir=(args.data_dir or args.work_dir/'data').resolve()
    args.evidence_dir=(args.evidence_dir or args.work_dir/'recovery').resolve()
    args.attempt_dir=(args.attempt_dir or args.evidence_dir/'continuation_200_to_300').resolve()
    if args.attempt_dir.is_relative_to(args.mirror) or args.attempt_dir.is_relative_to(args.run_dir):
        raise ValueError('operational recovery output must stay outside MIRROR and RUN')
    if args.attempt_dir.exists():raise ValueError('recovery attempt directory must be new; previous attempts stay preserved')
    args.attempt_dir.mkdir(parents=True)
    source=Path(__file__).resolve()
    if source.is_relative_to(args.mirror):raise ValueError('recovery source must stay outside the frozen implementation closure')
    recovery=dict(schema_version=RECOVERY_SCHEMA,source_path=str(source),source_sha256=file_sha(source),
        started_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        committed_boundary=200,resumed_steps=[201,300],frozen_implementation_edited=False,
        original_final_native_memory_hash_observed=False)
    sibling=source.with_name('test_recover_training.py')
    if sibling.exists():recovery['cpu_regression_source_sha256']=file_sha(sibling)
    published_markers=None
    try:
        pkg=_imports(args.mirror)
        torch.set_num_threads(8)
        plan=_preflight(args,pkg)
        recovery.update(original_live_inputs_sha256_at_preflight=plan['protected'],
            preserved_evidence_sha256={name:digest for name,digest in plan['protected'].items() if Path(name).is_relative_to(args.evidence_dir)},
            cache_files_sha256=plan['cache_files'],
            discarded_uncheckpointed_steps=[row['step'] for row in plan['structured_rows'][200:]],
            optimizer_boundary_proof=plan['optimizer_proofs'],original_training_code_sha256=plan['root']['code_sha256'])
        atomic_json(args.attempt_dir/'recovery_preflight.json',dict(status='preflight_passed',recovery=recovery))
        if args.preflight_only:
            print('RECOVERY_PREFLIGHT_PASSED',flush=True)
            return dict(status='preflight_passed_no_training_no_completion',recovery=recovery)
        with pkg.precision.qwen21_precision():
            torch.manual_seed(plan['root']['seed']);torch.cuda.manual_seed_all(plan['root']['seed'])
            pipe=pkg.runtime.load_native(Path(plan['root']['model_id']))
            before=pkg.runtime.tensor_model_sha256(pipe.transformer)
            if before!=plan['root']['backbone_sha256']:raise ValueError('reloaded native backbone differs from original initial SHA')
            actual_code={name:file_sha(inspect.getfile(type(component))) for name,component in
                (('transformer',pipe.transformer),('pipeline',pipe),('vae',pipe.vae),('text_encoder',pipe.text_encoder))}
            if actual_code!=plan['metadata']['native_code_sha256']:raise ValueError('native runtime source SHA mismatch')
            if file_sha(Path(plan['root']['model_id'])/'transformer/config.json')!=plan['root']['model_config_sha256']:
                raise ValueError('native config SHA mismatch')
            pipe.text_encoder.to('cpu');pipe.vae.to('cpu');gc.collect();torch.cuda.empty_cache()
            pipe.transformer.enable_gradient_checkpointing()
            if pipe.transformer.training or any(param.requires_grad or param.grad is not None for param in pipe.transformer.parameters()):
                raise ValueError('native recovery backbone is not frozen eval')
            adapter=plan['adapter'].to('cuda',torch.float32).train()
            # Rebuild moments on CUDA through normal load_state_dict casting.
            state=plan['optimizer'].state_dict()
            optimizer=torch.optim.AdamW([param for param in adapter.parameters() if param.requires_grad],lr=1e-4,weight_decay=.01)
            optimizer.load_state_dict(state);plan['optimizer']=optimizer;del state
            optimizer_proof(adapter,optimizer,plan['metadata']['optimizer_parameter_names'],expected_step=200)
            active_dropout=[name for model in (adapter,pipe.transformer) for name,module in model.named_modules()
                            if module.training and ((isinstance(module,torch.nn.modules.dropout._DropoutNd) and module.p>0)
                            or (isinstance(module,torch.nn.MultiheadAttention) and module.dropout>0))]
            if active_dropout:raise ValueError('active dropout has no saved RNG state; restart is required')
            candidates=args.attempt_dir/'structured';candidates.mkdir()
            best_path=args.run_dir/'structured/checkpoint_best.pt'
            historical=min(validation_history(plan['structured_rows'][:200]),key=lambda row:(row['loss'],row['step']))
            global_best=dict(historical)
            def train_forward(item,scale):
                sample=plan['caches']['structured']['train'][item.sample_index]
                if sample.sample_id!=item.sample_id:raise ValueError('recovery cached sample differs from original schedule')
                loss,stats=pkg.trainer.loss_forward(pipe,adapter,sample.generation.to('cuda',torch.float32),sample.source,
                    sample.clean.to('cuda',torch.bfloat16),torch.Generator(device='cuda').manual_seed(item.noise_seed),
                    backward_scale=scale,teacher_weight=.1,budget_weight=.01)
                if any(param.grad is not None for param in pipe.transformer.parameters()):raise ValueError('native backbone received gradients')
                return loss,stats
            def validate(step):
                def evaluate(index,seed):
                    sample=plan['caches']['structured']['val'][index]
                    loss,_=pkg.trainer.loss_forward(pipe,adapter,sample.generation.to('cuda',torch.float32),sample.source,
                        sample.clean.to('cuda',torch.bfloat16),torch.Generator(device='cuda').manual_seed(seed),teacher_weight=0,budget_weight=0)
                    return loss
                return pkg.pilot.validation_pass(adapter,pipe.transformer,optimizer,evaluate,sample_count=1)
            def save(step,score,is_best,record):
                nonlocal best_path
                if is_best:global_best.update(step=step,loss=score)
                info=dict(plan['metadata'],step=step,validation_loss=score,best_validation=global_best['loss'],
                    best_step=global_best['step'],validation=record['validation'],recovery_provenance=recovery)
                pkg.checkpoint.save_checkpoint(candidates/'checkpoint_last.pt',adapter,info,optimizer=optimizer,step=step)
                if is_best:
                    best_path=candidates/'checkpoint_best.pt'
                    pkg.checkpoint.save_checkpoint(best_path,adapter,info,optimizer=optimizer,step=step)
            resumed=resume_updates(adapter,optimizer,plan['schedule'],start_step=200,target_step=300,
                history=plan['structured_rows'][:200],archived={row['step']:row for row in plan['structured_rows'][200:]},
                train_forward=train_forward,validate=validate,save_every=50,on_save=save,
                metrics_path=args.attempt_dir/'structured_resumed_global_metrics.jsonl')
            if [row['step'] for row in resumed['replay_audits']]!=recovery['discarded_uncheckpointed_steps']:
                raise ValueError('not all discarded committed updates were replay-audited')
            recovery['replay_audits']=resumed['replay_audits']
            atomic_json(args.attempt_dir/'replay_audits.json',resumed['replay_audits'])
            after=pkg.runtime.tensor_model_sha256(pipe.transformer)
            scope=native_integrity_report(plan['root']['backbone_sha256'],before,after)
            recovery['native_integrity_scope']=scope
            if any(param.grad is not None for param in pipe.transformer.parameters()):raise ValueError('native gradients exist after recovery')
            for variant in plan['caches'].values():
                for samples in variant.values():
                    for sample in samples:sample.assert_unchanged()
            verify_files(plan['protected']);verify_files(plan['cache_files']);plan['data'].assert_unchanged()
            if file_sha(source)!=recovery['source_sha256'] or pkg.pilot.prompt_pilot_code_sha256(args.mirror)!=plan['root']['code_sha256']:
                raise ValueError('recovery/frozen source SHA changed during continuation')
            if file_sha(plan['root']['warm_start_path'])!=plan['root']['warm_start_sha256']:raise ValueError('warm start changed during continuation')
            prefix_elapsed=plan['structured_rows'][199]['elapsed_seconds']
            merged=plan['structured_rows'][:200]+[dict(row,elapsed_seconds=prefix_elapsed+row['elapsed_seconds']) for row in resumed['records']]
            branches={
                'short':_branch_completion(args,plan,'short',plan['short_rows'],args.run_dir/'short/checkpoint_last.pt',
                                          args.run_dir/'short/checkpoint_best.pt',scope,recovery,pkg),
                'structured':_branch_completion(args,plan,'structured',merged,candidates/'checkpoint_last.pt',best_path,scope,recovery,pkg)}
            root=dict(status='success',schema_version=TRAINING_SCHEMA,production_training=True,requested_steps=300,completed_steps=300,
                dataset_path=str(plan['data'].dataset_path),dataset_sha256=plan['data'].dataset_sha256,split_manifest=plan['data'].split_manifest,
                model_id=plan['root']['model_id'],backbone_sha256_before=before,backbone_sha256_after=after,backbone_unchanged=True,
                initialization_sha256_equal=True,initial_adapter_sha256=plan['manifests']['short']['initial_adapter_sha256'],
                initial_adapter_sha256_by_variant={name:manifest['initial_adapter_sha256'] for name,manifest in plan['manifests'].items()},
                warm_start_sha256=plan['root']['warm_start_sha256'],train_pair_ids=plan['root']['train_pair_ids'],val_pair_ids=plan['root']['val_pair_ids'],
                validation_selection='true_val_split_only',effective_batch=4,schedule_sha256=plan['root']['schedule_sha256'],
                test_data_read=False,variants=branches,recovery_provenance=recovery,native_integrity_scope=scope)
            published_markers=_stage_and_commit(args,plan,branches,root,candidates/'checkpoint_last.pt',best_path,merged,pkg)
            atomic_json(args.attempt_dir/'recovery_result.json',dict(status='success',recovery=recovery,
                completed_steps=300,structured_optimizer_proof=resumed['optimizer_proof'],completion_sha256=file_sha(args.run_dir/'training_complete.json')))
            print('RECOVERY_COMPLETE',str(args.run_dir/'training_complete.json'),flush=True)
            return root
    except BaseException as exc:
        if published_markers is not None:withdraw_own_completion_markers(published_markers)
        atomic_json(args.attempt_dir/'recovery_result.json',dict(status='failed_no_completion_claim',error_type=type(exc).__name__,
            error=str(exc),recovery=recovery))
        raise


if __name__=='__main__':main()
