"""Real-model gates and an explicitly bounded, one-pair overfit experiment."""
from __future__ import annotations

import argparse
import copy
import gc
import inspect
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.data.hi4d_smplx_dataset import Hi4DSMPLXDataset,file_sha256
from src.pose_control.qwen21_full.dual import DualV8Adapter
from src.pose_control.qwen21_full.dual.checkpoint import load_checkpoint,save_checkpoint,warm_start_from_body,encoder_provenance
from src.pose_control.qwen21_full.dual.conditions import ControlStrengths
from src.pose_control.qwen21_full.dual.context import DualV8ControlContext
from src.pose_control.qwen21_full.dual.experiment import (validate_resume_contract,
    validate_optimizer_contract,validate_metrics_resume,pilot_code_sha256,tensor_cache_sha256)
from src.pose_control.qwen21_full.dual.precision import qwen21_precision,precision_policy
from src.pose_control.qwen21_full.dual.runtime import (background_teacher_mask,flow_sample,load_native,
    native_inputs,prepared_for_qwen,source_cache,tensor_model_sha256,weighted_flow_loss)


def gradient_report(adapter):
    groups={name:0. for name in ['final_heads','query','global','relative','mutual','contact','appearance','depth']}
    for name,p in adapter.named_parameters():
        if p.grad is None:continue
        if not bool(torch.isfinite(p.grad).all()):raise FloatingPointError(f'nonfinite gradient: {name}')
        category=('final_heads' if 'output_heads' in name and '.2.' in name else
            'query' if 'query_projection' in name else 'global' if '.global32.' in name else
            'relative' if '.relative14.' in name else 'mutual' if '.mutual.' in name else
            'contact' if 'contact' in name else 'appearance' if 'appearance' in name else
            'depth' if 'depth_visibility' in name else None)
        if category:groups[category]+=float(p.grad.detach().float().square().sum())
    return {k:v**.5 for k,v in groups.items()}


@torch.no_grad()
def cache_probe(transformer,adapter,prepared,inputs,sigma,*,require_close=True):
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
    count=math.prod(prepared.target_hw)
    native=transformer(**inputs)[0]
    native_cache=QwenImage21KVCache(num_layers=32)
    transformer(**inputs,kv_cache=native_cache,kv_cache_mode='extract')
    native_cached=transformer(**inputs,kv_cache=native_cache,kv_cache_mode='cached')[0]
    with DualV8ControlContext(transformer,adapter,prepared,sigma=sigma,strengths=ControlStrengths(0,0,0)):
        disabled=transformer(**inputs)[0]
    cache=QwenImage21KVCache(num_layers=32)
    with DualV8ControlContext(transformer,adapter,prepared,sigma=sigma):
        extracted=transformer(**inputs,kv_cache=cache,kv_cache_mode='extract')[0]
        cached=transformer(**inputs,kv_cache=cache,kv_cache_mode='cached')[0]
        full=transformer(**inputs)[0]
    reference=full[:,-count:]
    difference=(cached.float()-reference.float()).abs()
    record=dict(prefix_exact=torch.equal(native[:,:-count],full[:,:-count]),
        disabled_exact=torch.equal(native,disabled),target_changed=not torch.equal(native[:,-count:],reference),
        cached_close=torch.allclose(cached,reference,atol=1e-2,rtol=1e-2),
        extract_close=torch.allclose(extracted,full,atol=1e-2,rtol=1e-2),
        cached_max_abs=float(difference.max()),cached_mean_abs=float(difference.mean()),
        cached_max_relative=float((difference/reference.float().abs().clamp_min(1e-3)).max()),
        native_cached_close=torch.allclose(native_cached,native[:,-count:],atol=1e-2,rtol=1e-2),
        native_cached_max_abs=float((native_cached.float()-native[:,-count:].float()).abs().max()),
        native_cached_mean_abs=float((native_cached.float()-native[:,-count:].float()).abs().mean()),
        adapter_increment_cache_max_abs=float(((cached.float()-native_cached.float())-(reference.float()-native[:,-count:].float())).abs().max()),
        atol=1e-2,rtol=1e-2)
    required=['prefix_exact','disabled_exact','target_changed','extract_close']
    if require_close:required+=['cached_close','native_cached_close']
    if not all(record[key] for key in required):
        raise AssertionError(f'native cache/control gate failed: {record}')
    return record


def loss_forward(pipe,adapter,generation,source,clean,generator,*,backward_scale=None,teacher_weight=.1,budget_weight=.01,probe_sigma=None):
    noise=torch.randn(clean.shape,device=clean.device,dtype=clean.dtype,generator=generator)
    sigma=torch.rand(1,device=clean.device,generator=generator).clamp(.01,.99)
    if probe_sigma is not None:sigma[:]=probe_sigma
    noisy,goal=flow_sample(clean,noise,sigma)
    inputs=native_inputs(source,noisy,sigma)
    teacher=None
    if teacher_weight:
        with torch.no_grad():teacher=pipe.transformer(**inputs)[0][:,-1024:].detach()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        prepared=prepared_for_qwen(adapter,generation)
        with DualV8ControlContext(pipe.transformer,adapter,prepared,sigma=sigma) as context:
            prediction=pipe.transformer(**inputs)[0][:,-1024:]
            body=F.adaptive_avg_pool2d(generation.geometry.amodal_mask.amax(1),(32,32)).flatten(2).transpose(1,2)
            contact=F.adaptive_avg_pool2d(generation.geometry.contact_map,(32,32)).flatten(2).transpose(1,2)
            weights=1+body+contact
            flow=weighted_flow_loss(prediction,goal,weights)
            bg=flow*0
            if teacher is not None:
                bgmask=background_teacher_mask(generation.source.masks,generation.geometry.amodal_mask,(32,32))
                bg=weighted_flow_loss(prediction,teacher,bgmask)
            budget=adapter.budget_loss(.05)
            loss=flow+teacher_weight*bg+budget_weight*budget
            diagnostics=copy.deepcopy(adapter.last_diagnostics)
            if backward_scale is not None:(loss/backward_scale).backward()
    return loss,dict(flow=float(flow.detach()),background=float(bg.detach()),budget=float(budget.detach()),
        sigma=float(sigma[0]),diagnostics=diagnostics)


@qwen21_precision()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--generation',type=Path,required=True)
    p.add_argument('--supervision',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path,default=Path('/home/shangguanrz/project/pic-edit/models/Qwen-Image-2.1'))
    p.add_argument('--warm-start',type=Path,default=Path('/home/shangguanrz/project/pic-edit/experiments/qwen21_iper_body/full_20k/checkpoint_best.pt'))
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--accumulation',type=int,default=4)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=14062)
    p.add_argument('--save-every',type=int,default=50)
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--resume',type=Path)
    a=p.parse_args()
    if not 1<=a.steps<=300 or a.accumulation!=4 or not math.isfinite(a.lr) or a.lr<=0 or a.save_every<1:
        raise ValueError('approved pilot requires 1..300 steps, effective batch four and positive finite LR/save interval')
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'checkpoint_last.pt').exists() and not a.resume:raise ValueError('use a new output directory or explicit --resume')
    if (a.output/'train_metrics.jsonl').exists() and not a.resume:raise ValueError('existing training metrics require explicit --resume')
    torch.set_num_threads(8);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    dataset=Hi4DSMPLXDataset(a.generation,a.supervision)
    if len(dataset)!=1:raise ValueError('this entry is restricted to one-sample overfit')
    generation=dataset.generation();truth=dataset.supervision()
    pipe=load_native(a.model)
    print('NATIVE_LOADED',flush=True)
    cache,source=source_cache(pipe,generation,a.output/'native_cache')
    clean=cache.target(pipe,to_pil_image(truth.rgb[0])).to('cuda',torch.bfloat16)
    # Keep VL/VAE off the training device after caching. Generation loads them
    # in its own process and never reads the supervision object.
    pipe.text_encoder.to('cpu');pipe.vae.to('cpu');gc.collect();torch.cuda.empty_cache()
    generation=generation.to('cuda',torch.float32)
    adapter=DualV8Adapter();warm=warm_start_from_body(adapter,a.warm_start,
        require_frozen_coverage=True,expected_source_step=19500)
    adapter.freeze_warm_geometry().to('cuda',torch.float32).train()
    pipe.transformer.enable_gradient_checkpointing()
    optimizer=torch.optim.AdamW([p for p in adapter.parameters()if p.requires_grad],lr=a.lr,weight_decay=.01)
    before=tensor_model_sha256(pipe.transformer)
    metadata=dict(model_id=str(a.model.resolve()),prompt=generation.prompts[0],seed=a.seed,**encoder_provenance(),
        model_config_sha256=file_sha256(a.model/'transformer/config.json'),generation_sha256=file_sha256(a.generation),
        source_features_sha256=file_sha256(dataset.rows[0]['source_features']),geometry_sha256=generation.geometry.geometry_hash,
        native_source_cache_content_sha256=tensor_cache_sha256(source),
        target_latent_cache_content_sha256=tensor_cache_sha256(clean),
        supervision_sha256=file_sha256(a.supervision),warm_start_sha256=file_sha256(a.warm_start),loading_report=warm,
        backbone_sha256=before,task='one_sample_overfit_target_geometry_known',effective_batch=4,
        precision_policy=precision_policy(),
        optimizer=dict(name='AdamW',**{key:(list(value) if isinstance(value,tuple) else value)
            for key,value in optimizer.param_groups[0].items() if key!='params'}),
        optimizer_parameter_names=[name for name,p in adapter.named_parameters() if p.requires_grad],
        frozen_scopes=['geometry_encoder.spatial_encoder','reasoner.person_reasoner','reasoner.shared_role_offset'],
        objective=dict(flow_body_extra=1.,flow_contact_extra=1.,background_teacher_weight=.1,budget_weight=.01,budget_ratio=.05),
        supervision_content={key:truth.metadata[key] for key in
            ('target_rgb_sha256','target_mask_sha256','target_geometry_sha256','person_uids','pair','action','target_frame','camera')},
        code_sha256=pilot_code_sha256(Path(__file__).resolve().parents[2]),
        runtime_versions={package:version(package) for package in ('torch','torchvision','diffusers','transformers','numpy','scipy','pillow')},
        native_code_sha256={name:file_sha256(inspect.getfile(type(component)))
            for name,component in (('transformer',pipe.transformer),('pipeline',pipe),('vae',pipe.vae),('text_encoder',pipe.text_encoder))})
    contract_keys=['model_id','model_config_sha256','backbone_sha256','generation_sha256','source_features_sha256',
        'native_source_cache_content_sha256','target_latent_cache_content_sha256','optimizer_parameter_names',
        'geometry_sha256','supervision_sha256','supervision_content','warm_start_sha256','prompt','seed','effective_batch',
        'optimizer','frozen_scopes','objective','code_sha256','native_code_sha256','runtime_versions','precision_policy',*encoder_provenance().keys()]
    metadata['run_contract']={key:metadata[key] for key in contract_keys}
    loaded=None;resume_payload=None;start=0;best=float('inf')
    if a.resume:
        loaded,info,resume_payload=load_checkpoint(a.resume)
        start=validate_resume_contract(dict(info,step=resume_payload['step']),metadata['run_contract'],requested_steps=a.steps)
        validate_optimizer_contract(resume_payload.get('optimizer_state_dict'),metadata['optimizer'],
            parameters=optimizer.param_groups[0]['params'],checkpoint_step=start)
        validate_metrics_resume(a.output/'train_metrics.jsonl',start)
        best=info.get('best_noise_validation',best)
        metadata['resume_parent']=dict(path=str(a.resume.resolve()),sha256=file_sha256(a.resume),step=start)
    (a.output/'run_manifest.json').write_text(json.dumps(dict(metadata,requested_steps=a.steps,save_every=a.save_every),indent=2)+'\n')
    fixed=torch.Generator(device='cuda').manual_seed(42)
    probe_noise=torch.randn(clean.shape,device='cuda',dtype=clean.dtype,generator=fixed)
    probe_sigma=torch.tensor([.65],device='cuda')
    noisy,_=flow_sample(clean,probe_noise,probe_sigma);inputs=native_inputs(source,noisy,probe_sigma)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        native=pipe.transformer(**inputs)[0]
        state=prepared_for_qwen(adapter,generation)
        with DualV8ControlContext(pipe.transformer,adapter,state,sigma=probe_sigma):controlled=pipe.transformer(**inputs)[0]
    if not torch.equal(native,controlled):raise AssertionError('zero initialization differs from native real Qwen')
    del native,controlled,state
    gate=dict(zero_init_exact=True,backbone_before=before,warm_tensors=len(warm['matched']),
              frozen_coverage=warm['frozen_coverage'],precision_policy=precision_policy())
    for update in range(2):
        optimizer.zero_grad(set_to_none=True)
        generator=torch.Generator(device='cuda').manual_seed(42)
        loss,stats=loss_forward(pipe,adapter,generation,source,clean,generator,backward_scale=1,probe_sigma=.65)
        gradients=gradient_report(adapter)
        if gradients['final_heads']<=0:raise AssertionError('final zero heads have no flow gradient')
        if update and any(gradients[k]<=0 for k in ['query','global','relative','mutual','appearance','depth','contact']):
            raise AssertionError(f'upstream controller gradients missing after one update: {gradients}')
        if any(p.grad is not None for p in pipe.transformer.parameters()):raise AssertionError('native backbone received gradients')
        torch.nn.utils.clip_grad_norm_([p for p in adapter.parameters()if p.requires_grad],1.)
        optimizer.step()
        gate[f'update_{update+1}']=dict(loss=float(loss.detach()),gradient_norms=gradients)
        print('GATE_UPDATE',update+1,gate[f'update_{update+1}'],flush=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        probe_state=prepared_for_qwen(adapter,generation)
        gate['native_cache_control']=cache_probe(pipe.transformer,adapter,probe_state,inputs,probe_sigma)
    del probe_state
    print('CACHE_GATE',gate['native_cache_control'],flush=True)
    gate['backbone_after']=tensor_model_sha256(pipe.transformer)
    if gate['backbone_after']!=before:raise AssertionError('native backbone parameters changed')
    gate['passed']=True
    (a.output/'real_model_gates.json').write_text(json.dumps(gate,indent=2)+'\n')
    print('REAL_MODEL_GATES_PASSED',flush=True)
    if a.verify_only:return
    # Restart from the actual single geometry import, so gate optimizer steps do
    # not become hidden pilot steps or change the declared initialization.
    del optimizer,adapter;gc.collect();torch.cuda.empty_cache()
    adapter=DualV8Adapter();warm_start_from_body(adapter,a.warm_start,
        require_frozen_coverage=True,expected_source_step=19500)
    adapter.freeze_warm_geometry().to('cuda',torch.float32).train()
    optimizer=torch.optim.AdamW([p for p in adapter.parameters()if p.requires_grad],lr=a.lr,weight_decay=.01)
    if loaded is not None:
        adapter.load_state_dict(loaded.state_dict());optimizer.load_state_dict(resume_payload['optimizer_state_dict'])
        del loaded,resume_payload
    metrics=(a.output/'train_metrics.jsonl').open('a',buffering=1)
    began=time.monotonic()
    for step in range(start,a.steps):
        optimizer.zero_grad(set_to_none=True)
        losses=[]
        for micro in range(a.accumulation):
            rng=torch.Generator(device='cuda').manual_seed(a.seed+step*a.accumulation+micro)
            loss,stats=loss_forward(pipe,adapter,generation,source,clean,rng,backward_scale=a.accumulation)
            losses.append(float(loss.detach()))
        gradients=gradient_report(adapter)
        torch.nn.utils.clip_grad_norm_([p for p in adapter.parameters()if p.requires_grad],1.)
        optimizer.step()
        record=dict(step=step+1,loss=sum(losses)/len(losses),last_microbatch=stats,
                    gradients=gradients,elapsed_seconds=time.monotonic()-began)
        metrics.write(json.dumps(record)+'\n')
        print('TRAIN',step+1,record['loss'],'elapsed',round(record['elapsed_seconds'],1),flush=True)
        if (step+1)%a.save_every==0 or step+1==a.steps:
            adapter.eval()
            with torch.no_grad():
                validation=[]
                for seed in [420001,420002,420003]:
                    loss,_=loss_forward(pipe,adapter,generation,source,clean,torch.Generator(device='cuda').manual_seed(seed),teacher_weight=0,budget_weight=0)
                    validation.append(float(loss))
            score=sum(validation)/len(validation)
            info=dict(metadata,noise_validation=score,best_noise_validation=min(best,score),
                      validation_scope='same image, held-out noise only; no generalization claim')
            save_checkpoint(a.output/'checkpoint_last.pt',adapter,info,optimizer=optimizer,step=step+1)
            if score<best:
                best=score;save_checkpoint(a.output/'checkpoint_best.pt',adapter,info,optimizer=optimizer,step=step+1)
            print('SAVE',step+1,'noise_validation',score,flush=True)
            adapter.train()
    metrics.close()
    after=tensor_model_sha256(pipe.transformer)
    if after!=before:raise AssertionError('native backbone changed during pilot')
    (a.output/'training_complete.json').write_text(json.dumps(dict(steps=a.steps,backbone_sha256=after,
        best_noise_validation=best,elapsed_seconds=time.monotonic()-began),indent=2)+'\n')


if __name__=='__main__':main()
