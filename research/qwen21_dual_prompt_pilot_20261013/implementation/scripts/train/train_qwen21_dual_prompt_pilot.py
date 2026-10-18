"""Train matched short/structured dual_v8 adapters on real train/val pairs."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
from importlib.metadata import version
import inspect
import math
from pathlib import Path
import sys

import torch
from torchvision.transforms.functional import to_pil_image

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))

from src.data.hi4d_smplx_dataset import file_sha256
from src.pose_control.qwen21_full.dual.checkpoint import encoder_provenance,load_checkpoint,save_checkpoint
from src.pose_control.qwen21_full.dual.context import DualV8ControlContext
from src.pose_control.qwen21_full.dual.experiment import tensor_cache_sha256
from src.pose_control.qwen21_full.dual.precision import precision_policy,qwen21_precision
from src.pose_control.qwen21_full.dual.prompt_pilot import (
    APPROVED_TRAINING_ROWS,CachedSample,DATA_SCHEMA_VERSION,SMOKE_TRAINING_SCHEMA_VERSION,
    TRAINING_SCHEMA_VERSION,VARIANTS,atomic_json,fresh_adapter,json_sha256,
    load_training_data,prompt_pilot_code_sha256,require_fresh_output,run_variant_updates,
    schedule_sha256,shared_schedule,source_prompt_cache_key)
from src.pose_control.qwen21_full.dual.runtime import (
    flow_sample,load_native,native_inputs,prepared_for_qwen,source_cache,tensor_model_sha256)
from src.pose_control.qwen21_full.qwen_runtime import Qwen21IPERCache
from scripts.train.train_qwen21_dual_v8_pilot import loss_forward


MODEL = '/home/shangguanrz/project/pic-edit/models/Qwen-Image-2.1'
WARM_START = '/home/shangguanrz/project/pic-edit/experiments/3.singel man/qwen21_iper_body/full_20k/checkpoint_best.pt'
PROMPT_PILOT_VERSION = 'qwen21_dual_prompt_pilot_v1'


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=300)
    parser.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--seed',type=int,default=14062)
    parser.add_argument('--save-every',type=int,default=50)
    parser.add_argument('--model',type=Path,default=Path(MODEL))
    parser.add_argument('--warm-start',type=Path,default=Path(WARM_START))
    args=parser.parse_args(argv)
    validate_training_settings(args)
    return args


def validate_training_settings(args, *, smoke=False):
    if type(smoke) is not bool:
        raise ValueError('smoke must be an explicit boolean')
    if not 1<=args.steps<=300 or not math.isfinite(args.lr) or args.lr<=0 or args.save_every<1:
        raise ValueError('training requires 1..300 steps and positive finite LR/save interval')
    if not args.variants or len(set(args.variants))!=len(args.variants) or any(variant not in VARIANTS for variant in args.variants):
        raise ValueError('prompt variants must be unique independent runs')
    if not 0<=args.seed<=2**63-1200:
        raise ValueError('prompt pilot seed must be a valid nonnegative integer')
    if not smoke and (args.steps!=300 or set(args.variants)!=set(VARIANTS) or args.lr!=1e-4 or args.save_every!=50):
        raise ValueError('approved production pilot requires 300 steps, both short/structured variants, LR 1e-4 and validation every50 steps')


def cache_training_samples(pipe,data,output,variants):
    """Cache each exact source/prompt, sharing target encoding across variants."""
    target_cache=Qwen21IPERCache(output/'native_cache'/'shared_targets',
        prompt='shared_target_latents_for_matched_prompt_pilot_v1',
        preprocessing='affine_letterbox_v8_bf16_full_reduction_v1')
    samples={variant:{name:[] for name in ('train','val')} for variant in variants}
    evidence={variant:{name:[] for name in ('train','val')} for variant in variants}
    for name,split in data.splits.items():
        for index,sample_id in enumerate(split.sample_ids):
            base_generation=split.generation(index,'short')
            truth=split.supervision(index)
            clean=target_cache.target(pipe,to_pil_image(truth.rgb[0]))
            reference=None
            conditions=[]
            for variant in variants:
                row=split.rows[variant][index]
                generation=replace(base_generation,prompts=(row['prompt'],))
                key=source_prompt_cache_key(row)
                directory=output/'native_cache'/variant/name/key
                _,source=source_cache(pipe,generation,directory)
                cached=CachedSample(sample_id,generation,source,clean)
                if reference is not None and not torch.equal(cached.source['reference_latent'],reference):
                    raise AssertionError('matched prompt variants have different encoded source image latents')
                reference=cached.source['reference_latent']
                conditions.append(cached.source)
                samples[variant][name].append(cached)
                evidence[variant][name].append(dict(sample_id=sample_id,prompt=row['prompt'],
                    source_prompt_cache_key=key,cache_directory=str(directory.resolve()),
                    source_content_sha256=tensor_cache_sha256(cached.source),
                    target_latent_content_sha256=tensor_cache_sha256(cached.clean),
                    combined_content_sha256=cached.content_sha256,geometry_sha256=generation.geometry.geometry_hash))
                print('CACHE',name,variant,sample_id,flush=True)
            if len(conditions)==2 and all(torch.equal(conditions[0][key],conditions[1][key])
                                         for key in ('prompt_embeds','prompt_mask')):
                raise AssertionError('different prompts produced identical text conditions')
            del truth,clean,base_generation,reference,conditions
    data.assert_unchanged()
    return samples,evidence


def zero_initialization_gate(pipe,adapter,sample):
    generation=sample.generation.to('cuda',torch.float32)
    clean=sample.clean.to('cuda',torch.bfloat16)
    generator=torch.Generator(device='cuda').manual_seed(42)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        noise=torch.randn(clean.shape,device=clean.device,dtype=clean.dtype,generator=generator)
        sigma=torch.tensor([.65],device='cuda')
        noisy,_=flow_sample(clean,noise,sigma)
        inputs=native_inputs(sample.source,noisy,sigma)
        native=pipe.transformer(**inputs)[0]
        prepared=prepared_for_qwen(adapter,generation)
        with DualV8ControlContext(pipe.transformer,adapter,prepared,sigma=sigma):
            controlled=pipe.transformer(**inputs)[0]
        if not torch.equal(native,controlled):
            raise AssertionError('fresh zero output heads differ from real native Qwen')
    return dict(zero_init_exact=True,probe_seed=42,probe_sigma=.65,
                gradients_none=all(param.grad is None for param in adapter.parameters()),optimizer_steps=0)


def manifest_hashes(data):
    return {name:{key:file_sha256(path) for key,path in split.manifest_paths.items()}
            for name,split in data.splits.items()}


def checkpoint_record(path,step):
    return dict(path=str(path.resolve()),sha256=file_sha256(path),step=step)


def train_one_variant(args,pipe,data,samples,cache_evidence,variant,common,schedule,expected_initial):
    output=args.output/variant
    require_fresh_output(output)
    adapter,warm,initial=fresh_adapter(args.seed,args.warm_start,device='cuda')
    if expected_initial is not None and initial!=expected_initial:
        raise AssertionError('matched variants did not start from identical adapter tensors')
    device=next(adapter.parameters()).device
    latent_dtype=torch.bfloat16 if device.type=='cuda' else torch.float32
    optimizer=torch.optim.AdamW([param for param in adapter.parameters() if param.requires_grad],
                               lr=args.lr,weight_decay=.01)
    optimizer_contract=dict(name='AdamW',**{key:(list(value) if isinstance(value,tuple) else value)
        for key,value in optimizer.param_groups[0].items() if key!='params'})
    metadata=dict(common,variant=variant,initial_adapter_sha256=initial,loading_report=warm,
        prompt_pilot_schema_version=PROMPT_PILOT_VERSION if common['production_training'] else 'qwen21_dual_prompt_smoke_v1',
        cache_content=cache_evidence,
        prompts={name:[sample.generation.prompts[0] for sample in rows] for name,rows in samples.items()},
        optimizer=optimizer_contract,
        optimizer_parameter_names=[name for name,param in adapter.named_parameters() if param.requires_grad],
        frozen_scopes=['geometry_encoder.spatial_encoder','reasoner.person_reasoner','reasoner.shared_role_offset'],
        objective=dict(flow_body_extra=1.,flow_contact_extra=1.,background_teacher_weight=.1,
                       budget_weight=.01,budget_ratio=.05),
        validation_selection='true_val_split_only',validation_seeds=[420001,420002,420003],
        warm_start_initialization='fresh_dual_v8_plus_frozen_iper_body_step19500')
    metadata['run_contract']={key:value for key,value in metadata.items() if key not in ('loading_report',)}
    gate=zero_initialization_gate(pipe,adapter,samples['train'][0])
    gate.update(initial_adapter_sha256=initial,frozen_coverage=warm['frozen_coverage'],source_step=warm['source_step'])
    atomic_json(output/'real_model_gates.json',gate)
    atomic_json(output/'run_manifest.json',dict(metadata,requested_steps=args.steps,save_every=args.save_every))
    current_best={'step':None,'loss':float('inf')}
    def training_forward(item,backward_scale):
        sample=samples['train'][item.sample_index]
        if sample.sample_id!=item.sample_id:
            raise AssertionError('cached sample disagrees with shared schedule')
        generation=sample.generation.to(device,torch.float32)
        clean=sample.clean.to(device,latent_dtype)
        generator=torch.Generator(device=device).manual_seed(item.noise_seed)
        return loss_forward(pipe,adapter,generation,sample.source,clean,generator,
                            backward_scale=backward_scale,teacher_weight=.1,budget_weight=.01)
    def validation_forward(index,seed):
        sample=samples['val'][index]
        generation=sample.generation.to(device,torch.float32)
        clean=sample.clean.to(device,latent_dtype)
        loss,_=loss_forward(pipe,adapter,generation,sample.source,clean,
            torch.Generator(device=device).manual_seed(seed),teacher_weight=0,budget_weight=0)
        return loss
    def on_save(step,score,is_best,evidence):
        if is_best:
            current_best.update(step=step,loss=score)
        info=dict(metadata,step=step,validation_loss=score,best_validation=current_best['loss'],
                  best_step=current_best['step'],validation=evidence,
                  validation_selection='true_val_split_only')
        save_checkpoint(output/'checkpoint_last.pt',adapter,info,optimizer=optimizer,step=step)
        if is_best:
            save_checkpoint(output/'checkpoint_best.pt',adapter,info,optimizer=optimizer,step=step)
        print('SAVE',variant,step,'validation',score,'best_step',current_best['step'],flush=True)
    result=run_variant_updates(adapter,pipe.transformer,optimizer,schedule,training_forward,validation_forward,
        sample_count=len(samples['val']),save_every=args.save_every,on_save=on_save,
        metrics_path=output/'train_metrics.jsonl')
    for rows in samples.values():
        for sample in rows:
            sample.assert_unchanged()
    data.assert_unchanged()
    # Check the actual exported artifact with the unchanged public loader.
    loaded,info,payload=load_checkpoint(output/'checkpoint_best.pt')
    if payload['step']!=result['best_step'] or info['best_validation']!=result['best_validation']:
        raise AssertionError('actual best checkpoint does not match true validation selection')
    if info['variant']!=variant or info['validation_selection']!='true_val_split_only':
        raise AssertionError('best checkpoint lost prompt variant or true validation provenance')
    result.update(variant=variant,status='success' if common['production_training'] else 'smoke_success',
        schema_version=TRAINING_SCHEMA_VERSION if common['production_training'] else SMOKE_TRAINING_SCHEMA_VERSION,
        production_training=common['production_training'],
        requested_steps=args.steps,dataset_sha256=data.dataset_sha256,split_manifest=data.split_manifest,
        model_id=common['model_id'],initial_adapter_sha256=initial,
        validation_selection='true_val_split_only',train_pair_ids=common['train_pair_ids'],
        val_pair_ids=common['val_pair_ids'],train_sample_ids=common['train_sample_ids'],val_sample_ids=common['val_sample_ids'],
        best_checkpoint=checkpoint_record(output/'checkpoint_best.pt',payload['step']),
        last_checkpoint=checkpoint_record(output/'checkpoint_last.pt',args.steps),
        best_adapter_sha256=tensor_model_sha256(loaded),
        trained_adapter_sha256=tensor_model_sha256(adapter),caches_unchanged=True,
        native_gradients_none=all(param.grad is None for param in pipe.transformer.parameters()),
        warm_start_sha256=common['warm_start_sha256'])
    del loaded,optimizer,adapter
    gc.collect();torch.cuda.empty_cache()
    return result,initial


@qwen21_precision()
def run_training(args, *, smoke=False):
    """Internal smoke runs produce distinct records that cannot pass test preflight."""
    validate_training_settings(args,smoke=smoke)
    args.output=args.output.resolve()
    require_fresh_output(args.output)
    data=load_training_data(args.data_dir)
    schedule=shared_schedule(data.splits['train'].sample_ids,steps=args.steps,seed=args.seed)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    warm_sha=file_sha256(args.warm_start)
    closure=prompt_pilot_code_sha256(PROJECT)
    atomic_json(args.output/'shared_schedule.json',[asdict(item) for item in schedule])
    pipe=load_native(args.model)
    print('NATIVE_LOADED',flush=True)
    before=tensor_model_sha256(pipe.transformer)
    cached,cache_evidence=cache_training_samples(pipe,data,args.output,args.variants)
    atomic_json(args.output/'cache_manifest.json',cache_evidence)
    pipe.text_encoder.to('cpu');pipe.vae.to('cpu');gc.collect();torch.cuda.empty_cache()
    pipe.transformer.enable_gradient_checkpointing()
    common=dict(production_training=not smoke,
        approved_training_selection_sha256=json_sha256(APPROVED_TRAINING_ROWS),
        model_id=str(args.model.resolve()),model_config_sha256=file_sha256(args.model/'transformer/config.json'),
        dataset_schema_version=DATA_SCHEMA_VERSION,dataset_path=str(data.dataset_path),dataset_sha256=data.dataset_sha256,
        split_manifest=data.split_manifest,manifest_sha256=manifest_hashes(data),
        data_content_sha256=data.content_sha256,
        train_pair_ids=list(data.splits['train'].pair_ids),val_pair_ids=list(data.splits['val'].pair_ids),
        authoritative_pair_ids={name:list(values) for name,values in data.authoritative_pair_ids.items()},
        train_sample_ids=list(data.splits['train'].sample_ids),val_sample_ids=list(data.splits['val'].sample_ids),
        seed=args.seed,effective_batch=4,accumulation=4,learning_rate=args.lr,requested_steps=args.steps,
        schedule_sha256=schedule_sha256(schedule),warm_start_path=str(args.warm_start.resolve()),warm_start_sha256=warm_sha,
        backbone_sha256=before,backbone_sha256_before=before,
        precision_policy=precision_policy(),code_sha256=closure,**encoder_provenance(),
        runtime_versions={package:version(package) for package in ('torch','torchvision','diffusers','transformers','numpy','scipy','pillow')},
        native_code_sha256={name:file_sha256(inspect.getfile(type(component))) for name,component in
            (('transformer',pipe.transformer),('pipeline',pipe),('vae',pipe.vae),('text_encoder',pipe.text_encoder))},
        task='matched_multisample_prompt_pilot_target_geometry_known',test_data_read=False)
    atomic_json(args.output/'run_manifest.json',dict(common,variants=args.variants,save_every=args.save_every))
    results={};initials={};expected_initial=None
    for variant in args.variants:
        print('VARIANT',variant,flush=True)
        result,initial=train_one_variant(args,pipe,data,cached[variant],cache_evidence[variant],variant,
                                        common,schedule,expected_initial)
        results[variant]=result;initials[variant]=initial
        expected_initial=initial if expected_initial is None else expected_initial
    after=tensor_model_sha256(pipe.transformer)
    if after!=before or any(param.grad is not None for param in pipe.transformer.parameters()):
        raise AssertionError('native Qwen backbone changed during prompt training')
    if file_sha256(args.warm_start)!=warm_sha or prompt_pilot_code_sha256(PROJECT)!=closure:
        raise ValueError('warm-start or runtime source code changed during training')
    data.assert_unchanged()
    if len(set(initials.values()))!=1:
        raise AssertionError('independent prompt variants have different initial adapter SHA256')
    common_complete=dict(status='smoke_success' if smoke else 'success',
        schema_version=SMOKE_TRAINING_SCHEMA_VERSION if smoke else TRAINING_SCHEMA_VERSION,
        production_training=not smoke,requested_steps=args.steps,
        completed_steps=args.steps,dataset_path=str(data.dataset_path),dataset_sha256=data.dataset_sha256,
        split_manifest=data.split_manifest,model_id=common['model_id'],
        backbone_sha256_before=before,backbone_sha256_after=after,backbone_unchanged=True,
        initialization_sha256_equal=True,initial_adapter_sha256=expected_initial,
        initial_adapter_sha256_by_variant=initials,warm_start_sha256=warm_sha,
        train_pair_ids=common['train_pair_ids'],val_pair_ids=common['val_pair_ids'],
        validation_selection='true_val_split_only',effective_batch=4,schedule_sha256=common['schedule_sha256'],
        test_data_read=False,variants=results)
    # Completion appears only after all variants and integrity checks succeed.
    completion_name='smoke_complete.json' if smoke else 'training_complete.json'
    for variant,result in results.items():
        result.update(backbone_sha256_before=before,backbone_sha256_after=after,backbone_unchanged=True,
                      initialization_sha256_equal=True)
        atomic_json(args.output/variant/completion_name,result)
    atomic_json(args.output/completion_name,common_complete)
    print('PROMPT_SMOKE_COMPLETE' if smoke else 'PROMPT_TRAINING_COMPLETE',str(args.output/completion_name),flush=True)
    return common_complete


def main(argv=None):
    return run_training(parse_args(argv))


if __name__=='__main__':
    main()
