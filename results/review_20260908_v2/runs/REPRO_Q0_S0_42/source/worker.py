"""A single isolated GPU job. Never run jobs concurrently on the same GPU."""
import argparse
import gc
import hashlib
import inspect
import shutil
import json
import sys
import time
import traceback
from pathlib import Path

from .common import MODEL, PROJECT, REVISION, TASKS, Monitor, environment, preprocess, sha256, source_path, write_json


def load_pipeline(monitor):
    import torch
    from diffusers import DiffusionPipeline
    sys.path.insert(0, str(MODEL))
    with monitor.section('load_complete_pipeline'):
        pipe = DiffusionPipeline.from_pretrained(str(MODEL), torch_dtype=torch.bfloat16).to('cuda')
        pipe._load_extras(attn_implementation='sdpa')
        for module in (pipe.transformer, pipe.vae, pipe.lmm, pipe.connector_module):
            module.eval().requires_grad_(False)
    return pipe


def input_image(config, out):
    from PIL import Image
    path = source_path(config['task'])
    original = Image.open(path).convert('RGB')
    image, box = preprocess(original, config['resolution'],config['preprocess'])
    config.update(input_sha256=sha256(path),input_path=str(path),original_size=list(original.size),content_box=box,
                  model_revision=REVISION, pipeline_sha256=sha256(MODEL/'deepgen_pipeline.py'))
    original.save(out/'input_original.png')
    image.save(out/'input_model.png')
    write_json(out/'config.json',config)
    return image


def native_quality(config, out, monitor):
    import torch
    image = input_image(config,out)
    pipe = load_pipeline(monitor)
    if config['kind']=='adapter_infer':
        sys.path.insert(0,str(PROJECT))
        from src.adapter.pose_adapter import PoseConditionAdapter
        with monitor.section('load_trained_adapter'):
            saved=torch.load(config['adapter_checkpoint'],map_location='cpu',weights_only=True)
            adapter=PoseConditionAdapter(num_blocks={'S':2,'M':4,'L':8}[saved['config']['adapter_size']],num_injection_layers=6).to('cuda')
            adapter.load_state_dict(saved['adapter'],strict=True)
            adapter.eval().requires_grad_(False)
            del saved
            config['adapter_checkpoint_sha256']=sha256(config['adapter_checkpoint'])
            value=controls(config,out)
            control=torch.from_numpy(value.copy()).permute(2,0,1)[None].to('cuda',dtype=torch.float32)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                residuals=[torch.nn.functional.pad(v,(0,0,0,v.shape[1])).to(torch.bfloat16) for v in adapter(control)]
            del control,value
            original_forward=pipe.transformer.forward
            def controlled_forward(*args,**kwargs):
                b=len(kwargs['hidden_states'])
                assert all(len(refs)==1 for refs in kwargs['cond_hidden_states'])
                kwargs['block_controlnet_hidden_states']=[v.expand(b,-1,-1) for v in residuals]
                return original_forward(*args,**kwargs)
            pipe.transformer.forward=controlled_forward
    # Source VAE sampling uses the global RNG; pipeline's seed only controls denoising.
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    config['rng_policy']='global_cpu_cuda_and_denoising_generator_same_seed'
    write_json(out/'config.json',config)
    timings = {}
    handles = []
    pending = {}
    for name,module in [('vlm',pipe.lmm.language_model),('visual_encoder',pipe.lmm.visual),('connector',pipe.connector_module.connector),
                        ('connector_proj1',pipe.connector_module.projector_1),('connector_proj2',pipe.connector_module.projector_2),
                        ('connector_proj3',pipe.connector_module.projector_3),('dit',pipe.transformer),('vae_encoder',pipe.vae.encoder),('vae_decoder',pipe.vae.decoder)]:
        def pre(module,args,n=name):
            e=torch.cuda.Event(enable_timing=True);e.record();pending[n]=e
            monitor.stage=n
        def post(module,args,result,n=name):
            e=torch.cuda.Event(enable_timing=True);e.record();timings.setdefault(n,[]).append((pending.pop(n),e))
            monitor.stage='inference_other'
        handles += [module.register_forward_pre_hook(pre),module.register_forward_hook(post)]
    with monitor.section('edit_end_to_end_no_file_io'):
        output=pipe(prompt=config['prompt'],image=image,negative_prompt=config['negative_prompt'],
                    height=config['resolution'],width=config['resolution'],num_inference_steps=config['steps'],
                    guidance_scale=config['cfg'],seed=config['seed']).images[0]
    for h in handles:
        h.remove()
    component_ms={n:sum(a.elapsed_time(b) for a,b in events) for n,events in timings.items()}
    start=time.perf_counter()
    output.save(out/'output.png')
    output.crop(config['content_box']).save(out/'output_content.png')
    return dict(component_cuda_ms=component_ms,file_save_seconds=time.perf_counter()-start,
                interpretation=('Reconstruction-trained adapter with simulated controls; not pose-transfer validation.' if config['kind']=='adapter_infer' else 'Native image+text edit; no structural adapter injected.'))


def cache_key(config, task=None):
    task=task or config['task']
    fields=dict(input_sha256=sha256(source_path(task)),prompt=TASKS[task],resolution=config['resolution'],
                preprocess=config['preprocess'],model_revision=REVISION,pipeline_sha256=sha256(MODEL/'deepgen_pipeline.py'),
                encoding_seed=1701,vae_sampling='sample',dtype='bfloat16',condition_batch=1,
                encoding_implementation_sha256=hashlib.sha256((inspect.getsource(encode)+inspect.getsource(preprocess)).encode()).hexdigest())
    digest=hashlib.sha256(json.dumps(fields,sort_keys=True).encode()).hexdigest()
    return digest,fields


def encode(pipe,image,prompt):
    """Positive condition only for training; same VLM/SCB/source VAE operations as native pipeline."""
    import numpy as np
    import torch
    torch.manual_seed(1701)
    with torch.no_grad():
        pixels=torch.from_numpy(np.asarray(image).copy()).float()/255.0
        pixels=(2*pixels-1).permute(2,0,1).unsqueeze(0).to('cuda',dtype=torch.bfloat16)
        embeds,grid=pipe.get_semantic_features_dynamic([pixels[0]])
        ti=pipe.prepare_image2image_prompts([prompt],num_refs=[1],ref_lens=[len(embeds[0])])
        ti.update(image_embeds=torch.cat(embeds),image_grid_thw=grid)
        queries=pipe.connector_module.meta_queries[None]
        args=pipe.prepare_forward_input(query_embeds=queries,**ti)
        output=pipe.llm(**args,return_dict=True,output_hidden_states=True)
        hs=output.hidden_states
        merged=torch.cat([hs[i] for i in range(len(hs)-2,0,-6)],dim=-1)
        pooled,seq=pipe.connector_module.llm2dit(merged)
        ref=pipe.pixels_to_latents(pixels)
        target=pipe.pixels_to_latents(pixels)
    return {'seq':seq.detach(),'pooled':pooled.detach(),'ref':ref.detach(),'target':target.detach()}


def controls(config,out):
    import numpy as np
    from PIL import Image
    sys.path.insert(0,str(PROJECT))
    from test_control_signals import create_synthetic_smplx_human
    from src.controls.generator import Camera,ControlSignalGenerator
    r=config['resolution'];s=r/512
    people=[create_synthetic_smplx_human(0,0)] if config['task'].startswith('S') else [create_synthetic_smplx_human(-.2,0),create_synthetic_smplx_human(.2,0)]
    camera=Camera(fx=600*s,fy=600*s,cx=r/2,cy=r/2,image_size=(r,r),T=np.array([0,0,2.8],dtype=np.float32))
    result=ControlSignalGenerator((r,r)).generate_all_control_maps(people,camera)
    value=result['stacked_control_8ch']
    assert value.shape==(r,r,8) and np.isfinite(value).all() and value.min()>=0 and value.max()<=1
    for name,key in [('normal','normal_rgb'),('depth','depth_map'),('skeleton','skeleton_rgb'),('contact','contact_heatmap')]:
        array=result[key]
        if array.dtype!=np.uint8:
            array=(array*255).astype('uint8')
        if array.shape[-1]==1:
            array=array[...,0]
        Image.fromarray(array).save(out/(name+'.png'))
    np.save(out/'control.npy',value)
    write_json(out/'control_metadata.json',dict(source='artificial skeleton and pointcloud; not SMPL-X ground truth',
               channel_order=['normal_r','normal_g','normal_b','depth','skeleton_r','skeleton_g','skeleton_b','contact'],
               shape=list(value.shape),channel_min=value.min((0,1)).tolist(),channel_max=value.max((0,1)).tolist(),
               contact_table=result['contact_table'],aligned_to_photo=False))
    return value


def cache_job(config,out,monitor,root):
    import torch
    image=input_image(config,out)
    pipe=load_pipeline(monitor)
    with monitor.section('cache_encode'):
        tensors=encode(pipe,image,TASKS[config['task']])
    key,fields=cache_key(config)
    target=root/'cache'/key
    target.mkdir(parents=True,exist_ok=True)
    cpu={k:v.cpu() for k,v in tensors.items()}
    torch.save(cpu,target/'condition.pt')
    restored=torch.load(target/'condition.pt',weights_only=True)
    assert all(torch.equal(cpu[k],restored[k]) for k in cpu)
    write_json(target/'metadata.json',dict(**fields,cache_key=key,source_run=config['id'],tensor_sha256=sha256(target/'condition.pt'),
                                          shapes={k:list(v.shape) for k,v in cpu.items()}))
    return dict(cache_key=key,shapes={k:list(v.shape) for k,v in cpu.items()},cache_roundtrip_exact=True)


def module_digest(module):
    import torch
    h=hashlib.sha256()
    for name,tensor in module.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def train_job(config,out,monitor,root):
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    sys.path.insert(0,str(PROJECT))
    from src.adapter.pose_adapter import PoseConditionAdapter
    from diffusers import DiffusionPipeline
    image=input_image(config,out)
    if config['encoding']=='online':
        pipe=load_pipeline(monitor)
    else:
        sys.path.insert(0,str(MODEL))
        with monitor.section('load_cached_training_backbone'):
            pipe=DiffusionPipeline.from_pretrained(str(MODEL),torch_dtype=torch.bfloat16)
            pipe.transformer.to('cuda').eval().requires_grad_(False)
            pipe.vae.eval().requires_grad_(False)
    transformer=pipe.transformer
    with monitor.section('adapter_optimizer_setup'):
        torch.manual_seed(config['seed'])
        n={'S':2,'M':4,'L':8}[config['adapter_size']]
        adapter=PoseConditionAdapter(in_channels=8,hidden_dim=1536,num_blocks=n,num_injection_layers=6).to('cuda',dtype=torch.float32)
        optimizer=torch.optim.AdamW(adapter.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01,foreach=False,fused=False)
        ckpt=config['checkpointing']
        if ckpt:
            transformer.enable_gradient_checkpointing(gradient_checkpointing_func=lambda fn,*a:checkpoint(fn,*a,use_reentrant=False))
        stages=[adapter.init_conv,adapter.down1,adapter.down2,adapter.down3,adapter.down4,*adapter.mid_blocks,adapter.out_proj]
    ctrl=controls(config,out)
    control=torch.from_numpy(ctrl.copy()).permute(2,0,1)[None].to('cuda',dtype=torch.float32)
    del ctrl
    images={config['task']:image}
    conditions={}
    tasks=['S0','D0'] if config.get('alternate') else [config['task']]
    from PIL import Image
    for task in tasks:
        images[task]=preprocess(Image.open(source_path(task)),config['resolution'],config['preprocess'])[0]
        if config['encoding']=='cache':
            key,fields=cache_key(config,task)
            path=root/'cache'/key
            metadata=json.loads((path/'metadata.json').read_text())
            assert all(metadata[k]==v for k,v in fields.items())
            assert sha256(path/'condition.pt')==metadata['tensor_sha256']
            conditions[task]=torch.load(path/'condition.pt',map_location='cpu',weights_only=True)
    modules={'transformer':transformer,'vae':pipe.vae}
    if config['encoding']=='online':
        modules.update(lmm=pipe.lmm,connector=pipe.connector_module)
    before={k:module_digest(v) for k,v in modules.items()}
    assert all(not p.requires_grad for m in modules.values() for p in m.parameters())
    count={'dit':0,'adapter':0}
    hd=transformer.transformer_blocks[1].register_forward_pre_hook(lambda *_:count.__setitem__('dit',count['dit']+1))
    ha=adapter.down1.register_forward_pre_hook(lambda *_:count.__setitem__('adapter',count['adapter']+1))

    def residuals():
        x=control
        for stage in stages:
            x=checkpoint(stage,x,use_reentrant=False) if ckpt and torch.is_grad_enabled() else stage(x)
        result=[head(x).flatten(2).transpose(1,2) for head in adapter.zero_convs]
        target_tokens=(config['resolution']//16)**2
        assert all(v.shape==(1,target_tokens,1536) for v in result)
        return [F.pad(v,(0,0,0,target_tokens)).to(torch.bfloat16) for v in result]

    def condition(task):
        if config['encoding']=='online':
            return encode(pipe,images[task],TASKS[task])
        return {k:v.to('cuda') for k,v in conditions[task].items()}

    generator=torch.Generator(device='cuda').manual_seed(config['seed'])
    def forward(c,rs,noise,index):
        sigma=pipe.scheduler.sigmas[index].to('cuda',dtype=torch.bfloat16).reshape(1,1,1,1)
        t=pipe.scheduler.timesteps[index].to('cuda').reshape(1)
        z=(1-sigma)*c['target']+sigma*noise
        result=transformer(hidden_states=z,encoder_hidden_states=c['seq'],pooled_projections=c['pooled'],
                           cond_hidden_states=[[c['ref'][0]]],timestep=t,block_controlnet_hidden_states=rs,return_dict=False)[0]
        return result,noise-c['target']

    with monitor.section('zero_equivalence_validation'):
        c=condition(tasks[0])
        noise=torch.randn(c['target'].shape,device='cuda',dtype=torch.bfloat16,generator=generator)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            zero=residuals()
            assert all(v.count_nonzero()==0 for v in zero)
            a,_=forward(c,None,noise,500);b,_=forward(c,zero,noise,500)
            max_diff=(a-b).abs().max().item()
            assert torch.allclose(a,b,rtol=.01,atol=.01)
        del c,noise,zero,a,b
    # No validation graph survives into benchmarking.
    gc.collect();torch.cuda.empty_cache()
    metrics=[];gradient_gate=None;recompute_gate=None;semantic_shapes={}
    total=config['warmup']+config['updates']
    for update in range(total):
        task=tasks[update%len(tasks)]
        if config.get('alternate') and update>0:
            cfg=dict(config,task=task)
            # Rebuild matching simulated controls, independently of the photo, outside timed update.
            value=controls(cfg,out)
            control=torch.from_numpy(value.copy()).permute(2,0,1)[None].to('cuda',dtype=torch.float32)
            del value
        stage='first_update' if update==0 else ('warmup' if update<config['warmup'] else 'steady_train')
        with monitor.section(stage):
            optimizer.zero_grad(set_to_none=True)
            c=condition(task)
            semantic_shapes[task]={k:list(v.shape) for k,v in c.items()}
            noise=torch.randn(c['target'].shape,device='cuda',dtype=torch.bfloat16,generator=generator)
            index=int(torch.randint(0,pipe.scheduler.config.num_train_timesteps,(1,),generator=generator,device='cuda').item())
            count.update(dit=0,adapter=0)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                if update==0:
                    monitor.stage='first_forward'
                    monitor.event(stage='first_forward',event='start')
                rs=residuals()
                pred,target=forward(c,rs,noise,index)
                loss=F.mse_loss(pred.float(),target.float())
            assert torch.isfinite(loss).item()
            if update==0:
                monitor.stage='first_backward'
                monitor.event(stage='first_backward',event='start',allocated_gib=torch.cuda.memory_allocated()/2**30)
            loss.backward()
            if update==2:
                heads=[sum(p.grad.float().abs().sum().item() for p in head.parameters() if p.grad is not None)>0 for head in adapter.zero_convs]
                body=sum(p.grad.float().abs().sum().item() for name,p in adapter.named_parameters() if not name.startswith('zero_convs') and p.grad is not None)>0
                assert all(heads) and body
                assert all(p.grad is None for m in modules.values() for p in m.parameters())
                gradient_gate=dict(heads_nonzero=heads,body_nonzero=body,backbone_gradients_absent=True)
                recompute_gate=dict(count)
                assert (count['dit']>1 and count['adapter']>1) if ckpt else (count['dit']==1 and count['adapter']==1)
            norm=torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.0,error_if_nonfinite=True)
            if update==0:
                monitor.stage='first_optimizer'
                monitor.event(stage='first_optimizer',event='start',allocated_gib=torch.cuda.memory_allocated()/2**30)
            optimizer.step()
            value=loss.item()
            del c,noise,rs,pred,target,loss
        metrics.append(dict(update=update+1,task=task,loss=value,gradient_norm=float(norm),seconds=monitor.stages[-1]['seconds'],phase=stage))
        print(json.dumps(metrics[-1]),flush=True)
    hd.remove();ha.remove()
    assert all(torch.isfinite(p).all().item() for p in adapter.parameters())
    after={k:module_digest(v) for k,v in modules.items()}
    assert before==after
    assert gradient_gate is not None
    optimizer_dtypes=sorted({str(v.dtype) for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v)})
    assert optimizer_dtypes==['torch.float32']
    write_json(out/'updates.json',metrics)
    write_json(out/'validation.json',dict(backbone_before=before,backbone_after=after,zero_max_abs_diff=max_diff,
               gradients=gradient_gate,recomputation=recompute_gate,optimizer_dtypes=optimizer_dtypes))
    torch.save(dict(adapter={k:v.detach().cpu() for k,v in adapter.state_dict().items()},config=config),out/'adapter.pt')
    import numpy as np
    times=[m['seconds'] for m in metrics if m['phase']=='steady_train']
    return dict(adapter_parameters=sum(p.numel() for p in adapter.parameters()),validation_passed=True,
                train_steps=len(metrics),steady_steps=len(times),step_p50_seconds=float(np.percentile(times,50)),
                step_p95_seconds=float(np.percentile(times,95)),semantic_shapes=semantic_shapes,
                interpretation='Real image reconstruction with simulated controls; not pose-edit quality training.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8'))
    out=args.root/'runs'/config['id']
    out.mkdir(parents=True,exist_ok=False)
    config['source_hashes']={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}
    snapshot=out/'source';snapshot.mkdir()
    for p in Path(__file__).parent.glob('*.py'):
        shutil.copy2(p,snapshot/p.name)
    shutil.copy2(PROJECT/'src/adapter/pose_adapter.py',snapshot/'pose_adapter.py')
    config['adapter_source_sha256']=sha256(PROJECT/'src/adapter/pose_adapter.py')
    config['scheduler_config_sha256']=sha256(MODEL/'scheduler/scheduler_config.json')
    write_json(out/'config.json',config)
    monitor=None
    summary={};status='INVALID';started=time.perf_counter()
    try:
        write_json(out/'environment.json',environment())
        monitor=Monitor(out)
        if config['kind'] in ('quality','adapter_infer'):
            summary=native_quality(config,out,monitor)
        elif config['kind']=='cache':
            summary=cache_job(config,out,monitor,args.root)
        elif config['kind']=='train':
            summary=train_job(config,out,monitor,args.root)
        else:
            raise ValueError(config['kind'])
        status='PASS'
    except BaseException as e:
        import torch
        status='OOM' if isinstance(e,torch.cuda.OutOfMemoryError) else ('CANCELLED' if isinstance(e,KeyboardInterrupt) else 'INVALID')
        summary.update(error=repr(e),traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        if monitor:
            summary.update(monitor.close())
            if status=='PASS' and (summary.get('nvml_error') or not summary.get('nvml_sample_count') or not summary.get('peak_nvml_gib')):
                status='INVALID'
                summary['error']='NVML measurement missing or failed; cannot publish a memory result'
        summary.update(status=status,process_wall_seconds=time.perf_counter()-started)
        write_json(out/'summary.json',summary)
    return 0 if status=='PASS' else 1

if __name__=='__main__':
    sys.exit(main())
