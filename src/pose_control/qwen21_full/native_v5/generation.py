"""Native-canvas sampling from the very same frozen multi-reference conditions.

No target RGB/latent is read. The installed Qwen pipeline's scheduler, VAE
normalisation and latent packing are retained; its area-based reference resize
is deliberately bypassed, without modifying the installed diffusers package.
"""
import numpy as np
import hashlib
from contextlib import nullcontext
import torch
from torch.nn import functional as F
from ..qwen_runtime import transformer_inputs
from ..native_full_data import identity_from_reference_v4
from .integration import bind_native_full_v5
from .conditioning import primary_source


@torch.no_grad()
def generate_native(pipe,runtime,batch,pair,*,options,base_options,seed=42,steps=20,enabled=True,observer=None,sampling_report=None):
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift,retrieve_timesteps
    device=next(pipe.transformer.parameters()).device
    pair={k:v.to(device) if torch.is_tensor(v) else v for k,v in pair.items() if k!='target_latent'}
    th,tw=pair['target_token_hw'];dtype=pair['prompt_embeds'].dtype
    generator=torch.Generator(device=device).manual_seed(seed)
    noise=torch.randn((1,1,64,th,tw),device=device,dtype=dtype,generator=generator)
    latent=pipe._pack_latents(noise,1,64,th,tw)
    if sampling_report is not None:
        sampling_report.update(initial_noise_sha256=hashlib.sha256(noise.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest(),
            noise_shape=list(noise.shape),noise_dtype=str(noise.dtype),seed=seed)
    scheduler=type(pipe.scheduler).from_config(pipe.scheduler.config)
    mu=calculate_shift(th*tw,scheduler.config.get('base_image_seq_len',256),scheduler.config.get('max_image_seq_len',4096),
        scheduler.config.get('base_shift',.5),scheduler.config.get('max_shift',1.15))
    timesteps,_=retrieve_timesteps(scheduler,steps,device,sigmas=np.linspace(1.,1/steps,steps),mu=mu)
    scheduler.set_begin_index(0)
    if sampling_report is not None:
        sampling_report.update(timesteps=timesteps.detach().float().cpu().tolist(),scheduler=type(scheduler).__name__)
    prepared=None
    if enabled:
        bank=batch.get('source_bank_v5')
        if pair.get('qwen_condition_profile')=='dual_source_5ref':
            if bank is None: raise ValueError('Five-reference Native generation requires two independent appearance sources')
            bank=bank.with_pair(pair)
            prepared=runtime.encode(batch,target_token_hw=(th,tw),source_bank=bank,options=options,base_options=base_options)
        else:
            if bank is not None: raise ValueError('Two-source Native generation requires the five-reference Qwen profile')
            source=primary_source(pair)
            identity=identity_from_reference_v4(batch,source,pair['source_token_hw']) if base_options.identity and base_options.identity_strength else None
            prepared=runtime.encode(batch,target_token_hw=(th,tw),identity=identity,source_latent=source,options=options,base_options=base_options)
    coverage=F.avg_pool2d(batch['valid_image_mask'].to(device).float(),16).flatten(1)

    def predict(t):
        sigma=t.expand(1).to(dtype)/1000
        values=transformer_inputs(pair,latent,sigma)
        if not enabled and hasattr(pipe.transformer,'_native_v5_execution'):
            values['native_enabled']=False
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            return pipe.transformer(**values)[0][:,-th*tw:]

    if enabled:
        if sampling_report is not None:
            sampling_report['condition_validity']=dict(person_valid=prepared.body.person_valid.tolist(),
                relation_tokens=int(prepared.relation_valid.sum()),local=None if prepared.local is None else dict(
                region_valid=prepared.local.region_valid.tolist(),token_valid=prepared.local.token_valid.sum(-1).tolist(),
                detail_valid=prepared.local.detail_valid.tolist()))
        with bind_native_full_v5(pipe.transformer,prepared,options=options,valid_mask=coverage), (observer.capture() if observer is not None else nullcontext()):
            for t in timesteps:latent=scheduler.step(predict(t),t,latent,return_dict=False)[0]
    else:
        for t in timesteps:latent=scheduler.step(predict(t),t,latent,return_dict=False)[0]
    old_device=next(pipe.vae.parameters()).device
    try:
        pipe.vae.to(device)
        unpacked=pipe._unpack_latents(latent,th*16,tw*16,pipe.vae_scale_factor).to(pipe.vae.dtype)
        mean=torch.tensor(pipe.vae.config.latents_mean,device=device,dtype=unpacked.dtype).view(1,64,1,1,1)
        std=torch.tensor(pipe.vae.config.latents_std,device=device,dtype=unpacked.dtype).view(1,64,1,1,1)
        image=pipe.vae.decode(unpacked*std+mean,return_dict=False)[0][:,:,0]
        return pipe.image_processor.postprocess(image,output_type='pil')[0]
    finally:pipe.vae.to(old_device)
