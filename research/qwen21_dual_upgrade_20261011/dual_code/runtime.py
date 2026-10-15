"""Native pipeline generation and flow objectives for the one-pair pilot."""
from __future__ import annotations

from contextlib import nullcontext
import hashlib

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

from .conditions import ControlStrengths,DualGenerationInput
from .diagnostics import GenerationDiagnostics
from .precision import precision_policy,qwen21_precision
from ..qwen_adapter import create_control_context
from ..qwen_runtime import Qwen21IPERCache,transformer_inputs


DETAILED_HUG_PROMPT = ('Edit the supplied photo of exactly two existing people. The person on the left and '
    'the person on the right move toward each other and hug. Their torsos touch and their arms wrap around '
    "each other's shoulders and backs. Preserve each person's face, hair, clothing and proportions, as well "
    'as the camera view, lighting and background. Do not add another person or change the setting.')


def load_native(model_path,device='cuda'):
    from diffusers import QwenImage21Pipeline
    pipe=QwenImage21Pipeline.from_pretrained(str(model_path),dtype=torch.bfloat16,local_files_only=True).to(device)
    for component in (pipe.transformer,pipe.text_encoder,pipe.vae):component.eval().requires_grad_(False)
    return pipe


def tensor_model_sha256(module):
    digest=hashlib.sha256()
    for name,value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def flow_sample(clean,noise,sigma):
    if clean.shape!=noise.shape or clean.ndim!=3:raise ValueError('clean/noise must share [B,N,C] shape')
    sigma=torch.as_tensor(sigma,device=clean.device,dtype=torch.float32).reshape(-1)
    if sigma.numel()==1:sigma=sigma.expand(clean.shape[0])
    if sigma.shape!=(clean.shape[0],) or not bool(torch.isfinite(sigma).all()) or bool(((sigma<0)|(sigma>1)).any()):
        raise ValueError('sigma must be in [0,1] with one value per sample')
    mixed=(1-sigma[:,None,None])*clean.float()+sigma[:,None,None]*noise.float()
    return mixed.to(clean.dtype),noise.float()-clean.float()


def weighted_flow_loss(prediction,goal,weights):
    if prediction.shape!=goal.shape or weights.shape!=(*goal.shape[:2],1):raise ValueError('flow loss shapes disagree')
    return ((prediction.float()-goal.float()).square()*weights.float()).sum()/(weights.float().sum().clamp_min(1e-8)*goal.shape[-1])


def background_teacher_mask(source_masks,target_amodal,target_hw):
    if source_masks.shape!=target_amodal.shape:raise ValueError('source/target masks must share the same letterbox grid')
    union=torch.maximum(source_masks.amax(1),target_amodal.amax(1))
    excluded=F.max_pool2d(union,17,1,8)
    return F.adaptive_avg_pool2d(1-excluded,target_hw).flatten(2).transpose(1,2)


def source_cache(pipe,generation,cache_dir):
    if not isinstance(generation,DualGenerationInput):raise ValueError('source cache requires typed generation input')
    if generation.source.rgb.shape[0]!=1:raise ValueError('pilot cache supports one sample')
    cache=Qwen21IPERCache(cache_dir,prompt=generation.prompts[0],preprocessing='affine_letterbox_v8_bf16_full_reduction_v1')
    image=to_pil_image(generation.source.rgb[0].detach().float().cpu())
    return cache,cache.source(pipe,image,prompt=generation.prompts[0])


def native_inputs(source,noisy,sigma):
    device=noisy.device
    target_hw=(32,32)
    target_slots=torch.ones(noisy.shape[0],1024//4,device=device,dtype=torch.bool)
    pair=dict(source_latent=source['reference_latent'].to(device=device,dtype=noisy.dtype),
        prompt_embeds=source['prompt_embeds'].to(device=device,dtype=noisy.dtype),
        prompt_mask=source['prompt_mask'].to(device),
        img_mask=torch.cat((source['image_mask'].to(device),target_slots),dim=1),
        source_token_hw=tuple(source['token_hw']),target_token_hw=target_hw)
    return transformer_inputs(pair,noisy,sigma)


def prepared_for_qwen(adapter,generation):
    with torch.autocast('cuda',dtype=torch.bfloat16):
        return adapter.prepare(generation).to(generation.source.rgb.device,torch.bfloat16)


@torch.no_grad()
@qwen21_precision()
def generate(pipe,generation,adapter=None,strengths=None,steps=40,cfg=1.,use_kv_cache=True,
             *,single_context=None):
    """Public generation boundary: source, external geometry, text, seed only."""
    generation.validate()
    pipe.last_dual_v8_precision_policy=precision_policy()
    if generation.source.rgb.shape[0]!=1:raise ValueError('pilot inference supports exactly one frame')
    device=next(pipe.transformer.parameters()).device
    image=to_pil_image(generation.source.rgb[0].detach().float().cpu())
    strengths=ControlStrengths() if strengths is None else strengths
    if hasattr(pipe,'last_dual_v8_diagnostics'):
        pipe.last_dual_v8_diagnostics=None
    if adapter is not None and bool(generation.geometry.person_valid.all()):
        adapter.last_diagnostics={}
        adapter.last_residual_maps={}
        adapter.last_inference_diagnostics=None
    context=nullcontext()
    if not bool(generation.geometry.person_valid.all()):
        context=create_control_context(pipe.transformer,mode='dual_v8',generation=generation,
            single_context=single_context)
    elif adapter is not None and not strengths.disabled:
        typed=generation.to(device,torch.float32)
        prepared=prepared_for_qwen(adapter,typed)
        def actual_sigma(inputs):
            index=pipe.scheduler.step_index
            if index is None:
                # set_begin_index(0) is called by the native pipeline before step0.
                index=pipe.scheduler.begin_index or 0
            return pipe.scheduler.sigmas[index]
        context=create_control_context(pipe.transformer,mode='dual_v8',generation=typed,
            dual_adapter=adapter,dual_prepared=prepared,sigma=actual_sigma,dual_strengths=strengths)
    observer=GenerationDiagnostics(pipe,adapter,cfg=cfg,target_tokens=1024) if adapter is not None and bool(generation.geometry.person_valid.all()) else None
    autocast=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()
    with context,(observer if observer is not None else nullcontext()),autocast:
        result=pipe(prompt=generation.prompts[0],image=image,height=512,width=512,output_resolution=512,
            num_inference_steps=steps,true_cfg_scale=cfg,negative_prompt='' if cfg>1 else None,
            generator=torch.Generator(device=device).manual_seed(generation.seeds[0]),use_kv_cache=use_kv_cache).images[0]
    if observer is not None:
        pipe.last_dual_v8_diagnostics=observer.summary
        adapter.last_inference_diagnostics=observer.summary
        adapter.last_inference_diagnostics['control_disabled']=strengths.disabled
    return result
