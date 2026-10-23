#!/usr/bin/env python3
"""Post-training fixed-noise, regional-error, and one-step reconstruction diagnosis."""

# Project root for the categorized entrypoint.
from pathlib import Path as _EntryPath
import sys as _entry_sys
_entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))

from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from diffusers import QwenImage21Pipeline
from torchvision.transforms.functional import to_pil_image
from scripts.run_qwen21_face_single_sample import OUT
from src.pose_control.qwen21_full.execution import (
    MODEL,RUN_ROOT,QwenFullDataset,FaceConditionStore,Qwen21IPERCache,new_adapter,
    load_adapter_checkpoint,json_write,run_loss,batch_one,prepare,transformer_inputs,decode_tokens)
from src.pose_control.qwen21_full.qwen_adapter import Qwen21FullControlContext

@torch.no_grad()
def main():
    torch.set_num_threads(4)
    dataset=QwenFullDataset(OUT/'single_pair.jsonl',512, align_mode='resize', legacy_policy_ok=True);cache=Qwen21IPERCache(RUN_ROOT/'native_cache')
    store=FaceConditionStore(RUN_ROOT/'face_cache')
    pipe=QwenImage21Pipeline.from_pretrained(str(MODEL),dtype=torch.bfloat16)
    for module in (pipe.transformer,pipe.text_encoder,pipe.vae):module.eval().requires_grad_(False)
    pipe.transformer.to('cuda');pipe.vae.to('cuda')
    adapter=new_adapter('cuda').eval().requires_grad_(False)
    sigmas=(.05,.10,.15,.20,.30,.40,.50,.60);all_rows=[]
    destination=OUT/'teacher_forced';destination.mkdir(exist_ok=True)
    for step in (0,100,300,500,1000,2000):
        checkpoint=OUT/'face_train'/f'checkpoint_step_{step:06d}.pt'
        if not checkpoint.exists():continue
        load_adapter_checkpoint(adapter,checkpoint);rows=[]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for i,sigma in enumerate(sigmas):
                rows.append(run_loss(pipe,adapter,cache,dataset,store,0,'face',seed=41042+i,
                                     backward=False,fixed_sigma=sigma,normalized_face_evaluators=True))
            batch=batch_one(dataset,0);pair=cache.pair(pipe,batch,torch.device('cuda'))
            target=pair['target_latent'];generator=torch.Generator(device='cuda').manual_seed(41045)
            torch.rand((1,),device='cuda',dtype=target.dtype,generator=generator) # same first draw as run_loss
            noise=torch.randn(target.shape,device='cuda',dtype=target.dtype,generator=generator)
            sigma=target.new_tensor([.2]);noisy=(1-sigma[:,None,None])*target+sigma[:,None,None]*noise
            prepared=prepare(adapter,batch,pair,[dataset.samples[0]],store,with_face=True)
            with Qwen21FullControlContext(pipe.transformer,adapter,prepared,face_strength=1,hand_strength=0,hand_mode='off'):
                prediction=pipe.transformer(**transformer_inputs(pair,noisy,sigma))[0][:,-1024:]
                pixels=decode_tokens(pipe,noisy-sigma[:,None,None]*prediction,512)
            path=destination/f'step_{step:06d}_sigma02.png';to_pil_image(pixels[0].cpu()).save(path)
        all_rows.append({'added_step':step,'fixed_noise_rows':rows,'one_step_reconstruction':str(path),
                         'Target_RGB_used_to_construct_noisy_supervised_latent':True,
                         'not_a_real_generation_result':True})
    json_write(OUT/'regional_probe.json',all_rows)
    print('regional and teacher-forced probes recorded')

if __name__=='__main__':main()
