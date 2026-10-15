#!/usr/bin/env python3
"""Real Face loss gradient and source/geometry boundaries; no optimizer update."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from diffusers import QwenImage21Pipeline
from scripts.run_qwen21_face_single_sample import OUT,BODY,FACE
from scripts.run_qwen21_weight_inheritance import source_pair,verify_gpu
from src.pose_control.qwen21_full.execution import (
    MODEL,RUN_ROOT,QwenFullDataset,FaceConditionStore,Qwen21IPERCache,new_adapter,
    load_adapter_checkpoint,json_write,set_training_phase,run_loss,frozen_digest,
    batch_one,prepare,transformer_inputs,dataset_image_to_pil)
from src.pose_control.qwen21_full.qwen_adapter import Qwen21FullControlContext
from src.pose_control.qwen21_full.face_supervision import FaceSupervisionStore

def main():
    torch.set_num_threads(4)
    dataset=QwenFullDataset(OUT/'single_pair.jsonl',resolution=512)
    store=FaceConditionStore(RUN_ROOT/'face_cache');cache=Qwen21IPERCache(RUN_ROOT/'native_cache')
    pipe=QwenImage21Pipeline.from_pretrained(str(MODEL),dtype=torch.bfloat16).to('cuda')
    for model in (pipe.transformer,pipe.text_encoder,pipe.vae):model.eval().requires_grad_(False)
    adapter=new_adapter('cuda').eval().requires_grad_(False)
    load_adapter_checkpoint(adapter,FACE)
    runtime=verify_gpu(pipe,adapter,cache,dataset,store,'current_face100')
    # Check the exact A/B arrays used by preprocessing, not only tensor shapes.
    meta=dataset[0]['pair_metadata']; fc,refs=store.for_pairs([meta],device='cpu',dtype=torch.float32)
    a=store._load(meta['source_appearance']);b=store._load(meta['target_appearance'])
    ai=a['stem_to_idx'][meta['source_stem']];bi=b['stem_to_idx'][meta['target_stem']]
    provenance={'identity_from_A':torch.equal(refs.arcface[0,0,0],a['arcface'][ai]),
                'dino_from_A':torch.equal(refs.dino_patches[0,0,0],a['dino_patches'][ai]),
                'landmarks_from_B':torch.equal(fc.landmarks[0,0],b['face_landmarks'][bi]),
                'target_box_from_B':torch.equal(fc.target_boxes[0,0],b['face_boxes'][bi])}
    if not all(provenance.values()):raise AssertionError(provenance)
    adapter.train();set_training_phase(adapter,'face',warmup=False)
    before=frozen_digest(adapter,'face')
    pipe.text_encoder.to('cpu');pipe.transformer.enable_gradient_checkpointing()
    evaluators=tuple(torch.jit.load(str(ROOT/'models/face_eval'/n),map_location='cuda').eval()
                     for n in ('identity_encoder.ts','perceptual_encoder.ts'))
    for model in evaluators:
        for parameter in model.parameters():parameter.requires_grad_(False)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        gradient=run_loss(pipe,adapter,cache,dataset,store,0,'face',seed=41042,
                          auxiliary=True,evaluators=evaluators,supervision=FaceSupervisionStore(RUN_ROOT/'face_supervision.json'),
                          fixed_sigma=.15,check_auxiliary_gradients=True,normalized_face_evaluators=True)
    if not all(value>0 for value in gradient['auxiliary_gradient_norms'].values()):raise AssertionError(gradient)
    if any(p.grad is not None for p in pipe.transformer.parameters()):raise AssertionError('Base has parameter gradients')
    if any(p.grad is not None for p in adapter.parameters() if not p.requires_grad):raise AssertionError('Frozen branch has gradients')
    if before!=frozen_digest(adapter,'face'):raise AssertionError('No-update check mutated frozen parameters')
    adapter.zero_grad(set_to_none=True);adapter.eval().requires_grad_(False)
    load_adapter_checkpoint(adapter,BODY)
    if torch.count_nonzero(adapter.output_bridges['face'].weight):raise AssertionError('Expected initial zero Face output bridge')
    batch=batch_one(dataset,0);pair=cache.pair(pipe,batch,torch.device('cuda'))
    sigma=pair['target_latent'].new_tensor([.2]);inputs=transformer_inputs(pair,pair['target_latent']*.8,sigma)
    prepared=prepare(adapter,batch,pair,[dataset.samples[0]],store,with_face=True)
    values=[]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for strength in (0,1):
            with Qwen21FullControlContext(pipe.transformer,adapter,prepared,face_strength=strength,hand_strength=0,hand_mode='off'):
                values.append(pipe.transformer(**inputs)[0].clone())
    zero_parity={'equal':torch.equal(*values),'max_abs':float((values[0].float()-values[1].float()).abs().max())}
    if not zero_parity['equal']:raise AssertionError(zero_parity)
    latent=pair['source_latent'];grid=32
    # Evaluate the lossless spatial rearrangement at identity initialization separately from the trained bridge.
    value=latent.transpose(1,2).reshape(1,64,grid,grid)
    rearranged=torch.nn.functional.pixel_shuffle(value,2)
    restored=torch.nn.functional.pixel_unshuffle(rearranged,2).flatten(2).transpose(1,2)
    if not torch.equal(latent,restored):raise AssertionError('Spatial rearrangement failed')
    json_write(OUT/'correctness.json',{'passed':True,'runtime':runtime,'source_geometry_provenance':provenance,
               'auxiliary_gradient_check':gradient,'initial_face_vs_body':zero_parity,
               'lossless_spatial_rearrangement':True,'optimizer_updates':0,
               'base_frozen':True,'only_face_gradients':True})
    print('correctness checks passed',flush=True)

if __name__=='__main__':main()
