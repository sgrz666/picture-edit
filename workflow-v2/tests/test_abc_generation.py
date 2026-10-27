"""Generation must retain five real roles and never need target RGB."""
from pathlib import Path
import sys
import json
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def test_five_reference_spec_keeps_sources_independent_and_target_canvas_exact(tmp_path):
    from abc_generate_worker import condition_spec
    from src.data.canvas_contract import native_pad
    a=native_pad((64,96));b=native_pad((96,64));canvas=native_pad((64,96))
    sources=[]
    for i,spec in enumerate((a,b)):
        path=tmp_path/f'source{i}.png';Image.new('RGB',(spec.original_hw[1],spec.original_hw[0])).save(path)
        sources.append({'person_id':'source_'+str('AB'[i]),'image':str(path),'canvas_image':str(path),'canvas':spec.metadata()})
    guides={}
    for key in ('person_0','person_1','combined'):
        path=tmp_path/(key+'.png');Image.new('RGB',(96,64)).save(path);guides[key]=str(path)
    prompt=tmp_path/'prompt.txt';prompt.write_text('Use <image1> and <image2>.')
    scene={'sources':sources,'camera':{'canvas':canvas.metadata()},'guides':guides}
    request={'prompt':str(prompt),'run_dir':str(tmp_path)}
    spec=condition_spec(request,scene)
    assert [x['role'] for x in spec['images']]==['source_a','source_b','person_0_pose','person_1_pose','combined_pose']
    assert spec['appearance_source_indices']==[0,1]
    assert spec['images'][0]['canvas_id']!=spec['images'][1]['canvas_id']
    assert all(x['canvas_id']==spec['target_canvas_id'] for x in spec['images'][2:])
    assert not any('target_rgb' in x for x in spec['images'])

def test_baseline_sampler_calls_a_plain_transformer_without_native_keywords():
    import torch
    from types import SimpleNamespace
    from diffusers import FlowMatchEulerDiscreteScheduler
    from src.pose_control.qwen21_full.native_v5.generation import generate_native
    class Transformer(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1))
        def forward(self,hidden_states,timestep,encoder_hidden_states,encoder_hidden_states_mask,img_shapes,img_mask,return_dict):
            return (torch.zeros_like(hidden_states),)
    class VAE(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1))
            self.config=SimpleNamespace(latents_mean=[0.]*64,latents_std=[1.]*64)
        @property
        def dtype(self):return self.weight.dtype
        def decode(self,value,return_dict):return (torch.zeros(1,3,1,64,64),)
    refs=torch.zeros(1,5*16,64)
    pair=dict(qwen_condition_profile='dual_source_5ref',reference_roles=['source_a','source_b','person_0_pose','person_1_pose','combined_pose'],
        reference_token_hws=[(4,4)]*5,reference_latents=refs,source_latent=refs[:,:16],source_token_hw=(4,4),target_token_hw=(4,4),
        prompt_embeds=torch.zeros(1,22,8),prompt_mask=torch.ones(1,22,dtype=torch.bool),
        img_mask=torch.tensor([[False,False]+[True]*24]))
    pipe=SimpleNamespace(transformer=Transformer(),vae=VAE(),scheduler=FlowMatchEulerDiscreteScheduler(),vae_scale_factor=16,
        _pack_latents=lambda v,b,c,h,w:v[:,0].flatten(2).transpose(1,2),
        _unpack_latents=lambda v,h,w,s:v.transpose(1,2).reshape(1,64,1,4,4),
        image_processor=SimpleNamespace(postprocess=lambda v,output_type:[Image.new('RGB',(64,64))]))
    image=generate_native(pipe,None,{'valid_image_mask':torch.ones(1,1,64,64)},pair,
        options=None,base_options=None,steps=1,enabled=False)
    assert image.size==(64,64)
