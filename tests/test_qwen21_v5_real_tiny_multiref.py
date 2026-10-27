"""Real native Qwen blocks, bounded CPU grids; no optimizer or model weights."""
from dataclasses import asdict
import pytest
import torch
from torch import nn


@pytest.mark.parametrize('references',[2,4,5])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_real_native_qwen_frozen_prefix_and_checkpoint_gradient(references,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel
    from src.pose_control.qwen21_full.native_body import NativeBodyConfig,Qwen21NativeBody
    from src.pose_control.qwen21_full.native_geometry import NativeGeometryFeatures
    from src.pose_control.qwen21_full.native_local_condition import PreparedNativeFullControlV4
    from src.pose_control.qwen21_full.native_v5.contracts import NativeConfigV5,NativeControlOptionsV5,PreparedNativeFullControlV5
    from src.pose_control.qwen21_full.native_v5.runtime import NativeFullRuntimeV5
    from src.pose_control.qwen21_full.native_v5.strength import NativeStrengthController
    from src.pose_control.qwen21_full.native_v5.integration import install_native_full_v5
    from src.pose_control.qwen21_full.native_v5.training.validation import prefix_trace
    from types import SimpleNamespace
    torch.manual_seed(123)
    model=QwenImage21Transformer2DModel(in_channels=64,out_channels=64,num_layers=32,num_attention_heads=2,
        attention_head_dim=8,context_in_dim=8,axes_dims_rope=(2,2,4),mlp_ratio=2).eval()
    model=model.to(device=device,dtype=torch.bfloat16 if device=='cuda' else torch.float32)
    # Only the Body stream is exercised here; other streams have separate CPU
    # tests and must pass the full-width, real-asset acceptance before training.
    runtime=NativeFullRuntimeV5.__new__(NativeFullRuntimeV5);nn.Module.__init__(runtime)
    runtime.base=nn.Module();runtime.base.body=Qwen21NativeBody(NativeBodyConfig(hidden_dim=16))
    runtime.config=NativeConfigV5(verification_only=True,hidden_dim=16)
    runtime.face_controller=nn.Identity();runtime.hand_controller=nn.Identity();runtime.interaction_head=nn.Identity()
    runtime.strength=NativeStrengthController(runtime.config)
    runtime.to(device)
    original={n:p for n,p in model.named_parameters()}
    install_native_full_v5(model,runtime);model.enable_gradient_checkpointing()
    options=NativeControlOptionsV5(face=False,hand=False,detail=False,interaction=False)
    geometry=torch.randn(1,8,512,requires_grad=True)
    features=NativeGeometryFeatures(geometry,torch.zeros_like(geometry),(2,4),torch.tensor([[True,False]]))
    control=PreparedNativeFullControlV5(PreparedNativeFullControlV4(features,None,({},)),
        torch.zeros_like(geometry),torch.zeros(1,8,dtype=torch.bool),None,({'options':asdict(options)},))
    mask=torch.tensor([[False,False]+[True]*(references+2)])
    values=dict(hidden_states=torch.randn(1,references*4+8,64),encoder_hidden_states=torch.randn(1,references+2,8),
        encoder_hidden_states_mask=torch.ones(1,references+2,dtype=torch.bool),timestep=torch.tensor([.5]),
        img_shapes=[[(1,2,2)]*references+[(1,2,4)]],img_mask=mask,return_dict=False)
    values={k:v.to(device=device,dtype=torch.bfloat16 if device=='cuda' and v.is_floating_point() else v.dtype) if torch.is_tensor(v) else v for k,v in values.items()}
    control=control.to(device)
    prefix=values['hidden_states'][:,:references*4]
    profiles={2:('single_2ref',['source','target_pose']),
        4:('dual_4ref',['source','combined_pose','person_0_pose','person_1_pose']),
        5:('dual_source_5ref',['source_a','source_b','person_0_pose','person_1_pose','combined_pose'])}
    pair={'qwen_condition_profile':profiles[references][0],
        'reference_latents':prefix,'reference_token_hws':[(2,2)]*references,
        'reference_roles':profiles[references][1],
        'source_latent':prefix[:,:4],'source_token_hw':(2,2),'target_token_hw':(2,4),
        'target_latent':values['hidden_states'][:,-8:],'prompt_embeds':values['encoder_hidden_states'],
        'prompt_mask':values['encoder_hidden_states_mask'],'img_mask':values['img_mask']}
    backend=SimpleNamespace(pipe=SimpleNamespace(transformer=model))
    with torch.no_grad(),torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'),prefix_trace(backend,pair,pair['target_latent'],values['timestep']) as (before,_):
        baseline=model(**values,native_enabled=False)[0]
    with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'),prefix_trace(backend,pair,pair['target_latent'],values['timestep']) as (after,_):
        output=model(**values,native_full_control=control,native_full_options=options,native_valid_mask=torch.ones(1,8,device=device))[0]
    assert torch.equal(output,baseline)
    assert before==after
    output[:,-8:].square().mean().backward()
    assert any(h.weight.grad is not None and h.weight.grad.count_nonzero() for h in runtime.body.output_heads)
    assert all(not p.requires_grad and p.grad is None for p in original.values())
    runtime.zero_grad(set_to_none=True)
    with torch.no_grad():
        for head in runtime.body.output_heads:head.weight.normal_(0,.001)
    with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):
        probe=model(**values,native_full_control=control,native_full_options=options,native_valid_mask=torch.ones(1,8,device=device))[0]
    probe[:,-8:].square().mean().backward()
    assert geometry.grad is not None and geometry.grad.count_nonzero()
    assert runtime.body.query_projection.weight.grad.count_nonzero()
    assert all(p.grad is None for p in original.values())
