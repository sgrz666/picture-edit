"""Native 32-block Qwen runtime boundaries, using a small real transformer."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from diffusers import QwenImage21Transformer2DModel
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

torch.set_num_threads(1)


def context_type():
    try:
        from src.pose_control.qwen21_full.dual.context import DualV8ControlContext
    except ModuleNotFoundError as exc:
        if exc.name == 'src.pose_control.qwen21_full.dual.context':
            pytest.fail('native DualV8 context is not implemented')
        raise
    return DualV8ControlContext


def fixture():
    torch.manual_seed(191)
    model = QwenImage21Transformer2DModel(num_layers=32, attention_head_dim=16,
        num_attention_heads=4, context_in_dim=32, mlp_ratio=2, axes_dims_rope=(4,6,6))
    model.requires_grad_(False)
    inputs = dict(hidden_states=torch.randn(1,8,64), encoder_hidden_states=torch.randn(1,3,32),
        timestep=torch.tensor([.7]), img_shapes=[[(1,2,2),(1,2,2)]],
        img_mask=torch.tensor([[False,True,False,True]]),
        encoder_hidden_states_mask=torch.ones(1,3,dtype=torch.bool), return_dict=False)
    state = SimpleNamespace(target_hw=(2,2), dense=torch.zeros(1,2,4,32))
    return model, inputs, state


class ProbeController(nn.Module):
    """Independent controller for measuring context behavior, not adapter quality."""
    hidden_size = 64
    layer_indices = (3,8,13,18,23,28)

    def __init__(self, zero=False):
        super().__init__()
        self.proj = nn.Linear(64,64,bias=False)
        nn.init.zeros_(self.proj.weight) if zero else nn.init.normal_(self.proj.weight,std=.001)
        self.inputs = []

    def residual(self, layer_index, target_hidden, prepared, timestep, sigma, strengths):
        self.inputs.append((layer_index, target_hidden.shape, timestep.detach().clone(), sigma.detach().clone()))
        return self.proj(target_hidden) * strengths.geometry


def test_native_zero_init_exact_and_final_head_receives_gradient():
    Context = context_type()
    model, inputs, state = fixture()
    adapter = ProbeController(zero=True)
    native = model(**inputs)[0]
    with Context(model,adapter,state,sigma=.6) as ctx:
        output = model(**inputs)[0]
        assert torch.equal(output,native)
        output[:,-4:].square().mean().backward()
        assert adapter.proj.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())
    assert ctx.call_counts == {k:1 for k in adapter.layer_indices}


def test_nonzero_control_leaves_every_native_prefix_token_exact():
    Context = context_type()
    model,inputs,state = fixture()
    adapter = ProbeController()
    with torch.no_grad():
        native = model(**inputs)[0]
        with Context(model,adapter,state,sigma=.6):
            result = model(**inputs)[0]
    assert torch.equal(result[:,:6], native[:,:6])
    assert not torch.equal(result[:,-4:], native[:,-4:])
    assert all(shape == (1,4,64) for _,shape,_,_ in adapter.inputs)
    assert all(torch.equal(t,inputs['timestep']) and torch.allclose(s,torch.tensor([.6])) for _,_,t,s in adapter.inputs)


def test_optional_residual_maps_measure_actual_target_delta_only():
    Context=context_type()
    model,inputs,state=fixture()
    adapter=ProbeController()
    adapter.capture_residual_maps=True
    expected={}
    original=adapter.residual
    def record(layer,*args,**kwargs):
        delta=original(layer,*args,**kwargs)
        expected[layer]=delta.float().square().mean(-1).sqrt().reshape(2,2).detach()
        return delta
    adapter.residual=record
    with torch.no_grad(),Context(model,adapter,state,sigma=.6):
        model(**inputs)
    assert set(adapter.last_residual_maps)==set(adapter.layer_indices)
    for layer,item in adapter.last_residual_maps.items():
        assert item['rms'].device.type=='cpu' and item['target_hw']==(2,2)
        torch.testing.assert_close(item['rms'][0],expected[layer],rtol=0,atol=0)
        assert item['sigma']==pytest.approx(.6)


def test_disabled_installs_no_hooks_and_is_exact():
    Context = context_type()
    from src.pose_control.qwen21_full.dual.conditions import ControlStrengths
    model,inputs,state = fixture()
    before = [len(block._forward_hooks) for block in model.transformer_blocks]
    with torch.no_grad():
        native = model(**inputs)[0]
        with Context(model,ProbeController(),state,sigma=.6,strengths=ControlStrengths(0,0,0)) as ctx:
            assert not ctx.handles
            assert before == [len(b._forward_hooks) for b in model.transformer_blocks]
            assert torch.equal(native,model(**inputs)[0])


def test_native_cached_equals_uncached_and_sees_only_target():
    Context = context_type()
    model,inputs,state = fixture()
    adapter = ProbeController()
    cache = QwenImage21KVCache(num_layers=32)
    with torch.no_grad(), Context(model,adapter,state,sigma=lambda kw: kw['timestep']*.9):
        extract = model(**inputs,kv_cache=cache,kv_cache_mode='extract')[0]
        cached = model(**inputs,kv_cache=cache,kv_cache_mode='cached')[0]
        full = model(**inputs)[0]
    assert cached.shape == (1,4,64)
    torch.testing.assert_close(cached,full[:,-4:],atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(extract,full,atol=1e-6,rtol=1e-5)


def test_nested_context_and_multiple_live_training_forwards_rejected():
    Context = context_type()
    model,inputs,state = fixture()
    adapter = ProbeController()
    with Context(model,adapter,state,sigma=.6):
        with pytest.raises(RuntimeError,match='active'):
            with Context(model,adapter,state,sigma=.6): pass
        output = model(**inputs)[0]
        with pytest.raises(RuntimeError,match='one training forward'):
            model(**inputs)
        output.square().mean().backward()
    assert not getattr(model,'_pic_edit_qwen21_control_active',False)


def test_checkpoint_recomputation_matches_native_gradient_and_requires_live_context():
    Context = context_type()
    model,inputs,state = fixture()
    adapter = ProbeController()
    with Context(model,adapter,state,sigma=.6):
        model(**inputs)[0].square().mean().backward()
    expected = adapter.proj.weight.grad.clone()
    adapter.zero_grad(set_to_none=True)
    model.enable_gradient_checkpointing()
    with Context(model,adapter,state,sigma=.6):
        model(**inputs)[0].square().mean().backward()
    torch.testing.assert_close(adapter.proj.weight.grad,expected,atol=1e-6,rtol=1e-5)
    with Context(model,adapter,state,sigma=.6):
        pending = model(**inputs)[0]
    with pytest.raises(RuntimeError,match='backward'):
        pending.square().mean().backward()


def test_reject_inconsistent_target_image_slots_before_transformer_execution():
    Context = context_type()
    model,inputs,state = fixture()
    inputs['img_mask'] = torch.tensor([[False,True,True,False]])
    with Context(model,ProbeController(),state,sigma=.6):
        with pytest.raises(ValueError,match='target.*suffix'):
            model(**inputs)
