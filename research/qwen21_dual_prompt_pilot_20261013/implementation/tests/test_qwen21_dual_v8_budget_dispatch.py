"""Cumulative ROI budget and full entry compatibility on real reduced modules."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from src.pose_control.qwen21_full.dual.conditions import ControlStrengths
from src.pose_control.qwen21_full.dual.controller import DualV8Adapter
from test_qwen21_dual_v8_conditions import make_input
from test_qwen21_dual_v8_context import fixture as native_fixture

torch.set_num_threads(1)


def controlled(*, geometry=.03, appearance=0., count=6):
    model = DualV8Adapter(width=32, hidden_size=64, rank=8,
                          layer_indices=(3,8,13,18,23,28)[:count])
    state = model.prepare(make_input(contact=False))
    weights = torch.zeros_like(state.weights_geometry)
    weights[:,0,0] = 1
    state = replace(state, weights_geometry=weights, weights_appearance=weights)
    with torch.no_grad():
        for heads in model.output_heads.values():
            heads['geometry'][-1].bias.fill_(geometry)
            heads['appearance'][-1].bias.fill_(appearance)
    return model, state


def test_cumulative_soft_budget_penalizes_six_layers_each_below_limit():
    model, state = controlled()
    hidden = torch.ones(1,4,64)
    for layer in model.layer_indices:
        model.residual(layer, hidden, state, .6, .6, ControlStrengths(1,0,0))
    penalty = model.budget_loss(.05)
    assert penalty.item() > 0
    penalty.backward()
    for layer in model.layer_indices:
        gradient = model.output_heads[str(layer)]['geometry'][-1].bias.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_cumulative_budget_uses_active_roi_without_background_dilution():
    model, state = controlled(count=2)
    hidden = torch.ones(1,4,64)
    for layer in model.layer_indices:
        model.residual(layer, hidden, state, .6, .6, ControlStrengths(1,0,0))
    small = model.budget_loss(.05)
    expanded = replace(state, dense=state.dense.repeat(1,1,4,1),
        weights_geometry=state.weights_geometry.repeat(1,1,4,1),
        weights_appearance=state.weights_appearance.repeat(1,1,4,1),
        weights_contact=state.weights_contact.repeat(1,4,1), target_hw=(4,4))
    # Keep only one token active; the other fifteen must not divide the penalty.
    weights = torch.zeros_like(expanded.weights_geometry)
    weights[:,0,0] = 1
    expanded = replace(expanded, weights_geometry=weights, weights_appearance=weights)
    for layer in model.layer_indices:
        model.residual(layer, torch.ones(1,16,64), expanded, .6, .6, ControlStrengths(1,0,0))
    torch.testing.assert_close(model.budget_loss(.05),small)


def test_disabled_appearance_roi_does_not_dilute_geometry_budget():
    model,state = controlled(count=2)
    hidden = torch.ones(1,4,64)
    for layer in model.layer_indices:
        model.residual(layer,hidden,state,.6,.6,ControlStrengths(1,0,0))
    expected = model.budget_loss(.05)
    weights = torch.zeros_like(state.weights_appearance)
    weights[:,0] = 1
    wider = replace(state,weights_appearance=weights)
    for layer in model.layer_indices:
        model.residual(layer,hidden,wider,.6,.6,ControlStrengths(1,0,0))
    torch.testing.assert_close(model.budget_loss(.05),expected)


def test_optional_cap_limits_total_injection_against_first_hidden():
    model, state = controlled(geometry=.2)
    total = torch.zeros(1,4,64)
    first = torch.ones_like(total)
    for index,layer in enumerate(model.layer_indices):
        hidden = first * (index+1)
        delta = model.residual(layer, hidden, state, .6, .6,
                               ControlStrengths(1,0,0,max_ratio=.05))
        total += delta
        assert total[:,0].square().mean().sqrt() <= .050001
        assert delta[:,0].square().mean().sqrt() <= .05*hidden[:,0].square().mean().sqrt()+1e-6
    assert model.last_diagnostics[28]['total']['cumulative_ratio'] <= .050001


def test_each_branch_cumulative_statistics_do_not_include_other_branches():
    model, state = controlled(geometry=.03,appearance=.01,count=2)
    hidden = torch.ones(1,4,64)
    for layer in model.layer_indices:
        model.residual(layer, hidden, state, .6, .6, ControlStrengths(1,1,0))
    reports = model.last_diagnostics[8]
    assert reports['geometry']['cumulative'] == pytest.approx(.06)
    assert reports['appearance']['cumulative'] == pytest.approx(.02)
    assert reports['contact']['cumulative'] == 0
    assert reports['total']['cumulative'] == pytest.approx(.08)


@pytest.mark.parametrize('max_ratio',[None,.05])
def test_zero_initialized_cumulative_budget_keeps_finite_flow_gradients(max_ratio):
    model, state = controlled(geometry=0.)
    deltas = [model.residual(layer,torch.ones(1,4,64),state,.6,.6,
                            ControlStrengths(max_ratio=max_ratio))
              for layer in model.layer_indices]
    loss = (sum(deltas)-1).square().mean()+model.budget_loss(.05)
    loss.backward()
    for layer in model.layer_indices:
        gradient = model.output_heads[str(layer)]['geometry'][-1].weight.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def legacy_fixture():
    from src.pose_control.qwen21_full.conditions import ConditionBundle
    from src.pose_control.qwen21_full.control_core import SharedRecurrentControlCore
    from src.pose_control.qwen21_full.deepgen_adapter import UnifiedSMPLXAdapterV6
    from src.pose_control.qwen21_full.qwen_adapter import Qwen21FullAdapter
    from src.pose_control.qwen21_full.reasoning import AdapterReasoningConfig

    torch.manual_seed(308)
    core = SharedRecurrentControlCore.build_tiny(hidden_dim=32,condition_channels=48,
        context_input_dim=32,pooled_input_dim=20,rank=8)
    legacy = UnifiedSMPLXAdapterV6(core,geometry_channels=32,cross_attention_dim=32,
        reasoning_config=AdapterReasoningConfig(hidden_dim=32))
    model = Qwen21FullAdapter(legacy,qwen_hidden_size=64,qwen_text_dim=32,pooled_input_dim=20).eval()
    with torch.no_grad():
        for head in core.geometry_zero_heads:
            head.weight.normal_(std=.01)
    bundle = ConditionBundle(person_a_spatial=torch.randn(1,256,4,4),
        person_b_spatial=None,person_a_mask=torch.ones(1,1,4,4),person_b_mask=None,
        person_a_global_tokens=torch.randn(1,4,256),person_b_global_tokens=None,
        relative_tokens=None,contact_spatial=None,contact_tokens=None,contact_mask=None,
        task_token=torch.randn(1,1,256),person_valid=torch.tensor([[True,False]]),
        person_count=torch.tensor([1])).validate()
    prepared = model.prepare_conditioning(condition_bundle=bundle,
        source_latent=torch.randn(1,4,64),target_token_hw=(2,2))
    return model,prepared


@pytest.mark.parametrize('mode', ['single_v6','dual_v8'])
def test_full_unified_entry_preserves_real_legacy_context_and_state_roundtrip(tmp_path,mode):
    from src.pose_control.qwen21_full import qwen_adapter

    transformer,inputs,_ = native_fixture()
    single,prepared = legacy_fixture()
    state_before = {name:value.clone() for name,value in single.state_dict().items()}
    path = tmp_path/'representative_single_full_state.pt'
    torch.save({'adapter_state_dict':state_before},path)
    loaded,_ = legacy_fixture()
    loaded.load_state_dict(torch.load(path,weights_only=True)['adapter_state_dict'],strict=True)
    with torch.no_grad():
        with qwen_adapter.Qwen21FullControlContext(transformer,single,prepared,
                                                   geometry_strength=.4) as original:
            expected = transformer(**inputs)[0]
        assert hasattr(qwen_adapter,'create_control_context'), 'full unified entry is missing'
        # The original prepared condition is explicitly supplied. Dual global32
        # may differ freely and cannot silently replace the old global protocol.
        generation = make_input(with_b=False,contact=False)
        with qwen_adapter.create_control_context(transformer,loaded,prepared,mode=mode,
                generation=generation,geometry_strength=.4) as dispatched:
            actual = transformer(**inputs)[0]
    assert isinstance(dispatched,qwen_adapter.Qwen21FullControlContext)
    assert dispatched.prepared_or_factory is prepared
    assert original.call_counts == dispatched.call_counts == {i:1 for i in range(32)}
    assert torch.equal(actual,expected)
    assert not torch.equal(expected,transformer(**inputs)[0])
    assert all(torch.equal(value,loaded.state_dict()[name]) for name,value in state_before.items())


def test_unified_entry_routes_typed_dual_to_six_native_layers():
    from src.pose_control.qwen21_full import qwen_adapter
    from src.pose_control.qwen21_full.dual.context import DualV8ControlContext

    transformer,inputs,_ = native_fixture()
    model,state = controlled()
    assert hasattr(qwen_adapter,'create_control_context'), 'full unified entry is missing'
    with torch.no_grad(), qwen_adapter.create_control_context(transformer,mode='dual_v8',
            generation=make_input(contact=False),dual_adapter=model,dual_prepared=state,
            sigma=.6) as context:
        transformer(**inputs)
    assert isinstance(context,DualV8ControlContext)
    assert context.call_counts == {layer:1 for layer in model.layer_indices}


def test_runtime_missing_b_uses_explicit_original_context():
    from src.pose_control.qwen21_full.dual.runtime import generate
    from src.pose_control.qwen21_full.qwen_adapter import Qwen21FullControlContext

    transformer,inputs,_ = native_fixture()
    single,prepared = legacy_fixture()
    class ReducedPipeline:
        def __init__(self): self.transformer = transformer
        def __call__(self,**kwargs):
            return SimpleNamespace(images=[self.transformer(**inputs)[0]])
    class UnusedDualAdapter:
        def prepare(self,generation):
            raise AssertionError('missing B must not prepare a dual condition')
    generation = make_input(with_b=False,contact=False)
    with torch.no_grad(), Qwen21FullControlContext(transformer,single,prepared,geometry_strength=.4):
        expected = transformer(**inputs)[0]
    original = Qwen21FullControlContext(transformer,single,prepared,geometry_strength=.4)
    actual = generate(ReducedPipeline(),generation,adapter=UnusedDualAdapter(),
        strengths=ControlStrengths(0,0,0),single_context=original)
    assert torch.equal(actual,expected)
    assert original.call_counts == {layer:1 for layer in range(32)}


def test_cumulative_budget_checkpoint_backward_matches_without_recomputation():
    from src.pose_control.qwen21_full.dual.context import DualV8ControlContext

    transformer,inputs,_ = native_fixture()
    model,state = controlled()
    # A fresh prepared graph for each context; freeze static test features.
    state = replace(state,dense=state.dense.detach(),nodes=state.nodes.detach(),
        appearance=state.appearance.detach(),contacts=state.contacts.detach())
    def backward():
        with DualV8ControlContext(transformer,model,state,sigma=.6):
            result = transformer(**inputs)[0]
            (result.square().mean()+model.budget_loss(.0001)).backward()
        return {name:p.grad.clone() for name,p in model.named_parameters() if p.grad is not None}
    expected = backward()
    model.zero_grad(set_to_none=True)
    transformer.enable_gradient_checkpointing()
    actual = backward()
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name],expected[name],atol=1e-6,rtol=1e-5,msg=name)
