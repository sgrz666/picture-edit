"""零输出启动、梯度、prefix 与浮点存储的 CPU 回归。"""
from dataclasses import replace

import pytest
import torch
from torch import nn

from test_qwen21_pro_contracts import api, make_condition, without_people


def make_adapter():
    torch.manual_seed(7)
    return api().ProTaskAdapter(api().ProAdapterConfig(hidden_dim=32, width=16, heads=4))


def test_zero_heads_preserve_joint_and_start_head_gradient():
    model, pack = make_adapter(), make_condition()
    x = torch.randn(1, 7, 32, requires_grad=True)
    target = torch.tensor([False, True, False, True, True, False, True])
    out = model.forward_layer(0, x, target, pack, torch.tensor([.5]))
    assert torch.equal(out, x)
    out.square().mean().backward()
    assert model.output_heads["0"].weight.grad.abs().sum() > 0
    assert set(model.output_heads) == {"0", "4", "8", "12"}
    assert all(head.weight.count_nonzero() == 0 for head in model.output_heads.values())
    assert torch.equal(model.forward_layer(1, x, target, pack, torch.tensor([.5])), x)


def test_one_head_update_opens_geometry_and_bank_gradients_without_prefix_write():
    model, pack = make_adapter(), make_condition()
    x = torch.randn(1, 7, 32, requires_grad=True)
    target = torch.tensor([False, False, False, True, True, True, True])
    optimizer = torch.optim.SGD(model.parameters(), lr=.5)
    model(x, stage=0, target_mask=target, conditions=pack, timestep=torch.tensor([.4])).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad()
    out = model(x, stage=0, target_mask=target, conditions=pack, timestep=torch.tensor([.4]))
    assert torch.equal(out[:, ~target], x[:, ~target])
    assert not torch.equal(out[:, target], x[:, target])
    out.square().mean().backward()
    for branch in (model.geometry_encoder, model.banks):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in branch.parameters())
    assert x.grad is not None and x.grad.abs().sum() > 0
    model.enabled = False
    assert torch.equal(model(x, stage=0, target_mask=target, conditions=pack, timestep=torch.tensor([.4])), x)


def test_module_call_runs_hooks_and_preserves_fp32_values_and_gradients():
    model, pack = make_adapter(), make_condition()
    seen = []
    hook = model.register_forward_hook(lambda *args: seen.append(True))
    x = torch.randn(1, 4, 32)
    model(x, stage=0, target_mask=torch.ones(4, dtype=torch.bool), conditions=pack, timestep=torch.tensor([.4])).sum().backward()
    assert seen
    before = {k: p.detach().clone() for k, p in model.named_parameters()}
    grads = {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}
    nn.Sequential(model).to(dtype=torch.bfloat16)
    for name, parameter in model.named_parameters():
        assert parameter.dtype == torch.float32
        assert torch.equal(parameter, before[name])
        if name in grads:
            assert torch.equal(parameter.grad, grads[name])
    out = model(x.bfloat16(), stage=0, target_mask=torch.ones(4, dtype=torch.bool), conditions=pack, timestep=torch.tensor([.4]))
    assert out.dtype == torch.bfloat16 and torch.equal(out, x.bfloat16())
    hook.remove()


def test_invalid_and_empty_rois_stay_finite_and_null_condition_is_identity():
    model, pack = make_adapter(), make_condition()
    with torch.no_grad():
        model.output_heads["0"].weight.normal_()
    x, mask = torch.randn(1, 4, 32), torch.ones(4, dtype=torch.bool)
    pack = replace(pack, source_token_valid=torch.zeros_like(pack.source_token_valid), face_identity_valid=torch.zeros_like(pack.face_identity_valid), face_mask=torch.zeros_like(pack.face_mask), hand_mask=torch.zeros_like(pack.hand_mask))
    out = model(x, stage=0, target_mask=mask, conditions=pack, timestep=torch.tensor([.1]))
    assert torch.isfinite(out).all()
    pack = without_people(pack)
    assert torch.equal(model(x, stage=0, target_mask=mask, conditions=pack, timestep=torch.tensor([.1])), x)


def test_target_layout_and_timestep_batch_must_match_hidden():
    model, pack = make_adapter(), make_condition()
    for mask, time in ((torch.ones(5, dtype=torch.bool), torch.tensor([.2])), (torch.ones(4), torch.tensor([.2])), (torch.ones(4, dtype=torch.bool), torch.tensor([.2, .3]))):
        with pytest.raises(ValueError):
            model(torch.zeros(1, 4, 32), stage=0, target_mask=mask, conditions=pack, timestep=time)


def test_state_roundtrip_includes_strict_configuration_and_trainable_audit():
    model, restored = make_adapter(), make_adapter()
    state = model.state_dict()
    assert state["_extra_state"]["config"] == model.config.to_dict()
    restored.load_state_dict(state)
    report = model.trainable_parameter_report()
    assert report["all_fp32"] and report["count"] > 0
    assert report["config_signature"] == model.config.signature()
    bad = dict(state)
    bad["_extra_state"] = dict(state["_extra_state"], config=dict(model.config.to_dict(), heads=2))
    with pytest.raises(ValueError, match="config"):
        restored.load_state_dict(bad)


def test_zero_delta_preserves_signed_zero_bits_and_head_gradient():
    model, pack = make_adapter(), make_condition()
    x = torch.full((1, 4, 32), -0.)
    out = model(x, stage=0, target_mask=torch.ones(4, dtype=torch.bool), conditions=pack, timestep=torch.tensor([.2]))
    assert torch.equal(out.view(torch.int32), x.view(torch.int32))
    out.sum().backward()
    assert model.output_heads["0"].weight.grad.abs().sum() > 0


def test_contact_valid_branch_receives_gradient_after_head_update():
    model, pack = make_adapter(), make_condition(batch=2)
    relations = api().ContactRelations(torch.tensor([[0], [0]]), torch.tensor([[3], [3]]), torch.tensor([[1], [1]]), torch.tensor([[7], [7]]), torch.randn(2, 1, 6), torch.ones(2, 1), torch.ones(2, 1, dtype=torch.bool))
    pack = replace(pack, relations=relations)
    with torch.no_grad():
        model.output_heads["0"].weight.normal_()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = model(torch.randn(2, 4, 32), stage=0, target_mask=torch.ones(4, dtype=torch.bool), conditions=pack, timestep=torch.tensor([.2, .4]))
        out.square().mean().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.geometry_encoder.contacts.parameters())


def test_empty_source_bank_still_reads_current_control_hidden_dynamically():
    model, pack = make_adapter(), make_condition()
    pack = replace(pack, source_token_valid=torch.zeros_like(pack.source_token_valid), face_identity_valid=torch.zeros_like(pack.face_identity_valid))
    with torch.no_grad():
        model.output_heads["0"].weight.normal_()
    first = torch.randn(1, 4, 32)
    # Change token content, since global per-token shifts are removed by LayerNorm.
    second = first + torch.linspace(-1, 1, 32)
    kwargs = dict(stage=0, target_mask=torch.ones(4, dtype=torch.bool), conditions=pack, timestep=torch.tensor([.3]))
    delta_first = model(first, **kwargs) - first
    delta_second = model(second, **kwargs) - second
    assert not torch.allclose(delta_first, delta_second, rtol=1e-5, atol=1e-5)


def test_base_gate_is_fixed_at_one():
    model = make_adapter()
    assert model.gate.item() == 1
    assert "gate" not in dict(model.named_parameters())


@pytest.mark.parametrize("visibility_valid", [True, False])
def test_invisible_person_cannot_change_appearance_residual(visibility_valid):
    model, pack = make_adapter(), make_condition()
    owner = pack.owner.clone()
    owner[:, 1] = 0
    visibility = pack.spatial["visibility"].clone()
    visibility[:, 1] = 0
    valid = pack.spatial_valid["visibility"].clone()
    valid[:, 1] = visibility_valid
    pack = replace(pack, owner=owner, spatial=dict(pack.spatial, visibility=visibility),
                   spatial_valid=dict(pack.spatial_valid, visibility=valid))
    pack.validate()
    with torch.no_grad():
        model.output_heads["0"].weight.normal_()
    x = torch.randn(1, 4, 32)
    kwargs = dict(stage=0, target_mask=torch.ones(4, dtype=torch.bool), timestep=torch.tensor([.2]))
    initial = model(x, conditions=pack, **kwargs)
    source, identity = pack.source_tokens.clone(), pack.face_identity.clone()
    source[:, 1] += 100
    identity[:, 1] += 100
    assert torch.equal(initial, model(x, conditions=replace(pack, source_tokens=source, face_identity=identity), **kwargs))


def test_occluded_owner_is_rejected_before_attention():
    model, pack = make_adapter(), make_condition()
    pack = replace(pack, spatial=dict(pack.spatial, visibility=torch.zeros_like(pack.spatial["visibility"])))
    with pytest.raises(ValueError, match="owner"):
        model(torch.zeros(1, 4, 32), stage=0, target_mask=torch.ones(4, dtype=torch.bool), conditions=pack, timestep=torch.tensor([.2]))


def test_query_and_first_head_gradient_are_stable_under_hidden_scale_and_shift():
    model, pack = make_adapter(), make_condition()
    hidden = torch.randn(1, 4, 32)
    queries = []
    hook = model.query_projection.register_forward_hook(lambda module, inputs, output: queries.append(output.detach().clone()))
    gradients = []
    for scale, shift in ((1., 0.), (1., 10.), (10., 35.), (100., 350.), (1000., 3500.)):
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = model(hidden * scale + shift, stage=0, target_mask=torch.ones(4, dtype=torch.bool),
                           conditions=pack, timestep=torch.tensor([.3]))
        output.sum().backward()
        gradients.append(model.output_heads["0"].weight.grad.detach().clone())
    hook.remove()
    assert all(query.dtype == torch.float32 for query in queries)
    for query, gradient in zip(queries[1:], gradients[1:]):
        torch.testing.assert_close(query, queries[0], rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(gradient, gradients[0], rtol=2e-5, atol=2e-5)
