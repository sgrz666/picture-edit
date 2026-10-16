"""Exact control-only LoRA contracts, independent of pretrained downloads."""
import importlib
import importlib.util

import pytest
import torch
from torch import nn


def weights_module():
    name = "src.pose_control.qwen21_union.weights"
    assert importlib.util.find_spec(name) is not None, "control LoRA implementation is missing"
    return importlib.import_module(name)


def make_transformer(width=4, device=None):
    model = nn.Module()
    model.control_img_in = nn.Linear(129, width, device=device)
    model.control_blocks = nn.ModuleList()
    for index in range(16):
        block = nn.Module()
        block.attn = nn.Module()
        for key in ("to_q", "to_k", "to_v"):
            setattr(block.attn, key, nn.Linear(width, width, device=device))
        block.attn.to_out = nn.ModuleList([nn.Linear(width, width, device=device)])
        block.img_mlp = nn.Module()
        block.img_mlp.proj = nn.Linear(width, 3 * width, device=device)
        block.img_mlp.gate_layer = nn.Linear(width, 3 * width, device=device)
        block.img_mlp.out = nn.Linear(3 * width, width, device=device)
        block.after_proj = nn.Linear(width, width, device=device)
        if index == 0:
            block.before_proj = nn.Linear(width, width, device=device)
        model.control_blocks.append(block)
    model.transformer_blocks = nn.ModuleList([nn.Linear(width, width, device=device)])
    return model


def test_lora_inventory_has_exact_130_targets_and_official_parameter_budget():
    module = weights_module()
    targets = module.control_lora_targets()
    assert len(targets) == len(set(targets)) == 130
    assert targets[:2] == ("control_img_in", "control_blocks.0.before_proj")
    assert not any("transformer_blocks" in name for name in targets)
    model = make_transformer(4096, "meta")
    report = module.install_control_lora(model)
    assert tuple(report["targets"]) == targets
    assert report["trainable_parameters"] == 23_267_344
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)


def test_lora_zero_b_preserves_output_and_only_ab_can_train():
    module = weights_module()
    torch.manual_seed(12)
    model = make_transformer()
    x = torch.randn(2, 129)
    before = model.control_img_in(x).detach().clone()
    original_weight = model.control_img_in.weight.detach().clone()
    module.install_control_lora(model, expected_parameters=None)
    torch.testing.assert_close(model.control_img_in(x), before, rtol=0, atol=0)
    assert torch.equal(model.control_img_in.weight, original_weight)
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == name.endswith((".lora_A", ".lora_B"))
        if name.endswith(".lora_B"):
            assert parameter.count_nonzero() == 0
    original_a = model.control_img_in.lora_A.detach().clone()
    model.control_img_in(x).sum().backward()
    original_b_grad = model.control_img_in.lora_B.grad.detach().clone()
    model.to(dtype=torch.bfloat16)
    assert model.control_img_in.weight.dtype == torch.bfloat16
    assert model.control_img_in.lora_A.dtype == torch.float32
    assert model.control_img_in.lora_B.dtype == torch.float32
    assert torch.equal(model.control_img_in.lora_A, original_a), "model.to must not round FP32 LoRA values through BF16"
    assert torch.equal(model.control_img_in.lora_B.grad, original_b_grad), "model.to must preserve FP32 gradient values"


def test_install_validates_whole_inventory_before_mutation():
    module = weights_module()
    model = make_transformer()
    model.control_blocks[7].attn.to_q = nn.Identity()
    with pytest.raises(ValueError, match="Linear"):
        module.install_control_lora(model, expected_parameters=None)
    assert isinstance(model.control_img_in, nn.Linear)
    with pytest.raises(ValueError, match="parameter"):
        module.install_control_lora(make_transformer())


def test_lora_state_roundtrip_and_disabled_output():
    module = weights_module()
    model = make_transformer()
    module.install_control_lora(model, expected_parameters=None)
    with torch.no_grad():
        model.control_img_in.lora_B.fill_(.2)
    x = torch.randn(1, 129)
    changed = model.control_img_in(x)
    state = module.lora_state_dict(model)
    assert len(state) == 260
    original = model.control_img_in.base_layer(x)
    module.set_lora_enabled(model, False)
    torch.testing.assert_close(model.control_img_in(x), original, rtol=0, atol=0)
    module.set_lora_enabled(model, True)
    with torch.no_grad():
        model.control_img_in.lora_B.zero_()
    module.load_lora_state_dict(model, state)
    torch.testing.assert_close(model.control_img_in(x), changed)
    bad = dict(state)
    bad.pop(next(iter(bad)))
    with pytest.raises(ValueError, match="keys"):
        module.load_lora_state_dict(model, bad)


def test_frozen_audit_detects_unexpected_trainable_parameter_and_gradient():
    module = weights_module()
    model = make_transformer()
    module.install_control_lora(model, expected_parameters=None)
    before = module.frozen_parameter_digest(model)
    model.control_img_in(torch.randn(1, 129)).square().mean().backward()
    module.audit_trainable_parameters(model, expected_parameters=None)
    assert module.frozen_parameter_digest(model) == before
    model.transformer_blocks[0].weight.requires_grad_(True)
    with pytest.raises(ValueError, match="trainable"):
        module.audit_trainable_parameters(model, expected_parameters=None)


def test_control_state_audit_rejects_missing_extra_and_shape_mismatch():
    module = weights_module()
    model = make_transformer()
    control = {key: value for key, value in model.state_dict().items() if key.startswith("control_")}
    report = module.audit_control_state(model, control, expected_tensors=None)
    assert report["tensor_count"] == len(control)
    missing = dict(control)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="missing"):
        module.audit_control_state(model, missing, expected_tensors=None)
    wrong = dict(control, **{"control_img_in.weight": torch.zeros(1)})
    with pytest.raises(ValueError, match="shape"):
        module.audit_control_state(model, wrong, expected_tensors=None)
    with pytest.raises(ValueError, match="unexpected"):
        module.audit_control_state(model, dict(control, unknown=torch.zeros(1)), expected_tensors=None)
