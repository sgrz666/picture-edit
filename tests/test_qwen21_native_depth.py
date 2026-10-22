from __future__ import annotations

import copy

import pytest
import torch


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def setup(scale=16, hw=(32, 64), width=16):
    from src.pose_control.qwen21_full.depth_condition import relative_depth_condition
    from src.pose_control.qwen21_full.native_depth import NativeDepthConfig, Qwen21NativeDepth
    from src.pose_control.qwen21_full.native_geometry import NativeGeometryFeatures
    h, w = hw
    d = torch.linspace(0, 1, h*w).reshape(1, 1, h, w)
    valid = torch.ones_like(d, dtype=torch.bool)
    condition = relative_depth_condition(d, valid, valid,
            [{"source_id": "pair-1", "camera_id": "camera-1", "canvas_id": "canvas-1",
              "units": "unknown", "scale_status": "unknown"}], polarity="near_one")
    features = NativeGeometryFeatures(torch.randn(1, h//16*w//16, width),
            torch.randn(1, h//16*w//16, width), (h//16, w//16), torch.tensor([[True, False]]))
    module = Qwen21NativeDepth(NativeDepthConfig(geometry_dim=width, attention_scale=scale))
    return module, features, condition


@pytest.mark.parametrize("scale", [8, 16])
@pytest.mark.parametrize("hw", [(32, 32), (32, 64), (64, 96)])
def test_zero_projection_preserves_full_geometry_and_interaction(scale, hw):
    module, features, condition = setup(scale, hw)
    result = module(features, condition)
    assert torch.equal(result.geometry, features.geometry)
    assert result.interaction is features.interaction
    assert result.person_valid is features.person_valid
    assert result.target_token_hw == features.target_token_hw


def test_off_and_zero_strength_return_the_original_object():
    module, features, condition = setup()
    assert module(features, None, enabled=False) is features
    assert module(features, None, strength=0) is features
    with pytest.raises(ValueError):
        module(features, None)


def test_projection_learns_immediately_then_encoder_and_depth_are_sensitive():
    module, features, condition = setup()
    module(features, condition).geometry.square().mean().backward()
    assert module.projection.weight.grad.norm() > 0
    assert module.encoder.conv_in.weight.grad.abs().sum() == 0
    module.zero_grad(set_to_none=True)
    torch.nn.init.normal_(module.projection.weight, std=.02)
    result = module(features, condition)
    result.geometry.square().mean().backward()
    assert module.encoder.conv_in.weight.grad.norm() > 0
    assert module.encoder.attention.proj_in.weight.grad.norm() > 0
    from dataclasses import replace
    other = replace(condition, relative=1-condition.relative)
    assert not torch.allclose(result.geometry, module(features, other).geometry)


def test_missing_conditions_are_zero_even_with_learned_bias():
    module, features, condition = setup()
    from src.pose_control.qwen21_full.depth_condition import relative_depth_condition
    missing = relative_depth_condition(torch.zeros_like(condition.relative), torch.zeros_like(condition.valid),
            condition.canvas_valid, condition.metadata, polarity="near_one")
    torch.nn.init.normal_(module.projection.weight)
    torch.nn.init.constant_(module.projection.bias, 3)
    assert module(features, missing) is features


def test_wrong_grid_and_non_fp32_weights_are_rejected():
    module, features, condition = setup()
    from dataclasses import replace
    with pytest.raises(ValueError, match="grid"):
        module(replace(features, target_token_hw=(4, 2)), condition)
    module.bfloat16()
    with pytest.raises(ValueError, match="FP32"):
        module(features, condition)


def test_state_roundtrip_and_config_mismatch():
    module, features, condition = setup()
    torch.nn.init.normal_(module.projection.weight, std=.01)
    other, _, _ = setup()
    other.load_state_dict(copy.deepcopy(module.state_dict()), strict=True)
    assert torch.equal(module(features, condition).geometry, other(features, condition).geometry)
    wrong, _, _ = setup(scale=8)
    with pytest.raises(ValueError, match="config"):
        wrong.load_state_dict(module.state_dict(), strict=True)


def official_like_state(module):
    state = {}
    for k, v in module.encoder.state_dict().items():
        if k.startswith("downsample."):
            continue
        name = "attentions.5." + k[len("attention."):] if k.startswith("attention.") else k
        state[name] = v.clone()
    state["conv_in.weight"] = torch.randn(16, 3, 3, 3)
    state["conv_out.weight"] = torch.randn(320, 256, 3, 3)
    state["conv_out.bias"] = torch.randn(320)
    return state


def test_champ_gray_kernel_sum_and_active_attention_are_strictly_loaded():
    module, _, _ = setup()
    state = official_like_state(module)
    report = module.load_champ_state_dict(state)
    assert torch.equal(module.encoder.conv_in.weight[:, 0], state["conv_in.weight"].sum(1))
    assert module.encoder.conv_in.weight[:, 1].eq(0).all()
    assert torch.equal(module.encoder.attention.proj_in.weight, state["attentions.5.proj_in.weight"])
    assert module.projection.weight.eq(0).all()
    assert report["discarded_sd_output_head"] is True
    assert report["attention_relocated"] is True


@pytest.mark.parametrize("corruption", ["unknown", "missing", "shape", "nan", "unused_unknown"])
def test_champ_load_rejects_partial_or_unrecognized_payload_without_mutation(corruption):
    module, _, _ = setup()
    state = official_like_state(module)
    if corruption == "unknown":
        state["unexpected.weight"] = torch.ones(1)
    elif corruption == "missing":
        del state["blocks.0.weight"]
    elif corruption == "shape":
        state["conv_in.weight"] = torch.ones(16, 1, 3, 3)
    elif corruption == "nan":
        state["blocks.0.weight"][0, 0, 0, 0] = float("nan")
    else:
        state["attentions.0.unknown.weight"] = torch.ones(1)
    before = module.encoder.conv_in.weight.clone()
    with pytest.raises(ValueError):
        module.load_champ_state_dict(state)
    assert torch.equal(before, module.encoder.conv_in.weight)
