"""CPU checks for the reversible trained-grid Face scatter boundary."""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from src.pose_control.qwen21_full.deepgen_adapter import UnifiedSMPLXAdapterV6
from src.pose_control.qwen21_full.face.conditions import PreparedFaceConditioning
from src.pose_control.qwen21_full.qwen_adapter import (
    LAYER_GROUPS, Qwen21FullAdapter, resize_face_residuals_from_reference_grid,
)


@pytest.fixture(autouse=True)
def limited_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def face_condition(*, valid=True):
    gate = torch.tensor([[float(valid), 0.]])
    return PreparedFaceConditioning(
        spatial_features=torch.randn(1, 2, 512, 16, 16) * gate[..., None, None, None],
        identity_tokens=torch.randn(1, 2, 4, 512) * gate[..., None, None],
        texture_tokens=torch.randn(1, 2, 8, 512) * gate[..., None, None],
        texture_delta=torch.randn(1, 2, 4, 512) * gate[..., None, None],
        geometry_tokens=torch.randn(1, 2, 4, 512) * gate[..., None, None],
        face_masks=torch.ones(1, 2, 128, 128) * gate[..., None, None],
        target_boxes=torch.tensor([[[.25, .25, .75, .75], [0., 0., 0., 0.]]]) * gate[..., None],
        face_valid=gate.bool(),
    ).validate()


def runtime(grid):
    torch.manual_seed(29)
    legacy = UnifiedSMPLXAdapterV6.build_tiny(
        hidden_dim=32, geometry_channels=16, context_input_dim=24,
        pooled_input_dim=20, num_heads=4, enable_face_adapter=True)
    for heads in (legacy.control_core.geometry_zero_heads,
                  legacy.control_interface.face_adapter.zero_heads):
        for head in heads:
            nn.init.normal_(head.weight, std=.02)
    model = Qwen21FullAdapter(legacy, qwen_hidden_size=16, qwen_text_dim=24, pooled_input_dim=20)
    nn.init.normal_(model.output_bridges['geometry'].weight, std=.02)
    nn.init.normal_(model.output_bridges['face'].weight, std=.02)
    bundle = model.condition_injector(
        normal_a=torch.zeros(1, 3, 32, 32), pose_heatmap_a=torch.zeros(1, 25, 32, 32),
        part_onehot_a=torch.zeros(1, 14, 32, 32), smplx_global_a=torch.zeros(1, 26),
        human_mask_a=torch.ones(1, 1, 32, 32), task_id=torch.zeros(1, dtype=torch.long))
    prepared = model.prepare_conditioning(
        condition_bundle=bundle, source_latent=torch.randn(1, grid * grid, 64),
        target_token_hw=(grid, grid))
    args = dict(noisy_target=torch.randn(1, grid * grid, 64),
                prepared=replace(prepared, face=face_condition()),
                prompt_embeds=torch.randn(1, 5, 24), prompt_mask=torch.ones(1, 5, dtype=torch.bool),
                sigma=torch.tensor([.2]), target_token_hw=(grid, grid), hand_mode='off')
    return model, args


def test_512_reference_grid_is_an_exact_bypass_without_a_second_mask():
    # Actual 512 layout: 32x32 tokens. Values outside the box deliberately remain
    # nonzero, so an accidental second mask would fail this exact bypass check.
    residuals = tuple(torch.randn(1, 1024, 8) for _ in range(6))
    result = resize_face_residuals_from_reference_grid(
        residuals, reference_token_hw=(32, 32), target_token_hw=(32, 32),
        prepared_face=face_condition())
    assert result is residuals
    for before, after in zip(residuals, result):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_1024_resize_is_zero_outside_face_and_preserves_gradient():
    # Actual 512 -> 1024 layouts, without any base-model or data dependencies.
    residuals = tuple(torch.ones(1, 1024, 8, requires_grad=True) for _ in range(6))
    condition = face_condition()
    resized = resize_face_residuals_from_reference_grid(
        residuals, reference_token_hw=(32, 32), target_token_hw=(64, 64),
        prepared_face=condition)
    outside = torch.ones(64, 64, dtype=torch.bool)
    outside[16:48, 16:48] = False
    for value in resized:
        assert value.shape == (1, 4096, 8)
        assert torch.count_nonzero(value[:, outside.flatten()]) == 0
        assert torch.count_nonzero(value[:, ~outside.flatten()]) > 0
    sum(value.square().sum() for value in resized).backward()
    assert all(value.grad is not None and value.grad.abs().sum() > 0 for value in residuals)


def test_equal_grid_integration_matches_historical_body_and_face_exactly():
    model, args = runtime(4)
    keys = tuple(model.state_dict())
    historical = model.build_residuals(**args)
    explicit = model.build_residuals(**args, face_reference_token_hw=(4, 4))
    assert explicit.diagnostics['face_scatter_resized'] is False
    assert tuple(model.state_dict()) == keys
    assert model.active_block_mapping == LAYER_GROUPS
    for before, after in zip(historical.residuals, explicit.residuals):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_face_only_reference_scatter_leaves_body_and_mapping_unchanged():
    model, args = runtime(8)
    native = model.build_residuals(**args)
    calls = []
    face = model.legacy.control_interface.face_adapter
    handle = face.register_forward_pre_hook(
        lambda module, inputs, kwargs: calls.append(kwargs['target_token_hw']), with_kwargs=True)
    try:
        result = model.build_residuals(**args, face_reference_token_hw=(4, 4))
    finally:
        handle.remove()
    assert calls == [(4, 4)]
    assert result.diagnostics['face_scatter_resized'] is True
    assert result.diagnostics['active_block_mapping'] == [list(group) for group in LAYER_GROUPS]
    for name in ('geometry', 'interaction'):
        for before, after in zip(getattr(native.raw_residuals, name), getattr(result.raw_residuals, name)):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
    outside = torch.ones(8, 8, dtype=torch.bool)
    outside[2:6, 2:6] = False
    for value in result.raw_residuals.face:
        assert torch.count_nonzero(value[:, outside.flatten()]) == 0
    # The bridge and retained Face core both remain differentiable.
    loss = sum(model.output_bridges['face'](value).square().sum() for value in result.raw_residuals.face)
    loss.backward()
    assert model.output_bridges['face'].weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in face.shared_block.parameters())


@pytest.mark.parametrize('reference', [None, (4, 4)])
def test_face_disabled_or_invalid_returns_the_same_body(reference):
    model, args = runtime(8)
    off = model.build_residuals(**args, face_strength=0, face_reference_token_hw=reference)
    body_args = dict(args, prepared=replace(args['prepared'], face=None))
    body = model.build_residuals(**body_args, face_strength=0)
    invalid_args = dict(args, prepared=replace(args['prepared'], face=face_condition(valid=False)))
    invalid = model.build_residuals(**invalid_args, face_reference_token_hw=reference)
    for expected, disabled, no_face in zip(body.residuals, off.residuals, invalid.residuals):
        torch.testing.assert_close(disabled, expected, rtol=0, atol=0)
        torch.testing.assert_close(no_face, expected, rtol=0, atol=0)


@pytest.mark.parametrize('reference', [(0, 32), (32,), (32., 32), (True, 32)])
def test_invalid_reference_grid_is_rejected(reference):
    model, args = runtime(4)
    with pytest.raises(ValueError, match='two positive integers'):
        model.build_residuals(**args, face_reference_token_hw=reference)
