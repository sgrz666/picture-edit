from __future__ import annotations

from dataclasses import replace
import importlib.util
import sys
import types

import pytest
import torch


try:
    _diffusers_spec = importlib.util.find_spec("diffusers")
except ValueError:
    _diffusers_spec = None
if _diffusers_spec is None and "diffusers.models" not in sys.modules:
    diffusers = types.ModuleType("diffusers")
    diffusers_models = types.ModuleType("diffusers.models")

    class _SD3ControlNetModel(torch.nn.Module):
        pass

    diffusers_models.SD3ControlNetModel = _SD3ControlNetModel
    diffusers.models = diffusers_models
    sys.modules["diffusers"] = diffusers
    sys.modules["diffusers.models"] = diffusers_models

from src.pose_control.v6.checkpoint import (
    ARCHITECTURE_VERSION,
    build_v64_checkpoint,
    build_v65_checkpoint,
    freeze_for_face_training,
    load_v63_checkpoint,
    load_v64_checkpoint,
    load_v65_checkpoint,
)
from src.pose_control.v6.control_core import BranchControlResiduals
from src.pose_control.v6.deepgen_adapter import UnifiedSMPLXAdapterV6
from src.pose_control.v6.detail import DetailReferenceBatch, FaceHandDetailCondition
from src.pose_control.v6.detail.hand import (
    HandDetailCondition,
    hand_to_legacy_detail,
    legacy_to_face_and_hand,
    references_to_hand_only,
)
from src.pose_control.v6.face import PreparedFaceConditioning
from src.pose_control.v6.face.adapter import (
    FaceControlAdapter,
    scatter_face_roi_residuals,
)
from src.pose_control.v6.interface import ControlStrengthController


def _prepared_face(*, batch_size: int = 1, both: bool = True) -> PreparedFaceConditioning:
    valid = torch.tensor([[True, both]], dtype=torch.bool).expand(batch_size, -1).clone()
    boxes = torch.tensor(
        [[[0.0, 0.0, 0.75, 0.75], [0.25, 0.25, 1.0, 1.0]]],
        dtype=torch.float32,
    ).expand(batch_size, -1, -1).clone()
    gate = valid.to(torch.float32)
    return PreparedFaceConditioning(
        spatial_features=torch.randn(batch_size, 2, 512, 16, 16)
        * gate[..., None, None, None],
        identity_tokens=torch.randn(batch_size, 2, 4, 512) * gate[..., None, None],
        texture_tokens=torch.randn(batch_size, 2, 8, 512) * gate[..., None, None],
        texture_delta=torch.randn(batch_size, 2, 4, 512) * gate[..., None, None],
        geometry_tokens=torch.randn(batch_size, 2, 4, 512) * gate[..., None, None],
        face_masks=torch.ones(batch_size, 2, 128, 128) * gate[..., None, None],
        target_boxes=boxes * gate[..., None],
        face_valid=valid,
    ).validate()


def _legacy_detail(batch_size: int = 1) -> FaceHandDetailCondition:
    boxes = torch.tensor([0.1, 0.1, 0.9, 0.9]).repeat(batch_size, 2, 3, 1)
    return FaceHandDetailCondition(
        face_keypoints=torch.rand(batch_size, 2, 68, 3),
        hand_keypoints=torch.rand(batch_size, 2, 2, 21, 3),
        smplx_detail=torch.randn(batch_size, 2, 103),
        source_boxes=boxes.clone(),
        target_boxes=boxes.clone(),
        region_valid=torch.ones(batch_size, 2, 3, dtype=torch.bool),
        source_indices=torch.tensor([[2, 5]], dtype=torch.long).expand(batch_size, -1).clone(),
    ).validate()


def test_legacy_condition_splits_face_and_hand_and_hand_roundtrips_without_face() -> None:
    legacy = _legacy_detail()
    face, hand = legacy_to_face_and_hand(legacy)
    assert face.landmarks.shape == (1, 2, 72, 3)
    assert torch.count_nonzero(face.landmarks[:, :, :4]) == 0
    torch.testing.assert_close(face.landmarks[:, :, 4:], legacy.face_keypoints)
    torch.testing.assert_close(face.expression, legacy.smplx_detail[..., :10])
    torch.testing.assert_close(face.jaw_pose, legacy.smplx_detail[..., 10:13])
    assert hand.hand_keypoints.shape == (1, 2, 2, 21, 3)
    torch.testing.assert_close(hand.hand_pose[:, :, 0], legacy.smplx_detail[..., 13:58])
    torch.testing.assert_close(hand.hand_pose[:, :, 1], legacy.smplx_detail[..., 58:103])

    hand_only = hand_to_legacy_detail(hand).validate()
    assert not hand_only.region_valid[:, :, 0].any()
    assert torch.count_nonzero(hand_only.face_keypoints) == 0
    torch.testing.assert_close(hand_only.hand_keypoints, legacy.hand_keypoints)

    references = DetailReferenceBatch(
        images=torch.randn(1, 2, 3, 1, 3, 224, 224),
        reference_valid=torch.ones(1, 2, 3, 1, dtype=torch.bool),
    ).validate()
    filtered = references_to_hand_only(references).validate()
    assert not filtered.reference_valid[:, :, 0].any()
    assert torch.count_nonzero(filtered.images[:, :, 0]) == 0


def test_hand_condition_contract_preserves_discrete_types_and_cfg_order() -> None:
    _, hand = legacy_to_face_and_hand(_legacy_detail(batch_size=1))
    assert isinstance(hand, HandDetailCondition)
    moved = hand.to(dtype=torch.float64)
    assert moved.hand_keypoints.dtype == torch.float64
    assert moved.region_valid.dtype == torch.bool
    assert moved.source_indices.dtype == torch.long
    expanded = moved.expand_to_batch(2)
    torch.testing.assert_close(expanded.hand_keypoints[0], expanded.hand_keypoints[1])
    assert expanded.source_indices.tolist() == [[2, 5], [2, 5]]


def test_face_adapter_zero_init_invalid_gradients_and_activated_signal() -> None:
    torch.manual_seed(4)
    adapter = FaceControlAdapter(face_dim=16, deepgen_dim=8, heads=4, rank=4)
    prepared = _prepared_face(both=False)
    prepared.spatial_features.requires_grad_(True)
    zero = adapter(
        prepared,
        target_token_hw=(8, 8),
        timestep=torch.tensor([500.0]),
        texture_scale=1.0,
    )
    assert len(zero) == 6
    assert all(value.shape == (1, 64, 8) for value in zero)
    assert all(torch.count_nonzero(value) == 0 for value in zero)

    with torch.no_grad():
        adapter.zero_heads[0].weight.fill_(0.02)
    active = adapter(
        prepared,
        target_token_hw=(8, 8),
        timestep=torch.tensor([500.0]),
        texture_scale=1.0,
    )
    assert torch.count_nonzero(active[0]) > 0
    active[0].square().mean().backward()
    assert prepared.spatial_features.grad is not None
    assert torch.count_nonzero(prepared.spatial_features.grad[:, 1]) == 0


def test_texture_inputs_are_exactly_inactive_before_texture_schedule() -> None:
    torch.manual_seed(8)
    adapter = FaceControlAdapter(face_dim=16, deepgen_dim=8, heads=4, rank=4).eval()
    with torch.no_grad():
        adapter.zero_heads[0].weight.fill_(0.02)
    prepared = _prepared_face(both=True)
    changed = replace(
        prepared,
        texture_tokens=prepared.texture_tokens + 100,
        texture_delta=prepared.texture_delta - 100,
    ).validate()
    kwargs = dict(
        target_token_hw=(4, 4),
        timestep=torch.tensor([500.0]),
        texture_scale=0.0,
    )
    original = adapter(prepared, **kwargs)
    without_texture = adapter(changed, **kwargs)
    for first, second in zip(original, without_texture):
        torch.testing.assert_close(first, second)


def test_scatter_overlap_is_weighted_average_times_max_mask_not_double_sum() -> None:
    roi = torch.ones(1, 2, 4, 4, 3)
    mask = torch.ones(1, 2, 4, 4)
    boxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]])
    both = scatter_face_roi_residuals(
        roi, mask, boxes, torch.tensor([[True, True]]), (4, 4)
    )
    one = scatter_face_roi_residuals(
        roi, mask, boxes, torch.tensor([[True, False]]), (4, 4)
    )
    torch.testing.assert_close(both, one)
    assert both.max().item() == pytest.approx(1.0)


def test_face_and_hand_strengths_are_independent_and_detail_is_total_switch() -> None:
    shape = (1, 4, 3)
    zeros = tuple(torch.zeros(shape) for _ in range(6))
    hand = tuple(torch.full(shape, 2.0) for _ in range(6))
    face = tuple(torch.full(shape, 3.0) for _ in range(6))
    residuals = BranchControlResiduals(
        geometry=zeros,
        interaction=zeros,
        target_token_hw=(2, 2),
        detail=hand,
        face=face,
    ).validate()
    masks = torch.zeros(1, 2, 3, 2, 2)
    masks[:, :, 1:] = 1
    controller = ControlStrengthController(enable_face=True)
    hand_only, _ = controller(
        residuals,
        denoise_progress=1.0,
        face_strength=0.0,
        hand_strength=1.0,
        detail_region_masks=masks,
    )
    face_only, _ = controller(
        residuals,
        denoise_progress=1.0,
        face_strength=1.0,
        hand_strength=0.0,
        detail_region_masks=masks,
    )
    disabled, _ = controller(
        residuals,
        denoise_progress=1.0,
        detail_strength=0.0,
        detail_region_masks=masks,
    )
    assert hand_only[0][0, 0, 0].item() == pytest.approx(2.0)
    assert face_only[0][0, 0, 0].item() == pytest.approx(3.0 * 0.1)
    assert all(torch.count_nonzero(value) == 0 for value in disabled)


def test_face_and_texture_schedules_have_independent_late_windows() -> None:
    controller = ControlStrengthController(enable_face=True)
    assert controller.face_schedule.multiplier(0.2) == 0.0
    assert 0.0 < controller.face_schedule.multiplier(0.5) < 1.0
    assert controller.face_texture_schedule.multiplier(0.5) == 0.0
    assert 0.0 < controller.face_texture_schedule.multiplier(0.8) < 1.0
    torch.testing.assert_close(
        controller.face_group_prior,
        torch.tensor([0.1, 0.2, 0.5, 0.8, 1.0, 1.0]),
    )


def test_v65_checkpoint_roundtrip_v64_migration_and_face_only_freeze() -> None:
    source = UnifiedSMPLXAdapterV6.build_tiny(
        hidden_dim=16, geometry_channels=4, enable_face_adapter=True, face_dim=16
    )
    assert ARCHITECTURE_VERSION == "v6.5"
    checkpoint = build_v65_checkpoint(source, feature_cache_fingerprint="test")
    assert checkpoint["metadata"]["branch_names"] == [
        "geometry", "interaction", "hand", "face"
    ]
    target = UnifiedSMPLXAdapterV6.build_tiny(
        hidden_dim=16, geometry_channels=4, enable_face_adapter=True, face_dim=16
    )
    load_v65_checkpoint(target, checkpoint)
    for key, value in source.state_dict().items():
        assert torch.equal(value, target.state_dict()[key]), key

    legacy = UnifiedSMPLXAdapterV6.build_tiny(hidden_dim=16, geometry_channels=4)
    migrated = UnifiedSMPLXAdapterV6.build_tiny(
        hidden_dim=16, geometry_channels=4, enable_face_adapter=True, face_dim=16
    )
    with torch.no_grad():
        for head in migrated.control_interface.face_adapter.zero_heads:
            head.weight.fill_(1)
    result = load_v64_checkpoint(migrated, build_v64_checkpoint(legacy))
    assert result.missing_keys
    assert all(
        key.startswith(("face_preparer.", "control_interface.face_adapter.",
                        "control_interface.strength_controller.face_"))
        for key in result.missing_keys
    )
    assert all(
        torch.count_nonzero(head.weight) == 0
        for head in migrated.control_interface.face_adapter.zero_heads
    )

    v63_state = {
        key: value.detach().clone()
        for key, value in legacy.state_dict().items()
        if not key.startswith("detail_preparer.")
        and not key.startswith("control_interface.control_core.detail_branch.")
        and key
        != "control_interface.strength_controller.detail_log_group_scale"
    }
    with torch.no_grad():
        for head in migrated.control_interface.face_adapter.zero_heads:
            head.weight.fill_(1)
    load_v63_checkpoint(migrated, {"adapter_state_dict": v63_state})
    assert all(
        torch.count_nonzero(head.weight) == 0
        for head in migrated.control_interface.face_adapter.zero_heads
    )

    trainable = freeze_for_face_training(migrated)
    assert trainable
    names = {name for name, value in migrated.named_parameters() if value.requires_grad}
    assert names
    assert all(
        name.startswith(("face_preparer.", "control_interface.face_adapter."))
        or name == "control_interface.strength_controller.face_log_group_scale"
        for name in names
    )


def test_prepared_face_cfg_expansion_and_branch_validation() -> None:
    from tests.test_deepgen_control_interface_v1 import _prepared

    prepared = replace(_prepared(batch_size=1), face=_prepared_face()).validate()
    expanded = prepared.expand_to_batch(2)
    assert expanded.face is not None
    torch.testing.assert_close(
        expanded.face.spatial_features[0], expanded.face.spatial_features[1]
    )
    base = torch.zeros(2, 4, 5)
    branches = BranchControlResiduals(
        geometry=(base,) * 6,
        interaction=(base,) * 6,
        detail=(base,) * 6,
        face=(base,) * 6,
        target_token_hw=(2, 2),
    ).validate()
    assert branches.face is not None


def test_face_input_changes_only_face_raw_residual_and_padding_stays_zero() -> None:
    from tests.test_deepgen_control_interface_v1 import _prepared

    torch.manual_seed(21)
    model = UnifiedSMPLXAdapterV6.build_tiny(
        hidden_dim=32,
        geometry_channels=16,
        enable_face_adapter=True,
        face_dim=16,
    ).eval()
    with torch.no_grad():
        model.control_interface.face_adapter.zero_heads[0].weight.fill_(0.02)
    face = _prepared_face(both=True)
    first_prepared = replace(_prepared(), face=face).validate()
    changed_face = replace(
        face,
        spatial_features=face.spatial_features + 3,
        identity_tokens=face.identity_tokens - 2,
    ).validate()
    second_prepared = replace(first_prepared, face=changed_face).validate()
    inputs = dict(
        target_latents=torch.randn(1, 16, 8, 12),
        encoder_hidden_states=torch.randn(1, 7, 24),
        pooled_projections=torch.randn(1, 20),
        timestep=torch.tensor([500.0]),
    )
    first = model.control_interface.build_raw_residuals(
        prepared=first_prepared, **inputs
    )
    second = model.control_interface.build_raw_residuals(
        prepared=second_prepared, **inputs
    )
    assert all(torch.equal(a, b) for a, b in zip(first.geometry, second.geometry))
    assert all(torch.equal(a, b) for a, b in zip(first.interaction, second.interaction))
    assert first.face is not None and second.face is not None
    assert any(not torch.equal(a, b) for a, b in zip(first.face, second.face))

    output = model.control_interface(
        prepared=first_prepared,
        cond_hidden_states=[[torch.randn(16, 8, 12), torch.randn(16, 4, 4)]],
        denoise_progress=1.0,
        geometry_strength=0.0,
        interaction_strength=0.0,
        **inputs,
    )
    assert output.block_controlnet_hidden_states[0].shape == (1, 52, 32)
    assert torch.count_nonzero(output.block_controlnet_hidden_states[0][:, :24]) > 0
    assert torch.count_nonzero(output.block_controlnet_hidden_states[0][:, 24:]) == 0


def test_face_branch_is_not_executed_when_face_strength_is_zero() -> None:
    from tests.test_deepgen_control_interface_v1 import _prepared

    model = UnifiedSMPLXAdapterV6.build_tiny(
        hidden_dim=32,
        geometry_channels=16,
        enable_face_adapter=True,
        face_dim=16,
    ).eval()
    prepared = replace(_prepared(), face=_prepared_face(both=True)).validate()
    calls = []
    hook = model.control_interface.face_adapter.register_forward_hook(
        lambda *args: calls.append(True)
    )
    inputs = dict(
        target_latents=torch.randn(1, 16, 8, 12),
        prepared=prepared,
        cond_hidden_states=None,
        encoder_hidden_states=torch.randn(1, 7, 24),
        pooled_projections=torch.randn(1, 20),
        timestep=torch.tensor([500.0]),
        denoise_progress=1.0,
    )
    disabled = model.control_interface(face_strength=0.0, **inputs)
    assert calls == []
    assert disabled.diagnostics["face_executed"] is False
    assert disabled.diagnostics["face_active_people"] == 0
    enabled = model.control_interface(face_strength=1.0, **inputs)
    hook.remove()
    assert calls == [True]
    assert enabled.diagnostics["face_executed"] is True
    assert enabled.diagnostics["face_active_people"] == 2


def test_legacy_prepare_is_operationally_hand_only() -> None:
    from src.pose_control.v6.conditions import AdapterIdentityCondition
    from tests.test_deepgen_control_interface_v1 import _control_state

    model = UnifiedSMPLXAdapterV6.build_tiny(hidden_dim=32, geometry_channels=16)
    prepared = model.prepare_control(
        state=_control_state(dual=True),
        identity_condition=AdapterIdentityCondition(
            source_person_latents=torch.randn(1, 2, 16, 8, 12),
            source_indices=torch.tensor([[2, 5]]),
        ),
        source_scene_latents=torch.randn(1, 16, 8, 12),
        target_latent_hw=(8, 12),
        detail_condition=_legacy_detail(),
    )
    assert prepared.face is None
    assert prepared.detail is not None
    assert not prepared.detail.region_valid[:, :, 0].any()
    assert torch.count_nonzero(prepared.detail.region_masks[:, :, 0]) == 0
    assert not prepared.detail.detail_token_mask.reshape(1, 2, 3, 8)[:, :, 0].any()


def test_face_optimizer_step_keeps_every_old_parameter_byte_identical() -> None:
    torch.manual_seed(31)
    model = UnifiedSMPLXAdapterV6.build_tiny(
        hidden_dim=16,
        geometry_channels=4,
        enable_face_adapter=True,
        face_dim=16,
    )
    freeze_for_face_training(model)
    frozen = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
        if not value.requires_grad
    }
    with torch.no_grad():
        model.control_interface.face_adapter.zero_heads[0].weight.fill_(0.01)
    optimizer = torch.optim.SGD(
        [value for value in model.parameters() if value.requires_grad], lr=1e-3
    )
    output = model.control_interface.face_adapter(
        _prepared_face(both=True),
        target_token_hw=(4, 4),
        timestep=torch.tensor([300.0]),
        texture_scale=1.0,
    )
    sum(value.square().mean() for value in output).backward()
    optimizer.step()
    for name, value in model.named_parameters():
        if name in frozen:
            assert torch.equal(value, frozen[name]), name
