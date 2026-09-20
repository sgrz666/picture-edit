from __future__ import annotations

import pytest
import torch
from dataclasses import replace

from src.pose_control.v6.checkpoint import (
    DETAIL_SCHEMA_VERSION,
    build_v64_checkpoint,
    freeze_for_detail_training,
    load_v63_checkpoint,
    load_v64_checkpoint,
)
from src.pose_control.v6.control_core import BranchControlResiduals, SharedRecurrentControlCore
from src.pose_control.v6.detail import PreparedFaceHandDetailConditioning
from src.pose_control.v6.interface import ControlStrengthController, PreparedControlConditioning


def _core() -> SharedRecurrentControlCore:
    return SharedRecurrentControlCore.build_tiny(
        hidden_dim=16, condition_channels=20, context_input_dim=12,
        pooled_input_dim=10, rank=4,
    ).eval()


def _core_inputs(detail_value: float = 1.0):
    return dict(
        target_latents=torch.randn(2, 16, 4, 4),
        geometry_condition=torch.randn(2, 20, 4, 4),
        interaction_condition=torch.randn(2, 20, 4, 4),
        interaction_valid=torch.tensor([True, False]),
        adapter_tokens=torch.randn(2, 3, 16),
        adapter_token_mask=torch.ones(2, 3, dtype=torch.bool),
        encoder_hidden_states=torch.randn(2, 5, 12),
        pooled_projections=torch.randn(2, 10),
        timestep=torch.tensor([900.0, 400.0]),
        detail_condition=torch.full((2, 144, 4, 4), detail_value),
        detail_tokens=torch.full((2, 4, 16), detail_value),
        detail_token_mask=torch.tensor([[True, True, False, False], [True, False, False, False]]),
        detail_valid=torch.tensor([True, False]),
    )


def _activate_heads(core: SharedRecurrentControlCore) -> None:
    with torch.no_grad():
        for branch in (core.geometry_branch, core.interaction_branch, core.detail_branch):
            for head in branch.zero_heads:
                head.weight.fill_(0.01)


def test_detail_branch_is_independent_zero_initialized_and_skips_invalid_rows() -> None:
    torch.manual_seed(4)
    core = _core()
    assert core.detail_branch.condition_embed.projection.in_channels == 144
    assert len(core.detail_branch.stage_adapters) == 6
    assert len(core.detail_branch.cross_norms) == 6
    assert len(core.detail_branch.zero_heads) == 6
    initial = core(**_core_inputs())
    assert initial.detail is not None
    assert all(torch.count_nonzero(value) == 0 for value in initial.detail)
    _activate_heads(core)
    inputs = _core_inputs(1.0)
    first = core(**inputs)
    changed = dict(inputs)
    changed["detail_condition"] = torch.full_like(inputs["detail_condition"], 8.0)
    changed["detail_tokens"] = torch.full_like(inputs["detail_tokens"], 8.0)
    second = core(**changed)
    assert all(torch.equal(a, b) for a, b in zip(first.geometry, second.geometry))
    assert all(torch.equal(a, b) for a, b in zip(first.interaction, second.interaction))
    assert any(not torch.equal(a, b) for a, b in zip(first.detail, second.detail))
    assert all(torch.count_nonzero(value[1]) == 0 for value in first.detail)


def test_missing_detail_skips_branch() -> None:
    core = _core()
    inputs = _core_inputs()
    without = {key: value for key, value in inputs.items() if not key.startswith("detail_")}
    assert core(**without).detail is None


def _residuals() -> BranchControlResiduals:
    geometry = tuple(torch.ones(2, 4, 3) for _ in range(6))
    interaction = tuple(torch.ones(2, 4, 3) * 2 for _ in range(6))
    detail = tuple(torch.ones(2, 4, 3) * 4 for _ in range(6))
    return BranchControlResiduals(geometry, interaction, (2, 2), detail=detail)


def test_per_sample_late_cosine_and_separate_region_strengths() -> None:
    controller = ControlStrengthController()
    masks = torch.zeros(2, 2, 3, 2, 2)
    masks[:, 0, 0, 0, 0] = 0.8
    masks[:, 1, 0, 0, 0] = 0.6
    masks[:, :, 1:, 1, 1] = 1.0
    output, diagnostics = controller(
        _residuals(), denoise_progress=torch.tensor([0.2, 1.0]),
        geometry_strength=0.0, interaction_strength=0.0,
        detail_strength=1.0, face_strength=0.5, hand_strength=0.25,
        detail_region_masks=masks,
    )
    assert torch.count_nonzero(output[0][0]) == 0
    assert output[0][1, 0, 0].item() == pytest.approx(4 * 0.8 * 0.5)
    assert output[0][1, 3, 0].item() == pytest.approx(4 * 0.25)
    assert output[0][1, 1:3].count_nonzero() == 0
    assert diagnostics["detail_schedule_multiplier"] == [0.0, 1.0]


@pytest.mark.parametrize(
    "detail_strength",
    (0.0, torch.tensor(0.0), torch.zeros(2)),
)
def test_detail_strength_zero_and_missing_detail_are_exact_old_parity(
    detail_strength,
) -> None:
    controller = ControlStrengthController()
    old = _residuals()
    no_detail = BranchControlResiduals(old.geometry, old.interaction, old.target_token_hw)
    expected, _ = controller(no_detail, denoise_progress=0.8)
    disabled, _ = controller(
        old, denoise_progress=0.8, detail_strength=detail_strength
    )
    assert all(torch.equal(a, b) for a, b in zip(expected, disabled))


@pytest.mark.parametrize(
    "invalid_strength",
    (torch.zeros(3), torch.zeros(2, 1), torch.empty(0)),
)
def test_all_zero_detail_strength_rejects_unsupported_tensor_shapes(
    invalid_strength,
) -> None:
    with pytest.raises(ValueError, match=r"shape \[B\]"):
        ControlStrengthController()(
            _residuals(),
            denoise_progress=0.8,
            detail_strength=invalid_strength,
        )


def test_mixed_batch_detail_strength_requires_masks_and_applies_per_sample() -> None:
    controller = ControlStrengthController()
    residuals = _residuals()
    strength = torch.tensor([0.0, 1.0])
    with pytest.raises(ValueError, match="detail_region_masks"):
        controller(residuals, denoise_progress=1.0, detail_strength=strength)
    expected, _ = controller(
        BranchControlResiduals(
            residuals.geometry, residuals.interaction, residuals.target_token_hw
        ),
        denoise_progress=1.0,
    )
    actual, _ = controller(
        residuals,
        denoise_progress=1.0,
        detail_strength=strength,
        detail_region_masks=torch.ones(2, 2, 3, 2, 2),
    )
    assert all(torch.equal(before[0], after[0]) for before, after in zip(expected, actual))
    assert all(not torch.equal(before[1], after[1]) for before, after in zip(expected, actual))


def test_prepared_control_preserves_optional_detail_and_cfg_order() -> None:
    detail = PreparedFaceHandDetailConditioning(
        detail_condition=torch.randn(2, 144, 4, 4),
        detail_tokens=torch.randn(2, 48, 8),
        detail_token_mask=torch.ones(2, 48, dtype=torch.bool),
        region_masks=torch.ones(2, 2, 3, 2, 2),
        region_valid=torch.ones(2, 2, 3, dtype=torch.bool),
        detail_valid=torch.ones(2, dtype=torch.bool),
    ).validate()
    from tests.test_deepgen_control_interface_v1 import _prepared
    old = _prepared(batch_size=2)
    prepared = replace(old, detail=detail).validate()
    expanded = prepared.expand_to_batch(4)
    assert expanded.detail is not None
    torch.testing.assert_close(expanded.detail.detail_condition[:2], detail.detail_condition)
    torch.testing.assert_close(expanded.detail.detail_condition[2:], detail.detail_condition)


def test_legacy_allowlist_strict_roundtrip_and_detail_only_freeze() -> None:
    from src.pose_control.v6.deepgen_adapter import UnifiedSMPLXAdapterV6
    legacy = UnifiedSMPLXAdapterV6.build_tiny(hidden_dim=16, geometry_channels=4)
    v63_state = {
        key: value.clone() for key, value in legacy.state_dict().items()
        if not key.startswith("detail_preparer.")
        and not key.startswith("control_interface.control_core.detail_branch.")
        and key != "control_interface.strength_controller.detail_log_group_scale"
    }
    loaded = UnifiedSMPLXAdapterV6.build_tiny(hidden_dim=16, geometry_channels=4)
    with pytest.raises(
        RuntimeError,
        match="V6.3 optimizer state is incompatible.*must not be restored",
    ):
        load_v63_checkpoint(
            loaded,
            {"adapter_state_dict": v63_state, "optimizer_state_dict": {"bad": True}},
        )
    result = load_v63_checkpoint(loaded, {"adapter_state_dict": v63_state})
    assert result.missing_keys and all("detail" in key for key in result.missing_keys)
    assert all(torch.count_nonzero(head.weight) == 0 for head in loaded.control_core.detail_zero_heads)
    corrupted = dict(v63_state)
    corrupted["corrupt.unexpected"] = torch.tensor(1)
    with pytest.raises(RuntimeError, match="unexpected"):
        load_v63_checkpoint(loaded, corrupted)
    checkpoint = build_v64_checkpoint(loaded)
    assert checkpoint["metadata"]["architecture_version"] == "v6.4"
    assert checkpoint["metadata"]["branch_names"] == ["geometry", "interaction", "detail"]
    assert checkpoint["metadata"]["detail_schema_version"] == DETAIL_SCHEMA_VERSION
    roundtrip = UnifiedSMPLXAdapterV6.build_tiny(hidden_dim=16, geometry_channels=4)
    load_v64_checkpoint(roundtrip, checkpoint)
    for key, value in loaded.state_dict().items():
        assert torch.equal(value, roundtrip.state_dict()[key])
    freeze_for_detail_training(roundtrip)
    trainable = {name for name, value in roundtrip.named_parameters() if value.requires_grad}
    assert trainable
    assert all(
        name.startswith("detail_preparer.")
        or name.startswith("control_interface.control_core.detail_branch.")
        or name == "control_interface.strength_controller.detail_log_group_scale"
        for name in trainable
    )
    assert len([key for key in roundtrip.state_dict() if "person_token_binder" in key]) == 2


def test_detail_only_optimizer_step_keeps_every_v63_parameter_byte_identical() -> None:
    from src.pose_control.v6.deepgen_adapter import UnifiedSMPLXAdapterV6

    backbone = torch.nn.Linear(4, 4)
    model = UnifiedSMPLXAdapterV6.build_tiny(
        hidden_dim=16,
        geometry_channels=4,
        context_input_dim=12,
        pooled_input_dim=10,
        deepgen_backbone=backbone,
    )
    freeze_for_detail_training(model)
    old = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
        if not value.requires_grad
    }
    with torch.no_grad():
        model.control_core.detail_zero_heads[0].weight.fill_(0.01)
    optimizer = torch.optim.SGD(
        [value for value in model.parameters() if value.requires_grad], lr=1e-3
    )
    output = model.control_core(**_core_inputs())
    assert output.detail is not None
    sum(value.square().mean() for value in output.detail).backward()
    head_grad = model.control_core.detail_zero_heads[0].weight.grad
    assert head_grad is not None
    assert torch.isfinite(head_grad).all() and torch.count_nonzero(head_grad) > 0
    optimizer.step()
    for name, value in model.named_parameters():
        if name in old:
            assert torch.equal(value, old[name]), name
    assert all(parameter.grad is None for parameter in backbone.parameters())


def test_adapter_prepares_null_references_without_duplicate_person_binder() -> None:
    from src.pose_control.v6.conditions import AdapterIdentityCondition
    from src.pose_control.v6.deepgen_adapter import UnifiedSMPLXAdapterV6
    from tests.test_deepgen_control_interface_v1 import _control_state
    from tests.test_v6_detail_contracts import make_condition

    model = UnifiedSMPLXAdapterV6.build_tiny(hidden_dim=32, geometry_channels=16)
    prepared = model.prepare_control(
        state=_control_state(dual=True),
        identity_condition=AdapterIdentityCondition(
            source_person_latents=torch.randn(1, 2, 16, 8, 12),
            source_indices=torch.tensor([[2, 5]]),
        ),
        source_scene_latents=torch.randn(1, 16, 8, 12),
        target_latent_hw=(8, 12),
        detail_condition=make_condition(1),
    )
    assert prepared.detail is not None
    assert prepared.detail.detail_condition.shape == (1, 144, 8, 12)
    assert prepared.detail.detail_tokens.shape == (1, 48, 32)
    assert len([key for key in model.state_dict() if "person_token_binder" in key]) == 2


def _state_snapshot(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _assert_state_unchanged(model, snapshot) -> None:
    assert snapshot.keys() == model.state_dict().keys()
    for key, value in model.state_dict().items():
        assert torch.equal(value, snapshot[key]), key


@pytest.mark.parametrize("corruption", ("unexpected", "missing", "shape", "non_tensor"))
def test_v63_loader_preflight_failures_are_atomic(corruption: str) -> None:
    from src.pose_control.v6.deepgen_adapter import UnifiedSMPLXAdapterV6

    source = UnifiedSMPLXAdapterV6.build_tiny(hidden_dim=16, geometry_channels=4)
    state = {
        key: value.detach().clone()
        for key, value in source.state_dict().items()
        if "detail" not in key
    }
    key = next(iter(state))
    if corruption == "unexpected":
        state["unexpected.key"] = torch.zeros(1)
    elif corruption == "missing":
        state.pop(key)
    elif corruption == "shape":
        state[key] = torch.zeros(1)
    else:
        state[key] = "not-a-tensor"
    target = UnifiedSMPLXAdapterV6.build_tiny(hidden_dim=16, geometry_channels=4)
    snapshot = _state_snapshot(target)
    with pytest.raises(RuntimeError):
        load_v63_checkpoint(target, {"adapter_state_dict": state})
    _assert_state_unchanged(target, snapshot)


@pytest.mark.parametrize("corruption", ("unexpected", "missing", "shape", "non_tensor"))
def test_v64_loader_preflight_failures_are_atomic(corruption: str) -> None:
    from src.pose_control.v6.deepgen_adapter import UnifiedSMPLXAdapterV6

    source = UnifiedSMPLXAdapterV6.build_tiny(hidden_dim=16, geometry_channels=4)
    checkpoint = build_v64_checkpoint(source)
    state = dict(checkpoint["adapter_state_dict"])
    key = next(iter(state))
    if corruption == "unexpected":
        state["unexpected.key"] = torch.zeros(1)
    elif corruption == "missing":
        state.pop(key)
    elif corruption == "shape":
        state[key] = torch.zeros(1)
    else:
        state[key] = "not-a-tensor"
    checkpoint["adapter_state_dict"] = state
    target = UnifiedSMPLXAdapterV6.build_tiny(hidden_dim=16, geometry_channels=4)
    snapshot = _state_snapshot(target)
    with pytest.raises(RuntimeError):
        load_v64_checkpoint(target, checkpoint)
    _assert_state_unchanged(target, snapshot)


def test_v63_loader_rejects_v64_metadata_and_detail_keys() -> None:
    from src.pose_control.v6.deepgen_adapter import UnifiedSMPLXAdapterV6

    model = UnifiedSMPLXAdapterV6.build_tiny(hidden_dim=16, geometry_channels=4)
    legacy = {key: value for key, value in model.state_dict().items() if "detail" not in key}
    with pytest.raises(RuntimeError, match="V6.4 metadata"):
        load_v63_checkpoint(
            model,
            {"adapter_state_dict": legacy, "metadata": {"architecture_version": "v6.4"}},
        )
    contaminated = dict(legacy)
    detail_key = next(key for key in model.state_dict() if "detail" in key)
    contaminated[detail_key] = model.state_dict()[detail_key]
    with pytest.raises(RuntimeError, match="detail-only"):
        load_v63_checkpoint(model, contaminated)


def test_v64_checkpoint_clones_tensors_and_rejects_reserved_extras() -> None:
    from src.pose_control.v6.deepgen_adapter import UnifiedSMPLXAdapterV6

    model = UnifiedSMPLXAdapterV6.build_tiny(hidden_dim=16, geometry_channels=4)
    checkpoint = build_v64_checkpoint(model)
    key, saved = next(iter(checkpoint["adapter_state_dict"].items()))
    before = saved.clone()
    with torch.no_grad():
        model.state_dict()[key].add_(1)
    assert torch.equal(saved, before)
    for reserved in ("metadata", "adapter_state_dict"):
        with pytest.raises(ValueError, match="reserved"):
            build_v64_checkpoint(model, **{reserved: {}})


def test_detail_context_physically_omits_masked_tokens() -> None:
    torch.manual_seed(23)
    core = _core()
    _activate_heads(core)
    inputs = _core_inputs()
    valid_tokens = inputs["detail_tokens"][:, :2].clone()
    omitted = dict(inputs)
    omitted["detail_tokens"] = valid_tokens
    omitted["detail_token_mask"] = torch.ones(2, 2, dtype=torch.bool)
    padded = dict(inputs)
    padded["detail_tokens"] = torch.cat(
        (valid_tokens, torch.full((2, 3, 16), 1e6)), dim=1
    )
    padded["detail_token_mask"] = torch.tensor(
        [[True, True, False, False, False], [True, True, False, False, False]]
    )
    expected = core(**omitted)
    actual = core(**padded)
    assert expected.detail is not None and actual.detail is not None
    assert all(torch.equal(a, b) for a, b in zip(expected.detail, actual.detail))


@pytest.mark.parametrize("progress", (torch.tensor(float("nan")), torch.tensor(float("inf"))))
def test_tensor_progress_must_be_finite(progress) -> None:
    with pytest.raises(ValueError, match="finite"):
        ControlStrengthController()(_residuals(), denoise_progress=progress)


def test_detail_region_mask_validation_is_explicit() -> None:
    controller = ControlStrengthController()
    base = torch.ones(2, 2, 3, 2, 2)
    invalid = (
        base.bool(),
        base.clone().index_put_((torch.tensor([0]),) * 5, torch.tensor(float("nan"))),
        base * 2,
        torch.empty(2, 2, 3, 2, 2, device="meta"),
    )
    for masks in invalid:
        with pytest.raises(ValueError):
            controller(
                _residuals(),
                denoise_progress=1.0,
                detail_region_masks=masks,
            )


def test_detail_identity_must_match_state_and_identity_condition() -> None:
    from src.pose_control.v6.conditions import AdapterIdentityCondition
    from src.pose_control.v6.deepgen_adapter import UnifiedSMPLXAdapterV6
    from tests.test_deepgen_control_interface_v1 import _control_state
    from tests.test_v6_detail_contracts import make_condition

    model = UnifiedSMPLXAdapterV6.build_tiny(hidden_dim=32, geometry_channels=16)
    identity = AdapterIdentityCondition(
        source_person_latents=torch.randn(1, 2, 16, 8, 12),
        source_indices=torch.tensor([[2, 5]]),
    )
    mismatch = make_condition(1)
    mismatch.source_indices[0, 1] = 6
    with pytest.raises(ValueError, match="source_indices"):
        model.prepare_control(
            state=_control_state(dual=True),
            identity_condition=identity,
            source_scene_latents=torch.randn(1, 16, 8, 12),
            target_latent_hw=(8, 12),
            detail_condition=mismatch,
        )
    with pytest.raises(ValueError, match="detail-valid person"):
        model.prepare_control(
            state=_control_state(dual=False),
            identity_condition=identity,
            source_scene_latents=torch.randn(1, 16, 8, 12),
            target_latent_hw=(8, 12),
            detail_condition=make_condition(1),
        )


def test_detail_compute_fast_path_reports_only_active_rows() -> None:
    from src.pose_control.v6.conditions import AdapterIdentityCondition
    from src.pose_control.v6.deepgen_adapter import UnifiedSMPLXAdapterV6
    from tests.test_deepgen_control_interface_v1 import _control_state
    from tests.test_v6_detail_contracts import make_condition

    model = UnifiedSMPLXAdapterV6.build_tiny(hidden_dim=32, geometry_channels=16).eval()
    prepared = model.prepare_control(
        state=_control_state(dual=True),
        identity_condition=AdapterIdentityCondition(
            source_person_latents=torch.randn(1, 2, 16, 8, 12),
            source_indices=torch.tensor([[2, 5]]),
        ),
        source_scene_latents=torch.randn(1, 16, 8, 12),
        target_latent_hw=(8, 12),
        detail_condition=make_condition(1),
    )

    def run(batch: int, progress, strength):
        output = model(
            target_latents=torch.randn(batch, 16, 8, 12),
            prepared=prepared,
            cond_hidden_states=None,
            encoder_hidden_states=torch.randn(batch, 5, 24),
            pooled_projections=torch.randn(batch, 20),
            timestep=torch.full((batch,), 500.0),
            denoise_progress=progress,
            detail_strength=strength,
        )
        return output.diagnostics

    assert run(1, 0.2, 1.0)["detail_active_rows"] == 0
    assert run(1, 1.0, 0.0)["detail_executed"] is False
    assert run(2, 1.0, torch.zeros(2))["detail_active_rows"] == 0
    mixed = run(2, 1.0, torch.tensor([0.0, 1.0]))
    assert mixed["detail_active_rows"] == 1
    assert mixed["detail_executed"] is True
