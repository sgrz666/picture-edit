from __future__ import annotations

import math

import pytest
import torch

from src.pose_control.v6.interface import (
    BranchControlResiduals,
    ControlStrengthController,
    DeepGenControlOutput,
    StrengthScheduleConfig,
)
from src.pose_control.v6.control_core import SharedRecurrentControlCore


def _branch_residuals(
    *, batch_size: int = 2, token_hw: tuple[int, int] = (4, 6), hidden_dim: int = 32
) -> BranchControlResiduals:
    token_count = math.prod(token_hw)
    geometry = tuple(
        torch.full((batch_size, token_count, hidden_dim), float(index + 1))
        for index in range(6)
    )
    interaction = tuple(
        torch.full((batch_size, token_count, hidden_dim), float(10 + index))
        for index in range(6)
    )
    return BranchControlResiduals(
        geometry=geometry,
        interaction=interaction,
        target_token_hw=token_hw,
    ).validate()


def test_branch_control_residuals_validate_dynamic_token_shape() -> None:
    residuals = _branch_residuals(token_hw=(3, 5))
    assert residuals.target_token_count == 15
    assert residuals.batch_size == 2
    assert residuals.hidden_dim == 32

    malformed = BranchControlResiduals(
        geometry=residuals.geometry,
        interaction=residuals.interaction[:-1],
        target_token_hw=(3, 5),
    )
    with pytest.raises(ValueError, match="six"):
        malformed.validate()


@pytest.mark.parametrize(
    ("curve", "progress", "expected"),
    [
        ("constant", 0.5, 1.0),
        ("linear", 0.5, 0.5),
        ("cosine", 0.5, 0.5),
    ],
)
def test_strength_schedule_curves(curve: str, progress: float, expected: float) -> None:
    schedule = StrengthScheduleConfig(
        curve=curve,
        start_progress=0.0,
        end_progress=1.0,
        start_multiplier=1.0,
        end_multiplier=0.0,
    )
    assert schedule.multiplier(progress) == pytest.approx(expected)


def test_strength_schedule_is_zero_outside_its_window() -> None:
    schedule = StrengthScheduleConfig(start_progress=0.2, end_progress=0.8)
    assert schedule.multiplier(0.1) == 0.0
    assert schedule.multiplier(0.9) == 0.0
    with pytest.raises(ValueError, match="progress"):
        schedule.multiplier(1.1)


def test_strength_controller_scales_branches_and_groups_independently() -> None:
    controller = ControlStrengthController(num_control_groups=6)
    raw = _branch_residuals(batch_size=1, token_hw=(2, 2), hidden_dim=4)
    with torch.no_grad():
        controller.geometry_log_group_scale[1] = math.log(2.0)
        controller.interaction_log_group_scale[1] = math.log(0.5)

    output, diagnostics = controller(
        raw,
        denoise_progress=0.5,
        geometry_strength=1.0,
        interaction_strength=0.8,
    )

    torch.testing.assert_close(output[0], raw.geometry[0] + raw.interaction[0] * 0.8)
    torch.testing.assert_close(
        output[1], raw.geometry[1] * 2.0 + raw.interaction[1] * 0.4
    )
    assert diagnostics["geometry_schedule_multiplier"] == pytest.approx(1.0)
    assert diagnostics["interaction_schedule_multiplier"] == pytest.approx(1.0)


def test_deepgen_control_output_requires_six_aligned_residuals() -> None:
    residuals = _branch_residuals(batch_size=1).geometry
    output = DeepGenControlOutput(
        block_controlnet_hidden_states=residuals,
        diagnostics={"target_token_count": 24},
    ).validate()
    assert len(output.block_controlnet_hidden_states) == 6
    with pytest.raises(ValueError, match="six"):
        DeepGenControlOutput(residuals[:-1], {}).validate()


def _tiny_core() -> SharedRecurrentControlCore:
    return SharedRecurrentControlCore.build_tiny(
        hidden_dim=32,
        condition_channels=32,
        context_input_dim=24,
        pooled_input_dim=20,
        patch_size=2,
        num_stages=6,
        rank=8,
    )


def _core_inputs(*, batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "target_latents": torch.randn(batch_size, 16, 8, 12),
        "geometry_condition": torch.randn(batch_size, 32, 8, 12),
        "interaction_condition": torch.randn(batch_size, 32, 8, 12),
        "interaction_valid": torch.tensor([True, False])[:batch_size],
        "adapter_tokens": torch.randn(batch_size, 5, 32),
        "adapter_token_mask": torch.ones(batch_size, 5, dtype=torch.bool),
        "encoder_hidden_states": torch.randn(batch_size, 7, 24),
        "pooled_projections": torch.randn(batch_size, 20),
        "timestep": torch.full((batch_size,), 500),
    }


def test_dynamic_core_has_independent_zero_heads_per_branch() -> None:
    core = _tiny_core()
    assert len(core.geometry_zero_heads) == 6
    assert len(core.interaction_zero_heads) == 6
    all_heads = [*core.geometry_zero_heads, *core.interaction_zero_heads]
    assert len({id(head) for head in all_heads}) == 12
    assert all(torch.count_nonzero(head.weight) == 0 for head in all_heads)
    assert all(torch.count_nonzero(head.bias) == 0 for head in all_heads)

    residuals = core(**_core_inputs())
    assert residuals.target_token_hw == (4, 6)
    assert all(torch.count_nonzero(value) == 0 for value in residuals.geometry)
    assert all(torch.count_nonzero(value) == 0 for value in residuals.interaction)


def test_dynamic_core_runs_interaction_only_for_valid_dual_rows() -> None:
    torch.manual_seed(19)
    core = _tiny_core().eval()
    with torch.no_grad():
        core.geometry_zero_heads[0].weight.copy_(torch.eye(32))
        core.interaction_zero_heads[0].weight.copy_(torch.eye(32))
    residuals = core(**_core_inputs())
    assert torch.count_nonzero(residuals.geometry[0]) > 0
    assert torch.count_nonzero(residuals.interaction[0][0]) > 0
    assert torch.count_nonzero(residuals.interaction[0][1]) == 0


def test_dynamic_core_is_sensitive_to_latent_text_and_timestep() -> None:
    torch.manual_seed(23)
    core = _tiny_core().eval()
    with torch.no_grad():
        core.geometry_zero_heads[0].weight.copy_(torch.eye(32))
    baseline_inputs = _core_inputs(batch_size=1)
    baseline_inputs["interaction_valid"] = torch.tensor([False])
    baseline = core(**baseline_inputs).geometry[0]
    for field in ("target_latents", "encoder_hidden_states", "timestep"):
        changed = {key: value.clone() for key, value in baseline_inputs.items()}
        if field == "timestep":
            changed[field] = changed[field] + 100
        else:
            changed[field] = changed[field] + 1
        assert not torch.allclose(baseline, core(**changed).geometry[0])
