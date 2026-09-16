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
