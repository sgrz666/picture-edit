from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from .outputs import BranchControlResiduals


@dataclass(frozen=True)
class StrengthScheduleConfig:
    curve: Literal["constant", "linear", "cosine"] = "constant"
    start_progress: float = 0.0
    end_progress: float = 1.0
    start_multiplier: float = 1.0
    end_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.curve not in {"constant", "linear", "cosine"}:
            raise ValueError("curve must be constant, linear, or cosine")
        if not 0.0 <= self.start_progress <= self.end_progress <= 1.0:
            raise ValueError("schedule progress window must lie within [0,1]")
        if self.start_progress == self.end_progress:
            raise ValueError("schedule progress window must have positive width")

    def multiplier(self, progress: float) -> float:
        progress = float(progress)
        if not 0.0 <= progress <= 1.0:
            raise ValueError("denoise progress must lie within [0,1]")
        if progress < self.start_progress or progress > self.end_progress:
            return 0.0
        if self.curve == "constant":
            return float(self.start_multiplier)
        position = (progress - self.start_progress) / (
            self.end_progress - self.start_progress
        )
        if self.curve == "cosine":
            position = 0.5 * (1.0 - math.cos(math.pi * position))
        return float(
            self.start_multiplier
            + position * (self.end_multiplier - self.start_multiplier)
        )


class ControlStrengthController(nn.Module):
    """Apply positive per-group gates and denoising schedules per branch."""

    def __init__(
        self,
        *,
        num_control_groups: int = 6,
        geometry_schedule: StrengthScheduleConfig | None = None,
        interaction_schedule: StrengthScheduleConfig | None = None,
    ) -> None:
        super().__init__()
        if num_control_groups != 6:
            raise ValueError("DeepGen V6.3 defines exactly six control groups")
        self.num_control_groups = num_control_groups
        self.geometry_schedule = geometry_schedule or StrengthScheduleConfig()
        self.interaction_schedule = interaction_schedule or StrengthScheduleConfig()
        self.geometry_log_group_scale = nn.Parameter(torch.zeros(num_control_groups))
        self.interaction_log_group_scale = nn.Parameter(torch.zeros(num_control_groups))

    def forward(
        self,
        residuals: BranchControlResiduals,
        *,
        denoise_progress: float,
        geometry_strength: float = 1.0,
        interaction_strength: float = 0.8,
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, object]]:
        residuals.validate()
        geo_schedule = self.geometry_schedule.multiplier(denoise_progress)
        int_schedule = self.interaction_schedule.multiplier(denoise_progress)
        geo_gates = self.geometry_log_group_scale.exp()
        int_gates = self.interaction_log_group_scale.exp()
        output = tuple(
            geometry
            * (float(geometry_strength) * geo_schedule * geo_gates[index])
            + interaction
            * (float(interaction_strength) * int_schedule * int_gates[index])
            for index, (geometry, interaction) in enumerate(
                zip(residuals.geometry, residuals.interaction)
            )
        )
        diagnostics: dict[str, object] = {
            "denoise_progress": float(denoise_progress),
            "geometry_schedule_multiplier": geo_schedule,
            "interaction_schedule_multiplier": int_schedule,
            "geometry_group_scales": geo_gates.detach().cpu().tolist(),
            "interaction_group_scales": int_gates.detach().cpu().tolist(),
        }
        return output, diagnostics
