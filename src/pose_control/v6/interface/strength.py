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

    def multiplier(self, progress: float | torch.Tensor) -> float | torch.Tensor:
        if torch.is_tensor(progress):
            if progress.ndim > 1 or not progress.is_floating_point():
                raise ValueError("denoise progress tensor must be scalar or have shape [B]")
            if not torch.isfinite(progress).all():
                raise ValueError("denoise progress tensor must contain only finite values")
            if torch.any((progress < 0) | (progress > 1)):
                raise ValueError("denoise progress must lie within [0,1]")
            active = (progress >= self.start_progress) & (progress <= self.end_progress)
            if self.curve == "constant":
                value = torch.full_like(progress, self.start_multiplier)
            else:
                position = (progress - self.start_progress) / (
                    self.end_progress - self.start_progress
                )
                if self.curve == "cosine":
                    position = 0.5 * (1.0 - torch.cos(torch.pi * position))
                value = self.start_multiplier + position * (
                    self.end_multiplier - self.start_multiplier
                )
            return torch.where(active, value, torch.zeros_like(value))
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
        return float(self.start_multiplier + position * (self.end_multiplier - self.start_multiplier))


class ControlStrengthController(nn.Module):
    """Apply positive per-group gates and denoising schedules per branch."""

    def __init__(
        self,
        *,
        num_control_groups: int = 6,
        geometry_schedule: StrengthScheduleConfig | None = None,
        interaction_schedule: StrengthScheduleConfig | None = None,
        detail_schedule: StrengthScheduleConfig | None = None,
    ) -> None:
        super().__init__()
        if num_control_groups != 6:
            raise ValueError("DeepGen V6.3 defines exactly six control groups")
        self.num_control_groups = num_control_groups
        self.geometry_schedule = geometry_schedule or StrengthScheduleConfig()
        self.interaction_schedule = interaction_schedule or StrengthScheduleConfig()
        self.detail_schedule = detail_schedule or StrengthScheduleConfig(
            curve="cosine",
            start_progress=0.35,
            end_progress=1.0,
            start_multiplier=0.0,
            end_multiplier=1.0,
        )
        self.geometry_log_group_scale = nn.Parameter(torch.zeros(num_control_groups))
        self.interaction_log_group_scale = nn.Parameter(torch.zeros(num_control_groups))
        self.detail_log_group_scale = nn.Parameter(torch.zeros(num_control_groups))

    @staticmethod
    def _sample_gate(value, reference: torch.Tensor) -> torch.Tensor | float:
        if not torch.is_tensor(value):
            return float(value)
        value = value.to(device=reference.device, dtype=reference.dtype)
        if value.ndim == 0:
            return value
        if value.shape != (reference.shape[0],):
            raise ValueError("per-sample strength or progress must have shape [B]")
        return value[:, None, None]

    def forward(
        self,
        residuals: BranchControlResiduals,
        *,
        denoise_progress: float | torch.Tensor,
        geometry_strength: float | torch.Tensor = 1.0,
        interaction_strength: float | torch.Tensor = 0.8,
        detail_strength: float | torch.Tensor = 1.0,
        face_strength: float | torch.Tensor = 1.0,
        hand_strength: float | torch.Tensor = 1.0,
        detail_region_masks: torch.Tensor | None = None,
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, object]]:
        residuals.validate()
        geo_schedule = self.geometry_schedule.multiplier(denoise_progress)
        int_schedule = self.interaction_schedule.multiplier(denoise_progress)
        detail_schedule = self.detail_schedule.multiplier(denoise_progress)
        geo_gates = self.geometry_log_group_scale.exp()
        int_gates = self.interaction_log_group_scale.exp()
        detail_gates = self.detail_log_group_scale.exp()
        reference = residuals.geometry[0]
        if detail_region_masks is not None:
            expected = (reference.shape[0], 2, 3, *residuals.target_token_hw)
            if tuple(detail_region_masks.shape) != expected:
                raise ValueError(f"detail_region_masks must have shape {expected}")
            if not detail_region_masks.is_floating_point():
                raise ValueError("detail_region_masks must use a floating dtype")
            if detail_region_masks.device != reference.device:
                raise ValueError("detail_region_masks must be on the residual device")
            if not torch.isfinite(detail_region_masks).all():
                raise ValueError("detail_region_masks must contain only finite values")
            if torch.any((detail_region_masks < 0) | (detail_region_masks > 1)):
                raise ValueError("detail_region_masks must be bounded in [0,1]")
        geo_schedule = self._sample_gate(geo_schedule, reference)
        int_schedule = self._sample_gate(int_schedule, reference)
        detail_schedule = self._sample_gate(detail_schedule, reference)
        geometry_strength = self._sample_gate(geometry_strength, reference)
        interaction_strength = self._sample_gate(interaction_strength, reference)
        output = tuple(
            geometry
            * (geometry_strength * geo_schedule * geo_gates[index])
            + interaction
            * (interaction_strength * int_schedule * int_gates[index])
            for index, (geometry, interaction) in enumerate(
                zip(residuals.geometry, residuals.interaction)
            )
        )
        detail_strength = self._sample_gate(detail_strength, reference)
        if torch.is_tensor(detail_strength):
            zero_detail_strength = bool(torch.count_nonzero(detail_strength) == 0)
        else:
            zero_detail_strength = float(detail_strength) == 0.0
        detail_is_disabled = residuals.detail is None or zero_detail_strength
        if not detail_is_disabled:
            if detail_region_masks is None:
                raise ValueError("detail_region_masks are required when detail residuals are active")
            face = detail_region_masks[:, :, 0].amax(dim=1).flatten(1)
            hands = detail_region_masks[:, :, 1:].amax(dim=(1, 2)).flatten(1)
            face_strength = self._sample_gate(face_strength, reference)
            hand_strength = self._sample_gate(hand_strength, reference)
            face_gate = face[:, :, None] * face_strength
            hand_gate = hands[:, :, None] * hand_strength
            region_gate = torch.maximum(face_gate, hand_gate)
            output = tuple(
                current + detail * region_gate * detail_strength * detail_schedule * detail_gates[index]
                for index, (current, detail) in enumerate(zip(output, residuals.detail))
            )

        def diagnostic(value):
            if torch.is_tensor(value):
                if value.ndim == 0:
                    return float(value.detach().cpu())
                flat = value.detach().reshape(value.shape[0], -1)[:, 0].cpu().tolist()
                return flat[0] if len(flat) == 1 else flat
            return float(value)
        diagnostics: dict[str, object] = {
            "denoise_progress": diagnostic(denoise_progress),
            "geometry_schedule_multiplier": diagnostic(geo_schedule),
            "interaction_schedule_multiplier": diagnostic(int_schedule),
            "detail_schedule_multiplier": diagnostic(detail_schedule),
            "geometry_group_scales": geo_gates.detach().cpu().tolist(),
            "interaction_group_scales": int_gates.detach().cpu().tolist(),
            "detail_group_scales": detail_gates.detach().cpu().tolist(),
        }
        return output, diagnostics
