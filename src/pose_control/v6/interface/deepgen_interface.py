from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..control_core import SharedRecurrentControlCore
from .outputs import (
    BranchControlResiduals,
    DeepGenControlOutput,
    PreparedControlConditioning,
)
from .strength import ControlStrengthController, StrengthScheduleConfig


def latent_token_count(latent: torch.Tensor, patch_size: int) -> int:
    if latent.ndim not in (3, 4):
        raise ValueError("latent must have shape [C,H,W] or [B,C,H,W]")
    height, width = latent.shape[-2:]
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"latent size {(height, width)} is not divisible by patch_size={patch_size}"
        )
    return (height // patch_size) * (width // patch_size)


def align_target_residuals(
    residuals: Sequence[torch.Tensor],
    *,
    target_latents: torch.Tensor,
    cond_hidden_states: Optional[Sequence[Sequence[torch.Tensor]]],
    patch_size: int,
) -> tuple[torch.Tensor, ...]:
    """Align target-only controls to DeepGen's target/source/padding sequence."""

    target_tokens = latent_token_count(target_latents, patch_size)
    batch_size = target_latents.shape[0]
    full_tokens = target_tokens
    if cond_hidden_states is not None:
        if len(cond_hidden_states) != batch_size:
            raise ValueError("cond_hidden_states batch does not match target latents")
        lengths = [
            target_tokens
            + sum(latent_token_count(reference, patch_size) for reference in references)
            for references in cond_hidden_states
        ]
        full_tokens = max(lengths)
    aligned = []
    for residual in residuals:
        if residual.shape[:2] != (batch_size, target_tokens):
            raise ValueError("control residual must contain target tokens only")
        padding = full_tokens - target_tokens
        if padding:
            residual = torch.cat(
                (
                    residual,
                    residual.new_zeros(batch_size, padding, residual.shape[-1]),
                ),
                dim=1,
            )
        aligned.append(residual)
    return tuple(aligned)


class DeepGenControlInterface(nn.Module):
    """Build, scale, and align dynamic Adapter residuals for frozen DeepGen."""

    def __init__(
        self,
        control_core: SharedRecurrentControlCore,
        *,
        num_transformer_layers: int = 24,
        context_pre_only_blocks: tuple[int, ...] | None = None,
        geometry_schedule: StrengthScheduleConfig | None = None,
        interaction_schedule: StrengthScheduleConfig | None = None,
        detail_schedule: StrengthScheduleConfig | None = None,
    ) -> None:
        super().__init__()
        if num_transformer_layers <= 0 or num_transformer_layers % control_core.num_stages:
            raise ValueError("transformer layers must divide evenly across control groups")
        self.control_core = control_core
        self.num_transformer_layers = num_transformer_layers
        self.context_pre_only_blocks = (
            (num_transformer_layers - 1,)
            if context_pre_only_blocks is None
            else tuple(context_pre_only_blocks)
        )
        self.strength_controller = ControlStrengthController(
            num_control_groups=control_core.num_stages,
            geometry_schedule=geometry_schedule,
            interaction_schedule=interaction_schedule,
            detail_schedule=detail_schedule,
        )

    @property
    def active_block_mapping(self) -> tuple[tuple[int, ...], ...]:
        interval = self.num_transformer_layers / self.control_core.num_stages
        groups: list[list[int]] = [
            [] for _ in range(self.control_core.num_stages)
        ]
        for block in range(self.num_transformer_layers):
            if block in self.context_pre_only_blocks:
                continue
            groups[int(block / interval)].append(block)
        return tuple(tuple(group) for group in groups)

    def build_raw_residuals(
        self,
        *,
        target_latents: torch.Tensor,
        prepared: PreparedControlConditioning,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
        joint_attention_kwargs: Optional[dict] = None,
        detail_active: torch.Tensor | None = None,
    ) -> BranchControlResiduals:
        prepared = prepared.validate().expand_to_batch(target_latents.shape[0])
        if detail_active is not None:
            if detail_active.shape != (target_latents.shape[0],) or detail_active.dtype != torch.bool:
                raise ValueError("detail_active must have shape [B] and dtype bool")
            detail_active = detail_active.to(device=target_latents.device)
        target_size = target_latents.shape[-2:]
        geometry_condition = F.interpolate(
            prepared.geometry_condition,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        interaction_condition = F.interpolate(
            prepared.interaction_condition,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        target_hw = (
            target_size[0] // self.control_core.patch_size,
            target_size[1] // self.control_core.patch_size,
        )
        token_count = target_hw[0] * target_hw[1]
        batch_size = target_latents.shape[0]
        geometry = tuple(
            target_latents.new_zeros(batch_size, token_count, self.control_core.hidden_dim)
            for _ in range(self.control_core.num_stages)
        )
        interaction = tuple(torch.zeros_like(value) for value in geometry)
        detail = (
            None
            if prepared.detail is None
            else tuple(torch.zeros_like(value) for value in geometry)
        )

        detail_condition = None
        if prepared.detail is not None:
            detail_condition = F.interpolate(
                prepared.detail.detail_condition,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        for token_pattern in torch.unique(prepared.adapter_token_mask, dim=0):
            indices = (
                (prepared.adapter_token_mask == token_pattern)
                .all(dim=1)
                .nonzero(as_tuple=False)
                .flatten()
            )
            group = self.control_core(
                target_latents=target_latents.index_select(0, indices),
                geometry_condition=geometry_condition.index_select(0, indices),
                interaction_condition=interaction_condition.index_select(0, indices),
                interaction_valid=prepared.interaction_valid.index_select(0, indices),
                adapter_tokens=prepared.adapter_tokens.index_select(0, indices)[
                    :, token_pattern
                ],
                adapter_token_mask=prepared.adapter_token_mask.index_select(0, indices)[
                    :, token_pattern
                ],
                encoder_hidden_states=encoder_hidden_states.index_select(0, indices),
                pooled_projections=pooled_projections.index_select(0, indices),
                timestep=timestep.index_select(0, indices),
                joint_attention_kwargs=joint_attention_kwargs,
                detail_condition=None if detail_condition is None else detail_condition.index_select(0, indices),
                detail_tokens=None if prepared.detail is None else prepared.detail.detail_tokens.index_select(0, indices),
                detail_token_mask=None if prepared.detail is None else prepared.detail.detail_token_mask.index_select(0, indices),
                detail_valid=(
                    None
                    if prepared.detail is None
                    else (
                        prepared.detail.detail_valid
                        if detail_active is None
                        else prepared.detail.detail_valid & detail_active
                    ).index_select(0, indices)
                ),
            )
            geometry = tuple(
                current.index_copy(0, indices, update)
                for current, update in zip(geometry, group.geometry)
            )
            interaction = tuple(
                current.index_copy(0, indices, update)
                for current, update in zip(interaction, group.interaction)
            )
            if detail is not None:
                assert group.detail is not None
                detail = tuple(
                    current.index_copy(0, indices, update)
                    for current, update in zip(detail, group.detail)
                )
        return BranchControlResiduals(
            geometry=geometry,
            interaction=interaction,
            target_token_hw=target_hw,
            detail=detail,
        ).validate()

    def forward(
        self,
        *,
        target_latents: torch.Tensor,
        prepared: PreparedControlConditioning,
        cond_hidden_states: Optional[Sequence[Sequence[torch.Tensor]]],
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
        denoise_progress: float | torch.Tensor,
        geometry_strength: float | torch.Tensor = 1.0,
        interaction_strength: float | torch.Tensor = 0.8,
        detail_strength: float | torch.Tensor = 1.0,
        face_strength: float | torch.Tensor = 1.0,
        hand_strength: float | torch.Tensor = 1.0,
        joint_attention_kwargs: Optional[dict] = None,
    ) -> DeepGenControlOutput:
        expanded = prepared.validate().expand_to_batch(target_latents.shape[0])
        detail_active = target_latents.new_zeros(
            target_latents.shape[0], dtype=torch.bool
        )
        if expanded.detail is not None:
            reference = target_latents.new_empty(target_latents.shape[0], 1, 1)

            def per_sample(value) -> torch.Tensor:
                gate = self.strength_controller._sample_gate(value, reference)
                if not torch.is_tensor(gate):
                    return reference.new_full((reference.shape[0],), gate)
                if gate.ndim == 0:
                    return gate.expand(reference.shape[0])
                return gate[:, 0, 0]

            schedule = per_sample(
                self.strength_controller.detail_schedule.multiplier(
                    denoise_progress
                )
            )
            detail_gate = per_sample(detail_strength)
            face_gate = per_sample(face_strength)
            hand_gate = per_sample(hand_strength)
            face_valid = expanded.detail.region_valid[:, :, 0].any(dim=1)
            hand_valid = expanded.detail.region_valid[:, :, 1:].any(dim=(1, 2))
            region_active = (face_valid & (face_gate != 0)) | (
                hand_valid & (hand_gate != 0)
            )
            detail_active = (
                expanded.detail.detail_valid
                & (schedule != 0)
                & (detail_gate != 0)
                & region_active
            )
        raw = self.build_raw_residuals(
            target_latents=target_latents,
            prepared=expanded,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            joint_attention_kwargs=joint_attention_kwargs,
            detail_active=detail_active,
        )
        detail_region_masks = None
        if expanded.detail is not None:
            expanded_detail = expanded.detail
            assert expanded_detail is not None
            masks = expanded_detail.region_masks.flatten(0, 2)[:, None]
            masks = F.interpolate(
                masks,
                size=raw.target_token_hw,
                mode="bilinear",
                align_corners=False,
            )
            detail_region_masks = masks[:, 0].reshape(
                target_latents.shape[0], 2, 3, *raw.target_token_hw
            )
        scaled, diagnostics = self.strength_controller(
            raw,
            denoise_progress=denoise_progress,
            geometry_strength=geometry_strength,
            interaction_strength=interaction_strength,
            detail_strength=detail_strength,
            face_strength=face_strength,
            hand_strength=hand_strength,
            detail_region_masks=detail_region_masks,
        )
        aligned = align_target_residuals(
            scaled,
            target_latents=target_latents,
            cond_hidden_states=cond_hidden_states,
            patch_size=self.control_core.patch_size,
        )
        diagnostics.update(
            {
                "detail_active_rows": int(detail_active.sum().item()),
                "detail_executed": bool(detail_active.any().item()),
                "target_token_hw": list(raw.target_token_hw),
                "target_token_count": raw.target_token_count,
                "full_token_count": int(aligned[0].shape[1]),
                "dtype": str(aligned[0].dtype),
                "device": str(aligned[0].device),
                "active_block_mapping": [
                    list(group) for group in self.active_block_mapping
                ],
                "residual_rms": [
                    float(value.detach().float().square().mean().sqrt().item())
                    for value in aligned
                ],
            }
        )
        return DeepGenControlOutput(aligned, diagnostics).validate()
