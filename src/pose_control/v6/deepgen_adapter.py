from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conditions import UnifiedAdapterCondition
from .control_core import SharedRecurrentControlCore
from .experts import DualPersonExpert, RoutingDecision, SinglePersonExpert, TaskRouter
from .geometry_encoder import SharedGeometryEncoder
from .token_encoders import (
    AppearanceTokenEncoder,
    GlobalGeometryTokenEncoder,
    PersonTokenBinder,
    TaskTokenEncoder,
)


@dataclass
class PreparedAdapterConditioning:
    scene_feature: torch.Tensor
    adapter_tokens: torch.Tensor
    adapter_token_mask: torch.Tensor
    contact_token_mask: torch.Tensor
    route: RoutingDecision


def latent_token_count(latent: torch.Tensor, patch_size: int) -> int:
    if latent.ndim not in (3, 4):
        raise ValueError("latent must have shape [C,H,W] or [B,C,H,W]")
    height, width = latent.shape[-2:]
    if height % patch_size or width % patch_size:
        raise ValueError(f"latent size {(height, width)} is not divisible by patch_size={patch_size}")
    return (height // patch_size) * (width // patch_size)


def align_target_residuals(
    residuals: Sequence[torch.Tensor],
    target_latents: torch.Tensor,
    cond_hidden_states: Optional[Sequence[Sequence[torch.Tensor]]],
    patch_size: int,
) -> list[torch.Tensor]:
    """Return DeepGen order: [target residual | source zeros | padding zeros]."""

    target_tokens = latent_token_count(target_latents, patch_size)
    batch_size = target_latents.shape[0]
    full_tokens = target_tokens
    if cond_hidden_states is not None:
        if len(cond_hidden_states) != batch_size:
            raise ValueError("cond_hidden_states batch does not match target latents")
        lengths = []
        for references in cond_hidden_states:
            lengths.append(target_tokens + sum(latent_token_count(item, patch_size) for item in references))
        full_tokens = max(lengths)

    aligned = []
    for residual in residuals:
        if residual.ndim != 3 or residual.shape[1] != target_tokens:
            raise ValueError("control residual must contain target tokens only")
        if batch_size % residual.shape[0]:
            raise ValueError("residual batch cannot be expanded to transformer batch")
        if residual.shape[0] != batch_size:
            residual = torch.cat([residual] * (batch_size // residual.shape[0]), dim=0)
        padding = full_tokens - target_tokens
        if padding:
            residual = torch.cat(
                (residual, residual.new_zeros(batch_size, padding, residual.shape[-1])), dim=1
            )
        aligned.append(residual)
    return aligned


class UnifiedSMPLXAdapterV6(nn.Module):
    """Unified SMPL-X control adapter for frozen DeepGen SD3-style DiT.

    Internal task/person tokens never alter the frozen tokenizer or base text
    sequence. Only six target-token residuals are exposed to DeepGen.
    """

    def __init__(
        self,
        control_core: SharedRecurrentControlCore,
        *,
        geometry_channels: int = 128,
        geometry_base_channels: int = 24,
        cross_attention_dim: int = 512,
        expert_heads: int = 8,
        num_parts: int = 24,
        contact_pair_dim: int = 4,
        contact_token_count: int = 8,
    ) -> None:
        super().__init__()
        self.control_core = control_core
        token_dim = control_core.hidden_dim
        self.geometry_encoder = SharedGeometryEncoder(
            base_channels=geometry_base_channels,
            output_channels=geometry_channels,
            num_parts=num_parts,
        )
        self.geometry_token_encoder = GlobalGeometryTokenEncoder(token_dim=token_dim)
        self.appearance_token_encoder = AppearanceTokenEncoder(token_dim=token_dim)
        self.task_token_encoder = TaskTokenEncoder(token_dim=token_dim)
        self.person_token_binder = PersonTokenBinder(token_dim=token_dim)
        self.router = TaskRouter()
        self.single_expert = SinglePersonExpert(contact_token_count=contact_token_count)
        self.dual_expert = DualPersonExpert(
            feature_channels=geometry_channels,
            cross_attention_dim=cross_attention_dim,
            token_dim=token_dim,
            num_heads=expert_heads,
            contact_pair_dim=contact_pair_dim,
            contact_token_count=contact_token_count,
        )
        self.geometry_channels = geometry_channels
        self.contact_token_count = contact_token_count

    @staticmethod
    def _freeze_backbone(backbone: nn.Module | None) -> None:
        if backbone is None:
            return
        backbone.eval()
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def freeze_deepgen_pipeline_components(cls, pipeline) -> None:
        """Freeze every DeepGen component that must remain outside training."""

        for name in ("transformer", "vae", "lmm", "vlm", "connector_module", "connector"):
            cls._freeze_backbone(getattr(pipeline, name, None))

    @classmethod
    def build_tiny(
        cls,
        *,
        hidden_dim: int = 32,
        geometry_channels: int = 16,
        context_input_dim: int = 24,
        pooled_input_dim: int = 20,
        num_heads: int = 4,
        deepgen_backbone: nn.Module | None = None,
    ) -> "UnifiedSMPLXAdapterV6":
        cls._freeze_backbone(deepgen_backbone)
        core = SharedRecurrentControlCore.build_tiny(
            hidden_dim=hidden_dim,
            condition_channels=16 + geometry_channels,
            context_input_dim=context_input_dim,
            pooled_input_dim=pooled_input_dim,
            rank=max(4, hidden_dim // 4),
        )
        return cls(
            core,
            geometry_channels=geometry_channels,
            geometry_base_channels=max(4, geometry_channels // 2),
            cross_attention_dim=max(hidden_dim, geometry_channels),
            expert_heads=num_heads,
        )

    @classmethod
    def from_deepgen_config(cls, config) -> "UnifiedSMPLXAdapterV6":
        core = SharedRecurrentControlCore.from_deepgen_config(config)
        return cls(core)

    @classmethod
    def from_deepgen(cls, deepgen_transformer: nn.Module) -> "UnifiedSMPLXAdapterV6":
        cls._freeze_backbone(deepgen_transformer)
        core = SharedRecurrentControlCore.from_deepgen(deepgen_transformer)
        return cls(core)

    @classmethod
    def from_deepgen_pipeline(cls, pipeline) -> "UnifiedSMPLXAdapterV6":
        cls.freeze_deepgen_pipeline_components(pipeline)
        return cls.from_deepgen(pipeline.transformer)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def prepare_conditioning(self, condition: UnifiedAdapterCondition) -> PreparedAdapterConditioning:
        condition.validate()
        pyramid = self.geometry_encoder(condition)
        geometry_tokens = self.geometry_token_encoder(condition)
        appearance_tokens = self.appearance_token_encoder(
            condition.source_person_latents, condition.person_valid
        )
        route = self.router.resolve(condition.person_valid, condition.task_spec)
        task_token = self.task_token_encoder(route.num_people, route.interaction_ids)
        person_tokens, person_token_mask = self.person_token_binder(
            geometry_tokens,
            appearance_tokens,
            task_token,
            condition.person_valid,
            condition.effective_source_indices(),
        )

        batch_size = condition.batch_size
        spatial_shape = pyramid.level_64.shape[2:]
        scene = pyramid.level_64.new_zeros(batch_size, *spatial_shape)
        routed_person_tokens = person_tokens.new_zeros(person_tokens.shape)
        contact_tokens = person_tokens.new_zeros(
            batch_size, self.contact_token_count, person_tokens.shape[-1]
        )
        contact_mask = torch.zeros(
            batch_size, self.contact_token_count, dtype=torch.bool, device=condition.device
        )

        for active_mask, expert in (
            (route.single_mask, self.single_expert),
            (route.dual_mask, self.dual_expert),
        ):
            indices = active_mask.nonzero(as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            output = expert(
                pyramid.level_64.index_select(0, indices),
                person_tokens.index_select(0, indices),
                condition.index_select(indices),
            )
            scene = scene.index_copy(0, indices, output.scene_feature)
            routed_person_tokens = routed_person_tokens.index_copy(0, indices, output.person_tokens)
            contact_tokens = contact_tokens.index_copy(0, indices, output.contact_tokens)
            contact_mask = contact_mask.index_copy(0, indices, output.contact_token_mask)

        flat_person_tokens = routed_person_tokens.flatten(1, 2)
        flat_person_mask = person_token_mask.flatten(1, 2)
        task_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=condition.device)
        adapter_tokens = torch.cat((flat_person_tokens, task_token, contact_tokens), dim=1)
        adapter_mask = torch.cat((flat_person_mask, task_mask, contact_mask), dim=1)
        adapter_tokens = adapter_tokens * adapter_mask[..., None].to(adapter_tokens.dtype)
        return PreparedAdapterConditioning(
            scene_feature=scene,
            adapter_tokens=adapter_tokens,
            adapter_token_mask=adapter_mask,
            contact_token_mask=contact_mask,
            route=route,
        )

    def forward(
        self,
        *,
        target_latents: torch.Tensor,
        condition: UnifiedAdapterCondition,
        source_scene_latents: torch.Tensor,
        cond_hidden_states: Optional[Sequence[Sequence[torch.Tensor]]],
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
        conditioning_scale: float = 1.0,
        joint_attention_kwargs: Optional[dict] = None,
    ) -> list[torch.Tensor]:
        parameter = next(self.parameters())
        device, dtype = target_latents.device, parameter.dtype
        condition = condition.to(device=device, dtype=dtype)
        prepared = self.prepare_conditioning(condition)
        target_size = target_latents.shape[-2:]
        scene = F.interpolate(
            prepared.scene_feature, size=target_size, mode="bilinear", align_corners=False
        )
        source_scene = F.interpolate(
            source_scene_latents.to(device=device, dtype=dtype),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        control_condition = torch.cat((source_scene, scene), dim=1)
        # Route groups separately so single-person execution physically omits
        # Person-B/contact tokens from joint attention (not merely zeroing them).
        residuals = [
            target_latents.new_zeros(
                target_latents.shape[0],
                latent_token_count(target_latents, self.control_core.patch_size),
                self.control_core.hidden_dim,
                dtype=dtype,
            )
            for _ in range(self.control_core.num_stages)
        ]
        for route_mask in (prepared.route.single_mask, prepared.route.dual_mask):
            indices = route_mask.nonzero(as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            group_mask = prepared.adapter_token_mask.index_select(0, indices)
            retained_columns = group_mask.any(dim=0)
            group_residuals = self.control_core(
                target_latents=target_latents.index_select(0, indices).to(dtype=dtype),
                control_condition=control_condition.index_select(0, indices),
                adapter_tokens=prepared.adapter_tokens.index_select(0, indices)[:, retained_columns].to(
                    dtype=dtype
                ),
                adapter_token_mask=group_mask[:, retained_columns],
                encoder_hidden_states=encoder_hidden_states.index_select(0, indices).to(
                    device=device, dtype=dtype
                ),
                pooled_projections=pooled_projections.index_select(0, indices).to(
                    device=device, dtype=dtype
                ),
                timestep=timestep.index_select(0, indices).to(device=device),
                conditioning_scale=conditioning_scale,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            residuals = [
                current.index_copy(0, indices, group)
                for current, group in zip(residuals, group_residuals)
            ]
        return align_target_residuals(
            residuals,
            target_latents=target_latents,
            cond_hidden_states=cond_hidden_states,
            patch_size=self.control_core.patch_size,
        )

    def forward_deepgen(
        self,
        deepgen_transformer: nn.Module,
        *,
        adapter_kwargs: dict,
        transformer_kwargs: dict,
    ):
        """Run frozen DeepGen with V6's six aligned block residuals."""

        residuals = self(**adapter_kwargs)
        kwargs = dict(transformer_kwargs)
        kwargs["block_controlnet_hidden_states"] = residuals
        return deepgen_transformer(**kwargs)
