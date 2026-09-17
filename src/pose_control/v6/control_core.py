from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from diffusers.models import SD3ControlNetModel



@dataclass
class BranchControlResiduals:
    """Unscaled target-token residuals from the dynamic control branches."""

    geometry: tuple[torch.Tensor, ...]
    interaction: tuple[torch.Tensor, ...]
    target_token_hw: tuple[int, int]

    @property
    def target_token_count(self) -> int:
        return int(self.target_token_hw[0] * self.target_token_hw[1])

    @property
    def batch_size(self) -> int:
        return int(self.geometry[0].shape[0])

    @property
    def hidden_dim(self) -> int:
        return int(self.geometry[0].shape[-1])

    def validate(self) -> "BranchControlResiduals":
        if len(self.geometry) != 6 or len(self.interaction) != 6:
            raise ValueError("geometry and interaction must each contain six residuals")
        if len(self.target_token_hw) != 2 or min(self.target_token_hw) <= 0:
            raise ValueError("target_token_hw must contain two positive dimensions")
        expected = None
        for branch_name, branch in (
            ("geometry", self.geometry),
            ("interaction", self.interaction),
        ):
            for index, residual in enumerate(branch):
                if residual.ndim != 3:
                    raise ValueError(
                        f"{branch_name} residual {index} must have shape [B,N,D]"
                    )
                if residual.shape[1] != self.target_token_count:
                    raise ValueError(
                        f"{branch_name} residual {index} token count does not match target_token_hw"
                    )
                if expected is None:
                    expected = residual.shape
                elif residual.shape != expected:
                    raise ValueError("all branch residuals must have the same shape")
        return self


class CrossNorm(nn.Module):
    """Align control-token statistics to the current target-token stream."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, control_hidden: torch.Tensor, target_hidden: torch.Tensor) -> torch.Tensor:
        control_mean = control_hidden.mean(dim=-1, keepdim=True)
        control_std = control_hidden.var(dim=-1, keepdim=True, unbiased=False).add(self.eps).sqrt()
        target_mean = target_hidden.mean(dim=-1, keepdim=True)
        target_std = target_hidden.var(dim=-1, keepdim=True, unbiased=False).add(self.eps).sqrt()
        return (control_hidden - control_mean) / control_std * target_std + target_mean


class LowRankStageAdapter(nn.Module):
    def __init__(self, hidden_dim: int, rank: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.activation = nn.SiLU()
        self.up = nn.Linear(rank, hidden_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.up(self.activation(self.down(self.norm(hidden_states))))


class _TinyPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, patch_size: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(in_channels, hidden_dim, patch_size, stride=patch_size)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.projection(value).flatten(2).transpose(1, 2)


class _TinyTimeTextEmbed(nn.Module):
    def __init__(self, pooled_input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(pooled_input_dim + 1, hidden_dim)

    def forward(self, timestep: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
        timestep = timestep.to(dtype=pooled.dtype).reshape(-1, 1) / 1000
        return self.projection(torch.cat((pooled, timestep), dim=-1))


class _TinyJointBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.hidden_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.context_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.context_to_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_to_context = nn.Linear(hidden_dim, hidden_dim)
        self.temb_to_hidden = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        joint_attention_kwargs: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del joint_attention_kwargs
        hidden_context = self.context_to_hidden(encoder_hidden_states.mean(dim=1, keepdim=True))
        hidden_states = hidden_states + self.hidden_mlp(self.hidden_norm(hidden_states))
        hidden_states = hidden_states + hidden_context + self.temb_to_hidden(temb)[:, None]
        context_update = self.hidden_to_context(hidden_states.mean(dim=1, keepdim=True))
        encoder_hidden_states = encoder_hidden_states + self.context_mlp(
            self.context_norm(encoder_hidden_states)
        ) + context_update
        return encoder_hidden_states, hidden_states


class _RecurrentControlBranch(nn.Module):
    def __init__(
        self,
        *,
        condition_embed: nn.Module,
        hidden_dim: int,
        num_stages: int,
        rank: int,
        first_zero_head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.condition_embed = condition_embed
        self.stage_embeddings = nn.Parameter(
            torch.randn(num_stages, hidden_dim) * 0.02
        )
        self.stage_adapters = nn.ModuleList(
            LowRankStageAdapter(hidden_dim, rank) for _ in range(num_stages)
        )
        self.cross_norms = nn.ModuleList(CrossNorm() for _ in range(num_stages))
        heads = []
        for stage in range(num_stages):
            head = (
                first_zero_head
                if stage == 0 and first_zero_head is not None
                else nn.Linear(hidden_dim, hidden_dim)
            )
            nn.init.zeros_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)
            heads.append(head)
        self.zero_heads = nn.ModuleList(heads)


class SharedRecurrentControlCore(nn.Module):
    """A shared DeepGen block with independent dynamic geometry/interaction states."""

    def __init__(
        self,
        *,
        pos_embed: nn.Module,
        geometry_condition_embed: nn.Module,
        interaction_condition_embed: nn.Module,
        time_text_embed: nn.Module,
        context_embedder: nn.Module,
        shared_block: nn.Module,
        hidden_dim: int,
        patch_size: int,
        num_stages: int = 6,
        rank: int = 64,
        first_geometry_zero_head: nn.Module | None = None,
        first_interaction_zero_head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.pos_embed = pos_embed
        self.time_text_embed = time_text_embed
        self.context_embedder = context_embedder
        self.shared_block = shared_block
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.num_stages = num_stages
        self.geometry_branch = _RecurrentControlBranch(
            condition_embed=geometry_condition_embed,
            hidden_dim=hidden_dim,
            num_stages=num_stages,
            rank=rank,
            first_zero_head=first_geometry_zero_head,
        )
        self.interaction_branch = _RecurrentControlBranch(
            condition_embed=interaction_condition_embed,
            hidden_dim=hidden_dim,
            num_stages=num_stages,
            rank=rank,
            first_zero_head=first_interaction_zero_head,
        )

    @property
    def geometry_zero_heads(self) -> nn.ModuleList:
        return self.geometry_branch.zero_heads

    @property
    def interaction_zero_heads(self) -> nn.ModuleList:
        return self.interaction_branch.zero_heads

    @classmethod
    def build_tiny(
        cls,
        *,
        hidden_dim: int,
        condition_channels: int,
        context_input_dim: int,
        pooled_input_dim: int,
        patch_size: int = 2,
        num_stages: int = 6,
        rank: int = 8,
    ) -> "SharedRecurrentControlCore":
        return cls(
            pos_embed=_TinyPatchEmbed(16, hidden_dim, patch_size),
            geometry_condition_embed=_TinyPatchEmbed(
                condition_channels, hidden_dim, patch_size
            ),
            interaction_condition_embed=_TinyPatchEmbed(
                condition_channels, hidden_dim, patch_size
            ),
            time_text_embed=_TinyTimeTextEmbed(pooled_input_dim, hidden_dim),
            context_embedder=nn.Linear(context_input_dim, hidden_dim),
            shared_block=_TinyJointBlock(hidden_dim),
            hidden_dim=hidden_dim,
            patch_size=patch_size,
            num_stages=num_stages,
            rank=rank,
        )

    @classmethod
    def from_deepgen_config(
        cls,
        config,
        *,
        geometry_channels: int = 128,
        num_stages: int = 6,
        rank: int = 64,
    ) -> "SharedRecurrentControlCore":
        dual_layers = tuple(
            index for index in getattr(config, "dual_attention_layers", ()) if index == 0
        )
        base = SD3ControlNetModel(
            sample_size=config.sample_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            num_layers=1,
            attention_head_dim=config.attention_head_dim,
            num_attention_heads=config.num_attention_heads,
            joint_attention_dim=config.joint_attention_dim,
            caption_projection_dim=config.caption_projection_dim,
            pooled_projection_dim=config.pooled_projection_dim,
            out_channels=config.out_channels,
            pos_embed_max_size=config.pos_embed_max_size,
            extra_conditioning_channels=geometry_channels,
            dual_attention_layers=dual_layers,
            qk_norm=getattr(config, "qk_norm", None),
            pos_embed_type=getattr(config, "pos_embed_type", "sincos"),
            use_pos_embed=getattr(config, "use_pos_embed", True),
            force_zeros_for_pooled_projection=getattr(
                config, "force_zeros_for_pooled_projection", True
            ),
        )
        inner_dim = config.num_attention_heads * config.attention_head_dim
        return cls(
            pos_embed=base.pos_embed,
            geometry_condition_embed=base.pos_embed_input,
            interaction_condition_embed=copy.deepcopy(base.pos_embed_input),
            time_text_embed=base.time_text_embed,
            context_embedder=base.context_embedder,
            shared_block=base.transformer_blocks[0],
            hidden_dim=inner_dim,
            patch_size=config.patch_size,
            num_stages=num_stages,
            rank=rank,
            first_geometry_zero_head=base.controlnet_blocks[0],
            first_interaction_zero_head=copy.deepcopy(base.controlnet_blocks[0]),
        )

    @classmethod
    def from_deepgen(
        cls,
        deepgen_transformer: nn.Module,
        *,
        geometry_channels: int = 128,
        num_stages: int = 6,
        rank: int = 64,
    ) -> "SharedRecurrentControlCore":
        for parameter in deepgen_transformer.parameters():
            parameter.requires_grad_(False)
        core = cls.from_deepgen_config(
            deepgen_transformer.config,
            geometry_channels=geometry_channels,
            num_stages=num_stages,
            rank=rank,
        )
        core.pos_embed.load_state_dict(deepgen_transformer.pos_embed.state_dict())
        core.time_text_embed.load_state_dict(deepgen_transformer.time_text_embed.state_dict())
        core.context_embedder.load_state_dict(deepgen_transformer.context_embedder.state_dict())
        incompatible = core.shared_block.load_state_dict(
            deepgen_transformer.transformer_blocks[0].state_dict(), strict=False
        )
        if incompatible.unexpected_keys:
            raise RuntimeError(
                f"unexpected keys while copying DeepGen block 0: {incompatible.unexpected_keys}"
            )
        return core

    def _run_branch(
        self,
        *,
        branch: _RecurrentControlBranch,
        target_hidden: torch.Tensor,
        control_condition: torch.Tensor,
        context: torch.Tensor,
        temb: torch.Tensor,
        joint_attention_kwargs: Optional[dict],
    ) -> tuple[torch.Tensor, ...]:
        hidden_states = target_hidden + branch.condition_embed(control_condition)
        branch_context = context
        residuals = []
        for stage_index in range(self.num_stages):
            hidden_states = (
                hidden_states + branch.stage_embeddings[stage_index][None, None]
            )
            branch_context, hidden_states = self.shared_block(
                hidden_states=hidden_states,
                encoder_hidden_states=branch_context,
                temb=temb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            hidden_states = hidden_states + branch.stage_adapters[stage_index](
                hidden_states
            )
            aligned = branch.cross_norms[stage_index](hidden_states, target_hidden)
            residuals.append(branch.zero_heads[stage_index](aligned))
        return tuple(residuals)

    def forward(
        self,
        *,
        target_latents: torch.Tensor,
        geometry_condition: torch.Tensor,
        interaction_condition: torch.Tensor,
        interaction_valid: torch.Tensor,
        adapter_tokens: torch.Tensor,
        adapter_token_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
        joint_attention_kwargs: Optional[dict] = None,
    ) -> BranchControlResiduals:
        if target_latents.shape[-2] % self.patch_size or target_latents.shape[-1] % self.patch_size:
            raise ValueError("target latent size must be divisible by patch_size")
        batch_size = target_latents.shape[0]
        if interaction_valid.shape != (batch_size,) or interaction_valid.dtype != torch.bool:
            raise ValueError("interaction_valid must have shape [B] and dtype bool")
        target_hidden = self.pos_embed(target_latents)
        temb = self.time_text_embed(timestep, pooled_projections)
        context = self.context_embedder(encoder_hidden_states)
        adapter_tokens = adapter_tokens * adapter_token_mask[..., None].to(adapter_tokens.dtype)
        context = torch.cat((context, adapter_tokens), dim=1)
        geometry = self._run_branch(
            branch=self.geometry_branch,
            target_hidden=target_hidden,
            control_condition=geometry_condition,
            context=context,
            temb=temb,
            joint_attention_kwargs=joint_attention_kwargs,
        )
        interaction = tuple(torch.zeros_like(item) for item in geometry)
        dual_indices = interaction_valid.nonzero(as_tuple=False).flatten()
        if dual_indices.numel():
            dual = self._run_branch(
                branch=self.interaction_branch,
                target_hidden=target_hidden.index_select(0, dual_indices),
                control_condition=interaction_condition.index_select(
                    0, dual_indices
                ),
                context=context.index_select(0, dual_indices),
                temb=temb.index_select(0, dual_indices),
                joint_attention_kwargs=joint_attention_kwargs,
            )
            interaction = tuple(
                current.index_copy(0, dual_indices, update)
                for current, update in zip(interaction, dual)
            )
        return BranchControlResiduals(
            geometry=geometry,
            interaction=interaction,
            target_token_hw=(
                target_latents.shape[-2] // self.patch_size,
                target_latents.shape[-1] // self.patch_size,
            ),
        ).validate()
