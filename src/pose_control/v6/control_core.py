from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from diffusers.models import SD3ControlNetModel


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


class SharedRecurrentControlCore(nn.Module):
    """One full-width joint block recurrently unrolled into six control stages."""

    def __init__(
        self,
        *,
        pos_embed: nn.Module,
        condition_embed: nn.Module,
        time_text_embed: nn.Module,
        context_embedder: nn.Module,
        shared_block: nn.Module,
        hidden_dim: int,
        patch_size: int,
        num_stages: int = 6,
        rank: int = 64,
        first_zero_head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.pos_embed = pos_embed
        self.condition_embed = condition_embed
        self.time_text_embed = time_text_embed
        self.context_embedder = context_embedder
        self.shared_block = shared_block
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.num_stages = num_stages
        self.stage_embeddings = nn.Parameter(torch.randn(num_stages, hidden_dim) * 0.02)
        self.stage_adapters = nn.ModuleList(
            LowRankStageAdapter(hidden_dim, rank) for _ in range(num_stages)
        )
        self.cross_norms = nn.ModuleList(CrossNorm() for _ in range(num_stages))
        heads = []
        for stage in range(num_stages):
            head = first_zero_head if stage == 0 and first_zero_head is not None else nn.Linear(hidden_dim, hidden_dim)
            nn.init.zeros_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)
            heads.append(head)
        self.zero_heads = nn.ModuleList(heads)

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
            condition_embed=_TinyPatchEmbed(condition_channels, hidden_dim, patch_size),
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
            condition_embed=base.pos_embed_input,
            time_text_embed=base.time_text_embed,
            context_embedder=base.context_embedder,
            shared_block=base.transformer_blocks[0],
            hidden_dim=inner_dim,
            patch_size=config.patch_size,
            num_stages=num_stages,
            rank=rank,
            first_zero_head=base.controlnet_blocks[0],
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

    def forward(
        self,
        *,
        target_latents: torch.Tensor,
        control_condition: torch.Tensor,
        adapter_tokens: torch.Tensor,
        adapter_token_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
        conditioning_scale: float = 1.0,
        joint_attention_kwargs: Optional[dict] = None,
    ) -> list[torch.Tensor]:
        target_hidden = self.pos_embed(target_latents)
        hidden_states = target_hidden + self.condition_embed(control_condition)
        temb = self.time_text_embed(timestep, pooled_projections)
        context = self.context_embedder(encoder_hidden_states)
        adapter_tokens = adapter_tokens * adapter_token_mask[..., None].to(adapter_tokens.dtype)
        context = torch.cat((context, adapter_tokens), dim=1)

        residuals = []
        for stage_index in range(self.num_stages):
            hidden_states = hidden_states + self.stage_embeddings[stage_index][None, None]
            context, hidden_states = self.shared_block(
                hidden_states=hidden_states,
                encoder_hidden_states=context,
                temb=temb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            hidden_states = hidden_states + self.stage_adapters[stage_index](hidden_states)
            aligned = self.cross_norms[stage_index](hidden_states, target_hidden)
            residuals.append(self.zero_heads[stage_index](aligned) * conditioning_scale)
        return residuals
