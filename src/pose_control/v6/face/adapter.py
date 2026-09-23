from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from third_party.v65_face.visual_persona import IPAttnProcessor2_0

from .conditions import PreparedFaceConditioning


def _as_batch_scalar(
    value: float | torch.Tensor,
    reference: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    if not torch.is_tensor(value):
        return reference.new_full((reference.shape[0],), float(value))
    value = value.to(device=reference.device, dtype=reference.dtype)
    if value.ndim == 0:
        return value.expand(reference.shape[0])
    if value.shape != (reference.shape[0],):
        raise ValueError(f"{name} must be scalar or have shape [B]")
    return value


def scatter_face_roi_residuals(
    roi_residuals: torch.Tensor,
    roi_masks: torch.Tensor,
    target_boxes: torch.Tensor,
    face_valid: torch.Tensor,
    target_token_hw: tuple[int, int],
) -> torch.Tensor:
    """Inverse-warp two ROI maps and blend overlap without increasing strength."""

    if roi_residuals.ndim != 5 or roi_residuals.shape[1] != 2:
        raise ValueError("roi_residuals must have shape [B,2,Hroi,Wroi,D]")
    batch_size, people, roi_height, roi_width, channels = roi_residuals.shape
    if roi_height <= 0 or roi_width <= 0:
        raise ValueError("ROI dimensions must be positive")
    if roi_masks.ndim != 4 or roi_masks.shape[:2] != (batch_size, people):
        raise ValueError("roi_masks must have shape [B,2,H,W]")
    if target_boxes.shape != (batch_size, people, 4):
        raise ValueError("target_boxes must have shape [B,2,4]")
    if face_valid.shape != (batch_size, people) or face_valid.dtype != torch.bool:
        raise ValueError("face_valid must have shape [B,2] and dtype bool")
    target_height, target_width = target_token_hw
    if target_height <= 0 or target_width <= 0:
        raise ValueError("target_token_hw must contain positive dimensions")

    dtype = roi_residuals.dtype
    device = roi_residuals.device
    y = (torch.arange(target_height, device=device, dtype=dtype) + 0.5) / target_height
    x = (torch.arange(target_width, device=device, dtype=dtype) + 0.5) / target_width
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    global_grid = torch.stack((grid_x, grid_y), dim=-1)[None, None]
    boxes = target_boxes.to(device=device, dtype=dtype)
    origin = boxes[..., :2][..., None, None, :]
    size = (boxes[..., 2:] - boxes[..., :2]).clamp_min(1e-6)
    local_grid = (global_grid - origin) / size[..., None, None, :]
    sample_grid = (local_grid * 2 - 1).reshape(
        batch_size * people, target_height, target_width, 2
    )

    roi_channels_first = roi_residuals.permute(0, 1, 4, 2, 3).reshape(
        batch_size * people, channels, roi_height, roi_width
    )
    sampled_residuals = F.grid_sample(
        roi_channels_first,
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).reshape(batch_size, people, channels, target_height, target_width)
    sampled_residuals = sampled_residuals.permute(0, 1, 3, 4, 2)
    sampled_masks = F.grid_sample(
        roi_masks.to(dtype=dtype).reshape(
            batch_size * people, 1, *roi_masks.shape[-2:]
        ),
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).reshape(batch_size, people, target_height, target_width)
    sampled_masks = sampled_masks * face_valid[..., None, None].to(dtype)
    weight_sum = sampled_masks.sum(dim=1)
    max_mask = sampled_masks.amax(dim=1)
    weighted = (sampled_residuals * sampled_masks[..., None]).sum(dim=1)
    blended = weighted / weight_sum.clamp_min(1e-6)[..., None]
    blended = blended * max_mask[..., None]
    return blended.reshape(batch_size, target_height * target_width, channels)


class _LowRankStageAdapter(nn.Module):
    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.up(F.silu(self.down(self.norm(hidden_states))))


class FaceAttentionBlock(IPAttnProcessor2_0):
    """ROI self-attention plus an independent face-token K/V path."""

    def __init__(self, dim: int, heads: int, ff_mult: int = 4) -> None:
        super().__init__(
            hidden_size=dim,
            cross_attention_dim=dim,
            scale=1.0,
            heads=heads,
        )
        self.self_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )

    def forward(self, hidden_states: torch.Tensor, face_tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.self_norm(hidden_states)
        update, _ = self.self_attention(
            normalized, normalized, normalized, need_weights=False
        )
        hidden_states = hidden_states + update
        context = self.context_norm(face_tokens)
        hidden_states = hidden_states + self.face_attention(
            self.cross_norm(hidden_states), context
        )
        return hidden_states + self.ffn(hidden_states)


class FaceControlAdapter(nn.Module):
    """Six-stage independent face control branch operating on per-person ROIs."""

    def __init__(
        self,
        *,
        face_dim: int = 512,
        deepgen_dim: int = 1536,
        num_stages: int = 6,
        rank: int = 64,
        heads: int = 8,
    ) -> None:
        super().__init__()
        if num_stages != 6:
            raise ValueError("V6.5 defines exactly six face control stages")
        if face_dim <= 0 or deepgen_dim <= 0 or rank <= 0:
            raise ValueError("adapter dimensions must be positive")
        self.face_dim = face_dim
        self.deepgen_dim = deepgen_dim
        self.num_stages = num_stages
        self.spatial_projection = nn.Linear(512, face_dim)
        self.identity_projection = nn.Linear(512, face_dim)
        self.texture_projection = nn.Linear(512, face_dim)
        self.geometry_projection = nn.Linear(512, face_dim)
        self.person_embedding = nn.Parameter(torch.randn(2, face_dim) * 0.02)
        self.token_type_embedding = nn.Parameter(torch.randn(3, face_dim) * 0.02)
        self.stage_embeddings = nn.Parameter(torch.randn(num_stages, face_dim) * 0.02)
        self.shared_block = FaceAttentionBlock(face_dim, heads)
        self.stage_adapters = nn.ModuleList(
            _LowRankStageAdapter(face_dim, rank) for _ in range(num_stages)
        )
        self.timestep_films = nn.ModuleList(
            nn.Sequential(nn.SiLU(), nn.Linear(1, face_dim * 2))
            for _ in range(num_stages)
        )
        self.film_norms = nn.ModuleList(nn.LayerNorm(face_dim) for _ in range(num_stages))
        self.zero_heads = nn.ModuleList(
            nn.Linear(face_dim, deepgen_dim) for _ in range(num_stages)
        )
        for head in self.zero_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _face_tokens(
        self,
        prepared: PreparedFaceConditioning,
        texture_scale: torch.Tensor,
        face_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = prepared.batch_size
        texture_scale = texture_scale[:, None, None, None]
        identity = self.identity_projection(
            prepared.identity_tokens + prepared.texture_delta * texture_scale
        )
        geometry = self.geometry_projection(prepared.geometry_tokens)
        person_binding = self.person_embedding[None, :, None]
        identity = identity + self.token_type_embedding[0] + person_binding
        texture = (
            self.texture_projection(prepared.texture_tokens)
            + self.token_type_embedding[1]
            + person_binding
        ) * texture_scale
        geometry = geometry + self.token_type_embedding[2] + person_binding
        tokens = torch.cat((identity, texture, geometry), dim=2)
        valid = face_valid[..., None, None].to(tokens.dtype)
        return (tokens * valid).reshape(batch_size * 2, 16, self.face_dim)

    def forward(
        self,
        prepared: PreparedFaceConditioning,
        *,
        target_token_hw: tuple[int, int],
        timestep: torch.Tensor,
        texture_scale: float | torch.Tensor = 1.0,
        face_active: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        prepared.validate()
        batch_size = prepared.batch_size
        if timestep.ndim == 0:
            timestep = timestep.expand(batch_size)
        if timestep.shape != (batch_size,):
            raise ValueError("timestep must be scalar or have shape [B]")
        timestep = timestep.to(
            device=prepared.device, dtype=prepared.spatial_features.dtype
        )
        texture_scale = _as_batch_scalar(
            texture_scale,
            prepared.spatial_features,
            name="texture_scale",
        )
        face_valid = prepared.face_valid
        if face_active is not None:
            if face_active.shape != (batch_size,) or face_active.dtype != torch.bool:
                raise ValueError("face_active must have shape [B] and dtype bool")
            face_valid = face_valid & face_active[:, None].to(face_valid.device)
        tokens = self._face_tokens(prepared, texture_scale, face_valid)
        hidden_states = prepared.spatial_features.flatten(3).transpose(2, 3)
        hidden_states = self.spatial_projection(hidden_states)
        hidden_states = hidden_states + self.person_embedding[None, :, None]
        valid = face_valid[..., None, None].to(hidden_states.dtype)
        hidden_states = (hidden_states * valid).reshape(
            batch_size * 2, 256, self.face_dim
        )
        person_timestep = timestep.repeat_interleave(2).reshape(-1, 1) / 1000.0
        residuals = []
        for stage_index in range(self.num_stages):
            hidden_states = hidden_states + self.stage_embeddings[stage_index]
            film = self.timestep_films[stage_index](person_timestep)
            scale, shift = film.chunk(2, dim=-1)
            modulated = self.film_norms[stage_index](hidden_states)
            hidden_states = hidden_states + modulated * (1 + scale[:, None]) + shift[:, None]
            hidden_states = self.shared_block(hidden_states, tokens)
            hidden_states = hidden_states + self.stage_adapters[stage_index](hidden_states)
            local = self.zero_heads[stage_index](hidden_states)
            local = local * face_valid.reshape(-1, 1, 1).to(local.dtype)
            local = local.reshape(batch_size, 2, 16, 16, self.deepgen_dim)
            residuals.append(
                scatter_face_roi_residuals(
                    local,
                    prepared.face_masks,
                    prepared.target_boxes,
                    face_valid,
                    target_token_hw,
                )
            )
        return tuple(residuals)


__all__ = [
    "FaceAttentionBlock",
    "FaceControlAdapter",
    "scatter_face_roi_residuals",
]
