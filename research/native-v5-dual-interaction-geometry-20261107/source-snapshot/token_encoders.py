from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conditions import ContactRelationBatch, PART_NAMES, TaskType
from .body_pose import BodyPoseConfig, axis_angle_to_matrix, matrix_to_rotation_6d, joint_tree_distances


class SkeletonAttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(256)
        self.attention = nn.MultiheadAttention(256, 8, dropout=0., batch_first=True)
        self.norm2 = nn.LayerNorm(256)
        self.ffn = nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, 256))
        self.register_buffer('topology', -joint_tree_distances(), persistent=False)

    def forward(self, tokens, valid):
        safe = valid.clone()
        safe[~safe.any(1), 0] = True
        bias = self.topology[None].expand(tokens.shape[0], -1, -1).clone()
        bias = bias.masked_fill(~safe[:, None], float('-inf')).repeat_interleave(8, dim=0)
        value = self.norm1(tokens)
        update = self.attention(value, value, value, attn_mask=bias, need_weights=False)[0]
        tokens = tokens + update
        return (tokens + self.ffn(self.norm2(tokens))) * valid[..., None].to(tokens.dtype)


class BodyPoseTokenEncoder(nn.Module):
    """Encode 21 anatomical joints once, before Qwen executes any blocks."""
    def __init__(self, config: BodyPoseConfig | None = None):
        super().__init__()
        self.config = config or BodyPoseConfig()
        width = 11 if self.config.rotation == 'rotation6d_columns' else 8
        self.projection = nn.Sequential(nn.Linear(width, 256), nn.SiLU(), nn.Linear(256, 256))
        self.joint_embedding = nn.Embedding(21, 256)
        self.skeleton = nn.ModuleList(SkeletonAttentionLayer() for _ in range(2))

    def forward(self, body_pose, joints_cam, joint_uv, fk_valid, projection_valid, body_height, canvas_hw):
        batch = body_pose.shape[0]
        specs = ((body_pose, (batch, 63)), (joints_cam, (batch, 22, 3)),
                 (joint_uv, (batch, 21, 2)), (fk_valid, (batch, 22)),
                 (projection_valid, (batch, 21)), (body_height, (batch,)), (canvas_hw, (batch, 2)))
        for value, shape in specs:
            if tuple(value.shape) != shape or value.device != body_pose.device:
                raise ValueError(f'Body Pose requires 63D SMPL-X and matching tensors; expected {shape}')
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError('Body Pose inputs must be finite, including invalid entries')
        if fk_valid.dtype != torch.bool or projection_valid.dtype != torch.bool:
            raise ValueError('FK/projection validity must be bool')
        if not body_pose.is_floating_point() or (body_height <= 0).any() or (canvas_hw <= 0).any():
            raise ValueError('Body Pose float and positive height/canvas required')
        if (projection_valid & ~fk_valid[:, 1:]).any():
            raise ValueError('Projection validity requires FK validity')
        in_canvas = ((joint_uv >= 0) & (joint_uv < canvas_hw[:, None, [1, 0]])).all(-1)
        if (projection_valid & (~in_canvas | (joints_cam[:, 1:, 2] <= 1e-8))).any():
            raise ValueError('projection_valid cannot mark a behind-camera or outside joint as projectable')
        rotation = body_pose.reshape(batch, 21, 3)
        if self.config.rotation == 'rotation6d_columns':
            rotation = matrix_to_rotation_6d(axis_angle_to_matrix(rotation))
        position = (joints_cam[:, 1:] - joints_cam[:, :1]) / body_height[:, None, None]
        valid = fk_valid[:, 1:] & fk_valid[:, :1]
        uv = joint_uv / canvas_hw[:, None, [1, 0]]
        uv = torch.where(projection_valid[..., None], uv, torch.zeros_like(uv))
        inputs = torch.cat((rotation, position, uv), dim=-1)
        inputs = torch.where(valid[..., None], inputs, torch.zeros_like(inputs))
        tokens = (self.projection(inputs) + self.joint_embedding.weight[None]) * valid[..., None].to(inputs.dtype)
        for block in self.skeleton:
            tokens = block(tokens, valid)
        return tokens


class AppearanceTokenEncoder(nn.Module):
    """IP-Adapter-style decoupled identity tokens from masked source latents."""

    def __init__(self, in_channels: int = 16, token_dim: int = 1536, token_count: int = 8) -> None:
        super().__init__()
        if token_count != 8:
            raise ValueError("V6 defines exactly eight appearance tokens per person")
        self.token_count = token_count
        self.projection = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )

    def forward(self, source_latents: torch.Tensor, person_valid: torch.Tensor) -> torch.Tensor:
        if source_latents.ndim != 5 or source_latents.shape[1:3] != (2, 16):
            raise ValueError("source latents must have shape [B,2,16,h,w]")
        batch_size = source_latents.shape[0]
        flat = source_latents.reshape(batch_size * 2, *source_latents.shape[2:])
        pooled = F.adaptive_avg_pool2d(flat, (2, 4)).flatten(2).transpose(1, 2)
        tokens = self.projection(pooled).reshape(batch_size, 2, self.token_count, -1)
        return tokens * person_valid[..., None, None].to(tokens.dtype)


class PersonTokenBinder(nn.Module):
    """Bind geometry and appearance tokens to the same source/role slot."""

    def __init__(self, token_dim: int, max_sources: int = 16) -> None:
        super().__init__()
        self.slot_embedding = nn.Embedding(2, token_dim)
        self.source_embedding = nn.Embedding(max_sources, token_dim)

    def binding_for(
        self,
        source_indices: torch.Tensor,
        person_valid: torch.Tensor | None = None,
        task_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return validated person/source bindings without adding new state."""

        if source_indices.ndim != 2 or source_indices.shape[1] != 2:
            raise ValueError("source_indices must have shape [B,2]")
        if source_indices.dtype != torch.long:
            raise ValueError("source_indices must have dtype int64")
        batch_size = source_indices.shape[0]
        if person_valid is None:
            person_valid = torch.ones_like(source_indices, dtype=torch.bool)
        if tuple(person_valid.shape) != (batch_size, 2) or person_valid.dtype != torch.bool:
            raise ValueError("person_valid must have shape [B,2] and dtype bool")
        if person_valid.device != source_indices.device:
            raise ValueError("person_valid and source_indices must be on the same device")
        valid_indices = source_indices[person_valid]
        maximum = min(self.source_embedding.num_embeddings - 1, 15)
        if valid_indices.numel() and torch.any(
            (valid_indices < 0) | (valid_indices > maximum)
        ):
            raise ValueError(f"valid source_indices must be in [0, {maximum}]")
        both_valid = person_valid.all(dim=1)
        if torch.any(
            both_valid & (source_indices[:, 0] == source_indices[:, 1])
        ):
            raise ValueError("valid persons must use unique source_indices within each sample")

        safe_indices = torch.where(
            person_valid, source_indices, torch.zeros_like(source_indices)
        )
        slots = torch.arange(2, device=source_indices.device).expand(batch_size, -1)
        binding = self.slot_embedding(slots) + self.source_embedding(safe_indices)
        if task_token is not None:
            if tuple(task_token.shape) == (batch_size, binding.shape[-1]):
                task_token = task_token[:, None]
            if tuple(task_token.shape) != (batch_size, 1, binding.shape[-1]):
                raise ValueError("task_token must have shape [B,D] or [B,1,D]")
            binding = binding + task_token
        return binding

    def forward(
        self,
        geometry_tokens: torch.Tensor,
        appearance_tokens: torch.Tensor,
        person_valid: torch.Tensor,
        source_indices: torch.Tensor,
        task_token: torch.Tensor | None = None,
        geometry_token_mask: torch.Tensor | None = None,
        person_binding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = geometry_tokens.shape[0]
        if person_binding is None:
            person_binding = self.binding_for(source_indices, person_valid, task_token)
        elif tuple(person_binding.shape) != (batch_size, 2, geometry_tokens.shape[-1]):
            raise ValueError("person_binding must have shape [B,2,D]")
        binding = person_binding[:, :, None]
        geometry_count = geometry_tokens.shape[2]
        appearance_count = appearance_tokens.shape[2]
        geometry_valid = person_valid[..., None].expand(-1, -1, geometry_count)
        if geometry_token_mask is not None:
            if geometry_token_mask.shape != geometry_valid.shape:
                raise ValueError("geometry_token_mask must match geometry tokens")
            geometry_valid = geometry_valid & geometry_token_mask
        appearance_valid = person_valid[..., None].expand(-1, -1, appearance_count)
        tokens = torch.cat((geometry_tokens + binding, appearance_tokens + binding), dim=2)
        valid = torch.cat((geometry_valid, appearance_valid), dim=2)
        tokens = tokens * valid[..., None].to(tokens.dtype)
        return tokens, valid


class GlobalConditionTokenEncoder(nn.Module):
    """Tokenize beta, root rotation, translation and normalized camera only."""

    def __init__(self, token_dim: int = 256, token_count: int = 4) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.token_count = token_count
        self.network = nn.Sequential(
            nn.LayerNorm(26),
            nn.Linear(26, 512),
            nn.SiLU(),
            nn.Linear(512, 1024),
            nn.SiLU(),
            nn.Linear(1024, token_count * token_dim),
        )

    def forward(self, value: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        tokens = self.network(value).reshape(value.shape[0], self.token_count, self.token_dim)
        if valid is not None:
            tokens = tokens * valid.to(tokens.dtype)[:, None, None]
        return tokens


class RelativeGeometryTokenEncoder(nn.Module):
    def __init__(self, token_dim: int = 256, token_count: int = 2) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.token_count = token_count
        self.network = nn.Sequential(
            nn.LayerNorm(11),
            nn.Linear(11, 512),
            nn.SiLU(),
            nn.Linear(512, token_count * token_dim),
        )

    def forward(self, value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        tokens = self.network(value).reshape(value.shape[0], self.token_count, self.token_dim)
        return tokens * valid.to(tokens.dtype)[:, None, None]


class ContactRasterEncoder(nn.Module):
    def __init__(self, output_channels: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(2, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, output_channels, 3, stride=2, padding=1),
            nn.GroupNorm(8, output_channels),
            nn.SiLU(),
        )

    def forward(self, value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        output = self.network(value)
        present = value.abs().flatten(1).sum(dim=1) > 0
        active = valid & present
        return output * active.to(output.dtype)[:, None, None, None]


class StructuredContactRelationEncoder(nn.Module):
    def __init__(
        self,
        *,
        token_dim: int = 256,
        max_relations: int = 8,
        contact_type_count: int = 16,
    ) -> None:
        super().__init__()
        self.max_relations = max_relations
        self.contact_type_count = contact_type_count
        self.person_embedding = nn.Embedding(2, token_dim)
        self.part_embedding = nn.Embedding(len(PART_NAMES), token_dim)
        self.type_embedding = nn.Embedding(contact_type_count, token_dim)
        self.distance_mlp = nn.Sequential(
            nn.Linear(1, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        self.output = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, token_dim))

    def forward(
        self,
        relations: ContactRelationBatch,
        person_b_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        relations.validate(
            batch_size=person_b_valid.shape[0],
            max_relations=self.max_relations,
            contact_type_count=self.contact_type_count,
        )
        # Padding is outside the semantic contract. Replace sentinel values
        # before embedding lookup, then zero the projected tokens with mask.
        padding_mask = relations.valid_mask
        safe = lambda value: torch.where(padding_mask, value, torch.zeros_like(value))
        token = (
            self.person_embedding(safe(relations.src_person))
            + self.part_embedding(safe(relations.src_part))
            + self.person_embedding(safe(relations.dst_person))
            + self.part_embedding(safe(relations.dst_part))
            + self.type_embedding(safe(relations.contact_type))
            + self.distance_mlp(relations.distance)
        )
        mask = relations.valid_mask & person_b_valid[:, None]
        token = self.output(token) * mask[..., None].to(token.dtype)
        return token, mask


class ExplicitTaskTokenEncoder(nn.Module):
    def __init__(self, token_dim: int = 256) -> None:
        super().__init__()
        self.embedding = nn.Embedding(len(TaskType), token_dim)

    def forward(self, task_id: torch.Tensor) -> torch.Tensor:
        return self.embedding(task_id).unsqueeze(1)
