from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import CrossAttentionFFN, ReasoningResBlock
from .config import AdapterReasoningConfig
from ..body_pose import BodyPoseConfig


class JointGeometryAttention(nn.Module):
    """Independent attention UPDATE; only its final projection starts at zero."""
    def __init__(self, width, config: BodyPoseConfig):
        super().__init__()
        if width % 8:
            raise ValueError('Joint attention width must support 8 heads')
        self.config = config
        self.context_projection = nn.Linear(256, width)
        self.query_norm = nn.LayerNorm(width)
        self.context_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, 8, batch_first=True, dropout=0.)
        self.projection = nn.Linear(width, width)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.last_projectable = None

    @staticmethod
    def spatial_bias(grid, uv, sigma, canvas_hw):
        h, w = grid
        yy, xx = torch.meshgrid(torch.arange(h, device=uv.device, dtype=uv.dtype) + .5,
                                torch.arange(w, device=uv.device, dtype=uv.dtype) + .5, indexing='ij')
        unit = torch.stack((xx / w, yy / h), -1).reshape(1, h * w, 2)
        centers = unit * canvas_hw[:, None, [1, 0]]
        return -(centers[:, :, None] - uv[:, None]).square().sum(-1) / (2 * sigma[:, None, None].square())

    def forward(self, query, tokens, uv, valid, sigma, canvas_hw, grid):
        active = valid.any(1)
        self.last_projectable = active.detach().clone()
        # Selecting active rows avoids a completely masked softmax, even in mixed batches.
        indices = active.nonzero(as_tuple=False).flatten()
        if not indices.numel():
            return torch.zeros_like(query)
        q, j, points, mask = (x.index_select(0, indices) for x in (query, tokens, uv, valid))
        if self.config.spatial_bias:
            bias = self.spatial_bias(grid, points, sigma[indices], canvas_hw[indices])
        else:
            bias = q.new_zeros(q.shape[0], q.shape[1], 21)
        bias = bias.masked_fill(~mask[:, None], float('-inf')).repeat_interleave(8, 0)
        context = self.context_norm(self.context_projection(j))
        update = self.attention(self.query_norm(q), context, context, attn_mask=bias, need_weights=False)[0]
        delta = self.projection(update)
        return torch.zeros_like(query).index_copy(0, indices, delta)


@dataclass
class PersonReasoningOutput:
    local_feature: torch.Tensor
    tokens: torch.Tensor
    token_mask: torch.Tensor


class PersonGeometryReasoner(nn.Module):
    """Shared local/global reasoning path used identically for A and B."""

    def __init__(self, config: AdapterReasoningConfig, pose_config: BodyPoseConfig | None = None) -> None:
        super().__init__()
        self.config = config
        self.pose_config = pose_config
        if pose_config is not None:
            self.joint_attention = JointGeometryAttention(config.hidden_dim, pose_config)
        hidden_dim = config.hidden_dim
        self.input_projection = nn.Conv2d(config.condition_dim, hidden_dim, 1)
        self.local_blocks = nn.ModuleList(
            ReasoningResBlock(hidden_dim) for _ in range(2)
        )
        self.downsample = nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1)
        self.context_projection = nn.Linear(config.condition_dim, hidden_dim)
        self.global_attention = nn.ModuleList(
            CrossAttentionFFN(
                hidden_dim,
                config.global_attention_heads,
                ffn_ratio=config.ffn_ratio,
                dropout=config.dropout,
            )
            for _ in range(config.global_attention_layers)
        )

    def forward(
        self,
        spatial: torch.Tensor,
        mask: torch.Tensor,
        global_tokens: torch.Tensor,
        task_token: torch.Tensor,
        *, joint_tokens=None, joint_uv=None, projection_valid=None, joint_sigma=None, pose_canvas_hw=None,
    ) -> PersonReasoningOutput:
        if spatial.ndim != 4 or spatial.shape[1] != self.config.condition_dim:
            raise ValueError(
                f"person spatial feature must have shape [B,{self.config.condition_dim},H,W]"
            )
        if tuple(mask.shape) != (spatial.shape[0], 1, *spatial.shape[-2:]):
            raise ValueError("person mask must have shape [B,1,H,W]")
        if torch.any(~mask.bool().flatten(1).any(dim=1)):
            raise ValueError("valid person has an empty mask")
        expected_global = (spatial.shape[0], 4, self.config.condition_dim)
        if tuple(global_tokens.shape) != expected_global:
            raise ValueError(f"global tokens must have shape {expected_global}")
        if tuple(task_token.shape) != (spatial.shape[0], 1, self.config.condition_dim):
            raise ValueError("task token must have shape [B,1,condition_dim]")

        mask = mask > 0
        local = self.input_projection(spatial * mask.to(spatial.dtype))
        for block in self.local_blocks:
            local = block(local, mask)
        low_mask = F.max_pool2d(mask.to(local.dtype), kernel_size=2, stride=2) > 0
        low = self.downsample(local * mask.to(local.dtype))
        low = low * low_mask.to(low.dtype)
        tokens = low.flatten(2).transpose(1, 2)
        token_mask = low_mask.flatten(1)
        context = self.context_projection(torch.cat((global_tokens, task_token), dim=1))
        for block in self.global_attention:
            tokens = block(tokens, context, query_mask=token_mask)
        if self.pose_config is not None:
            if any(v is None for v in (joint_tokens, joint_uv, projection_valid, joint_sigma, pose_canvas_hw)):
                raise ValueError('Body Pose v2 requires joint conditions')
            delta = self.joint_attention(tokens, joint_tokens, joint_uv, projection_valid, joint_sigma,
                                         pose_canvas_hw, tuple(low.shape[-2:]))
            tokens = tokens + delta * token_mask[..., None].to(delta.dtype)
        return PersonReasoningOutput(local, tokens, token_mask)
