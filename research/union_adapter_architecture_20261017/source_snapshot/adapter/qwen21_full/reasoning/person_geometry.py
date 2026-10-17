from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import CrossAttentionFFN, ReasoningResBlock
from .config import AdapterReasoningConfig


@dataclass
class PersonReasoningOutput:
    local_feature: torch.Tensor
    tokens: torch.Tensor
    token_mask: torch.Tensor


class PersonGeometryReasoner(nn.Module):
    """Shared local/global reasoning path used identically for A and B."""

    def __init__(self, config: AdapterReasoningConfig) -> None:
        super().__init__()
        self.config = config
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
        return PersonReasoningOutput(local, tokens, token_mask)
