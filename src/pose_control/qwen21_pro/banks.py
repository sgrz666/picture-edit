"""共享参数、人物独立 K/V 的源外观 bank。"""
from __future__ import annotations

import torch
from torch import nn

from .contracts import ProAdapterConfig, TaskCondition


class PersonBankAttention(nn.Module):
    """返回 [B,P,N,width]；调用方按可见 owner 融合，不做跨 bank 查询。"""

    def __init__(self, config: ProAdapterConfig):
        super().__init__()
        self.width, self.heads = config.width, config.heads
        self.query_projection = nn.Linear(config.width, config.width, bias=False)
        self.source_key = nn.Linear(config.source_dim, config.width, bias=False)
        self.source_value = nn.Linear(config.source_dim, config.width, bias=False)
        self.source_output = nn.Linear(config.width, config.width, bias=False)
        # Face ID is never appended to the globally visible appearance sequence.
        self.identity_projection = nn.Linear(config.identity_dim, config.width, bias=False)

    def forward(self, query, conditions: TaskCondition):
        b, p, n, width = query.shape
        if width != self.width or conditions.person_valid.shape != (b, p):
            raise ValueError("bank query must match condition [B,P,N,width]")
        tokens = conditions.source_tokens.float()
        m = tokens.shape[2]
        if m:
            size = width // self.heads
            q = self.query_projection(query.float()).reshape(b, p, n, self.heads, size).transpose(2, 3)
            k = self.source_key(tokens).reshape(b, p, m, self.heads, size).transpose(2, 3)
            v = self.source_value(tokens).reshape(b, p, m, self.heads, size).transpose(2, 3)
            valid = (conditions.source_token_valid & conditions.person_valid[..., None])[:, :, None, None]
            logits = torch.matmul(q, k.transpose(-2, -1)) * (size ** -.5)
            # Finite logits and explicit renormalization make all-masked rows zero.
            probability = logits.masked_fill(~valid, torch.finfo(logits.dtype).min).softmax(dim=-1) * valid
            probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            context = torch.matmul(probability, v).transpose(2, 3).reshape(b, p, n, width)
            context = self.source_output(context)
        else:
            context = query.new_zeros(b, p, n, width, dtype=torch.float32)
        identity = self.identity_projection(conditions.face_identity.float())[:, :, None]
        identity_valid = (conditions.face_identity_valid & conditions.person_valid)[..., None, None]
        return context + identity * conditions.face_mask.float()[..., None] * identity_valid
