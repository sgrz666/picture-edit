"""Own-person 8 VAE + 8 DINO tokens; neither bank enters geometry reasoning."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .conditions import SourceCondition


class MaskedAttentionBlock(nn.Module):
    def __init__(self,width,heads=8):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.context_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width,heads,batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width*2),nn.GELU(),nn.Linear(width*2,width))

    def forward(self, query, context, valid):
        # Select rows with real context instead of relying on all-masked softmax.
        usable = valid.any(-1).nonzero(as_tuple=False).flatten()
        result = torch.zeros_like(query)
        if usable.numel():
            q = query.index_select(0,usable)
            kv = self.context_norm(context.index_select(0,usable))
            update = self.attention(self.query_norm(q),kv,kv,
                                    key_padding_mask=~valid.index_select(0,usable),need_weights=False)[0]
            q = q+update
            q = q+self.ffn(q)
            result = result.index_copy(0,usable,q)
        return result


class DualAppearanceEncoder(nn.Module):
    def __init__(self,width=512):
        super().__init__()
        self.vae_projection = nn.Linear(64,width)
        self.dino_projection = nn.Linear(1536,width)
        self.modality_embedding = nn.Embedding(2,width)
        self.dino_queries = nn.Parameter(torch.randn(8,width)*width**-.5)
        self.dino_resampler = nn.ModuleList(MaskedAttentionBlock(width) for _ in range(2))

    def forward(self,source: SourceCondition):
        batch = source.rgb.shape[0]
        latents, mask = source.person_latents.flatten(0,1), source.latent_masks.flatten(0,1)
        mass = F.adaptive_avg_pool2d(mask,(2,4))
        weighted = F.adaptive_avg_pool2d(latents*mask,(2,4))/mass.clamp_min(1e-8)
        vae_valid = mass.flatten(1)>0
        vae = (self.vae_projection(weighted.flatten(2).transpose(1,2))+self.modality_embedding.weight[0])*vae_valid[...,None]
        patches = self.dino_projection(source.dino_patches.flatten(0,1))
        valid = source.dino_valid.flatten(0,1)
        dino_valid = valid.any(-1)[:,None].expand(-1,8)
        queries = self.dino_queries[None].expand(batch*2,-1,-1)
        for block in self.dino_resampler:
            queries = block(queries,patches,valid)
        bank = torch.cat((vae,(queries+self.modality_embedding.weight[1])*dino_valid[...,None]),dim=1)
        bank_valid = torch.cat((vae_valid,dino_valid),dim=1) & source.person_valid.flatten()[:,None]
        bank = bank*bank_valid[...,None]
        return bank.reshape(batch,2,16,-1), bank_valid.reshape(batch,2,16)
