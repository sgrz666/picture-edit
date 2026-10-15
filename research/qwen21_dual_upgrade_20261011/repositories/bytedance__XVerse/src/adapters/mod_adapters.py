# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass
from inspect import isfunction

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

from src.utils.data_utils import pad_to_square, pad_to_target

from transformers import CLIPProcessor, CLIPModel, CLIPVisionModelWithProjection, CLIPVisionModel

from collections import OrderedDict

class SquaredReLU(nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.square(torch.relu(x))

class AdaLayerNorm(nn.Module):
    def __init__(self, embedding_dim: int, time_embedding_dim: Optional[int] = None, ln_bias=True):
        super().__init__()

        if time_embedding_dim is None:
            time_embedding_dim = embedding_dim

        self.silu = nn.SiLU()
        self.linear = nn.Linear(time_embedding_dim, 2 * embedding_dim, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6, bias=ln_bias)

    def forward(
        self, x: torch.Tensor, timestep_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(timestep_embedding))
        shift, scale = emb.view(len(x), 1, -1).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x

class PerceiverAttentionBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, 
        time_embedding_dim: Optional[int] = None,
        double_kv: Optional[bool] = True,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.n_heads = n_heads

        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("sq_relu", SquaredReLU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.double_kv = double_kv
        self.ln_1 = AdaLayerNorm(d_model, time_embedding_dim)
        self.ln_2 = AdaLayerNorm(d_model, time_embedding_dim)
        self.ln_ff = AdaLayerNorm(d_model, time_embedding_dim)

    def attention(self, q: torch.Tensor, kv: torch.Tensor, attn_mask: torch.Tensor = None):
        attn_output, attn_output_weights = self.attn(q, kv, kv, need_weights=False, key_padding_mask=attn_mask)
        return attn_output

    def forward(
        self,
        x: torch.Tensor,
        latents: torch.Tensor,
        timestep_embedding: torch.Tensor = None,
        attn_mask: torch.Tensor = None
    ):
        normed_latents = self.ln_1(latents, timestep_embedding)
        normed_x = self.ln_2(x, timestep_embedding)
        if self.double_kv:
            kv = torch.cat([normed_latents, normed_x], dim=1)
        else:
            kv = normed_x
        attn = self.attention(
            q=normed_latents,
            kv=kv,
            attn_mask=attn_mask,
        )
        if attn_mask is not None:
            query_padding_mask = attn_mask.chunk(2, -1)[0].unsqueeze(-1) # (B, 2S) -> (B, S, 1)
            latents = latents + attn * (~query_padding_mask).to(attn)
        else:
            latents = latents + attn
        latents = latents + self.mlp(self.ln_ff(latents, timestep_embedding))
        return latents
   

class CLIPModAdapter(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self, 
        out_dim=3072,
        width=1024,
        pblock_width=512,
        layers=6,
        pblock_layers=1,
        heads=8,
        input_text_dim=4096,
        input_image_dim=1024,
        pblock_single_blocks=0,
    ):
        super().__init__()
        self.out_dim = out_dim

        self.net = TextImageResampler(
            width=width,
            layers=layers,
            heads=heads,
            input_text_dim=input_text_dim,
            input_image_dim=input_image_dim,
            time_embedding_dim=64,
            output_dim=out_dim,
        )
        self.net2 = TextImageResampler(
            width=pblock_width,
            layers=pblock_layers,
            heads=heads,
            input_text_dim=input_text_dim,
            input_image_dim=input_image_dim,
            time_embedding_dim=64,
            output_dim=out_dim*(19+pblock_single_blocks),
        )
        
    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True
        self.net.enable_gradient_checkpointing()
        self.net2.enable_gradient_checkpointing()


    def forward(self, t_emb, llm_hidden_states, clip_outputs):
        if len(llm_hidden_states.shape) > 3:
            llm_hidden_states = llm_hidden_states[..., -1, :]
        batch_size, seq_length = llm_hidden_states.shape[:2]

        img_cls_feat = clip_outputs["image_embeds"] # (B, 768)
        img_last_feat = clip_outputs["last_hidden_state"] # (B, 257, 1024)
        img_layer_feats = clip_outputs["hidden_states"] # [(B, 257, 1024) * 25]
        img_second_last_feat = img_layer_feats[-2] # (B, 257, 1024)

        img_hidden_states = img_second_last_feat # (B, 257, 1024)

        x = self.net(llm_hidden_states, img_hidden_states) # (B, S, 3072)
        x2 = self.net2(llm_hidden_states, img_hidden_states).view(batch_size, seq_length, -1, self.out_dim) # (B, S, N, 3072)
        return x, x2


class TextImageResampler(nn.Module):
    def __init__(
        self,
        width: int = 768,
        layers: int = 6,
        heads: int = 8,
        output_dim: int = 3072,
        input_text_dim: int = 4096,
        input_image_dim: int = 1024,
        time_embedding_dim: int = 64,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.input_text_dim = input_text_dim
        self.input_image_dim = input_image_dim
        self.time_embedding_dim = time_embedding_dim

        self.text_proj_in = nn.Linear(input_text_dim, width)
        self.image_proj_in = nn.Linear(input_image_dim, width)

        self.perceiver_blocks = nn.Sequential(
            *[
                PerceiverAttentionBlock(
                    width, heads, time_embedding_dim=self.time_embedding_dim
                )
                for _ in range(layers)
            ]
        )

        self.proj_out = nn.Sequential(
            nn.Linear(width, output_dim), nn.LayerNorm(output_dim)
        )
        
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True
        

    def forward(
        self, 
        text_hidden_states: torch.Tensor, 
        image_hidden_states: torch.Tensor,
    ):
        timestep_embedding = torch.zeros((text_hidden_states.shape[0], 1, self.time_embedding_dim)).to(text_hidden_states)

        text_hidden_states = self.text_proj_in(text_hidden_states)
        image_hidden_states = self.image_proj_in(image_hidden_states)

        for p_block in self.perceiver_blocks:
            if self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                text_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(p_block),
                    image_hidden_states,
                    text_hidden_states,
                    timestep_embedding
                )
            else:
                text_hidden_states = p_block(image_hidden_states, text_hidden_states, timestep_embedding=timestep_embedding)

        text_hidden_states = self.proj_out(text_hidden_states)

        return text_hidden_states
