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

import torch
from typing import List, Union, Optional, Tuple, Dict, Any, Callable
from diffusers.models.attention_processor import Attention, F
from .lora_controller import enable_lora
from einops import rearrange
import math
from diffusers.models.embeddings import apply_rotary_emb


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    B = query.size(0)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(B, 1, L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        assert False
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias.to(attn_weight.device)
    attn_weight = torch.softmax(attn_weight, dim=-1)

    return torch.dropout(attn_weight, dropout_p, train=True) @ value, attn_weight

def attn_forward(
    attn: Attention,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor = None,
    condition_latents: torch.FloatTensor = None,
    text_cond_mask: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
    cond_rotary_emb: Optional[torch.Tensor] = None,
    model_config: Optional[Dict[str, Any]] = {},
    store_attn_map: bool = False,
    latent_height: Optional[int] = None,
    timestep: Optional[torch.Tensor] = None,
    last_attn_map: Optional[torch.Tensor] = None,
    condition_sblora_weight: Optional[float] = None,
    latent_sblora_weight: Optional[float] = None,
) -> torch.FloatTensor:
    batch_size, _, _ = (
        hidden_states.shape
        if encoder_hidden_states is None
        else encoder_hidden_states.shape
    )
    
    is_sblock = encoder_hidden_states is None
    is_dblock = not is_sblock
    
    with enable_lora(
        (attn.to_q, attn.to_k, attn.to_v), 
        (is_dblock and model_config["latent_lora"]) or (is_sblock and model_config["sblock_lora"]), latent_sblora_weight=latent_sblora_weight
    ):
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
         
    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
    if encoder_hidden_states is not None:
        # `context` projections.
        with enable_lora((attn.add_q_proj, attn.add_k_proj, attn.add_v_proj), model_config["text_lora"]):
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

        # attention
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

    if image_rotary_emb is not None:
        query = apply_rotary_emb(query, image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb)

    if condition_latents is not None:
        assert condition_latents.shape[0] == batch_size
        cond_length = condition_latents.shape[1]

        cond_lora_activate = (is_dblock and model_config["use_condition_dblock_lora"]) or (is_sblock and model_config["use_condition_sblock_lora"])
        with enable_lora(
            (attn.to_q, attn.to_k, attn.to_v), 
            dit_activated=not cond_lora_activate, cond_activated=cond_lora_activate, latent_sblora_weight=condition_sblora_weight  #TODO implementation for condition lora not share
        ):
            cond_query = attn.to_q(condition_latents)
            cond_key = attn.to_k(condition_latents)
            cond_value = attn.to_v(condition_latents)

        cond_query = cond_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        cond_key = cond_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        cond_value = cond_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            cond_query = attn.norm_q(cond_query)
        if attn.norm_k is not None:
            cond_key = attn.norm_k(cond_key)

        if cond_rotary_emb is not None:
            cond_query = apply_rotary_emb(cond_query, cond_rotary_emb)
            cond_key = apply_rotary_emb(cond_key, cond_rotary_emb)

        if model_config.get("text_cond_attn", False):
            if encoder_hidden_states is not None:
                assert text_cond_mask is not None
                img_length = hidden_states.shape[1]
                seq_length = encoder_hidden_states_query_proj.shape[2]
                assert len(text_cond_mask.shape) == 2 or len(text_cond_mask.shape) == 3
                if len(text_cond_mask.shape) == 2:
                    text_cond_mask = text_cond_mask.unsqueeze(-1)
                N = text_cond_mask.shape[-1] # num_condition
            else:
                raise NotImplementedError()

            query = torch.cat([query, cond_query], dim=2) # (B, 24, S+HW+NC)
            key = torch.cat([key, cond_key], dim=2)
            value = torch.cat([value, cond_value], dim=2)
            
            assert query.shape[2] == key.shape[2]
            assert query.shape[2] == cond_length + img_length + seq_length

            attention_mask = torch.ones(batch_size, 1, query.shape[2], key.shape[2], device=query.device, dtype=torch.bool)
            attention_mask[..., -cond_length:, :-cond_length] = False
            attention_mask[..., :-cond_length, -cond_length:] = False

            if encoder_hidden_states is not None:
                tokens_per_cond = cond_length // N
                for i in range(batch_size):
                    for j in range(N):
                        start = seq_length + img_length + tokens_per_cond * j
                        attention_mask[i, 0, :seq_length, start:start+tokens_per_cond] = text_cond_mask[i, :, j].unsqueeze(-1)

        elif model_config.get("union_cond_attn", False):
            query = torch.cat([query, cond_query], dim=2) # (B, 24, S+HW+NC)
            key = torch.cat([key, cond_key], dim=2)
            value = torch.cat([value, cond_value], dim=2)
            
            attention_mask = torch.ones(batch_size, 1, query.shape[2], key.shape[2], device=query.device, dtype=torch.bool)
            cond_length = condition_latents.shape[1]
            assert len(text_cond_mask.shape) == 2 or len(text_cond_mask.shape) == 3
            if len(text_cond_mask.shape) == 2:
                text_cond_mask = text_cond_mask.unsqueeze(-1)
            N = text_cond_mask.shape[-1] # num_condition
            tokens_per_cond = cond_length // N
            
            seq_length = 0
            if encoder_hidden_states is not None:
                seq_length = encoder_hidden_states_query_proj.shape[2]
                img_length = hidden_states.shape[1]
            else:
                seq_length = 128 # TODO, pass it here
                img_length = hidden_states.shape[1] - seq_length
            
            if not model_config.get("cond_cond_cross_attn", True):
                # no cross attention between different conds
                cond_start = seq_length + img_length
                attention_mask[:, :, cond_start:, cond_start:] = False
                
                for j in range(N):
                    start = cond_start + tokens_per_cond * j
                    end = cond_start + tokens_per_cond * (j + 1)
                    attention_mask[..., start:end, start:end] = True
            
            # double block
            if encoder_hidden_states is not None:
                
                # no cross attention
                attention_mask[..., :-cond_length, -cond_length:] = False

                if model_config.get("use_attention_double", False) and last_attn_map is not None:
                    attention_mask = torch.zeros(batch_size, 1, query.shape[2], key.shape[2], device=query.device, dtype=torch.bfloat16)
                    last_attn_map = last_attn_map.to(query.device)
                    attention_mask[..., seq_length:-cond_length, :seq_length] = torch.log(last_attn_map/last_attn_map.mean()*model_config["use_atten_lambda"]).view(-1, seq_length)
                
            # single block
            else:
                # print(last_attn_map)
                if model_config.get("use_attention_single", False) and last_attn_map is not None:
                    attention_mask = torch.zeros(batch_size, 1, query.shape[2], key.shape[2], device=query.device, dtype=torch.bfloat16)
                    attention_mask[..., :seq_length, -cond_length:] = float('-inf')
                    # 确保 use_atten_lambda 是列表
                    use_atten_lambdas = model_config["use_atten_lambda"] if len(model_config["use_atten_lambda"])!=1 else model_config["use_atten_lambda"] * (N+1)
                    attention_mask[..., -cond_length:, seq_length:-cond_length] = math.log(use_atten_lambdas[0])
                    last_attn_map = last_attn_map.to(query.device)
                    
                    cond2latents = []
                    for i in range(batch_size):
                        AM = last_attn_map[i] # (H, W, S)
                        for j in range(N):
                            start = seq_length + img_length + tokens_per_cond * j
                            mask = text_cond_mask[i, :, j] # (S,)
                            weighted_AM = AM * mask.unsqueeze(0).unsqueeze(0)  # 扩展 mask 维度以匹配 AM

                            cond2latent = weighted_AM.mean(-1)
                            if model_config.get("attention_norm", "mean") == "max":
                                cond2latent = cond2latent / cond2latent.max()  # 归一化
                            else:
                                cond2latent = cond2latent / cond2latent.mean()  # 归一化
                            cond2latent = cond2latent.view(-1,) # (WH,)

                            # 使用对应 condition 的 lambda 值
                            current_lambda = use_atten_lambdas[j+1]
                            # 将 cond2latent 复制到 attention_mask[i, 0, :seq_length, start:start+tokens_per_cond]
                            attention_mask[i, 0, seq_length:-cond_length, start:start+tokens_per_cond] = torch.log(current_lambda * cond2latent.unsqueeze(-1))
                            
                            # 将 text_cond_mask[i, :, j].unsqueeze(-1) 为 true 的位置设置为当前 lambda 值
                            cond = mask.unsqueeze(-1).expand(-1, tokens_per_cond)
                            sub_mask = attention_mask[i, 0, :seq_length, start:start+tokens_per_cond]
                            attention_mask[i, 0, :seq_length, start:start+tokens_per_cond] = torch.where(cond, math.log(current_lambda), sub_mask)
                            cond2latents.append(
                                cond2latent.reshape(latent_height, -1).detach().cpu()
                            )
                    if store_attn_map:
                        if not hasattr(attn, "cond2latents"):
                            attn.cond2latents = []
                            attn.cond_timesteps = []
                        attn.cond2latents.append(torch.stack(cond2latents, dim=0)) # (N, H, W)
                        attn.cond_timesteps.append(timestep.cpu())

                pass
        else:
            raise NotImplementedError()
        if hasattr(attn, "c_factor"):
            assert False
            attention_mask = torch.zeros(
                query.shape[2], key.shape[2], device=query.device, dtype=query.dtype
            )
            bias = torch.log(attn.c_factor[0])
            attention_mask[-cond_length:, :-cond_length] = bias
            attention_mask[:-cond_length, -cond_length:] = bias

    ####################################################################################################
    if store_attn_map and encoder_hidden_states is not None:
        seq_length = encoder_hidden_states_query_proj.shape[2]
        img_length = hidden_states.shape[1]
        hidden_states, attention_probs = scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        # (B, 24, S+HW, S+HW) -> (B, 24, HW, S)
        t2i_attention_probs = attention_probs[:, :, seq_length:seq_length+img_length, :seq_length]
        # (B, 24, S+HW, S+HW) -> (B, 24, S, HW) -> (B, 24, HW, S)
        i2t_attention_probs = attention_probs[:, :, :seq_length, seq_length:seq_length+img_length].transpose(-1, -2)
        
        if not hasattr(attn, "attn_maps"):
            attn.attn_maps = []
            attn.timestep = []

        attn.attn_maps.append(
            (
                rearrange(t2i_attention_probs, 'B attn_head (H W) attn_dim -> B attn_head H W attn_dim', H=latent_height),
                rearrange(i2t_attention_probs, 'B attn_head (H W) attn_dim -> B attn_head H W attn_dim', H=latent_height),
            )
        )

        attn.timestep.append(timestep.cpu())
        has_nan = torch.isnan(hidden_states).any().item()
        if has_nan:
            print("[attn_forward] detect nan hidden_states in store_attn_map")
    else:
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        has_nan = torch.isnan(hidden_states).any().item()
        if has_nan:
            print("[attn_forward] detect nan hidden_states")
    ####################################################################################################
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).to(query.dtype)

    if encoder_hidden_states is not None:
        if condition_latents is not None:
            encoder_hidden_states, hidden_states, condition_latents = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[
                    :, encoder_hidden_states.shape[1] : -condition_latents.shape[1]
                ],
                hidden_states[:, -condition_latents.shape[1] :],
            )
            if model_config.get("latent_cond_by_text_attn", False):
                # hidden_states += add_latent # (B, HW, D)
                hidden_states = new_hidden_states # (B, HW, D)
            
        else:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )


        with enable_lora((attn.to_out[0],), model_config["latent_lora"]):
            hidden_states = attn.to_out[0](hidden_states) # linear proj
            hidden_states = attn.to_out[1](hidden_states) # dropout
        with enable_lora((attn.to_add_out,), model_config["text_lora"]):
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if condition_latents is not None:
            cond_lora_activate = model_config["use_condition_dblock_lora"]
            with enable_lora(
                (attn.to_out[0],), 
                dit_activated=not cond_lora_activate, cond_activated=cond_lora_activate,
            ):
                condition_latents = attn.to_out[0](condition_latents)
                condition_latents = attn.to_out[1](condition_latents)


        return (
            (hidden_states, encoder_hidden_states, condition_latents)
            if condition_latents is not None
            else (hidden_states, encoder_hidden_states)
        )
    elif condition_latents is not None:
        hidden_states, condition_latents = (
            hidden_states[:, : -condition_latents.shape[1]],
            hidden_states[:, -condition_latents.shape[1] :],
        )
        return hidden_states, condition_latents
    else:
        return hidden_states


def set_delta_by_start_end(
    start_ends, 
    src_delta_emb, src_delta_emb_pblock, 
    delta_emb, delta_emb_pblock, delta_emb_mask, 
):
    for (i, j, src_s, src_e, tar_s, tar_e) in start_ends:
        if src_delta_emb is not None:
            delta_emb[i, tar_s:tar_e] = src_delta_emb[j, src_s:src_e]
        if src_delta_emb_pblock is not None:
            delta_emb_pblock[i, tar_s:tar_e] = src_delta_emb_pblock[j, src_s:src_e]
        delta_emb_mask[i, tar_s:tar_e] = True
    return delta_emb, delta_emb_pblock, delta_emb_mask

def norm1_context_forward(
    norm1_context,
    x: torch.Tensor,
    condition_latents: Optional[torch.Tensor] = None, 
    timestep: Optional[torch.Tensor] = None,
    class_labels: Optional[torch.LongTensor] = None,
    hidden_dtype: Optional[torch.dtype] = None,
    emb: Optional[torch.Tensor] = None,
    delta_emb: Optional[torch.Tensor] = None,
    delta_emb_cblock: Optional[torch.Tensor] = None,
    delta_emb_mask: Optional[torch.Tensor] = None,
    delta_start_ends = None,
    mod_adapter = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, seq_length = x.shape[:2]
    
    if mod_adapter is not None:
        assert False

    if delta_emb is None:
        emb = norm1_context.linear(norm1_context.silu(emb)) # (B, 3072) -> (B, 18432)
        emb = emb.unsqueeze(1) # (B, 1, 18432)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1) # (B, 1, 3072)
        x = norm1_context.norm(x) * (1 + scale_msa) + shift_msa # (B, 1, 3072)
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp
    else:
        # (B, 3072) > (B, 18432) -> (B, S, 18432)
        emb_orig = norm1_context.linear(norm1_context.silu(emb)).unsqueeze(1).expand((-1, seq_length, -1))
        # (B, 3072) -> (B, 1, 3072) -> (B, S, 3072) -> (B, S, 18432)
        if delta_emb_cblock is None:
            emb_new = norm1_context.linear(norm1_context.silu(emb.unsqueeze(1) + delta_emb))
        else:
            emb_new = norm1_context.linear(norm1_context.silu(emb.unsqueeze(1) + delta_emb + delta_emb_cblock))
        emb = torch.where(delta_emb_mask.unsqueeze(-1), emb_new, emb_orig) # (B, S, 18432)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1) # (B, S, 3072)
        x = norm1_context.norm(x) * (1 + scale_msa) + shift_msa # (B, S, 3072)
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp
        

def norm1_forward(
    norm1,
    x: torch.Tensor,
    timestep: Optional[torch.Tensor] = None,
    class_labels: Optional[torch.LongTensor] = None,
    hidden_dtype: Optional[torch.dtype] = None,
    emb: Optional[torch.Tensor] = None,
    delta_emb: Optional[torch.Tensor] = None,
    delta_emb_cblock: Optional[torch.Tensor] = None,
    delta_emb_mask: Optional[torch.Tensor] = None,
    t2i_attn_map: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if delta_emb is None:
        emb = norm1.linear(norm1.silu(emb)) # (B, 3072) -> (B, 18432)
        emb = emb.unsqueeze(1) # (B, 1, 18432)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1) # (B, 1, 3072)
        x = norm1.norm(x) * (1 + scale_msa) + shift_msa # (B, 1, 3072)
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp
    else:
        raise NotImplementedError()
        batch_size, HW = x.shape[:2]
        seq_length = t2i_attn_map.shape[-1]
        # (B, 3072) > (B, 18432) -> (B, S, 18432)
        emb_orig = norm1.linear(norm1.silu(emb)).unsqueeze(1).expand((-1, seq_length, -1))
        # (B, 3072) -> (B, 1, 3072) -> (B, S, 3072) -> (B, S, 18432)
        if delta_emb_cblock is None:
            emb_new = norm1.linear(norm1.silu(emb.unsqueeze(1) + delta_emb))
        else:
            emb_new = norm1.linear(norm1.silu(emb.unsqueeze(1) + delta_emb + delta_emb_cblock))
        # attn_weight (B, HW, S)
        emb = torch.where(delta_emb_mask.unsqueeze(-1), emb_new, emb_orig) # (B, S, 18432)
        emb = t2i_attn_map @ emb    # (B, HW, 18432)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1) # (B, HW, 3072)
        x = norm1.norm(x) * (1 + scale_msa) + shift_msa # (B, HW, 3072)
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


def block_forward(
    block,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor,
    condition_latents: torch.FloatTensor,
    temb: torch.FloatTensor,
    cond_temb: torch.FloatTensor,
    text_cond_mask: Optional[torch.FloatTensor] = None,
    delta_emb: Optional[torch.FloatTensor] = None,
    delta_emb_cblock: Optional[torch.FloatTensor] = None,
    delta_emb_mask: Optional[torch.Tensor] = None,
    delta_start_ends = None,
    cond_rotary_emb=None,
    image_rotary_emb=None,
    model_config: Optional[Dict[str, Any]] = {},
    store_attn_map: bool = False,
    use_text_mod: bool = True,
    use_img_mod: bool = False,
    mod_adapter = None,
    latent_height: Optional[int] = None,
    timestep: Optional[torch.Tensor] = None,
    last_attn_map: Optional[torch.Tensor] = None,
):
    batch_size = hidden_states.shape[0]
    use_cond = condition_latents is not None

    train_partial_latent_lora = model_config.get("train_partial_latent_lora", False)
    train_partial_text_lora = model_config.get("train_partial_text_lora", False)
    if train_partial_latent_lora:
        train_partial_latent_lora_layers = model_config.get("train_partial_latent_lora_layers", "")
        activate_norm1 = activate_ff = True
        if "norm1" not in train_partial_latent_lora_layers:
            activate_norm1 = False
        if "ff" not in train_partial_latent_lora_layers:
            activate_ff = False
    
    if train_partial_text_lora:
        train_partial_text_lora_layers = model_config.get("train_partial_text_lora_layers", "")
        activate_norm1_context = activate_ff_context = True
        if "norm1" not in train_partial_text_lora_layers:
            activate_norm1_context = False
        if "ff" not in train_partial_text_lora_layers:
            activate_ff_context = False
    
    if use_cond:
        cond_lora_activate = model_config["use_condition_dblock_lora"]
        with enable_lora(
            (block.norm1.linear,), 
            dit_activated=activate_norm1 if train_partial_latent_lora else not cond_lora_activate, cond_activated=cond_lora_activate,
        ):
            norm_condition_latents, cond_gate_msa, cond_shift_mlp, cond_scale_mlp, cond_gate_mlp = (
                norm1_forward(
                    block.norm1,
                    condition_latents,
                    emb=cond_temb, 
                )
            )
    delta_emb_img = delta_emb_img_cblock = None
    if use_img_mod and use_text_mod:
        if delta_emb is not None:
            delta_emb_img, delta_emb = delta_emb.chunk(2, dim=-1)
        if delta_emb_cblock is not None:
            delta_emb_img_cblock, delta_emb_cblock = delta_emb_cblock.chunk(2, dim=-1)

    with enable_lora((block.norm1.linear,), activate_norm1 if train_partial_latent_lora else model_config["latent_lora"]):
        if use_img_mod and encoder_hidden_states is not None:
            with torch.no_grad():
                attn = block.attn

                norm_img = block.norm1(hidden_states, emb=temb)[0]
                norm_text = block.norm1_context(encoder_hidden_states, emb=temb)[0]
                
                img_query = attn.to_q(norm_img)
                img_key = attn.to_k(norm_img)
                text_query = attn.add_q_proj(norm_text)
                text_key = attn.add_k_proj(norm_text)

                inner_dim = img_key.shape[-1]
                head_dim = inner_dim // attn.heads

                img_query = img_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)   # (B, N, HW, D)
                img_key = img_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)       # (B, N, HW, D)
                text_query = text_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) # (B, N, S, D)
                text_key = text_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)     # (B, N, S, D)

                if attn.norm_q is not None:
                    img_query = attn.norm_q(img_query)
                if attn.norm_added_q is not None:
                    text_query = attn.norm_added_q(text_query)
                if attn.norm_k is not None:
                    img_key = attn.norm_k(img_key)
                if attn.norm_added_k is not None:
                    text_key = attn.norm_added_k(text_key)
                
                query = torch.cat([text_query, img_query], dim=2) # (B, N, S+HW, D)
                key = torch.cat([text_key, img_key], dim=2)       # (B, N, S+HW, D)
                if image_rotary_emb is not None:
                    query = apply_rotary_emb(query, image_rotary_emb)
                    key = apply_rotary_emb(key, image_rotary_emb)

                seq_length = text_query.shape[2]

                scale_factor = 1 / math.sqrt(query.size(-1))
                t2i_attn_map = query @ key.transpose(-2, -1) * scale_factor # (B, N, S+HW, S+HW)
                t2i_attn_map = t2i_attn_map.mean(1)[:, seq_length:, :seq_length] # (B, S+HW, S+HW) -> (B, HW, S)
                t2i_attn_map = torch.softmax(t2i_attn_map, dim=-1) # (B, HW, S)

        else:
            t2i_attn_map = None

        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            norm1_forward(
                block.norm1,
                hidden_states,
                emb=temb,
                delta_emb=delta_emb_img,
                delta_emb_cblock=delta_emb_img_cblock,
                delta_emb_mask=delta_emb_mask,
                t2i_attn_map=t2i_attn_map,
            )
        )
    # Modulation for double block
    with enable_lora((block.norm1_context.linear,), activate_norm1_context if train_partial_text_lora else model_config["text_lora"]):
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
            norm1_context_forward(
                block.norm1_context, 
                encoder_hidden_states, 
                emb=temb, 
                delta_emb=delta_emb if use_text_mod else None,
                delta_emb_cblock=delta_emb_cblock if use_text_mod else None,
                delta_emb_mask=delta_emb_mask if use_text_mod else None,
                delta_start_ends=delta_start_ends if use_text_mod else None,
                mod_adapter=mod_adapter,
                condition_latents=condition_latents,
            )
        )

    # Attention.
    result = attn_forward(
        block.attn,
        model_config=model_config,
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        condition_latents=norm_condition_latents if use_cond else None,
        text_cond_mask=text_cond_mask if use_cond else None,
        image_rotary_emb=image_rotary_emb,
        cond_rotary_emb=cond_rotary_emb if use_cond else None,
        store_attn_map=store_attn_map,
        latent_height=latent_height,
        timestep=timestep,
        last_attn_map=last_attn_map,
    )
    attn_output, context_attn_output = result[:2]
    cond_attn_output = result[2] if use_cond else None

    # Process attention outputs for the `hidden_states`.
    # 1. hidden_states
    attn_output = gate_msa * attn_output # NOTE: changed by img mod
    hidden_states = hidden_states + attn_output
    # 2. encoder_hidden_states
    context_attn_output = c_gate_msa * context_attn_output # NOTE: changed by delta_temb
    encoder_hidden_states = encoder_hidden_states + context_attn_output
    # 3. condition_latents
    if use_cond:
        cond_attn_output = cond_gate_msa * cond_attn_output  # NOTE: changed by img mod
        condition_latents = condition_latents + cond_attn_output
        if model_config.get("add_cond_attn", False):
            hidden_states += cond_attn_output

    # LayerNorm + MLP.
    # 1. hidden_states
    norm_hidden_states = block.norm2(hidden_states)
    norm_hidden_states = (
        norm_hidden_states * (1 + scale_mlp) + shift_mlp  # NOTE: changed by img mod
    )
    # 2. encoder_hidden_states
    norm_encoder_hidden_states = block.norm2_context(encoder_hidden_states)
    norm_encoder_hidden_states = (
        norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp  # NOTE: changed by delta_temb
    )
    # 3. condition_latents
    if use_cond:
        norm_condition_latents = block.norm2(condition_latents)
        norm_condition_latents = (
            norm_condition_latents * (1 + cond_scale_mlp) + cond_shift_mlp # NOTE: changed by img mod
        )

    # Feed-forward.
    with enable_lora((block.ff.net[2],), activate_ff if train_partial_latent_lora else model_config["latent_lora"]):
        # 1. hidden_states
        ff_output = block.ff(norm_hidden_states)
        ff_output = gate_mlp * ff_output  # NOTE: changed by img mod
    # 2. encoder_hidden_states
    with enable_lora((block.ff_context.net[2],), activate_ff_context if train_partial_text_lora else model_config["text_lora"]):
        context_ff_output = block.ff_context(norm_encoder_hidden_states)
        context_ff_output = c_gate_mlp * context_ff_output  # NOTE: changed by delta_temb
    # 3. condition_latents
    if use_cond:
        cond_lora_activate = model_config["use_condition_dblock_lora"]
        with enable_lora(
            (block.ff.net[2],), 
            dit_activated=activate_ff if train_partial_latent_lora else not cond_lora_activate, cond_activated=cond_lora_activate,
        ):
            cond_ff_output = block.ff(norm_condition_latents)
            cond_ff_output = cond_gate_mlp * cond_ff_output  # NOTE: changed by img mod

    # Process feed-forward outputs.
    hidden_states = hidden_states + ff_output
    encoder_hidden_states = encoder_hidden_states + context_ff_output
    if use_cond:
        condition_latents = condition_latents + cond_ff_output

    # Clip to avoid overflow.
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

    return encoder_hidden_states, hidden_states, condition_latents if use_cond else None

def single_norm_forward(
    block,
    x: torch.Tensor,
    timestep: Optional[torch.Tensor] = None,
    class_labels: Optional[torch.LongTensor] = None,
    hidden_dtype: Optional[torch.dtype] = None,
    emb: Optional[torch.Tensor] = None,
    delta_emb: Optional[torch.Tensor] = None,
    delta_emb_cblock: Optional[torch.Tensor] = None,
    delta_emb_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if delta_emb is None:
        emb = block.linear(block.silu(emb)) # (B, 3072) -> (B, 9216)
        emb = emb.unsqueeze(1) # (B, 1, 9216)
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=-1) # (B, 1, 3072)
        x = block.norm(x) * (1 + scale_msa) + shift_msa # (B, S, 3072) * (B, 1, 3072)
        return x, gate_msa
    else:
        img_text_seq_length = x.shape[1] # S+
        text_seq_length = delta_emb_mask.shape[1] # S
        # (B, 3072) -> (B, 9216) -> (B, S+, 9216)
        emb_orig = block.linear(block.silu(emb)).unsqueeze(1).expand((-1, img_text_seq_length, -1))
        # (B, 3072) -> (B, 1, 3072) -> (B, S, 3072) -> (B, S, 9216)
        if delta_emb_cblock is None:
            emb_new = block.linear(block.silu(emb.unsqueeze(1) + delta_emb))
        else:
            emb_new = block.linear(block.silu(emb.unsqueeze(1) + delta_emb + delta_emb_cblock))

        emb_text = torch.where(delta_emb_mask.unsqueeze(-1), emb_new, emb_orig[:, :text_seq_length]) # (B, S, 9216)
        emb_img = emb_orig[:, text_seq_length:] # (B, s, 9216)
        emb = torch.cat([emb_text, emb_img], dim=1) # (B, S+, 9216)

        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=-1) # (B, S+, 3072)
        x = block.norm(x) * (1 + scale_msa) + shift_msa # (B, S+, 3072)
        return x, gate_msa


def single_block_forward(
    block,
    hidden_states: torch.FloatTensor,
    temb: torch.FloatTensor,
    image_rotary_emb=None,
    condition_latents: torch.FloatTensor = None,
    text_cond_mask: torch.FloatTensor = None,
    cond_temb: torch.FloatTensor = None,
    delta_emb: Optional[torch.FloatTensor] = None,
    delta_emb_cblock: Optional[torch.FloatTensor] = None,
    delta_emb_mask: Optional[torch.Tensor] = None,
    use_text_mod: bool = True,
    use_img_mod: bool = False,
    cond_rotary_emb=None,
    latent_height: Optional[int] = None,
    timestep: Optional[torch.Tensor] = None,
    store_attn_map: bool = False,
    model_config: Optional[Dict[str, Any]] = {},
    last_attn_map: Optional[torch.Tensor] = None,
    latent_sblora_weight=None,
    condition_sblora_weight=None,
):

    using_cond = condition_latents is not None
    residual = hidden_states
    
    train_partial_lora = model_config.get("train_partial_lora", False)
    if train_partial_lora:
        train_partial_lora_layers = model_config.get("train_partial_lora_layers", "")
        activate_norm = activate_projmlp = activate_projout = True
        
        if "norm" not in train_partial_lora_layers:
            activate_norm = False
        if "projmlp" not in train_partial_lora_layers:
            activate_projmlp = False
        if "projout" not in train_partial_lora_layers:
            activate_projout = False

    with enable_lora((block.norm.linear,), activate_norm if train_partial_lora else model_config["sblock_lora"], latent_sblora_weight=latent_sblora_weight):
        # Modulation for single block
        norm_hidden_states, gate = single_norm_forward(
            block.norm, 
            hidden_states, 
            emb=temb,
            delta_emb=delta_emb if use_text_mod else None,
            delta_emb_cblock=delta_emb_cblock if use_text_mod else None,
            delta_emb_mask=delta_emb_mask if use_text_mod else None,
        )
    with enable_lora((block.proj_mlp,), activate_projmlp if train_partial_lora else model_config["sblock_lora"], latent_sblora_weight=latent_sblora_weight):
        mlp_hidden_states = block.act_mlp(block.proj_mlp(norm_hidden_states))
    if using_cond:
        cond_lora_activate = model_config["use_condition_sblock_lora"]
        with enable_lora(
            (block.norm.linear,), 
            dit_activated=activate_norm if train_partial_lora else not cond_lora_activate, cond_activated=cond_lora_activate, latent_sblora_weight=condition_sblora_weight
        ):
            residual_cond = condition_latents
            norm_condition_latents, cond_gate = block.norm(condition_latents, emb=cond_temb)
        with enable_lora(
            (block.proj_mlp,), 
            dit_activated=activate_projmlp if train_partial_lora else not cond_lora_activate, cond_activated=cond_lora_activate, latent_sblora_weight=condition_sblora_weight
        ):
            mlp_cond_hidden_states = block.act_mlp(block.proj_mlp(norm_condition_latents))

    attn_output = attn_forward(
        block.attn,
        model_config=model_config,
        hidden_states=norm_hidden_states,
        image_rotary_emb=image_rotary_emb,
        last_attn_map=last_attn_map,
        latent_height=latent_height,
        store_attn_map=store_attn_map,
        timestep=timestep,
        latent_sblora_weight=latent_sblora_weight,
        condition_sblora_weight=condition_sblora_weight,
        **(
            {
                "condition_latents": norm_condition_latents,
                "cond_rotary_emb": cond_rotary_emb if using_cond else None,
                "text_cond_mask": text_cond_mask if using_cond else None,
            }
            if using_cond
            else {}
        ),
    )
    if using_cond:
        attn_output, cond_attn_output = attn_output

    with enable_lora((block.proj_out,), activate_projout if train_partial_lora else model_config["sblock_lora"], latent_sblora_weight=latent_sblora_weight):
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        # gate = (B, 1, 3072) or (B, S+, 3072)
        hidden_states = gate * block.proj_out(hidden_states)
        hidden_states = residual + hidden_states
    if using_cond:
        cond_lora_activate = model_config["use_condition_sblock_lora"]
        with enable_lora(
            (block.proj_out,), 
            dit_activated=activate_projout if train_partial_lora else not cond_lora_activate, cond_activated=cond_lora_activate, latent_sblora_weight=condition_sblora_weight
        ):
            condition_latents = torch.cat([cond_attn_output, mlp_cond_hidden_states], dim=2)
            cond_gate = cond_gate.unsqueeze(1)
            condition_latents = cond_gate * block.proj_out(condition_latents)
            condition_latents = residual_cond + condition_latents

    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)

    return hidden_states if not using_cond else (hidden_states, condition_latents)
