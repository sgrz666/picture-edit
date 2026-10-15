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
from diffusers.pipelines import FluxPipeline
from typing import List, Union, Optional, Dict, Any, Callable
from .block import block_forward, single_block_forward
from .lora_controller import enable_lora
from diffusers.models.transformers.transformer_flux import (
    FluxTransformer2DModel,
    Transformer2DModelOutput,
    USE_PEFT_BACKEND,
    is_torch_version,
    scale_lora_layers,
    unscale_lora_layers,
    logger,
)
import numpy as np

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def prepare_params(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_block_samples=None,
    controlnet_single_block_samples=None,
    return_dict: bool = True,
    **kwargs: dict,
):
    return (
        hidden_states,
        encoder_hidden_states,
        pooled_projections,
        timestep,
        img_ids,
        txt_ids,
        guidance,
        joint_attention_kwargs,
        controlnet_block_samples,
        controlnet_single_block_samples,
        return_dict,
    )


def transformer_forward(
    transformer: FluxTransformer2DModel,
    condition_latents: torch.Tensor,
    condition_ids: torch.Tensor,
    condition_type_ids: torch.Tensor,
    model_config: Optional[Dict[str, Any]] = {},
    c_t=0,
    text_cond_mask: Optional[torch.FloatTensor] = None,
    delta_emb: Optional[torch.FloatTensor] = None,
    delta_emb_pblock: Optional[torch.FloatTensor] = None,
    delta_emb_mask: Optional[torch.FloatTensor] = None,
    delta_start_ends = None,
    store_attn_map: bool = False,
    use_text_mod: bool = True,
    use_img_mod: bool = False,
    mod_adapter = None,
    latent_height: Optional[int] = None,
    last_attn_map = None,
    **params: dict,
):
    
    use_condition = condition_latents is not None

    (
        hidden_states,
        encoder_hidden_states,
        pooled_projections,
        timestep,
        img_ids,
        txt_ids,
        guidance,
        joint_attention_kwargs,
        controlnet_block_samples,
        controlnet_single_block_samples,
        return_dict,
    ) = prepare_params(**params)

    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        latent_sblora_weight = joint_attention_kwargs.pop("latent_sblora_weight", None)
        condition_sblora_weight = joint_attention_kwargs.pop("condition_sblora_weight", None)
    else:
        lora_scale = 1.0
        latent_sblora_weight = None
        condition_sblora_weight = None
    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(transformer, lora_scale)
    else:
        if (
            joint_attention_kwargs is not None
            and joint_attention_kwargs.get("scale", None) is not None
        ):
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )

    train_partial_text_lora = model_config.get("train_partial_text_lora", False)
    train_partial_latent_lora = model_config.get("train_partial_latent_lora", False)

    if train_partial_text_lora or train_partial_latent_lora:
        train_partial_text_lora_layers = model_config.get("train_partial_text_lora_layers", "")
        train_partial_latent_lora_layers = model_config.get("train_partial_latent_lora_layers", "")
        activate_x_embedder = True
        if "x_embedder" not in train_partial_text_lora_layers or "x_embedder" not in train_partial_latent_lora_layers:
            activate_x_embedder = False
    if train_partial_text_lora or train_partial_latent_lora:
        activate_x_embedder_ = activate_x_embedder
    else:
        activate_x_embedder_ = model_config["latent_lora"] or model_config["text_lora"]
    
    with enable_lora((transformer.x_embedder,), activate_x_embedder_):
        hidden_states = transformer.x_embedder(hidden_states)
    cond_lora_activate = model_config["use_condition_dblock_lora"] or model_config["use_condition_sblock_lora"]
    with enable_lora(
        (transformer.x_embedder,), 
        dit_activated=activate_x_embedder if train_partial_text_lora or train_partial_latent_lora else not cond_lora_activate, cond_activated=cond_lora_activate,
    ):
        condition_latents = transformer.x_embedder(condition_latents) if use_condition else None

    timestep = timestep.to(hidden_states.dtype) * 1000

    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
    else:
        guidance = None

    temb = (
        transformer.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else transformer.time_text_embed(timestep, guidance, pooled_projections)
    ) # (B, 3072)

    cond_temb = (
        transformer.time_text_embed(torch.ones_like(timestep) * c_t * 1000, pooled_projections)
        if guidance is None
        else transformer.time_text_embed(
            torch.ones_like(timestep) * c_t * 1000, guidance, pooled_projections
        )
    )
    encoder_hidden_states = transformer.context_embedder(encoder_hidden_states)

    if txt_ids.ndim == 3:
        logger.warning(
            "Passing `txt_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        logger.warning(
            "Passing `img_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        img_ids = img_ids[0]

    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = transformer.pos_embed(ids)
    if use_condition:
        cond_rotary_emb = transformer.pos_embed(condition_ids)

    for index_block, block in enumerate(transformer.transformer_blocks):
        if delta_emb_pblock is None:
            delta_emb_cblock = None
        else:
            delta_emb_cblock = delta_emb_pblock[:, :, index_block]
        condition_pass_to_double = use_condition and (model_config["double_use_condition"] or model_config["single_use_condition"])
        if transformer.training and transformer.gradient_checkpointing:
            ckpt_kwargs: Dict[str, Any] = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )
            
            encoder_hidden_states, hidden_states, condition_latents = (
                torch.utils.checkpoint.checkpoint(
                    block_forward,
                    block=block,
                    model_config=model_config,
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    condition_latents=condition_latents if condition_pass_to_double else None,
                    cond_temb=cond_temb if condition_pass_to_double else None,
                    cond_rotary_emb=cond_rotary_emb if condition_pass_to_double else None,
                    temb=temb,
                    text_cond_mask=text_cond_mask,
                    delta_emb=delta_emb,
                    delta_emb_cblock=delta_emb_cblock,
                    delta_emb_mask=delta_emb_mask,
                    delta_start_ends=delta_start_ends,
                    image_rotary_emb=image_rotary_emb,
                    store_attn_map=store_attn_map,
                    use_text_mod=use_text_mod,
                    use_img_mod=use_img_mod,
                    mod_adapter=mod_adapter,
                    latent_height=latent_height,
                    timestep=timestep,
                    last_attn_map=last_attn_map,
                    **ckpt_kwargs,
                )
            )

        else:
            encoder_hidden_states, hidden_states, condition_latents = block_forward(
                block,
                model_config=model_config,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                condition_latents=condition_latents if condition_pass_to_double else None,
                cond_temb=cond_temb if condition_pass_to_double else None,
                cond_rotary_emb=cond_rotary_emb if condition_pass_to_double else None,
                temb=temb,
                text_cond_mask=text_cond_mask,
                delta_emb=delta_emb,
                delta_emb_cblock=delta_emb_cblock,
                delta_emb_mask=delta_emb_mask,
                delta_start_ends=delta_start_ends,
                image_rotary_emb=image_rotary_emb,
                store_attn_map=store_attn_map,
                use_text_mod=use_text_mod,
                use_img_mod=use_img_mod,
                mod_adapter=mod_adapter,
                latent_height=latent_height,
                timestep=timestep,
                last_attn_map=last_attn_map,
            )

        # controlnet residual
        if controlnet_block_samples is not None:
            interval_control = len(transformer.transformer_blocks) / len(
                controlnet_block_samples
            )
            interval_control = int(np.ceil(interval_control))
            hidden_states = (
                hidden_states
                + controlnet_block_samples[index_block // interval_control]
            )
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    for index_block, block in enumerate(transformer.single_transformer_blocks):
        if delta_emb_pblock is not None and delta_emb_pblock.shape[2] > 19+index_block:
            delta_emb_single = delta_emb
            delta_emb_cblock = delta_emb_pblock[:, :, index_block+19]
        else:
            delta_emb_single = None
            delta_emb_cblock = None
        if transformer.training and transformer.gradient_checkpointing:
            ckpt_kwargs: Dict[str, Any] = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )
            result = torch.utils.checkpoint.checkpoint(
                single_block_forward,
                block=block,
                model_config=model_config,
                hidden_states=hidden_states,
                temb=temb,
                delta_emb=delta_emb_single,
                delta_emb_cblock=delta_emb_cblock,
                delta_emb_mask=delta_emb_mask,
                use_text_mod=use_text_mod,
                use_img_mod=use_img_mod,
                image_rotary_emb=image_rotary_emb,
                last_attn_map=last_attn_map,
                latent_height=latent_height,
                timestep=timestep,
                store_attn_map=store_attn_map,
                **(
                    {
                        "condition_latents": condition_latents,
                        "cond_temb": cond_temb,
                        "cond_rotary_emb": cond_rotary_emb,
                        "text_cond_mask": text_cond_mask,
                    }
                    if use_condition and model_config["single_use_condition"]
                    else {}
                ),
                **ckpt_kwargs,
            )

        else:
            result = single_block_forward(
                block,
                model_config=model_config,
                hidden_states=hidden_states,
                temb=temb,
                delta_emb=delta_emb_single,
                delta_emb_cblock=delta_emb_cblock,
                delta_emb_mask=delta_emb_mask,
                use_text_mod=use_text_mod,
                use_img_mod=use_img_mod,
                image_rotary_emb=image_rotary_emb,
                last_attn_map=last_attn_map,
                latent_height=latent_height,
                timestep=timestep,
                store_attn_map=store_attn_map,
                latent_sblora_weight=latent_sblora_weight,
                condition_sblora_weight=condition_sblora_weight,
                **(
                    {
                        "condition_latents": condition_latents,
                        "cond_temb": cond_temb,
                        "cond_rotary_emb": cond_rotary_emb,
                        "text_cond_mask": text_cond_mask,
                    }
                    if use_condition and model_config["single_use_condition"]
                    else {}
                ),
            )
        if use_condition and model_config["single_use_condition"]:
            hidden_states, condition_latents = result
        else:
            hidden_states = result

        # controlnet residual
        if controlnet_single_block_samples is not None:
            interval_control = len(transformer.single_transformer_blocks) / len(
                controlnet_single_block_samples
            )
            interval_control = int(np.ceil(interval_control))
            hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                + controlnet_single_block_samples[index_block // interval_control]
            )

    hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

    hidden_states = transformer.norm_out(hidden_states, temb)
    output = transformer.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(transformer, lora_scale)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)

