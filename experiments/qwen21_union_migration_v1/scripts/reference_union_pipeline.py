# -*- coding: utf-8 -*-
"""ReferenceUnionPipeline：官方 Qwen-Image-2.1-Fun-Controlnet-Union 的参考图接入层。

官方 control pipeline 把参考图路径写死为 None（见 pipeline_qwenimage21_control.py 第 630 行），
所以它的示例无法证明源人物身份能被保持。本文件只做一件事：在同一条官方流程里把参考图
重新接回 Qwen3-VL 与 VAE，同时把 129 通道控制条件对齐到「参考图 token + 目标 token」的联合序列。

不改动官方仓库的任何文件（固定 commit 的 checkout 保持原样），也不改动本项目现有 workflow 与 Adapter。
"""
from __future__ import annotations

import pathlib
from typing import Callable, List, Optional, Union

import torch

import videox_fun.pipeline.pipeline_qwenimage21_control as official
from videox_fun.pipeline import QwenImage21ControlPipeline

# 全部复用官方模块已导入的符号，保证与原版走同一套实现
calculate_dimensions = official.calculate_dimensions
calculate_shift = official.calculate_shift
retrieve_timesteps = official.retrieve_timesteps
randn_tensor = official.randn_tensor
QwenImagePipelineOutput = official.QwenImagePipelineOutput
F = official.F
logger = official.logger
np = official.np


class ReferenceUnionPipeline(QwenImage21ControlPipeline):
    """官方控制 pipeline + `reference_images`。不传参考图时与原版逐行等价（控制条件构造完全相同）。"""

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Optional[torch.Tensor] = None,                      # 仍是 Inpaint 底图，语义不变
        reference_images: Optional[List] = None,                    # 新增：源人物图 / 骨架参考图
        control_image: Optional[torch.Tensor] = None,
        mask_image: Optional[torch.Tensor] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 40,
        sigmas: Optional[List[float]] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_resolution: int = 1024,
        control_context_scale: float = 1.0,
        use_kv_cache: bool = False,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[dict] = None,
        callback_on_step_end: Optional[Callable] = None,
        callback_on_step_end_tensor_inputs: Optional[List[str]] = None,
    ):
        device = self._execution_device

        # 1. 参考图预处理：与官方 ti2i 路径一致，一次 resize 同时喂文本编码器和 VAE
        input_image_sizes, input_images, vae_images = [], None, None
        if reference_images is not None:
            input_images, vae_images = [], []
            for img in reference_images:
                if hasattr(img, "mode") and img.mode != "RGBA":
                    img = img.convert("RGBA")
                image_width, image_height = img.size
                input_width, input_height, _ = calculate_dimensions(
                    output_resolution * output_resolution, image_width / image_height
                )
                input_image_sizes.append((input_width, input_height))
                input_images.append(self.image_processor.resize(img, width=input_width, height=input_height))
                vae_images.append(
                    self.image_processor.preprocess(img, width=input_width, height=input_height).unsqueeze(2)
                )

        if image is not None:
            if height is None or width is None:
                image_width, image_height = image.size if hasattr(image, "size") else (image.shape[-1], image.shape[-2])
                calculated_width, calculated_height, _ = calculate_dimensions(
                    output_resolution * output_resolution, image_width / image_height
                )
                height = height or calculated_height
                width = width or calculated_width
        height = height or output_resolution
        width = width or output_resolution

        callback_on_step_end_tensor_inputs = callback_on_step_end_tensor_inputs or []
        self.check_inputs(prompt, height, width, prompt_embeds, callback_on_step_end_tensor_inputs)

        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        self._attention_kwargs = attention_kwargs or {}
        self._current_timestep = None
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 2. 编码提示词（参考图进 Qwen3-VL，得到图文联合 embedding 与 image_pad_mask）
        has_neg_prompt = negative_prompt is not None or negative_prompt_embeds is not None
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        prompt_embeds, prompt_embeds_mask, image_pad_mask = self.encode_prompt(
            image=input_images,
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask, negative_image_pad_mask = self.encode_prompt(
                image=input_images,
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )

        # 3. 目标噪声 latent 与参考图 latent
        num_channels_latents = self.transformer.config.in_channels
        latents, input_images_latents = self.prepare_latents(
            vae_images,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 3b. 129 通道控制条件：64 控制 latent + 1 keep mask + 64 被遮罩图 latent（官方格式，不重新解释）
        weight_dtype = prompt_embeds.dtype
        ctrl_batch = batch_size * num_images_per_prompt
        lat_h = 2 * (int(height) // (self.vae_scale_factor * 2))
        lat_w = 2 * (int(width) // (self.vae_scale_factor * 2))

        def _to_ctrl_batch(t):
            return t.repeat(ctrl_batch, *[1] * (t.dim() - 1)) if (t.shape[0] == 1 and ctrl_batch > 1) else t

        def _rgba(t):
            return torch.cat([t, torch.ones_like(t[:, :1])], dim=1) if t.shape[1] == 3 else t

        if mask_image is not None:
            mask_condition = self.mask_processor.preprocess(mask_image, height=height, width=width)
            mask_condition = (mask_condition >= 0.5).to(weight_dtype)[:, :1]
        else:
            mask_condition = torch.ones(ctrl_batch, 1, height, width, dtype=weight_dtype)
        mask_condition = _to_ctrl_batch(mask_condition).to(device)

        if image is not None:
            inpaint_image = self.image_processor.preprocess(image, height=height, width=width).to(device)
            inpaint_image = inpaint_image * (mask_condition < 0.5)
            inpaint_latent = self._encode_vae_image(
                _rgba(inpaint_image).unsqueeze(2).to(device=device, dtype=weight_dtype), generator
            )
        else:
            inpaint_latent = torch.zeros(
                ctrl_batch, self.latent_channels, 1, lat_h, lat_w, device=device, dtype=weight_dtype
            )
        inpaint_latent = _to_ctrl_batch(inpaint_latent)

        if control_image is not None:
            ctrl_image = self.image_processor.preprocess(control_image, height=height, width=width)
            control_latents = self._encode_vae_image(
                _rgba(ctrl_image).unsqueeze(2).to(device=device, dtype=weight_dtype), generator
            )
        else:
            control_latents = torch.zeros_like(inpaint_latent)
        control_latents = _to_ctrl_batch(control_latents)

        mask_latent = F.interpolate(1 - mask_condition, size=(lat_h, lat_w), mode="nearest").unsqueeze(2)

        control_context = torch.cat([control_latents, mask_latent, inpaint_latent], dim=1)
        control_context = self._pack_latents(control_context, ctrl_batch, control_context.shape[1], lat_h, lat_w)

        # 关键适配：control_context 原本只覆盖目标 token；参考图进入联合序列后，image_pad_mask 覆盖
        # 「参考 + 目标」两段，投影前的输入长度必须等长。参考位置补零只表示"没有控制信号"，
        # 不等于 control_img_in 的输出为零——该层带已训练 bias，必须保留。
        n_ref = int(input_images_latents.shape[1]) if input_images_latents is not None else 0
        if n_ref:
            head = control_context.new_zeros(ctrl_batch, n_ref, control_context.shape[2])
            control_context = torch.cat([head, control_context], dim=1)
        control_context = control_context.to(device=device, dtype=weight_dtype)

        img_shapes = [
            [
                *[
                    (1, vae_height // self.vae_scale_factor, vae_width // self.vae_scale_factor)
                    for vae_width, vae_height in input_image_sizes
                ],
                (1, height // self.vae_scale_factor, width // self.vae_scale_factor),
            ]
        ] * batch_size

        # 4. 采样时间步
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        mu = calculate_shift(
            latents.shape[1],
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        def append_target_slots(mask):
            return torch.cat([mask, mask.new_ones(mask.shape[0], latents.shape[1] // 4)], dim=1)

        image_pad_mask = append_target_slots(image_pad_mask)
        if do_true_cfg:
            negative_image_pad_mask = append_target_slots(negative_image_pad_mask)

        # 首轮按方案关闭 KV cache：控制分支每步都要重算条件，不能借主干前缀缓存
        num_blocks = len(self.transformer.transformer_blocks)
        cache_enabled = (
            use_kv_cache and self.transformer.config.causal_condition and control_context is None
        )
        cond_cache = official.QwenImage21KVCache(num_blocks) if cache_enabled else None
        neg_cache = official.QwenImage21KVCache(num_blocks) if cache_enabled and do_true_cfg else None

        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    break

                self._current_timestep = t
                kv_mode = "extract" if (cache_enabled and i == 0) else ("cached" if cache_enabled else None)

                latent_model_input = latents
                if input_images_latents is not None:
                    latent_model_input = torch.cat([input_images_latents, latents], dim=1)

                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    img_shapes=img_shapes,
                    img_mask=image_pad_mask,
                    attention_kwargs=self.attention_kwargs,
                    kv_cache=cond_cache,
                    kv_cache_mode=kv_mode,
                    control_context=control_context,
                    control_context_scale=control_context_scale,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred[:, -latents.size(1):]

                if do_true_cfg:
                    neg_noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_hidden_states_mask=negative_prompt_embeds_mask,
                        img_shapes=img_shapes,
                        img_mask=negative_image_pad_mask,
                        attention_kwargs=self.attention_kwargs,
                        kv_cache=neg_cache,
                        kv_cache_mode=kv_mode,
                        control_context=control_context,
                        control_context_scale=control_context_scale,
                        return_dict=False,
                    )[0]
                    neg_noise_pred = neg_noise_pred[:, -latents.size(1):]
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self._current_timestep = None
        if output_type == "latent":
            image_out = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents = latents * latents_std + latents_mean
            image_out = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            image_out = self.image_processor.postprocess(image_out, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image_out,)
        return QwenImagePipelineOutput(images=image_out)
