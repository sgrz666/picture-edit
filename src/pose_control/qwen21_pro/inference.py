"""变分辨率的原生 unpatched64 Euler 采样；条件与监督完全分离。"""

from __future__ import annotations

from dataclasses import dataclass
import copy
from collections.abc import Mapping

import torch

from .training import make_training_schedule, transformer_inputs
from .cache import _payload_digest


@dataclass(frozen=True)
class SamplingResult:
    latents: torch.Tensor
    images: object
    audit: dict


@torch.no_grad()
def decode_latents(pipe, latents, canvas_hw, *, output_type="pil"):
    h, w = canvas_hw
    scale = getattr(pipe, "vae_scale_factor", 16)
    if h % scale or w % scale or latents.shape != (1, (h // scale) * (w // scale), 64):
        raise ValueError("Decode canvas and unpatched64 latent grid differ")
    value = (
        latents.transpose(1, 2)
        .reshape(1, 64, 1, h // scale, w // scale)
        .to(dtype=pipe.vae.dtype)
    )
    config = pipe.vae.config
    if config.z_dim != 64:
        raise ValueError("Qwen21 requires unpatched64 VAE")
    mean = torch.as_tensor(
        config.latents_mean, device=value.device, dtype=value.dtype
    ).reshape(1, 64, 1, 1, 1)
    std = torch.as_tensor(
        config.latents_std, device=value.device, dtype=value.dtype
    ).reshape(1, 64, 1, 1, 1)
    if (
        not torch.isfinite(mean).all()
        or not torch.isfinite(std).all()
        or (std <= 0).any()
    ):
        raise ValueError("Invalid VAE normalization statistics")
    decoded = pipe.vae.decode(value * std + mean, return_dict=False)[0][:, :, 0]
    return pipe.image_processor.postprocess(decoded, output_type=output_type)


@torch.no_grad()
def sample_cached(
    pipe,
    condition,
    *,
    task_condition,
    num_inference_steps=40,
    seed=3407,
    true_cfg_scale=1.0,
    output_type="latent",
    use_kv_cache=False,
    device=None,
    latents=None,
    control_context_scale=1.0,
):
    """消费 legacy BoundUnionCondition 或 EncodedCondition；不走缺失 image_pad_mask 的 VLM 接口。

    正负 prompt 各自使用实际 mask，同一个 task pack/source bank 和原生 129 通道。
    scheduler sigma/shift 与训练完全同源，CFG 使用 Qwen21 官方 L2 norm rescale。
    """
    if (
        use_kv_cache
        or not 1 <= true_cfg_scale
        or type(num_inference_steps) is not int
        or num_inference_steps < 1
    ):
        raise ValueError("Invalid sampling options; Pro requires disabled KV cache")
    if hasattr(condition, "verify"):
        condition.verify(task_condition)
    task_condition.validate(pipe.transformer.task_adapter.config)
    hw = tuple(condition["target_hw"])
    canvas = tuple(condition.get("canvas_hw", tuple(v * 16 for v in hw)))
    if (
        hw != task_condition.target_hw
        or tuple(v // 16 for v in canvas) != hw
        or any(v % 32 for v in canvas)
    ):
        raise ValueError(
            "Encoded canvas and task target grid differ or cannot form four-token image slots"
        )
    if (
        "person_order" in condition
        and tuple(condition["person_order"]) != task_condition.person_ids
    ):
        raise ValueError("Encoded source person order differs")
    if true_cfg_scale > 1 and "negative" not in condition:
        raise ValueError("CFG requires independently encoded negative prompt masks")
    device = torch.device(device or pipe._execution_device)
    frozen = next(
        (
            p
            for p in pipe.transformer.parameters()
            if p.is_floating_point() and not p.requires_grad
        ),
        None,
    )
    dtype = frozen.dtype if frozen is not None else condition["control_latents"].dtype
    shape = (1, hw[0] * hw[1], 64)
    if latents is None:
        latents = (
            torch.randn(
                (1, 64, 1, *hw),
                device=device,
                dtype=dtype,
                generator=torch.Generator(device=device).manual_seed(seed),
            )
            .flatten(2)
            .transpose(1, 2)
        )
    elif tuple(latents.shape) != shape or not torch.isfinite(latents).all():
        raise ValueError("Initial noise does not match current target grid")
    else:
        latents = latents.detach().clone().to(device=device, dtype=dtype)
    schedule = make_training_schedule(
        pipe.scheduler, shape[1], steps=num_inference_steps
    )
    pipe.scheduler.timesteps = schedule.timesteps.to(device)
    pipe.scheduler.set_begin_index(0)
    for t in schedule.timesteps.to(device):

        def prediction(branch):
            values = transformer_inputs(
                branch, latents, t.reshape(1) / 1000, device=device, dtype=dtype
            )
            values.update(
                task_condition=task_condition,
                kv_cache=None,
                kv_cache_mode=None,
                control_context_scale=control_context_scale,
                return_dict=False,
            )
            result = pipe.transformer(**values)[0][:, -shape[1] :]
            if result.shape != shape or not torch.isfinite(result).all():
                raise ValueError("Transformer prediction must match finite target grid")
            return result

        positive = prediction(condition)
        if true_cfg_scale > 1:
            negative = prediction({**dict(condition), **condition["negative"]})
            guided = negative + true_cfg_scale * (positive - negative)
            positive = guided * (
                positive.norm(dim=-1, keepdim=True)
                / guided.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            )
        latents = pipe.scheduler.step(positive, t, latents, return_dict=False)[0]
    images = (
        None
        if output_type == "latent"
        else decode_latents(pipe, latents, canvas, output_type=output_type)
    )
    if hasattr(pipe, "maybe_free_model_hooks"):
        pipe.maybe_free_model_hooks()
    native_binding = getattr(
        condition, "binding", getattr(condition, "attestation", None)
    )
    if isinstance(native_binding, Mapping):
        native_binding = dict(native_binding)
    return SamplingResult(
        latents,
        images,
        {
            "target_hw": list(hw),
            "canvas_hw": list(canvas),
            "seed": seed,
            "steps": num_inference_steps,
            "shift": schedule.shift,
            "true_cfg_scale": true_cfg_scale,
            "cfg_norm_rescale": True,
            "use_kv_cache": False,
            "sampling": "native_dynamic_shift_euler_unpatched64_v1",
            "noise_layout": "channel_major_1x64x1xHxW_then_unpatched64",
            "source_fingerprint": task_condition.source_fingerprint,
            "geometry_signature": copy.deepcopy(
                task_condition.provenance.get("geometry", {})
            ),
            "task_condition_sha256": _payload_digest(task_condition.to_dict()),
            "native_condition_binding": copy.deepcopy(native_binding),
            "trained_model_signature": copy.deepcopy(
                getattr(pipe, "trained_model_signature", None)
            ),
        },
    )
