"""Union flow-matching 原语：仅目标加噪、确定性轨迹与冻结审计。"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch

from .weights import EXPECTED_LORA_PARAMETERS, audit_trainable_parameters

DATASET_ORDER = ("iper", "tiktok", "hi4d")


def configure_determinism():
    """在首次 CUDA 初始化前配置严格可复现计算；已初始化的错误配置拒绝续跑。"""
    required = ":4096:8"
    current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if torch.cuda.is_initialized() and current != required:
        raise RuntimeError("Restart this process with CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA "
                           f"context creation; current initialized CUDA has {current!r}")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = required
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {"cublas_workspace_config": required,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark}


@dataclass(frozen=True)
class TrainingSchedule:
    timesteps: torch.Tensor
    sigmas: torch.Tensor
    shift: float
    target_tokens: int


def make_training_schedule(scheduler, target_tokens: int, steps: int = 1000) -> TrainingSchedule:
    if not isinstance(target_tokens, int) or target_tokens <= 0:
        raise ValueError("target_tokens must be a positive target resolution")
    if not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be positive")
    config = scheduler.config
    if not config.get("use_dynamic_shifting", False):
        raise ValueError("Training scheduler must enable dynamic shifting")
    base_len, max_len = config.get("base_image_seq_len", 256), config.get("max_image_seq_len", 4096)
    base_shift, max_shift = config.get("base_shift", .5), config.get("max_shift", 1.15)
    if max_len <= base_len:
        raise ValueError("Invalid scheduler sequence length bounds")
    slope = (max_shift - base_shift) / (max_len - base_len)
    shift = base_shift + (target_tokens - base_len) * slope
    scheduler.set_timesteps(sigmas=np.linspace(1., 1. / steps, steps), device="cpu", mu=shift)
    timesteps = scheduler.timesteps.detach().cpu().clone()
    sigmas = scheduler.sigmas[:steps].detach().cpu().clone()
    if len(timesteps) != steps or len(sigmas) != steps:
        raise ValueError("Scheduler did not produce the requested paired positions")
    return TrainingSchedule(timesteps=timesteps, sigmas=sigmas, shift=shift, target_tokens=target_tokens)


def transformer_inputs(condition: Mapping, noisy_target: torch.Tensor, timestep: torch.Tensor,
                       *, device=None, dtype=None) -> dict:
    """只接收条件缓存；监督 target latent/路径没有入口。"""
    device = device if device is not None else noisy_target.device
    dtype = dtype if dtype is not None else noisy_target.dtype
    def tensor(key, *, boolean=False):
        value = condition[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Condition {key} must be a tensor")
        # clone removes inference-tensor status before a trainable Linear saves its input.
        return value.detach().clone().to(device=device, dtype=torch.bool if boolean else dtype)
    references, control = tensor("reference_latents"), tensor("control_latents")
    noisy_target = noisy_target.to(device=device, dtype=dtype)
    if noisy_target.ndim != 3 or noisy_target.shape[0] != 1 or noisy_target.shape[-1] != 64:
        raise ValueError("Plan B requires one [1,Ntarget,64] target per microstep")
    batch, target_tokens, _ = noisy_target.shape
    if references.ndim != 3 or references.shape[0] != batch or references.shape[-1] != 64:
        raise ValueError("Reference latent shape must be [1,Nreference,64]")
    if control.shape != noisy_target.shape:
        raise ValueError("Control latent token count differs from target")
    shapes = condition["img_shapes"]
    if len(shapes) != batch or np.prod(shapes[0][-1]) != target_tokens:
        raise ValueError("img_shapes target token count differs")
    if sum(np.prod(shape) for shape in shapes[0][:-1]) != references.shape[1]:
        raise ValueError("img_shapes reference token count differs")
    target_context = torch.cat([control, control.new_zeros(batch, target_tokens, 65)], dim=-1)
    context = torch.cat([control.new_zeros(batch, references.shape[1], 129), target_context], dim=1)
    prompt, prompt_mask, image_mask = tensor("prompt_embeds"), tensor("prompt_mask", boolean=True), tensor("img_mask", boolean=True)
    if prompt.ndim != 3 or prompt.shape[0] != batch or prompt_mask.shape != prompt.shape[:2]:
        raise ValueError("Prompt embedding/mask shape differs")
    if image_mask.ndim != 2 or image_mask.shape[0] != batch:
        raise ValueError("img_mask must be the per-slot [1,L] mask")
    if int(image_mask.sum()) * 4 != references.shape[1] + target_tokens:
        raise ValueError("Image slot count differs from reference plus target tokens")
    if image_mask.shape[1] != prompt.shape[1] + target_tokens // 4:
        raise ValueError("Image slots must include final target slots")
    return {"hidden_states": torch.cat([references, noisy_target], dim=1),
            "encoder_hidden_states": prompt, "encoder_hidden_states_mask": prompt_mask,
            "img_shapes": shapes, "img_mask": image_mask,
            "control_context": context, "control_context_scale": 1.,
            "timestep": timestep.detach().clone().to(device=device, dtype=dtype).reshape(1)}


def flow_loss(transformer, condition, supervision, index, noise, scheduler, device):
    """返回内容区 flow MSE；保留冻结主干的计算图，使 hints 接收梯度。"""
    target = supervision["target_latent"]
    if not isinstance(target, torch.Tensor) or target.ndim != 3 or target.shape[0] != 1 or target.shape[-1] != 64:
        raise ValueError("Supervision target_latent must be [1,Ntarget,64]")
    floating = next((p for p in transformer.parameters() if not p.requires_grad and p.is_floating_point()), None)
    dtype = floating.dtype if floating is not None else target.dtype
    target = target.detach().clone().to(device=device, dtype=dtype)
    if noise.shape != target.shape:
        raise ValueError("Noise shape differs from target")
    noise = noise.to(device=device, dtype=dtype)
    valid = supervision["valid_target_tokens"].detach().clone().to(device=device, dtype=torch.float32)
    if valid.shape != target.shape[:2] or not torch.isfinite(valid).all() or (valid < 0).any() or (valid > 1).any():
        raise ValueError("Valid token coverage must be finite [1,Ntarget] values in [0,1]")
    if valid.sum() <= 0:
        raise ValueError("Target must contain positive valid content")
    schedule = scheduler if isinstance(scheduler, TrainingSchedule) else make_training_schedule(scheduler, target.shape[1])
    if schedule.target_tokens != target.shape[1]:
        raise ValueError("Training schedule target resolution differs")
    if isinstance(index, torch.Tensor):
        if index.numel() != 1:
            raise ValueError("One timestep index is required per microstep")
        index = index.item()
    if not isinstance(index, int) or not 0 <= index < len(schedule.timesteps):
        raise ValueError("Schedule index is outside available paired positions")
    sigma = schedule.sigmas[index].to(device=device, dtype=target.dtype).reshape(1, 1, 1)
    timestep = schedule.timesteps[index:index + 1].to(device=device) / 1000
    noisy_target = (1 - sigma) * target + sigma * noise
    inputs = transformer_inputs(condition, noisy_target, timestep, device=device, dtype=dtype)
    prediction = transformer(**inputs, return_dict=False)[0][:, -target.shape[1]:]
    if prediction.shape != target.shape:
        raise ValueError("Transformer target prediction shape differs")
    square = (prediction.float() - (noise - target).float()).square()
    weights = valid.unsqueeze(-1)
    return (square * weights).sum() / (weights.sum() * square.shape[-1]).clamp_min(1)


def deterministic_seed(train_seed: int, update: int, microstep: int, sample_id: str, stream: str) -> int:
    payload = json.dumps([int(train_seed), int(update), int(microstep), str(sample_id), str(stream)],
                         ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def balanced_sample_order(samples: Mapping, sampler_seed=20261015) -> tuple[tuple[str, str], ...]:
    if set(samples) != set(DATASET_ORDER):
        raise ValueError("Balanced sampling requires iper, tiktok, hi4d")
    sizes = {len(samples[name]) for name in DATASET_ORDER}
    if len(sizes) != 1 or not next(iter(sizes)):
        raise ValueError("All three datasets require equal nonzero quotas")
    ordered = {}
    for dataset in DATASET_ORDER:
        ids = tuple(str(value) for value in samples[dataset])
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate sample ids: {dataset}")
        ordered[dataset] = sorted(ids, key=lambda sample: hashlib.sha256(
            f"union_b_order_v1:{sampler_seed}:{dataset}:{sample}".encode()).hexdigest())
    return tuple((dataset, ordered[dataset][index]) for index in range(next(iter(sizes))) for dataset in DATASET_ORDER)


def training_draw(*, train_seed, update, microstep, sample_id, target_shape, schedule_steps=1000,
                  device="cpu", dtype=torch.float32):
    """独立 noise/timestep Generator，不消耗全局或验证 RNG。"""
    noise_seed = deterministic_seed(train_seed, update, microstep, sample_id, "noise")
    timestep_seed = deterministic_seed(train_seed, update, microstep, sample_id, "timestep")
    generator = torch.Generator(device=device).manual_seed(noise_seed)
    time_generator = torch.Generator(device="cpu").manual_seed(timestep_seed)
    noise = torch.randn(target_shape, generator=generator, device=device, dtype=dtype)
    index = int(torch.randint(schedule_steps, (1,), generator=time_generator).item())
    return {"noise": noise, "index": index, "noise_seed": noise_seed, "timestep_seed": timestep_seed,
            "sample_id": sample_id, "update": update, "microstep": microstep}


def make_optimizer(model, *, lr=1e-4, expected_parameters=EXPECTED_LORA_PARAMETERS):
    audit_trainable_parameters(model, expected_parameters=expected_parameters)
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                             betas=(.9, .999), eps=1e-8, weight_decay=.01)


def make_lr_scheduler(optimizer, *, warmup_updates=32):
    if not isinstance(warmup_updates, int) or warmup_updates < 0:
        raise ValueError("warmup_updates must be a nonnegative integer")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda index: min((index + 1) / warmup_updates, 1.) if warmup_updates else 1.)
    scheduler.union_warmup_updates = warmup_updates
    return scheduler


class TrainingResourceError(RuntimeError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def run_training_updates(transformer, *, optimizer, lr_scheduler, scheduler, sample_order,
                         load_sample, updates: int, device="cuda", train_seed=3407,
                         accumulation_steps=3, cursor=None, on_update=None, on_microstep=None,
                         stop_requested=None, stop_after_updates=None, repeat_samples=False,
                         max_memory_fraction=.9, expected_parameters=EXPECTED_LORA_PARAMETERS):
    """执行有界训练；回调在更新边界保存 checkpoint，支持 probe 和安全停止。

    updates 是整个 run 的最终更新数，stop_after_updates 限制本次调用。
    load_sample(sample_id) 返回独立的 (condition, supervision) 缓存字典。
    on_update(event) 接收已完成边界的 TrainingCursor，可据 step 0/128/256/512
    调用 save_checkpoint；本函数不会替调用者决定输出路径或覆盖产物。
    """
    from .checkpoint import TrainingCursor

    if not isinstance(updates, int) or updates < 0 or accumulation_steps < 1:
        raise ValueError("Training update/accumulation budget is invalid")
    if stop_after_updates is not None and (not isinstance(stop_after_updates, int) or stop_after_updates < 0):
        raise ValueError("Probe stop_after_updates must be a nonnegative integer")
    if not 0 < max_memory_fraction <= 1:
        raise ValueError("Memory fraction must be in (0,1]")
    cursor = cursor or TrainingCursor(accumulation_steps=accumulation_steps)
    cursor.validate()
    if cursor.accumulation_steps != accumulation_steps or cursor.optimizer_step > updates:
        raise ValueError("Cursor differs from the requested update budget")
    sample_order = tuple(sample_order)
    if not sample_order or (not repeat_samples and updates * accumulation_steps > len(sample_order)):
        raise ValueError("Training budget exceeds the locked sample order")
    if any(len(entry) != 2 for entry in sample_order):
        raise ValueError("sample_order must contain (dataset, sample_id) pairs")
    if accumulation_steps == 3:
        if len(sample_order) % 3 or any(tuple(item[0] for item in sample_order[i:i + 3]) != DATASET_ORDER
                                       for i in range(0, len(sample_order), 3)):
            raise ValueError("Each optimizer update must follow iper,tiktok,hi4d order")
    audit_trainable_parameters(transformer, expected_parameters=expected_parameters)
    parameters = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
    transformer.train()
    device = torch.device(device)
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        total_memory = torch.cuda.get_device_properties(device).total_memory
    else:
        total_memory = 0
    final_update = updates if stop_after_updates is None else min(updates, cursor.optimizer_step + stop_after_updates)
    schedule_cache, trajectory, update_logs = {}, [], []
    optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()

    def resource_report():
        return {"max_memory_allocated": int(torch.cuda.max_memory_allocated(device)) if use_cuda else 0,
                "max_memory_reserved": int(torch.cuda.max_memory_reserved(device)) if use_cuda else 0,
                "total_memory": total_memory, "elapsed_seconds": time.perf_counter() - start}

    try:
        for update in range(cursor.optimizer_step, final_update):
            losses, step_trajectory = [], []
            for microstep in range(accumulation_steps):
                position = update * accumulation_steps + microstep
                dataset, sample_id = sample_order[position % len(sample_order)]
                condition, supervision = load_sample(sample_id)
                target = supervision["target_latent"]
                target_tokens = target.shape[1]
                if target_tokens not in schedule_cache:
                    schedule_cache[target_tokens] = make_training_schedule(scheduler, target_tokens)
                schedule = schedule_cache[target_tokens]
                draw = training_draw(train_seed=train_seed, update=update, microstep=microstep,
                                     sample_id=sample_id, target_shape=target.shape, device=device,
                                     dtype=target.dtype, schedule_steps=len(schedule.timesteps))
                autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_cuda else nullcontext()
                with autocast:
                    loss = flow_loss(transformer, condition, supervision, draw["index"], draw["noise"], schedule, device)
                if not torch.isfinite(loss):
                    raise ValueError(f"Nonfinite flow loss at {sample_id}, update {update}")
                (loss / accumulation_steps).backward()
                if use_cuda:
                    torch.cuda.synchronize(device)
                entry = {key: value for key, value in draw.items() if key != "noise"}
                entry.update(dataset=dataset, position=position, sigma=float(schedule.sigmas[draw["index"]]),
                             timestep=float(schedule.timesteps[draw["index"]]), loss=float(loss.detach()))
                trajectory.append(entry)
                step_trajectory.append(entry)
                losses.append(entry["loss"])
                resources = resource_report()
                if on_microstep is not None:
                    on_microstep({**entry, **resources})
                if use_cuda and resources["max_memory_reserved"] > max_memory_fraction * total_memory:
                    raise TrainingResourceError("Training memory exceeded the fixed 90% gate", {
                        "cursor": cursor, "trajectory": trajectory, "updates": update_logs, **resources})
            audit_trainable_parameters(transformer, expected_parameters=expected_parameters)
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            if gradient_norm <= 0:
                raise ValueError("No nonzero LoRA gradient reached the optimizer")
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            cursor = TrainingCursor(optimizer_step=update + 1, microstep=(update + 1) * accumulation_steps,
                next_sample_position=(update + 1) * accumulation_steps,
                loss_log_cursor=cursor.loss_log_cursor + accumulation_steps,
                gradient_log_cursor=cursor.gradient_log_cursor + 1, accumulation_steps=accumulation_steps)
            event = {"optimizer_step": update + 1, "loss": sum(losses) / accumulation_steps,
                     "gradient_norm": float(gradient_norm), "lr": optimizer.param_groups[0]["lr"],
                     "cursor": cursor, "trajectory": step_trajectory, **resource_report()}
            update_logs.append(event)
            if on_update is not None:
                on_update(event)
            if stop_requested is not None and stop_requested():
                break
    except torch.cuda.OutOfMemoryError as error:
        optimizer.zero_grad(set_to_none=True)
        raise TrainingResourceError("CUDA OOM; stopped without changing training variables", {
            "cursor": cursor, "trajectory": trajectory, "updates": update_logs, **resource_report()}) from error
    except Exception:
        optimizer.zero_grad(set_to_none=True)
        raise
    return {"cursor": cursor, "trajectory": trajectory, "updates": update_logs,
            "stopped_early": cursor.optimizer_step < updates, **resource_report()}
