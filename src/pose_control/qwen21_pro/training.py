"""Pro 的独立优化器、冻结审计与更新循环；复用 Union 噪声及调度语义。"""
from __future__ import annotations

import hashlib
import time
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from ..qwen21_union.training import (DATASET_ORDER, TrainingResourceError, TrainingSchedule,
    balanced_sample_order, configure_determinism, deterministic_seed, make_lr_scheduler,
    make_training_schedule, training_draw, transformer_inputs as _union_inputs)
from ..qwen21_union.weights import ControlLoRALinear, control_lora_targets as _official_lora_targets
from .adapter import ProTaskAdapter
from .cache import BoundUnionCondition, _payload_digest, stable_hash
from .losses import RegionalLossConfig, regional_flow_loss
from .offload import audit_memory_policy


@dataclass(frozen=True)
class SampleAttestation:
    """由显式 sample_validator 返回的 typed 证明，供 refiner 等独立缓存使用。

    callback 负责磁盘与载入后 payload/source 完整性验证；runner 检查样本、
    source bank、人物顺序、target grid，不能用任意 dict 自称 verified。
    """
    sample_id: str
    condition_fingerprint: str
    payload_sha256: str
    source_fingerprint: str
    target_hw: tuple[int, int]
    person_ids: tuple[int, ...]

    def validate(self, pack, sample_id):
        if (self.sample_id != sample_id or self.source_fingerprint != pack.source_fingerprint
                or self.target_hw != pack.target_hw or self.person_ids != pack.person_ids):
            raise ValueError("Custom sample attestation binding differs from typed condition")
        for value in (self.condition_fingerprint, self.payload_sha256):
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Custom sample attestation requires SHA256 fingerprints")
        return self


def audit_trainable_parameters(model, *, control_lora_targets=()):
    """任务分支全量 FP32；额外训练只允许显式列出的原生 control LoRA A/B。"""
    adapter = getattr(model, "task_adapter", None)
    if not isinstance(adapter, ProTaskAdapter) or not adapter.enabled:
        raise ValueError("Training requires an enabled Pro task_adapter at model root")
    if not torch.isfinite(adapter.gate).all() or adapter.gate == 0:
        raise ValueError("Training requires an enabled nonzero task gate")
    targets = tuple(control_lora_targets)
    if len(set(targets)) != len(targets) or not set(targets) <= set(_official_lora_targets()):
        raise ValueError("Explicit control LoRA targets differ from official control paths")
    modules = {n: m for n, m in model.named_modules() if isinstance(m, ControlLoRALinear)}
    if set(modules) != set(targets):
        raise ValueError("Installed LoRA inventory differs from explicit enabled control LoRA targets")
    for name, module in modules.items():
        if not module.enabled or module.rank <= 0 or module.alpha <= 0:
            raise ValueError(f"Control LoRA must be enabled with positive rank/alpha: {name}")
    allowed = {f"task_adapter.{n}" for n, _ in adapter.named_parameters()}
    allowed |= {f"{name}.{suffix}" for name in targets for suffix in ("lora_A", "lora_B")}
    actual = {n for n, p in model.named_parameters() if p.requires_grad}
    if actual != allowed:
        raise ValueError(f"Unexpected trainable parameters: extra={sorted(actual - allowed)}, missing={sorted(allowed - actual)}")
    aliases = [(n, p) for n, p in model.named_parameters(remove_duplicate=False) if p.requires_grad]
    if len({id(p) for _, p in aliases}) != len(aliases):
        raise ValueError("Trainable parameter aliases are not supported")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if parameter.dtype != torch.float32:
                raise ValueError(f"Trainable storage must be FP32: {name}")
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise ValueError(f"Nonfinite trainable gradient: {name}")
        elif parameter.grad is not None:
            raise ValueError(f"Frozen parameter has a gradient: {name}")
    return {"trainable_names": tuple(n for n, p in model.named_parameters() if p.requires_grad),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "config_signature": adapter.config.signature(),
            "control_lora": tuple({"name": name, "rank": modules[name].rank,
                "alpha": modules[name].alpha, "enabled": modules[name].enabled} for name in targets)}


def optimizer_name_bindings(model, optimizer, *, control_lora_targets=()):
    report = audit_trainable_parameters(model, control_lora_targets=control_lora_targets)
    by_id = {id(p): n for n, p in model.named_parameters() if p.requires_grad}
    bindings, seen = [], []
    for group in optimizer.param_groups:
        names = []
        for parameter in group["params"]:
            if id(parameter) not in by_id:
                raise ValueError("Optimizer contains a frozen or foreign parameter")
            names.append(by_id[id(parameter)])
            seen.append(id(parameter))
        bindings.append(names)
    if len(seen) != len(set(seen)) or set(seen) != set(by_id):
        raise ValueError("Optimizer bindings must cover each exact trainable parameter once")
    if not report["trainable_names"]:
        raise ValueError("Empty Pro optimizer")
    return bindings


def make_optimizer(model, *, lr=1e-4, control_lora_targets=()):
    audit_trainable_parameters(model, control_lora_targets=control_lora_targets)
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                             betas=(.9, .999), eps=1e-8, weight_decay=.01)


def frozen_parameter_digest(model):
    """哈希覆盖全部冻结参数的 name/dtype/shape/原始 bytes，不包含任务 gate buffer。"""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            header = str((name, str(parameter.dtype), tuple(parameter.shape))).encode()
            digest.update(len(header).to_bytes(8, "big"))
            digest.update(header)
            value = parameter.detach().contiguous().cpu().reshape(-1).view(torch.uint8)
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def transformer_inputs(condition, noisy_target, timestep, *, device=None, dtype=None):
    """复用原生 64+65 context；refiner 可显式提供 keep1 + inpaint64。"""
    inputs = _union_inputs(condition, noisy_target, timestep, device=device, dtype=dtype)
    keep, inpaint = condition.get("keep_mask"), condition.get("inpaint_latents")
    if (keep is None) != (inpaint is None):
        raise ValueError("keep_mask and inpaint_latents must be present together")
    if keep is not None:
        shape = noisy_target.shape
        if (not isinstance(keep, torch.Tensor) or keep.shape != (*shape[:2], 1)
                or not keep.is_floating_point() or not torch.isfinite(keep).all() or (keep < 0).any() or (keep > 1).any()
                or not isinstance(inpaint, torch.Tensor) or inpaint.shape != shape
                or not inpaint.is_floating_point() or not torch.isfinite(inpaint).all()):
            raise ValueError("Invalid keep_mask/inpaint_latents target grid, values or dtype")
        context = inputs["control_context"]
        keep = keep.detach().clone().to(device=context.device, dtype=context.dtype)
        inpaint = inpaint.detach().clone().to(device=context.device, dtype=context.dtype)
        context[:, -shape[1]:, 64:] = torch.cat([keep, inpaint], dim=-1)
    return inputs


def _verify_sample(condition, pack, sample_id=None, *, required=True):
    if not isinstance(condition, BoundUnionCondition):
        if required:
            raise ValueError("Training requires a verified BoundUnionCondition from load_bound_condition")
        return
    if _payload_digest(condition.payload) != condition._payload_digest:
        raise ValueError("Verified legacy condition payload changed after loading")
    if sample_id is not None and condition.attestation["sample_id"] != sample_id:
        raise ValueError("Verified legacy sample id differs from locked sample order")
    if tuple(condition["target_hw"]) != pack.target_hw:
        raise ValueError("Verified legacy target grid differs from typed pack")
    if "person_order" in condition and tuple(condition["person_order"]) != pack.person_ids:
        raise ValueError("Verified legacy person order differs from typed pack")
    keys = ("sample_id", "condition_fingerprint", "payload_sha256", "prepared_source_sha256", "source_latent_sha256")
    expected = {key: condition.attestation[key] for key in keys}
    if pack.provenance.get("cache_binding") != expected:
        raise ValueError("Typed pack cache binding differs from verified legacy condition")
    # VAE64 packs expose this direct source binding; source1600 packs are bound
    # by RawSourceFeatures.validate_binding in preparation and the runner cache.
    source = pack.provenance.get("source", {})
    for key in ("source_latent_sha256", "prepared_source_sha256"):
        if key in source and source[key] != condition.attestation[key]:
            raise ValueError(f"Typed source bank differs from verified legacy condition: {key}")


def flow_loss(transformer, condition, supervision, index, noise, scheduler, device, *, task_condition,
              loss_config=None, return_metrics=False):
    audit_memory_policy(transformer)
    task_condition.validate(transformer.task_adapter.config)
    _verify_sample(condition, task_condition, required=False)
    target = supervision["target_latent"]
    if (not isinstance(target, torch.Tensor) or target.ndim != 3 or target.shape[0] != 1 or target.shape[-1] != 64
            or not torch.isfinite(target).all()):
        raise ValueError("Supervision target_latent must be finite [1,Ntarget,64]")
    if task_condition.target_hw[0] * task_condition.target_hw[1] != target.shape[1]:
        raise ValueError("Task condition and target grid differ")
    frozen = next((p for p in transformer.parameters() if not p.requires_grad and p.is_floating_point()), None)
    dtype = frozen.dtype if frozen is not None else target.dtype
    target = target.detach().clone().to(device=device, dtype=dtype)
    if not isinstance(noise, torch.Tensor) or noise.shape != target.shape or not torch.isfinite(noise).all():
        raise ValueError("Noise must be finite and match target shape")
    noise = noise.detach().to(device=device, dtype=dtype)
    schedule = scheduler if isinstance(scheduler, TrainingSchedule) else make_training_schedule(scheduler, target.shape[1])
    if schedule.target_tokens != target.shape[1]:
        raise ValueError("Training schedule target grid differs")
    if isinstance(index, torch.Tensor) and index.numel() == 1:
        index = index.item()
    if type(index) is not int or not 0 <= index < len(schedule.timesteps):
        raise ValueError("Schedule index is outside available paired positions")
    sigma = schedule.sigmas[index].to(device=device, dtype=dtype).reshape(1, 1, 1)
    timestep = schedule.timesteps[index:index + 1].to(device=device) / 1000
    inputs = transformer_inputs(condition, (1 - sigma) * target + sigma * noise, timestep, device=device, dtype=dtype)
    # Never wrap the frozen transformer in no_grad: task hints need its Jacobian.
    prediction = transformer(**inputs, task_condition=task_condition, return_dict=False)[0][:, -target.shape[1]:]
    if prediction.shape != target.shape:
        raise ValueError("Transformer target prediction shape differs")
    loss, metrics = regional_flow_loss(prediction, noise - target, supervision["valid_target_tokens"], task_condition,
        config=loss_config or RegionalLossConfig(), adapter_config=transformer.task_adapter.config)
    return (loss, metrics) if return_metrics else loss


def trajectory_signature(sample_order, *, train_seed=3407, accumulation_steps=3, loss_config=None,
                         precision="cuda_bfloat16_autocast_fp32_trainables", memory_mode="resident"):
    """给 checkpoint trajectory_signature 使用的完整静态训练协议。"""
    return stable_hash({"version": 1, "sample_order": list(sample_order), "train_seed": train_seed,
        "accumulation_steps": accumulation_steps, "loss": (loss_config or RegionalLossConfig()).to_dict(),
        "precision": precision, "memory_mode": memory_mode,
        "noise_timestep_stream": "qwen21_union_sha256_v1", "schedule_steps": 1000,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark})


def _gradient_norms(model):
    sums = {"head": 0., "upstream": 0., "control_lora": 0.}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is not None:
            group = "head" if name.startswith("task_adapter.output_heads.") else (
                "upstream" if name.startswith("task_adapter.") else "control_lora")
            sums[group] += float(parameter.grad.detach().double().square().sum())
    return {key: value ** .5 for key, value in sums.items()}


def run_training_updates(transformer, *, optimizer, lr_scheduler, scheduler, sample_order, load_sample,
                         updates, device="cuda", train_seed=3407, accumulation_steps=3, cursor=None,
                         on_update=None, on_microstep=None, stop_requested=None, stop_after_updates=None,
                         repeat_samples=False, max_memory_fraction=.9, max_memory_bytes=None,
                         control_lora_targets=(), loss_config=None, require_verified_conditions=True,
                         sample_validator=None):
    """load_sample(id) -> (verified legacy condition, supervision, TaskCondition)。

    updates 是 run 最终边界；stop_after_updates 限本次调用。默认要求 verified
    loader，CPU 原生小 fixture 可显式关闭这一项。回调只在已完成更新上保存断点。
    """
    from .checkpoint import TrainingCursor
    if type(updates) is not int or updates < 0 or type(accumulation_steps) is not int or accumulation_steps < 1:
        raise ValueError("Invalid training update/accumulation budget")
    if stop_after_updates is not None and (type(stop_after_updates) is not int or stop_after_updates < 0):
        raise ValueError("Invalid stop_after_updates")
    if not 0 < max_memory_fraction <= 1 or (max_memory_bytes is not None and max_memory_bytes <= 0):
        raise ValueError("Invalid memory budget")
    cursor = cursor or TrainingCursor(accumulation_steps=accumulation_steps)
    cursor.validate()
    if cursor.accumulation_steps != accumulation_steps or cursor.optimizer_step > updates:
        raise ValueError("Cursor differs from training update budget")
    sample_order = tuple(sample_order)
    if not sample_order or any(not isinstance(e, (tuple, list)) or len(e) != 2 for e in sample_order):
        raise ValueError("sample_order must contain (dataset,sample_id) pairs")
    if not repeat_samples and updates * accumulation_steps > len(sample_order):
        raise ValueError("Training budget exceeds locked sample order")
    if accumulation_steps == 3 and (len(sample_order) % 3 or any(
            tuple(e[0] for e in sample_order[i:i + 3]) != DATASET_ORDER for i in range(0, len(sample_order), 3))):
        raise ValueError("Each update must follow balanced iper,tiktok,hi4d order")
    optimizer_name_bindings(transformer, optimizer, control_lora_targets=control_lora_targets)
    identities = tuple((n, id(p)) for n, p in transformer.named_parameters())
    memory = audit_memory_policy(transformer)
    parameters = [p for p in transformer.parameters() if p.requires_grad]
    transformer.train()
    device, start = torch.device(device), time.perf_counter()
    use_cuda = device.type == "cuda"
    total_memory = torch.cuda.get_device_properties(device).total_memory if use_cuda else 0
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    def resources():
        return {"max_memory_allocated": int(torch.cuda.max_memory_allocated(device)) if use_cuda else 0,
                "max_memory_reserved": int(torch.cuda.max_memory_reserved(device)) if use_cuda else 0,
                "total_memory": total_memory, "elapsed_seconds": time.perf_counter() - start,
                "memory_mode": memory["memory_mode"]}
    final = updates if stop_after_updates is None else min(updates, cursor.optimizer_step + stop_after_updates)
    schedule_cache, trajectory, update_logs = {}, [], []
    optimizer.zero_grad(set_to_none=True)
    try:
        for update in range(cursor.optimizer_step, final):
            step_entries = []
            for microstep in range(accumulation_steps):
                position = update * accumulation_steps + microstep
                dataset, sample_id = sample_order[position % len(sample_order)]
                condition, supervision, pack = load_sample(sample_id)
                pack.validate(transformer.task_adapter.config)
                _verify_sample(condition, pack, sample_id,
                               required=require_verified_conditions and sample_validator is None)
                if sample_validator is not None:
                    attestation = sample_validator(condition, pack, sample_id)
                    if not isinstance(attestation, SampleAttestation):
                        raise ValueError("sample_validator must return a SampleAttestation")
                    attestation.validate(pack, sample_id)
                target = supervision["target_latent"]
                tokens = target.shape[1]
                if tokens not in schedule_cache:
                    schedule_cache[tokens] = make_training_schedule(scheduler, tokens)
                schedule = schedule_cache[tokens]
                draw = training_draw(train_seed=train_seed, update=update, microstep=microstep,
                    sample_id=sample_id, target_shape=target.shape, device=device, dtype=target.dtype,
                    schedule_steps=len(schedule.timesteps))
                autocast = torch.autocast("cuda", dtype=torch.bfloat16) if use_cuda else nullcontext()
                with autocast:
                    loss, metrics = flow_loss(transformer, condition, supervision, draw["index"], draw["noise"],
                        schedule, device, task_condition=pack, loss_config=loss_config, return_metrics=True)
                if not torch.isfinite(loss):
                    raise ValueError(f"Nonfinite flow loss at {sample_id}, update {update}")
                (loss / accumulation_steps).backward()
                entry = {k: v for k, v in draw.items() if k != "noise"}
                entry.update(dataset=dataset, position=position, sigma=float(schedule.sigmas[draw["index"]]),
                    timestep=float(schedule.timesteps[draw["index"]]), loss=float(loss.detach()), **metrics)
                trajectory.append(entry)
                step_entries.append(entry)
                if use_cuda:
                    torch.cuda.synchronize(device)
                report = resources()
                if on_microstep is not None:
                    on_microstep({**entry, "accumulated_gradient_norms": _gradient_norms(transformer), **report})
                limit = min(total_memory * max_memory_fraction, max_memory_bytes or float("inf"))
                if use_cuda and report["max_memory_reserved"] > limit:
                    raise TrainingResourceError("Training memory exceeded fixed budget", {
                        "cursor": cursor, "trajectory": trajectory, "updates": update_logs, **report})
            if tuple((n, id(p)) for n, p in transformer.named_parameters()) != identities:
                raise ValueError("Model parameter names or objects changed during training")
            audit_trainable_parameters(transformer, control_lora_targets=control_lora_targets)
            gradient_norms = _gradient_norms(transformer)
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            if norm <= 0:
                raise ValueError("No nonzero task/control gradient reached optimizer")
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            complete = (update + 1) * accumulation_steps
            cursor = TrainingCursor(optimizer_step=update + 1, microstep=complete, next_sample_position=complete,
                noise_draw_cursor=complete, timestep_draw_cursor=complete, loss_log_cursor=complete,
                gradient_log_cursor=update + 1, accumulation_steps=accumulation_steps)
            event = {"optimizer_step": update + 1, "loss": sum(e["loss"] for e in step_entries) / accumulation_steps,
                "gradient_norm": float(norm), "gradient_norms": gradient_norms,
                "lr": [g["lr"] for g in optimizer.param_groups], "cursor": cursor,
                "trajectory": step_entries, **resources()}
            update_logs.append(event)
            if on_update is not None:
                on_update(event)
            if stop_requested is not None and stop_requested():
                break
    except torch.cuda.OutOfMemoryError as error:
        optimizer.zero_grad(set_to_none=True)
        raise TrainingResourceError("CUDA OOM; training variables were not changed", {
            "cursor": cursor, "trajectory": trajectory, "updates": update_logs, **resources()}) from error
    except Exception:
        optimizer.zero_grad(set_to_none=True)
        raise
    return {"cursor": cursor, "trajectory": trajectory, "updates": update_logs,
            "stopped_early": cursor.optimizer_step < updates, "memory_policy": memory, **resources()}
