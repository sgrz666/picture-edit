"""Pro 独立完整断点；校验先于载入，完成 manifest 最后写入。"""
from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .cache import file_sha256, ordinary_cpu
from .contracts import ProAdapterConfig
from .offload import MEMORY_MODES, _reject_hooks, audit_memory_policy
from .training import audit_trainable_parameters, frozen_parameter_digest, optimizer_name_bindings

FORMAT_VERSION = "qwen21_pro_training_checkpoint_v1"
REQUIRED_METADATA = frozenset({"config", "scope", "task_signature", "source_signature", "cache_signature",
    "code_signature", "data_signature", "trajectory_signature", "memory_mode", "control_lora_targets"})


@dataclass(frozen=True)
class TrainingCursor:
    optimizer_step: int = 0
    microstep: int = 0
    next_sample_position: int = 0
    noise_draw_cursor: int = 0
    timestep_draw_cursor: int = 0
    loss_log_cursor: int = 0
    gradient_log_cursor: int = 0
    accumulation_steps: int = 3

    def validate(self):
        if any(type(value) is not int or value < 0 for value in asdict(self).values()) or self.accumulation_steps < 1:
            raise ValueError("Training cursor fields must be nonnegative integers with positive accumulation")
        complete = self.optimizer_step * self.accumulation_steps
        if any(value != complete for value in (self.microstep, self.next_sample_position, self.noise_draw_cursor,
                                                self.timestep_draw_cursor, self.loss_log_cursor)):
            raise ValueError("Checkpoint cursor must be at a complete optimizer update boundary")
        if self.gradient_log_cursor != self.optimizer_step:
            raise ValueError("Gradient log cursor differs from optimizer boundary")
        return self


def capture_rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [s.cpu().clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0}


def _validate_rng(state):
    try:
        devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if state["cuda_device_count"] != devices or len(state["torch_cuda"]) != devices:
            raise ValueError("CUDA RNG device count differs")
        random.Random().setstate(state["python"])
        np.random.RandomState().set_state(state["numpy"])
        torch.Generator(device="cpu").set_state(state["torch_cpu"])
        for index, value in enumerate(state["torch_cuda"]):
            torch.Generator(device=f"cuda:{index}").set_state(value)
    except (KeyError, TypeError, RuntimeError, OverflowError) as error:
        raise ValueError("Invalid checkpoint RNG state") from error


def restore_rng_state(state):
    _validate_rng(state)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["cuda_device_count"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _canonical_metadata(metadata):
    if not isinstance(metadata, dict) or not REQUIRED_METADATA <= metadata.keys():
        raise ValueError("Pro checkpoint metadata is incomplete")
    value = json.loads(json.dumps(metadata, sort_keys=True, allow_nan=False))
    ProAdapterConfig.from_dict(value["config"])
    if value["scope"] not in ("target", "joint") or value["memory_mode"] not in MEMORY_MODES:
        raise ValueError("Invalid Pro scope or memory_mode metadata")
    if not isinstance(value["control_lora_targets"], list) or any(not isinstance(n, str) for n in value["control_lora_targets"]):
        raise ValueError("control_lora_targets must be an explicit ordered list")
    for name in REQUIRED_METADATA - {"config", "scope", "memory_mode", "control_lora_targets"}:
        if not isinstance(value[name], str) or not value[name]:
            raise ValueError(f"Metadata signature must be nonempty: {name}")
    return value


def _validate_model(model, metadata, *, require_memory_mode=True):
    report = audit_trainable_parameters(model, control_lora_targets=metadata["control_lora_targets"])
    if model.task_adapter.config.to_dict() != metadata["config"]:
        raise ValueError("Checkpoint Pro adapter config differs")
    if getattr(model, "injection_scope", metadata["scope"]) != metadata["scope"]:
        raise ValueError("Checkpoint injection scope differs")
    if require_memory_mode and audit_memory_policy(model)["memory_mode"] != metadata["memory_mode"]:
        raise ValueError("Checkpoint memory_mode differs from model")
    return report


def _incremental_state(model, targets):
    parameters = dict(model.named_parameters())
    return {"task": ordinary_cpu(model.task_adapter.state_dict()),
        "control_lora": {f"{name}.{suffix}": ordinary_cpu(parameters[f"{name}.{suffix}"].detach())
                         for name in targets for suffix in ("lora_A", "lora_B")}}


def _validate_incremental(model, state, targets):
    expected = _incremental_state(model, targets)
    for section in ("task", "control_lora"):
        saved = state.get(section)
        if not isinstance(saved, dict) or set(saved) != set(expected[section]):
            raise ValueError(f"Checkpoint {section} keys differ")
        for name, current in expected[section].items():
            item = saved[name]
            if isinstance(current, torch.Tensor):
                if (not isinstance(item, torch.Tensor) or item.shape != current.shape or item.dtype != current.dtype
                        or not torch.isfinite(item).all()):
                    raise ValueError(f"Checkpoint {section} state shape/dtype/finite differs: {name}")
                if section == "task" and name == "gate" and item == 0:
                    raise ValueError("Checkpoint task gate must be enabled")
            elif item != current:
                # Includes the adapter config/enable dict in _extra_state.
                raise ValueError(f"Checkpoint {section} extra state differs: {name}")


def _apply_incremental(model, state):
    model.task_adapter.load_state_dict(state["task"], strict=True)
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, tensor in state["control_lora"].items():
            parameters[name].copy_(tensor)


def _optimizer_signature(optimizer):
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("Pro full checkpoint currently requires AdamW")
    return {"class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
        "groups": [{k: v for k, v in group.items() if k not in ("params", "lr")}
                   for group in optimizer.state_dict()["param_groups"]]}


def _json_equal(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def _validate_optimizer(model, optimizer, lr_scheduler, state, cursor, targets):
    names = optimizer_name_bindings(model, optimizer, control_lora_targets=targets)
    if state["optimizer_name_bindings"] != names:
        raise ValueError("Checkpoint optimizer ordered name bindings differ")
    if not _json_equal(state["optimizer_signature"], _optimizer_signature(optimizer)):
        raise ValueError("Checkpoint optimizer hyperparameters or initial LR differ")
    saved = state["optimizer"]
    groups = saved.get("param_groups", [])
    if len(groups) != len(names) or any(len(g["params"]) != len(n) for g, n in zip(groups, names)):
        raise ValueError("Checkpoint optimizer parameter groups differ")
    saved_static = [{k: v for k, v in g.items() if k not in ("params", "lr")} for g in groups]
    if not _json_equal(saved_static, state["optimizer_signature"]["groups"]):
        raise ValueError("Checkpoint optimizer state hyperparameters differ from metadata")
    stored_lr, current_lr = state["lr_scheduler"], lr_scheduler.state_dict()
    static_keys = ("base_lrs", "union_warmup_updates", "lr_lambdas")
    if any(not _json_equal(stored_lr.get(key), current_lr.get(key)) for key in static_keys):
        raise ValueError("Checkpoint LR scheduler policy differs")
    if stored_lr.get("last_epoch") != cursor.optimizer_step or stored_lr.get("_step_count") != cursor.optimizer_step + 1:
        raise ValueError("Checkpoint LR scheduler cursor differs")
    warmup = stored_lr.get("union_warmup_updates")
    if type(warmup) is not int or warmup < 0 or len(stored_lr["base_lrs"]) != len(groups):
        raise ValueError("Invalid checkpoint LR warmup metadata")
    expected_lr = [base * min((cursor.optimizer_step + 1) / warmup, 1.) if warmup else base
                   for base in stored_lr["base_lrs"]]
    if stored_lr.get("_last_lr") != expected_lr or [g.get("lr") for g in groups] != expected_lr:
        raise ValueError("Checkpoint optimizer/LR values disagree with schedule metadata")
    parameters = dict(model.named_parameters())
    ids = [index for group in groups for index in group["params"]]
    if len(set(ids)) != len(ids) or not set(saved.get("state", {})) <= set(ids):
        raise ValueError("Checkpoint optimizer parameter IDs differ")
    for group, group_names in zip(groups, names):
        for index, name in zip(group["params"], group_names):
            item = saved["state"].get(index)
            if item is None:
                # Some valid task branches can stay unused when geometry/source
                # availability is zero; AdamW has no state for those parameters.
                continue
            allowed = {"step", "exp_avg", "exp_avg_sq"} | ({"max_exp_avg_sq"} if group.get("amsgrad") else set())
            if set(item) != allowed:
                raise ValueError(f"Checkpoint AdamW state keys differ: {name}")
            for key, tensor in item.items():
                shape = () if key == "step" else parameters[name].shape
                if (not isinstance(tensor, torch.Tensor) or tensor.shape != shape or not torch.isfinite(tensor).all()
                        or tensor.dtype != torch.float32):
                    raise ValueError(f"Checkpoint AdamW state shape/dtype/finite differs: {name}.{key}")
            step = float(item["step"])
            if step != int(step) or not 0 <= step <= cursor.optimizer_step:
                raise ValueError("Checkpoint AdamW step differs from update cursor")


def _read_checkpoint(path):
    path = Path(path)
    try:
        manifest = json.loads((path / "checkpoint.json").read_text(encoding="utf-8"))
        if manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError("Checkpoint format version differs; old B checkpoints cannot resume Pro")
        metadata = _canonical_metadata(manifest["metadata"])
        if file_sha256(path / "state.pt") != manifest["state_sha256"]:
            raise ValueError("Checkpoint state checksum differs")
        # Local training artifacts contain optimizer/RNG objects and are not
        # interchangeable with downloaded model weights.
        state = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
        if state.get("format_version") != FORMAT_VERSION or _canonical_metadata(state["metadata"]) != metadata:
            raise ValueError("Checkpoint state format or metadata differs from manifest")
        for key in ("cursor", "frozen_sha256", "trainable_names", "control_lora_inventory", "optimizer_name_bindings"):
            if state[key] != manifest[key]:
                raise ValueError(f"Checkpoint state differs from manifest: {key}")
        cursor = TrainingCursor(**state["cursor"]).validate()
    except (OSError, KeyError, TypeError, RuntimeError, EOFError) as error:
        raise ValueError("Incomplete or malformed Pro checkpoint") from error
    return manifest, state, cursor


def save_checkpoint(path, model, optimizer, lr_scheduler, *, metadata, cursor):
    metadata = _canonical_metadata(metadata)
    cursor.validate()
    report = _validate_model(model, metadata)
    if optimizer is None or lr_scheduler is None:
        raise ValueError("Full checkpoint requires optimizer and LR scheduler")
    if any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.parameters()):
        raise ValueError("Checkpoint must be saved after complete update with cleared gradients")
    targets = metadata["control_lora_targets"]
    state = {"format_version": FORMAT_VERSION, "metadata": metadata, "cursor": asdict(cursor),
        "frozen_sha256": frozen_parameter_digest(model), "trainable_names": list(report["trainable_names"]),
        "control_lora_inventory": list(report["control_lora"]),
        "optimizer_name_bindings": optimizer_name_bindings(model, optimizer, control_lora_targets=targets),
        **_incremental_state(model, targets), "optimizer": ordinary_cpu(optimizer.state_dict()),
        "optimizer_signature": _optimizer_signature(optimizer), "lr_scheduler": copy.deepcopy(lr_scheduler.state_dict()),
        "rng": capture_rng_state()}
    _validate_incremental(model, state, targets)
    _validate_optimizer(model, optimizer, lr_scheduler, state, cursor, targets)
    _validate_rng(state["rng"])
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    with (path / "state.pt").open("xb") as stream:
        torch.save(state, stream)
    manifest = {key: state[key] for key in ("format_version", "metadata", "cursor", "frozen_sha256", "trainable_names",
                "control_lora_inventory", "optimizer_name_bindings")}
    manifest["state_sha256"] = file_sha256(path / "state.pt")
    with (path / "checkpoint.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    return path


def _validate_weights(model, manifest, state, *, require_memory_mode):
    metadata = manifest["metadata"]
    report = _validate_model(model, metadata, require_memory_mode=require_memory_mode)
    if list(report["trainable_names"]) != state["trainable_names"]:
        raise ValueError("Checkpoint ordered trainable parameter names differ")
    if list(report["control_lora"]) != state["control_lora_inventory"]:
        raise ValueError("Checkpoint explicit control LoRA rank/alpha/inventory differs")
    if frozen_parameter_digest(model) != state["frozen_sha256"]:
        raise ValueError("Checkpoint frozen parameter hash differs")
    _validate_incremental(model, state, metadata["control_lora_targets"])


def load_checkpoint(path, model, optimizer, lr_scheduler, *, expected_metadata):
    _reject_hooks(model)
    manifest, state, cursor = _read_checkpoint(path)
    if manifest["metadata"] != _canonical_metadata(expected_metadata):
        raise ValueError("Checkpoint metadata differs from requested resume config/data/scope/protocol")
    _validate_weights(model, manifest, state, require_memory_mode=True)
    _validate_optimizer(model, optimizer, lr_scheduler, state, cursor, manifest["metadata"]["control_lora_targets"])
    _validate_rng(state["rng"])
    # All signatures, tensor layouts, optimizer bindings and RNG are checked
    # before the first model/optimizer/global RNG mutation.
    _apply_incremental(model, state)
    optimizer.load_state_dict(state["optimizer"])
    lr_scheduler.load_state_dict(state["lr_scheduler"])
    optimizer.zero_grad(set_to_none=True)
    restore_rng_state(state["rng"])
    return cursor


def load_adapter_weights(path, model, *, expected_metadata):
    """推理只加载 task/LoRA，不载入 optimizer/RNG；返回完整 manifest。

    expected_metadata 是调用者明确绑定的签名子集，至少包含 config/scope/
    task_signature/code_signature/source_signature。推理允许换验证集与内存策略。
    必须在 inference group offload hook 捕获 CPU 权重快照之前调用。
    """
    _reject_hooks(model)
    required = {"config", "scope", "task_signature", "code_signature", "source_signature"}
    if not isinstance(expected_metadata, dict) or not required <= expected_metadata.keys():
        raise ValueError("Inference weight loading requires explicit config/scope/task/code/source signature bindings")
    manifest, state, _ = _read_checkpoint(path)
    for key, value in expected_metadata.items():
        if key not in manifest["metadata"] or not _json_equal(manifest["metadata"][key], value):
            raise ValueError(f"Inference checkpoint binding differs: {key}")
    _validate_weights(model, manifest, state, require_memory_mode=False)
    _apply_incremental(model, state)
    return manifest
