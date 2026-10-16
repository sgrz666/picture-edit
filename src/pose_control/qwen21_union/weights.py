"""官方控制权重审计与精确 control-only LoRA，不依赖 PEFT 后缀匹配。"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

OFFICIAL_SOURCE_COMMIT = "13bb3c625e8b1e457d7124d92207b49e00e4a6fe"
OFFICIAL_UNION_REVISION = "8a4702014d4dabb5f896fcba917e2ee0a961465f"
EXPECTED_LORA_PARAMETERS = 23_267_344
LORA_SUFFIXES = (
    "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
    "img_mlp.proj", "img_mlp.gate_layer", "img_mlp.out", "after_proj",
)


def control_lora_targets() -> tuple[str, ...]:
    return ("control_img_in", "control_blocks.0.before_proj") + tuple(
        f"control_blocks.{index}.{suffix}" for index in range(16) for suffix in LORA_SUFFIXES
    )


class ControlLoRALinear(nn.Module):
    """冻结原 Linear；A/B 以 FP32 存储，增量随基座输出 dtype 返回。"""

    def __init__(self, base_layer: nn.Linear, rank: int = 16, alpha: float = 16):
        super().__init__()
        self.base_layer = base_layer.requires_grad_(False)
        self.rank, self.alpha = rank, float(alpha)
        self.enabled = True
        self.lora_A = nn.Parameter(torch.empty(rank, base_layer.in_features,
                                               device=base_layer.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, rank,
                                               device=base_layer.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    @property
    def in_features(self):
        return self.base_layer.in_features

    @property
    def out_features(self):
        return self.base_layer.out_features

    def _apply(self, fn, recurse=True):
        originals = {name: (getattr(self, name).data,
                           getattr(self, name).grad.detach() if getattr(self, name).grad is not None else None)
                     for name in ("lora_A", "lora_B")}
        super()._apply(fn, recurse=recurse)
        # Parent .to(dtype=...) must not silently convert trainable storage to BF16.
        # Recover from the original FP32 storage, never from a rounded BF16 copy.
        for name, (original_data, original_grad) in originals.items():
            parameter = getattr(self, name)
            if original_data.is_meta and not parameter.is_meta:
                # to_empty materialization has no values to preserve.
                parameter.data = torch.empty_like(parameter, dtype=torch.float32)
            else:
                parameter.data = original_data.to(device=parameter.device, dtype=torch.float32)
            if original_grad is not None:
                parameter.grad = original_grad.to(device=parameter.device, dtype=torch.float32)
        return self

    def forward(self, hidden_states):
        output = self.base_layer(hidden_states)
        if not self.enabled:
            return output
        delta = F.linear(F.linear(hidden_states.float(), self.lora_A), self.lora_B)
        return output + delta.to(output.dtype) * (self.alpha / self.rank)


def install_control_lora(model: nn.Module, *, rank=16, alpha=16,
                         expected_parameters=EXPECTED_LORA_PARAMETERS) -> dict:
    if rank != 16 or float(alpha) != 16:
        raise ValueError("Plan B requires rank=16 and alpha=16")
    targets = control_lora_targets()
    modules = dict(model.named_modules())
    bad = [name for name in targets if not isinstance(modules.get(name), nn.Linear)]
    if bad:
        raise ValueError(f"LoRA targets must be unwrapped Linear modules: {bad}")
    count = sum(rank * (modules[name].in_features + modules[name].out_features) for name in targets)
    if expected_parameters is not None and count != expected_parameters:
        raise ValueError(f"LoRA parameter budget differs: {count} != {expected_parameters}")
    # Validate every target before freezing or replacing anything.
    model.requires_grad_(False)
    for name in targets:
        parent_path, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, child, ControlLoRALinear(modules[name], rank, alpha))
    return audit_trainable_parameters(model, expected_parameters=expected_parameters)


def _lora_modules(model):
    return {name: module for name, module in model.named_modules() if isinstance(module, ControlLoRALinear)}


def audit_trainable_parameters(model, *, expected_parameters=EXPECTED_LORA_PARAMETERS):
    modules = _lora_modules(model)
    targets = control_lora_targets()
    if set(modules) != set(targets):
        raise ValueError("LoRA module inventory differs from exact 130-module whitelist")
    allowed = {f"{name}.{suffix}" for name in targets for suffix in ("lora_A", "lora_B")}
    actual = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if actual != allowed:
        raise ValueError(f"Unexpected trainable parameters: extra={sorted(actual - allowed)}, missing={sorted(allowed - actual)}")
    count = 0
    for name, parameter in model.named_parameters():
        if name in allowed:
            if parameter.dtype != torch.float32:
                raise ValueError(f"LoRA storage must be FP32: {name}")
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise ValueError(f"Nonfinite LoRA gradient: {name}")
            count += parameter.numel()
        elif parameter.grad is not None:
            raise ValueError(f"Frozen parameter has gradient: {name}")
    if expected_parameters is not None and count != expected_parameters:
        raise ValueError(f"LoRA parameter budget differs: {count} != {expected_parameters}")
    return {"targets": targets, "trainable_parameters": count, "rank": 16, "alpha": 16,
            "trainable_tensors": len(allowed)}


def lora_state_dict(model) -> dict[str, torch.Tensor]:
    return {f"{name}.{suffix}": getattr(module, suffix).detach().cpu().clone()
            for name, module in _lora_modules(model).items() for suffix in ("lora_A", "lora_B")}


def validate_lora_state_dict(model, state: Mapping):
    expected = {f"{name}.{suffix}": getattr(module, suffix)
                for name, module in _lora_modules(model).items() for suffix in ("lora_A", "lora_B")}
    if not expected or set(state) != set(expected):
        raise ValueError("LoRA state keys differ")
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or tensor.shape != expected[name].shape:
            raise ValueError(f"LoRA state shape differs: {name}")
        if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError(f"LoRA state must be finite FP32: {name}")


def load_lora_state_dict(model, state: Mapping):
    validate_lora_state_dict(model, state)
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, tensor in state.items():
            parameters[name].copy_(tensor)


def set_lora_enabled(model, enabled: bool):
    for module in _lora_modules(model).values():
        module.enabled = bool(enabled)


def frozen_parameter_digest(model) -> str:
    """显式调用的完整冻结状态哈希；大型模型调用会发生 CPU 传输。"""
    lora_keys = {f"{name}.{suffix}" for name in _lora_modules(model)
                 for suffix in ("lora_A", "lora_B")}
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name in lora_keys:
            continue
        digest.update(f"{name}|{tensor.dtype}|{tuple(tensor.shape)}".encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def audit_control_state(model, state: Mapping, *, expected_tensors=180) -> dict:
    """在安装 LoRA 之前审计原始 Union 文件，拒绝任何缺失/多余/shape 变化。"""
    expected = {key: value for key, value in model.state_dict().items()
                if key.startswith(("control_blocks.", "control_img_in."))}
    missing, extra = set(expected) - set(state), set(state) - set(expected)
    if missing:
        raise ValueError(f"Union control state missing keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"Union control state unexpected keys: {sorted(extra)}")
    if expected_tensors is not None and len(state) != expected_tensors:
        raise ValueError(f"Union tensor count differs: {len(state)} != {expected_tensors}")
    for key, tensor in state.items():
        if tuple(tensor.shape) != tuple(expected[key].shape):
            raise ValueError(f"Union control shape differs: {key}")
    return {"tensor_count": len(state), "source_commit": OFFICIAL_SOURCE_COMMIT,
            "union_revision": OFFICIAL_UNION_REVISION}
