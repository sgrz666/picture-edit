"""16 个原生 Union 控制块上的逐实例人物任务接入。

install_pro_adapter 原地升级实例并返回 audit；既有参数对象和 state key 不变。
bind_task_condition 只在 with 作用域提供默认条件。每次 transformer.forward 都会
重建 joint target mask，退出时清理执行上下文；checkpoint kwargs 持有该次执行。
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import inspect
import math
import weakref

import torch
from torch import nn

from .adapter import ProTaskAdapter
from .contracts import ProAdapterConfig, TaskCondition

_UNSET = object()
_CONTROL_LAYERS = tuple(range(0, 32, 2))


@dataclass(frozen=True)
class _TaskBinding:
    condition: TaskCondition | None
    enabled: bool


@dataclass(frozen=True)
class _TaskExecution:
    condition: TaskCondition | None
    enabled: bool
    target_mask: torch.Tensor
    timestep: torch.Tensor


def _require_installed(model):
    if not isinstance(model, _ProTransformerMixin):
        raise ValueError("Pro adapter is not installed on this transformer")


def _execution(model, values, condition, enabled):
    """Match native four-token image slots to the actual reference/target grids."""
    hidden, text = values["hidden_states"], values["encoder_hidden_states"]
    if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[0] != 1
            or not isinstance(text, torch.Tensor) or text.ndim != 3 or text.shape[0] != 1):
        raise ValueError("Pro requires hidden and encoder batch B=1")
    if not getattr(model.config, "causal_condition", False):
        raise ValueError("Pro requires causal_condition=True")
    if getattr(model, "sp_world_size", 1) != 1:
        raise ValueError("Pro supports single-card execution only")
    if values.get("kv_cache") is not None or values.get("kv_cache_mode") is not None:
        raise ValueError("Pro requires KV cache disabled")
    timestep = values["timestep"]
    if (not isinstance(timestep, torch.Tensor) or timestep.shape != (1,)
            or not torch.isfinite(timestep).all()):
        raise ValueError("timestep must have finite shape [B=1]")
    shapes = values["img_shapes"]
    if not isinstance(shapes, (list, tuple)) or len(shapes) != 1 or not shapes[0]:
        raise ValueError("img_shapes must contain exactly one sample")
    shapes = shapes[0]
    if any(not isinstance(shape, (list, tuple)) or len(shape) != 3
           or any(type(n) is not int or n <= 0 for n in shape) or shape[0] != 1
           for shape in shapes):
        raise ValueError("img_shapes must contain positive (1,H,W) image grids")
    lengths = [math.prod(shape) for shape in shapes]
    if any(length % 4 for length in lengths) or sum(lengths) != hidden.shape[1]:
        raise ValueError("img_shapes must match image latent tokens in four-token slots")
    mask = values["img_mask"]
    target_slots = lengths[-1] // 4
    if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
            or mask.shape != (1, text.shape[1] + target_slots)):
        raise ValueError("img_mask must be bool [1,text_length+target_slots]")
    if not bool(mask[0, -target_slots:].all()):
        raise ValueError("img_mask must append the target image slots after the text/reference sequence")
    if int(mask[0, :text.shape[1]].sum()) * 4 != sum(lengths[:-1]):
        raise ValueError("img_mask reference slots must match img_shapes")
    text_mask = values.get("encoder_hidden_states_mask")
    if text_mask is not None and (not isinstance(text_mask, torch.Tensor) or text_mask.shape != text.shape[:2]):
        raise ValueError("encoder_hidden_states_mask must match encoder batch and length")
    control = values.get("control_context")
    if control is not None and (not isinstance(control, torch.Tensor) or control.ndim != 3
                                or control.shape[:2] != hidden.shape[:2]
                                or control.shape[-1] != model.control_img_in.in_features):
        raise ValueError("control_context must match all reference and target latent tokens")
    if enabled:
        if not isinstance(condition, TaskCondition):
            raise ValueError("enabled Pro requires task_condition; pass task_enabled=False for Union baseline")
        condition.validate(model.task_adapter.config)
        if condition.person_valid.shape[0] != 1:
            raise ValueError("task_condition requires batch B=1")
        if condition.target_hw != tuple(shapes[-1][1:]):
            raise ValueError("task_condition target_hw must match actual target img_shapes grid")
        if control is None:
            raise ValueError("enabled Pro requires control_context to execute the Union control chain")
    mask = mask.to(device=hidden.device)
    repeats = torch.where(mask, 4, 1)[0]
    image_pad_mask = torch.repeat_interleave(mask[0], repeats)
    _, target = model.build_token_metadata(image_pad_mask, shapes)
    if target.dtype != torch.bool or target.ndim != 1 or int(target.sum()) != lengths[-1]:
        raise ValueError("native build_token_metadata returned an invalid target mask")
    return _TaskExecution(condition, enabled, target, timestep.detach().clone())


class _ProControlBlockMixin:
    def forward(self, c, x, *, task_execution=None, **native_kwargs):
        if self.block_id == 0:
            c = self.before_proj(c) + x
            history = []
        else:
            history = list(torch.unbind(c))
            c = history.pop()
        # Deliberately bypass only the original control wrapper. Native attention,
        # MLP and all their parameter objects stay on this same module instance.
        c = self._pro_native_forward(self, c, **native_kwargs)
        if task_execution is not None and task_execution.enabled:
            owner = self._pro_owner_ref()
            if owner is None:
                raise RuntimeError("Pro control block lost its owning transformer")
            # First positional CUDA hidden triggers normal module offload hooks.
            # forward_layer returns the updated hidden, not an additive delta.
            c = owner.task_adapter(c, stage=self._pro_stage,
                                   target_mask=task_execution.target_mask,
                                   conditions=task_execution.condition,
                                   timestep=task_execution.timestep)
        return torch.stack([*history, self.after_proj(c), c])


class _ProTransformerMixin:
    def forward(self, *args, task_condition=_UNSET, task_enabled=_UNSET, **native_kwargs):
        binding = self._pro_binding_context.get()
        if task_condition is _UNSET:
            task_condition = binding.condition if binding is not None else None
        if task_enabled is _UNSET:
            task_enabled = binding.enabled if binding is not None else True
        if type(task_enabled) is not bool:
            raise ValueError("task_enabled must be bool")
        bound = self._pro_forward_signature.bind(self, *args, **native_kwargs)
        bound.apply_defaults()
        execution = _execution(self, bound.arguments, task_condition,
                               task_enabled and self.task_adapter.enabled)
        token = self._pro_execution_context.set(execution)
        try:
            return super().forward(*args, **native_kwargs)
        finally:
            self._pro_execution_context.reset(token)

    def forward_control(self, control_joint, x_joint, kwargs):
        execution = self._pro_execution_context.get()
        if execution is None:
            raise RuntimeError("Pro forward_control requires an active transformer.forward")
        # The official nonreentrant checkpoint captures these kwargs. Recompute
        # reads this immutable execution, never a reset ContextVar or last batch.
        kwargs = dict(kwargs, task_execution=execution)
        return super().forward_control(control_joint, x_joint, kwargs)


def install_pro_adapter(model: nn.Module, config: ProAdapterConfig | None = None) -> dict:
    """原地注册 root task_adapter，保留原 Union 参数；返回配置/参数/层次审计。

    只冻结原模型参数，保留已安装 control LoRA 的 requires_grad。必须在任何
    offload hook 注册前调用；不要在此后重新调用旧 install_control_lora（会 refreeze）。
    """
    if isinstance(model, _ProTransformerMixin) or hasattr(model, "task_adapter"):
        raise ValueError("Pro adapter is already installed")
    config = ProAdapterConfig() if config is None else config
    if not isinstance(config, ProAdapterConfig):
        raise ValueError("config must be ProAdapterConfig")
    blocks = getattr(model, "control_blocks", None)
    if (not isinstance(blocks, nn.ModuleList) or len(blocks) != 16
            or tuple(getattr(model, "control_layers", ())) != _CONTROL_LAYERS
            or tuple(getattr(block, "block_id", None) for block in blocks) != _CONTROL_LAYERS):
        raise ValueError("Pro requires all 16 official control blocks at base layers 0,2,...,30")
    if getattr(model, "inner_dim", None) != config.hidden_dim:
        raise ValueError("Pro hidden_dim must match transformer.inner_dim")
    if not hasattr(model, "forward_control") or not callable(getattr(model, "build_token_metadata", None)):
        raise ValueError("transformer must implement native forward_control and build_token_metadata")
    for index, block in enumerate(blocks):
        projections = [getattr(block, "after_proj", None)]
        if index == 0:
            projections.append(getattr(block, "before_proj", None))
        if any(getattr(proj, "in_features", None) != config.hidden_dim
               or getattr(proj, "out_features", None) != config.hidden_dim for proj in projections):
            raise ValueError("control before_proj/after_proj dimensions differ from hidden_dim")
        if len(type(block).__mro__) < 3 or type(block).__mro__[1].forward is nn.Module.forward:
            raise ValueError("control block must inherit the native transformer block")
    original_parameters = dict(model.named_parameters())
    control_keys = tuple(name for name in original_parameters if name.startswith(("control_blocks.", "control_img_in.")))
    # LoRA adds .base_layer. to the old Linear names; that is an existing opt-in
    # loader policy. Pro itself never renames or wraps any of these modules.
    native_control_keys = tuple(name.replace(".base_layer.", ".") for name in control_keys
                                if not name.endswith((".lora_A", ".lora_B")))
    if len(native_control_keys) != 180:
        raise ValueError("Pro requires all 180 native Union control parameter keys")
    from ..qwen21_union.weights import ControlLoRALinear
    lora_parameters = {id(p) for module in model.modules() if isinstance(module, ControlLoRALinear)
                       for p in (module.lora_A, module.lora_B)}
    # Finish construction and validate the signature before modifying the model.
    signature = inspect.signature(type(model).forward)
    required = {"hidden_states", "encoder_hidden_states", "timestep", "img_shapes", "img_mask"}
    if not required <= set(signature.parameters):
        raise ValueError("transformer forward lacks native image-slot layout arguments")
    adapter = ProTaskAdapter(config).to(device=blocks[0].after_proj.weight.device)
    block_types = {}
    for block in blocks:
        original_type = type(block)
        if original_type not in block_types:
            block_types[original_type] = type(
                f"Pro{original_type.__name__}", (_ProControlBlockMixin, original_type),
                {"_pro_native_forward": staticmethod(original_type.__mro__[1].forward), "__module__": __name__})
    transformer_type = type(f"Pro{type(model).__name__}", (_ProTransformerMixin, type(model)), {"__module__": __name__})
    for parameter in original_parameters.values():
        if id(parameter) not in lora_parameters:
            parameter.requires_grad_(False)
    model.task_adapter = adapter
    model._pro_binding_context = ContextVar(f"pro_binding_{id(model)}", default=None)
    model._pro_execution_context = ContextVar(f"pro_execution_{id(model)}", default=None)
    model._pro_forward_signature = signature
    owner = weakref.ref(model)
    for stage, block in enumerate(blocks):
        original_type = type(block)
        block.__class__ = block_types[original_type]
        block._pro_owner_ref = owner
        block._pro_stage = stage
    model.__class__ = transformer_type
    additions = [*(cls.__name__ for cls in block_types.values()), "ProTaskAdapter"]
    for attr in ("_no_split_modules", "_repeated_blocks"):
        setattr(model, attr, list(dict.fromkeys([*getattr(model, attr, []), *additions])))
    return {"config": config.to_dict(), "config_signature": config.signature(),
            "num_control_blocks": 16, "control_layers": _CONTROL_LAYERS,
            "control_parameter_keys": control_keys, "native_control_parameter_keys": native_control_keys,
            "native_control_parameter_count": len(native_control_keys),
            "parameter_objects_preserved": all(dict(model.named_parameters())[name] is parameter
                                                for name, parameter in original_parameters.items()),
            "task_adapter": adapter.trainable_parameter_report(),
            "task_enabled": adapter.enabled, "lora_present": bool(lora_parameters),
            "disable_all_increments": "set_pro_enabled(model, False, include_lora=True)"}


@contextmanager
def bind_task_condition(model: nn.Module, pack: TaskCondition | None, enabled: bool = True):
    """作用域内的默认 pack；forward 的显式 task_condition/task_enabled 优先。"""
    _require_installed(model)
    if type(enabled) is not bool:
        raise ValueError("enabled must be bool")
    if enabled and model.task_adapter.enabled:
        if not isinstance(pack, TaskCondition):
            raise ValueError("enabled Pro requires task_condition")
        pack.validate(model.task_adapter.config)
    token = model._pro_binding_context.set(_TaskBinding(pack, enabled))
    try:
        yield model
    finally:
        model._pro_binding_context.reset(token)


def set_pro_enabled(model: nn.Module, enabled: bool, *, include_lora: bool = False) -> dict:
    """切换任务分支；include_lora=True 才同时切换旧控制 LoRA。

    task_enabled=False 或 adapter.enabled=False 只跳过任务分支；恢复原 Union
    数值还需关闭已安装的 LoRA。返回本次切换后的实际状态。
    """
    _require_installed(model)
    if type(enabled) is not bool or type(include_lora) is not bool:
        raise ValueError("enabled and include_lora must be bool")
    from ..qwen21_union.weights import ControlLoRALinear
    modules = [module for module in model.modules() if isinstance(module, ControlLoRALinear)]
    model.task_adapter.enabled = enabled
    if include_lora:
        for module in modules:
            module.enabled = enabled
    return {"task_enabled": model.task_adapter.enabled, "lora_present": bool(modules),
            "lora_enabled": any(module.enabled for module in modules), "include_lora": include_lora}

