"""训练专用冻结 Linear CPU 权重；输入梯度保留，参数对象与 key 不变。"""
from __future__ import annotations

from collections import Counter

import torch
from torch import nn
from torch.nn import functional as F

MEMORY_MODES = ("resident", "frozen_linear_cpu")


class _FrozenLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, bias):
        ctx.save_for_backward(weight)
        ctx.input_dtype = inputs.dtype
        output = F.linear(inputs, weight.to(inputs.device), None if bias is None else bias.to(inputs.device))
        ctx.compute_dtype = output.dtype
        return output

    @staticmethod
    def backward(ctx, gradient):
        weight, = ctx.saved_tensors
        # Native Linear backward uses flattened 2-D mm, including leading batch
        # dimensions. No GPU copy of W is retained across forward/backward.
        with torch.autocast(device_type=gradient.device.type, enabled=False):
            temporary = weight.to(device=gradient.device, dtype=ctx.compute_dtype)
            inputs_gradient = gradient.to(ctx.compute_dtype).reshape(-1, gradient.shape[-1]).mm(temporary)
        return inputs_gradient.reshape(*gradient.shape[:-1], weight.shape[1]).to(ctx.input_dtype), None, None


class FrozenLinearCPU(nn.Linear):
    """只通过逐实例 __class__ 升级；原 nn.Parameter 不替换、不改名。"""
    def forward(self, inputs):
        if self.weight.device.type != "cpu" or self.weight.requires_grad or (self.bias is not None and self.bias.requires_grad):
            raise ValueError("FrozenLinearCPU requires frozen canonical CPU weights")
        return _FrozenLinear.apply(inputs, self.weight, self.bias)

    def _apply(self, fn, recurse=True):
        raise ValueError("Use place_training_model; whole-model .to would move canonical CPU weights")


def _reject_hooks(model):
    for name, module in model.named_modules():
        if any(getattr(module, key, None) is not None for key in
               ("_hf_hook", "_diffusers_hook", "_group_offload_hook", "_offload_hook", "_auto_device_hooks")):
            raise ValueError(f"Existing inference/offload hook is incompatible with training: {name}")


def place_training_model(model, device, *, memory_mode="resident"):
    """在 CPU 构建并装好 Pro/LoRA 后调用；只移动非 offload 参数与 buffer。

    不先 model.to(cuda)，不量化，不改变 dtype，不使用 VideoX inference group
    offload。任务分支与 A/B 保持原 FP32，冻结 LayerNorm 等保持原始 dtype。
    """
    if memory_mode not in MEMORY_MODES:
        raise ValueError(f"Unknown memory_mode: {memory_mode}")
    _reject_hooks(model)
    device = torch.device(device)
    installed = getattr(model, "_pro_memory_mode", None)
    if installed is not None and installed != memory_mode:
        raise ValueError("memory_mode cannot change after training placement")
    modules = tuple(model.named_modules())
    if installed is None and any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("Construct on CPU before selective training placement")
    candidates = []
    if memory_mode == "frozen_linear_cpu":
        for name, module in modules:
            if (type(module) is nn.Linear and not name.startswith("task_adapter")
                    and not module.weight.requires_grad and (module.bias is None or not module.bias.requires_grad)):
                candidates.append(module)
        for module in candidates:
            module.__class__ = FrozenLinearCPU
    # Do not recurse via _apply: every Tensor is visited once, with no temporary
    # full-model CUDA residency. Preserve all original Parameter identities.
    offloaded = {id(p) for _, module in modules if isinstance(module, FrozenLinearCPU)
                 for p in module.parameters(recurse=False)}
    with torch.no_grad():
        for parameter in model.parameters():
            destination = torch.device("cpu") if id(parameter) in offloaded else device
            parameter.data = parameter.data.to(destination)
            if parameter.grad is not None:
                parameter.grad = parameter.grad.to(destination)
        for _, module in modules:
            for key, value in module._buffers.items():
                if value is not None:
                    module._buffers[key] = value.to(device)
    model._pro_memory_mode = memory_mode
    model._pro_training_device = str(device)
    return audit_memory_policy(model)


def audit_memory_policy(model):
    _reject_hooks(model)
    mode = getattr(model, "_pro_memory_mode", "resident")
    layers = {name: module for name, module in model.named_modules() if isinstance(module, FrozenLinearCPU)}
    if mode not in MEMORY_MODES or (mode == "resident" and layers):
        raise ValueError("memory_mode differs from installed Linear policy")
    for name, module in layers.items():
        for parameter in module.parameters(recurse=False):
            if parameter.requires_grad or parameter.device.type != "cpu":
                raise ValueError(f"Offloaded parameter must remain frozen on CPU: {name}")
    return {"memory_mode": mode, "offloaded_linear_count": len(layers), "offloaded_names": tuple(layers),
            "offloaded_dtype_counts": dict(Counter(str(module.weight.dtype) for module in layers.values())),
            "offloaded_parameter_bytes": sum(p.numel() * p.element_size() for m in layers.values() for p in m.parameters()),
            "parameter_device_counts": dict(Counter(str(p.device) for p in model.parameters()))}
