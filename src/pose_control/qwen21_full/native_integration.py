"""Per-instance native Qwen block integration, including checkpoint-safe context."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial
import inspect
import math
import weakref

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .native_body import PreparedNativeGeometry, Qwen21NativeBody
from .native_geometry import NativeGeometryFeatures

_UNSET = object()


@dataclass(frozen=True)
class _Binding:
    features: NativeGeometryFeatures | None
    enabled: bool
    valid_mask: torch.Tensor | None
    strength: float
    conv_input_gate: bool = False


@dataclass(frozen=True)
class _Execution:
    enabled: bool
    target_mask: torch.Tensor
    sigma: torch.Tensor
    geometry: PreparedNativeGeometry | None
    valid_mask: torch.Tensor | None
    strength: float


def _prepare_execution(model, values, features, enabled, valid_mask, strength, conv_input_gate=False):
    if type(enabled) is not bool:
        raise ValueError("native_enabled must be bool")
    if values.get("kv_cache") is not None or values.get("kv_cache_mode") is not None:
        raise ValueError("Native Body requires KV cache disabled in this first integration")
    hidden, text = values["hidden_states"], values["encoder_hidden_states"]
    if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[0] != 1
            or not isinstance(text, torch.Tensor) or text.ndim != 3 or text.shape[0] != 1):
        raise ValueError("Native Body integration currently requires batch B=1")
    shapes = values["img_shapes"]
    if not isinstance(shapes, (list, tuple)) or len(shapes) != 1 or not shapes[0]:
        raise ValueError("img_shapes must contain one sample with reference/target grids")
    shapes = shapes[0]
    if any(not isinstance(shape, (list, tuple)) or len(shape) != 3
           or any(type(i) is not int or i <= 0 for i in shape) or shape[0] != 1 for shape in shapes):
        raise ValueError("Image grids must be positive (1,H,W)")
    lengths = [math.prod(shape) for shape in shapes]
    if any(length % 4 for length in lengths) or sum(lengths) != hidden.shape[1]:
        raise ValueError("img_shapes must match all image latents in four-token slots")
    mask = values["img_mask"]
    target_slots = lengths[-1] // 4
    if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
            or mask.shape != (1, text.shape[1] + target_slots)
            or not bool(mask[0, -target_slots:].all())
            or int(mask[0, :text.shape[1]].sum()) * 4 != sum(lengths[:-1])):
        raise ValueError("img_mask reference/target slots do not match the actual image grids")
    repeats = torch.where(mask[0], 4, 1)
    image_pad_mask = torch.repeat_interleave(mask[0], repeats)
    _, target_mask = model.build_token_metadata(image_pad_mask, shapes)
    sigma = values["timestep"]
    if (not isinstance(sigma, torch.Tensor) or sigma.shape != (1,)
            or not bool(torch.isfinite(sigma).all()) or bool(((sigma < 0) | (sigma > 1)).any())):
        raise ValueError("timestep must be normalized sigma [1] in [0,1]")
    geometry = None
    if valid_mask is not None:
        # Mirrors Qwen21NativeBody.forward: bool for an all-or-nothing region, 0..1 for canvas
        # coverage so padded boundary tokens are damped instead of injected at full strength.
        if (not isinstance(valid_mask, torch.Tensor) or valid_mask.shape != (1, lengths[-1])
                or not (valid_mask.dtype == torch.bool or valid_mask.is_floating_point())
                or (valid_mask.is_floating_point() and (
                    not bool(torch.isfinite(valid_mask).all())
                    or bool((valid_mask < 0).any()) or bool((valid_mask > 1).any())))):
            raise ValueError(
                "native_valid_mask must be bool or 0..1 area coverage [1,N_target]")
    if type(strength) not in (int, float) or not math.isfinite(strength) or strength < 0:
        raise ValueError("native_strength must be finite and nonnegative")
    if enabled:
        if not isinstance(features, NativeGeometryFeatures) or features.target_token_hw != tuple(shapes[-1][1:]):
            raise ValueError("native_geometry must match the current target grid")
        if conv_input_gate and valid_mask is None:
            raise ValueError("conv_input_gate needs a valid_mask: with nothing declared invalid it "
                             "would silently gate every token")
        # A token whose 16x16 block holds no real image area must not feed the local convolution:
        # otherwise padding still reaches its neighbours and §8's "fully invalid tokens get zero
        # gradient" cannot hold however the residual is damped afterwards.
        conv_input_valid = None if not conv_input_gate else (valid_mask != 0)
        geometry = model.native_body.prepare_geometry(features, conv_input_valid=conv_input_valid)
        if geometry.tokens.device != hidden.device:
            raise ValueError("Prepared geometry and Qwen inputs must share a device")
    return _Execution(enabled, target_mask.to(hidden.device), sigma.clone(), geometry,
                      None if valid_mask is None else valid_mask.clone(), float(strength))


class _NativeBlockMixin:
    def forward(self, *args, native_execution=None, **kwargs):
        result = super().forward(*args, **kwargs)
        owner = self._native_body_owner()
        if owner is None:
            raise RuntimeError("Native Body block lost its owning transformer")
        execution = native_execution if native_execution is not None else owner._native_body_execution.get()
        if execution is None or not execution.enabled:
            return result
        return owner.native_body(
            result, prepared=execution.geometry, layer=self._native_body_layer,
            sigma=execution.sigma, target_mask=execution.target_mask,
            valid_mask=execution.valid_mask, strength=execution.strength,
        )


class _NativeTransformerMixin:
    def forward(self, *args, native_geometry=_UNSET, native_enabled=_UNSET,
                native_valid_mask=_UNSET, native_strength=_UNSET,
                native_conv_input_gate=_UNSET, **kwargs):
        if self._native_body_execution.get() is not None:
            raise RuntimeError("Concurrent/reentrant Native Body forward is unsupported")
        if getattr(self, "_pic_edit_qwen21_control_active", False):
            raise ValueError("Disable the legacy Full control context before using Native Body")
        binding = self._native_body_binding.get()
        if native_geometry is _UNSET:
            native_geometry = None if binding is None else binding.features
        if native_enabled is _UNSET:
            native_enabled = True if binding is None else binding.enabled
        if native_valid_mask is _UNSET:
            native_valid_mask = None if binding is None else binding.valid_mask
        if native_strength is _UNSET:
            native_strength = 1.0 if binding is None else binding.strength
        if native_conv_input_gate is _UNSET:
            native_conv_input_gate = False if binding is None else binding.conv_input_gate
        bound = self._native_body_signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        execution = _prepare_execution(self, bound.arguments, native_geometry,
                                       native_enabled, native_valid_mask, native_strength,
                                       native_conv_input_gate)
        token = self._native_body_execution.set(execution)
        checkpoint_function = getattr(self, "_gradient_checkpointing_func", None)
        checkpointing = torch.is_grad_enabled() and self.gradient_checkpointing
        try:
            if checkpointing:
                standard = (getattr(checkpoint_function, "__module__", "") == "diffusers.models.modeling_utils"
                            and getattr(checkpoint_function, "__name__", "") == "_gradient_checkpointing_func")
                explicit_nonreentrant = (isinstance(checkpoint_function, partial)
                                         and checkpoint_function.func is torch_checkpoint
                                         and checkpoint_function.keywords.get("use_reentrant") is False)
                if not standard and not explicit_nonreentrant:
                    raise ValueError("Native Body supports standard non-reentrant gradient checkpointing only")

                def checkpoint_bound(block, *block_args):
                    # The saved callable owns THIS immutable execution. During
                    # backward it never reads a reset ContextVar or a later batch.
                    return checkpoint_function(partial(block, native_execution=execution), *block_args)

                self._gradient_checkpointing_func = checkpoint_bound
            return super().forward(*args, **kwargs)
        finally:
            if checkpointing:
                self._gradient_checkpointing_func = checkpoint_function
            self._native_body_execution.reset(token)


def install_native_body(model: nn.Module, adapter: Qwen21NativeBody) -> dict:
    """Install before offload; original base parameter objects/keys stay intact.

    Convert the backbone's dtype before installing the FP32 task module. Save
    adapter/frontend state separately; this does not export a new base checkpoint.
    """
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel

    if isinstance(model, _NativeTransformerMixin) or hasattr(model, "native_body"):
        raise ValueError("Native Body is already installed")
    if not isinstance(model, QwenImage21Transformer2DModel) or hasattr(model, "control_blocks"):
        raise ValueError("Native Body requires the native Qwen21 transformer, without Union control blocks")
    if not isinstance(adapter, Qwen21NativeBody):
        raise ValueError("adapter must be Qwen21NativeBody")
    if (not model.config.causal_condition or model.inner_dim != adapter.config.hidden_dim
            or len(model.transformer_blocks) != adapter.config.num_layers):
        raise ValueError("Native Body width/layer configuration must match causal Qwen21")
    if getattr(model, "_pic_edit_qwen21_control_active", False):
        raise ValueError("A legacy Full context is already active")
    adapter._require_fp32()
    before = dict(model.named_parameters())
    signature = inspect.signature(type(model).forward)
    model.requires_grad_(False)
    model.add_module("native_body", adapter)
    adapter.requires_grad_(True).train(model.training)
    model._native_body_signature = signature
    model._native_body_execution = ContextVar(f"native_body_{id(model)}", default=None)
    model._native_body_binding = ContextVar(f"native_body_binding_{id(model)}", default=None)
    model.__class__ = type(f"NativeBody{type(model).__name__}", (_NativeTransformerMixin, type(model)), {})
    for layer, block in enumerate(model.transformer_blocks):
        block._native_body_owner = weakref.ref(model)
        block._native_body_layer = layer
        block.__class__ = type(f"NativeBody{type(block).__name__}", (_NativeBlockMixin, type(block)), {})
    after = dict(model.named_parameters())
    preserved = all(after[name] is parameter for name, parameter in before.items())
    if not preserved:
        raise RuntimeError("Native Body installation replaced a base parameter")
    return {"schema": "qwen21_full_native_install_v1", "config": adapter.config.to_dict(),
            "injection_layers": list(adapter.config.injection_layers),
            "base_parameter_objects_preserved": preserved,
            "trainable_parameter_count": sum(p.numel() for p in adapter.parameters()),
            "base_parameter_count": sum(p.numel() for p in before.values()),
            "kv_cache_supported": False, "batch_size": 1}


@contextmanager
def bind_native_body(model, features: NativeGeometryFeatures | None, *, enabled=True,
                     valid_mask=None, strength=1.0, conv_input_gate=False):
    """Supply geometry to the original pipeline's repeated transformer calls.

    Each call still constructs its own layout/sigma execution. Checkpoint
    recomputation owns that execution after this sampling/binding scope exits.

    ``conv_input_gate`` stops fully-invalid canvas tokens from entering the local geometry
    convolution at all. It changes what trained weights saw, so it stays off unless asked for.
    """
    if not isinstance(model, _NativeTransformerMixin):
        raise ValueError("Install Native Body before binding its geometry")
    if type(enabled) is not bool or (enabled and not isinstance(features, NativeGeometryFeatures)):
        raise ValueError("Enabled Native Body binding requires NativeGeometryFeatures")
    if type(conv_input_gate) is not bool:
        raise ValueError("conv_input_gate must be bool")
    token = model._native_body_binding.set(
        _Binding(features, enabled, valid_mask, strength, conv_input_gate))
    try:
        yield model
    finally:
        model._native_body_binding.reset(token)
