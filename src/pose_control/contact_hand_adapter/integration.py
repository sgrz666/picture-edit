"""An independent hand K/V branch inside the frozen Qwen attention residual."""

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field, replace
from functools import partial, wraps
import inspect
import math

import torch
from torch.nn import functional as F

from src.data.canvas_contract import TOKEN_SLOT_MULTIPLE

from .conditions import verify_appearance_sources
from .geometry import pixel_geometry
from .model import ContactHandAdapter, HandMemory
from .prior import HandPrior, prior_attention_delta


@dataclass(frozen=True)
class Binding:
    condition: object
    strength: float
    memory: HandMemory | None = None
    prior_strength: float = 0.
    prior_support: torch.Tensor | None = None


@dataclass
class Execution:
    adapter: ContactHandAdapter
    memory: HandMemory | None
    state: torch.Tensor | None
    roi_positions: torch.Tensor | None
    anchor_positions: torch.Tensor | None
    sigma: torch.Tensor
    strength: float
    prior: HandPrior | None
    prior_positions: torch.Tensor | None
    prior_gain: torch.Tensor | None


@dataclass
class QueryCapture:
    attention: object
    roi: torch.Tensor | None
    prior_roi: torch.Tensor | None
    query: torch.Tensor | None = None
    native_values: torch.Tensor | None = None
    prior_projected: dict = field(default_factory=dict)
    handle: Token | None = None


_QUERY_CAPTURE = ContextVar("contact_hand_native_query", default=None)


def _capture_query(norm, _inputs, output):
    capture = _QUERY_CAPTURE.get()
    if capture is not None and capture.roi is not None and capture.attention.norm_q is norm:
        capture.query = output[0].index_select(0, capture.roi)


def _capture_native_values(module, inputs):
    capture = _QUERY_CAPTURE.get()
    if capture is not None and capture.roi is not None and capture.attention.to_out[0] is module:
        capture.native_values = inputs[0][0].index_select(0, capture.roi)


def _capture_prior_projection(_module, _args, output, *, name):
    capture = _QUERY_CAPTURE.get()
    if capture is not None and capture.prior_roi is not None:
        capture.prior_projected[name] = output[:, capture.prior_roi] if name == "query" else output


def _begin_attention(attention, _args, *, execution, layer, prior_layer):
    current = execution.get()
    if current is not None:
        roi = current.roi_positions if layer is not None else None
        prior_roi = current.prior_positions if prior_layer is not None else None
        if roi is not None or prior_roi is not None:
            capture = QueryCapture(attention, roi, prior_roi)
            capture.handle = _QUERY_CAPTURE.set(capture)


def _end_attention(attention, args, kwargs, native, *, execution, layer, prior_layer):
    capture = _QUERY_CAPTURE.get()
    if capture is None or capture.attention is not attention:
        return None
    _QUERY_CAPTURE.reset(capture.handle)
    if native is None:  # Exception or early stop during non-reentrant recomputation.
        return None
    current = execution.get()
    hidden_states = args[0] if args else kwargs["hidden_states"]
    rotary = kwargs.get("rotary_emb")
    if capture.prior_roi is not None:
        if len(capture.prior_projected) != 3:
            raise RuntimeError("The Qwen attention processor did not call its native Q/K/V projections")
        delta = prior_attention_delta(attention, current.prior.blocks[prior_layer], hidden_states, capture.prior_roi,
                                      current.prior_gain, capture.prior_projected, rotary, kwargs.get("key_valid"))
        native = native.index_add(1, capture.prior_roi, delta.to(native.dtype))
    if capture.roi is None:
        return native
    if capture.query is None:
        raise RuntimeError("The Qwen attention processor did not call its native query normalization")
    query = capture.query.to(native.dtype)
    hand_rope = None
    if rotary is not None:
        from diffusers.models.transformers.transformer_qwenimage21 import apply_rotary_emb_qwen

        query = apply_rotary_emb_qwen(query[None], rotary[current.roi_positions], use_real=False)[0]

        def hand_rope(keys):
            return apply_rotary_emb_qwen(keys[None], rotary[current.anchor_positions], use_real=False)[0]

    with torch.autocast(device_type=native.device.type, enabled=False):
        update, current.state = current.adapter.update_attention(
            current.state, current.memory, layer, current.sigma,
            hidden_states[0, current.roi_positions].float(), query.float(), hand_rope,
        )
        lora = current.adapter.native_lora
        if lora:
            if capture.native_values is None:
                raise RuntimeError("Native Qwen attention output projection was not observed")
            support = current.memory.support[:, current.memory.roi].sum(0)[:, None]
            update = update + lora[layer](capture.native_values.float()) * support
        update = update * current.strength
    return native.index_add(1, current.roi_positions, update[None].to(native.dtype))


def _target_positions(model, arguments, token_hw):
    image_mask = arguments["img_mask"]
    shapes = arguments["img_shapes"]
    latents = arguments["hidden_states"]
    text = arguments["encoder_hidden_states"]
    if (latents.ndim != 3 or text.ndim != 3 or latents.shape[0] != 1 or text.shape[0] != 1
            or image_mask.dtype != torch.bool or image_mask.ndim != 2 or image_mask.shape[0] != 1
            or len(shapes) != 1 or not shapes[0]):
        raise ValueError("Contact Hand Adapter requires one causal Qwen sample")
    shapes = shapes[0]
    if any(len(shape) != 3 or shape[0] != 1 or any(type(x) is not int or x <= 0 for x in shape)
           for shape in shapes):
        raise ValueError("Expected positive Qwen image grids")
    lengths = [math.prod(shape) for shape in shapes]
    target_slots = lengths[-1] // TOKEN_SLOT_MULTIPLE
    if (tuple(shapes[-1][1:]) != token_hw or sum(lengths) != latents.shape[1]
            or any(length % TOKEN_SLOT_MULTIPLE for length in lengths)
            or image_mask.shape[1] != text.shape[1] + target_slots
            or not image_mask[0, -target_slots:].all()
            or image_mask[0, :text.shape[1]].sum().item() * TOKEN_SLOT_MULTIPLE != sum(lengths[:-1])):
        raise ValueError("Hand canvas and Qwen image slot layout disagree")
    repeats = torch.where(image_mask[0], TOKEN_SLOT_MULTIPLE, 1)
    mask = torch.repeat_interleave(image_mask[0], repeats)
    _, target = model.build_token_metadata(mask, shapes)
    return torch.where(target)[0]


def _adapter_positions(memory, target_positions):
    height, width = memory.token_hw
    anchors = memory.xy.floor().long()
    anchor = anchors[:, 1].clamp(0, height - 1) * width + anchors[:, 0].clamp(0, width - 1)
    return target_positions[memory.roi], target_positions[anchor]


def _prior_support(condition):
    condition.validate()
    _, support = pixel_geometry(condition)
    return F.avg_pool2d(support.amax(0, keepdim=True), condition.canvas.stride).flatten()


class _TransformerIntegration:
    def enable_gradient_checkpointing(self, gradient_checkpointing_func=None):
        super().enable_gradient_checkpointing(gradient_checkpointing_func)
        self._install_hand_checkpoint_bridge()

    def _install_hand_checkpoint_bridge(self):
        original = self._gradient_checkpointing_func
        if getattr(original, "_contact_hand_bridge", False):
            return
        execution = self._contact_hand_execution
        layers = self._contact_hand_blocks

        @wraps(original)
        def checkpoint_block(block, *args):
            parent = execution.get()
            if parent is None or block not in layers:
                return original(block, *args)
            frozen = replace(parent)

            def run(state, tokens, spatial, *block_args):
                memory = None if frozen.memory is None else replace(frozen.memory, tokens=tokens, spatial=spatial)
                local = replace(frozen, state=state, memory=memory)
                handle = execution.set(local)
                try:
                    hidden = block(*block_args)
                    return hidden, local.state
                finally:
                    execution.reset(handle)

            memory = parent.memory
            tokens, spatial = (None, None) if memory is None else (memory.tokens, memory.spatial)
            hidden, parent.state = original(run, parent.state, tokens, spatial, *args)
            return hidden

        checkpoint_block._contact_hand_bridge = True
        self._gradient_checkpointing_func = checkpoint_block

    def forward(self, *args, **kwargs):
        binding = self._contact_hand_binding.get()
        if binding is None or not (binding.strength or binding.prior_strength):
            return super().forward(*args, **kwargs)
        if self._contact_hand_execution.get() is not None:
            raise RuntimeError("Reentrant Contact Hand Adapter forward is unsupported")
        bound = self._contact_hand_signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        if values.get("kv_cache") is not None or values.get("kv_cache_mode") is not None:
            raise ValueError("Contact Hand Adapter requires use_kv_cache=False")
        sigma = values["timestep"]
        if sigma.shape != (1,) or not torch.isfinite(sigma).all() or not ((sigma >= 0) & (sigma <= 1)).all():
            raise ValueError("Hand Adapter requires one normalized sigma in [0,1]")
        adapter = self.contact_hand_adapter
        device = values["hidden_states"].device
        memory = binding.memory
        prior_support = binding.prior_support
        prior_gain = None
        prior_scale = binding.prior_strength * (1 - sigma.detach().float().reshape(()))
        prior_active = binding.prior_strength and bool(prior_scale > 0)
        with torch.autocast(device_type=device.type, enabled=False):
            if (binding.strength and memory is None) or (prior_active and prior_support is None):
                condition = binding.condition.to(device)
                if binding.strength and memory is None:
                    memory = adapter.prepare(condition)
                if prior_active and prior_support is None:
                    prior_support = _prior_support(condition)
        roi, anchors, prior_positions = None, None, None
        if (memory is not None and memory.roi.numel()) or (prior_active and prior_support.any()):
            target = _target_positions(self, values, binding.condition.canvas.token_hw)
            if memory is not None and memory.roi.numel():
                if memory.tokens.device != device:
                    raise ValueError("Qwen and bound hand geometry must share a device")
                roi, anchors = _adapter_positions(memory, target)
            if prior_active and prior_support.any():
                if prior_support.device != device or next(self.contact_hand_prior.parameters()).device != device:
                    raise ValueError("Qwen and the bound hand prior must share a device")
                indices = torch.where(prior_support > 0)[0]
                prior_positions, prior_gain = target[indices], prior_support[indices] * prior_scale
        if roi is None and prior_positions is None:
            return super().forward(*args, **kwargs)
        if self.gradient_checkpointing and torch.is_grad_enabled():
            self._install_hand_checkpoint_bridge()
        active = Execution(adapter, memory, None if memory is None else memory.tokens, roi, anchors, sigma,
                           binding.strength, self.contact_hand_prior, prior_positions, prior_gain)
        handle = self._contact_hand_execution.set(active)
        try:
            return super().forward(*args, **kwargs)
        finally:
            self._contact_hand_execution.reset(handle)


def install_hand_adapter(model, adapter, *, hand_prior=None):
    """Install once on a clean causal Qwen transformer and freeze its original parameters."""
    from diffusers.models.transformers.transformer_qwenimage21 import (
        QwenImage21AttnProcessor, QwenImage21FlexAttnProcessor, QwenImage21Transformer2DModel,
    )

    if not isinstance(model, QwenImage21Transformer2DModel) or not isinstance(adapter, ContactHandAdapter):
        raise TypeError("Contact Hand Adapter requires the QwenImage21Transformer2DModel")
    if any(hasattr(model, attr) for attr in ("contact_hand_adapter", "native_body", "control_blocks")):
        raise ValueError("Install on a clean Qwen transformer once")
    config = adapter.config
    if (not model.config.causal_condition or config.base_width != model.inner_dim
            or config.base_heads != model.config.num_attention_heads or config.layers[-1] >= len(model.transformer_blocks)):
        raise ValueError("Adapter dimensions or layer indices disagree with the causal Qwen backbone")
    if hand_prior is not None and (not isinstance(hand_prior, HandPrior) or hand_prior.width != model.inner_dim
                                   or len(hand_prior.blocks) != len(model.transformer_blocks)):
        raise ValueError("Hand prior projections must match all causal Qwen blocks")
    adapter_indices = {layer: index for index, layer in enumerate(config.layers)}
    blocks = {block: (adapter_indices.get(layer), layer if hand_prior is not None else None)
              for layer, block in enumerate(model.transformer_blocks)
              if layer in adapter_indices or hand_prior is not None}
    if any(type(block.attn.processor) not in (QwenImage21AttnProcessor, QwenImage21FlexAttnProcessor) for block in blocks):
        raise ValueError("Expected a supported native Qwen attention processor with observable query normalization")
    model.requires_grad_(False)
    model.add_module("contact_hand_adapter", adapter)
    model.add_module("contact_hand_prior", None if hand_prior is None else hand_prior.requires_grad_(False).eval())
    model._contact_hand_signature = inspect.signature(type(model).forward)
    model._contact_hand_binding = ContextVar("contact_hand_binding", default=None)
    model._contact_hand_execution = ContextVar("contact_hand_execution", default=None)
    model._contact_hand_blocks = frozenset(blocks)
    model.__class__ = type("HandControlled" + type(model).__name__, (_TransformerIntegration, type(model)), {})
    for block, (index, prior_index) in blocks.items():
        attention = block.attn
        if index is not None:
            attention.norm_q.register_forward_hook(_capture_query)
            if adapter.native_lora:
                attention.to_out[0].register_forward_pre_hook(_capture_native_values)
        if prior_index is not None:
            for name, projection in (("query", attention.to_q), ("key", attention.to_k), ("value", attention.to_v)):
                projection.register_forward_hook(partial(_capture_prior_projection, name=name))
        arguments = dict(execution=model._contact_hand_execution, layer=index, prior_layer=prior_index)
        attention.register_forward_pre_hook(partial(_begin_attention, **arguments))
        attention.register_forward_hook(
            partial(_end_attention, **arguments),
            with_kwargs=True, always_call=True,
        )
    if model.gradient_checkpointing:
        model._install_hand_checkpoint_bridge()
    return model


@contextmanager
def bind_hand_condition(model, condition, *, strength=1.0, source_hashes=None, prior_strength=0.):
    if not isinstance(model, _TransformerIntegration) or model._contact_hand_binding.get() is not None:
        raise ValueError("Expected a clean installed Contact Hand Adapter binding")
    if not isinstance(strength, (int, float)) or not math.isfinite(strength) or strength < 0:
        raise ValueError("Hand strength must be finite and nonnegative")
    if not isinstance(prior_strength, (int, float)) or not math.isfinite(prior_strength) or prior_strength < 0:
        raise ValueError("Hand prior strength must be finite and nonnegative")
    if prior_strength and model.contact_hand_prior is None:
        raise ValueError("Positive hand prior strength requires installed SHA-pinned prior weights")
    memory, prior_support = None, None
    if strength:
        verify_appearance_sources(condition, source_hashes)
    if (strength or prior_strength) and not torch.is_grad_enabled():
        adapter = model.contact_hand_adapter
        device = next(adapter.parameters()).device
        placed = condition.to(device)
        with torch.autocast(device_type=device.type, enabled=False):
            if strength:
                memory = adapter.prepare(placed)
            if prior_strength:
                prior_support = _prior_support(placed)
    handle = model._contact_hand_binding.set(Binding(condition, strength, memory, prior_strength, prior_support))
    try:
        yield model
    finally:
        model._contact_hand_binding.reset(handle)
