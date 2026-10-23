"""Opt-in attention-level exclusion of canvas padding tokens (remediation plan §5).

Why this exists: `transformer_qwenimage21` builds `joint_key_valid` as all-True for every image
position and only overwrites the text slots, so a padded portrait's margin tokens are attendable
keys. Measured on the real 20B model (research/canvas_native_portrait_acceptance_20261022/):
perturbing only the 41 fully-padding tokens of a 26x16 grid moved *valid* token outputs by 501% of
their signal, and merely changing the margin fill value moved them by 404%. Excluding those keys
through this module drove the perturbation shift to exactly 0.000000.

That same measurement is the warning: excluding padding keys changes valid-token outputs by ~5x
signal on its own, so this cannot be switched on for a checkpoint that was trained with padding
attending. It is opt-in, per-call, and refuses to run next to a `BlockMask` path it does not own.
"""
from __future__ import annotations

import weakref
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import torch
from torch import nn

SCHEMA = "qwen21_full_padding_attention_v1"
STATE_KEY = "_padding_attention_state"
OWNER_KEY = "_padding_attention_owner"


class _PaddingBlockMixin:
    """Drop fully-padding target tokens from `key_valid` before the block attends."""

    def forward(self, *args, **kwargs):
        owner_ref = getattr(self, OWNER_KEY, None)
        owner = owner_ref() if owner_ref is not None else None
        binding = None if owner is None else getattr(owner, STATE_KEY).get()
        if binding is None:
            return super().forward(*args, **kwargs)
        padding, tokens, counters = binding
        if kwargs.get("attention_mask") is not None:
            raise RuntimeError(
                "a BlockMask/segment attention path is active; padding exclusion here only edits "
                "key_valid, which that path would ignore")
        key_valid = kwargs.get("key_valid")
        if key_valid is None:
            raise RuntimeError("the transformer did not pass key_valid; this hook has nothing to edit")
        if key_valid.shape[-1] < tokens:
            raise ValueError(f"key_valid covers {key_valid.shape[-1]} positions but the target grid "
                             f"needs {tokens}; the target tokens must sit at the end of the sequence")
        if key_valid.shape[0] != 1:
            raise ValueError(f"padding exclusion supports batch size 1, got {key_valid.shape[0]}")
        counters["calls"] += 1
        masked = key_valid.clone()
        tail = masked[:, -tokens:]
        tail[0, padding] = False
        kwargs = dict(kwargs)
        kwargs["key_valid"] = masked
        counters["tokens_dropped"] = int(padding.sum())
        return super().forward(*args, **kwargs)


def install_padding_key_mask(model: nn.Module) -> dict[str, Any]:
    """Wrap every transformer block. Idempotent-guarded and parameter-preserving."""
    if getattr(model, STATE_KEY, None) is not None:
        raise ValueError("padding key masking is already installed on this model")
    blocks = getattr(model, "transformer_blocks", None)
    if blocks is None:
        raise ValueError("expected a transformer exposing transformer_blocks")
    setattr(model, STATE_KEY, ContextVar(f"padding_attention_{id(model)}", default=None))
    counters = {"calls": 0, "tokens_dropped": 0}
    setattr(model, "_padding_attention_counters", counters)
    for block in blocks:
        setattr(block, OWNER_KEY, weakref.ref(model))
        block.__class__ = type(f"Padded{type(block).__name__}", (_PaddingBlockMixin, type(block)), {})
    return {"schema": SCHEMA, "blocks_wrapped": len(blocks), "default": "disabled"}


def padding_attention_active(model: nn.Module) -> bool:
    state = getattr(model, STATE_KEY, None)
    return state is not None and state.get() is not None


@contextmanager
def padding_key_masked(model: nn.Module, canvas, *, allow_semantic_change: bool = False):
    """Exclude `canvas`'s fully-padding target tokens from attention for this scope.

    `allow_semantic_change` must be True because the outputs differ materially from a model that
    was trained with padding tokens attendable; the flag exists so that difference is a decision
    recorded at the call site rather than a default someone inherits.
    """
    state = getattr(model, STATE_KEY, None)
    if state is None:
        raise RuntimeError("call install_padding_key_mask(model) first")
    if not allow_semantic_change:
        raise ValueError(
            "excluding padding tokens changes the function the backbone computes by an amount "
            "comparable to the signal itself; pass allow_semantic_change=True only for a run that "
            "trains or evaluates consistently under it")
    padding = canvas.padding_token_mask().reshape(-1).to(torch.bool)
    tokens = int(padding.numel())
    if state.get() is not None:
        raise RuntimeError("padding_key_masked is not reentrant; nest one canvas per call")
    counters = getattr(model, "_padding_attention_counters", {"calls": 0, "tokens_dropped": 0})
    token = state.set((padding, tokens, counters))
    try:
        yield {"padding_tokens": int(padding.sum()), "target_tokens": tokens,
               "valid_tokens": tokens - int(padding.sum()), "token_hw": list(canvas.token_hw)}
    finally:
        state.reset(token)


__all__ = ["SCHEMA", "install_padding_key_mask", "padding_key_masked", "padding_attention_active"]
