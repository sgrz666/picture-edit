from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn


ARCHITECTURE_VERSION = "v6.4"
DETAIL_SCHEMA_VERSION = "1"
PREPROCESSING_SCHEMA_VERSION = "1"
BRANCH_NAMES = ["geometry", "interaction", "detail"]

DETAIL_ONLY_PREFIXES = (
    "detail_preparer.",
    "control_interface.control_core.detail_branch.",
    "control_interface.strength_controller.detail_log_group_scale",
)


def _adapter_state(checkpoint: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    state = checkpoint.get("adapter_state_dict", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint adapter_state_dict must be a mapping")
    return state


def load_v63_checkpoint(
    model: nn.Module, checkpoint: Mapping[str, Any]
) -> torch.nn.modules.module._IncompatibleKeys:
    """Load V6.3 adapter weights while allowing only newly introduced detail keys."""

    if "optimizer_state_dict" in checkpoint:
        raise RuntimeError(
            "V6.3 optimizer state is incompatible with V6.4 and must not be restored"
        )
    result = model.load_state_dict(_adapter_state(checkpoint), strict=False)
    if result.unexpected_keys:
        raise RuntimeError(
            f"unexpected legacy checkpoint keys: {result.unexpected_keys}"
        )
    rejected = [
        key
        for key in result.missing_keys
        if not any(key.startswith(prefix) for prefix in DETAIL_ONLY_PREFIXES)
    ]
    if rejected:
        raise RuntimeError(f"legacy checkpoint is missing non-detail keys: {rejected}")
    detail_branch = model.get_submodule("control_interface.control_core.detail_branch")
    with torch.no_grad():
        for head in detail_branch.zero_heads:
            nn.init.zeros_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)
    return result


def v64_checkpoint_metadata() -> dict[str, object]:
    return {
        "architecture_version": ARCHITECTURE_VERSION,
        "branch_names": list(BRANCH_NAMES),
        "detail_schema_version": DETAIL_SCHEMA_VERSION,
        "preprocessing_schema_version": PREPROCESSING_SCHEMA_VERSION,
    }


def build_v64_checkpoint(
    model: nn.Module, **extra: Any
) -> dict[str, Any]:
    checkpoint = {
        "adapter_state_dict": model.state_dict(),
        "metadata": v64_checkpoint_metadata(),
    }
    checkpoint.update(extra)
    return checkpoint


def load_v64_checkpoint(model: nn.Module, checkpoint: Mapping[str, Any]) -> None:
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError("V6.4 checkpoint metadata is missing")
    expected = v64_checkpoint_metadata()
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"invalid V6.4 checkpoint metadata field {key!r}")
    model.load_state_dict(_adapter_state(checkpoint), strict=True)


def freeze_for_detail_training(model: nn.Module) -> tuple[nn.Parameter, ...]:
    """Freeze V6.3 and expose only V6.4 detail modules and detail group gates."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable_prefixes = (
        "detail_preparer.",
        "control_interface.control_core.detail_branch.",
        "control_interface.strength_controller.detail_log_group_scale",
    )
    selected = []
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in trainable_prefixes):
            parameter.requires_grad_(True)
            selected.append(parameter)
    return tuple(selected)


__all__ = [
    "ARCHITECTURE_VERSION",
    "BRANCH_NAMES",
    "DETAIL_ONLY_PREFIXES",
    "DETAIL_SCHEMA_VERSION",
    "PREPROCESSING_SCHEMA_VERSION",
    "build_v64_checkpoint",
    "freeze_for_detail_training",
    "load_v63_checkpoint",
    "load_v64_checkpoint",
    "v64_checkpoint_metadata",
]
