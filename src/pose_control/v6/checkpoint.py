from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn


ARCHITECTURE_VERSION = "v6.5"
DETAIL_SCHEMA_VERSION = "1"
PREPROCESSING_SCHEMA_VERSION = "1"
FACE_SCHEMA_VERSION = "1"
BRANCH_NAMES = ["geometry", "interaction", "hand", "face"]
V64_BRANCH_NAMES = ["geometry", "interaction", "detail"]

DETAIL_ONLY_PREFIXES = (
    "detail_preparer.",
    "control_interface.control_core.detail_branch.",
    "control_interface.strength_controller.detail_log_group_scale",
)

FACE_ONLY_PREFIXES = (
    "face_preparer.",
    "control_interface.face_adapter.",
    "control_interface.strength_controller.face_",
)


def _has_prefix(key: str, prefixes: tuple[str, ...]) -> bool:
    return any(key.startswith(prefix) for prefix in prefixes)


def _adapter_state(checkpoint: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    state = checkpoint.get("adapter_state_dict", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint adapter_state_dict must be a mapping")
    return state


def _preflight_state(
    model: nn.Module,
    state: Mapping[str, Any],
    expected_keys: set[str],
) -> None:
    provided_keys = set(state)
    unexpected = sorted(provided_keys - expected_keys)
    missing = sorted(expected_keys - provided_keys)
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint keys: {unexpected}")
    if missing:
        raise RuntimeError(f"checkpoint is missing keys: {missing}")
    expected_state = model.state_dict()
    for key in sorted(expected_keys):
        value = state[key]
        expected = expected_state[key]
        if not torch.is_tensor(value):
            raise RuntimeError(f"checkpoint value for {key!r} must be a tensor")
        if value.shape != expected.shape:
            raise RuntimeError(
                f"checkpoint tensor {key!r} has shape {tuple(value.shape)}; "
                f"expected {tuple(expected.shape)}"
            )
        if value.dtype != expected.dtype:
            raise RuntimeError(
                f"checkpoint tensor {key!r} has dtype {value.dtype}; "
                f"expected {expected.dtype}"
            )
        if value.layout != torch.strided or value.layout != expected.layout:
            raise RuntimeError(
                f"checkpoint tensor {key!r} has layout {value.layout}; "
                f"expected strided layout {expected.layout}"
            )


def _transactional_load(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
    *,
    strict: bool,
    after_load=None,
):
    snapshot = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    try:
        result = model.load_state_dict(state, strict=strict)
        if after_load is not None:
            after_load()
        return result
    except Exception:
        current = model.state_dict()
        with torch.no_grad():
            for key, value in snapshot.items():
                current[key].copy_(value)
        raise


def load_v63_checkpoint(
    model: nn.Module, checkpoint: Mapping[str, Any]
) -> torch.nn.modules.module._IncompatibleKeys:
    """Load V6.3 adapter weights while allowing only newly introduced detail keys."""

    if "optimizer_state_dict" in checkpoint:
        raise RuntimeError(
            "V6.3 optimizer state is incompatible with V6.4 and must not be restored"
        )
    metadata = checkpoint.get("metadata")
    if isinstance(metadata, Mapping) and metadata.get("architecture_version") in {
        "v6.4",
        "v6.5",
    }:
        version = metadata.get("architecture_version")
        raise RuntimeError(f"V6.3 loader rejects {version.upper()} metadata")
    state = _adapter_state(checkpoint)
    detail_keys = sorted(
        key
        for key in state
        if _has_prefix(key, DETAIL_ONLY_PREFIXES)
    )
    if detail_keys:
        raise RuntimeError(f"V6.3 checkpoint contains detail-only keys: {detail_keys}")
    face_keys = sorted(
        key for key in state if _has_prefix(key, FACE_ONLY_PREFIXES)
    )
    if face_keys:
        raise RuntimeError(f"V6.3 checkpoint contains face-only keys: {face_keys}")
    model_keys = set(model.state_dict())
    expected_keys = {
        key
        for key in model_keys
        if not _has_prefix(key, DETAIL_ONLY_PREFIXES + FACE_ONLY_PREFIXES)
    }
    _preflight_state(model, state, expected_keys)
    detail_branch = model.get_submodule("control_interface.control_core.detail_branch")

    def rezero_new_heads() -> None:
        with torch.no_grad():
            for head in detail_branch.zero_heads:
                nn.init.zeros_(head.weight)
                if head.bias is not None:
                    nn.init.zeros_(head.bias)
        _rezero_face_heads(model)

    result = _transactional_load(
        model,
        state,
        strict=False,
        after_load=rezero_new_heads,
    )
    return result


def v64_checkpoint_metadata() -> dict[str, object]:
    return {
        "architecture_version": "v6.4",
        "branch_names": list(V64_BRANCH_NAMES),
        "detail_schema_version": DETAIL_SCHEMA_VERSION,
        "preprocessing_schema_version": PREPROCESSING_SCHEMA_VERSION,
    }


def build_v64_checkpoint(
    model: nn.Module, **extra: Any
) -> dict[str, Any]:
    reserved = {"metadata", "adapter_state_dict"} & set(extra)
    if reserved:
        raise ValueError(f"checkpoint extra keys are reserved: {sorted(reserved)}")
    checkpoint = {
        "adapter_state_dict": {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if not _has_prefix(key, FACE_ONLY_PREFIXES)
        },
        "metadata": v64_checkpoint_metadata(),
    }
    checkpoint.update(extra)
    return checkpoint


def _rezero_face_heads(model: nn.Module) -> None:
    interface = getattr(model, "control_interface", None)
    face_adapter = getattr(interface, "face_adapter", None)
    if face_adapter is None:
        return
    with torch.no_grad():
        for head in face_adapter.zero_heads:
            nn.init.zeros_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)


def load_v64_checkpoint(
    model: nn.Module, checkpoint: Mapping[str, Any]
) -> torch.nn.modules.module._IncompatibleKeys:
    if "optimizer_state_dict" in checkpoint:
        raise RuntimeError(
            "V6.4 optimizer state is incompatible with V6.5 and must not be restored"
        )
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError("V6.4 checkpoint metadata is missing")
    expected = v64_checkpoint_metadata()
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"invalid V6.4 checkpoint metadata field {key!r}")
    state = _adapter_state(checkpoint)
    expected_keys = {
        key for key in model.state_dict() if not _has_prefix(key, FACE_ONLY_PREFIXES)
    }
    _preflight_state(model, state, expected_keys)
    return _transactional_load(
        model,
        state,
        strict=expected_keys == set(model.state_dict()),
        after_load=lambda: _rezero_face_heads(model),
    )


def v65_checkpoint_metadata(
    *, feature_cache_fingerprint: str | None = None
) -> dict[str, object]:
    return {
        "architecture_version": ARCHITECTURE_VERSION,
        "branch_names": list(BRANCH_NAMES),
        "detail_schema_version": DETAIL_SCHEMA_VERSION,
        "face_schema_version": FACE_SCHEMA_VERSION,
        "preprocessing_schema_version": PREPROCESSING_SCHEMA_VERSION,
        "feature_cache_fingerprint": feature_cache_fingerprint,
    }


def build_v65_checkpoint(
    model: nn.Module,
    *,
    feature_cache_fingerprint: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    reserved = {"metadata", "adapter_state_dict"} & set(extra)
    if reserved:
        raise ValueError(f"checkpoint extra keys are reserved: {sorted(reserved)}")
    checkpoint = {
        "adapter_state_dict": {
            key: value.detach().clone() for key, value in model.state_dict().items()
        },
        "metadata": v65_checkpoint_metadata(
            feature_cache_fingerprint=feature_cache_fingerprint
        ),
    }
    checkpoint.update(extra)
    return checkpoint


def load_v65_checkpoint(model: nn.Module, checkpoint: Mapping[str, Any]) -> None:
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError("V6.5 checkpoint metadata is missing")
    expected = v65_checkpoint_metadata(
        feature_cache_fingerprint=metadata.get("feature_cache_fingerprint")
    )
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"invalid V6.5 checkpoint metadata field {key!r}")
    state = _adapter_state(checkpoint)
    _preflight_state(model, state, set(model.state_dict()))
    _transactional_load(model, state, strict=True)


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


def freeze_for_face_training(model: nn.Module) -> tuple[nn.Parameter, ...]:
    """Freeze every old branch and expose only the independent V6.5 face path."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected = []
    for name, parameter in model.named_parameters():
        if _has_prefix(name, FACE_ONLY_PREFIXES):
            parameter.requires_grad_(True)
            selected.append(parameter)
    return tuple(selected)


__all__ = [
    "ARCHITECTURE_VERSION",
    "BRANCH_NAMES",
    "DETAIL_ONLY_PREFIXES",
    "DETAIL_SCHEMA_VERSION",
    "FACE_ONLY_PREFIXES",
    "FACE_SCHEMA_VERSION",
    "PREPROCESSING_SCHEMA_VERSION",
    "build_v64_checkpoint",
    "build_v65_checkpoint",
    "freeze_for_detail_training",
    "freeze_for_face_training",
    "load_v63_checkpoint",
    "load_v64_checkpoint",
    "load_v65_checkpoint",
    "v64_checkpoint_metadata",
    "v65_checkpoint_metadata",
]
