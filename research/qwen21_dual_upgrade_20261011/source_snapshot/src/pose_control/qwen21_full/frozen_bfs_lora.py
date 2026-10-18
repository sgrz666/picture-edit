"""Load an unfused BFS prior while keeping the Qwen Transformer frozen.

Only Transformer parameters are frozen. This module does not install a no-grad
forward wrapper, so gradients can still pass through Qwen to an external Adapter.
Use a fresh pipeline for each experiment arm; existing LoRAs are rejected.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import torch
from torch import nn


_SOURCE_KEY = re.compile(r"diffusion_model\.(transformer_blocks\.\d+\..+)\.lora_([AB])\.weight")
_CONVERTED_KEY = re.compile(r"transformer\.(.+)\.lora_([AB])\.weight")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_digest(tensors) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors, key=lambda item: item[0]):
        if tensor.device.type == "meta":
            raise RuntimeError(f"Cannot fingerprint an unmaterialized tensor: {name}")
        value = tensor.detach().contiguous().cpu()
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        # The byte view also supports BF16, which NumPy cannot represent directly.
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _is_lora_parameter(name: str) -> bool:
    return any(part in {"lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B", "lora_magnitude_vector"}
               for part in name.split("."))


def _freeze_report(transformer: nn.Module) -> dict:
    parameters = list(transformer.named_parameters())
    trainable = [name for name, value in parameters if value.requires_grad]
    return {
        "all_parameters_frozen": not trainable,
        "trainable_parameter_names": trainable,
        "parameter_tensors": len(parameters),
        "parameter_elements": sum(value.numel() for _, value in parameters),
        "lora_parameter_tensors": sum(_is_lora_parameter(name) for name, _ in parameters),
        "lora_parameter_elements": sum(value.numel() for name, value in parameters if _is_lora_parameter(name)),
        "parameter_names_sha256": hashlib.sha256("\n".join(sorted(name for name, _ in parameters)).encode()).hexdigest(),
    }


def frozen_transformer_digest(transformer: nn.Module, *, lora_only: bool = True) -> str:
    """Fingerprint frozen parameters, with inexpensive LoRA-only hashing by default.

    All Transformer parameters must have ``requires_grad=False`` in either mode.
    ``lora_only=False`` hashes the complete base and LoRA parameter values, which
    involves copying the entire model to CPU one parameter at a time. Names,
    shapes and dtypes are included; devices and gradient buffers are excluded.
    An unloaded Qwen prior has the SHA256 of an empty byte stream in LoRA mode.
    """
    if not _freeze_report(transformer)["all_parameters_frozen"]:
        raise RuntimeError("Transformer must be fully frozen before fingerprinting")
    return _tensor_digest((name, value) for name, value in transformer.named_parameters()
                          if not lora_only or _is_lora_parameter(name))


def _require_clean_transformer(transformer: nn.Module) -> None:
    if getattr(transformer, "peft_config", None):
        raise RuntimeError("BFS loading requires a fresh Transformer with no existing adapters")
    for name, module in transformer.named_modules():
        if getattr(module, "merged_adapters", None):
            raise RuntimeError(f"Transformer already contains fused adapters: {name}")
        if any(len(getattr(module, part, {})) for part in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")):
            raise RuntimeError(f"Transformer already contains a LoRA: {name}")


def _base_weight(transformer: nn.Module, name: str) -> torch.Tensor:
    try:
        module = transformer.get_submodule(name)
    except (AttributeError, KeyError) as error:
        raise ValueError(f"Unknown BFS target module: {name}") from error
    base = module.get_base_layer() if hasattr(module, "get_base_layer") else module
    weight = getattr(base, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError(f"BFS target must have a 2D Linear weight: {name}")
    return weight


def _convert_state(path: Path, transformer: nn.Module) -> tuple[dict, int]:
    # Keep this dependency lazy: a plain frozen Qwen arm needs only torch.
    from safetensors.torch import load_file

    source = load_file(str(path), device="cpu")
    if not source:
        raise ValueError("BFS checkpoint contains no tensors")
    converted = {}
    for key, value in source.items():
        match = _SOURCE_KEY.fullmatch(key)
        if match is None:
            raise ValueError(f"Unknown BFS tensor mapping: {key}")
        name, part = match.groups()
        if value.ndim != 2 or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"Invalid BFS source tensor: {key}")
        if name.endswith(".img_mlp.gate_up"):
            prefix = name.removesuffix("gate_up")
            gate = _base_weight(transformer, prefix + "gate_layer")
            up = _base_weight(transformer, prefix + "proj")
            if gate.shape != up.shape:
                raise ValueError(f"Incompatible SwiGLU gate/up target shapes: {name}")
            if part == "A":
                pieces = ((prefix + "gate_layer", value), (prefix + "proj", value))
            else:
                width = gate.shape[0]
                if value.shape[0] != width + up.shape[0]:
                    raise ValueError(f"Invalid fused gate-first BFS tensor shape: {key}")
                pieces = ((prefix + "gate_layer", value[:width]), (prefix + "proj", value[width:]))
        else:
            pieces = ((name, value),)
        for target, tensor in pieces:
            mapped = f"transformer.{target}.lora_{part}.weight"
            if mapped in converted:
                raise ValueError(f"Duplicate BFS mapping: {mapped}")
            converted[mapped] = tensor

    targets = {match.group(1) for key in converted if (match := _CONVERTED_KEY.fullmatch(key))}
    for name in sorted(targets):
        weight = _base_weight(transformer, name)
        a = converted.get(f"transformer.{name}.lora_A.weight")
        b = converted.get(f"transformer.{name}.lora_B.weight")
        if a is None or b is None:
            raise ValueError(f"Missing BFS A/B tensor pair: {name}")
        if a.shape[0] < 1 or a.shape[1] != weight.shape[1] or b.shape != (weight.shape[0], a.shape[0]):
            raise ValueError(f"BFS rank/shape does not match target {name}: A={tuple(a.shape)}, B={tuple(b.shape)}, base={tuple(weight.shape)}")
    return converted, len(source)


def _loaded_inventory(transformer: nn.Module, adapter_name: str) -> set[str]:
    keys = set()
    for name, module in transformer.named_modules():
        for part in ("A", "B"):
            collection = getattr(module, "lora_" + part, {})
            if adapter_name in collection:
                keys.add(f"transformer.{name}.lora_{part}.weight")
    return keys


def _verify_loaded(transformer: nn.Module, state: dict, adapter_name: str, strength: float) -> dict:
    if _loaded_inventory(transformer, adapter_name) != set(state):
        raise RuntimeError("Loaded BFS tensor inventory differs from converted checkpoint")
    expected_parameters = {
        f"{name}.lora_{part}.{adapter_name}.weight"
        for key in state for name, part in [_CONVERTED_KEY.fullmatch(key).groups()]
    }
    actual_parameters = {name for name, _ in transformer.named_parameters() if _is_lora_parameter(name)}
    if actual_parameters != expected_parameters:
        raise RuntimeError("Loaded BFS contains unexpected LoRA parameters or adapters")
    tensors, modules = [], {}
    base_cast_count = 0
    for key, expected in sorted(state.items()):
        name, part = _CONVERTED_KEY.fullmatch(key).groups()
        module = transformer.get_submodule(name)
        live = getattr(module, "lora_" + part)[adapter_name].weight.detach().cpu()
        if live.shape != expected.shape or not torch.isfinite(live).all():
            raise RuntimeError(f"Invalid loaded BFS tensor: {key}")
        conversion = "direct_dtype"
        if not torch.equal(live, expected.to(live.dtype)):
            # Diffusers/PEFT may initialize in BF16, then promote LoRA parameters
            # to FP32. Accept exactly that rounding, never a numeric tolerance.
            if not torch.equal(live, expected.to(_base_weight(transformer, name).dtype).to(live.dtype)):
                raise RuntimeError(f"Loaded BFS tensor differs from checkpoint: {key}")
            conversion = "via_base_dtype"
            base_cast_count += 1
        tensors.append({"key": key, "shape": list(live.shape), "source_dtype": str(expected.dtype),
                        "loaded_dtype": str(live.dtype), "value_check": conversion})
        if name in modules:
            continue
        if getattr(module, "merged_adapters", None) or getattr(module, "disable_adapters", False):
            raise RuntimeError(f"BFS adapter is merged or disabled: {name}")
        active = getattr(module, "active_adapters", None)
        if active is None or list(active) != [adapter_name]:
            raise RuntimeError(f"BFS adapter is not the only active adapter: {name}")
        scale = float(module.scaling[adapter_name])
        if not math.isfinite(scale) or not math.isclose(scale, strength, rel_tol=1e-7, abs_tol=1e-12):
            raise RuntimeError(f"Unexpected effective BFS scale for {name}: {scale} != {strength}")
        modules[name] = {"rank": int(state[f"transformer.{name}.lora_A.weight"].shape[0]),
                         "base_shape": list(_base_weight(transformer, name).shape), "effective_scale": scale}
    return {"tensors_checked": len(tensors), "modules_checked": len(modules), "tensor_checks": tensors,
            "module_checks": modules, "module_scales": sorted({row["effective_scale"] for row in modules.values()}),
            "all_values_match_after_dtype_conversion": True, "tensors_cast_through_base_dtype": base_cast_count}


def load_frozen_bfs(pipe, path: str | Path | None, adapter_name: str = "bfs_joint", strength: float = 1.0) -> dict:
    """Validate, load and freeze one BFS prior; return a JSON-serializable report.

    ``path=None`` freezes plain Qwen and does not load a LoRA. The source must use
    the BFS ComfyUI ``diffusion_model.*.lora_[AB].weight`` format; fused SwiGLU
    projections are split gate-first using the actual target module dimensions.
    Effective PEFT scales must match ``strength`` (the pilot uses 1.0).

    A failure after pipeline loading can leave partially attached adapters. The
    Transformer is still frozen; discard that pipeline and create a fresh one.
    This function never fuses or merges adapters and never freezes pipeline
    neighbors or external Body/Face Adapter parameters.
    """
    strength = float(strength)
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("BFS strength must be finite and nonnegative")
    if not isinstance(adapter_name, str) or not adapter_name or "." in adapter_name:
        raise ValueError("BFS adapter_name must be a nonempty name without dots")
    transformer = pipe.transformer
    _require_clean_transformer(transformer)
    report = {
        "schema_version": 1, "variant": "qwen" if path is None else "bfs", "path": None,
        "file_sha256": None, "adapter_name": None if path is None else adapter_name,
        "gate_first": None if path is None else True, "strength": 0.0 if path is None else strength,
        "source_tensors": 0, "tensors_checked": 0, "modules_checked": 0,
        "tensor_checks": [], "module_checks": {}, "module_scales": [],
        "all_values_match_after_dtype_conversion": True, "tensors_cast_through_base_dtype": 0,
        "converted_lora_sha256": None,
    }
    if path is not None:
        resolved = Path(path).expanduser().resolve(strict=True)
        state, source_count = _convert_state(resolved, transformer)
        report.update(path=str(resolved), file_sha256=_file_sha256(resolved), source_tensors=source_count,
                      converted_lora_sha256=_tensor_digest(state.items()))
        try:
            # Preserve the source tensors for verification if a loader mutates
            # its input dictionary or tensors in place.
            pipe.load_lora_weights({key: value.clone() for key, value in state.items()}, adapter_name=adapter_name)
            pipe.set_adapters([adapter_name], adapter_weights=[strength])
            report.update(_verify_loaded(transformer, state, adapter_name, strength))
        finally:
            transformer.eval().requires_grad_(False)
    else:
        transformer.eval().requires_grad_(False)
    report["freeze"] = _freeze_report(transformer)
    report["loaded_lora_sha256"] = frozen_transformer_digest(transformer)
    report["loaded_lora_digest_scope"] = "lora_parameters_only"
    return report
