"""更新边界上的完整训练断点；新目录独占创建，已有产物不覆盖。"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .weights import load_lora_state_dict, lora_state_dict, validate_lora_state_dict

FORMAT_VERSION = "qwen21_union_training_checkpoint_v1"
REQUIRED_METADATA = frozenset({"scope", "base_fingerprint", "union_fingerprint", "lock_sha256",
    "preprocessing_version", "rank", "alpha", "lora_targets", "dataset_order", "trajectory_sha256"})


@dataclass(frozen=True)
class TrainingCursor:
    optimizer_step: int = 0
    microstep: int = 0
    next_sample_position: int = 0
    loss_log_cursor: int = 0
    gradient_log_cursor: int = 0
    accumulation_steps: int = 3

    def validate(self):
        values = asdict(self)
        if any(not isinstance(value, int) or value < 0 for value in values.values()):
            raise ValueError("Training cursor fields must be nonnegative integers")
        if self.accumulation_steps < 1 or self.microstep != self.optimizer_step * self.accumulation_steps:
            raise ValueError("Checkpoint must be saved at an optimizer update boundary")
        if self.next_sample_position != self.microstep:
            raise ValueError("Sample position differs from completed microsteps")


def capture_rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [state.clone().cpu() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available() else [],
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0}


def restore_rng_state(state):
    expected_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if state["cuda_device_count"] != expected_cuda:
        raise ValueError("CUDA RNG device count differs from checkpoint")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if expected_cuda:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _canonical_metadata(metadata):
    if not isinstance(metadata, dict) or not REQUIRED_METADATA <= metadata.keys():
        missing = REQUIRED_METADATA - metadata.keys() if isinstance(metadata, dict) else REQUIRED_METADATA
        raise ValueError(f"Checkpoint metadata is incomplete: {sorted(missing)}")
    if metadata["scope"] not in ("joint", "target"):
        raise ValueError("Checkpoint metadata scope must be joint or target")
    if metadata["rank"] != 16 or metadata["alpha"] != 16:
        raise ValueError("Checkpoint metadata requires rank=16 alpha=16")
    if metadata["dataset_order"] != ["iper", "tiktok", "hi4d"]:
        raise ValueError("Checkpoint metadata dataset_order differs from locked protocol")
    if not metadata["lora_targets"] or len(metadata["lora_targets"]) != 130:
        raise ValueError("Checkpoint metadata requires the exact 130 LoRA targets")
    for key in ("base_fingerprint", "union_fingerprint", "lock_sha256", "preprocessing_version", "trajectory_sha256"):
        if not isinstance(metadata[key], str) or not metadata[key]:
            raise ValueError(f"Checkpoint metadata {key} must be nonempty")
    # JSON roundtrip prevents tuple/list ambiguity and rejects unserializable signatures.
    return json.loads(json.dumps(metadata, sort_keys=True, allow_nan=False))


def _validate_metadata(stored, expected):
    left, right = _canonical_metadata(stored), _canonical_metadata(expected)
    if left.keys() != right.keys():
        raise ValueError(f"Checkpoint metadata fields differ: {sorted(left.keys() ^ right.keys())}")
    for key in left:
        if left[key] != right[key]:
            raise ValueError(f"Checkpoint metadata differs: {key}")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_model_metadata(model, metadata):
    modules = {name: module for name, module in model.named_modules()
               if hasattr(module, "lora_A") and hasattr(module, "lora_B")}
    if tuple(modules) != tuple(metadata["lora_targets"]):
        # named_modules traverses structural order, so compare the full exact set.
        if set(modules) != set(metadata["lora_targets"]):
            raise ValueError("Checkpoint model lora_targets differ from metadata")
    for module in modules.values():
        if module.rank != metadata["rank"] or module.alpha != metadata["alpha"]:
            raise ValueError("Checkpoint model LoRA rank/alpha differs from metadata")
        if not module.enabled:
            raise ValueError("Training checkpoints require enabled LoRA modules")
    scope = getattr(model, "injection_scope", metadata["scope"])
    if scope != metadata["scope"]:
        raise ValueError("Checkpoint model scope differs from metadata")


def save_checkpoint(path, model, optimizer, lr_scheduler, *, metadata, cursor: TrainingCursor) -> Path:
    path = Path(path)
    metadata = _canonical_metadata(metadata)
    cursor.validate()
    _validate_model_metadata(model, metadata)
    if lr_scheduler is None or optimizer is None:
        raise ValueError("Full checkpoint requires optimizer and LR scheduler")
    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
           for parameter in model.parameters() if parameter.requires_grad):
        raise ValueError("Cannot save nonfinite training gradients")
    state = {"format_version": FORMAT_VERSION, "metadata": metadata, "cursor": asdict(cursor),
             "lora": lora_state_dict(model), "optimizer": optimizer.state_dict(),
             "lr_scheduler": lr_scheduler.state_dict(), "rng": capture_rng_state()}
    path.mkdir(parents=True, exist_ok=False)
    # Completion manifest is written last. Interrupted directories remain inspectable.
    with (path / "state.pt").open("xb") as stream:
        torch.save(state, stream)
    manifest = {"format_version": FORMAT_VERSION, "metadata": metadata, "cursor": asdict(cursor),
                "state_sha256": _sha256(path / "state.pt")}
    with (path / "checkpoint.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return path


def load_checkpoint(path, model, optimizer, lr_scheduler, *, expected_metadata) -> TrainingCursor:
    path = Path(path)
    manifest = json.loads((path / "checkpoint.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("Checkpoint format version differs")
    _validate_metadata(manifest["metadata"], expected_metadata)
    _validate_model_metadata(model, manifest["metadata"])
    if _sha256(path / "state.pt") != manifest["state_sha256"]:
        raise ValueError("Checkpoint state checksum differs")
    # This is a locally created full optimizer/RNG artifact, not a remote model weight file.
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
    if state.get("format_version") != FORMAT_VERSION:
        raise ValueError("Checkpoint state format version differs")
    _validate_metadata(state["metadata"], expected_metadata)
    if state["cursor"] != manifest["cursor"]:
        raise ValueError("Checkpoint cursor differs from manifest")
    cursor = TrainingCursor(**state["cursor"])
    cursor.validate()
    validate_lora_state_dict(model, state["lora"])
    groups = optimizer.state_dict()["param_groups"]
    saved_groups = state["optimizer"]["param_groups"]
    if len(groups) != len(saved_groups) or any(len(a["params"]) != len(b["params"]) for a, b in zip(groups, saved_groups)):
        raise ValueError("Checkpoint optimizer parameter groups differ")
    if state["lr_scheduler"].get("union_warmup_updates") != lr_scheduler.state_dict().get("union_warmup_updates"):
        raise ValueError("Checkpoint LR warmup function differs")
    expected_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if state["rng"]["cuda_device_count"] != expected_cuda:
        raise ValueError("CUDA RNG device count differs from checkpoint")
    # Signature, tensor layout and optimizer structure checks precede mutations.
    load_lora_state_dict(model, state["lora"])
    optimizer.load_state_dict(state["optimizer"])
    lr_scheduler.load_state_dict(state["lr_scheduler"])
    optimizer.zero_grad(set_to_none=True)
    restore_rng_state(state["rng"])
    return cursor
