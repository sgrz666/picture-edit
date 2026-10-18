"""预登记重训 arm；仅在 forward 的局部副本屏蔽任务条件。"""
from __future__ import annotations

from dataclasses import replace

import torch

from .adapter import ProTaskAdapter
from .cache import tensor_sha256

ARMS = ("full4", "no_dense", "no_appearance", "no_contact", "full8")


def arm_stages(arm):
    if arm not in ARMS:
        raise ValueError(f"Unknown retrained arm: {arm}")
    return tuple(range(0, 16, 2)) if arm == "full8" else (0, 4, 8, 12)


class AblatedTaskAdapter(ProTaskAdapter):
    """参数对象/名称保持原样；arm 属于不可变 checkpoint extra state。"""

    def __init__(self, config=None, *, arm="full4"):
        super().__init__(config)
        self._bind_arm(arm)

    def _bind_arm(self, arm):
        if tuple(self.config.stage_indices) != arm_stages(arm):
            raise ValueError("Ablation stage configuration differs from arm")
        if hasattr(self, "_ablation_arm") and self._ablation_arm != arm:
            raise ValueError("Ablation arm is immutable")
        self._ablation_arm = arm

    @property
    def arm(self):
        return self._ablation_arm

    def __setattr__(self, name, value):
        if name == "arm" or (name == "_ablation_arm" and hasattr(self, name)):
            raise AttributeError("Ablation arm is immutable")
        return super().__setattr__(name, value)

    def get_extra_state(self):
        return {**super().get_extra_state(), "ablation_arm": self.arm}

    def set_extra_state(self, state):
        if not isinstance(state, dict) or state.get("ablation_arm") != self.arm:
            raise ValueError("Checkpoint ablation arm differs")
        super().set_extra_state({k: v for k, v in state.items() if k != "ablation_arm"})

    def forward_layer(self, stage, hidden, target_mask, conditions, timestep):
        # The original pack is still used by cache verification and regional loss.
        # Do not mutate its tensors or provenance, even for an in-memory diagnostic.
        if self.arm == "no_appearance":
            conditions = replace(conditions,
                source_tokens=torch.zeros_like(conditions.source_tokens),
                source_token_valid=torch.zeros_like(conditions.source_token_valid),
                face_identity=torch.zeros_like(conditions.face_identity),
                face_identity_valid=torch.zeros_like(conditions.face_identity_valid))
        disabled = {"normal", "depth", "part"} if self.arm == "no_dense" else (
            {"contact"} if self.arm == "no_contact" else set())
        if disabled:
            conditions = replace(conditions,
                spatial={k: torch.zeros_like(v) if k in disabled else v for k, v in conditions.spatial.items()},
                spatial_valid={k: torch.zeros_like(v) if k in disabled else v for k, v in conditions.spatial_valid.items()})
        if self.arm == "no_contact":
            conditions = replace(conditions, relations=replace(conditions.relations,
                valid=torch.zeros_like(conditions.relations.valid),
                confidence=torch.zeros_like(conditions.relations.confidence)))
        return super().forward_layer(stage, hidden, target_mask, conditions, timestep)


def tensor_inventory(adapter):
    return {name: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": tensor_sha256(value)}
            for name, value in adapter.state_dict().items() if isinstance(value, torch.Tensor)}


def configure_ablation(model, *, arm="full4", seed=3407):
    """builder callback：在加载 checkpoint/offload 前安装，不换 Parameter。"""
    adapter = model.task_adapter
    if type(adapter) is not ProTaskAdapter:
        raise ValueError("Ablation setup requires a fresh ProTaskAdapter")
    if tuple(adapter.config.stage_indices) != arm_stages(arm):
        raise ValueError("Ablation stage configuration differs from arm")
    adapter.__class__ = AblatedTaskAdapter
    adapter._bind_arm(arm)
    # Independent seed scope makes shared tensor bytes identical for every arm.
    # New full8 output heads start at zero; they never perturb earlier modules.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        initial = AblatedTaskAdapter(adapter.config, arm=arm)
    adapter.load_state_dict(initial.state_dict(), strict=True)
    return {"arm": arm, "initialization_seed": seed, "tensors": tensor_inventory(adapter),
            "configured_trainable": adapter.trainable_parameter_report()}


def active_gradient_report(model):
    configured, active = {}, {}
    for name, value in model.named_parameters():
        if value.requires_grad:
            configured[name] = value.numel()
            if value.grad is not None and torch.count_nonzero(value.grad).item():
                active[name] = value.numel()
    return {"configured_parameters": sum(configured.values()), "configured_names": list(configured),
            "active_parameters": sum(active.values()), "active_names": list(active)}
