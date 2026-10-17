"""官方 control block 后、after_proj 前的 target-only 人物任务增量。"""
from __future__ import annotations

import torch
from torch import nn

from .banks import PersonBankAttention
from .contracts import ProAdapterConfig, TaskCondition
from .encoders import GeometryEncoder


class _IdentitySafeAdd(torch.autograd.Function):
    """数值为零的增量保留原 bits（包括 -0），反向仍是普通加法。"""

    @staticmethod
    def forward(ctx, hidden, delta):
        return torch.where(delta == 0, hidden, hidden + delta)

    @staticmethod
    def backward(ctx, gradient):
        return gradient, gradient


class ProTaskAdapter(nn.Module):
    """只负责本层 target scatter；官方 joint self-attention 仍可跨人物。"""

    def __init__(self, config: ProAdapterConfig | None = None):
        super().__init__()
        self.config = config or ProAdapterConfig()
        self.enabled = True
        cfg = self.config
        self.query_norm = nn.LayerNorm(cfg.hidden_dim, eps=1e-6, elementwise_affine=False)
        self.query_projection = nn.Linear(cfg.hidden_dim, cfg.width)
        self.geometry_encoder = GeometryEncoder(cfg)
        self.banks = PersonBankAttention(cfg)
        self.time_projection = nn.Sequential(nn.Linear(1, cfg.width), nn.SiLU(), nn.Linear(cfg.width, cfg.width))
        self.fusion = nn.Sequential(nn.Linear(cfg.width, cfg.width), nn.SiLU(), nn.Linear(cfg.width, cfg.width))
        self.register_buffer("gate", torch.ones(()))
        self.output_heads = nn.ModuleDict({str(stage): nn.Linear(cfg.width, cfg.hidden_dim) for stage in cfg.stage_indices})
        for head in self.output_heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _apply(self, fn, recurse=True):
        originals = {name: (parameter.data, parameter.grad.detach() if parameter.grad is not None else None)
                     for name, parameter in self.named_parameters() if parameter.is_floating_point()}
        super()._apply(fn, recurse=recurse)
        for name, parameter in self.named_parameters():
            if name not in originals:
                continue
            original, gradient = originals[name]
            if original.is_meta and not parameter.is_meta:
                parameter.data = torch.empty_like(parameter, dtype=torch.float32)
            else:
                parameter.data = original.to(device=parameter.device, dtype=torch.float32)
            if gradient is not None:
                parameter.grad = gradient.to(device=parameter.device, dtype=torch.float32)
        return self

    def trainable_parameter_report(self) -> dict:
        parameters = {name: parameter for name, parameter in self.named_parameters() if parameter.requires_grad}
        return {"names": tuple(parameters), "count": sum(p.numel() for p in parameters.values()),
                "all_fp32": all(p.dtype == torch.float32 for p in parameters.values()),
                "config_signature": self.config.signature()}

    def get_extra_state(self):
        return {"version": 1, "config": self.config.to_dict(), "config_signature": self.config.signature(),
                "enabled": self.enabled}

    def set_extra_state(self, state):
        if not isinstance(state, dict) or set(state) != {"version", "config", "config_signature", "enabled"}:
            raise ValueError("invalid Pro adapter config state")
        if state["version"] != 1 or state["config"] != self.config.to_dict() or state["config_signature"] != self.config.signature():
            raise ValueError("Pro adapter config differs from checkpoint")
        if type(state["enabled"]) is not bool:
            raise ValueError("invalid Pro adapter enabled state")
        self.enabled = state["enabled"]

    def forward_layer(self, stage, hidden, target_mask, conditions: TaskCondition, timestep):
        if not self.enabled or stage not in self.config.stage_indices:
            return hidden
        if hidden.ndim != 3 or hidden.shape[-1] != self.config.hidden_dim or not hidden.is_floating_point():
            raise ValueError("hidden must have shape [B,L,hidden_dim] and floating dtype")
        if not isinstance(target_mask, torch.Tensor) or target_mask.dtype != torch.bool or target_mask.ndim != 1 or target_mask.shape[0] != hidden.shape[1]:
            raise ValueError("target_mask must be a bool [L] tensor")
        conditions.validate(self.config)
        b = hidden.shape[0]
        if conditions.person_valid.shape[0] != b:
            raise ValueError("condition batch must match hidden batch")
        if target_mask.sum().item() != conditions.target_hw[0] * conditions.target_hw[1]:
            raise ValueError("target token count must match target_hw")
        if not isinstance(timestep, torch.Tensor) or timestep.shape != (b,) or not torch.isfinite(timestep).all():
            raise ValueError("timestep must have finite shape [B]")
        conditions = conditions.to(hidden.device, torch.float32)
        target_mask = target_mask.to(hidden.device)
        # Keep the trainable branch FP32 even under an enclosing mixed precision context.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            query = self.query_projection(self.query_norm(hidden[:, target_mask].float()))
            geometry = self.geometry_encoder(conditions)
            appearance = self.banks(query[:, None] + geometry, conditions)
            owner = conditions.visible_owner().float()
            appearance = (appearance * owner[..., None]).sum(dim=1)
            time = self.time_projection(timestep.to(device=hidden.device, dtype=torch.float32)[:, None])[:, None]
            fused = self.fusion(query + geometry.sum(dim=1) + appearance + time)
            support = (conditions.person_mask * conditions.person_valid[..., None]).sum(dim=1).clamp(0, 1)
            delta = self.output_heads[str(stage)](fused) * self.gate * support[..., None]
        output = hidden.clone()
        output[:, target_mask] = _IdentitySafeAdd.apply(hidden[:, target_mask], delta.to(hidden.dtype))
        return output

    def forward(self, hidden, *, stage, target_mask, conditions, timestep):
        # __call__ is required by integration so offload hooks run on this module.
        return self.forward_layer(stage, hidden, target_mask, conditions, timestep)
