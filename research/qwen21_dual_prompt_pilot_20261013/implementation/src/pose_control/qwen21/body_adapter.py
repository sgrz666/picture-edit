from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from .face_adapter import Qwen21FaceAdapter, Qwen21FaceState

from src.pose_control.v6.condition_bridge import ReasonerControlBridge
from src.pose_control.v6.condition_injector import SMPLXConditionInjector
from src.pose_control.v6.reasoning import SMPLXAdapterReasoner


LAYER_INDICES = (3, 8, 13, 18, 23, 28)


class Qwen21BodyAdapter(nn.Module):
    """Reuse V6 geometry reasoning and produce target-only Qwen21 residuals."""

    def __init__(self, *, width: int = 512, hidden_size: int = 4096) -> None:
        super().__init__()
        self.condition_injector = SMPLXConditionInjector(use_depth=False)
        self.reasoner = SMPLXAdapterReasoner()
        self.condition_bridge = ReasonerControlBridge(
            reasoning_dim=512, geometry_channels=128, token_dim=1536
        )
        self.geometry_in = nn.Conv2d(128, width, kernel_size=1)
        self.hidden_in = nn.Linear(hidden_size, width, bias=False)
        self.time_in = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width))
        self.stage_embedding = nn.Embedding(len(LAYER_INDICES), width)
        self.fusion = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width)
        )
        self.zero_heads = nn.ModuleDict({str(index): nn.Linear(width, hidden_size) for index in LAYER_INDICES})
        for head in self.zero_heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.hidden_size = hidden_size

    def load_v6_geometry(self, checkpoint: str) -> dict[str, int]:
        """Load only shape-compatible encoder/reasoner/bridge tensors from a V6 checkpoint."""
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        source = saved.get("adapter_state_dict", saved)
        current = self.state_dict()
        prefixes = ("condition_injector.", "reasoner.", "condition_bridge.")
        selected = {
            name: value for name, value in source.items()
            if name.startswith(prefixes) and name in current and current[name].shape == value.shape
        }
        self.load_state_dict(selected, strict=False)
        return {prefix.rstrip("."): sum(name.startswith(prefix) for name in selected) for prefix in prefixes}

    def encode_geometry(self, batch: dict[str, Any], target_hw: tuple[int, int]) -> torch.Tensor:
        parameter = next(self.parameters())
        device, dtype = parameter.device, parameter.dtype
        required = ("normal", "pose_heatmap", "part_onehot", "smplx_global", "human_mask")
        inputs = {name: batch[name].to(device=device, dtype=dtype) for name in required}
        task_id = batch["task_id"].to(device=device, dtype=torch.long)
        bundle = self.condition_injector(
            normal_a=inputs["normal"],
            pose_heatmap_a=inputs["pose_heatmap"],
            part_onehot_a=inputs["part_onehot"],
            smplx_global_a=inputs["smplx_global"],
            human_mask_a=inputs["human_mask"],
            task_id=task_id,
        )
        geometry = self.condition_bridge(self.reasoner(bundle)).geometry_scene
        geometry = F.interpolate(geometry, size=target_hw, mode="bilinear", align_corners=False)
        return self.geometry_in(geometry).flatten(2).transpose(1, 2)

    def residual(
        self, layer_index: int, target_hidden: torch.Tensor, geometry: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        if layer_index not in LAYER_INDICES:
            raise ValueError(f"layer {layer_index} has no control head")
        if target_hidden.ndim != 3 or target_hidden.shape[-1] != self.hidden_size:
            raise ValueError("target hidden has an unexpected shape")
        if geometry.shape[:2] != target_hidden.shape[:2]:
            raise ValueError("geometry token grid does not match target hidden")
        if sigma.shape != (target_hidden.shape[0],):
            raise ValueError("sigma must have shape [B]")
        fusion_dtype = self.hidden_in.weight.dtype
        stage = LAYER_INDICES.index(layer_index)
        state = (
            self.hidden_in(target_hidden.to(fusion_dtype))
            + geometry.to(fusion_dtype)
            + self.time_in(sigma.to(fusion_dtype).view(-1, 1))[:, None]
            + self.stage_embedding.weight[stage][None, None]
        )
        delta = self.zero_heads[str(layer_index)](self.fusion(state))
        return delta.to(target_hidden.dtype)


class Qwen21ControlContext(AbstractContextManager):
    """Inject residuals after selected blocks without altering native prefix attention.

    Keep this context alive through backward when gradient checkpointing is enabled.
    A single pipeline/transformer may have only one active control context.
    """

    def __init__(
        self,
        transformer: nn.Module,
        adapter: Qwen21BodyAdapter,
        geometry: torch.Tensor,
        *,
        sigma: torch.Tensor | None = None,
        target_tokens: int | None = None,
        strength: float = 1.0,
        face_adapter: Qwen21FaceAdapter | None = None,
        face_state: Qwen21FaceState | None = None,
        face_strength: float = 0.0,
        body_face_scale: float = 1.0,
    ) -> None:
        self.transformer = transformer
        self.adapter = adapter
        self.geometry = geometry
        self.sigma = sigma
        self.target_tokens = target_tokens
        self.strength = strength
        if not 0 <= body_face_scale <= 1:
            raise ValueError('body_face_scale must lie in [0,1]')
        if face_strength != 0 and (face_adapter is None or face_state is None):
            raise ValueError('active face control requires face adapter and state')
        self.face_adapter = face_adapter
        self.face_state = face_state
        self.face_strength = face_strength
        self.body_face_scale = body_face_scale
        self.handles: list[Any] = []
        self.call_counts = {index: 0 for index in LAYER_INDICES}

    def _before_transformer(self, module, args, kwargs):
        if "timestep" in kwargs:
            self.sigma = kwargs["timestep"].reshape(-1)
        if "img_shapes" in kwargs:
            target = kwargs["img_shapes"][0][-1]
            self.target_tokens = target[1] * target[2]

    def _after_block(self, layer_index: int):
        def inject(module, args, output):
            if self.sigma is None or self.target_tokens is None:
                raise RuntimeError("Qwen21 control context lacks sigma or target token count")
            if not isinstance(output, torch.Tensor) or output.shape[1] < self.target_tokens:
                raise ValueError("Qwen21 block output does not contain the target token segment")
            self.call_counts[layer_index] += 1
            if self.strength == 0 and self.face_strength == 0:
                return output
            prefix = output.shape[1] - self.target_tokens
            target = output[:, prefix:]
            controlled = target
            if self.strength != 0:
                delta = self.adapter.residual(layer_index, target, self.geometry, self.sigma)
                if self.body_face_scale != 1 and self.face_state is not None and layer_index >= 18:
                    delta = delta * (1 - (1 - self.body_face_scale) * self.face_state.mask.to(delta.dtype))
                controlled = controlled + self.strength * delta
            if self.face_strength != 0:
                assert self.face_adapter is not None and self.face_state is not None
                controlled = controlled + self.face_strength * self.face_adapter.residual(
                    layer_index, controlled, self.face_state, self.sigma,
                )
            return controlled if prefix == 0 else torch.cat((output[:, :prefix], controlled), dim=1)

        return inject

    def __enter__(self):
        if getattr(self.transformer, "_pic_edit_qwen21_control_active", False):
            raise RuntimeError("Qwen21 transformer already has an active control context")
        if len(self.transformer.transformer_blocks) != 32:
            raise ValueError("expected the official 32-block Qwen-Image-2.1 transformer")
        self.transformer._pic_edit_qwen21_control_active = True
        self.handles.append(self.transformer.register_forward_pre_hook(self._before_transformer, with_kwargs=True))
        for index in LAYER_INDICES:
            self.handles.append(self.transformer.transformer_blocks[index].register_forward_hook(self._after_block(index)))
        return self

    def __exit__(self, exception_type, exception, traceback):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self.transformer._pic_edit_qwen21_control_active = False
        return False
