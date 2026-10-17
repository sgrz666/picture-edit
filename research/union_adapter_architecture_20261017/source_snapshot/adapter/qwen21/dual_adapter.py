"""Two-person geometry, contact and source-appearance control for Qwen-Image-2.1."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
from torch import nn
import torch.nn.functional as F

from src.pose_control.v6.token_encoders import GlobalConditionTokenEncoder, PersonTokenBinder
from .body_adapter import LAYER_INDICES, Qwen21BodyAdapter, Qwen21ControlContext


@dataclass
class Qwen21DualState:
    geometry: torch.Tensor            # [B,N,512]
    interaction: torch.Tensor         # [B,N,512]
    appearance: torch.Tensor          # [B,N,512], bound to target A/B regions
    target_mask: torch.Tensor         # [B,N,1]
    interaction_valid: torch.Tensor   # [B]
    interaction_strength: float = 1.0
    appearance_strength: float = 1.0

    def geometry_only(self) -> "Qwen21DualState":
        return replace(self, interaction_strength=0.0, appearance_strength=0.0)


class Qwen21DualAdapter(Qwen21BodyAdapter):
    """Preserve the iPER single path and add a trainable dual-person path.

    Pass Qwen21DualState as the `geometry` value to Qwen21ControlContext.
    Its existing hooks still preserve all Qwen reference/prompt prefix tokens.
    """

    def __init__(self, *, width: int = 512, hidden_size: int = 4096) -> None:
        super().__init__(width=width, hidden_size=hidden_size)
        # Hi4D uses SMPL plus a different camera vector convention, so this
        # encoder must not overwrite the pre-trained iPER global encoder.
        self.dual_global_encoder = GlobalConditionTokenEncoder()
        self.source_projection = nn.Sequential(
            nn.LayerNorm(64), nn.Linear(64, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.person_binder = PersonTokenBinder(token_dim=width)
        self.interaction_in = nn.Conv2d(128, width, 1)
        self.depth_in = nn.Conv2d(2, width, 1)
        nn.init.zeros_(self.depth_in.weight)
        nn.init.zeros_(self.depth_in.bias)
        self.interaction_fusion = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width)
        )
        self.appearance_fusion = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width)
        )
        self.interaction_heads = nn.ModuleDict({str(i): nn.Linear(width, hidden_size) for i in LAYER_INDICES})
        self.appearance_heads = nn.ModuleDict({str(i): nn.Linear(width, hidden_size) for i in LAYER_INDICES})
        for head in list(self.interaction_heads.values()) + list(self.appearance_heads.values()):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def load_single_checkpoint(self, checkpoint: str) -> dict[str, int]:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        source = saved.get("adapter_state_dict", saved)
        current = self.state_dict()
        selected = {key: value for key, value in source.items()
                    if key in current and current[key].shape == value.shape}
        self.load_state_dict(selected, strict=False)
        return {"loaded": len(selected), "new": len(current) - len(selected)}

    @staticmethod
    def _pool_geometry(tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, people, count, width = tokens.shape
        side = math.isqrt(count)
        if side * side != count:
            raise ValueError("person geometry tokens must form a square grid")
        features = tokens.reshape(batch * people, side, side, width).permute(0, 3, 1, 2)
        weights = mask.reshape(batch * people, 1, side, side).to(features.dtype)
        numerator = F.adaptive_avg_pool2d(features * weights, (2, 2))
        denominator = F.adaptive_avg_pool2d(weights, (2, 2))
        pooled = (numerator / denominator.clamp_min(1e-6)).flatten(2).transpose(1, 2)
        valid = denominator.flatten(2).squeeze(1) > 0
        return pooled.reshape(batch, people, 4, width), valid.reshape(batch, people, 4)

    def _source_tokens(self, source_latent: torch.Tensor, source_masks: torch.Tensor) -> torch.Tensor:
        if source_latent.ndim != 3 or source_latent.shape[-1] != 64:
            raise ValueError("Qwen source latent must have shape [B,H*W,64]")
        batch, count, _ = source_latent.shape
        side = math.isqrt(count)
        if side * side != count or source_masks.shape[:2] != (batch, 2):
            raise ValueError("source latent grid or A/B source masks are invalid")
        latent = source_latent.transpose(1, 2).reshape(batch, 64, side, side)
        masks = F.interpolate(source_masks.flatten(0, 1).float(), (side, side), mode="nearest")
        tokens = []
        for item in range(batch):
            persons = []
            for person in range(2):
                mask = masks[item * 2 + person, 0]
                yy, xx = torch.where(mask > 0.5)
                if not len(xx):
                    raise ValueError(f"source person {person} mask is empty")
                x0, x1 = int(xx.min()), int(xx.max()) + 1
                y0, y1 = int(yy.min()), int(yy.max()) + 1
                cropped = latent[item:item + 1, :, y0:y1, x0:x1]
                masked = cropped * mask[y0:y1, x0:x1][None, None]
                pooled = F.adaptive_avg_pool2d(masked, (2, 4)).flatten(2).transpose(1, 2)
                persons.append(self.source_projection(pooled))
            tokens.append(torch.stack(persons, dim=1))
        return torch.cat(tokens, dim=0)

    def encode_dual(self, batch: dict, source_latent: torch.Tensor,
                    target_hw: tuple[int, int] = (32, 32), *,
                    include_contact: bool = True) -> Qwen21DualState:
        parameter = next(self.parameters())
        device, dtype = parameter.device, parameter.dtype
        float_names = ("normal_a", "normal_b", "pose_heatmap_a", "pose_heatmap_b",
                       "part_onehot_a", "part_onehot_b", "smplx_global_a", "smplx_global_b",
                       "human_mask_a", "human_mask_b", "depth_a", "depth_b",
                       "relative_geometry", "contact_raster", "source_mask_a", "source_mask_b")
        values = {name: batch[name].to(device=device, dtype=dtype) for name in float_names}
        task_id = batch["task_id"].to(device=device, dtype=torch.long)
        valid_b = batch["person_b_valid"].to(device=device, dtype=torch.bool)
        if not valid_b.all():
            raise ValueError("encode_dual expects two valid persons; use encode_geometry for single person")
        relations = (batch["contact_relations"].to(device=device, dtype=dtype)
                     if include_contact else None)
        bundle = self.condition_injector(
            normal_a=values["normal_a"], pose_heatmap_a=values["pose_heatmap_a"],
            part_onehot_a=values["part_onehot_a"], smplx_global_a=values["smplx_global_a"],
            human_mask_a=values["human_mask_a"], normal_b=values["normal_b"],
            pose_heatmap_b=values["pose_heatmap_b"], part_onehot_b=values["part_onehot_b"],
            smplx_global_b=values["smplx_global_b"], human_mask_b=values["human_mask_b"],
            person_b_valid=valid_b, relative_geometry=values["relative_geometry"],
            contact_raster=values["contact_raster"] if include_contact else None,
            contact_relations=relations, task_id=task_id,
        )
        bundle.person_a_global_tokens = self.dual_global_encoder(values["smplx_global_a"], bundle.person_valid[:, 0])
        bundle.person_b_global_tokens = self.dual_global_encoder(values["smplx_global_b"], bundle.person_valid[:, 1])
        reasoning = self.reasoner(bundle)
        bridged = self.condition_bridge(reasoning)
        geometry = F.interpolate(bridged.geometry_scene, size=target_hw, mode="bilinear", align_corners=False)
        interaction = F.interpolate(bridged.interaction_scene, size=target_hw, mode="bilinear", align_corners=False)
        depth = F.interpolate(torch.cat((values["depth_a"], values["depth_b"]), dim=1),
                              size=target_hw, mode="bilinear", align_corners=False)
        geometry_tokens = (self.geometry_in(geometry) + self.depth_in(depth)).flatten(2).transpose(1, 2)
        interaction_tokens = self.interaction_in(interaction).flatten(2).transpose(1, 2)
        pooled_geometry, pooled_mask = self._pool_geometry(reasoning.person_tokens, reasoning.person_token_mask)
        source_masks = torch.stack((values["source_mask_a"], values["source_mask_b"]), dim=1)
        appearance = self._source_tokens(source_latent.to(device=device, dtype=dtype), source_masks)
        source_indices = torch.arange(2, device=device, dtype=torch.long)[None].expand(geometry.shape[0], -1)
        bound, bound_mask = self.person_binder(pooled_geometry, appearance, reasoning.person_valid,
                                              source_indices, geometry_token_mask=pooled_mask)
        role_code = (bound * bound_mask[..., None]).sum(dim=2) / bound_mask.sum(dim=2).clamp_min(1)[..., None]
        target_masks = torch.stack([
            F.interpolate(values[f"human_mask_{side}"], target_hw, mode="nearest").flatten(2).transpose(1, 2)
            for side in ("a", "b")
        ], dim=1)
        appearance_map = (role_code[:, :, None] * target_masks).sum(dim=1)
        union = target_masks.sum(dim=1).clamp(0, 1)
        appearance_map = appearance_map / target_masks.sum(dim=1).clamp_min(1)
        return Qwen21DualState(geometry_tokens, interaction_tokens, appearance_map, union,
                               reasoning.interaction_valid)

    def residual(self, layer_index: int, target_hidden: torch.Tensor,
                 geometry: torch.Tensor | Qwen21DualState, sigma: torch.Tensor) -> torch.Tensor:
        if isinstance(geometry, torch.Tensor):
            return super().residual(layer_index, target_hidden, geometry, sigma)
        state = geometry
        delta = super().residual(layer_index, target_hidden, state.geometry, sigma) * state.target_mask
        dtype = self.hidden_in.weight.dtype
        stage = LAYER_INDICES.index(layer_index)
        common = (self.hidden_in(target_hidden.to(dtype))
                  + self.time_in(sigma.to(dtype).reshape(-1, 1))[:, None]
                  + self.stage_embedding.weight[stage][None, None])
        if state.appearance_strength:
            appearance = self.appearance_heads[str(layer_index)](
                self.appearance_fusion(common + state.appearance.to(dtype)))
            delta = delta + state.appearance_strength * appearance.to(delta.dtype) * state.target_mask
        if state.interaction_strength:
            interaction = self.interaction_heads[str(layer_index)](
                self.interaction_fusion(common + state.interaction.to(dtype)))
            active = state.interaction_valid[:, None, None].to(delta.dtype) * state.target_mask
            delta = delta + state.interaction_strength * interaction.to(delta.dtype) * active
        return delta


__all__ = ["Qwen21DualAdapter", "Qwen21DualState", "Qwen21ControlContext"]
