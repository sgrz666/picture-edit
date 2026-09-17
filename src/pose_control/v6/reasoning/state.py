from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from ..conditions import _to_preserving_discrete_dtype


@dataclass
class InternalControlState:
    """Batch-safe output of Adapter reasoning before DeepGen projection."""

    geometry_feature: torch.Tensor
    geometry_highres: torch.Tensor
    interaction_feature: torch.Tensor
    interaction_highres: torch.Tensor
    person_tokens: torch.Tensor
    person_token_mask: torch.Tensor
    person_valid: torch.Tensor
    person_count: torch.Tensor
    interaction_valid: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.geometry_feature.shape[0])

    def validate(self) -> "InternalControlState":
        if self.geometry_feature.ndim != 4:
            raise ValueError("geometry_feature must have shape [B,C,H,W]")
        batch_size, channels, height, width = self.geometry_feature.shape
        expected_high = (batch_size, channels, height * 2, width * 2)
        if tuple(self.geometry_highres.shape) != expected_high:
            raise ValueError(f"geometry_highres must have shape {expected_high}")
        if tuple(self.interaction_feature.shape) != tuple(self.geometry_feature.shape):
            raise ValueError("interaction_feature must match geometry_feature")
        if tuple(self.interaction_highres.shape) != expected_high:
            raise ValueError("interaction_highres must match geometry_highres")
        token_shape = (batch_size, 2, height * width, channels)
        mask_shape = token_shape[:-1]
        if tuple(self.person_tokens.shape) != token_shape:
            raise ValueError(f"person_tokens must have shape {token_shape}")
        if tuple(self.person_token_mask.shape) != mask_shape or self.person_token_mask.dtype != torch.bool:
            raise ValueError(f"person_token_mask must have shape {mask_shape} and dtype bool")
        if tuple(self.person_valid.shape) != (batch_size, 2) or self.person_valid.dtype != torch.bool:
            raise ValueError("person_valid must have shape [B,2] and dtype bool")
        if tuple(self.person_count.shape) != (batch_size,) or self.person_count.dtype != torch.long:
            raise ValueError("person_count must have shape [B] and dtype int64")
        if tuple(self.interaction_valid.shape) != (batch_size,) or self.interaction_valid.dtype != torch.bool:
            raise ValueError("interaction_valid must have shape [B] and dtype bool")
        if not torch.equal(self.person_count, self.person_valid.sum(dim=1).long()):
            raise ValueError("person_count must equal person_valid.sum(dim=1)")
        if not torch.equal(self.interaction_valid, self.person_count == 2):
            raise ValueError("interaction_valid must identify exactly the dual-person rows")
        if torch.any(self.person_token_mask & ~self.person_valid[..., None]):
            raise ValueError("invalid person slots cannot contain valid tokens")

        single = ~self.interaction_valid
        if single.any():
            if torch.count_nonzero(self.person_tokens[single, 1]) != 0:
                raise ValueError("single-person rows must have zero Person-B tokens")
            if self.person_token_mask[single, 1].any():
                raise ValueError("single-person rows must mask every Person-B token")
            if torch.count_nonzero(self.interaction_feature[single]) != 0:
                raise ValueError("single-person rows must have zero interaction_feature")
            if torch.count_nonzero(self.interaction_highres[single]) != 0:
                raise ValueError("single-person rows must have zero interaction_highres")
        return self

    def index_select(self, indices: torch.Tensor) -> "InternalControlState":
        return type(self)(
            **{
                item.name: getattr(self, item.name).index_select(0, indices)
                for item in fields(self)
            }
        )

    def to(self, *args, **kwargs) -> "InternalControlState":
        return type(self)(
            **{
                item.name: _to_preserving_discrete_dtype(
                    getattr(self, item.name), *args, **kwargs
                )
                for item in fields(self)
            }
        )
