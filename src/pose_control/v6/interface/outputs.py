from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..conditions import _to_preserving_discrete_dtype
from ..control_core import BranchControlResiduals
from ..reasoning import InternalControlState


@dataclass
class PreparedControlConditioning:
    """Static per-image conditions consumed by every denoising step."""

    geometry_condition: torch.Tensor
    interaction_condition: torch.Tensor
    adapter_tokens: torch.Tensor
    adapter_token_mask: torch.Tensor
    person_count: torch.Tensor
    interaction_valid: torch.Tensor
    control_state: InternalControlState

    @property
    def batch_size(self) -> int:
        return int(self.geometry_condition.shape[0])

    def validate(self) -> "PreparedControlConditioning":
        if self.geometry_condition.ndim != 4:
            raise ValueError("geometry_condition must have shape [B,C,H,W]")
        if self.interaction_condition.shape != self.geometry_condition.shape:
            raise ValueError("interaction_condition must match geometry_condition")
        batch_size = self.batch_size
        if self.adapter_tokens.ndim != 3 or self.adapter_tokens.shape[0] != batch_size:
            raise ValueError("adapter_tokens must have shape [B,M,D]")
        if (
            self.adapter_token_mask.shape != self.adapter_tokens.shape[:2]
            or self.adapter_token_mask.dtype != torch.bool
        ):
            raise ValueError("adapter_token_mask must have shape [B,M] and dtype bool")
        if self.person_count.shape != (batch_size,) or self.person_count.dtype != torch.long:
            raise ValueError("person_count must have shape [B] and dtype int64")
        if (
            self.interaction_valid.shape != (batch_size,)
            or self.interaction_valid.dtype != torch.bool
        ):
            raise ValueError("interaction_valid must have shape [B] and dtype bool")
        if not torch.equal(self.interaction_valid, self.person_count == 2):
            raise ValueError("interaction_valid must identify exactly the dual-person rows")
        if self.control_state.batch_size != batch_size:
            raise ValueError("control_state batch does not match prepared conditions")
        invalid = ~self.interaction_valid
        if invalid.any() and torch.count_nonzero(self.interaction_condition[invalid]) != 0:
            raise ValueError("single-person interaction_condition must be exactly zero")
        if torch.count_nonzero(
            self.adapter_tokens * (~self.adapter_token_mask)[..., None]
        ) != 0:
            raise ValueError("masked adapter tokens must be exactly zero")
        return self

    def index_select(self, indices: torch.Tensor) -> "PreparedControlConditioning":
        return type(self)(
            geometry_condition=self.geometry_condition.index_select(0, indices),
            interaction_condition=self.interaction_condition.index_select(0, indices),
            adapter_tokens=self.adapter_tokens.index_select(0, indices),
            adapter_token_mask=self.adapter_token_mask.index_select(0, indices),
            person_count=self.person_count.index_select(0, indices),
            interaction_valid=self.interaction_valid.index_select(0, indices),
            control_state=self.control_state.index_select(indices),
        )

    def expand_to_batch(self, batch_size: int) -> "PreparedControlConditioning":
        if batch_size % self.batch_size:
            raise ValueError("prepared batch cannot be expanded to transformer batch")
        factor = batch_size // self.batch_size
        indices = torch.arange(self.batch_size, device=self.geometry_condition.device).repeat(
            factor
        )
        return self.index_select(indices).validate()

    def to(self, *args, **kwargs) -> "PreparedControlConditioning":
        return type(self)(
            geometry_condition=self.geometry_condition.to(*args, **kwargs),
            interaction_condition=self.interaction_condition.to(*args, **kwargs),
            adapter_tokens=self.adapter_tokens.to(*args, **kwargs),
            adapter_token_mask=_to_preserving_discrete_dtype(
                self.adapter_token_mask, *args, **kwargs
            ),
            person_count=_to_preserving_discrete_dtype(
                self.person_count, *args, **kwargs
            ),
            interaction_valid=_to_preserving_discrete_dtype(
                self.interaction_valid, *args, **kwargs
            ),
            control_state=self.control_state.to(*args, **kwargs),
        )


@dataclass
class DeepGenControlOutput:
    """Six aligned residual groups and lightweight runtime diagnostics."""

    block_controlnet_hidden_states: tuple[torch.Tensor, ...]
    diagnostics: dict[str, Any]

    def validate(self) -> "DeepGenControlOutput":
        if len(self.block_controlnet_hidden_states) != 6:
            raise ValueError("DeepGen control output must contain six residuals")
        expected = None
        for index, residual in enumerate(self.block_controlnet_hidden_states):
            if residual.ndim != 3:
                raise ValueError(f"control residual {index} must have shape [B,N,D]")
            if expected is None:
                expected = residual.shape
            elif residual.shape != expected:
                raise ValueError("all DeepGen residuals must have the same shape")
        return self
