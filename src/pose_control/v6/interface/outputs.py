from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class BranchControlResiduals:
    """Unscaled target-token residuals from the dynamic control branches."""

    geometry: tuple[torch.Tensor, ...]
    interaction: tuple[torch.Tensor, ...]
    target_token_hw: tuple[int, int]

    @property
    def target_token_count(self) -> int:
        return int(self.target_token_hw[0] * self.target_token_hw[1])

    @property
    def batch_size(self) -> int:
        return int(self.geometry[0].shape[0])

    @property
    def hidden_dim(self) -> int:
        return int(self.geometry[0].shape[-1])

    def validate(self) -> "BranchControlResiduals":
        if len(self.geometry) != 6 or len(self.interaction) != 6:
            raise ValueError("geometry and interaction must each contain six residuals")
        if len(self.target_token_hw) != 2 or min(self.target_token_hw) <= 0:
            raise ValueError("target_token_hw must contain two positive dimensions")
        expected = None
        for branch_name, branch in (
            ("geometry", self.geometry),
            ("interaction", self.interaction),
        ):
            for index, residual in enumerate(branch):
                if residual.ndim != 3:
                    raise ValueError(
                        f"{branch_name} residual {index} must have shape [B,N,D]"
                    )
                if residual.shape[1] != self.target_token_count:
                    raise ValueError(
                        f"{branch_name} residual {index} token count does not match target_token_hw"
                    )
                if expected is None:
                    expected = residual.shape
                elif residual.shape != expected:
                    raise ValueError("all branch residuals must have the same shape")
        return self


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
