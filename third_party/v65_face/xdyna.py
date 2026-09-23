"""Small pipeline utilities directly adapted from pinned X-Dyna control flow.

Upstream file ``animatediff/pipelines/pipeline_xdyna.py`` at commit
``9a54f8e9b90c195eb1f21641c791896bcefe4ce0`` (Apache-2.0).
"""

from __future__ import annotations

import torch


def repeat_condition_batch(condition: torch.Tensor, factor: int) -> torch.Tensor:
    """Repeat complete batches in X-Dyna CFG order: [uncond batch, cond batch]."""

    if factor <= 0:
        raise ValueError("factor must be positive")
    if factor == 1:
        return condition
    return torch.cat([condition] * factor, dim=0)


def add_optional_face_residual(
    base_residual: torch.Tensor,
    face_residual: torch.Tensor | None,
) -> torch.Tensor:
    """Add an independently executed face-control residual at the same level."""

    if face_residual is None:
        return base_residual
    if face_residual.shape != base_residual.shape:
        raise ValueError("face residual must match the base residual shape")
    return base_residual + face_residual


__all__ = ["add_optional_face_residual", "repeat_condition_batch"]
