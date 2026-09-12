"""Integration helpers for external pose-control datasets and DeepGen."""

from .champ_deepgen import (
    ChampSample,
    align_control_residuals,
    image_diagnostics,
    load_champ_sample,
)

__all__ = [
    "ChampSample",
    "align_control_residuals",
    "image_diagnostics",
    "load_champ_sample",
]
