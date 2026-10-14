"""Prompt study package for PE-I2I x Body/Face joint training.

Provides protocol constants, deterministic sampling schedules, pair locking,
prompt compilation, and results tracking. Importing this package does not load
GPU models.
"""
from __future__ import annotations

__all__ = [
    "protocol",
    "sampling",
]
