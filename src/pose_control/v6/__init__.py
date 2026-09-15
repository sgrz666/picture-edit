"""Unified single/dual-person SMPL-X Adapter V6."""

from .conditions import TaskSpec, UnifiedAdapterCondition
from .deepgen_adapter import UnifiedSMPLXAdapterV6

__all__ = ["TaskSpec", "UnifiedAdapterCondition", "UnifiedSMPLXAdapterV6"]
