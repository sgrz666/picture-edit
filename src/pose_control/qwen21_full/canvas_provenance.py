"""Canvas provenance for checkpoints (plan §7), kept out of the copied v6 modules.

`src/pose_control/qwen21_full/` holds 47 files copied verbatim from `src/pose_control/v6/`, and
`tests/test_qwen21_full_preservation.py` pins their SHA256 against a migration manifest. So the
canvas version cannot live in `checkpoint.py` without breaking that invariant; it lives here.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

CANVAS_SCHEMA_VERSION = "canvas_v2"
PRE_CANVAS = "pre_canvas_warm_start"


def require_canvas_compatible(checkpoint: Mapping[str, Any] | None, *, warm_start_ok: bool = False,
                              where: str = "native canvas training") -> str:
    """Say which canvas a checkpoint was trained under, and refuse to guess on new runs.

    Weights from before the contract may still be inherited, but only as an explicit warm
    start: features learned on a square-stretched person have never seen a portrait canvas,
    so treating them as canvas-ready would smuggle the old distortion back in.
    """
    recorded = None if checkpoint is None else checkpoint.get("canvas_schema_version")
    if recorded == CANVAS_SCHEMA_VERSION:
        return CANVAS_SCHEMA_VERSION
    if warm_start_ok:
        return PRE_CANVAS
    raise ValueError(
        f"checkpoint has no {CANVAS_SCHEMA_VERSION} canvas record; {where} may only inherit it as an "
        "explicit warm start (warm_start_ok=True), because the old square preprocessing never saw a "
        "portrait canvas")


__all__ = ["CANVAS_SCHEMA_VERSION", "PRE_CANVAS", "require_canvas_compatible"]
