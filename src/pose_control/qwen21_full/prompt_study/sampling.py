from __future__ import annotations

import random
from typing import Any


def make_order() -> list[int]:
    second = list(range(512))
    random.Random(1709).shuffle(second)
    return list(range(512)) + second


def noise_seed(update_index: int) -> int:
    if not 0 <= update_index < 1024:
        raise ValueError("update_index must be in 0..1023")
    return 42 + 256 + update_index


def warmup_scale(update_index: int) -> float:
    noise_seed(update_index)
    return min((update_index + 1) / 20, 1.0)


def validate_update_bounds(start_step: int, stop_after_step: int) -> None:
    if not isinstance(start_step, int) or not isinstance(stop_after_step, int) or not 0 <= start_step <= stop_after_step <= 1024:
        raise ValueError("Require 0 <= start_step <= stop_after_step <= 1024")


def validate_trajectory(
    saved: dict[str, Any], metrics: list[dict[str, Any]], order: list[int], *,
    pair_ids: list[str] | None = None, prompt_hashes: dict[str, str] | None = None,
) -> None:
    step = saved["step"]
    validate_update_bounds(step, 1024)
    if order != make_order():
        raise ValueError("Resume trajectory differs from locked order")
    if saved.get("next_update_index", step) != step:
        raise ValueError("Resume next_update_index differs from step")
    if saved["sample_order"] != order[:step] or len(metrics) != step:
        raise ValueError("Resume sample_order/metrics differs")
    for u, row in enumerate(metrics):
        if (row["step"], row["pair_index"], row["noise_seed"]) != (
            u + 1,
            order[u],
            noise_seed(u),
        ):
            raise ValueError(f"Resume trajectory differs at update {u}")
        if pair_ids is not None:
            expected_id = pair_ids[order[u]]
            if row.get("pair_id") != expected_id or row.get("prompt_sha256") != (prompt_hashes or {}).get(expected_id):
                raise ValueError(f"Resume trajectory pair/prompt differs at update {u}")
