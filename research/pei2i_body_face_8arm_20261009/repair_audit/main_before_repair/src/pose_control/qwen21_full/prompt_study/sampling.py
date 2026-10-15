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
    return min((update_index + 1) / 20, 1.0)


def validate_trajectory(
    saved: dict[str, Any], metrics: list[dict[str, Any]], order: list[int]
) -> None:
    step = int(saved["step"])
    if saved["sample_order"] != order[:step] or len(metrics) != step:
        raise ValueError("Resume sample_order/metrics differs")
    for u, row in enumerate(metrics):
        if (row["step"], row["pair_index"], row["noise_seed"]) != (
            u + 1,
            order[u],
            noise_seed(u),
        ):
            raise ValueError(f"Resume trajectory differs at update {u}")
