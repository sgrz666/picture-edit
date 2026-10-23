"""The pair contract for geometry-conditioned training (remediation plan §3/§4).

`src/data/canvas_contract.py` makes a single canvas un-stretchable. This module is the other
half: the rules a *pair* must satisfy so that no trainer can quietly re-introduce a square
assumption between the dataset and the adapter — the plan's "训练时禁止任何隐式回退".

Deliberately a validator rather than a second loader. `TikTokV3PoseDataset._native_item`
already builds two independent per-side canvases and rasterises every condition through them;
copying that logic here would create exactly the per-module resize divergence §1 warns about.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Mapping, Sequence

import torch

from .canvas_contract import (
    CANVAS_MULTIPLE,
    VAE_STRIDE,
    CanvasPolicyError,
    CanvasSpec,
    require_canonical,
)

SCHEMA = "geometry_canvas_v2"

# What a geometry pair must carry. Names on the right are the repo's existing keys; the plan's
# table (§3) uses canvas_valid_mask / source_canvas / target_canvas, both spellings are read.
REQUIRED_TENSORS = {
    "source_rgb": "src_image",
    "target_rgb": "tgt_image",
    "pose_heatmap": "pose_heatmap",
    "normal": "normal",
    "part_onehot": "part_onehot",
    "human_mask": "human_mask",
    "canvas_valid_mask": "valid_image_mask",
    "smplx_global": "smplx_global",
}
CANVAS_KEYS = {"source_canvas": ("source_canvas", "source_canvas_spec"),
               "target_canvas": ("target_canvas", "target_canvas_spec")}


class PairContractError(ValueError):
    """A pair cannot be trusted to sit on one rectangular, aspect-preserving canvas."""


def _resolve(sample: Mapping[str, Any], plan_name: str) -> Any:
    for key in (plan_name, *CANVAS_KEYS.get(plan_name, ())):
        if key in sample:
            return sample[key]
    raise PairContractError(
        f"pair is missing {plan_name!r}; has {sorted(k for k in sample if not k.startswith('_'))}")


def _canvas_of(sample: Mapping[str, Any], plan_name: str) -> CanvasSpec:
    value = _resolve(sample, plan_name)
    return require_canonical(value, where=f"pair contract ({plan_name})")


def assert_pair_contract(sample: Mapping[str, Any], *, where: str = "training") -> dict[str, Any]:
    """Validate one pair and return its resolved canvas facts.

    Raises instead of repairing: the plan's whole point is that a mismatch must stop the run
    rather than be interpolated away.
    """
    for plan_name, repo_key in REQUIRED_TENSORS.items():
        if plan_name not in sample and repo_key not in sample:
            raise PairContractError(
                f"{where}: pair lacks {plan_name} (expected tensor {repo_key!r}); "
                f"a missing condition must not be zero-filled")

    source = _canvas_of(sample, "source_canvas")
    target = _canvas_of(sample, "target_canvas")

    canvas_h, canvas_w = target.canvas_hw
    if canvas_h % VAE_STRIDE or canvas_w % VAE_STRIDE:
        raise PairContractError(f"{where}: target canvas {(canvas_h, canvas_w)} is not on the "
                                f"{VAE_STRIDE}x VAE stride")
    declared = sample.get("target_token_hw")
    if declared is None:
        raise PairContractError(
            f"{where}: pair carries no target_token_hw; the grid must travel with the sample "
            "instead of being derived from an integer resolution")
    declared = (int(declared[0]), int(declared[1]))
    if declared != target.token_hw:
        raise PairContractError(f"{where}: target_token_hw {declared} disagrees with the target "
                                f"canvas {(canvas_h, canvas_w)} -> {target.token_hw}")

    spatial = {plan_name: sample.get(plan_name, sample.get(repo_key))
               for plan_name, repo_key in REQUIRED_TENSORS.items()
               if plan_name != "smplx_global"}
    if not torch.is_tensor(spatial["source_rgb"]) or spatial["source_rgb"].ndim < 2:
        raise PairContractError(f"{where}: source_rgb must be a spatial tensor")
    # The reference image rides on its own canvas; requiring it to match the target's is what
    # would force A and B to share an aspect ratio.
    if tuple(spatial["source_rgb"].shape[-2:]) != source.canvas_hw:
        raise PairContractError(
            f"{where}: source_rgb sits on {tuple(spatial['source_rgb'].shape[-2:])} but the source "
            f"canvas is {source.canvas_hw}")
    del spatial["source_rgb"]
    for plan_name, tensor in spatial.items():
        if not torch.is_tensor(tensor) or tensor.ndim < 2:
            raise PairContractError(f"{where}: {plan_name} must be a spatial tensor")
        if tuple(tensor.shape[-2:]) != (canvas_h, canvas_w):
            raise PairContractError(
                f"{where}: {plan_name} sits on {tuple(tensor.shape[-2:])} but the target canvas is "
                f"{(canvas_h, canvas_w)}; conditions must be rasterised on the target canvas, "
                "not resized to it later")

    valid = spatial["canvas_valid_mask"]
    if bool((valid < 0).any()) or bool((valid > 1).any()):
        raise PairContractError(f"{where}: canvas_valid_mask must be a 0..1 coverage mask")
    expected_area = canvas_h * canvas_w - sum(
        (target.canvas_hw[i] - target.resized_hw[i]) * canvas_w if i == 0 else
        (target.canvas_hw[i] - target.resized_hw[i]) * target.resized_hw[0]
        for i in range(2))
    covered = float(valid.sum())
    if abs(covered - expected_area) > max(1.0, 0.01 * expected_area):
        raise PairContractError(
            f"{where}: canvas_valid_mask covers {covered:.0f}px but the declared canvas leaves "
            f"{expected_area}px of padding; the mask and the CanvasSpec describe different images")
    human = spatial["human_mask"]
    if bool((human * (1 - valid)).max() > 0):
        raise PairContractError(f"{where}: human_mask marks pixels outside the valid canvas area")

    depth = sample.get("depth")
    if torch.is_tensor(depth) and "depth_metadata" not in sample:
        raise PairContractError(
            f"{where}: depth is present without depth_metadata; a normalised grayscale visualisation "
            "must not be consumed as metric depth (plan §6)")

    return {"schema": SCHEMA, "source_canvas": source, "target_canvas": target,
            "target_token_hw": declared, "canvas_hw": (canvas_h, canvas_w),
            "policy": target.policy}


def canvas_fingerprint(sample: Mapping[str, Any]) -> str:
    """Stable per-pair fingerprint binding caches and checkpoints to both canvases."""
    source = _canvas_of(sample, "source_canvas")
    target = _canvas_of(sample, "target_canvas")
    return f"{source.fingerprint()}:{target.fingerprint()}"


def collate_geometry_pairs(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Stack only rows that share both canvases.

    Mixed sizes are a bucketing bug, not something to solve by resampling to a common square.
    """
    from torch.utils.data._utils.collate import default_collate

    if not rows:
        raise PairContractError("empty batch")
    keys = {(tuple(r["target_canvas"]["canvas_hw"] if isinstance(r.get("target_canvas"), Mapping)
                  else r["target_canvas"].canvas_hw),
             tuple(r["source_canvas"]["canvas_hw"] if isinstance(r.get("source_canvas"), Mapping)
                   else r["source_canvas"].canvas_hw)) for r in rows}
    if len(keys) > 1:
        raise PairContractError(
            f"batch mixes canvases {sorted(keys)}; bucket by (H, W) or use batch_size=1 rather "
            "than stacking everything onto one square")
    from torch.utils.data._utils.collate import default_collate as _default

    tensors, extras = [], {}
    for row in rows:
        tensors.append({k: v for k, v in row.items() if torch.is_tensor(v)})
        extras.setdefault("__non_tensors__", []).append(
            {k: v for k, v in row.items() if not torch.is_tensor(v)})
    batch = _default(tensors)
    if len(rows) == 1:
        for key, value in extras["__non_tensors__"][0].items():
            batch.setdefault(key, [value])
    else:
        for key in extras["__non_tensors__"][0]:
            batch.setdefault(key, [row[key] for row in extras["__non_tensors__"]])
    return batch


def bucket_by_canvas(dataset: Sequence[Mapping[str, Any]],
                     *, key: Callable[[Mapping[str, Any]], tuple] | None = None
                     ) -> dict[tuple[int, int, int, int], list[int]]:
    """Group pair indices by (target H, W, source H, W) so a batch never has to distort."""
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for index in range(len(dataset)):
        row = dataset[index] if not isinstance(dataset, Mapping) else dataset[index]
        target = row["target_canvas"]
        source = row["source_canvas"]
        get = lambda canvas, name: (canvas[name] if isinstance(canvas, Mapping)
                                    else getattr(canvas, name))
        signature = key(row) if key is not None else (
            tuple(get(target, "canvas_hw")), tuple(get(source, "canvas_hw")))
        buckets[tuple(tuple(part) for part in signature)].append(index)
    return dict(buckets)


__all__ = ["SCHEMA", "REQUIRED_TENSORS", "PairContractError", "assert_pair_contract",
           "canvas_fingerprint", "collate_geometry_pairs", "bucket_by_canvas",
           "CANVAS_MULTIPLE"]
