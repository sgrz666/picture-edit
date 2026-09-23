"""Self-contained iPER single-pair loader for V6.5 Face overfitting.

This module intentionally lives with the V6.5 Face implementation.  The
training entry point must not depend on the experimental ``src.data`` tree or
on its untracked V6.4 dataset implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..detail.conditions import DetailReferenceBatch, FaceHandDetailCondition
from ..detail.hand import HandDetailCondition, hand_to_legacy_detail


@dataclass(frozen=True)
class IPERV65OverfitInputs:
    """One already-batched iPER pair plus the retained Hand branch payload."""

    batch: dict[str, Any]
    hand_detail_condition: FaceHandDetailCondition
    hand_detail_references: DetailReferenceBatch


def _load_bundle(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(bundle, Mapping):
        raise ValueError(f"{label} must contain a mapping: {path}")
    return bundle


def _frame_index(bundle: Mapping[str, Any], stem: str, label: str) -> int:
    stem_to_idx = bundle.get("stem_to_idx")
    if not isinstance(stem_to_idx, Mapping) or stem not in stem_to_idx:
        raise KeyError(f"{stem!r} is absent from {label} stem_to_idx")
    return int(stem_to_idx[stem])


def _tensor(bundle: Mapping[str, Any], name: str) -> torch.Tensor:
    value = bundle.get(name)
    if not torch.is_tensor(value):
        raise ValueError(f"iPER cache field {name!r} must be a tensor")
    return value.detach().cpu()


def _read_image(path: Path, mode: str) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"missing iPER image asset: {path}")
    with path.open("rb") as handle:
        return Image.open(handle).convert(mode)


def _rgb_tensor(path: Path, resolution: int) -> torch.Tensor:
    image = _read_image(path, "RGB").resize(
        (resolution, resolution), Image.Resampling.BILINEAR
    )
    array = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).div_(127.5).sub_(1.0)


def _gray_tensor(path: Path, resolution: int) -> torch.Tensor:
    image = _read_image(path, "L").resize(
        (resolution, resolution), Image.Resampling.NEAREST
    )
    array = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(array).unsqueeze(0).div_(255.0)


def _part_onehot(path: Path, resolution: int) -> torch.Tensor:
    image = _read_image(path, "L").resize(
        (resolution, resolution), Image.Resampling.NEAREST
    )
    labels = torch.from_numpy(np.asarray(image, dtype=np.int64).copy())
    if torch.any((labels < 0) | (labels > 14)):
        raise ValueError(f"part labels must lie in [0,14]: {path}")
    return F.one_hot(labels, num_classes=15)[..., 1:].permute(2, 0, 1).float()


def _pose_heatmaps(
    keypoints: torch.Tensor,
    *,
    resolution: int,
    sigma: float,
) -> torch.Tensor:
    if tuple(keypoints.shape) != (25, 3):
        raise ValueError("kps_25_coords frame must have shape [25,3]")
    keypoints = keypoints.float()
    if not torch.isfinite(keypoints).all():
        raise ValueError("kps_25_coords must contain finite values")
    yy, xx = torch.meshgrid(
        torch.arange(resolution, dtype=torch.float32),
        torch.arange(resolution, dtype=torch.float32),
        indexing="ij",
    )
    x = keypoints[:, 0, None, None] * resolution
    y = keypoints[:, 1, None, None] * resolution
    confidence = keypoints[:, 2, None, None]
    heatmaps = confidence * torch.exp(
        -((xx[None] - x).square() + (yy[None] - y).square()) / (2.0 * sigma**2)
    )
    return torch.where(confidence > 0.05, heatmaps, 0.0).clamp_(0.0, 1.0)


class IPERV65FaceOverfitLoader:
    """Load exactly one deterministic source/target pair for V6.5 training."""

    def __init__(
        self,
        *,
        sampled_root: str | Path,
        assets_root: str | Path,
        appearance: str,
        source_stem: str,
        source_role: str,
        target_stem: str,
        target_role: str,
        resolution: int = 512,
        heatmap_sigma: float = 4.0,
    ) -> None:
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        if heatmap_sigma <= 0:
            raise ValueError("heatmap_sigma must be positive")
        self.sampled_root = Path(sampled_root)
        self.assets_root = Path(assets_root)
        self.appearance = str(appearance)
        self.source_stem = str(source_stem)
        self.source_role = str(source_role)
        self.target_stem = str(target_stem)
        self.target_role = str(target_role)
        self.resolution = int(resolution)
        self.heatmap_sigma = float(heatmap_sigma)

    def _load_hand_payload(
        self, detail: Mapping[str, Any]
    ) -> tuple[FaceHandDetailCondition, DetailReferenceBatch]:
        source_index = _frame_index(detail, self.source_stem, "v6_detail_conditions")
        target_index = _frame_index(detail, self.target_stem, "v6_detail_conditions")
        hand_keypoints_cache = _tensor(detail, "hand_keypoints").float()
        smplx_detail_cache = _tensor(detail, "smplx_detail").float()
        boxes_cache = _tensor(detail, "boxes").float()
        valid_cache = _tensor(detail, "region_valid").bool()
        crops_cache = _tensor(detail, "reference_crops").float()
        if hand_keypoints_cache.ndim != 4 or tuple(hand_keypoints_cache.shape[1:]) != (2, 21, 3):
            raise ValueError("hand_keypoints must have shape [N,2,21,3]")
        if smplx_detail_cache.ndim != 2 or smplx_detail_cache.shape[1] != 103:
            raise ValueError("smplx_detail must have shape [N,103]")
        if boxes_cache.ndim != 3 or tuple(boxes_cache.shape[1:]) != (3, 4):
            raise ValueError("boxes must have shape [N,3,4]")
        if valid_cache.ndim != 2 or valid_cache.shape[1] != 3:
            raise ValueError("region_valid must have shape [N,3]")
        if crops_cache.ndim != 5 or tuple(crops_cache.shape[1:]) != (3, 3, 224, 224):
            raise ValueError("reference_crops must have shape [N,3,3,224,224]")

        hand_keypoints = torch.zeros(1, 2, 2, 21, 3, dtype=torch.float32)
        hand_pose = torch.zeros(1, 2, 2, 45, dtype=torch.float32)
        source_boxes = torch.zeros(1, 2, 2, 4, dtype=torch.float32)
        target_boxes = torch.zeros_like(source_boxes)
        region_valid = torch.zeros(1, 2, 2, dtype=torch.bool)
        hand_keypoints[0, 0] = hand_keypoints_cache[target_index]
        hand_pose[0, 0, 0] = smplx_detail_cache[target_index, 13:58]
        hand_pose[0, 0, 1] = smplx_detail_cache[target_index, 58:103]
        source_boxes[0, 0] = boxes_cache[source_index, 1:]
        target_boxes[0, 0] = boxes_cache[target_index, 1:]
        region_valid[0, 0] = valid_cache[target_index, 1:]
        hand = HandDetailCondition(
            hand_keypoints=hand_keypoints,
            hand_pose=hand_pose,
            source_boxes=source_boxes,
            target_boxes=target_boxes,
            region_valid=region_valid,
            source_indices=torch.tensor([[0, 1]], dtype=torch.long),
        ).validate()
        legacy_condition = hand_to_legacy_detail(hand).validate()

        images = torch.zeros(1, 2, 3, 1, 3, 224, 224, dtype=torch.float32)
        reference_valid = torch.zeros(1, 2, 3, 1, dtype=torch.bool)
        images[0, 0, 1:, 0] = crops_cache[source_index, 1:]
        reference_valid[0, 0, 1:, 0] = valid_cache[source_index, 1:]
        references = DetailReferenceBatch(
            images=images,
            reference_valid=reference_valid,
        ).validate()
        return legacy_condition, references

    def load(self) -> IPERV65OverfitInputs:
        app_sampled = self.sampled_root / self.appearance
        app_assets = self.assets_root / self.appearance
        global_conditions = _load_bundle(
            app_assets / "v6_conditions.pt", "v6_conditions.pt"
        )
        detail_conditions = _load_bundle(
            app_assets / "v6_detail_conditions.pt", "v6_detail_conditions.pt"
        )
        target_index = _frame_index(
            global_conditions, self.target_stem, "v6_conditions"
        )
        smplx_cache = _tensor(global_conditions, "smplx_global").float()
        keypoint_cache = _tensor(global_conditions, "kps_25_coords").float()
        if smplx_cache.ndim != 2 or smplx_cache.shape[1] != 26:
            raise ValueError("smplx_global must have shape [N,26]")
        if keypoint_cache.ndim != 3 or tuple(keypoint_cache.shape[1:]) != (25, 3):
            raise ValueError("kps_25_coords must have shape [N,25,3]")

        source_path = (
            app_sampled / "rgb_1024" / self.source_role / f"{self.source_stem}.png"
        )
        target_path = (
            app_sampled / "rgb_1024" / self.target_role / f"{self.target_stem}.png"
        )
        normal_path = (
            app_assets / "normal_512" / self.target_role / f"{self.target_stem}.png"
        )
        mask_path = (
            app_assets / "mask_512" / self.target_role / f"{self.target_stem}.png"
        )
        part_path = (
            app_assets / "part_512" / self.target_role / f"{self.target_stem}.png"
        )
        source_image = _rgb_tensor(source_path, self.resolution)
        target_image = _rgb_tensor(target_path, self.resolution)
        human_mask = _gray_tensor(mask_path, self.resolution)
        normal = _rgb_tensor(normal_path, self.resolution) * human_mask
        part_onehot = _part_onehot(part_path, self.resolution)
        pose_heatmap = _pose_heatmaps(
            keypoint_cache[target_index],
            resolution=self.resolution,
            sigma=self.heatmap_sigma,
        )
        smplx_global = smplx_cache[target_index]
        if not torch.isfinite(smplx_global).all():
            raise ValueError("smplx_global must contain finite values")

        batch: dict[str, Any] = {
            "src_image": source_image.unsqueeze(0),
            "tgt_image": target_image.unsqueeze(0),
            "normal": normal.unsqueeze(0),
            "part_onehot": part_onehot.unsqueeze(0),
            "pose_heatmap": pose_heatmap.unsqueeze(0),
            "smplx_global": smplx_global.unsqueeze(0),
            "human_mask": human_mask.unsqueeze(0),
            "task_id": torch.zeros(1, dtype=torch.long),
            "appearance": [self.appearance],
            "source_stem": [self.source_stem],
            "target_stem": [self.target_stem],
        }
        hand_condition, hand_references = self._load_hand_payload(detail_conditions)
        return IPERV65OverfitInputs(
            batch=batch,
            hand_detail_condition=hand_condition,
            hand_detail_references=hand_references,
        )


__all__ = ["IPERV65FaceOverfitLoader", "IPERV65OverfitInputs"]
