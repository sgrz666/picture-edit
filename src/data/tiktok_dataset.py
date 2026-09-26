"""TikTok Dataset loader and single-pair overfit loader for DeepGen V6.

Provides:
1. `TikTokV65OverfitInputs`: Dataclass holding the batch dict, Hand conditions, and Face conditions.
2. `TikTokV65FaceOverfitLoader`: Deterministic single-pair loader for Sequence 00001 (or any sequence).
3. `TikTokPoseDataset`: PyTorch Dataset for paired sequence training.
4. `build_tiktok_face_inputs`: Helper to convert `v65_face_features.pt` into `FaceFineCondition` and `FaceReferenceFeatures`.
"""

from __future__ import annotations

import importlib.util
from importlib.machinery import ModuleSpec
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

if "diffusers" not in sys.modules:
    try:
        _diffusers_spec = importlib.util.find_spec("diffusers")
    except ValueError:
        _diffusers_spec = None
    if _diffusers_spec is None:
        diffusers = types.ModuleType("diffusers")
        diffusers_models = types.ModuleType("diffusers.models")
        diffusers.__spec__ = ModuleSpec("diffusers", loader=None)
        diffusers_models.__spec__ = ModuleSpec("diffusers.models", loader=None)

        class _SD3ControlNetModel(torch.nn.Module):
            pass

        diffusers_models.SD3ControlNetModel = _SD3ControlNetModel
        diffusers.models = diffusers_models
        sys.modules["diffusers"] = diffusers
        sys.modules["diffusers.models"] = diffusers_models

from src.pose_control.v6.detail.conditions import DetailReferenceBatch, FaceHandDetailCondition
from src.pose_control.v6.detail.hand import HandDetailCondition
from src.pose_control.v6.face.conditions import FaceFineCondition, FaceReferenceFeatures


@dataclass(frozen=True)
class TikTokV65OverfitInputs:
    """One batched TikTok pair plus Hand and Face condition payloads."""

    batch: Dict[str, Any]
    hand_detail_condition: HandDetailCondition
    hand_detail_references: DetailReferenceBatch
    face_condition: FaceFineCondition
    face_references: FaceReferenceFeatures


def _load_bundle(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(bundle, Mapping):
        raise ValueError(f"{label} must contain a mapping: {path}")
    return bundle


def _frame_index(bundle: Mapping[str, Any], stem: str, label: str) -> int:
    stem_to_idx = bundle.get("stem_to_idx")
    if isinstance(stem_to_idx, Mapping) and stem in stem_to_idx:
        return int(stem_to_idx[stem])
    stems = bundle.get("frame_stems") or bundle.get("stems")
    if isinstance(stems, Sequence):
        stems_str = [str(s) for s in stems]
        if stem in stems_str:
            return stems_str.index(stem)
    raise KeyError(f"Stem {stem!r} is absent from {label}")


def _tensor(bundle: Mapping[str, Any], name: str) -> torch.Tensor:
    value = bundle.get(name)
    if not torch.is_tensor(value):
        raise ValueError(f"Cache field {name!r} must be a tensor")
    return value.detach().cpu()


def _read_image(path: Path, mode: str = "RGB") -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Missing image file: {path}")
    with path.open("rb") as f:
        return Image.open(f).convert(mode)


def _rgb_tensor(path: Path, resolution: int, crop_box: Optional[Tuple[int, int, int, int]] = None) -> torch.Tensor:
    image = _read_image(path, "RGB")
    if crop_box is not None:
        image = image.crop(crop_box)
    image = image.resize((resolution, resolution), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).div_(127.5).sub_(1.0)


def _gray_tensor(path: Path, resolution: int, crop_box: Optional[Tuple[int, int, int, int]] = None) -> torch.Tensor:
    image = _read_image(path, "L")
    if crop_box is not None:
        image = image.crop(crop_box)
    image = image.resize((resolution, resolution), Image.Resampling.NEAREST)
    array = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(array).unsqueeze(0).div_(255.0)


def _part_onehot(path: Path, resolution: int, crop_box: Optional[Tuple[int, int, int, int]] = None) -> torch.Tensor:
    image = _read_image(path, "L")
    if crop_box is not None:
        image = image.crop(crop_box)
    image = image.resize((resolution, resolution), Image.Resampling.NEAREST)
    labels = torch.from_numpy(np.asarray(image, dtype=np.int64).copy())
    if torch.any((labels < 0) | (labels > 14)):
        raise ValueError(f"Part labels must lie in [0, 14]: {path}")
    return F.one_hot(labels, num_classes=15)[..., 1:].permute(2, 0, 1).float()


def _pose_heatmaps(
    keypoints: torch.Tensor,
    *,
    resolution: int,
    sigma: float,
) -> torch.Tensor:
    if tuple(keypoints.shape) != (25, 3):
        raise ValueError(f"kps_25_coords frame must have shape [25, 3], got {tuple(keypoints.shape)}")
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


def apply_tiktok_condition_override(
    batch: Dict[str, Any],
    path: str | Path,
    *,
    sequence: str,
    target_stem: str,
    resolution: int,
    heatmap_sigma: float,
) -> Dict[str, Any]:
    """Apply one audited target-frame condition without changing cached assets."""
    override = _load_bundle(Path(path), "TikTok condition override")
    if str(override.get("sequence")) != sequence:
        raise ValueError("condition override sequence does not match sample")
    if str(override.get("target_stem")) != target_stem:
        raise ValueError("condition override target_stem does not match sample")
    points = torch.as_tensor(override["kps_25_coords"], dtype=torch.float32)
    if points.shape != (25, 3) or not torch.isfinite(points).all():
        raise ValueError("override kps_25_coords must be finite [25,3]")
    updated = dict(batch)
    updated["pose_heatmap"] = _pose_heatmaps(
        points, resolution=resolution, sigma=heatmap_sigma
    ).unsqueeze(0)
    if "part_512" in override:
        labels = torch.as_tensor(override["part_512"])
        if labels.shape != (resolution, resolution) or labels.dtype != torch.uint8:
            raise ValueError("override part_512 must be uint8 [resolution,resolution]")
        if torch.any(labels > 14):
            raise ValueError("override part_512 labels must be in [0,14]")
        updated["part_onehot"] = F.one_hot(
            labels.long(), num_classes=15
        )[..., 1:].permute(2, 0, 1).float().unsqueeze(0)
    return updated


def build_tiktok_face_inputs(
    cache: Mapping[str, Any],
    *,
    source_frame: int | str,
    target_frame: int | str,
) -> Tuple[FaceFineCondition, FaceReferenceFeatures]:
    """Assemble one TikTok pair as [B=1, P=2, R=1] V6.5 face inputs."""
    source_stem = f"{int(source_frame):04d}" if isinstance(source_frame, int) else str(source_frame)
    target_stem = f"{int(target_frame):04d}" if isinstance(target_frame, int) else str(target_frame)

    source_index = _frame_index(cache, source_stem, "v65_face_features")
    target_index = _frame_index(cache, target_stem, "v65_face_features")
    slot_valid = bool(cache["valid"][source_index] and cache["valid"][target_index])

    landmarks = torch.zeros(1, 2, 72, 3, dtype=torch.float32)
    jaw = torch.zeros(1, 2, 3, dtype=torch.float32)
    expression = torch.zeros(1, 2, 10, dtype=torch.float32)
    source_boxes = torch.zeros(1, 2, 4, dtype=torch.float32)
    target_boxes = torch.zeros(1, 2, 4, dtype=torch.float32)
    face_valid = torch.zeros(1, 2, dtype=torch.bool)
    arcface = torch.zeros(1, 2, 1, 512, dtype=torch.float16)
    dino = torch.zeros(1, 2, 1, 256, 1536, dtype=torch.float16)
    reference_valid = torch.zeros(1, 2, 1, dtype=torch.bool)

    if slot_valid:
        landmarks[0, 0] = cache["face_landmarks"][target_index].float()
        jaw[0, 0] = cache["jaw_pose"][target_index].float()
        expression[0, 0] = cache["expression"][target_index].float()
        source_boxes[0, 0] = cache["face_boxes"][source_index].float()
        target_boxes[0, 0] = cache["face_boxes"][target_index].float()
        face_valid[0, 0] = True
        arcface[0, 0, 0] = cache["arcface"][source_index]
        dino[0, 0, 0] = cache["dino_patches"][source_index]
        reference_valid[0, 0, 0] = True

    condition = FaceFineCondition(
        landmarks=landmarks,
        jaw_pose=jaw,
        expression=expression,
        source_boxes=source_boxes,
        target_boxes=target_boxes,
        face_valid=face_valid,
        source_indices=torch.tensor([[0, 1]], dtype=torch.long),
    ).validate()

    references = FaceReferenceFeatures(
        arcface=arcface,
        dino_patches=dino,
        reference_valid=reference_valid,
    ).validate()

    return condition, references


class TikTokV65FaceOverfitLoader:
    """Deterministic single-pair loader for TikTok V6.5 Face overfitting."""

    def __init__(
        self,
        *,
        raw_root: str | Path,
        assets_root: str | Path,
        sequence: str = "00001",
        source_stem: str = "0014",
        target_stem: str = "0074",
        resolution: int = 512,
        heatmap_sigma: float = 4.0,
        align_mode: str = "resize",
        condition_override: str | Path | None = None,
    ) -> None:
        self.raw_root = Path(raw_root)
        self.assets_root = Path(assets_root)
        self.sequence = str(sequence)
        self.source_stem = str(source_stem)
        self.target_stem = str(target_stem)
        self.resolution = int(resolution)
        self.heatmap_sigma = float(heatmap_sigma)
        self.align_mode = str(align_mode)
        self.condition_override = None if condition_override is None else Path(condition_override)

    def _resolve_image_path(self, folder: Path, stem: str, exts: Sequence[str] = (".png", ".jpg")) -> Path:
        for ext in exts:
            p = folder / f"{stem}{ext}"
            if p.is_file():
                return p
        raise FileNotFoundError(f"Missing file for stem {stem} in {folder}")

    def load(self) -> TikTokV65OverfitInputs:
        seq_raw = self.raw_root / self.sequence
        seq_assets = self.assets_root / self.sequence

        global_bundle = _load_bundle(seq_assets / "v6_conditions.pt", "v6_conditions.pt")
        face_bundle = _load_bundle(seq_assets / "v65_face_features.pt", "v65_face_features.pt")

        target_idx = _frame_index(global_bundle, self.target_stem, "v6_conditions")
        source_idx = _frame_index(global_bundle, self.source_stem, "v6_conditions")
        smplx_cache = _tensor(global_bundle, "smplx_global").float()
        kps_cache = _tensor(global_bundle, "kps_25_coords").float()
        params = _load_bundle(seq_assets / "smplestx" / "params.pt", "SMPLest-X params")

        def project_hand(frame_index: int) -> torch.Tensor:
            joints = _tensor(params, "joint_proj_model")[frame_index]
            focal_key = "focal_length_xy_raw" if "focal_length_xy_raw" in params else "focal_length_xy"
            principal_key = "princpt_raw" if "princpt_raw" in params else "princpt"
            focal = _tensor(params, focal_key)[frame_index]
            principal = _tensor(params, principal_key)[frame_index]
            width = float(_tensor(params, "width").reshape(-1)[frame_index if _tensor(params, "width").numel() > 1 else 0])
            height = float(_tensor(params, "height").reshape(-1)[frame_index if _tensor(params, "height").numel() > 1 else 0])
            box_width = focal[0] * (192.0 / 5000.0)
            box_height = focal[1] * (256.0 / 5000.0)
            x = principal[0] - box_width * 0.5 + joints[25:65, 0] / 12.0 * box_width
            y = principal[1] - box_height * 0.5 + joints[25:65, 1] / 16.0 * box_height
            points = torch.stack((x / width, y / height), dim=-1).reshape(2, 20, 2)
            # SMPLest-X has 20 hand joints per side; HandDetailCondition uses 21.
            wrist = points[:, :1]
            points = torch.cat((wrist, points), dim=1)
            valid = ((points >= 0) & (points <= 1)).all(dim=-1).float()
            return torch.cat((points.clamp(0, 1), valid[..., None]), dim=-1)

        hand_source = project_hand(source_idx)
        hand_target = project_hand(target_idx)
        hand_keypoints = torch.zeros(1, 2, 2, 21, 3)
        hand_keypoints[:, 0] = hand_target
        hand_pose = torch.stack(
            (
                _tensor(params, "left_hand_pose")[[target_idx, target_idx]],
                _tensor(params, "right_hand_pose")[[target_idx, target_idx]],
            ), dim=1
        ).unsqueeze(0)
        hand_pose[:, 1] = 0

        def boxes(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            output = torch.zeros(2, 2, 4)
            valid = torch.zeros(2, 2, dtype=torch.bool)
            for person in range(2):
                for side in range(2):
                    visible = points[person, side, :, 2] > 0.05
                    if int(visible.sum()) < 2:
                        continue
                    xy = points[person, side, visible, :2]
                    lo, hi = xy.amin(0), xy.amax(0)
                    pad = (hi - lo).clamp_min(1e-3) * 0.2 + 0.01
                    box = torch.cat(((lo - pad).clamp(0, 1), (hi + pad).clamp(0, 1)))
                    output[person, side] = box
                    valid[person, side] = bool((box[2:] > box[:2]).all())
            return output, valid

        source_points = torch.zeros(2, 2, 21, 3)
        source_points[0] = hand_source
        source_boxes, source_valid = boxes(source_points)
        target_boxes, target_valid = boxes(hand_keypoints[0])
        hand_cond = HandDetailCondition(
            hand_keypoints=hand_keypoints,
            hand_pose=hand_pose,
            source_boxes=source_boxes.unsqueeze(0),
            target_boxes=target_boxes.unsqueeze(0),
            region_valid=(source_valid & target_valid).unsqueeze(0),
            source_indices=torch.tensor([[0, 1]], dtype=torch.long),
        ).validate()

        crop_box = None
        if self.align_mode == "square_crop":
            # 604x1080 -> 604x604 center crop in Y
            crop_box = (0, 100, 604, 704)

        source_img_p = self._resolve_image_path(seq_raw / "images", self.source_stem)
        target_img_p = self._resolve_image_path(seq_raw / "images", self.target_stem)
        normal_p = self._resolve_image_path(seq_assets / "normal", self.target_stem)
        mask_p = self._resolve_image_path(seq_raw / "masks", self.target_stem)
        source_image = _rgb_tensor(source_img_p, self.resolution, crop_box=crop_box)
        target_image = _rgb_tensor(target_img_p, self.resolution, crop_box=crop_box)
        human_mask = _gray_tensor(mask_p, self.resolution, crop_box=crop_box)
        normal = _rgb_tensor(normal_p, self.resolution, crop_box=crop_box) * human_mask

        try:
            part_p = self._resolve_image_path(seq_assets / "part_512", self.target_stem)
            part_onehot = _part_onehot(part_p, self.resolution, crop_box=crop_box)
        except (FileNotFoundError, OSError):
            part_onehot = torch.zeros(14, self.resolution, self.resolution, dtype=torch.float32)
            part_onehot[1] = (human_mask[0] > 0.5).float()

        pose_heatmap = _pose_heatmaps(
            kps_cache[target_idx],
            resolution=self.resolution,
            sigma=self.heatmap_sigma,
        )
        smplx_global = smplx_cache[target_idx]

        batch: Dict[str, Any] = {
            "src_image": source_image.unsqueeze(0),
            "tgt_image": target_image.unsqueeze(0),
            "normal": normal.unsqueeze(0),
            "part_onehot": part_onehot.unsqueeze(0),
            "pose_heatmap": pose_heatmap.unsqueeze(0),
            "smplx_global": smplx_global.unsqueeze(0),
            "human_mask": human_mask.unsqueeze(0),
            "task_id": torch.zeros(1, dtype=torch.long),
            "appearance": [self.sequence],
            "source_stem": [self.source_stem],
            "target_stem": [self.target_stem],
        }
        if self.condition_override is not None:
            batch = apply_tiktok_condition_override(
                batch,
                self.condition_override,
                sequence=self.sequence,
                target_stem=self.target_stem,
                resolution=self.resolution,
                heatmap_sigma=self.heatmap_sigma,
            )

        # Build Face Fine Condition & Face References
        face_cond, face_refs = build_tiktok_face_inputs(
            face_bundle,
            source_frame=self.source_stem,
            target_frame=self.target_stem,
        )

        hand_refs = DetailReferenceBatch(
            images=torch.zeros(1, 2, 3, 1, 3, 224, 224, dtype=torch.float32),
            reference_valid=torch.zeros(1, 2, 3, 1, dtype=torch.bool),
        ).validate()

        return TikTokV65OverfitInputs(
            batch=batch,
            hand_detail_condition=hand_cond,
            hand_detail_references=hand_refs,
            face_condition=face_cond,
            face_references=face_refs,
        )


class TikTokPoseDataset(Dataset):
    """PyTorch Dataset for TikTok paired video sequences."""

    def __init__(
        self,
        raw_root: str | Path,
        assets_root: str | Path,
        sequences: Optional[Sequence[str]] = None,
        resolution: int = 512,
        heatmap_sigma: float = 4.0,
        frame_interval: int = 15,
    ) -> None:
        self.raw_root = Path(raw_root)
        self.assets_root = Path(assets_root)
        self.resolution = int(resolution)
        self.heatmap_sigma = float(heatmap_sigma)
        self.frame_interval = max(1, int(frame_interval))

        if sequences is not None:
            self.sequences = list(sequences)
        else:
            self.sequences = sorted([d.name for d in self.assets_root.iterdir() if d.is_dir() and not d.name.startswith(".")])

        # Build pair index (seq, src_stem, tgt_stem)
        self.pairs: List[Tuple[str, str, str]] = []
        for seq in self.sequences:
            seq_assets = self.assets_root / seq
            v6_cond = seq_assets / "v6_conditions.pt"
            if not v6_cond.is_file():
                continue
            bundle = torch.load(v6_cond, map_location="cpu", weights_only=False)
            stems = bundle.get("stems", [])
            n = len(stems)
            if n < 2:
                continue
            # Self-driven pairs
            for i in range(0, n - self.frame_interval, self.frame_interval):
                self.pairs.append((seq, stems[i], stems[i + self.frame_interval]))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seq, src_stem, tgt_stem = self.pairs[idx]
        loader = TikTokV65FaceOverfitLoader(
            raw_root=self.raw_root,
            assets_root=self.assets_root,
            sequence=seq,
            source_stem=src_stem,
            target_stem=tgt_stem,
            resolution=self.resolution,
            heatmap_sigma=self.heatmap_sigma,
        )
        inputs = loader.load()
        item = {k: v.squeeze(0) if torch.is_tensor(v) else v[0] for k, v in inputs.batch.items()}
        return item


__all__ = [
    "TikTokV65OverfitInputs",
    "TikTokV65FaceOverfitLoader",
    "TikTokPoseDataset",
    "build_tiktok_face_inputs",
]
