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


# The margin is filled differently for RGB and for the condition maps today: images get
# mid-grey, geometry rasters get black. This is a legacy state, not something to unify quietly —
# research/canvas_native_portrait_acceptance_20261022/probe_padding_attention.py measured that the
# margin value moves *valid*-token DiT outputs by ~4x their signal, so changing either constant
# changes what every existing checkpoint was trained against. Pinned and asserted in
# tests/test_geometry_pair_dataset_v2.py so any future unification is deliberate.
RGB_PAD_VALUE = (128, 128, 128)
CONDITION_PAD_VALUE = 0


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


def _part_onehot(
    path: Path,
    resolution: int,
    crop_box: Optional[Tuple[int, int, int, int]] = None,
    native_size: Tuple[int, int] = (604, 1080),
) -> torch.Tensor:
    image = _read_image(path, "L")
    if crop_box is not None:
        if image.size != native_size and native_size[0] > 0 and native_size[1] > 0:
            sx = image.size[0] / float(native_size[0])
            sy = image.size[1] / float(native_size[1])
            scaled_box = (
                int(round(crop_box[0] * sx)),
                int(round(crop_box[1] * sy)),
                int(round(crop_box[2] * sx)),
                int(round(crop_box[3] * sy)),
            )
            image = image.crop(scaled_box)
        else:
            image = image.crop(crop_box)
    image = image.resize((resolution, resolution), Image.Resampling.NEAREST)
    labels = torch.from_numpy(np.asarray(image, dtype=np.int64).copy())
    if torch.any((labels < 0) | (labels > 14)):
        raise ValueError(f"Part labels must lie in [0, 14]: {path}")
    return F.one_hot(labels, num_classes=15)[..., 1:].permute(2, 0, 1).float()


def transform_keypoints_for_crop(
    keypoints: torch.Tensor,
    norm_crop: Optional[Tuple[float, float, float, float]] = None,
) -> torch.Tensor:
    """Remap normalized [25, 3] keypoints into a normalized crop box (u0, v0, u1, v1)."""
    if norm_crop is None or norm_crop == (0.0, 0.0, 1.0, 1.0):
        return keypoints
    u0, v0, u1, v1 = norm_crop
    du = max(float(u1 - u0), 1e-6)
    dv = max(float(v1 - v0), 1e-6)
    pts = keypoints.clone().float()
    x = (pts[:, 0] - float(u0)) / du
    y = (pts[:, 1] - float(v0)) / dv
    inside = (pts[:, 2] > 0.0) & (x >= 0.0) & (x <= 1.0) & (y >= 0.0) & (y <= 1.0)
    conf = torch.where(inside, pts[:, 2], torch.zeros_like(pts[:, 2]))
    return torch.stack((x.clamp(0.0, 1.0), y.clamp(0.0, 1.0), conf), dim=-1)


def transform_box_for_crop(
    box: torch.Tensor,
    norm_crop: Optional[Tuple[float, float, float, float]] = None,
) -> torch.Tensor:
    """Remap normalized [4] xyxy box into a normalized crop box (u0, v0, u1, v1)."""
    b = box.clone().float()
    if norm_crop is None or norm_crop == (0.0, 0.0, 1.0, 1.0):
        return b
    u0, v0, u1, v1 = norm_crop
    du = max(float(u1 - u0), 1e-6)
    dv = max(float(v1 - v0), 1e-6)
    out = torch.tensor(
        [
            (float(b[0]) - float(u0)) / du,
            (float(b[1]) - float(v0)) / dv,
            (float(b[2]) - float(u0)) / du,
            (float(b[3]) - float(v0)) / dv,
        ],
        dtype=torch.float32,
    ).clamp(0.0, 1.0)
    valid = bool((out[2] > out[0] + 1e-3) and (out[3] > out[1] + 1e-3))
    if not valid:
        out.zero_()
    return out


def _pose_heatmaps(
    keypoints: torch.Tensor,
    *,
    resolution: int | tuple[int, int] | Sequence[int],
    sigma: float,
) -> torch.Tensor:
    if tuple(keypoints.shape) != (25, 3):
        raise ValueError(f"kps_25_coords frame must have shape [25, 3], got {tuple(keypoints.shape)}")
    keypoints = keypoints.float()
    if not torch.isfinite(keypoints).all():
        raise ValueError("kps_25_coords must contain finite values")
    h, w = (resolution, resolution) if isinstance(resolution, int) else (resolution[0], resolution[1])
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    x = keypoints[:, 0, None, None] * w
    y = keypoints[:, 1, None, None] * h
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
    norm_crop: Optional[Tuple[float, float, float, float]] = None,
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
        src_box = transform_box_for_crop(cache["face_boxes"][source_index].float(), norm_crop)
        tgt_box = transform_box_for_crop(cache["face_boxes"][target_index].float(), norm_crop)
        src_ok = bool((src_box[2] > src_box[0] + 1e-3) and (src_box[3] > src_box[1] + 1e-3))
        tgt_ok = bool((tgt_box[2] > tgt_box[0] + 1e-3) and (tgt_box[3] > tgt_box[1] + 1e-3))
        if src_ok and tgt_ok:
            lmk = cache["face_landmarks"][target_index].float().clone()
            if norm_crop is not None and norm_crop != (0.0, 0.0, 1.0, 1.0):
                u0, v0, u1, v1 = norm_crop
                lmk[:, 0] = ((lmk[:, 0] - float(u0)) / max(float(u1 - u0), 1e-6)).clamp(0.0, 1.0)
                lmk[:, 1] = ((lmk[:, 1] - float(v0)) / max(float(v1 - v0), 1e-6)).clamp(0.0, 1.0)
            landmarks[0, 0] = lmk
            jaw[0, 0] = cache["jaw_pose"][target_index].float()
            expression[0, 0] = cache["expression"][target_index].float()
            source_boxes[0, 0] = src_box
            target_boxes[0, 0] = tgt_box
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

        params_path = seq_assets / "smplestx" / "params.pt"
        if params_path.is_file():
            params = _load_bundle(params_path, "SMPLest-X params")

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
        else:
            hand_cond = HandDetailCondition(
                hand_keypoints=torch.zeros(1, 2, 2, 21, 3, dtype=torch.float32),
                hand_pose=torch.zeros(1, 2, 2, 45, dtype=torch.float32),
                source_boxes=torch.zeros(1, 2, 2, 4, dtype=torch.float32),
                target_boxes=torch.zeros(1, 2, 2, 4, dtype=torch.float32),
                region_valid=torch.zeros(1, 2, 2, dtype=torch.bool),
                source_indices=torch.tensor([[0, 1]], dtype=torch.long),
            ).validate()

        crop_box = None
        norm_crop = (0.0, 0.0, 1.0, 1.0)
        if self.align_mode == "square_crop":
            # 604x1080 -> 604x604 center crop in Y
            crop_box = (0, 100, 604, 704)
            norm_crop = (0.0, 100.0 / 1080.0, 1.0, 704.0 / 1080.0)

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

        kps_target = transform_keypoints_for_crop(kps_cache[target_idx], norm_crop)
        pose_heatmap = _pose_heatmaps(
            kps_target,
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
            norm_crop=norm_crop,
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


class TikTokV3PoseDataset(Dataset):
    """PyTorch Dataset for TikTok_v3 pair manifests (pairs_strict_v1 / pairs).

    Matches the exact sample and batch contract of :class:`IPERPoseDataset` so
    ``Qwen21BodyAdapter``, ``Qwen21FaceAdapter``, ``Qwen21IPERCache``, and
    ``FaceFeatureStore`` work seamlessly on TikTok_v3.
    """

    def __init__(
        self,
        pairs_jsonl: str | Path,
        raw_root: str | Path = "/home/shangguanrz/project/pic-edit/datasets/TikTokDataset/TikTok_dataset/TikTok_dataset",
        assets_root: str | Path = "/home/shangguanrz/project/pic-edit/datasets/TikTokDataset/TikTok_3d_assets_native",
        dwpose_root: str | Path = "/home/shangguanrz/project/pic-edit/datasets/TikTokDataset/TikTok_v3/dwpose_v1",
        resolution: int = 512,
        heatmap_sigma: float = 4.0,
        align_mode: str | None = None,
        max_pixels: int | None = None,
        augment: bool = False,
        strict_conditions: bool = False,
    ) -> None:
        self.pairs_jsonl = Path(pairs_jsonl)
        self.raw_root = Path(raw_root)
        self.assets_root = Path(assets_root)
        self.dwpose_root = Path(dwpose_root)
        self.sampled_root = ""
        self.resolution = resolution if isinstance(resolution, str) else int(resolution)
        self.heatmap_sigma = float(heatmap_sigma)
        if align_mode is None:
            raise ValueError(
                "align_mode must be chosen explicitly: 'native', 'aspect_preserving' and 'letterbox' "
                "keep body proportions, while 'resize' stretches them and 'square_crop' crops the "
                "field of view")
        self.align_mode = str(align_mode)
        self.max_pixels = None if max_pixels is None else int(max_pixels)
        if self.align_mode == "aspect_preserving" and not self.max_pixels:
            raise ValueError("aspect_preserving needs max_pixels: the budget is what keeps a "
                             "3000x5000 portrait trainable without stretching it")
        self.augment = bool(augment)
        self.strict_conditions = bool(strict_conditions)
        self._v6_cache: Dict[str, Mapping[str, Any]] = {}

        if self.align_mode not in ("resize", "square_crop", "letterbox", "native", "aspect_preserving"):
            raise ValueError(f"Unsupported align_mode: {self.align_mode!r}")

        self.samples: List[Dict[str, Any]] = []
        with self.pairs_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                seq_id = str(row.get("sequence_id") or row.get("appearance"))
                src_stem = str(row["source_stem"] if "source_stem" in row else row["source"]["stem"])
                tgt_stem = str(row["target_stem"] if "target_stem" in row else row["target"]["stem"])

                src_rgb = self._resolve_manifest_file(
                    row.get("source_rgb"),
                    self.raw_root / seq_id / "images",
                    src_stem,
                )
                tgt_rgb = self._resolve_manifest_file(
                    row.get("target_rgb"),
                    self.raw_root / seq_id / "images",
                    tgt_stem,
                )
                tgt_mask = self._resolve_manifest_file(
                    row.get("target_person_mask"),
                    self.raw_root / seq_id / "masks",
                    tgt_stem,
                )
                tgt_normal = self._resolve_manifest_file(
                    row.get("target_normal"),
                    self.assets_root / seq_id / "normal",
                    tgt_stem,
                )
                tgt_depth = self._resolve_manifest_file(
                    row.get("target_depth_vis"),
                    self.assets_root / seq_id / "depth_vis",
                    tgt_stem,
                    required=False,
                )
                tgt_dwpose = self._resolve_manifest_file(
                    row.get("target_dwpose"),
                    self.dwpose_root / seq_id / "dwpose",
                    tgt_stem,
                    required=False,
                )

                normalized = dict(row)
                normalized["sequence_id"] = seq_id
                normalized["appearance"] = seq_id
                normalized["source_stem"] = src_stem
                normalized["target_stem"] = tgt_stem
                normalized["source_rgb"] = str(src_rgb)
                normalized["target_rgb"] = str(tgt_rgb)
                normalized["target_person_mask"] = str(tgt_mask)
                normalized["target_normal"] = str(tgt_normal)
                normalized["target_depth_vis"] = str(tgt_depth) if tgt_depth else None
                normalized["target_dwpose"] = str(tgt_dwpose) if tgt_dwpose else None
                _, default_norm_crop = self._crop_params((604, 1080))
                normalized["norm_crop"] = default_norm_crop
                # Compatibility with IPERPoseDataset sample dict access in precompute & FaceFeatureStore
                normalized["source"] = {"stem": src_stem, "rgb_1024": str(src_rgb)}
                normalized["target"] = {"stem": tgt_stem, "rgb_1024": str(tgt_rgb), "role": "target"}
                self.samples.append(normalized)

    @staticmethod
    def _resolve_manifest_file(
        manifest_path: Optional[str],
        fallback_dir: Path,
        stem: str,
        *,
        exts: Sequence[str] = (".png", ".jpg"),
        required: bool = True,
    ) -> Optional[Path]:
        if manifest_path:
            return Path(manifest_path)
        for ext in exts:
            candidate = fallback_dir / f"{stem}{ext}"
            if candidate.is_file():
                return candidate
        if required:
            return fallback_dir / f"{stem}.png"
        return None

    def _crop_params(self, size: Tuple[int, int] = (604, 1080)) -> Tuple[Optional[Tuple[int, int, int, int]], Tuple[float, float, float, float]]:
        if self.align_mode == "square_crop":
            w, h = size
            side = min(w, h)
            top = int(round(100.0 * h / 1080.0)) if h > w else 0
            top = max(0, min(h - side, top))
            left = max(0, (w - side) // 2)
            crop_box = (left, top, left + side, top + side)
            norm_crop = (left / float(w), top / float(h), (left + side) / float(w), (top + side) / float(h))
            return crop_box, norm_crop
        return None, (0.0, 0.0, 1.0, 1.0)

    def _load_image(self, path: str | Path) -> Image.Image:
        """Load RGB image and apply alignment crop (compatible with train_iper_qwen21_body.precompute)."""
        image = _read_image(Path(path), "RGB")
        crop_box, _ = self._crop_params(image.size)
        if crop_box is not None:
            image = image.crop(crop_box)
        return image

    def rgb_transform(self, image: Image.Image) -> torch.Tensor:
        """Put RGB on the canvas through the shared contract, then normalise to [-1, 1]."""
        from .canvas_contract import canvas_for_mode
        if self.align_mode in ('native', 'letterbox', 'aspect_preserving'):
            canvas = canvas_for_mode(self.align_mode, (image.height, image.width),
                                     resolution=self.resolution, max_pixels=self.max_pixels)
            resized = canvas.image(image, fill=RGB_PAD_VALUE)
        elif self.align_mode == 'square_crop':
            # _load_image already cropped to a square, so this resize stays isotropic; the
            # defect this mode retains is lost field of view, not stretched limbs.
            resized = image.resize((self.resolution, self.resolution), Image.Resampling.BILINEAR)
        else:
            # 'resize': 604x1080 -> 512x512 widens the body by 1080/604. Kept only so
            # historical runs reproduce; require_canonical refuses it for new training.
            resized = image.resize((self.resolution, self.resolution), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32).copy()
        return torch.from_numpy(array).permute(2, 0, 1).div_(127.5).sub_(1.0)

    def _get_v6_bundle(self, seq_id: str) -> Optional[Mapping[str, Any]]:
        if seq_id not in self._v6_cache:
            path = self.assets_root / seq_id / "v6_conditions.pt"
            if path.is_file():
                self._v6_cache[seq_id] = _load_bundle(path, "v6_conditions.pt")
            else:
                return None
        return self._v6_cache[seq_id]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        if self.align_mode == 'letterbox':
            return self._letterbox_item(sample)
        if self.align_mode == 'native':
            return self._native_item(sample)
        if self.align_mode == 'aspect_preserving':
            return self._budget_item(sample)
        seq_id = sample["sequence_id"]
        src_stem = sample["source_stem"]
        tgt_stem = sample["target_stem"]


        src_pil = self._load_image(sample["source_rgb"])
        tgt_pil = self._load_image(sample["target_rgb"])
        src_image = self.rgb_transform(src_pil)
        tgt_image = self.rgb_transform(tgt_pil)

        native_img = _read_image(Path(sample["target_rgb"]), "RGB")
        crop_box, norm_crop = self._crop_params(native_img.size)

        human_mask = _gray_tensor(Path(sample["target_person_mask"]), self.resolution, crop_box=crop_box)
        normal_rgb = _rgb_tensor(Path(sample["target_normal"]), self.resolution, crop_box=crop_box)
        normal = normal_rgb * human_mask
        normal_raw = (normal_rgb + 1.0) * 0.5

        if sample.get("target_depth_vis") and Path(sample["target_depth_vis"]).is_file():
            depth = _gray_tensor(Path(sample["target_depth_vis"]), self.resolution, crop_box=crop_box)
        else:
            depth = torch.zeros(1, self.resolution, self.resolution, dtype=torch.float32)

        if sample.get("target_dwpose") and Path(sample["target_dwpose"]).is_file():
            dwpose = (_rgb_tensor(Path(sample["target_dwpose"]), self.resolution, crop_box=crop_box) + 1.0) * 0.5
        else:
            dwpose = torch.zeros(3, self.resolution, self.resolution, dtype=torch.float32)

        part_dir = self.assets_root / seq_id / "part_512"
        part_path = None
        for ext in (".png", ".jpg"):
            candidate = part_dir / f"{tgt_stem}{ext}"
            if candidate.is_file():
                part_path = candidate
                break
        if part_path is not None:
            part_onehot = _part_onehot(part_path, self.resolution, crop_box=crop_box, native_size=native_img.size)
        else:
            if self.strict_conditions:
                raise FileNotFoundError(f"Missing target part condition: {part_dir}/{tgt_stem}.png")
            part_onehot = torch.zeros(14, self.resolution, self.resolution, dtype=torch.float32)
            part_onehot[1] = (human_mask[0] > 0.5).float()

        v6_bundle = self._get_v6_bundle(seq_id)
        if v6_bundle is not None and tgt_stem in v6_bundle.get("stem_to_idx", {}):
            frame_idx = int(v6_bundle["stem_to_idx"][tgt_stem])
            smplx_global = _tensor(v6_bundle, "smplx_global")[frame_idx].float().clone()
            kps_raw = _tensor(v6_bundle, "kps_25_coords")[frame_idx].float().clone()
        else:
            if self.strict_conditions:
                raise ValueError(f"Missing v6 condition for {seq_id}/{tgt_stem}")
            smplx_global = torch.zeros(26, dtype=torch.float32)
            kps_raw = torch.zeros(25, 3, dtype=torch.float32)

        if self.strict_conditions:
            if smplx_global.shape != (26,) or not torch.isfinite(smplx_global).all():
                raise ValueError(f"SMPL-X condition must be finite [26]: {seq_id}/{tgt_stem}")
            if kps_raw.shape != (25, 3) or not torch.isfinite(kps_raw).all():
                raise ValueError(f"Body keypoint condition must be finite [25,3]: {seq_id}/{tgt_stem}")
            visible = (kps_raw[:, 2] > .05) & ((kps_raw[:, :2] >= 0) & (kps_raw[:, :2] <= 1)).all(-1)
            if int(visible.sum()) < 8:
                raise ValueError(f"Body condition has fewer than 8 visible keypoints: {seq_id}/{tgt_stem}")
            if not bool((part_onehot.sum() > 0) & (human_mask.sum() > 0)):
                raise ValueError(f"Empty target part/person condition: {seq_id}/{tgt_stem}")

        kps_aligned = transform_keypoints_for_crop(kps_raw, norm_crop)
        pose_heatmap = _pose_heatmaps(kps_aligned, resolution=self.resolution, sigma=self.heatmap_sigma)
        control_map = torch.cat([normal_raw, depth, dwpose, human_mask], dim=0)

        return {
            "src_image": src_image,
            "tgt_image": tgt_image,
            "control_map": control_map,
            "normal": normal,
            "depth": depth,
            "part_onehot": part_onehot,
            "pose_heatmap": pose_heatmap,
            "smplx_global": smplx_global,
            "human_mask": human_mask,
            "task_id": torch.tensor(0, dtype=torch.long),
            "appearance": seq_id,
            "source_stem": src_stem,
            "target_stem": tgt_stem,
        }

    def _letterbox_item(self, sample):
        from .letterbox import Letterbox
        source = _read_image(Path(sample['source_rgb']), 'RGB')
        target = _read_image(Path(sample['target_rgb']), 'RGB')
        a = Letterbox.from_size(source.size, self.resolution)
        b = Letterbox.from_size(target.size, self.resolution)
        seq, stem = sample['sequence_id'], sample['target_stem']
        bundle = self._get_v6_bundle(seq)
        if bundle is None or stem not in bundle.get('stem_to_idx', {}):
            raise ValueError(f'Missing letterbox geometry: {seq}/{stem}')
        index = int(bundle['stem_to_idx'][stem])
        vector = _tensor(bundle,'smplx_global')[index].float().clone()
        points = _tensor(bundle,'kps_25_coords')[index].float().clone()
        if vector.shape != (26,) or points.shape != (25,3) or not torch.isfinite(vector).all() or not torch.isfinite(points).all():
            raise ValueError(f'Invalid letterbox geometry: {seq}/{stem}')
        visible = (points[:,2]>.05) & ((points[:,:2]>=0)&(points[:,:2]<=1)).all(-1)
        if int(visible.sum()) < 8:
            raise ValueError(f'Body condition has fewer than 8 visible keypoints: {seq}/{stem}')
        def raster(path, mode, *, nearest=False):
            image = b.image(_read_image(Path(path),mode), nearest=nearest, fill=CONDITION_PAD_VALUE)
            array = np.asarray(image,dtype=np.float32).copy()
            tensor = torch.from_numpy(array).unsqueeze(0) if mode=='L' else torch.from_numpy(array).permute(2,0,1)
            return tensor/255
        mask = raster(sample['target_person_mask'],'L',nearest=True)
        raw_normal = raster(sample['target_normal'],'RGB')
        normal = (raw_normal*2-1)*mask
        part_path = self.assets_root/seq/'part_512'/f'{stem}.png'
        if not part_path.is_file():
            part_path = part_path.with_suffix('.jpg')
        labels = torch.from_numpy(np.asarray(b.image(_read_image(part_path,'L'),nearest=True),dtype=np.int64).copy())
        if (labels<0).any() or (labels>14).any():
            raise ValueError(f'Invalid part labels: {part_path}')
        part = F.one_hot(labels,num_classes=15)[...,1:].permute(2,0,1).float()
        if not bool((mask.sum()>0)&(part.sum()>0)):
            raise ValueError(f'Empty body mask/part: {seq}/{stem}')
        depth = raster(sample['target_depth_vis'],'L') if sample.get('target_depth_vis') else torch.zeros_like(mask)
        dwpose = raster(sample['target_dwpose'],'RGB') if sample.get('target_dwpose') else torch.zeros(3,self.resolution,self.resolution)
        points = b.points(points)
        points[:,2] = torch.where(visible,points[:,2],torch.zeros_like(points[:,2]))
        valid = b.valid_mask()
        return {'src_image': self.rgb_transform(source), 'tgt_image': self.rgb_transform(target),
                'control_map': torch.cat([raw_normal,depth,dwpose,mask]), 'normal':normal,'depth':depth,
                'part_onehot':part,'pose_heatmap':_pose_heatmaps(points,resolution=self.resolution,sigma=self.heatmap_sigma)*valid,
                'smplx_global':b.global_camera(vector),'human_mask':mask,'valid_image_mask':valid,
                'task_id':torch.tensor(0,dtype=torch.long),'appearance':seq,
                'source_stem':sample['source_stem'],'target_stem':stem,
                'alignment_meta':json.dumps({'source':a.metadata(),'target':b.metadata()},sort_keys=True)}

    def _native_item(self, sample):
        from .letterbox import NativePad
        source = _read_image(Path(sample['source_rgb']), 'RGB')
        target = _read_image(Path(sample['target_rgb']), 'RGB')
        a = NativePad.from_size(source.size)
        b = NativePad.from_size(target.size)
        seq, stem = sample['sequence_id'], sample['target_stem']
        bundle = self._get_v6_bundle(seq)
        if bundle is None or stem not in bundle.get('stem_to_idx', {}):
            raise ValueError(f'Missing native geometry: {seq}/{stem}')
        index = int(bundle['stem_to_idx'][stem])
        vector = _tensor(bundle, 'smplx_global')[index].float().clone()
        points = _tensor(bundle, 'kps_25_coords')[index].float().clone()
        if vector.shape != (26,) or points.shape != (25, 3) or not torch.isfinite(vector).all() or not torch.isfinite(points).all():
            raise ValueError(f'Invalid native geometry: {seq}/{stem}')
        visible = (points[:, 2] > .05) & ((points[:, :2] >= 0) & (points[:, :2] <= 1)).all(-1)
        if int(visible.sum()) < 8:
            raise ValueError(f'Body condition has fewer than 8 visible keypoints: {seq}/{stem}')
        def raster(path, mode, *, nearest=False):
            img = _read_image(Path(path), mode)
            if img.size != (b.width, b.height):
                resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
                img = img.resize((b.width, b.height), resample)
            padded = b.image(img, nearest=nearest, fill=CONDITION_PAD_VALUE)
            array = np.asarray(padded, dtype=np.float32).copy()
            tensor = torch.from_numpy(array).unsqueeze(0) if mode == 'L' else torch.from_numpy(array).permute(2, 0, 1)
            return tensor / 255
        mask = raster(sample['target_person_mask'], 'L', nearest=True)
        raw_normal = raster(sample['target_normal'], 'RGB')
        normal = (raw_normal * 2 - 1) * mask
        part_path = self.assets_root / seq / 'part_512' / f'{stem}.png'
        if not part_path.is_file():
            part_path = part_path.with_suffix('.jpg')
        part_img = _read_image(part_path, 'L')
        if part_img.size != (b.width, b.height):
            part_img = part_img.resize((b.width, b.height), Image.Resampling.NEAREST)
        labels = torch.from_numpy(np.asarray(b.image(part_img, nearest=True), dtype=np.int64).copy())
        if (labels < 0).any() or (labels > 14).any():
            raise ValueError(f'Invalid part labels: {part_path}')
        part = F.one_hot(labels, num_classes=15)[..., 1:].permute(2, 0, 1).float()
        if not bool((mask.sum() > 0) & (part.sum() > 0)):
            raise ValueError(f'Empty body mask/part: {seq}/{stem}')
        depth = raster(sample['target_depth_vis'], 'L') if sample.get('target_depth_vis') and Path(sample['target_depth_vis']).is_file() else torch.zeros_like(mask)
        dwpose = raster(sample['target_dwpose'], 'RGB') if sample.get('target_dwpose') and Path(sample['target_dwpose']).is_file() else torch.zeros(3, b.padded_height, b.padded_width)
        points = b.points(points)
        points[:, 2] = torch.where(visible, points[:, 2], torch.zeros_like(points[:, 2]))
        valid = b.valid_mask()
        # CanvasSpec records describe this pair's geometry for the downstream contract. NativePad
        # produced the pixels and is proven pixel-identical to native_pad() in
        # tests/test_canvas_contract.py, so the two records cannot disagree about the canvas.
        from .canvas_contract import native_pad
        from .geometry_pair_dataset_v2 import assert_pair_contract
        source_spec = native_pad((source.height, source.width))
        target_spec = native_pad((target.height, target.width))
        remapped_camera = b.global_camera(vector)
        has_depth = bool(sample.get('target_depth_vis')) and Path(sample['target_depth_vis']).is_file()
        pair = {
            'src_image': self.rgb_transform(source),
            'tgt_image': self.rgb_transform(target),
            'control_map': torch.cat([raw_normal, depth, dwpose, mask]),
            'normal': normal,
            'depth': depth,
            'part_onehot': part,
            'pose_heatmap': _pose_heatmaps(points, resolution=(b.padded_height, b.padded_width), sigma=self.heatmap_sigma) * valid,
            'smplx_global': remapped_camera,
            'human_mask': mask,
            'valid_image_mask': valid,
            'task_id': torch.tensor(0, dtype=torch.long),
            'appearance': seq,
            'source_stem': sample['source_stem'],
            'target_stem': stem,
            'source_canvas': source_spec.metadata(),
            'target_canvas': target_spec.metadata(),
            'source_token_hw': source_spec.token_hw,
            'target_token_hw': target_spec.token_hw,
            'camera_metadata': {
                'convention': 'legacy 26D SMPL-X global; normalised fx,fy,cx,cy at indices 19..22',
                'original_smplx_global': vector,
                'canvas_smplx_global': remapped_camera},
            # depth_vis is a normalised visualisation, never metric depth (plan §6).
            'depth_metadata': {'metric': False, 'source': 'depth_vis' if has_depth else 'absent'},
            'alignment_meta': json.dumps({'source': a.metadata(), 'target': b.metadata()}, sort_keys=True)
        }
        assert_pair_contract(pair, where='TikTokV3PoseDataset native pair')
        return pair

    def _budget_item(self, sample):
        """Shrink both axes by one factor to a pixel budget, then pad (the plan's §4 default).

        Kept separate from `_native_item` on purpose: the native path also feeds legacy Face
        records whose field names assume NativePad's no-scaling transform, and a downscaled
        canvas cannot be expressed there. Merging the two would make those records lie.
        """
        from .canvas_contract import aspect_preserving
        from .geometry_pair_dataset_v2 import assert_pair_contract
        source = _read_image(Path(sample['source_rgb']), 'RGB')
        target = _read_image(Path(sample['target_rgb']), 'RGB')
        a = aspect_preserving((source.height, source.width), max_pixels=self.max_pixels)
        b = aspect_preserving((target.height, target.width), max_pixels=self.max_pixels)
        seq, stem = sample['sequence_id'], sample['target_stem']
        bundle = self._get_v6_bundle(seq)
        if bundle is None or stem not in bundle.get('stem_to_idx', {}):
            raise ValueError(f'Missing budget geometry: {seq}/{stem}')
        index = int(bundle['stem_to_idx'][stem])
        vector = _tensor(bundle, 'smplx_global')[index].float().clone()
        points = _tensor(bundle, 'kps_25_coords')[index].float().clone()
        if (vector.shape != (26,) or points.shape != (25, 3)
                or not torch.isfinite(vector).all() or not torch.isfinite(points).all()):
            raise ValueError(f'Invalid budget geometry: {seq}/{stem}')
        visible = (points[:, 2] > .05) & ((points[:, :2] >= 0) & (points[:, :2] <= 1)).all(-1)
        if int(visible.sum()) < 8:
            raise ValueError(f'Body condition has fewer than 8 visible keypoints: {seq}/{stem}')
        frame = (b.original_hw[1], b.original_hw[0])

        def raster(path, mode, *, nearest=False):
            image = _read_image(Path(path), mode)
            # Conditions live in the frame's normalised space, so they land on the frame grid
            # first and then follow the very same canvas transform as RGB.
            if image.size != frame:
                image = image.resize(frame, Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR)
            array = np.asarray(b.image(image, nearest=nearest, fill=CONDITION_PAD_VALUE), dtype=np.float32).copy()
            tensor = torch.from_numpy(array).unsqueeze(0) if mode == 'L' else torch.from_numpy(array).permute(2, 0, 1)
            return tensor / 255

        def rgb(image):
            # Must be the same call precompute() uses, or the latent cache is keyed on different
            # pixels than the ones training feeds.
            return self.rgb_transform(image)

        mask = raster(sample['target_person_mask'], 'L', nearest=True)
        raw_normal = raster(sample['target_normal'], 'RGB')
        normal = (raw_normal * 2 - 1) * mask
        part_path = self.assets_root / seq / 'part_512' / f'{stem}.png'
        if not part_path.is_file():
            part_path = part_path.with_suffix('.jpg')
        part_img = _read_image(part_path, 'L')
        if part_img.size != frame:
            part_img = part_img.resize(frame, Image.Resampling.NEAREST)
        labels = torch.from_numpy(np.asarray(b.image(part_img, nearest=True), dtype=np.int64).copy())
        if (labels < 0).any() or (labels > 14).any():
            raise ValueError(f'Invalid part labels: {part_path}')
        part = F.one_hot(labels, num_classes=15)[..., 1:].permute(2, 0, 1).float()
        if not bool((mask.sum() > 0) & (part.sum() > 0)):
            raise ValueError(f'Empty body mask/part: {seq}/{stem}')
        has_depth = bool(sample.get('target_depth_vis')) and Path(sample['target_depth_vis']).is_file()
        depth = raster(sample['target_depth_vis'], 'L') if has_depth else torch.zeros_like(mask)
        canvas_h, canvas_w = b.canvas_hw
        has_dwpose = bool(sample.get('target_dwpose')) and Path(sample['target_dwpose']).is_file()
        dwpose = raster(sample['target_dwpose'], 'RGB') if has_dwpose else torch.zeros(3, canvas_h, canvas_w)
        points = b.points(points)
        points[:, 2] = torch.where(visible, points[:, 2], torch.zeros_like(points[:, 2]))
        valid = b.valid_mask()
        camera = b.global_camera(vector)
        pair = {
            'src_image': rgb(source),
            'tgt_image': rgb(target),
            'control_map': torch.cat([raw_normal, depth, dwpose, mask]),
            'normal': normal,
            'depth': depth,
            'part_onehot': part,
            'pose_heatmap': _pose_heatmaps(points, resolution=b.canvas_hw, sigma=self.heatmap_sigma) * valid,
            'smplx_global': camera,
            'human_mask': mask,
            'valid_image_mask': valid,
            'task_id': torch.tensor(0, dtype=torch.long),
            'appearance': seq,
            'source_stem': sample['source_stem'],
            'target_stem': stem,
            'source_canvas': a.metadata(),
            'target_canvas': b.metadata(),
            'source_token_hw': a.token_hw,
            'target_token_hw': b.token_hw,
            'camera_metadata': {
                'convention': 'legacy 26D SMPL-X global; normalised fx,fy,cx,cy at indices 19..22',
                'original_smplx_global': vector,
                'canvas_smplx_global': camera},
            'depth_metadata': {'metric': False, 'source': 'depth_vis' if has_depth else 'absent'},
        }
        assert_pair_contract(pair, where='TikTokV3PoseDataset budget pair')
        return pair



def is_tiktok_v3_manifest(pairs_jsonl: str | Path) -> bool:
    """Detect whether a JSONL manifest uses the TikTok_v3 pair schema."""
    path = Path(pairs_jsonl)
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            return "sequence_id" in row or "source_rgb" in row
    return False


def load_pose_dataset(
    pairs_jsonl: str | Path,
    *,
    dataset_type: str = "auto",
    sampled_root: str | Path = "/home/shangguanrz/project/pic-edit/datasets/iPER/iper_sampled_6src64tgt",
    assets_root: str | Path = "/home/shangguanrz/project/pic-edit/datasets/iPER/iper_assets_512_nvdiffrast",
    tiktok_raw_root: str | Path = "/home/shangguanrz/project/pic-edit/datasets/TikTokDataset/TikTok_dataset/TikTok_dataset",
    tiktok_assets_root: str | Path = "/home/shangguanrz/project/pic-edit/datasets/TikTokDataset/TikTok_3d_assets_native",
    resolution: int = 512,
    align_mode: str | None = None,
    max_pixels: int | None = None,
    augment: bool = False,
    legacy_policy_ok: bool = False,
) -> Dataset:
    """Unified factory returning TikTokV3PoseDataset or IPERPoseDataset.

    The TikTok branch enforces the same canvas gate as QwenFullDataset: the legacy iPER
    trainers reach TikTok through here, and without it an omitted --align-mode still meant
    "silently squeeze the portrait into a square".
    """
    resolved_type = dataset_type
    if resolved_type == "auto":
        resolved_type = "tiktok_v3" if is_tiktok_v3_manifest(pairs_jsonl) else "iper"
    if resolved_type == "tiktok_v3":
        from .canvas_contract import LEGACY_POLICIES, POLICY_FOR_MODE, CanvasPolicyError
        if align_mode is None:
            raise ValueError("align_mode must be given explicitly for TikTok: 'native', "
                             "'aspect_preserving' or 'letterbox' keep body proportions")
        policy = POLICY_FOR_MODE.get(str(align_mode), str(align_mode))
        if policy in LEGACY_POLICIES and not legacy_policy_ok:
            raise CanvasPolicyError(
                f"align_mode={align_mode!r} is refused here: it "
                + ("stretches the person to a square" if policy == "resize_square"
                   else "crops the field of view")
                + ". Pass legacy_policy_ok=True only to reproduce a run trained that way.")
        # If caller explicitly passed a TikTok assets directory via assets_root, honor it
        assets_p = Path(assets_root)
        if "TikTok" in str(assets_p) or (assets_p / "00001").exists():
            tiktok_assets_root = assets_p
        return TikTokV3PoseDataset(
            pairs_jsonl=pairs_jsonl,
            raw_root=tiktok_raw_root,
            assets_root=tiktok_assets_root,
            resolution=resolution,
            align_mode=align_mode,
            max_pixels=max_pixels,
            augment=augment,
        )
    if resolved_type == "iper":
        from src.data.iper_dataset import IPERPoseDataset

        return IPERPoseDataset(
            pairs_jsonl=str(pairs_jsonl),
            sampled_root=str(sampled_root),
            assets_root=str(assets_root),
            resolution=resolution,
            augment=augment,
        )
    raise ValueError(f"Unsupported dataset_type: {dataset_type!r}")


__all__ = [
    "TikTokV65OverfitInputs",
    "TikTokV65FaceOverfitLoader",
    "TikTokPoseDataset",
    "TikTokV3PoseDataset",
    "build_tiktok_face_inputs",
    "is_tiktok_v3_manifest",
    "load_pose_dataset",
    "transform_box_for_crop",
    "transform_keypoints_for_crop",
]
