"""CHAMP sample validation and DeepGen pose-residual alignment."""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


CONTROL_CHANNELS = (
    "normal_r",
    "normal_g",
    "normal_b",
    "depth",
    "pose_r",
    "pose_g",
    "pose_b",
    "semantic",
)


@dataclass(frozen=True)
class ChampSample:
  """One aligned CHAMP source/target pair and its structural controls."""

  source: Image.Image
  target: Image.Image
  control: torch.Tensor
  paths: dict[str, Path]
  metadata: dict[str, object]


def _resolve_paths(root: Path) -> dict[str, Path]:
  paths = {
      "source": root / "real_src_frame_0.png",
      "target": root / "real_tgt_frame_60.png",
      "normal": root / "ctrl_normal.png",
      "depth": root / "ctrl_depth.png",
      "pose": root / "ctrl_openpose.png",
      "semantic": root / "ctrl_semantic.png",
  }
  for path in paths.values():
    if not path.is_file():
      raise FileNotFoundError(f"required CHAMP input is missing: {path}")
  return paths


def _resize_rgb(image: Image.Image, resolution: int) -> Image.Image:
  return image.convert("RGB").resize(
      (resolution, resolution), Image.Resampling.LANCZOS
  )


def _load_rgb(path: Path, resolution: int) -> np.ndarray:
  image = Image.open(path).convert("RGB").resize(
      (resolution, resolution), Image.Resampling.BILINEAR
  )
  return np.asarray(image, dtype=np.float32) / 255.0


def _load_gray(path: Path, resolution: int) -> np.ndarray:
  image = Image.open(path).convert("L").resize(
      (resolution, resolution), Image.Resampling.BILINEAR
  )
  return (np.asarray(image, dtype=np.float32) / 255.0)[:, :, None]


def _validate_target_rgb(target: Image.Image, path: Path) -> None:
  array = np.asarray(target.convert("RGB"), dtype=np.uint8)
  channel_equal = np.array_equal(array[:, :, 0], array[:, :, 1]) and np.array_equal(
      array[:, :, 1], array[:, :, 2]
  )
  luminance = np.rint(array.astype(np.float32).mean(axis=2)).astype(np.uint8)
  unique_luminance = np.unique(luminance)
  near_binary = np.mean((luminance <= 2) | (luminance >= 253)) > 0.95
  if channel_equal and (len(unique_luminance) < 32 or near_binary):
    raise ValueError(f"target RGB looks like a mask, not a video frame: {path}")


def load_champ_sample(root: Path | str, resolution: int = 512) -> ChampSample:
  """Load one aligned CHAMP pair using the fixed eight-channel contract."""
  if resolution <= 0 or resolution % 16 != 0:
    raise ValueError(f"resolution must be a positive multiple of 16, got {resolution}")

  root_path = Path(root)
  paths = _resolve_paths(root_path)
  source_raw = Image.open(paths["source"]).convert("RGB")
  target_raw = Image.open(paths["target"]).convert("RGB")
  _validate_target_rgb(target_raw, paths["target"])

  normal = _load_rgb(paths["normal"], resolution)
  depth = _load_gray(paths["depth"], resolution)
  pose = _load_rgb(paths["pose"], resolution)
  semantic = _load_gray(paths["semantic"], resolution)
  stacked = np.concatenate([normal, depth, pose, semantic], axis=-1)
  if stacked.shape != (resolution, resolution, 8):
    raise ValueError(
        "CHAMP controls must form [H,W,8], "
        f"got {stacked.shape} from {root_path}"
    )
  if not np.isfinite(stacked).all():
    raise ValueError(f"CHAMP controls contain NaN or Inf: {root_path}")

  control = torch.from_numpy(stacked.copy()).permute(2, 0, 1).unsqueeze(0)
  return ChampSample(
      source=_resize_rgb(source_raw, resolution),
      target=_resize_rgb(target_raw, resolution),
      control=control,
      paths=paths,
      metadata={
          "root": str(root_path),
          "resolution": resolution,
          "control_channels": CONTROL_CHANNELS,
          "source_size": source_raw.size,
          "target_size": target_raw.size,
      },
  )


def align_control_residuals(
    residuals: Sequence[torch.Tensor],
    target_tokens: int,
    reference_tokens: int,
    batch_size: int,
) -> list[torch.Tensor]:
  """Pad reference-token positions with zero without changing batch or scale."""
  if target_tokens <= 0 or reference_tokens < 0 or batch_size <= 0:
    raise ValueError("target_tokens and batch_size must be positive")
  if not residuals:
    raise ValueError("at least one control residual layer is required")

  aligned = []
  for index, residual in enumerate(residuals):
    if residual.ndim != 3 or residual.shape[1] != target_tokens:
      raise ValueError(
          f"control layer {index}: expected [B,{target_tokens},D], "
          f"got {tuple(residual.shape)}"
      )
    if residual.shape[0] != batch_size:
      raise ValueError(
          f"control layer {index}: expected batch {batch_size}, "
          f"got {residual.shape[0]}"
      )
    if not torch.isfinite(residual).all():
      raise ValueError(f"control layer {index} contains NaN or Inf")
    aligned.append(F.pad(residual, (0, 0, 0, reference_tokens)))
  return aligned


def image_diagnostics(image: Image.Image) -> dict[str, float | bool]:
  """Return cheap numerical guards against collapsed generated images."""
  array = np.asarray(image.convert("RGB"), dtype=np.float32)
  mean = float(array.mean())
  std = float(array.std())
  near_white_ratio = float(np.mean(np.all(array >= 250.0, axis=2)))
  near_black_ratio = float(np.mean(np.all(array <= 5.0, axis=2)))
  finite = bool(np.isfinite(array).all())
  valid = finite and std >= 5.0 and near_white_ratio <= 0.95 and near_black_ratio <= 0.95
  return {
      "mean": mean,
      "std": std,
      "near_white_ratio": near_white_ratio,
      "near_black_ratio": near_black_ratio,
      "finite": finite,
      "valid": valid,
  }
