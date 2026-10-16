"""Serializable contracts shared by Union data, caches, and model wrappers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

DatasetName = Literal["iper", "tiktok", "hi4d", "3dpw"]
Split = Literal["train", "val", "test"]


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class CanvasTransform:
    original_wh: tuple[int, int]
    resized_wh: tuple[int, int]
    offset_xy: tuple[int, int]
    canvas_wh: tuple[int, int] = (512, 512)

    def __post_init__(self):
        for name in ("original_wh", "resized_wh", "offset_xy", "canvas_wh"):
            object.__setattr__(self, name, tuple(map(int, getattr(self, name))))
        if any(value <= 0 for value in (*self.original_wh, *self.resized_wh, *self.canvas_wh)):
            raise ValueError("canvas dimensions must be positive")
        if any(offset < 0 or offset + resized > canvas for offset, resized, canvas in
               zip(self.offset_xy, self.resized_wh, self.canvas_wh)):
            raise ValueError("resized image must fit inside the canvas")

    @classmethod
    def from_size(cls, original_wh, canvas_wh=(512, 512)):
        width, height = map(int, original_wh)
        if min(width, height) <= 0:
            raise ValueError("original dimensions must be positive")
        scale = min(canvas_wh[0] / width, canvas_wh[1] / height)
        resized = (max(1, round(width * scale)), max(1, round(height * scale)))
        offset = tuple((canvas - length) // 2 for canvas, length in zip(canvas_wh, resized))
        return cls((width, height), resized, offset, tuple(canvas_wh))

    def to_dict(self):
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, value):
        return value if isinstance(value, cls) else cls(**value)


@dataclass(frozen=True)
class ConditionRecord:
    sample_id: str
    dataset: DatasetName
    split: Split
    group_id: str
    source_rgb: Path
    reference_pose_images: tuple[Path, ...]
    control_image: Path
    prompt: str
    person_order: tuple[int, ...]
    source_transform: CanvasTransform
    target_transform: CanvasTransform
    provenance: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "source_rgb", Path(self.source_rgb))
        object.__setattr__(self, "control_image", Path(self.control_image))
        object.__setattr__(self, "reference_pose_images", tuple(map(Path, self.reference_pose_images)))
        object.__setattr__(self, "person_order", tuple(self.person_order))
        for name in ("source_transform", "target_transform"):
            object.__setattr__(self, name, CanvasTransform.from_dict(getattr(self, name)))
        if self.dataset not in ("iper", "tiktok", "hi4d", "3dpw") or self.split not in ("train", "val", "test"):
            raise ValueError("unknown dataset or split")
        if len(self.person_order) != len(self.reference_pose_images):
            raise ValueError("one pose reference is required per person")

    def to_dict(self):
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, value):
        if any(key in value for key in ("target_rgb", "target_latent", "target_masks", "supervision")):
            raise ValueError("inference conditions must not contain supervision fields")
        return cls(**value)


@dataclass(frozen=True)
class SupervisionRecord:
    sample_id: str
    target_rgb: Path
    target_masks: tuple[Path, ...] = ()
    requested_keypoints: tuple[Path, ...] = ()
    target_transform: CanvasTransform | None = None
    person_order: tuple[int, ...] = (0,)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "target_rgb", Path(self.target_rgb))
        for name in ("target_masks", "requested_keypoints"):
            object.__setattr__(self, name, tuple(map(Path, getattr(self, name))))
        object.__setattr__(self, "person_order", tuple(self.person_order))
        if self.target_transform is not None:
            object.__setattr__(self, "target_transform", CanvasTransform.from_dict(self.target_transform))

    def to_dict(self):
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


@dataclass(frozen=True)
class CachedCondition:
    sample_id: str
    payload: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key):
        return self.payload[key]

    def get(self, key, default=None):
        return self.payload.get(key, default)


@dataclass(frozen=True)
class UnionRunConfig:
    arm: str = "union_frozen"
    injection_scope: Literal["joint", "target"] = "joint"
    width: int = 512
    height: int = 512
    num_inference_steps: int = 40
    guidance_scale: float = 1.0
    control_context_scale: float = 1.0
    use_kv_cache: bool = False
    seeds: tuple[int, ...] = (42, 43)
    rank: int = 16
    alpha: int = 16
    metadata: dict[str, Any] = field(default_factory=dict)
