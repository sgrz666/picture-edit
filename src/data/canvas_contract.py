"""Single authoritative canvas protocol for RGB, geometry, camera and token grids.

A ``CanvasSpec`` is the only record allowed to decide how a sample reaches a model
canvas. It is built from integer pixel geometry and *cannot represent a non-uniform
scale*: ``resized_hw`` must agree with one shared factor up to the half-pixel rounding
bound, otherwise construction raises :class:`NonUniformScaleError`. Stretching a person
is therefore unconstructible here rather than merely discouraged.

The two affine matrices are derived properties instead of stored fields on purpose: a
stored 3x3 tensor could silently disagree with the integer geometry it describes, and
deriving them makes that disagreement impossible.

Modes already shipped that do distort (``resize_square``) or discard field-of-view
(``square_crop``) are represented by :class:`LegacyStretch`, which never poses as a
valid canvas and is refused by :func:`require_canonical`.

Coordinate conventions match ``src/data/letterbox.py``: ``*_hw`` fields are
``(height, width)``, ``offset_xy`` is ``(x, y)``, ``points`` speak normalised units of
the image they describe (input: original, output: canvas).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

VERSION = "canvas_v2"

NATIVE_PAD = "native_pad"
ASPECT_PRESERVING = "aspect_preserving"
LETTERBOX = "letterbox"
CANONICAL_POLICIES = (NATIVE_PAD, ASPECT_PRESERVING, LETTERBOX)

RESIZE_SQUARE = "resize_square"
SQUARE_CROP = "square_crop"
LEGACY_POLICIES = (RESIZE_SQUARE, SQUARE_CROP)

# The shipped align_mode spellings, mapped onto the policies they really are. Datasets,
# the training gate and cache fingerprints all speak this one vocabulary.
POLICY_FOR_MODE = {"native": NATIVE_PAD, "letterbox": LETTERBOX,
                   "aspect_preserving": ASPECT_PRESERVING, "native_pad": NATIVE_PAD,
                   "resize": RESIZE_SQUARE, "resize_square": RESIZE_SQUARE,
                   "square_crop": SQUARE_CROP}

# Qwen's VAE downsamples 16x; diffusers' _unpack_latents assumes 32-px canvases and the
# DiT VLM slots require the total token count to be divisible by 4.
VAE_STRIDE = 16
CANVAS_MULTIPLE = 32
TOKEN_SLOT_MULTIPLE = 4


class CanvasPolicyError(ValueError):
    """A sample was not built on a canonical, aspect-preserving canvas."""


class NonUniformScaleError(CanvasPolicyError):
    """The requested canvas changes the aspect ratio of the image."""


def _hw(value: Any, name: str) -> tuple[int, int]:
    """Normalise a (height, width) pair. Explicitly rejects PIL's (width, height)."""
    if isinstance(value, (CanvasSpec, LegacyStretch)):
        raise TypeError(f"{name} expects an (H, W) pair, not a canvas record")
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise TypeError(f"{name} must be an (height, width) pair, got {value!r}")
    height, width = int(value[0]), int(value[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} must be positive, got {(height, width)!r}")
    return (height, width)


def _xy(value: Any, name: str, *, allow_none: bool = False) -> tuple[int, int]:
    if value is None and allow_none:
        return (0, 0)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise TypeError(f"{name} must be an (x, y) pair, got {value!r}")
    x, y = int(value[0]), int(value[1])
    if x < 0 or y < 0:
        raise ValueError(f"{name} must be non-negative, got {(x, y)!r}")
    return (x, y)


@dataclass(frozen=True)
class CanvasSpec:
    """Exact, reversible placement of one original image on one model canvas."""

    original_hw: tuple[int, int]
    resized_hw: tuple[int, int]
    canvas_hw: tuple[int, int]
    offset_xy: tuple[int, int] = (0, 0)
    policy: str = NATIVE_PAD
    stride: int = VAE_STRIDE
    version: str = VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "original_hw", _hw(self.original_hw, "original_hw"))
        object.__setattr__(self, "resized_hw", _hw(self.resized_hw, "resized_hw"))
        object.__setattr__(self, "canvas_hw", _hw(self.canvas_hw, "canvas_hw"))
        object.__setattr__(self, "offset_xy", _xy(self.offset_xy, "offset_xy", allow_none=True))
        if self.policy not in CANONICAL_POLICIES:
            raise CanvasPolicyError(
                f"policy={self.policy!r} is not aspect-preserving; a canonical canvas must use one of "
                f"{CANONICAL_POLICIES}. Distorting legacy modes go through LegacyStretch."
            )
        if int(self.stride) <= 0:
            raise ValueError("stride must be positive")
        original_h, original_w = self.original_hw
        resized_h, resized_w = self.resized_hw
        canvas_h, canvas_w = self.canvas_hw
        offset_x, offset_y = self.offset_xy
        if offset_x + resized_w > canvas_w or offset_y + resized_h > canvas_h:
            raise ValueError(
                f"resized {self.resized_hw} at offset {self.offset_xy} does not fit canvas {self.canvas_hw}"
            )
        if canvas_h % self.stride or canvas_w % self.stride:
            raise ValueError(f"canvas {self.canvas_hw} must be a multiple of stride {self.stride}")
        tokens = self.token_count
        if tokens % TOKEN_SLOT_MULTIPLE:
            raise ValueError(f"token count {tokens} must be divisible by {TOKEN_SLOT_MULTIPLE}")
        # One shared scale, verified against the half-pixel error any integer resize can
        # introduce. Independent sx and sy is exactly the defect this module forbids.
        scale_x, scale_y = self.scale_xy
        bound = 0.5 / original_w + 0.5 / original_h
        if abs(scale_x - scale_y) > bound:
            raise NonUniformScaleError(
                f"canvas {self.canvas_hw} stretches {self.original_hw} by sx={scale_x:.6f} "
                f"sy={scale_y:.6f} (anisotropy {abs(scale_x - scale_y):.6f} exceeds rounding bound "
                f"{bound:.6f}); shrink both axes together and pad instead"
            )

    # ---- derived geometry -------------------------------------------------------

    @property
    def scale_xy(self) -> tuple[float, float]:
        """Realised per-axis scales, including integer-quantisation error."""
        original_h, original_w = self.original_hw
        resized_h, resized_w = self.resized_hw
        return (resized_w / original_w, resized_h / original_h)

    @property
    def valid_rect(self) -> tuple[int, int, int, int]:
        offset_x, offset_y = self.offset_xy
        resized_h, resized_w = self.resized_hw
        return (offset_x, offset_y, offset_x + resized_w, offset_y + resized_h)

    @property
    def token_hw(self) -> tuple[int, int]:
        canvas_h, canvas_w = self.canvas_hw
        return (canvas_h // self.stride, canvas_w // self.stride)

    @property
    def token_count(self) -> int:
        height, width = self.token_hw
        return height * width

    @property
    def transform(self) -> torch.Tensor:
        """3x3 affine taking original pixel coordinates onto the canvas."""
        scale_x, scale_y = self.scale_xy
        offset_x, offset_y = self.offset_xy
        return torch.tensor([[scale_x, 0.0, offset_x],
                             [0.0, scale_y, offset_y],
                             [0.0, 0.0, 1.0]], dtype=torch.float64)

    @property
    def inverse_transform(self) -> torch.Tensor:
        return torch.linalg.inv(self.transform).to(torch.float64)

    # ---- sampling ---------------------------------------------------------------

    def image(self, image: Image.Image, *, nearest: bool = False, fill: Any = 0) -> Image.Image:
        if image.size != (self.original_hw[1], self.original_hw[0]):
            raise ValueError(
                f"image size {image.size} (W, H) does not match original_hw {self.original_hw} (H, W)"
            )
        method = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
        target = (self.resized_hw[1], self.resized_hw[0])
        resized = image if image.size == target else image.resize(target, method)
        canvas = Image.new(image.mode, (self.canvas_hw[1], self.canvas_hw[0]), fill)
        canvas.paste(resized, tuple(self.offset_xy))
        return canvas

    def stack(self, value: torch.Tensor, *, nearest: bool = False, fill: float = 0.0) -> torch.Tensor:
        """Put a [..., H, W] RGB or condition tensor on the canvas.

        Conditions generated on the original grid and RGB share this one transform, which
        is what keeps pose, normal, part and mask aligned with the pixels.
        """
        if value.shape[-2:] != torch.Size(self.original_hw):
            raise ValueError(f"tensor grid {tuple(value.shape[-2:])} != original_hw {self.original_hw}")
        leading = tuple(value.shape[:-2])
        flat = value.reshape(-1, *self.original_hw)[:, None]
        if self.resized_hw != self.original_hw:
            flat = torch.nn.functional.interpolate(
                flat, size=self.resized_hw,
                mode="nearest" if nearest else "bilinear",
                align_corners=None if nearest else False)
        resized_h, resized_w = self.resized_hw
        canvas_h, canvas_w = self.canvas_hw
        offset_x, offset_y = self.offset_xy
        padded = torch.full((flat.shape[0], 1, canvas_h, canvas_w), float(fill),
                            dtype=flat.dtype, device=flat.device)
        padded[..., offset_y:offset_y + resized_h, offset_x:offset_x + resized_w] = flat
        return padded.reshape(*leading, canvas_h, canvas_w)

    def valid_mask(self, *, device: Any = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Canvas-validity mask. Deliberately distinct from the human mask: background
        inside the valid rect still has to be generated."""
        canvas_h, canvas_w = self.canvas_hw
        mask = torch.zeros(1, canvas_h, canvas_w, dtype=dtype, device=device)
        x0, y0, x1, y1 = self.valid_rect
        mask[:, y0:y1, x0:x1] = 1
        return mask

    def token_valid_coverage(self) -> torch.Tensor:
        """Per-token fraction (0..1) of each stride x stride block that holds real image.

        Row-major over ``token_hw``, which is the order the VAE latents are flattened in, so
        this lines up with the token axis of the DiT sequence without a further transpose.
        """
        canvas_h, canvas_w = self.canvas_hw
        offset_x, offset_y = self.offset_xy
        resized_h, resized_w = self.resized_hw
        stride = self.stride
        x1, y1 = offset_x + resized_w, offset_y + resized_h
        rows = []
        for row in range(canvas_h // stride):
            top, bottom = row * stride, (row + 1) * stride
            overlap_y = max(0, min(bottom, y1) - max(top, offset_y))
            rows.append([overlap_y * max(0, min((col + 1) * stride, x1) - max(col * stride, offset_x))
                         / float(stride * stride) for col in range(canvas_w // stride)])
        return torch.tensor(rows, dtype=torch.float32)

    def padding_token_mask(self) -> torch.Tensor:
        """True only where a token block lies entirely outside the image.

        Partially covered boundary tokens stay False: they carry real pixels and must not be
        dropped. This is the mask an attention-level exclusion is allowed to use.
        """
        return self.token_valid_coverage() == 0

    def unpad(self, image: Any) -> Any:
        x0, y0, x1, y1 = self.valid_rect
        if torch.is_tensor(image):
            return image[..., y0:y1, x0:x1]
        return image.crop((x0, y0, x1, y1))

    # ---- coordinates ------------------------------------------------------------

    def points(self, points: Any) -> Any:
        """Normalised original coords -> normalised canvas coords."""
        scale_x, scale_y = self.scale_xy
        canvas_h, canvas_w = self.canvas_hw
        offset_x, offset_y = self.offset_xy
        out = points.copy() if isinstance(points, np.ndarray) else points.clone()
        out[..., 0] = points[..., 0] * (scale_x * self.original_hw[1] / canvas_w) + offset_x / canvas_w
        out[..., 1] = points[..., 1] * (scale_y * self.original_hw[0] / canvas_h) + offset_y / canvas_h
        return out

    def unmap_points(self, points: Any) -> Any:
        """Inverse of :meth:`points`: normalised canvas coords -> normalised original."""
        scale_x, scale_y = self.scale_xy
        canvas_h, canvas_w = self.canvas_hw
        offset_x, offset_y = self.offset_xy
        out = points.copy() if isinstance(points, np.ndarray) else points.clone()
        out[..., 0] = (points[..., 0] * canvas_w - offset_x) / (scale_x * self.original_hw[1])
        out[..., 1] = (points[..., 1] * canvas_h - offset_y) / (scale_y * self.original_hw[0])
        return out

    def pixel_points(self, points: Any) -> Any:
        """Original pixel coords -> canvas pixel coords."""
        scale_x, scale_y = self.scale_xy
        offset_x, offset_y = self.offset_xy
        out = points.copy() if isinstance(points, np.ndarray) else points.clone()
        out[..., 0] = points[..., 0] * scale_x + offset_x
        out[..., 1] = points[..., 1] * scale_y + offset_y
        return out

    def camera(self, intrinsics: Any) -> Any:
        """K' = T K for the canvas, keeping SMPL-X reprojection aligned with the pixels.

        ``intrinsics`` is a 3x3 matrix in original pixel units. Per-axis realised scales
        are used rather than one nominal factor so that this agrees with :meth:`points`
        to within the same rounding bound the canvas itself guarantees.
        """
        if isinstance(intrinsics, np.ndarray):
            matrix = torch.from_numpy(np.asarray(intrinsics, dtype=np.float64))
            to_numpy = True
        else:
            matrix = intrinsics.to(torch.float64) if torch.is_tensor(intrinsics) else \
                torch.tensor(intrinsics, dtype=torch.float64)
            to_numpy = False
        if matrix.shape[-2:] != (3, 3):
            raise ValueError(f"camera intrinsics must be 3x3, got {tuple(matrix.shape)}")
        scale_x, scale_y = self.scale_xy
        offset_x, offset_y = self.offset_xy
        transform = torch.zeros_like(matrix)
        transform[..., 0, 0] = scale_x
        transform[..., 1, 1] = scale_y
        transform[..., 0, 2] = offset_x
        transform[..., 1, 2] = offset_y
        transform[..., 2, 2] = 1
        result = transform @ matrix
        return result.numpy() if to_numpy else result.to(matrix.dtype if matrix.dtype.is_floating_point
                                                        else torch.float32)

    def global_camera(self, vector: torch.Tensor) -> torch.Tensor:
        """Remap the legacy 26D SMPL-X global condition (normalised fx, fy, cx, cy)."""
        if vector.shape[-1] != 26:
            raise ValueError("Expected the legacy 26D SMPL-X global condition")
        scale_x, scale_y = self.scale_xy
        canvas_h, canvas_w = self.canvas_hw
        offset_x, offset_y = self.offset_xy
        out = vector.clone()
        fx_factor = scale_x * self.original_hw[1] / canvas_w
        fy_factor = scale_y * self.original_hw[0] / canvas_h
        out[..., 19] = vector[..., 19] * fx_factor
        out[..., 20] = vector[..., 20] * fy_factor
        out[..., 21] = vector[..., 21] * fx_factor + offset_x / canvas_w
        out[..., 22] = vector[..., 22] * fy_factor + offset_y / canvas_h
        return out

    # ---- records ----------------------------------------------------------------

    def metadata(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "policy": self.policy,
            "stride": int(self.stride),
            "original_hw": list(self.original_hw),
            "resized_hw": list(self.resized_hw),
            "canvas_hw": list(self.canvas_hw),
            "offset_xy": list(self.offset_xy),
            "valid_rect": list(self.valid_rect),
            "token_hw": list(self.token_hw),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "CanvasSpec":
        if record.get("policy") in LEGACY_POLICIES or record.get("version") == LegacyStretch.VERSION:
            raise CanvasPolicyError(
                f"record carries legacy policy {record.get('policy')!r}; it distorts or crops and "
                "cannot be read as a canonical canvas")
        missing = {"original_hw", "resized_hw", "canvas_hw"} - set(record)
        if missing:
            raise CanvasPolicyError(f"canvas record missing {sorted(missing)}")
        return cls(
            original_hw=tuple(int(v) for v in record["original_hw"]),
            resized_hw=tuple(int(v) for v in record["resized_hw"]),
            canvas_hw=tuple(int(v) for v in record["canvas_hw"]),
            offset_xy=tuple(int(v) for v in record.get("offset_xy", (0, 0))),
            policy=str(record.get("policy", NATIVE_PAD)),
            stride=int(record.get("stride", VAE_STRIDE)),
            version=str(record.get("version", VERSION)),
        )

    def fingerprint(self) -> str:
        """Stable hash to bind caches and checkpoints to this exact canvas."""
        blob = json.dumps(self.metadata(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LegacyStretch:
    """Self-declaring record for the shipped distorting/cropping modes.

    Kept only so historical runs stay reproducible. It is *not* a canvas contract: it
    carries no uniform scale and :func:`require_canonical` always refuses it.
    """

    policy: str
    original_hw: tuple[int, int]
    canvas_hw: tuple[int, int]
    crop_rect: tuple[int, int, int, int] | None = None
    version: str = "legacy_stretch_v1"

    VERSION = "legacy_stretch_v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "original_hw", _hw(self.original_hw, "original_hw"))
        object.__setattr__(self, "canvas_hw", _hw(self.canvas_hw, "canvas_hw"))
        if self.policy not in LEGACY_POLICIES:
            raise CanvasPolicyError(f"{self.policy!r} is not a legacy policy; use CanvasSpec")
        if self.crop_rect is not None:
            rect = tuple(int(v) for v in self.crop_rect)
            if len(rect) != 4:
                raise ValueError("crop_rect must be (x0, y0, x1, y1)")
            object.__setattr__(self, "crop_rect", rect)

    @property
    def aspect_distortion(self) -> float:
        """How much the mode changes body proportions; 1.0 means no stretch."""
        original_h, original_w = self.original_hw
        if self.crop_rect is not None:
            x0, y0, x1, y1 = self.crop_rect
            original_w, original_h = x1 - x0, y1 - y0
        canvas_h, canvas_w = self.canvas_hw
        return (canvas_w / original_w) / (canvas_h / original_h)

    def metadata(self) -> dict[str, Any]:
        return {"version": self.version, "policy": self.policy,
                "original_hw": list(self.original_hw), "canvas_hw": list(self.canvas_hw),
                "crop_rect": list(self.crop_rect) if self.crop_rect else None,
                "aspect_distortion": round(self.aspect_distortion, 6)}

    def describe(self) -> str:
        if self.policy == RESIZE_SQUARE:
            return (f"non-uniform resize {self.original_hw} -> {self.canvas_hw} changes body "
                    f"proportions by x{self.aspect_distortion:.3f}")
        return f"{self.original_hw} -> {self.canvas_hw} crops the field of view"


def native_pad(original_hw: Sequence[int], *, stride: int = VAE_STRIDE,
               pad_multiple: int | None = None, center: bool = False) -> CanvasSpec:
    """Keep the original pixels; only pad up to a canvas multiple."""
    height, width = _hw(original_hw, "original_hw")
    step = pad_multiple if pad_multiple is not None else max(stride, CANVAS_MULTIPLE)
    if step % stride:
        raise ValueError(f"pad_multiple {step} must be a multiple of stride {stride}")
    canvas_w = -(-width // step) * step
    canvas_h = -(-height // step) * step
    while ((canvas_h // stride) * (canvas_w // stride)) % TOKEN_SLOT_MULTIPLE:
        canvas_h += stride
    offset = ((canvas_w - width) // 2, (canvas_h - height) // 2) if center else (0, 0)
    return CanvasSpec((height, width), (height, width), (canvas_h, canvas_w), offset, NATIVE_PAD, stride)


def letterbox(original_hw: Sequence[int], side: int, *, stride: int = VAE_STRIDE) -> CanvasSpec:
    """Uniformly shrink the long axis to ``side`` and pad the short axis."""
    height, width = _hw(original_hw, "original_hw")
    if min(height, width, side) <= 0:
        raise ValueError("image and canvas dimensions must be positive")
    scale = side / max(height, width)
    resized_h = max(1, round(height * scale))
    resized_w = max(1, round(width * scale))
    return CanvasSpec((height, width), (resized_h, resized_w), (side, side),
                      ((side - resized_w) // 2, (side - resized_h) // 2), LETTERBOX, stride)


def aspect_preserving(original_hw: Sequence[int], *, max_pixels: int, pad_multiple: int | None = None,
                      stride: int = VAE_STRIDE, allow_upscale: bool = False,
                      center: bool = True) -> CanvasSpec:
    """Shrink both axes by one factor to fit a pixel budget, then pad.

    Exists because keeping the ratio and bounding the compute must both hold: a 3000x5000
    portrait is still too large for training even when nothing is stretched.
    """
    height, width = _hw(original_hw, "original_hw")
    if max_pixels <= 0:
        raise ValueError("max_pixels must be positive")
    step = pad_multiple if pad_multiple is not None else max(stride, CANVAS_MULTIPLE)
    if step % stride:
        raise ValueError(f"pad_multiple {step} must be a multiple of stride {stride}")
    scale = math.sqrt(max_pixels / (height * width))
    if scale > 1.0 and not allow_upscale:
        scale = 1.0
    resized_h = max(1, round(height * scale))
    resized_w = max(1, round(width * scale))
    canvas_h = -(-resized_h // step) * step
    canvas_w = -(-resized_w // step) * step
    while ((canvas_h // stride) * (canvas_w // stride)) % TOKEN_SLOT_MULTIPLE:
        canvas_h += stride
    offset = ((canvas_w - resized_w) // 2, (canvas_h - resized_h) // 2) if center else (0, 0)
    return CanvasSpec((height, width), (resized_h, resized_w), (canvas_h, canvas_w),
                      offset, ASPECT_PRESERVING, stride)


def resize_square(original_hw: Sequence[int], side: int) -> LegacyStretch:
    """The shipped stretch. Returns a refusal record, never a canvas."""
    return LegacyStretch(RESIZE_SQUARE, _hw(original_hw, "original_hw"), (int(side), int(side)))


def square_crop(original_hw: Sequence[int], *, crop_rect: Sequence[int]) -> LegacyStretch:
    height, width = _hw(original_hw, "original_hw")
    x0, y0, x1, y1 = (int(v) for v in crop_rect)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"crop rect {(x0, y0, x1, y1)} outside {original_hw}")
    return LegacyStretch(SQUARE_CROP, (height, width), (y1 - y0, x1 - x0), (x0, y0, x1, y1))


def canvas_for_mode(mode: str, original_hw: Sequence[int], *, resolution: int | str = 512,
                    max_pixels: int | None = None, **kwargs: Any) -> CanvasSpec | LegacyStretch:
    """One place that turns a mode name into geometry, for datasets and eval alike."""
    key = str(mode)
    if key == "native":
        return native_pad(original_hw, **kwargs)
    if key == "letterbox":
        return letterbox(original_hw, int(resolution), **kwargs)
    if key == "aspect_preserving":
        budget = max_pixels if max_pixels is not None else int(resolution) ** 2
        return aspect_preserving(original_hw, max_pixels=budget, **kwargs)
    if key in ("resize", "resize_square"):
        if kwargs:
            raise ValueError(f"resize accepts no options, got {sorted(kwargs)}")
        return resize_square(original_hw, int(resolution))
    if key == "square_crop":
        if kwargs:
            raise ValueError(f"square_crop takes only crop_rect via original_hw callers, got {sorted(kwargs)}")
        height, width = _hw(original_hw, "original_hw")
        side = min(height, width)
        top = max(0, min(height - side, round(100.0 * height / 1080.0))) if height > width else 0
        left = max(0, (width - side) // 2)
        return square_crop(original_hw, crop_rect=(left, top, left + side, top + side))
    raise ValueError(f"Unsupported align_mode: {mode!r}; allowed: native, letterbox, "
                     f"aspect_preserving, resize, square_crop")


def sampling_size(sample: Mapping[str, Any], *, fallback_side: int | None = None,
                  where: str = "sampling") -> tuple[int, int]:
    """Return the (width, height) a generated frame must have for this sample's canvas.

    Evaluation and run scripts used to hard-code `width=height=resolution`, which draws a square
    even when the pair was loaded onto a portrait canvas — the same distortion the loaders were
    fixed to refuse, reintroduced one stage later. With no canonical canvas this raises instead of
    quietly falling back; pass `fallback_side` only where a pre-contract square is the intent.
    """
    record = sample.get("target_canvas") if hasattr(sample, "get") else None
    if record is None:
        if fallback_side is None:
            raise CanvasPolicyError(
                f"{where}: no target_canvas on the sample, so the output size cannot be derived; "
                "load the pair through a canvas-aware dataset")
        return int(fallback_side), int(fallback_side)
    if isinstance(record, LegacyStretch) or (isinstance(record, Mapping)
                                             and record.get("version") == LegacyStretch.VERSION):
        # Checked before CanvasSpec.from_record, which refuses a legacy record outright: without
        # this order an explicit fallback_side could never reach the pre-contract path.
        if fallback_side is None:
            described = (record.describe() if isinstance(record, LegacyStretch)
                         else f"legacy {record.get('policy')!r} record")
            raise CanvasPolicyError(f"{where}: refusing to sample from {described}")
        return int(fallback_side), int(fallback_side)
    spec = record if isinstance(record, CanvasSpec) else CanvasSpec.from_record(record)
    height, width = spec.canvas_hw
    return int(width), int(height)


def require_canonical(canvas: Any, *, where: str = "training") -> CanvasSpec:
    """Fail closed unless the sample declares a canonical aspect-preserving canvas.

    ``None`` is a failure, not a fallback: silently assuming a square canvas is how the
    original defect reached training.
    """
    if canvas is None:
        raise CanvasPolicyError(
            f"missing CanvasSpec: {where} may not infer a canvas from an integer resolution")
    if isinstance(canvas, LegacyStretch):
        raise CanvasPolicyError(f"legacy canvas refused by {where}: {canvas.describe()}")
    if isinstance(canvas, str):
        canvas = {"policy": POLICY_FOR_MODE.get(canvas, canvas)}
    if isinstance(canvas, Mapping):
        policy = POLICY_FOR_MODE.get(str(canvas.get("policy")), canvas.get("policy"))
        if policy in LEGACY_POLICIES or canvas.get("version") == LegacyStretch.VERSION:
            raise CanvasPolicyError(
                f"legacy canvas {policy!r} refused by {where}; rebuild the cache with an "
                "aspect-preserving canvas policy")
        canvas = CanvasSpec.from_record({**canvas, "policy": policy})
    if not isinstance(canvas, CanvasSpec):
        raise CanvasPolicyError(f"{where} expected a CanvasSpec, got {type(canvas).__name__}")
    return canvas


__all__ = [
    "VERSION", "CANVAS_MULTIPLE", "VAE_STRIDE", "CANONICAL_POLICIES", "LEGACY_POLICIES",
    "NATIVE_PAD", "ASPECT_PRESERVING", "LETTERBOX", "RESIZE_SQUARE", "SQUARE_CROP",
    "CanvasPolicyError", "NonUniformScaleError", "CanvasSpec", "LegacyStretch", "POLICY_FOR_MODE",
    "native_pad", "letterbox", "aspect_preserving", "resize_square", "square_crop",
    "canvas_for_mode", "require_canonical",
]
