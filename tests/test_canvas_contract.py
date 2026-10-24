"""Acceptance tests for the canvas contract that replaces the square Resize.

Covers the size-preservation and grid criteria of the remediation plan: no portrait may
be stretched, 768x1344 must reach an 84x48 token grid rather than 64x64, camera and
key points must follow the same transform, and the legacy modes must be refused instead
of silently assumed.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.canvas_contract import (
    ASPECT_PRESERVING,
    CANVAS_MULTIPLE,
    CanvasPolicyError,
    CanvasSpec,
    LegacyStretch,
    NonUniformScaleError,
    RESIZE_SQUARE,
    SQUARE_CROP,
    aspect_preserving,
    canvas_for_mode,
    letterbox,
    native_pad,
    require_canonical,
)
from src.data.letterbox import Letterbox, NativePad

PORTRAIT_HW = (1344, 768)
TIKTOK_HW = (1080, 604)
CLOSER_HW = (1333, 750)


def _gradient(height: int, width: int) -> Image.Image:
    rows = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    cols = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    band = ((rows.astype(np.int32) + cols.astype(np.int32)) % 256).astype(np.uint8)
    return Image.fromarray(np.dstack([band, band[::-1], np.roll(band, 7, axis=1)]), "RGB")


def test_native_pad_keeps_portrait_pixels_and_produces_a_rectangular_grid():
    spec = native_pad(PORTRAIT_HW)
    assert spec.resized_hw == PORTRAIT_HW, "native must not rescale"
    assert spec.canvas_hw == PORTRAIT_HW
    assert spec.token_hw == (84, 48), spec.token_hw
    assert spec.token_count == 4032
    assert spec.token_hw != (64, 64), "the square grid assumption is the defect being removed"


def test_odd_portrait_is_padded_to_the_canvas_multiple_without_rescaling():
    spec = native_pad(CLOSER_HW)
    assert spec.original_hw == CLOSER_HW
    assert spec.resized_hw == CLOSER_HW, "750x1333 must keep its pixels"
    assert spec.canvas_hw == (1344, 768)
    assert spec.offset_xy == (0, 0)
    assert spec.token_hw == (84, 48)
    assert all(v % CANVAS_MULTIPLE == 0 for v in spec.canvas_hw)


def test_a_non_uniform_scale_cannot_be_constructed_at_all():
    with pytest.raises(NonUniformScaleError) as info:
        CanvasSpec(PORTRAIT_HW, (512, 512), (512, 512))
    assert "stretches" in str(info.value)
    with pytest.raises(NonUniformScaleError):  # 604x1080 -> 512x512, the shipped defect
        CanvasSpec(TIKTOK_HW, (512, 512), (512, 512))
    with pytest.raises(NonUniformScaleError):  # 270x151 -> 512x512 stretches width harder
        CanvasSpec((270, 151), (512, 512), (512, 512))
    # Isotropic upscaling is a resampling choice, not a distortion, so it must stay legal.
    uniform = CanvasSpec((270, 270), (512, 512), (512, 512))
    assert pytest.approx(uniform.scale_xy[0], rel=1e-9) == uniform.scale_xy[1]


def test_square_canvases_are_still_allowed_because_they_are_not_stretches():
    assert native_pad((1024, 1024)).token_hw == (64, 64)
    square = letterbox((1024, 1024), 512)
    assert square.resized_hw == (512, 512) and square.token_hw == (32, 32)


def test_legacy_resize_reports_its_distortion_and_is_refused():
    legacy = canvas_for_mode("resize", TIKTOK_HW, resolution=512)
    assert isinstance(legacy, LegacyStretch)
    assert legacy.policy == RESIZE_SQUARE
    # 604x1080 squeezed into 512x512 widens the body by 1080/604.
    assert pytest.approx(legacy.aspect_distortion, rel=1e-6) == 1080 / 604
    with pytest.raises(CanvasPolicyError, match="non-uniform resize"):
        require_canonical(legacy, where="formal training")
    with pytest.raises(CanvasPolicyError, match="missing CanvasSpec"):
        require_canonical(None, where="formal training")
    # The old mode spelling is recognised and refused with the real reason.
    with pytest.raises(CanvasPolicyError, match="resize_square"):
        require_canonical({"policy": "resize"}, where="formal training")


def test_square_crop_is_refused_for_full_body_training_but_stays_reproducible():
    legacy = canvas_for_mode("square_crop", TIKTOK_HW, resolution=512)
    assert legacy.policy == SQUARE_CROP
    assert legacy.crop_rect == (0, 100, 604, 704), legacy.crop_rect
    assert pytest.approx(legacy.aspect_distortion, rel=1e-6) == 1.0
    with pytest.raises(CanvasPolicyError, match="crops the field of view"):
        require_canonical(legacy, where="formal training")


def test_aspect_preserving_honours_the_pixel_budget_without_distorting():
    spec = aspect_preserving((5000, 3000), max_pixels=1024 * 1024)
    assert spec.resized_hw == (1322, 793), spec.resized_hw
    assert spec.canvas_hw == (1344, 800) and spec.offset_xy == (3, 11)
    scale_x, scale_y = spec.scale_xy
    assert abs(scale_x - scale_y) <= 0.5 / 3000 + 0.5 / 5000
    assert spec.canvas_hw[0] * spec.canvas_hw[1] <= 1024 * 1024 + 32 * (1344 + 800)
    assert spec.token_hw == (84, 50)
    assert spec.token_hw[0] != spec.token_hw[1], "a 5:3 portrait must not become square"
    assert spec.policy == ASPECT_PRESERVING
    # Under budget, nothing is resampled: no silent upscaling either.
    small = aspect_preserving(CLOSER_HW, max_pixels=1024 * 1024)
    assert small.resized_hw == native_pad(CLOSER_HW).resized_hw == CLOSER_HW


def test_native_pad_and_letterbox_reproduce_the_legacy_transforms_exactly():
    for original_hw in (TIKTOK_HW, CLOSER_HW, PORTRAIT_HW, (1024, 1024), (512, 512)):
        image = _gradient(*original_hw)
        width, height = original_hw[1], original_hw[0]

        native = native_pad(original_hw)
        old_native = NativePad.from_size((width, height))
        assert native.canvas_hw == (old_native.padded_height, old_native.padded_width)
        assert np.array_equal(np.asarray(native.image(image, fill=(128, 128, 128))),
                              np.asarray(old_native.image(image, fill=(128, 128, 128))))
        assert torch.equal(native.valid_mask(), old_native.valid_mask())

        boxed = letterbox(original_hw, 512)
        old_box = Letterbox.from_size((width, height), 512)
        assert boxed.resized_hw == (old_box.resized_height, old_box.resized_width)
        assert np.array_equal(np.asarray(boxed.image(image, fill=(128, 128, 128))),
                              np.asarray(old_box.image(image, fill=(128, 128, 128))))
        assert torch.equal(boxed.valid_mask(), old_box.valid_mask())

        points = torch.rand(18, 3)
        assert torch.allclose(native.points(points), old_native.points(points), atol=1e-7)
        assert torch.allclose(boxed.points(points), old_box.points(points), atol=1e-7)
        vector = torch.rand(26)
        assert torch.allclose(native.global_camera(vector), old_native.global_camera(vector), atol=1e-7)
        assert torch.allclose(boxed.global_camera(vector), old_box.global_camera(vector), atol=1e-7)


def test_points_round_trip_and_stay_inside_the_valid_rect():
    spec = native_pad(TIKTOK_HW)
    canvas_h, canvas_w = spec.canvas_hw
    points = torch.tensor([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.5, 0.25, 1.0]])
    on_canvas = spec.points(points)
    assert torch.allclose(spec.unmap_points(on_canvas), points, atol=1e-6)
    x0, y0, x1, y1 = spec.valid_rect
    assert (x0, y0, x1, y1) == (0, 0, 604, 1080)
    assert on_canvas[1, 0].item() == pytest.approx(x1 / canvas_w, abs=1e-6)
    assert on_canvas[1, 1].item() == pytest.approx(y1 / canvas_h, abs=1e-6)
    assert bool((on_canvas[:, 0] * canvas_w <= x1 + 1e-6).all())


def test_transform_and_camera_agree_on_a_real_keypoint_reprojection():
    for spec in (native_pad(TIKTOK_HW), letterbox(TIKTOK_HW, 512),
                 aspect_preserving(PORTRAIT_HW, max_pixels=512 * 512)):
        height, width = spec.original_hw
        intrinsics = torch.tensor([[float(width), 0.0, width / 2 - 0.5],
                                   [0.0, float(height), height / 2 - 0.5],
                                   [0.0, 0.0, 1.0]])
        moved = spec.camera(intrinsics)
        # K' must equal the canvas affine composed with K, exactly as the plan specifies.
        assert torch.allclose(moved.double(), spec.transform.double() @ intrinsics.double(), atol=1e-9)
        # A 3D joint reprojected through K' must land where the pixel map puts p = K X.
        for joint in ([0.05, -0.3, 2.4], [-0.2, 0.1, 3.1]):
            camera_point = torch.tensor(joint, dtype=torch.float64)
            original_px = (intrinsics.double() @ camera_point)
            original_px = original_px[:2] / original_px[2]
            canvas_px = (moved.double() @ camera_point)
            canvas_px = canvas_px[:2] / canvas_px[2]
            assert torch.allclose(canvas_px, spec.pixel_points(original_px).double(), atol=1e-6)
        principal = torch.tensor([width / 2 - 0.5, height / 2 - 0.5])
        assert torch.allclose(moved[:2, 2], spec.pixel_points(principal.double()), atol=1e-6)


def test_shrunk_canvas_transforms_a_numpy_intrinsics_matrix():
    spec = letterbox(TIKTOK_HW, 512)
    intrinsics = np.array([[604.0, 0.0, 301.5], [0.0, 1080.0, 539.5], [0.0, 0.0, 1.0]])
    moved = spec.camera(intrinsics)
    assert isinstance(moved, np.ndarray)
    scale_x, scale_y = spec.scale_xy
    offset_x, offset_y = spec.offset_xy
    assert moved[0, 0] == pytest.approx(604.0 * scale_x, rel=1e-9)
    assert moved[1, 1] == pytest.approx(1080.0 * scale_y, rel=1e-9)
    assert moved[0, 2] == pytest.approx(301.5 * scale_x + offset_x, rel=1e-9)
    assert moved[1, 2] == pytest.approx(539.5 * scale_y + offset_y, rel=1e-9)


def test_unpadded_native_canvas_is_bit_identical_to_the_original():
    spec = native_pad(CLOSER_HW)
    tensor = torch.from_numpy(np.asarray(_gradient(*CLOSER_HW), dtype=np.float32)).permute(2, 0, 1)
    canvas = spec.stack(tensor)
    assert canvas.shape == (3, 1344, 768)
    assert torch.equal(spec.unpad(canvas), tensor), "padding must be pure, not a resample"
    assert float(canvas[:, 1333:1344, :].abs().sum()) == 0.0


def test_condition_maps_and_rgb_share_one_transform_on_the_canvas():
    spec = native_pad(TIKTOK_HW)
    rgb = torch.from_numpy(np.asarray(_gradient(*TIKTOK_HW), dtype=np.float32)).permute(2, 0, 1)
    mask = torch.zeros(1, *TIKTOK_HW)
    mask[..., 100:700, 50:550] = 1
    heatmap = torch.zeros(25, *TIKTOK_HW)
    heatmap[..., 540, 302] = 1
    rgb_canvas, mask_canvas, heat_canvas = (spec.stack(rgb), spec.stack(mask, nearest=True),
                                           spec.stack(heatmap, nearest=True))
    assert rgb_canvas.shape == (3, 1088, 608)
    assert mask_canvas.shape == (1, 1088, 608)
    assert heat_canvas.shape == (25, 1088, 608)
    # The joint at (302, 540) stays at (302, 540): no axis is rescaled.
    assert heat_canvas[0, 540, 302].item() == 1.0
    assert mask_canvas[0, 100:700, 50:550].sum().item() == mask[0].sum().item()
    assert float(heat_canvas[:, 1080:, :].sum()) == 0.0, "padding must hold no signal"


def test_a_resampled_canvas_keeps_geometry_and_rgb_on_the_same_grid():
    spec = letterbox(TIKTOK_HW, 512)
    heatmap = torch.zeros(25, *TIKTOK_HW)
    heatmap[..., 540, 302] = 1
    moved = spec.stack(heatmap, nearest=True)
    assert moved.shape == (25, 512, 512)
    scale_x, scale_y = spec.scale_xy
    offset_x, offset_y = spec.offset_xy
    expected = (int(round(302 * scale_x)) + offset_x, int(round(540 * scale_y)) + offset_y)
    assert moved[0].flatten().argmax().item() == expected[1] * 512 + expected[0], expected


def test_source_and_target_can_differ_in_aspect_without_sharing_a_transform():
    source = native_pad(TIKTOK_HW)
    target = native_pad(PORTRAIT_HW)
    assert source.token_hw == (68, 38)
    assert target.token_hw == (84, 48)
    assert source.token_count != target.token_count
    assert source.fingerprint() != target.fingerprint()
    require_canonical(source, where="training")
    require_canonical(target, where="training")


def test_metadata_round_trips_and_fingerprint_binds_the_canvas():
    spec = aspect_preserving(PORTRAIT_HW, max_pixels=768 * 768)
    assert spec.canvas_hw == (1024, 608) and spec.offset_xy == (13, 4)
    record = spec.metadata()
    assert CanvasSpec.from_record(record) == spec
    assert spec.fingerprint() == CanvasSpec.from_record(record).fingerprint()
    for key, value in (("canvas_hw", [1344, 800]), ("offset_xy", [16, 0]), ("stride", 32)):
        mutated = CanvasSpec.from_record({**record, key: value})
        assert mutated.fingerprint() != spec.fingerprint(), key
    with pytest.raises(CanvasPolicyError):
        CanvasSpec.from_record({**record, "policy": "resize_square"})
    with pytest.raises(CanvasPolicyError):
        CanvasSpec.from_record({"version": LegacyStretch.VERSION, "policy": RESIZE_SQUARE,
                               "original_hw": [1080, 604], "canvas_hw": [512, 512]})


def test_a_pil_size_misread_as_hw_is_caught_by_the_image_check():
    spec = native_pad(TIKTOK_HW)
    assert spec.canvas_hw == (1088, 608)
    assert spec.image(_gradient(1080, 604)).size == (608, 1088)
    with pytest.raises(ValueError, match="does not match original_hw"):
        spec.image(_gradient(604, 1080))
    with pytest.raises(ValueError):
        native_pad((0, 604))
    with pytest.raises(TypeError):
        native_pad("1080x604")


def test_cache_binding_keeps_existing_native_caches_valid(tmp_path):
    """A declared canvas reuses the on-disk namespace, and legacy caches stay resumable."""
    import json

    from src.pose_control.qwen21_full.qwen_runtime import Qwen21IPERCache

    native = Qwen21IPERCache(tmp_path / "native", resolution="native", canvas_policy="native")
    assert native.preprocessing == "native_v1", "must not force a cache rebuild of running experiments"
    assert native.canvas_policy == "native_pad"
    recorded = json.loads((tmp_path / "native" / "cache_policy.json").read_text())
    assert recorded["canvas_policy"] == "native_pad"
    assert recorded["preprocessing"] == "native_v1"

    with pytest.raises(ValueError, match="stretches or crops"):
        Qwen21IPERCache(tmp_path / "bad", canvas_policy="resize")
    with pytest.raises(ValueError, match="unknown canvas policy"):
        Qwen21IPERCache(tmp_path / "worse", canvas_policy="magic_crop")
    with pytest.raises(ValueError, match="differs"):
        Qwen21IPERCache(tmp_path / "native", resolution="native", canvas_policy="letterbox")

    # Cache directories written before the contract carry no canvas key; reopening one must
    # still succeed, otherwise this change would invalidate runs already in flight.
    from src.pose_control.qwen21_full.qwen_runtime import MODEL_REVISION, PROMPT

    pre_contract = tmp_path / "pre_contract"
    pre_contract.mkdir()
    (pre_contract / "cache_policy.json").write_text(json.dumps({
        "preprocessing": "native_v1", "resolution": "native",
        "model_revision": MODEL_REVISION, "prompt": PROMPT}), encoding="utf-8")
    reopened = Qwen21IPERCache(pre_contract, resolution="native", preprocessing="native_v1")
    assert reopened.canvas_policy is None
    assert "canvas_policy" not in json.loads(
        (pre_contract / "cache_policy.json").read_text(encoding="utf-8"))


def test_a_checkpoint_records_which_canvas_it_was_trained_under():
    from src.pose_control.qwen21_full.canvas_provenance import (
        CANVAS_SCHEMA_VERSION, require_canvas_compatible)

    fresh = {"architecture": "qwen21_full_v1", "canvas_schema_version": CANVAS_SCHEMA_VERSION,
             "canvas_policy": "aspect-preserving", "max_pixels": 65536}
    assert require_canvas_compatible(fresh) == CANVAS_SCHEMA_VERSION
    # Pre-contract checkpoints may still be used, but only as a declared warm start.
    legacy = {"architecture_version": "v6.5", "preprocessing_schema_version": "1"}
    with pytest.raises(ValueError, match="warm start"):
        require_canvas_compatible(legacy, where="native canvas training")
    assert require_canvas_compatible(legacy, warm_start_ok=True) == "pre_canvas_warm_start"
    assert require_canvas_compatible(None, warm_start_ok=True) == "pre_canvas_warm_start"


def test_unsupported_mode_names_are_rejected():
    for mode in ("", "nativepad", "resize-square", "crop", None):
        with pytest.raises(ValueError):
            canvas_for_mode(mode, TIKTOK_HW, resolution=512)


def test_legacy_record_cannot_masquerade_as_a_canvas():
    with pytest.raises(CanvasPolicyError):
        CanvasSpec.from_record(LegacyStretch(RESIZE_SQUARE, TIKTOK_HW, (512, 512)).metadata())
    with pytest.raises(CanvasPolicyError, match="not aspect-preserving"):
        CanvasSpec(TIKTOK_HW, TIKTOK_HW, (1088, 608), policy="resize_square")


def test_sampling_size_follows_the_portrait_canvas_not_a_square_default():
    # Plan 4: eval and run scripts must share the training CanvasSpec. Sampling used to be
    # hard-coded width=height=resolution, which redraws a portrait as a square.
    from src.data.canvas_contract import CanvasPolicyError, sampling_size

    spec = aspect_preserving(TIKTOK_HW, max_pixels=65536)
    width, height = sampling_size({"target_canvas": spec.metadata()})
    assert (width, height) == (spec.canvas_hw[1], spec.canvas_hw[0])
    assert width != height, "a portrait canvas must not come back square"
    assert sampling_size({}, fallback_side=512) == (512, 512)
    with pytest.raises(CanvasPolicyError, match="no target_canvas"):
        sampling_size({})
    legacy = LegacyStretch(RESIZE_SQUARE, TIKTOK_HW, (512, 512))
    with pytest.raises(CanvasPolicyError, match="refusing to sample|cannot masquerade"):
        sampling_size({"target_canvas": legacy.metadata()})
    assert sampling_size({"target_canvas": legacy.metadata()},
                         fallback_side=512) == (512, 512)


def test_transformer_inputs_refuses_an_inferred_reference_grid():
    import torch

    from src.pose_control.qwen21_full.qwen_runtime import transformer_inputs

    def pair(**extra):
        base = {"prompt_embeds": torch.zeros(1, 3, 8),
                "prompt_mask": torch.ones(1, 3, dtype=torch.bool),
                "img_mask": torch.ones(1, 5, dtype=torch.bool),
                "source_latent": torch.zeros(1, 96, 64)}
        base.update(extra)
        return base

    declared = transformer_inputs(pair(target_token_hw=(16, 10), source_token_hw=(12, 8)),
                                  torch.zeros(1, 160, 64), torch.tensor([.5]))
    assert declared["img_shapes"] == [[(1, 12, 8), (1, 16, 10)]]
    with pytest.raises(ValueError, match="source_token_hw is required"):
        transformer_inputs(pair(target_token_hw=(16, 10)), torch.zeros(1, 160, 64),
                           torch.tensor([.5]))
    with pytest.raises(ValueError, match="does not match the reference latent"):
        transformer_inputs(pair(target_token_hw=(16, 10), source_token_hw=(16, 9)),
                           torch.zeros(1, 160, 64), torch.tensor([.5]))
    agreeing = transformer_inputs(pair(target_token_hw=(16, 10),
                                       source_latent=torch.zeros(1, 160, 64)),
                                  torch.zeros(1, 160, 64), torch.tensor([.5]))
    assert agreeing["img_shapes"] == [[(1, 16, 10), (1, 16, 10)]]
