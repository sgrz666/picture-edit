from __future__ import annotations

import copy

import numpy as np
from PIL import Image
import pytest
import torch


def api():
    from src.pose_control.qwen21_full import depth_condition
    return depth_condition


def provenance():
    return [{"source_id": "pair-1", "camera_id": "camera-1", "canvas_id": "canvas-1",
             "units": "unknown", "scale_status": "unknown"}]


def canvas():
    return {"original_hw": [32, 16], "resized_hw": [32, 16],
            "canvas_hw": [32, 32], "offset_xy": [0, 0]}


def test_camera_z_near_one_far_zero_and_background_distinct():
    z = torch.zeros(1, 1, 32, 32)
    z[..., 0, :3] = torch.tensor([2., 3., 4.])
    c = api().camera_depth_condition(z, z > 0, torch.ones_like(z, dtype=torch.bool), provenance())
    assert torch.equal(c.relative[0, 0, 0, :4], torch.tensor([1., .5, 0., 0.]))
    assert torch.equal(c.valid[0, 0, 0, :4], torch.tensor([True, True, True, False]))
    assert torch.equal(c.camera_z, z)
    assert c.metadata[0]["bounds"] == [2., 4.]
    assert c.metadata[0]["metric"] is False
    assert c.validate(target_token_hw=(2, 2)) is c


def test_people_share_depth_scale_and_nearest_surface():
    z = torch.stack([torch.full((32, 32), 2.), torch.full((32, 32), 4.)])[None]
    valid = torch.zeros_like(z, dtype=torch.bool)
    valid[:, 0, :, :16] = True
    valid[:, 1, :, 8:] = True
    c = api().camera_depth_condition(z, valid, torch.ones(1, 1, 32, 32, dtype=torch.bool), provenance())
    assert torch.equal(c.relative[..., :16], torch.ones(1, 1, 32, 16))
    assert torch.equal(c.relative[..., 16:], torch.zeros(1, 1, 32, 16))
    assert c.valid.all()
    assert c.metadata[0]["normalization_scope"] == "scene"


def test_constant_plane_is_not_missing():
    z = torch.full((1, 1, 32, 32), 4.)
    c = api().camera_depth_condition(z, z > 0, z > 0, provenance())
    assert c.available.tolist() == [True]
    assert c.relative.eq(.5).all()


def test_all_invalid_is_explicit_and_invalid_sentinels_do_not_leak():
    z = torch.full((1, 1, 32, 32), float("nan"))
    c = api().camera_depth_condition(z, torch.zeros_like(z, dtype=torch.bool),
                                     torch.ones_like(z, dtype=torch.bool), provenance())
    assert c.available.tolist() == [False]
    assert c.relative.eq(0).all() and c.camera_z.eq(0).all()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0., -1.])
def test_bad_valid_camera_z_rejected(value):
    z = torch.ones(1, 1, 32, 32)
    z[..., 0, 0] = value
    with pytest.raises(ValueError):
        api().camera_depth_condition(z, torch.ones_like(z, dtype=torch.bool),
                                     torch.ones_like(z, dtype=torch.bool), provenance())


def test_relative_zero_foreground_and_explicit_polarity():
    d = torch.zeros(1, 1, 32, 32)
    valid = torch.ones_like(d, dtype=torch.bool)
    a = api().relative_depth_condition(d, valid, valid, provenance(), polarity="near_one")
    b = api().relative_depth_condition(d, valid, valid, provenance(), polarity="near_zero")
    assert a.available.all() and a.valid.all() and a.relative.eq(0).all()
    assert b.relative.eq(1).all()
    assert a.camera_z is None and a.metadata[0]["metric"] is False
    with pytest.raises(ValueError, match="polarity"):
        api().relative_depth_condition(d, valid, valid, provenance(), polarity="unknown")


def test_canvas_valid_is_distinct_and_depth_cannot_exist_in_padding():
    d = torch.ones(1, 1, 32, 32)
    valid = d.bool()
    cv = valid.clone()
    cv[..., 16:] = False
    with pytest.raises(ValueError, match="canvas"):
        api().relative_depth_condition(d, valid, cv, provenance(), polarity="near_one")


def test_metadata_and_grid_are_strict():
    d = torch.ones(1, 1, 32, 64)
    meta = provenance()
    del meta[0]["camera_id"]
    with pytest.raises(ValueError, match="camera_id"):
        api().relative_depth_condition(d, d.bool(), d.bool(), meta, polarity="near_one")
    c = api().relative_depth_condition(d, d.bool(), d.bool(), provenance(), polarity="near_one")
    with pytest.raises(ValueError, match="grid"):
        c.validate(target_token_hw=(4, 2))
    with pytest.raises(ValueError, match="canvas"):
        c.validate(expected_canvas_ids=("another-canvas",))
    assert c.to("cpu").valid.dtype == torch.bool


def test_metric_flag_needs_units_and_calibration():
    z = torch.ones(1, 1, 32, 32)
    meta = provenance()
    meta[0].update(units="metre", scale_status="calibrated")
    assert api().camera_depth_condition(z, z.bool(), z.bool(), meta).metadata[0]["metric"] is True


def test_png_and_npy_readers_have_same_direction_no_fallback(tmp_path):
    z = np.zeros((32, 16), np.float32)
    z[:, :8] = 2
    z[:, 8:] = 4
    np.save(tmp_path / "depth.npy", z)
    Image.fromarray(np.where(z == 2, 255, 0).astype(np.uint8)).save(tmp_path / "depth.png")
    Image.fromarray(np.full(z.shape, 255, np.uint8)).save(tmp_path / "valid.png")
    base = {"valid_path": str(tmp_path / "valid.png"), "source_id": "pair-1", "camera_id": "camera-1",
            "units": "unknown", "scale_status": "unknown"}
    raw = api().load_depth_record({**base, "kind": "camera_z", "depth_path": str(tmp_path / "depth.npy")}, canvas())
    png = api().load_depth_record({**base, "kind": "relative_png", "polarity": "near_one",
                                  "depth_path": str(tmp_path / "depth.png")}, canvas())
    assert torch.equal(raw.relative, png.relative)
    assert torch.equal(raw.valid, png.valid)
    assert raw.relative.shape == (1, 1, 32, 32)
    assert raw.canvas_valid.sum() == 32 * 16
    assert raw.metadata[0]["source_sha256"]
    bad = {**base, "kind": "camera_z", "depth_path": str(tmp_path / "missing.npy")}
    with pytest.raises(FileNotFoundError):
        api().load_depth_record(bad, canvas())


def test_masked_resample_does_not_turn_background_into_near_depth(tmp_path):
    z = np.zeros((64, 32), np.float32)
    z[:, :16] = 4
    np.save(tmp_path / "z.npy", z)
    Image.fromarray((z > 0).astype(np.uint8) * 255).save(tmp_path / "valid.png")
    record = {"kind": "camera_z", "depth_path": str(tmp_path / "z.npy"),
              "valid_path": str(tmp_path / "valid.png"), **provenance()[0]}
    del record["canvas_id"]
    spec = {"original_hw": [64, 32], "resized_hw": [32, 16], "canvas_hw": [32, 32], "offset_xy": [0, 0]}
    c = api().load_depth_record(record, spec)
    assert c.camera_z[c.valid].eq(4).all()
    assert c.relative[c.valid].eq(.5).all()
    bad = copy.deepcopy(spec)
    bad["resized_hw"] = [32, 32]
    with pytest.raises(ValueError, match="aspect"):
        api().load_depth_record(record, bad)
