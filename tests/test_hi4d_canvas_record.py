"""Hi4D controls must declare and match the canvas they were rasterised on.

1389 control caches already exist on the server, so the guard has to catch a changed canvas
without forcing a recompute: new writes carry an explicit ``canvas_hw``, and pre-contract files
are checked through the shapes of their own tensors.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.data import hi4d_qwen21_dataset as hi4d


def row(tmp_path, index=0):
    return {
        "target_geometry_npz": str(tmp_path / f"geo{index}.npz"),
        "target_dwpose_A": str(tmp_path / f"dwA{index}.json"),
        "target_dwpose_B": str(tmp_path / f"dwB{index}.json"),
        "target_action": "hug",
        "source_rgb": str(tmp_path / "src.png"),
        "target_rgb": str(tmp_path / "tgt.png"),
        "source_person_mask_A": str(tmp_path / "mA.png"),
        "source_person_mask_B": str(tmp_path / "mB.png"),
    }


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    # The canvas guard runs before any pixel is read, so RGB/mask loading is stubbed: this test
    # must not depend on image files existing.
    monkeypatch.setattr(hi4d, "_rgb", lambda path: torch.zeros(3, hi4d.SIZE, hi4d.SIZE))
    monkeypatch.setattr(hi4d, "_mask", lambda path: torch.zeros(1, hi4d.SIZE, hi4d.SIZE))
    rows = [row(tmp_path)]
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    built = hi4d.Hi4DQwen21Dataset(manifest, tmp_path / "cache", load_target=False)
    built.cache_dir.mkdir(parents=True, exist_ok=True)
    return built


def save(dataset, tensors) -> None:
    torch.save(tensors, dataset.control_path(dataset.rows[0]))


def spatial(size=512, **extra):
    """The 24-key vocabulary observed in the 1389 caches on the server, measured 2026-10-22.

    Every square tensor larger than 16 px in that vocabulary is a canvas raster, which is what
    makes the pre-contract shape fallback safe to run over the existing files.
    """
    tensors = {
        "normal_a": torch.zeros(3, size, size), "normal_b": torch.zeros(3, size, size),
        "depth_a": torch.zeros(1, size, size), "depth_b": torch.zeros(1, size, size),
        "human_mask": torch.zeros(1, size, size), "human_mask_a": torch.zeros(1, size, size),
        "human_mask_b": torch.zeros(1, size, size), "contact_raster": torch.zeros(2, size, size),
        "part_onehot_a": torch.zeros(14, size, size), "part_onehot_b": torch.zeros(14, size, size),
        "pose_heatmap_a": torch.zeros(25, size, size), "pose_heatmap_b": torch.zeros(25, size, size),
        "relative_geometry": torch.zeros(11), "smplx_global_a": torch.zeros(26),
        "smplx_global_b": torch.zeros(26), "contact_distance": torch.zeros(8, 1),
        "contact_contact_type": torch.zeros(8, dtype=torch.long),
        "contact_src_part": torch.zeros(8, dtype=torch.long),
        "contact_dst_part": torch.zeros(8, dtype=torch.long),
        "contact_src_person": torch.zeros(8, dtype=torch.long),
        "contact_dst_person": torch.zeros(8, dtype=torch.long),
        "contact_valid_mask": torch.zeros(8, dtype=torch.bool),
        "task_id": torch.tensor(0, dtype=torch.long), "person_b_valid": torch.tensor(True),
    }
    tensors.update(extra)
    return tensors


def test_a_matching_canvas_record_loads_and_is_consumed(dataset):
    save(dataset, spatial(canvas_hw=torch.tensor([512, 512]), native_hw=torch.tensor([96, 128])))
    sample = dataset[0]
    assert sample["normal_a"].shape[-2:] == (512, 512)
    assert "canvas_hw" not in sample, "the record is verified and consumed, not passed downstream"
    assert "native_hw" not in sample


def test_a_canvas_record_disagreeing_with_the_build_is_refused(dataset):
    save(dataset, spatial(canvas_hw=torch.tensor([768, 768]), native_hw=torch.tensor([96, 128])))
    with pytest.raises(ValueError, match=r"letterboxed onto \[768, 768\]"):
        dataset[0]


def test_a_pre_contract_cache_is_still_caught_through_tensor_shapes(dataset):
    save(dataset, spatial(size=384))
    with pytest.raises(ValueError, match=r"sits on \(384, 384\)"):
        dataset[0]


def test_the_production_vocabulary_opens_and_non_spatial_keys_are_left_alone(dataset):
    # Mirrors the 1389 files on disk: no canvas_hw at all, so only the shape fallback protects them.
    save(dataset, spatial())
    sample = dataset[0]
    assert "canvas_hw" not in sample and "native_hw" not in sample
    for key in ("relative_geometry", "contact_distance", "contact_valid_mask", "task_id",
                "person_b_valid", "smplx_global_a"):
        assert sample[key].shape in {(), (8,), (8, 1), (11,), (26,)}, (key, tuple(sample[key].shape))
    assert sample["contact_raster"].shape == (2, hi4d.SIZE, hi4d.SIZE)


def test_letterbox_keeps_the_source_aspect_instead_of_stretching_it():
    # Hi4D was never part of the square-stretch hazard: the canvas is square, the content is not.
    content = np.full((128, 96), 255, dtype=np.uint8)  # height 128, width 96 -> portrait
    boxed = hi4d.letterbox_array(content)
    assert boxed.shape[:2] == (hi4d.SIZE, hi4d.SIZE)
    filled = boxed > 0
    rows, columns = np.nonzero(filled)
    out_h, out_w = rows.max() - rows.min() + 1, columns.max() - columns.min() + 1
    assert (out_w, out_h) == (384, 512)
    assert out_w / out_h == pytest.approx(96 / 128, rel=1e-3), \
        "a stretched Hi4D control would report a square content box here"
