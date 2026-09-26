import importlib.util
from importlib.machinery import ModuleSpec
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pytest
import torch
from PIL import Image

if "diffusers" in sys.modules:
    if getattr(sys.modules["diffusers"], "__spec__", None) is None:
        sys.modules["diffusers"].__spec__ = ModuleSpec("diffusers", loader=None)
    if "diffusers.models" in sys.modules and getattr(
        sys.modules["diffusers.models"], "__spec__", None
    ) is None:
        sys.modules["diffusers.models"].__spec__ = ModuleSpec(
            "diffusers.models", loader=None
        )
else:
    try:
        _diffusers_spec = importlib.util.find_spec("diffusers")
    except ValueError:
        _diffusers_spec = None

if "diffusers" not in sys.modules and _diffusers_spec is None:
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

from scripts.build_tiktok_v6_conditions import (
    axis_angle_to_rotation_6d,
    extract_25_keypoints_from_smplx_joints,
)
from scripts import build_tiktok_v6_conditions as tiktok_conditions_builder
from scripts.precompute_tiktok_face_features_v65 import (
    CACHE_SCHEMA_VERSION,
    atomic_save_cache,
    build_tiktok_v65_face_cache,
    compute_cache_fingerprint,
    compute_face_box,
    load_v65_face_cache,
    project_tiktok_face72_to_rgb_normalized,
    validate_v65_face_cache,
)
from src.data.tiktok_dataset import (
    TikTokPoseDataset,
    TikTokV65FaceOverfitLoader,
    TikTokV65OverfitInputs,
    build_tiktok_face_inputs,
)
from src.data import tiktok_dataset as tiktok_data_module
from src.pose_control.v6.face.conditions import FaceFineCondition, FaceReferenceFeatures


def test_tiktok_condition_cli_help() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    res = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "build_tiktok_v6_conditions.py"), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    assert "Preprocess TikTok dataset for V6.3 Native Adapter" in res.stdout


def test_tiktok_face_precompute_cli_help() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    res = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "precompute_tiktok_face_features_v65.py"), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    assert "Precompute V6.5 TikTok face features" in res.stdout


def test_axis_angle_to_rotation_6d() -> None:
    # Identity rotation: axis-angle (0, 0, 0)
    aa = torch.zeros(2, 3, dtype=torch.float32)
    rot6d = axis_angle_to_rotation_6d(aa)
    assert rot6d.shape == (2, 6)
    # First 3 should be column 0 (1, 0, 0), second 3 column 1 (0, 1, 0)
    expected = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    assert torch.allclose(rot6d, expected, atol=1e-5)


def test_extract_25_keypoints_from_smplx_joints() -> None:
    # 55 SMPL-X joints in camera space (e.g. at z=2.0)
    joints_3d = np.zeros((55, 3), dtype=np.float32)
    joints_3d[:, 2] = 2.0  # z = 2.0
    joints_3d[15] = [0.0, -0.5, 2.0]  # Head

    fx, fy, cx, cy = 1000.0, 1000.0, 302.0, 540.0
    w, h = 604.0, 1080.0
    kps = extract_25_keypoints_from_smplx_joints(joints_3d, fx, fy, cx, cy, w, h)
    assert kps.shape == (25, 3)
    # Head joint (index 0 in OpenPose 25)
    u = 1000.0 * 0.0 / 2.0 + 302.0  # 302.0
    v = 1000.0 * (-0.5) / 2.0 + 540.0  # 290.0
    assert np.isclose(kps[0, 0], 302.0 / 604.0, atol=1e-4)
    assert np.isclose(kps[0, 1], 290.0 / 1080.0, atol=1e-4)
    assert kps[0, 2] == 1.0


def test_body25_projection_uses_named_smplestx_joints_not_mesh_vertices() -> None:
    joints = torch.full((1, 137, 2), 6.0)
    joints[..., 1] = 8.0
    joints[0, 24] = torch.tensor([6.0, 4.0])   # nose
    joints[0, 13] = torch.tensor([3.0, 6.0])   # right wrist
    joints[0, 12] = torch.tensor([9.0, 6.0])   # left wrist
    joints[0, 0] = torch.tensor([6.0, 12.0])   # pelvis
    joints[0, 5] = torch.tensor([6.0, 100.0])  # left ankle outside frame
    params = {
        "joint_proj_model": joints,
        "focal_length_xy": torch.tensor([[2500.0, 2500.0]]),
        "princpt": torch.tensor([[302.0, 540.0]]),
        "width": torch.tensor([[604.0]]),
        "height": torch.tensor([[1080.0]]),
    }

    points = tiktok_conditions_builder.project_smplestx_body25_to_openpose(params, 0)

    assert points.shape == (25, 3)
    assert np.isclose(points[0, 1], (540.0 - 64.0 + 32.0) / 1080.0)  # nose from index 24
    assert points[4, 0] < points[7, 0]  # anatomical right/left wrist
    assert np.isclose(points[8, 1], (540.0 + 32.0) / 1080.0)  # pelvis from index 0
    assert points[14, 2] == 0.0  # out-of-frame left ankle


def test_tiktok_single_frame_override_changes_only_requested_conditions(tmp_path: Path) -> None:
    batch = {
        "pose_heatmap": torch.zeros(1, 25, 32, 32),
        "part_onehot": torch.zeros(1, 14, 32, 32),
        "src_image": torch.ones(1, 3, 32, 32),
    }
    points = torch.zeros(25, 3)
    points[0] = torch.tensor([0.5, 0.5, 1.0])
    part = torch.full((32, 32), 5, dtype=torch.uint8)
    path = tmp_path / "condition_override.pt"
    torch.save({"sequence": "00001", "target_stem": "0074", "kps_25_coords": points,
                "part_512": part}, path)

    updated = tiktok_data_module.apply_tiktok_condition_override(
        batch, path, sequence="00001", target_stem="0074", resolution=32, heatmap_sigma=2.0
    )

    assert updated["pose_heatmap"][0, 0].max() > 0.9
    assert updated["part_onehot"][0, 4].sum() == 32 * 32
    assert torch.equal(updated["src_image"], batch["src_image"])
    assert batch["pose_heatmap"].count_nonzero() == 0
    with pytest.raises(ValueError, match="target_stem"):
        tiktok_data_module.apply_tiktok_condition_override(
            batch, path, sequence="00001", target_stem="0014", resolution=32, heatmap_sigma=2.0
        )


def test_tiktok_sequence_builder_uses_projected_body_joints(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    assets_root = tmp_path / "assets"
    smplestx = assets_root / "00001" / "smplestx"
    smplestx.mkdir(parents=True)
    (smplestx / "frame_map.json").write_text(
        json.dumps({"frames": [{"stem": "0074", "index": 0}]}), encoding="utf-8"
    )
    joints = torch.full((1, 137, 2), 6.0)
    joints[..., 1] = 8.0
    joints[0, 24, 1] = 4.0
    params = {
        "mesh_cam": torch.zeros(1, 10475, 3),
        "betas": torch.zeros(1, 10),
        "global_orient": torch.zeros(1, 3),
        "transl": torch.zeros(1, 3),
        "joint_proj_model": joints,
        "focal_length_xy": torch.tensor([[2500.0, 2500.0]]),
        "princpt": torch.tensor([[302.0, 540.0]]),
        "width": torch.tensor([[604.0]]),
        "height": torch.tensor([[1080.0]]),
    }
    torch.save(params, smplestx / "params.pt")

    assert tiktok_conditions_builder.process_tiktok_sequence(
        "00001", raw_root, assets_root, renderer=None
    )
    bundle = torch.load(assets_root / "00001" / "v6_conditions.pt", weights_only=True)
    expected = tiktok_conditions_builder.project_smplestx_body25_to_openpose(params, 0)
    assert np.allclose(bundle["kps_25_coords"][0].numpy(), expected)


def test_single_frame_override_writer_does_not_touch_sequence_cache(tmp_path: Path) -> None:
    assets = tmp_path / "assets" / "00001"
    smplestx = assets / "smplestx"
    smplestx.mkdir(parents=True)
    (smplestx / "frame_map.json").write_text(
        json.dumps({"frames": [{"stem": "0074", "index": 0}]}), encoding="utf-8"
    )
    joints = torch.full((1, 137, 2), 6.0)
    joints[..., 1] = 8.0
    torch.save({
        "joint_proj_model": joints,
        "focal_length_xy": torch.tensor([[2500.0, 2500.0]]),
        "princpt": torch.tensor([[302.0, 540.0]]),
        "width": torch.tensor([[604.0]]),
        "height": torch.tensor([[1080.0]]),
    }, smplestx / "params.pt")
    existing = assets / "v6_conditions.pt"
    existing.write_bytes(b"old user cache")
    output = tmp_path / "experiment" / "pose_only.pt"

    points = tiktok_conditions_builder.write_single_frame_override(
        assets, sequence="00001", target_stem="0074", output_path=output
    )

    saved = torch.load(output, weights_only=True)
    assert saved["target_stem"] == "0074"
    assert np.allclose(saved["kps_25_coords"].numpy(), points)
    assert existing.read_bytes() == b"old user cache"
    with pytest.raises(FileExistsError):
        tiktok_conditions_builder.write_single_frame_override(
            assets, sequence="00001", target_stem="0074", output_path=output
        )


def test_project_tiktok_face72_joint_proj_model() -> None:
    frame_count = 2
    joints = torch.zeros(frame_count, 137, 2, dtype=torch.float32)
    ramp = torch.linspace(0.0, 1.0, 72)
    joints[:, 65:, 0] = ramp * 12.0
    joints[:, 65:, 1] = ramp * 16.0
    params = {
        "joint_proj_model": joints,
        "focal_length_xy": torch.tensor([[2500.0, 2500.0]]).repeat(frame_count, 1),
        "princpt": torch.tensor([[302.0, 540.0]]).repeat(frame_count, 1),
        "width": torch.full((frame_count,), 604.0),
        "height": torch.full((frame_count,), 1080.0),
    }
    landmarks = project_tiktok_face72_to_rgb_normalized(params)
    assert landmarks.shape == (frame_count, 72, 3)
    assert torch.isfinite(landmarks).all()
    assert torch.all(landmarks[..., 2] == 1.0)


def test_project_tiktok_face72_mesh_cam() -> None:
    frame_count = 2
    mesh_cam = torch.zeros(frame_count, 10475, 3, dtype=torch.float32)
    mesh_cam[..., 2] = 2.0  # z = 2m
    params = {
        "mesh_cam": mesh_cam,
        "focal_length_xy": torch.tensor([[1000.0, 1000.0]]).repeat(frame_count, 1),
        "princpt": torch.tensor([[302.0, 540.0]]).repeat(frame_count, 1),
        "width": torch.tensor([604.0]),
        "height": torch.tensor([1080.0]),
    }
    landmarks = project_tiktok_face72_to_rgb_normalized(params)
    assert landmarks.shape == (frame_count, 72, 3)
    assert torch.isfinite(landmarks).all()
    assert torch.all((landmarks[..., :2] >= 0.0) & (landmarks[..., :2] <= 1.0))


def test_compute_face_box() -> None:
    # Valid face landmarks
    lmk = torch.zeros(72, 3)
    lmk[:, 0] = torch.linspace(0.4, 0.6, 72)
    lmk[:, 1] = torch.linspace(0.2, 0.4, 72)
    lmk[:, 2] = 1.0
    box, valid = compute_face_box(lmk)
    assert valid is True
    assert box.shape == (4,)
    assert (box[2] > box[0]) and (box[3] > box[1])
    # Box should be square: (x2 - x1) == (y2 - y1) within clamp boundaries
    w = box[2] - box[0]
    h = box[3] - box[1]
    assert torch.isclose(w, h, atol=1e-4)

    # Invalid face landmarks
    lmk_inv = torch.zeros(72, 3)  # all confidence = 0
    box_inv, valid_inv = compute_face_box(lmk_inv)
    assert valid_inv is False
    assert torch.all(box_inv == 0.0)


class MockExtractor:
    arcface_model_id = "mock_arcface"
    arcface_revision = "local"
    arcface_weight_sha256 = "mock_arc_sha"
    dino_model_id = "mock_dino"
    dino_revision = "local"
    dino_weight_sha256 = "mock_dino_sha"

    def extract(self, tight: np.ndarray, expanded: np.ndarray, full_image: Any = None):
        return torch.ones(512), torch.ones(256, 1536)


class FailingMockExtractor(MockExtractor):
    def extract(self, tight: np.ndarray, expanded: np.ndarray, full_image: Any = None):
        raise RuntimeError("Extractor device failure")


def test_build_tiktok_v65_face_cache_and_serialization(tmp_path: Path) -> None:
    frame_count = 2
    joints = torch.zeros(frame_count, 137, 2, dtype=torch.float32)
    ramp = torch.linspace(0.0, 1.0, 72)
    joints[:, 65:, 0] = ramp * 12.0
    joints[:, 65:, 1] = ramp * 16.0
    params = {
        "joint_proj_model": joints,
        "focal_length_xy": torch.tensor([[2500.0, 2500.0]]).repeat(frame_count, 1),
        "princpt": torch.tensor([[302.0, 540.0]]).repeat(frame_count, 1),
        "width": torch.full((frame_count,), 604.0),
        "height": torch.full((frame_count,), 1080.0),
        "jaw_pose": torch.zeros(frame_count, 3),
        "expression": torch.zeros(frame_count, 10),
    }
    stems = ["0001", "0002"]

    def dummy_loader(stem: str) -> np.ndarray:
        return np.zeros((1080, 604, 3), dtype=np.uint8)

    cache = build_tiktok_v65_face_cache(
        params=params,
        frame_stems=stems,
        image_loader=dummy_loader,
        extractor=MockExtractor(),
        source_hashes={"test": "abc"},
        sequence_id="00001",
    )
    validate_v65_face_cache(cache)
    assert cache["arcface"].shape == (2, 512)
    assert cache["dino_patches"].shape == (2, 256, 1536)
    assert cache["valid"].all()

    # Roundtrip save/load
    dest = tmp_path / "v65_face_features.pt"
    atomic_save_cache(cache, dest)
    loaded = load_v65_face_cache(dest)
    assert loaded["metadata"]["cache_fingerprint"] == cache["metadata"]["cache_fingerprint"]


def test_build_tiktok_face_inputs() -> None:
    cache = {
        "metadata": {"schema_version": 1},
        "frame_stems": ["0014", "0074"],
        "stem_to_idx": {"0014": 0, "0074": 1},
        "valid": torch.tensor([True, True]),
        "face_landmarks": torch.rand(2, 72, 3),
        "jaw_pose": torch.rand(2, 3),
        "expression": torch.rand(2, 10),
        "face_boxes": torch.tensor([[0.2, 0.1, 0.5, 0.4], [0.25, 0.15, 0.55, 0.45]]),
        "arcface": torch.rand(2, 512, dtype=torch.float16),
        "dino_patches": torch.rand(2, 256, 1536, dtype=torch.float16),
    }
    cond, refs = build_tiktok_face_inputs(cache, source_frame="0014", target_frame="0074")
    assert isinstance(cond, FaceFineCondition)
    assert isinstance(refs, FaceReferenceFeatures)
    assert cond.landmarks.shape == (1, 2, 72, 3)
    assert cond.jaw_pose.shape == (1, 2, 3)
    assert cond.expression.shape == (1, 2, 10)
    assert cond.source_boxes.shape == (1, 2, 4)
    assert cond.target_boxes.shape == (1, 2, 4)
    assert bool(cond.face_valid[0, 0]) is True
    assert refs.arcface.shape == (1, 2, 1, 512)
    assert refs.dino_patches.shape == (1, 2, 1, 256, 1536)
    assert bool(refs.reference_valid[0, 0, 0]) is True


def test_tiktok_v65_face_overfit_loader(tmp_path: Path) -> None:
    seq_id = "00001"
    raw_root = tmp_path / "raw"
    assets_root = tmp_path / "assets"
    seq_raw = raw_root / seq_id
    seq_assets = assets_root / seq_id

    (seq_raw / "images").mkdir(parents=True)
    (seq_raw / "masks").mkdir(parents=True)
    (seq_assets / "normal").mkdir(parents=True)
    (seq_assets / "part_512").mkdir(parents=True)
    (seq_assets / "smplestx").mkdir(parents=True)

    # Create dummy images
    dummy_img = Image.new("RGB", (604, 1080), color=(128, 128, 128))
    dummy_mask = Image.new("L", (604, 1080), color=255)
    dummy_part = Image.new("L", (512, 512), color=1)

    for stem in ["0014", "0074"]:
        dummy_img.save(seq_raw / "images" / f"{stem}.png")
        dummy_mask.save(seq_raw / "masks" / f"{stem}.png")
        dummy_img.save(seq_assets / "normal" / f"{stem}.png")
        dummy_part.save(seq_assets / "part_512" / f"{stem}.png")

    # Global bundle
    v6_bundle = {
        "smplx_global": torch.randn(2, 26),
        "kps_25_coords": torch.rand(2, 25, 3),
        "stems": ["0014", "0074"],
        "stem_to_idx": {"0014": 0, "0074": 1},
    }
    torch.save(v6_bundle, seq_assets / "v6_conditions.pt")

    # Face bundle
    cache = {
        "metadata": {"schema_version": 1},
        "frame_stems": ["0014", "0074"],
        "stem_to_idx": {"0014": 0, "0074": 1},
        "valid": torch.tensor([True, True]),
        "face_landmarks": torch.rand(2, 72, 3),
        "jaw_pose": torch.rand(2, 3),
        "expression": torch.rand(2, 10),
        "face_boxes": torch.tensor([[0.2, 0.1, 0.5, 0.4], [0.25, 0.15, 0.55, 0.45]]),
        "arcface": torch.rand(2, 512, dtype=torch.float16),
        "dino_patches": torch.rand(2, 256, 1536, dtype=torch.float16),
    }
    cache["metadata"]["cache_fingerprint"] = compute_cache_fingerprint(cache)
    atomic_save_cache(cache, seq_assets / "v65_face_features.pt")

    loader = TikTokV65FaceOverfitLoader(
        raw_root=raw_root,
        assets_root=assets_root,
        sequence=seq_id,
        source_stem="0014",
        target_stem="0074",
        resolution=512,
    )
    inputs = loader.load()
    assert isinstance(inputs, TikTokV65OverfitInputs)
    b = inputs.batch
    assert b["src_image"].shape == (1, 3, 512, 512)
    assert b["tgt_image"].shape == (1, 3, 512, 512)
    assert b["normal"].shape == (1, 3, 512, 512)
    assert b["part_onehot"].shape == (1, 14, 512, 512)
    assert b["pose_heatmap"].shape == (1, 25, 512, 512)
    assert b["smplx_global"].shape == (1, 26)
    assert b["human_mask"].shape == (1, 1, 512, 512)
    assert b["source_stem"] == ["0014"]
    assert b["target_stem"] == ["0074"]

    assert bool(inputs.face_condition.face_valid[0, 0]) is True
    assert bool(inputs.face_references.reference_valid[0, 0, 0]) is True


def test_smoke_test_runner_dry_run(tmp_path: Path) -> None:
    from run_tiktok_sample_00001_smoke import run_smoke_test
    import argparse

    seq_id = "00001"
    raw_root = tmp_path / "raw"
    assets_root = tmp_path / "assets"
    seq_raw = raw_root / seq_id
    seq_assets = assets_root / seq_id

    (seq_raw / "images").mkdir(parents=True)
    (seq_raw / "masks").mkdir(parents=True)
    (seq_assets / "normal").mkdir(parents=True)
    (seq_assets / "part_512").mkdir(parents=True)

    dummy_img = Image.new("RGB", (604, 1080), color=(128, 128, 128))
    dummy_mask = Image.new("L", (604, 1080), color=255)
    dummy_part = Image.new("L", (512, 512), color=1)

    for stem in ["0014", "0074"]:
        dummy_img.save(seq_raw / "images" / f"{stem}.png")
        dummy_mask.save(seq_raw / "masks" / f"{stem}.png")
        dummy_img.save(seq_assets / "normal" / f"{stem}.png")
        dummy_part.save(seq_assets / "part_512" / f"{stem}.png")

    v6_bundle = {
        "smplx_global": torch.randn(2, 26),
        "kps_25_coords": torch.rand(2, 25, 3),
        "stems": ["0014", "0074"],
        "stem_to_idx": {"0014": 0, "0074": 1},
    }
    torch.save(v6_bundle, seq_assets / "v6_conditions.pt")

    cache = {
        "metadata": {"schema_version": 1},
        "frame_stems": ["0014", "0074"],
        "stem_to_idx": {"0014": 0, "0074": 1},
        "valid": torch.tensor([True, True]),
        "face_landmarks": torch.rand(2, 72, 3),
        "jaw_pose": torch.rand(2, 3),
        "expression": torch.rand(2, 10),
        "face_boxes": torch.tensor([[0.2, 0.1, 0.5, 0.4], [0.25, 0.15, 0.55, 0.45]]),
        "arcface": torch.rand(2, 512, dtype=torch.float16),
        "dino_patches": torch.rand(2, 256, 1536, dtype=torch.float16),
    }
    cache["metadata"]["cache_fingerprint"] = compute_cache_fingerprint(cache)
    atomic_save_cache(cache, seq_assets / "v65_face_features.pt")

    args = argparse.Namespace(
        raw_root=str(raw_root),
        assets_root=str(assets_root),
        sequence=seq_id,
        source_stem="0014",
        target_stem="0074",
        resolution=512,
        output_dir=str(tmp_path / "out"),
        base_checkpoint=str(tmp_path / "nonexistent.pt"),
        steps=2,
        dry_run=True,
        device="cpu",
    )
    report = run_smoke_test(args)
    assert report["sequence"] == "00001"
    assert len(report["steps"]) == 2
    assert (tmp_path / "out" / "preview_smoke.png").is_file()
    assert (tmp_path / "out" / "smoke_report.json").is_file()
