from __future__ import annotations

import importlib.util
from importlib.machinery import ModuleSpec
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image


if "diffusers" in sys.modules:
    # Other V6 tests may have installed the same minimal stub first. Repair its
    # spec so later find_spec calls are order-independent, without replacing it.
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

from scripts.precompute_iper_face_features_v65 import (
    CACHE_SCHEMA_VERSION,
    atomic_save_cache,
    build_v65_face_cache,
    compute_cache_fingerprint,
    compute_face_box,
    load_v65_face_cache,
    parse_args as parse_precompute_args,
    project_smplx_face72_to_rgb_normalized,
    safe_expanded_crop,
)
from train_iper_v65_face import (
    build_face_inputs,
    build_face_optimizer,
    compute_v65_face_losses,
    make_fixed_training_state,
    parse_args as parse_train_args,
    resolve_frame_selector,
    select_face_trainable_parameters,
    update_face_optimizer_lrs,
)
from src.pose_control.v6.face.iper_dataset import IPERV65FaceOverfitLoader


def test_precompute_cli_can_run_directly_from_repository_root() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts" / "precompute_iper_face_features_v65.py"),
            "--help",
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Precompute V6.5 iPER face features" in result.stdout


def _camera_params(frame_count: int = 2) -> dict[str, torch.Tensor]:
    joints = torch.zeros(frame_count, 137, 2, dtype=torch.float32)
    ramp = torch.linspace(0.0, 1.0, 72)
    joints[:, 65:, 0] = ramp * 12.0
    joints[:, 65:, 1] = ramp * 16.0
    return {
        "joint_proj_model": joints,
        "focal_length_xy_raw": torch.tensor([[2500.0, 2500.0]]).repeat(frame_count, 1),
        "princpt_raw": torch.tensor([[100.0, 200.0]]).repeat(frame_count, 1),
        "width": torch.full((frame_count,), 200.0),
        "height": torch.full((frame_count,), 400.0),
        "jaw_pose": torch.arange(frame_count * 3, dtype=torch.float32).reshape(frame_count, 3),
        "expression": torch.arange(frame_count * 10, dtype=torch.float32).reshape(frame_count, 10),
    }


def test_smplx_137_face_slice_and_raw_camera_inverse_are_exact() -> None:
    params = _camera_params(1)
    result = project_smplx_face72_to_rgb_normalized(params)

    assert result.shape == (1, 72, 3)
    assert result.dtype == torch.float32
    # bw=2500*192/5000=96; bx=100-48=52.  bh=128; by=136.
    torch.testing.assert_close(result[0, 0], torch.tensor([52 / 200, 136 / 400, 1.0]))
    torch.testing.assert_close(result[0, -1], torch.tensor([148 / 200, 264 / 400, 1.0]))


def test_retargeted_camera_without_raw_fields_fails_clearly() -> None:
    params = _camera_params(1)
    params.pop("focal_length_xy_raw")
    params.pop("princpt_raw")
    params["retargeted_cam"] = torch.ones(1, dtype=torch.bool)

    with pytest.raises(ValueError, match="retargeted_cam.*raw camera"):
        project_smplx_face72_to_rgb_normalized(params)


def test_face_box_is_square_expanded_and_crop_pads_safely() -> None:
    landmarks = torch.zeros(72, 3)
    landmarks[:, 0] = torch.linspace(0.20, 0.40, 72)
    landmarks[:, 1] = torch.linspace(0.30, 0.50, 72)
    landmarks[:, 2] = 1.0

    box, valid = compute_face_box(landmarks, expansion=0.15)
    assert valid
    torch.testing.assert_close(box, torch.tensor([0.17, 0.27, 0.43, 0.53]), atol=1e-6, rtol=0)

    image = np.full((10, 10, 3), 127, dtype=np.uint8)
    edge_box = torch.tensor([0.0, 0.0, 0.2, 0.2])
    crop = safe_expanded_crop(image, edge_box, scale=1.35)
    assert crop.ndim == 3 and crop.shape[2] == 3
    assert crop.shape[0] == crop.shape[1]
    assert crop.shape[0] >= 3
    assert (crop[0, 0] == 0).all(), "out-of-image area must use explicit safe padding"


class _FakeExtractor:
    arcface_model_id = "fake-arcface"
    arcface_revision = "fake-a"
    arcface_weight_sha256 = "a" * 64
    dino_model_id = "fake-dino"
    dino_revision = "fake-d"
    dino_weight_sha256 = "d" * 64

    def __init__(self, fail_on: int | None = None) -> None:
        self.calls = 0
        self.fail_on = fail_on

    def extract(self, tight_crop: np.ndarray, expanded_crop: np.ndarray):
        call = self.calls
        self.calls += 1
        if call == self.fail_on:
            raise RuntimeError("synthetic extraction failure")
        return (
            torch.full((512,), float(call + 1)),
            torch.full((256, 1536), float(call + 10)),
        )


def _image_loader(_stem: str) -> np.ndarray:
    return np.full((40, 20, 3), 127, dtype=np.uint8)


def test_cache_schema_dtypes_fingerprint_and_failed_frame_zeroing(tmp_path: Path) -> None:
    params = _camera_params(2)
    cache = build_v65_face_cache(
        params=params,
        frame_stems=["source_f000001", "target_f000002"],
        image_loader=_image_loader,
        extractor=_FakeExtractor(fail_on=1),
        source_hashes={"params.pt": "1" * 64, "frame_map.json": "2" * 64},
        appearance="024_6",
    )

    assert cache["metadata"]["schema_version"] == CACHE_SCHEMA_VERSION
    assert cache["metadata"]["appearance"] == "024_6"
    assert cache["frame_stems"] == ["source_f000001", "target_f000002"]
    assert cache["stem_to_idx"] == {"source_f000001": 0, "target_f000002": 1}
    assert cache["arcface"].shape == (2, 512) and cache["arcface"].dtype == torch.float16
    assert cache["dino_patches"].shape == (2, 256, 1536)
    assert cache["dino_patches"].dtype == torch.float16
    assert cache["valid"].dtype == torch.bool
    assert cache["face_landmarks"].shape == (2, 72, 3)
    assert cache["jaw_pose"].shape == (2, 3)
    assert cache["expression"].shape == (2, 10)
    assert cache["face_boxes"].shape == (2, 4)
    assert cache["valid"].tolist() == [True, False]
    assert torch.count_nonzero(cache["arcface"][1]) == 0
    assert torch.count_nonzero(cache["dino_patches"][1]) == 0
    assert cache["metadata"]["cache_fingerprint"] == compute_cache_fingerprint(cache)

    target = tmp_path / "v65_face_features.pt"
    atomic_save_cache(cache, target)
    assert target.exists()
    assert not list(tmp_path.glob("*.tmp"))
    loaded = load_v65_face_cache(target)
    assert loaded["metadata"]["cache_fingerprint"] == cache["metadata"]["cache_fingerprint"]
    torch.testing.assert_close(loaded["arcface"], cache["arcface"])


def test_cache_rejects_invalid_dino_shape() -> None:
    class BadExtractor(_FakeExtractor):
        def extract(self, tight_crop: np.ndarray, expanded_crop: np.ndarray):
            return torch.zeros(512), torch.zeros(257, 1536)

    cache = build_v65_face_cache(
        params=_camera_params(1),
        frame_stems=["frame"],
        image_loader=_image_loader,
        extractor=BadExtractor(),
        source_hashes={},
    )
    assert not cache["valid"][0]
    assert torch.count_nonzero(cache["dino_patches"]) == 0


def _valid_cache() -> dict:
    return build_v65_face_cache(
        params=_camera_params(2),
        frame_stems=["source", "target"],
        image_loader=_image_loader,
        extractor=_FakeExtractor(),
        source_hashes={},
    )


def test_single_reference_two_person_face_inputs_have_invalid_zero_slot() -> None:
    condition, references = build_face_inputs(_valid_cache(), source_frame=0, target_frame=1)
    condition.validate()
    references.validate()

    assert condition.landmarks.shape == (1, 2, 72, 3)
    assert references.arcface.shape == (1, 2, 1, 512)
    assert condition.face_valid.tolist() == [[True, False]]
    assert references.reference_valid.tolist() == [[[True], [False]]]
    torch.testing.assert_close(condition.source_boxes[0, 0], _valid_cache()["face_boxes"][0])
    torch.testing.assert_close(condition.target_boxes[0, 0], _valid_cache()["face_boxes"][1])
    assert condition.source_indices.tolist() == [[0, 1]]
    assert torch.count_nonzero(condition.landmarks[0, 1]) == 0
    assert torch.count_nonzero(condition.jaw_pose[0, 1]) == 0
    assert torch.count_nonzero(condition.expression[0, 1]) == 0
    assert torch.count_nonzero(references.arcface[0, 1]) == 0
    assert torch.count_nonzero(references.dino_patches[0, 1]) == 0


def _write_rgb(path: Path, value: int, *, size: int = 12) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((size, size, 3), value, dtype=np.uint8)).save(path)


def _write_gray(path: Path, value: int, *, size: int = 12) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((size, size), value, dtype=np.uint8)).save(path)


def test_v65_overfit_loader_is_self_contained_and_emits_hand_only_legacy_inputs(
    tmp_path: Path,
) -> None:
    appearance = "024_6"
    source_stem, target_stem = "source_f000001", "target_f000002"
    sampled = tmp_path / "sampled"
    assets = tmp_path / "assets"
    app_sampled = sampled / appearance
    app_assets = assets / appearance
    _write_rgb(app_sampled / "rgb_1024" / "source_motion" / f"{source_stem}.png", 64)
    _write_rgb(app_sampled / "rgb_1024" / "target" / f"{target_stem}.png", 192)
    _write_rgb(app_assets / "normal_512" / "target" / f"{target_stem}.png", 128)
    _write_gray(app_assets / "mask_512" / "target" / f"{target_stem}.png", 255)
    _write_gray(app_assets / "part_512" / "target" / f"{target_stem}.png", 3)

    kps = torch.zeros(2, 25, 3)
    kps[1, 0] = torch.tensor([0.5, 0.5, 1.0])
    torch.save(
        {
            "stem_to_idx": {source_stem: 0, target_stem: 1},
            "smplx_global": torch.arange(52, dtype=torch.float32).reshape(2, 26),
            "kps_25_coords": kps,
        },
        app_assets / "v6_conditions.pt",
    )
    detail = {
        "stem_to_idx": {source_stem: 0, target_stem: 1},
        # Non-zero face values deliberately prove that the V6.5 loader cannot
        # leak the deprecated shared Face payload into the retained Hand branch.
        "face_keypoints": torch.ones(2, 68, 3),
        "hand_keypoints": torch.zeros(2, 2, 21, 3),
        "smplx_detail": torch.zeros(2, 103),
        "boxes": torch.tensor(
            [
                [[0.1, 0.1, 0.4, 0.4], [0.1, 0.2, 0.3, 0.5], [0.6, 0.2, 0.8, 0.5]],
                [[0.2, 0.1, 0.5, 0.4], [0.2, 0.2, 0.4, 0.5], [0.5, 0.2, 0.7, 0.5]],
            ],
            dtype=torch.float32,
        ),
        "region_valid": torch.ones(2, 3, dtype=torch.bool),
        "reference_crops": torch.ones(2, 3, 3, 224, 224),
    }
    detail["hand_keypoints"][1, :, :, :2] = 0.5
    detail["hand_keypoints"][1, :, :, 2] = 1.0
    detail["smplx_detail"][1, 13:103] = torch.arange(90, dtype=torch.float32)
    torch.save(detail, app_assets / "v6_detail_conditions.pt")

    loaded = IPERV65FaceOverfitLoader(
        sampled_root=sampled,
        assets_root=assets,
        appearance=appearance,
        source_stem=source_stem,
        source_role="source_motion",
        target_stem=target_stem,
        target_role="target",
        resolution=16,
    ).load()

    assert set(loaded.batch) == {
        "src_image", "tgt_image", "normal", "part_onehot", "pose_heatmap",
        "smplx_global", "human_mask", "task_id", "appearance", "source_stem",
        "target_stem",
    }
    assert loaded.batch["src_image"].shape == (1, 3, 16, 16)
    assert loaded.batch["pose_heatmap"].shape == (1, 25, 16, 16)
    assert loaded.batch["part_onehot"].shape == (1, 14, 16, 16)
    assert loaded.batch["appearance"] == [appearance]
    condition = loaded.hand_detail_condition.validate()
    references = loaded.hand_detail_references.validate()
    assert condition.region_valid.tolist() == [[[False, True, True], [False, False, False]]]
    assert torch.count_nonzero(condition.face_keypoints) == 0
    assert torch.count_nonzero(condition.smplx_detail[..., :13]) == 0
    assert torch.count_nonzero(condition.source_boxes[:, :, 0]) == 0
    assert torch.count_nonzero(condition.target_boxes[:, :, 0]) == 0
    assert not references.reference_valid[:, :, 0].any()
    assert torch.count_nonzero(references.images[:, :, 0]) == 0
    torch.testing.assert_close(condition.smplx_detail[0, 0, 13:103], torch.arange(90.0))


def test_frame_selector_supports_index_and_stem_and_cli_aliases() -> None:
    cache = _valid_cache()
    assert resolve_frame_selector(cache, 1) == 1
    assert resolve_frame_selector(cache, "target") == 1
    assert resolve_frame_selector(cache, "0") == 0
    with pytest.raises(KeyError, match="unknown frame stem"):
        resolve_frame_selector(cache, "missing")

    args = parse_train_args(
        [
            "--stage", "overfit", "--appearance", "024_6",
            "--source-index", "0", "--target-stem", "target", "--seed", "42",
        ]
    )
    assert args.stage == "overfit" and args.source_index == "0"
    assert args.target_stem == "target" and args.seed == 42

    pre_args = parse_precompute_args(
        ["--appearance", "024_6", "--dino-model", "/models/dino", "--insightface-root", "/models/arc"]
    )
    assert pre_args.appearance == "024_6"


class _TinyFaceTrainingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.face_preparer = nn.ModuleDict({
            "resampler": nn.Linear(2, 2),
            "geometry_tokenizer": nn.Linear(2, 2),
        })
        self.control_interface = nn.Module()
        self.control_interface.face_adapter = nn.Module()
        self.control_interface.face_adapter.zero_heads = nn.ModuleList([nn.Linear(2, 2)])
        self.control_interface.face_adapter.stage_adapters = nn.ModuleList([nn.Linear(2, 2)])
        self.control_interface.face_adapter.attention = nn.Linear(2, 2)
        self.control_interface.strength_controller = nn.Module()
        self.control_interface.strength_controller.face_log_group_scale = nn.Parameter(torch.zeros(6))


def test_phase0_parameter_selection_and_layered_optimizer_cover_every_face_parameter() -> None:
    model = _TinyFaceTrainingModel()
    selected = select_face_trainable_parameters(model, phase="warmup")
    selected_names = {name for name, _ in selected}
    assert selected_names == {
        "control_interface.face_adapter.zero_heads.0.weight",
        "control_interface.face_adapter.zero_heads.0.bias",
        "control_interface.strength_controller.face_log_group_scale",
    }
    assert not model.backbone.weight.requires_grad

    optimizer, grouped_names = build_face_optimizer(model, phase="full", weight_decay=0.0)
    all_face = {name for name, _ in select_face_trainable_parameters(model, phase="full")}
    assert set().union(*map(set, grouped_names.values())) == all_face
    assert len(optimizer.param_groups) == 3
    update_face_optimizer_lrs(optimizer, step=0, warmup_steps=1000)
    assert [group["lr"] for group in optimizer.param_groups] == [1e-4, 0.0, 0.0]
    update_face_optimizer_lrs(optimizer, step=1000, warmup_steps=1000)
    assert [group["lr"] for group in optimizer.param_groups] == [1e-4, 5e-5, 1e-5]


def test_loss_uses_outside_mask_and_exact_weights() -> None:
    prediction = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    target = torch.zeros_like(prediction)
    face_mask = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    result = compute_v65_face_losses(
        prediction,
        target,
        face_mask,
        face_id_loss=torch.tensor([2.0]),
        face_perceptual_loss=torch.tensor([4.0]),
        auxiliary_selector=torch.tensor([True]),
    )
    assert result["flow"].item() == pytest.approx(7.5)
    assert result["outside"].item() == pytest.approx((9.0 + 16.0) / 2.0)
    assert result["total"].item() == pytest.approx(7.5 + 0.1 * 2 + 0.05 * 4 + 0.02 * 12.5)

    skipped = compute_v65_face_losses(
        prediction,
        target,
        face_mask,
        face_id_loss=torch.tensor([99.0]),
        face_perceptual_loss=torch.tensor([99.0]),
        auxiliary_selector=torch.tensor([False]),
    )
    assert skipped["face_id"].item() == 0.0
    assert skipped["face_perceptual"].item() == 0.0

    # L_outside is explicitly relative to the frozen Face-OFF prediction when
    # provided, not the flow target.
    matched_off = prediction.clone()
    matched_off[:, :, 0] = 0.0
    controlled = compute_v65_face_losses(
        prediction,
        target,
        face_mask,
        outside_reference=matched_off,
    )
    assert controlled["outside"].item() == 0.0


def test_fixed_noise_and_timestep_are_seed_deterministic() -> None:
    first = make_fixed_training_state((1, 4, 8, 8), seed=42, max_timestep=650)
    second = make_fixed_training_state((1, 4, 8, 8), seed=42, max_timestep=650)
    third = make_fixed_training_state((1, 4, 8, 8), seed=43, max_timestep=650)
    torch.testing.assert_close(first["noise"], second["noise"])
    torch.testing.assert_close(first["timestep"], second["timestep"])
    assert not torch.equal(first["noise"], third["noise"])
