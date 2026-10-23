"""Real dataset boundaries for TikTok + the preserved full Qwen controller."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.tiktok_dataset import TikTokV3PoseDataset
from src.pose_control.qwen21_full.data_runtime import QwenFullDataset, pair_metadata


@pytest.fixture
def tiktok_pair(tmp_path):
    raw = tmp_path / "raw" / "00001"
    assets = tmp_path / "assets" / "00001"
    for folder in (raw / "images", raw / "masks", assets / "normal", assets / "part_512"):
        folder.mkdir(parents=True)
    for stem, color in (("0001", 32), ("0002", 224)):
        Image.new("RGB", (60, 100), (color, color, color)).save(raw / "images" / f"{stem}.png")
    Image.new("L", (60, 100), 255).save(raw / "masks/0002.png")
    Image.new("RGB", (60, 100), (128, 128, 255)).save(assets / "normal/0002.png")
    Image.fromarray(np.ones((32, 32), dtype=np.uint8)).save(assets / "part_512/0002.png")
    keypoints = torch.full((2, 25, 3), .5)
    keypoints[..., 2] = 1
    torch.save({"stem_to_idx": {"0001": 0, "0002": 1}, "smplx_global": torch.ones(2, 26),
                "kps_25_coords": keypoints}, assets / "v6_conditions.pt")
    row = dict(pair_id="train_001", split="train", sequence_id="00001", source_video_group="group_a",
               source_stem="0001", target_stem="0002", source_rgb=str(raw / "images/0001.png"),
               target_rgb=str(raw / "images/0002.png"), target_normal=str(assets / "normal/0002.png"),
               target_person_mask=str(raw / "masks/0002.png"))
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return manifest, row, raw.parent, assets.parent


def test_full_qwen_reads_native_tiktok_manifest_without_iper_schema(tiktok_pair):
    path, row, raw, assets = tiktok_pair
    dataset = QwenFullDataset(path, resolution=32, tiktok_raw_root=raw, tiktok_assets_root=assets,
                              align_mode="resize", legacy_policy_ok=True)
    sample = dataset[0]
    assert sample["src_image"].mean() < 0
    assert sample["tgt_image"].mean() > 0
    assert sample["part_onehot"].shape == (14, 32, 32)
    assert sample["pose_heatmap"].count_nonzero() > 0
    assert dataset.samples[0]["source"]["stem"] == "0001"
    assert sample["pair_metadata"]["source_rgb"] == row["source_rgb"]
    assert sample["pair_metadata"]["source_appearance"] == "00001"


def test_pair_metadata_preserves_tiktok_paths(tiktok_pair):
    _, row, _, _ = tiktok_pair
    meta = pair_metadata(row)
    assert meta["source_rgb"] == row["source_rgb"]
    assert meta["target_rgb"] == row["target_rgb"]
    assert meta["source_appearance"] == meta["target_appearance"] == "00001"
    assert meta["cross_person"] is False


@pytest.mark.parametrize("missing", ["v6_conditions.pt", "part_512/0002.png"])
def test_strict_tiktok_rejects_missing_geometry_instead_of_silent_fallback(tiktok_pair, missing):
    path, _, raw, assets = tiktok_pair
    (assets / "00001" / missing).unlink()
    with pytest.raises((ValueError, FileNotFoundError), match="condition|part"):
        TikTokV3PoseDataset(path, raw_root=raw, assets_root=assets, resolution=32,
                            align_mode="resize", strict_conditions=True)[0]


def test_align_mode_must_be_chosen_explicitly(tiktok_pair):
    """No default canvas: an omitted mode used to silently mean the square stretch."""
    path, _, raw, assets = tiktok_pair
    with pytest.raises(ValueError, match="explicitly"):
        QwenFullDataset(path, resolution=32, tiktok_raw_root=raw, tiktok_assets_root=assets)
    with pytest.raises(ValueError, match="explicitly"):
        TikTokV3PoseDataset(path, raw_root=raw, assets_root=assets, resolution=32)


def test_full_training_refuses_the_square_stretch_unless_marked_legacy(tiktok_pair):
    from src.pose_control.qwen21_full.data_runtime import CanvasPolicyError
    path, _, raw, assets = tiktok_pair
    with pytest.raises(CanvasPolicyError, match="squeezes every portrait"):
        QwenFullDataset(path, resolution=32, tiktok_raw_root=raw, tiktok_assets_root=assets,
                        align_mode="resize")
    # Reproducing a historical run stays possible, but only by name.
    legacy = QwenFullDataset(path, resolution=32, tiktok_raw_root=raw, tiktok_assets_root=assets,
                             align_mode="resize", legacy_policy_ok=True)
    assert legacy.canvas_policy == "resize_square"
    native = QwenFullDataset(path, resolution=32, tiktok_raw_root=raw, tiktok_assets_root=assets,
                             align_mode="native")
    assert native.canvas_policy == "native_pad"
    assert native.base.align_mode == "native"


def test_strict_tiktok_rejects_nonfinite_global_parameters(tiktok_pair):
    path, _, raw, assets = tiktok_pair
    cache_path = assets / "00001/v6_conditions.pt"
    cache = torch.load(cache_path, weights_only=False)
    cache["smplx_global"][1, 0] = float("nan")
    torch.save(cache, cache_path)
    with pytest.raises(ValueError, match="finite"):
        TikTokV3PoseDataset(path, raw_root=raw, assets_root=assets, resolution=32,
                            align_mode="resize", strict_conditions=True)[0]


def test_full_tiktok_rejects_crop_until_face_coordinates_share_it(tiktok_pair):
    path, _, raw, assets = tiktok_pair
    with pytest.raises(ValueError, match="resize"):
        QwenFullDataset(path, tiktok_raw_root=raw, tiktok_assets_root=assets, align_mode="square_crop")


def test_native_cache_precompute_reads_each_rgb_once_without_geometry(tiktok_pair):
    from src.pose_control.qwen21_full.execution import precompute
    from src.pose_control.qwen21_full.qwen_runtime import dataset_image_to_pil
    path, row, raw, assets = tiktok_pair
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    dataset = QwenFullDataset(path, resolution=32, tiktok_raw_root=raw, tiktok_assets_root=assets,
                              align_mode="resize", legacy_policy_ok=True)
    # Native-image preprocessing requires no body geometry; a missing part must
    # still block training when dataset[0] is subsequently accessed.
    (assets / "00001/part_512/0002.png").unlink()
    expected_source = dataset_image_to_pil(dataset.rgb_transform(dataset._load_image(row['source_rgb'])))
    expected_target = dataset_image_to_pil(dataset.rgb_transform(dataset._load_image(row['target_rgb'])))

    class RecordingCache:
        resolution = 32

        def __init__(self):
            self.sources, self.targets = [], []

        def source(self, pipe, image):
            self.sources.append(image.tobytes())

        def target(self, pipe, image):
            self.targets.append(image.tobytes())

    cache = RecordingCache()
    precompute(None, cache, (dataset,))
    assert cache.sources == [expected_source.tobytes()]
    assert cache.targets == [expected_target.tobytes()]
    with pytest.raises(FileNotFoundError, match='part'):
        dataset[0]
