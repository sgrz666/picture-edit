import json

import pytest
import torch

from scripts.prepare_tiktok_qwen21_training import check_split_disjoint, inspect_body_pair


def test_preflight_rejects_source_video_leakage():
    train = [{"sequence_id": "00001", "source_video_group": "same"}]
    val = [{"sequence_id": "00002", "source_video_group": "same"}]
    with pytest.raises(ValueError, match="overlap"):
        check_split_disjoint({"train": train, "val": val})


def test_preflight_rejects_missing_group_provenance():
    with pytest.raises(ValueError, match="source_video_group"):
        check_split_disjoint({"train": [{"sequence_id": "00001"}], "val": []})


def test_preflight_does_not_accept_zero_keypoints(tmp_path):
    assets = tmp_path / "assets/00001"
    assets.mkdir(parents=True)
    torch.save({"stem_to_idx": {"0002": 0}, "smplx_global": torch.ones(1, 26),
                "kps_25_coords": torch.zeros(1, 25, 3)}, assets / "v6_conditions.pt")
    row = {"sequence_id": "00001", "source_stem": "0001", "target_stem": "0002"}
    _, reasons = inspect_body_pair(row, assets.parent, {})
    assert "fewer_than_8_body_keypoints" in reasons


def test_preflight_rejects_source_equals_target(tmp_path):
    row = {"sequence_id": "00001", "source_stem": "0002", "target_stem": "0002"}
    _, reasons = inspect_body_pair(row, tmp_path, {})
    assert "source_equals_target" in reasons
