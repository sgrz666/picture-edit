from __future__ import annotations

import pytest
import torch

from scripts.prepare_tiktok_qwen21_face import assemble_cache
from scripts.data.prepare_qwen21_prompt_study_faces import requested_frames


def test_target_only_frame_has_geometry_but_no_identity():
    geometry = [
        dict(
            face_landmarks=torch.zeros(72, 3),
            face_boxes=torch.tensor([0.2, 0.1, 0.6, 0.4]),
            jaw_pose=torch.ones(3),
            expression=torch.ones(10),
            geometry_valid=True,
        )
        for _ in range(2)
    ]
    source_features = {"0001": (torch.ones(512), torch.ones(256, 1536))}
    cache = assemble_cache(
        ["0001", "0002"],
        geometry,
        source_features,
        {"schema_version": 1, "appearance": "00001"},
    )
    assert cache["valid"].tolist() == [True, False]
    assert cache["geometry_valid"].tolist() == [True, True]
    assert cache["metadata"]["geometry_valid"] == [True, True]
    assert not cache["arcface"][1].any()
    assert not cache["dino_patches"][1].any()
    assert cache["jaw_pose"][1].any()


def test_requested_frames_aggregates_sources_and_targets():
    rows = [
        dict(
            pair_id="tiktok_00",
            sequence_id="00232",
            source_appearance="00130",
            target_appearance="00232",
            source_stem="0032",
            target_stem="0223",
            source_rgb="/a.png",
            target_rgb="/b.png",
        ),
        dict(
            pair_id="tiktok_01",
            sequence_id="00232",
            source_appearance="00130",
            target_appearance="00232",
            source_stem="0032",
            target_stem="0250",
            source_rgb="/a.png",
            target_rgb="/c.png",
        ),
    ]
    groups = requested_frames(rows)
    assert "00130" in groups
    assert "00232" in groups
    assert groups["00130"]["stems"] == {"0032"}
    assert groups["00130"]["sources"]["0032"] == "/a.png"
    assert groups["00232"]["stems"] == {"0223", "0250"}
