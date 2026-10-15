from __future__ import annotations

import pytest
from src.pose_control.qwen21_full.prompt_study.pairs import (
    select_test24,
    audit_video_groups,
    normalize_diagnostic_row,
)
from src.pose_control.qwen21_full.data_runtime import pair_metadata


def test_test24_is_order_independent_and_spaced():
    groups = ("seq_00010_00028", "seq_00241_00244", "seq_00289_00307")
    rows = [
        dict(
            pair_id=f"g{g}_{i}",
            source_video_group=group,
            sequence_id=f"{g:05d}",
            target_stem=f"{i * 16:04d}",
        )
        for g, group in enumerate(groups)
        for i in range(12)
    ]
    picked = select_test24(rows)
    assert picked == select_test24(list(reversed(rows)))
    assert len(picked) == 24
    for group in groups:
        frames = sorted(
            int(r["target_stem"])
            for r in picked
            if r["source_video_group"] == group
        )
        assert len(frames) == 8
        assert all(b - a >= 8 for a, b in zip(frames, frames[1:]))


def test_test24_fails_when_insufficient_samples():
    groups = ("seq_00010_00028", "seq_00241_00244", "seq_00289_00307")
    rows = [
        dict(
            pair_id=f"g0_{i}",
            source_video_group=groups[0],
            sequence_id="00000",
            target_stem=f"{i * 16:04d}",
        )
        for i in range(5)  # only 5 rows, needs 8
    ]
    with pytest.raises(ValueError, match="Cannot lock eight spaced targets"):
        select_test24(rows)


def test_cross_tiktok_metadata_preserves_two_appearances():
    row = dict(
        pair_id="tiktok_00",
        sequence_id="00232",
        source_appearance="00130",
        target_appearance="00232",
        source_stem="0032",
        target_stem="0223",
        source_rgb="/a.png",
        target_rgb="/b.png",
    )
    meta = pair_metadata(row)
    assert (meta["source_appearance"], meta["target_appearance"]) == ("00130", "00232")
    assert meta["cross_person"] is True
    assert meta["sequence_id"] if "sequence_id" in meta else meta["target_appearance"] == "00232"


def test_audit_video_groups_disjoint_and_overlap():
    train = [{"source_video_group": "train_g1"}, {"source_video_group": "train_g2"}]
    val = [{"source_video_group": "val_g1"}]
    test = [{"source_video_group": "seq_00010_00028"}]

    result = audit_video_groups(train, val, test)
    assert result["disjoint"] is True

    bad_test = [{"source_video_group": "train_g1"}]
    with pytest.raises(ValueError, match="overlap"):
        audit_video_groups(train, val, bad_test)


def test_normalize_diagnostic_row():
    raw = {
        "pair_id": "tiktok_00",
        "source_appearance": "00130",
        "target_appearance": "00232",
        "source_stem": "0032",
        "target_stem": "0223",
        "source_rgb": "/data/a.png",
        "target_rgb": "/data/b.png",
        "action_description": "FORBIDDEN MANUAL TEXT",
        "nested_garbage": {"unwanted": True},
    }
    normalized = normalize_diagnostic_row(raw)
    assert normalized["pair_id"] == "tiktok_00"
    assert normalized["sequence_id"] == "00232"
    assert normalized["cross_person"] is True
    assert "action_description" not in normalized
    assert "nested_garbage" not in normalized


def test_diagnostic_preserves_guides_and_uses_real_tiktok_asset_layout():
    row = dict(pair_id='tiktok_00', source_appearance='00130', target_appearance='00232',
               source_stem='0032', target_stem='0223', source_rgb='/raw/a.png', target_rgb='/raw/b.png',
               source_person_mask='/raw/am.png', target_person_mask='/raw/bm.png',
               target_dwpose='/datasets/TikTok_v3/dwpose_v1/00232/dwpose/0223.png',
               target_normal='/datasets/TikTok_3d_assets_native/00232/normal/0223.png')
    actual = normalize_diagnostic_row(row)
    assert actual['target_dwpose'] == row['target_dwpose']
    assert actual['target_dwpose_json'].replace('\\', '/') == '/datasets/TikTok_v3/dwpose_v1/00232/dwpose_json/0223.json'
    assert actual['part_512'].replace('\\', '/') == '/datasets/TikTok_3d_assets_native/00232/part_512/0223.png'
    assert actual['v6_conditions'].replace('\\', '/') == '/datasets/TikTok_3d_assets_native/00232/v6_conditions.pt'
    assert actual['source_person_mask'] == '/raw/am.png'
