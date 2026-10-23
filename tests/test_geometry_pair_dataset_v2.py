"""Pair-contract tests: what a geometry pair must carry, and every way it can be refused.

Deliberately includes both directions for each gate — a validator that only ever passes proves
nothing, and one that only rejects would break real training.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.data.canvas_contract import native_pad
from src.data.geometry_pair_dataset_v2 import (
    PairContractError,
    assert_pair_contract,
    bucket_by_canvas,
    canvas_fingerprint,
    collate_geometry_pairs,
)

TARGET_HW = (1080, 604)
SOURCE_HW = (1333, 750)


def make_pair(target_hw=TARGET_HW, source_hw=SOURCE_HW):
    target = native_pad(target_hw)
    source = native_pad(source_hw)
    th, tw = target.canvas_hw
    sh, sw = source.canvas_hw
    return {
        "src_image": torch.zeros(3, sh, sw),
        "tgt_image": torch.zeros(3, th, tw),
        "pose_heatmap": torch.zeros(25, th, tw),
        "normal": torch.zeros(3, th, tw),
        "part_onehot": torch.zeros(14, th, tw),
        "human_mask": torch.zeros(1, th, tw),
        "valid_image_mask": target.valid_mask(),
        "smplx_global": torch.zeros(26),
        "depth": torch.zeros(1, th, tw),
        "depth_metadata": {"metric": False, "source": "depth_vis"},
        "source_canvas": source.metadata(),
        "target_canvas": target.metadata(),
        "target_token_hw": target.token_hw,
    }


def test_a_well_formed_portrait_pair_passes_and_reports_its_facts():
    pair = make_pair()
    facts = assert_pair_contract(pair, where="test")
    assert facts["target_token_hw"] == (68, 38)
    assert facts["canvas_hw"] == (1088, 608)
    assert facts["policy"] == "native_pad"
    assert facts["source_canvas"].token_hw == (84, 48), "A rides its own 1333x750 canvas"
    assert facts["target_canvas"] != facts["source_canvas"]


def test_a_missing_canvas_is_a_refusal_not_a_default():
    pair = make_pair()
    del pair["target_canvas"]
    with pytest.raises(PairContractError, match="target_canvas"):
        assert_pair_contract(pair, where="test")
    pair = make_pair()
    pair["target_canvas"] = None
    with pytest.raises(Exception, match="missing CanvasSpec"):
        assert_pair_contract(pair, where="test")


def test_a_legacy_canvas_record_cannot_satisfy_the_contract():
    pair = make_pair()
    pair["target_canvas"] = {"version": "legacy_stretch_v1", "policy": "resize_square",
                             "original_hw": list(TARGET_HW), "canvas_hw": [512, 512]}
    with pytest.raises(Exception, match="legacy canvas"):
        assert_pair_contract(pair, where="test")


def test_the_token_grid_must_travel_with_the_sample():
    pair = make_pair()
    del pair["target_token_hw"]
    with pytest.raises(PairContractError, match="carries no target_token_hw"):
        assert_pair_contract(pair, where="test")
    pair = make_pair()
    pair["target_token_hw"] = (64, 64)  # the square guess this contract exists to reject
    with pytest.raises(PairContractError, match="disagrees with the target canvas"):
        assert_pair_contract(pair, where="test")


def test_a_condition_off_the_target_canvas_is_refused_not_resized():
    pair = make_pair()
    pair["pose_heatmap"] = torch.zeros(25, 512, 512)
    with pytest.raises(PairContractError, match="rasterised on the target canvas"):
        assert_pair_contract(pair, where="test")
    pair = make_pair()
    pair["src_image"] = torch.zeros(3, *native_pad(TARGET_HW).canvas_hw)  # A forced onto B's canvas
    with pytest.raises(PairContractError, match="source canvas"):
        assert_pair_contract(pair, where="test")


def test_a_missing_condition_is_not_zero_filled():
    pair = make_pair()
    del pair["part_onehot"]
    with pytest.raises(PairContractError, match="part_onehot"):
        assert_pair_contract(pair, where="test")


def test_masks_must_agree_with_the_declared_padding():
    pair = make_pair()
    pair["valid_image_mask"] = torch.ones(1, 1088, 608)  # claims no padding at all
    with pytest.raises(PairContractError, match="leaves .* of padding"):
        assert_pair_contract(pair, where="test")
    pair = make_pair()
    pair["human_mask"][0, 1085, 605] = 1.0  # marks a pixel in the padding margin
    with pytest.raises(PairContractError, match="outside the valid canvas"):
        assert_pair_contract(pair, where="test")


def test_depth_needs_its_provenance_because_visualisation_is_not_meters():
    pair = make_pair()
    del pair["depth_metadata"]
    with pytest.raises(PairContractError, match="depth_metadata"):
        assert_pair_contract(pair, where="test")


def test_a_batch_may_not_stack_two_canvases_together():
    batch = collate_geometry_pairs([make_pair(), make_pair()])
    assert batch["tgt_image"].shape == (2, 3, 1088, 608)
    with pytest.raises(PairContractError, match="mixes canvases"):
        collate_geometry_pairs([make_pair(), make_pair(target_hw=(900, 500))])


def test_bucketing_groups_by_both_canvases():
    rows = [make_pair(), make_pair(), make_pair(target_hw=(900, 500))]
    for row in rows:
        row["target_canvas"] = native_pad(row["tgt_image"].shape[-2:]).metadata()
    buckets = bucket_by_canvas(rows)
    assert sorted(len(v) for v in buckets.values()) == [1, 2]


def test_fingerprint_binds_both_canvases_and_changes_with_either():
    first = canvas_fingerprint(make_pair())
    assert first == canvas_fingerprint(make_pair())
    assert first != canvas_fingerprint(make_pair(source_hw=(1024, 1024)))
    assert first != canvas_fingerprint(make_pair(target_hw=(1024, 1024)))


@pytest.mark.skipif(not Path("/home/shangguanrz/project/pic-edit/datasets/TikTokDataset/"
                             "TikTok_v3/manifests/pairs/val_pairs.jsonl").is_file(),
                    reason="real TikTok v3 assets only exist on the sg working machine")
def test_the_budget_policy_keeps_aspect_and_satisfies_the_contract():
    from src.data.tiktok_dataset import TikTokV3PoseDataset

    root = Path("/home/shangguanrz/project/pic-edit/datasets/TikTokDataset")
    common = dict(raw_root=root / "TikTok_dataset/TikTok_dataset",
                  assets_root=root / "TikTok_3d_assets_native", strict_conditions=True)
    manifest = root / "TikTok_v3/manifests/pairs/val_pairs.jsonl"
    budget = 256 * 256
    dataset = TikTokV3PoseDataset(manifest, align_mode="aspect_preserving",
                                  max_pixels=budget, **common)
    facts = assert_pair_contract(dataset[0], where="budget pair")
    spec = facts["target_canvas"]
    before = spec.original_hw[1] / spec.original_hw[0]
    after = spec.resized_hw[1] / spec.resized_hw[0]
    assert abs(before - after) < 2e-3, (before, after)  # integer rounding only, never a stretch
    assert spec.canvas_hw[0] * spec.canvas_hw[1] <= budget + 32 * sum(spec.canvas_hw)
    assert facts["target_token_hw"] == (spec.canvas_hw[0] // 16, spec.canvas_hw[1] // 16)
    assert dataset[0]["tgt_image"].shape[-2:] == spec.canvas_hw
    assert spec.resized_hw != spec.original_hw, "a 256^2 budget must actually shrink 604x1080"


@pytest.mark.skipif(not Path("/home/shangguanrz/project/pic-edit/datasets/TikTokDataset/"
                             "TikTok_v3/manifests/pairs/val_pairs.jsonl").is_file(),
                    reason="real TikTok v3 assets only exist on the sg working machine")
def test_the_native_path_still_produces_the_same_rectangular_canvas():
    from src.data.tiktok_dataset import TikTokV3PoseDataset

    root = Path("/home/shangguanrz/project/pic-edit/datasets/TikTokDataset")
    dataset = TikTokV3PoseDataset(root / "TikTok_v3/manifests/pairs/val_pairs.jsonl",
                                  raw_root=root / "TikTok_dataset/TikTok_dataset",
                                  assets_root=root / "TikTok_3d_assets_native",
                                  align_mode="native", strict_conditions=True)
    pair = dataset[0]
    facts = assert_pair_contract(pair, where="native pair")
    assert facts["canvas_hw"] == (1088, 608)
    assert facts["target_token_hw"] == (68, 38)
    assert facts["target_canvas"].resized_hw == (1080, 604), "native must not rescale"
    assert pair["tgt_image"].shape == (3, 1088, 608)


def test_the_budget_policy_demands_its_budget():
    from src.data.tiktok_dataset import TikTokV3PoseDataset

    with pytest.raises(ValueError, match="max_pixels"):
        TikTokV3PoseDataset("/nonexistent.jsonl", align_mode="aspect_preserving",
                            raw_root="/nonexistent", assets_root="/nonexistent")


@pytest.mark.skipif(not Path("/home/shangguanrz/project/pic-edit/datasets/TikTokDataset/"
                             "TikTok_v3/manifests/pairs/val_pairs.jsonl").is_file(),
                    reason="real TikTok v3 assets only exist on the sg working machine")
def test_the_item_and_the_cache_precompute_share_one_rgb_transform():
    """precompute() keys the latent cache on rgb_transform output.

    If the item path pads differently, training feeds pixels the cache was never built from: every
    lookup silently misses and the run quietly re-encodes on GPU instead of failing.
    """
    from src.data.tiktok_dataset import TikTokV3PoseDataset

    root = Path("/home/shangguanrz/project/pic-edit/datasets/TikTokDataset")
    for mode, kwargs in (("aspect_preserving", {"max_pixels": 65536}), ("native", {})):
        dataset = TikTokV3PoseDataset(root / "TikTok_v3/manifests/pairs/val_pairs.jsonl",
                                      raw_root=root / "TikTok_dataset/TikTok_dataset",
                                      assets_root=root / "TikTok_3d_assets_native",
                                      align_mode=mode, resolution="native",
                                      strict_conditions=True, **kwargs)
        path = dataset.samples[0]["target_rgb"]
        precomputed = dataset.rgb_transform(dataset._load_image(path))
        assert torch.equal(precomputed, dataset[0]["tgt_image"]), mode


def test_the_two_margin_conventions_are_pinned_until_an_expensive_decision_is_made():
    """RGB pads with grey and conditions pad with black; that is known, not hidden.

    probe_padding_attention.py measured the margin value moving valid-token DiT outputs by ~4x
    their signal, so unifying these constants changes what existing checkpoints were trained
    against. This test exists so the day someone changes one of them, it is a decision on record.
    """
    from src.data.tiktok_dataset import CONDITION_PAD_VALUE, RGB_PAD_VALUE

    assert RGB_PAD_VALUE == (128, 128, 128)
    assert CONDITION_PAD_VALUE == 0
    assert RGB_PAD_VALUE[0] != CONDITION_PAD_VALUE, "intentionally divergent; see the docstring"


def test_an_iper_pair_is_refused_naming_the_stretch_not_a_missing_field():
    from src.data.canvas_contract import CanvasPolicyError, LegacyStretch
    from src.data.iper_dataset import CANVAS_POLICY

    assert CANVAS_POLICY == "resize_square"
    pair = make_pair()
    pair["target_canvas"] = LegacyStretch(CANVAS_POLICY, (1024, 1024), (512, 512)).metadata()
    with pytest.raises(CanvasPolicyError, match="legacy canvas"):
        assert_pair_contract(pair, where="iPER trainer")


def test_the_factory_gate_refuses_a_stretching_canvas_for_tiktok():
    """`load_pose_dataset` is how the legacy iPER trainers reach TikTok; it must not be a side door."""
    from src.data.canvas_contract import CanvasPolicyError
    from src.data.tiktok_dataset import load_pose_dataset

    root = Path("/home/shangguanrz/project/pic-edit/datasets/TikTokDataset")
    manifest = root / "TikTok_v3/manifests/pairs/val_pairs.jsonl"
    common = dict(dataset_type="tiktok_v3",
                  tiktok_raw_root=root / "TikTok_dataset/TikTok_dataset",
                  tiktok_assets_root=root / "TikTok_3d_assets_native", resolution=512)
    with pytest.raises(ValueError, match="explicitly"):
        load_pose_dataset(manifest, **common)
    with pytest.raises(CanvasPolicyError, match="stretches the person to a square"):
        load_pose_dataset(manifest, align_mode="resize", **common)
    with pytest.raises(CanvasPolicyError, match="crops the field of view"):
        load_pose_dataset(manifest, align_mode="square_crop", **common)
    # Reproduction stays possible, but only by naming it.
    assert load_pose_dataset(manifest, align_mode="resize", legacy_policy_ok=True, **common) is not None
    assert load_pose_dataset(manifest, align_mode="native", **common) is not None


def test_unsupported_mode_names_are_rejected_by_the_dataset():
    from src.data.tiktok_dataset import TikTokV3PoseDataset

    for mode in ("stretch", "native_pad", ""):
        with pytest.raises(ValueError, match="Unsupported align_mode"):
            TikTokV3PoseDataset("/nonexistent.jsonl", align_mode=mode,
                                raw_root="/nonexistent", assets_root="/nonexistent")


@pytest.mark.skipif(not Path("/home/shangguanrz/project/pic-edit/datasets/TikTokDataset/"
                             "TikTok_v3/manifests/pairs/val_pairs.jsonl").is_file(),
                    reason="real TikTok v3 assets only exist on the sg working machine")
def test_the_real_native_loader_output_satisfies_the_contract():
    from src.data.tiktok_dataset import TikTokV3PoseDataset

    root = Path("/home/shangguanrz/project/pic-edit/datasets/TikTokDataset")
    manifest = root / "TikTok_v3/manifests/pairs/val_pairs.jsonl"
    dataset = TikTokV3PoseDataset(manifest, raw_root=root / "TikTok_dataset/TikTok_dataset",
                                  assets_root=root / "TikTok_3d_assets_native",
                                  align_mode="native", strict_conditions=True)
    checked = 0
    for index in range(min(len(dataset), 5)):
        pair = dataset[index]
        facts = assert_pair_contract(pair, where="real TikTok native pair")
        height, width = facts["canvas_hw"]
        assert height != width or pair["tgt_image"].shape[-1] == pair["tgt_image"].shape[-2]
        assert facts["target_token_hw"] == (height // 16, width // 16)
        assert json.loads(pair["alignment_meta"])["target"]["version"] == "native_pad_v1"
        assert pair["depth_metadata"]["metric"] is False
        checked += 1
    assert checked > 0
