"""Action text and source cache contract for Qwen21 dual-person editing."""

import hashlib
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.pose_control.qwen21.runtime import MODEL_REVISION, PROMPT, Qwen21IPERCache


MANIFEST_ACTIONS = {
    "train": {
        "hug01", "talk01", "fight01", "hug09", "talk09", "bend09", "dance10", "talk10",
        "argue10", "highfive12", "fight12", "talk12", "hug14", "talk14", "hug16",
        "jump16", "talk16",
    },
    "val": {"game22", "hug22", "kiss22", "basketball28", "dance28", "fight28"},
    "test": {"hug00", "taichi00", "yoga00", "dance02", "pose02", "backhug02"},
}


def build_dual_action_prompts(*args):
    from src.pose_control.qwen21.action_prompts import build_dual_action_prompts as build

    return build(*args)


def _masks(*, a_on_left: bool = True):
    left = torch.zeros(1, 6, 10)
    right = torch.zeros_like(left)
    left[:, 1:5, 1:3] = 1
    right[:, 1:5, 7:9] = 1
    return (left, right) if a_on_left else (right, left)


def test_hug_prompt_matches_approved_text():
    native, adapter = build_dual_action_prompts("hug16", _masks())
    assert native == (
        "Edit the supplied photo of exactly two existing people. The person on the left and the person "
        "on the right move toward each other and hug. Their torsos touch and their arms wrap around "
        "each other's shoulders and backs. Preserve each person's face, hair, clothing and proportions, "
        "as well as the camera view, lighting and background. Do not add another person or change the setting."
    )
    assert adapter == "Make the two existing people hug. Preserve their appearance and background."
    assert build_dual_action_prompts("HUG01", _masks(a_on_left=False)) == (native, adapter)


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ("highfive12", "high-five"),
        ("fight01", "fight"),
        ("talk09", "talk"),
        ("argue10", "argue"),
        ("dance10", "dance"),
        ("jump16", "jump"),
        ("bend09", "bend"),
    ],
)
def test_action_mapping_is_specific_and_source_order_aware(action, expected):
    native_a, adapter_a = build_dual_action_prompts(action, _masks())
    native_b, adapter_b = build_dual_action_prompts(action, _masks(a_on_left=False))
    assert expected in native_a.lower()
    assert expected in adapter_a.lower()
    assert "person on the left" in native_a.lower()
    assert "person on the right" in native_a.lower()
    assert "source person A" in native_a
    assert "source person B" in native_a
    assert native_a != native_b
    assert adapter_a == adapter_b
    assert len(adapter_a) < len(native_a)
    assert "background" in native_a.lower() and "background" in adapter_a.lower()


def test_invalid_action_or_missing_source_person_is_rejected():
    with pytest.raises(ValueError, match="unsupported action"):
        build_dual_action_prompts("handshake01", _masks())
    empty, right = _masks()
    with pytest.raises(ValueError, match="empty"):
        build_dual_action_prompts("hug01", (empty * 0, right))


@pytest.mark.parametrize("label", sorted(set().union(*MANIFEST_ACTIONS.values())))
def test_all_train_val_and_test_action_labels_are_supported(label):
    native, adapter = build_dual_action_prompts(label, _masks())
    assert native.startswith("Edit the supplied photo of exactly two existing people.")
    assert adapter.startswith("Make the two existing people ")
    assert "Preserve their appearance and background." in adapter


def test_manifest_action_sets_match_test_coverage_when_manifests_are_available():
    root = Path(__file__).resolve().parents[1] / "datasets/Hi4D_pilot_v1/native_interaction_v4_2"
    for split, expected in MANIFEST_ACTIONS.items():
        manifest = root / f"main_pairs_{split}.jsonl"
        if manifest.exists():
            observed = {json.loads(line)["target_action"] for line in manifest.read_text().splitlines()}
            assert observed == expected


@pytest.mark.parametrize(
    ("label", "phrase"),
    [
        ("basketball28", "basketball"),
        ("game22", "play a game together"),
        ("kiss22", "kiss"),
        ("backhug02", "back hug"),
        ("pose02", "pose together"),
        ("taichi00", "tai chi"),
        ("yoga00", "yoga"),
    ],
)
def test_holdout_prompts_name_only_the_action_without_inventing_props(label, phrase):
    native, adapter = build_dual_action_prompts(label, _masks())
    assert phrase in native.lower()
    assert all(prop not in native.lower() for prop in ("hoop", "court", "racket"))
    assert "background" in native.lower() and "background" in adapter.lower()


def test_source_cache_key_uses_resolved_prompt_without_changing_legacy_default(tmp_path):
    image = Image.new("RGB", (4, 4), "white")
    cache = Qwen21IPERCache(tmp_path)
    legacy_digest = hashlib.sha256()
    legacy_digest.update((MODEL_REVISION + PROMPT + "source" + "512").encode())
    legacy_digest.update(image.tobytes())
    assert cache._path(image, "source") == tmp_path / f"source_{legacy_digest.hexdigest()[:24]}.pt"
    assert cache._path(image, "source", prompt=PROMPT) == cache._path(image, "source")
    assert cache._path(image, "source", prompt="Make them hug.") != cache._path(image, "source")
    assert cache._path(image, "source", prompt="Make them talk.") != cache._path(image, "source", prompt="Make them hug.")
    assert cache._path(image, "target", prompt="Make them hug.") == cache._path(image, "target")


def test_source_reads_only_the_matching_prompt_cache(tmp_path):
    image = Image.new("RGB", (4, 4), "white")
    cache = Qwen21IPERCache(tmp_path)
    for prompt, value in ((PROMPT, 1), ("Make them hug.", 2)):
        torch.save({"prompt_embeds": torch.tensor([value])}, cache._path(image, "source", prompt=prompt))
    assert cache.source(None, image)["prompt_embeds"].item() == 1
    assert cache.source(None, image, prompt="Make them hug.")["prompt_embeds"].item() == 2


def test_pair_selects_each_sample_prompt_and_preserves_default(tmp_path):
    cache = Qwen21IPERCache(tmp_path)
    black = Image.new("RGB", (2, 2), "black")
    white = Image.new("RGB", (2, 2), "white")
    for image, prompt, value in ((black, PROMPT, 1), (black, "hug", 2), (white, "talk", 3)):
        entry = {
            "prompt_embeds": torch.full((1, 2, 4), value, dtype=torch.float32),
            "prompt_mask": torch.ones(1, 2, dtype=torch.bool),
            "image_mask": torch.ones(1, 2, dtype=torch.bool),
            "reference_latent": torch.zeros(1, 1024, 4),
        }
        torch.save(entry, cache._path(image, "source", prompt=prompt))
        torch.save(torch.zeros(1, 1024, 4), cache._path(image, "target"))
    batch = {
        "src_image": torch.stack((torch.full((3, 2, 2), -1.0), torch.ones(3, 2, 2))),
        "tgt_image": torch.stack((torch.full((3, 2, 2), -1.0), torch.ones(3, 2, 2))),
    }
    pair = cache.pair(None, batch, torch.device("cpu"), prompts=["hug", "talk"])
    assert pair["prompt_embeds"][:, 0, 0].tolist() == [2.0, 3.0]
    default = cache.pair(None, {key: value[:1] for key, value in batch.items()}, torch.device("cpu"))
    assert default["prompt_embeds"][0, 0, 0].item() == 1
    with pytest.raises(ValueError, match="prompts length"):
        cache.pair(None, batch, torch.device("cpu"), prompts=["hug"])
