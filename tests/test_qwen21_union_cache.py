"""Cache provenance, no supervision leakage, and deterministic VAE mode."""
import json
import sys
import types

import pytest
import torch
from PIL import Image


class FakeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = types.SimpleNamespace(z_dim=64, latents_mean=[1.0] * 64, latents_std=[2.0] * 64)
        self.calls = 0

    def encode(self, image):
        self.calls += 1
        assert image.shape == (1, 4, 1, 512, 512)
        # The mean of the RGB distinguishes ordered reference contents.
        value = image[:, :3].mean()
        posterior = types.SimpleNamespace(mode=lambda: torch.full((1, 64, 1, 32, 32), value))
        return types.SimpleNamespace(latent_dist=posterior)


class FakeProcessor:
    def preprocess(self, image, height, width):
        assert image.size == (512, 512) and (height, width) == (512, 512)
        color = image.getpixel((256, 256))
        return torch.tensor(color, dtype=torch.float32).view(1, 4, 1, 1).expand(1, 4, height, width) / 127.5 - 1


class FakePipe:
    def __init__(self):
        self.vae = FakeVAE()
        self.text_encoder = torch.nn.Linear(1, 1)
        self.image_processor = FakeProcessor()
        self.vae_scale_factor = 16
        self._execution_device = torch.device("cpu")
        self.prompt_calls = 0

    def encode_prompt(self, image, prompt, device, num_images_per_prompt):
        self.prompt_calls += 1
        assert len(image) == 3 and all(i.size == (512, 512) for i in image)
        slots = torch.cat([torch.ones(1, 256, dtype=torch.bool), torch.zeros(1, 2, dtype=torch.bool)] * 3, dim=1)
        return torch.zeros(1, slots.shape[1], 4096), torch.ones_like(slots), slots


@pytest.fixture
def record(tmp_path):
    for index, name in enumerate(("source", "pose_a", "pose_b", "control", "target")):
        Image.new("RGB", (512, 512), (index * 35,) * 3).save(tmp_path / f"{name}.png")
    return {"sample_id": "sample/001", "split": "train", "source_rgb": str(tmp_path / "source.png"),
            "reference_pose_images": [str(tmp_path / "pose_a.png"), str(tmp_path / "pose_b.png")],
            "control_image": str(tmp_path / "control.png"), "target_rgb": str(tmp_path / "target.png"),
            "target_masks": ["DO-NOT-READ"], "prompt": "Picture 1 follows Picture 2 and Picture 3.",
            "transforms": {"target": {"resized_wh": [376, 512], "offset_xy": [68, 0]}}}


def test_condition_cache_never_reads_target_and_reuses_content(tmp_path, record):
    from src.pose_control.qwen21_union.cache import precompute_conditions, load_condition

    record["target_rgb"] = "NO-SUCH-TARGET"
    pipe = FakePipe()
    rng = torch.get_rng_state().clone()
    first = precompute_conditions(pipe, [record], tmp_path / "cache", "model-v1")
    assert torch.equal(torch.get_rng_state(), rng)
    payload = load_condition(tmp_path / "cache", record["sample_id"])
    assert payload["reference_latents"].shape == (1, 3072, 64)
    assert payload["control_latents"].shape == (1, 1024, 64)
    assert payload["prompt_embeds"].shape[-1] == 4096
    assert payload["img_mask"].shape[1] == payload["prompt_embeds"].shape[1] + 256
    assert payload["img_mask"].sum() * 4 == 4096
    assert payload["img_shapes"] == [[(1, 32, 32)] * 4]
    assert payload["target_hw"] == (32, 32)
    assert payload["canvas_hw"] == (512, 512)
    assert torch.allclose(payload["reference_latents"][:, :1024], torch.full((1, 1024, 64), -1.0))
    assert not any(key.startswith("target_") and key != "target_hw" for key in payload)
    assert not any(t.is_inference() for t in payload.values() if torch.is_tensor(t))
    assert "NO-SUCH-TARGET" not in json.dumps(first)
    assert pipe.vae.calls == 4 and pipe.prompt_calls == 1
    again = precompute_conditions(pipe, [record], tmp_path / "cache", "model-v1")
    assert again == first and pipe.vae.calls == 4 and pipe.prompt_calls == 1
    record["reference_pose_images"].reverse()
    changed = precompute_conditions(pipe, [record], tmp_path / "cache", "model-v1")
    assert changed[record["sample_id"]]["hash"] != first[record["sample_id"]]["hash"]
    assert pipe.vae.calls == 8
    with pytest.raises(ValueError, match="fingerprint"):
        load_condition(tmp_path / "cache", record["sample_id"], expected_hash=first[record["sample_id"]]["hash"])


def test_cache_fingerprint_tracks_bytes_prompt_and_model(tmp_path, record):
    from src.pose_control.qwen21_union.cache import condition_fingerprint

    original = condition_fingerprint(record, "model-v1")
    record["target_rgb"] = "UNREADABLE"
    record["target_masks"] = ["UNREADABLE"]
    assert condition_fingerprint(record, "model-v1") == original
    assert condition_fingerprint(record, "model-v2") != original
    record["prompt"] += " changed"
    assert condition_fingerprint(record, "model-v1") != original
    record["prompt"] = record["prompt"].removesuffix(" changed")
    Image.new("RGB", (512, 512), (100, 0, 0)).save(record["control_image"])
    assert condition_fingerprint(record, "model-v1") != original


def test_supervision_cache_is_train_only_and_fractional(tmp_path, record):
    from src.pose_control.qwen21_union.cache import precompute_supervision, load_supervision

    pipe = FakePipe()
    precompute_supervision(pipe, [record], tmp_path / "cache", "model-v1")
    payload = load_supervision(tmp_path / "cache", record["sample_id"])
    assert payload["target_latent"].shape == (1, 1024, 64)
    assert payload["valid_target_tokens"].sum() == 752
    assert set(payload) == {"target_latent", "valid_target_tokens"}
    record["split"] = "test"
    with pytest.raises(ValueError, match="train"):
        precompute_supervision(pipe, [record], tmp_path / "cache", "model-v1")


def test_prepared_images_are_not_silently_resized(tmp_path, record):
    from src.pose_control.qwen21_union.cache import precompute_conditions

    Image.new("RGB", (128, 256)).save(record["source_rgb"])
    with pytest.raises(ValueError, match="512"):
        precompute_conditions(FakePipe(), [record], tmp_path / "cache", "model-v1")


def test_single_sample_prompt_without_attention_mask_is_fully_valid(tmp_path, record):
    from src.pose_control.qwen21_union.cache import precompute_conditions, load_condition

    class SingleSamplePipe(FakePipe):
        def encode_prompt(self, **kwargs):
            prompt, _, slots = super().encode_prompt(**kwargs)
            return prompt, None, slots

    precompute_conditions(SingleSamplePipe(), [record], tmp_path / "cache", "model-v1")
    payload = load_condition(tmp_path / "cache", record["sample_id"])
    assert payload["prompt_mask"].dtype == torch.bool
    assert payload["prompt_mask"].shape == payload["prompt_embeds"].shape[:2]
    assert payload["prompt_mask"].all()


def test_image_slot_mask_is_required_even_without_prompt_attention_mask(tmp_path, record):
    from src.pose_control.qwen21_union.cache import precompute_conditions

    class MissingSlotsPipe(FakePipe):
        def encode_prompt(self, **kwargs):
            prompt, _, _ = super().encode_prompt(**kwargs)
            return prompt, None, None

    with pytest.raises(ValueError, match="slot"):
        precompute_conditions(MissingSlotsPipe(), [record], tmp_path / "cache", "model-v1")


def test_rectangular_target_uses_exact_recorded_transform(tmp_path, record):
    from src.pose_control.qwen21_union.cache import precompute_supervision, load_supervision

    Image.new("RGB", (940, 1280), (64, 64, 64)).save(record["target_rgb"])
    record["target_transform"] = {"original_wh": [940, 1280], "resized_wh": [376, 512],
                                  "offset_xy": [68, 0], "canvas_wh": [512, 512]}

    class InspectProcessor(FakeProcessor):
        def preprocess(self, image, height, width):
            assert image.getpixel((0, 0)) == (255, 255, 255, 255)
            assert image.getpixel((67, 256)) == (255, 255, 255, 255)
            assert image.getpixel((68, 256)) == (64, 64, 64, 255)
            assert image.getpixel((443, 256)) == (64, 64, 64, 255)
            assert image.getpixel((444, 256)) == (255, 255, 255, 255)
            return super().preprocess(image, height, width)

    pipe = FakePipe()
    pipe.image_processor = InspectProcessor()
    precompute_supervision(pipe, [record], tmp_path / "cache", "model-v1")
    assert load_supervision(tmp_path / "cache", record["sample_id"])["valid_target_tokens"].sum() == 752


def test_corrupt_cache_fails_without_relabeling_corrupt_bytes(tmp_path, record):
    from src.pose_control.qwen21_union.cache import precompute_conditions, load_condition

    pipe = FakePipe()
    metadata = precompute_conditions(pipe, [record], tmp_path / "cache", "model-v1")[record["sample_id"]]
    path = tmp_path / "cache" / "conditions" / f"{metadata['hash']}.pt"
    with path.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="payload fingerprint"):
        load_condition(tmp_path / "cache", record["sample_id"])
    with pytest.raises(ValueError, match="payload fingerprint"):
        precompute_conditions(pipe, [record], tmp_path / "cache", "model-v1")


def test_target_transform_size_mismatch_is_rejected(tmp_path, record):
    from src.pose_control.qwen21_union.cache import precompute_supervision

    record["target_transform"] = {"original_wh": [940, 1280], "resized_wh": [376, 512],
                                  "offset_xy": [68, 0], "canvas_wh": [512, 512]}
    with pytest.raises(ValueError, match="original_wh"):
        precompute_supervision(FakePipe(), [record], tmp_path / "cache", "model-v1")


def test_fingerprints_recompute_from_config_without_encoder_instance(tmp_path, record):
    from src.pose_control.qwen21_union.cache import (
        precompute_conditions, precompute_supervision, expected_condition_fingerprint,
        expected_supervision_fingerprint, cache_model_signature, condition_fingerprint,
    )

    pipe = FakePipe()
    config = vars(pipe.vae.config)
    conditions = precompute_conditions(pipe, [record], tmp_path / "cache", "model-v1")
    targets = precompute_supervision(pipe, [record], tmp_path / "cache", "model-v1")
    expected = expected_condition_fingerprint(record, "model-v1", config, dtype=torch.float32)
    assert expected == conditions[record["sample_id"]]["hash"]
    assert expected == condition_fingerprint(record, cache_model_signature("model-v1", config), torch.float32)
    assert expected_supervision_fingerprint(record, "model-v1", config, dtype=torch.float32) == targets[record["sample_id"]]["hash"]
    changed_config = dict(config, latents_std=[3.0] * 64)
    assert expected_condition_fingerprint(record, "model-v1", changed_config, dtype=torch.float32) != expected


def test_target_resampling_matches_shared_rgb_preprocessing(tmp_path):
    from src.pose_control.qwen21_union.cache import _open_target

    image = Image.new("RGB", (48, 96))
    for x in range(48):
        for y in range(96):
            image.putpixel((x, y), (255, 255, 255) if (x + y) % 2 else (0, 0, 0))
    path = tmp_path / "target.png"
    image.save(path)
    transform = {"original_wh": [48, 96], "resized_wh": [256, 512], "offset_xy": [128, 0], "canvas_wh": [512, 512]}
    actual = _open_target(path, transform)
    expected = Image.new("RGBA", (512, 512), (255, 255, 255, 255))
    expected.paste(image.convert("RGBA").resize((256, 512), Image.Resampling.LANCZOS), (128, 0))
    assert actual.tobytes() == expected.tobytes()
