from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from test_qwen21_pro_inference import grid_pack


class Encoder:
    _execution_device = "cpu"
    vae_scale_factor = 16

    def __init__(self):
        class VAE(torch.nn.Module):
            config = SimpleNamespace(
                z_dim=64, latents_mean=[2.0] * 64, latents_std=[3.0] * 64
            )

            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.ones(()))

            def encode(self, pixels):
                z = (
                    torch.ones(1, 64, 1, pixels.shape[-2] // 16, pixels.shape[-1] // 16)
                    * 5
                )
                return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: z))

        self.vae, self.text_encoder = VAE(), torch.nn.Identity()
        self.image_processor = SimpleNamespace(
            preprocess=lambda image, height, width: torch.zeros(1, 4, height, width)
        )

    def encode_prompt(self, *, image, prompt, device, num_images_per_prompt):
        count = sum((im.width // 16) * (im.height // 16) // 4 for im in image)
        length = count + len(prompt)
        slots = torch.zeros(1, length, dtype=torch.bool)
        slots[:, :count] = True
        return torch.zeros(1, length, 8), None, slots


def encoded(tmp_path):
    from src.pose_control.qwen21_pro.encoded import encode_condition
    from src.pose_control.qwen21_pro.contracts import ProAdapterConfig

    source, control = tmp_path / "source.png", tmp_path / "control.png"
    Image.new("RGB", (512, 512)).save(source)
    Image.new("RGB", (768, 768)).save(control)
    pack = grid_pack((48, 48))
    value = encode_condition(
        Encoder(),
        sample_id="roi",
        source_images=[source],
        control_image=control,
        canvas_hw=(768, 768),
        person_ids=(0,),
        prompt="positive",
        negative_prompt="negative text",
        producer_signature={"weights": "frozen"},
        geometry_signature={"request": "current"},
        source_fingerprint=pack.source_fingerprint,
    )
    pack = replace(pack, provenance={"geometry": value.binding["geometry"]})
    value = value.with_task_condition(pack, config=ProAdapterConfig())
    return value, pack, source, control


def test_encode_native_variable_grid_and_own_cache_attestation(tmp_path):
    from src.pose_control.qwen21_pro.encoded import (
        EncodedConditionCache,
        sample_validator,
    )
    from src.pose_control.qwen21_pro.contracts import ProAdapterConfig

    value, pack, source, control = encoded(tmp_path)
    assert value.payload["reference_latents"].shape == (1, 1024, 64)
    assert value.payload["control_latents"].shape == (1, 2304, 64)
    assert value.payload["img_shapes"] == [[(1, 32, 32), (1, 48, 48)]]
    cache = EncodedConditionCache(tmp_path / "cache")
    cache.put(value, pack, config=ProAdapterConfig())
    restored, bound_pack = cache.load(
        "roi", expected_binding=value.binding, config=ProAdapterConfig()
    )
    sample_validator(restored, bound_pack, "roi").validate(bound_pack, "roi")
    restored["control_latents"].add_(1)
    with pytest.raises(ValueError, match="payload"):
        sample_validator(restored, bound_pack, "roi")


@pytest.mark.parametrize("mutate", ["source", "geometry", "bank", "producer", "pack"])
def test_encoded_cache_rejects_changed_source_geometry_bank_producer_and_pack(
    tmp_path, mutate
):
    from src.pose_control.qwen21_pro.encoded import (
        EncodedConditionCache,
        sample_validator,
    )
    from src.pose_control.qwen21_pro.contracts import ProAdapterConfig

    value, pack, source, control = encoded(tmp_path)
    cache = EncodedConditionCache(tmp_path / "cache")
    cache.put(value, pack, config=ProAdapterConfig())
    if mutate in ("source", "geometry"):
        Image.new("RGB", (512, 512), "red").save(
            source if mutate == "source" else control
        )
        with pytest.raises(ValueError):
            cache.load("roi", expected_binding=value.binding, config=ProAdapterConfig())
    elif mutate == "producer":
        import copy

        binding = copy.deepcopy(value.binding)
        binding["producer"]["weights"] = "new"
        with pytest.raises(ValueError):
            cache.load("roi", expected_binding=binding, config=ProAdapterConfig())
    else:
        restored, bound_pack = cache.load(
            "roi", expected_binding=value.binding, config=ProAdapterConfig()
        )
        changed = (
            replace(bound_pack, source_tokens=bound_pack.source_tokens + 1)
            if mutate == "bank"
            else replace(bound_pack, owner=bound_pack.owner * 0)
        )
        with pytest.raises(ValueError, match="pack|binding"):
            sample_validator(restored, changed, "roi")


def test_new_record_source_features_need_no_legacy_b_cache(tmp_path):
    from src.pose_control.qwen21_pro.encoded import encode_condition
    from src.pose_control.qwen21_pro.features import FrozenSourceFeatureExtractor
    from src.pose_control.qwen21_pro.preparation import prepare_source_task_condition
    from src.pose_control.qwen21_pro.contracts import ProAdapterConfig
    from test_qwen21_pro_preparation import fixture_assets

    record, legacy, assets = fixture_assets(tmp_path, dual=True)
    native = encode_condition(
        Encoder(),
        sample_id="new-record",
        source_images=[record["source_rgb"]],
        control_image=record["control_image"],
        canvas_hw=(32, 32),
        person_ids=(1, 0),
        prompt="new pose",
        producer_signature={"frozen": "actual"},
        geometry_signature={"mode": "driver_exact"},
        source_fingerprint="initial-source-files",
    )

    class Dino(torch.nn.Module):
        def forward(self, **kw):
            return SimpleNamespace(last_hidden_state=torch.ones(1, 257, 1536))

    class Processor:
        def __call__(self, **kw):
            return {"pixel_values": torch.zeros(1, 3, 224, 224)}

    extractor = FrozenSourceFeatureExtractor(
        Dino(), Processor(), model_signature={"current": "dino"}
    )
    source = extractor.extract(record, native, project_root=tmp_path, assets=assets)
    pack = prepare_source_task_condition(
        source, config=ProAdapterConfig(source_dim=1600), target_hw=(2, 2)
    )
    assert source.person_ids == (1, 0) and pack.source_tokens.shape == (1, 2, 256, 1600)
    assert not pack.owner.any() and not any(
        v.any() for v in pack.spatial_valid.values()
    )
    pack = replace(pack, provenance={"geometry": native.binding["geometry"]})
    bound = native.with_source_binding(pack, config=ProAdapterConfig(source_dim=1600))
    assert bound.binding["source_fingerprint"] == source.fingerprint
    bound.verify(pack)
    native["reference_latents"].add_(1)
    with pytest.raises(ValueError, match="payload"):
        extractor.extract(record, native, project_root=tmp_path, assets=assets)


def test_encoded_cache_attests_raw_geometry_files_in_current_request(tmp_path):
    from src.pose_control.qwen21_pro.encoded import (
        encode_condition,
        EncodedConditionCache,
    )
    from src.pose_control.qwen21_pro.cache import file_sha256
    from src.pose_control.qwen21_pro.contracts import ProAdapterConfig

    source = tmp_path / "source.png"
    control = tmp_path / "control.png"
    raw = tmp_path / "driver_raw.npz"
    Image.new("RGB", (512, 512)).save(source)
    Image.new("RGB", (768, 768)).save(control)
    raw.write_bytes(b"raw driver-v1")
    pack = grid_pack((48, 48))
    value = encode_condition(
        Encoder(),
        sample_id="current-raw",
        source_images=[source],
        control_image=control,
        canvas_hw=(768, 768),
        person_ids=(0,),
        prompt="current pose",
        producer_signature={"current": "weights"},
        geometry_signature={
            "mode": "driver_exact",
            "input_asset_files": {"driver_raw": str(raw)},
            "input_asset_hashes": {"driver_raw": file_sha256(raw)},
        },
        source_fingerprint=pack.source_fingerprint,
    )
    pack = replace(pack, provenance={"geometry": value.binding["geometry"]})
    value = value.with_task_condition(pack, config=ProAdapterConfig())
    cache = EncodedConditionCache(tmp_path / "raw-cache")
    cache.put(value, pack, config=ProAdapterConfig())
    raw.write_bytes(b"raw driver-v2")
    with pytest.raises(ValueError, match="geometry|SHA"):
        cache.load(
            "current-raw", expected_binding=value.binding, config=ProAdapterConfig()
        )


@pytest.mark.parametrize("entry", ["verify", "cache", "inference"])
@pytest.mark.parametrize("geometry", ["FOREIGN_GEOMETRY", "current"])
def test_foreign_geometry_pack_is_rejected_before_first_encoded_cache_put(
    tmp_path, entry, geometry
):
    from src.pose_control.qwen21_pro.encoded import EncodedConditionCache
    from src.pose_control.qwen21_pro.contracts import ProAdapterConfig
    from src.pose_control.qwen21_pro.inference import sample_cached
    from test_qwen21_pro_inference import Transformer
    from diffusers import FlowMatchEulerDiscreteScheduler

    value, pack, *_ = encoded(tmp_path)
    wrong = replace(
        pack,
        spatial={**pack.spatial, "normal": pack.spatial["normal"] + 0.123},
        provenance={"geometry": {"request": geometry}},
    )
    with pytest.raises(ValueError, match="geometry|pack|binding"):
        if entry == "verify":
            value.verify(wrong)
        elif entry == "cache":
            EncodedConditionCache(tmp_path / "wrong-cache").put(
                value, wrong, config=ProAdapterConfig()
            )
        else:
            sample_cached(
                SimpleNamespace(
                    transformer=Transformer(),
                    scheduler=FlowMatchEulerDiscreteScheduler(
                        use_dynamic_shifting=True
                    ),
                    _execution_device="cpu",
                ),
                value,
                task_condition=wrong,
                num_inference_steps=1,
            )


def test_unsealed_source_encoding_is_not_a_cache_or_sampling_attestation(tmp_path):
    from src.pose_control.qwen21_pro.encoded import (
        EncodedCondition,
        EncodedConditionCache,
    )
    from src.pose_control.qwen21_pro.contracts import ProAdapterConfig

    value, pack, *_ = encoded(tmp_path)
    binding = {
        k: v
        for k, v in value.binding.items()
        if k not in ("task_pack_sha256", "config")
    }
    pending = EncodedCondition(value.payload, binding)
    pending.verify()  # source feature extraction may inspect the verified raw encoding.
    with pytest.raises(ValueError, match="creation-time.*pack binding"):
        EncodedConditionCache(tmp_path / "unsealed").put(
            pending, pack, config=ProAdapterConfig()
        )
    wrong = replace(pack, provenance={"geometry": {"request": "other"}})
    with pytest.raises(ValueError, match="geometry"):
        pending.with_task_condition(wrong, config=ProAdapterConfig())
