import numpy as np
import pytest
import torch
from PIL import Image


def test_square_person_crop_preserves_tall_support_and_vae_alignment():
    from src.pose_control.qwen21_pro.features import square_person_crop
    rgb = np.zeros((100, 40, 3), np.uint8)
    rgb[:, 18:22] = 255
    mask = np.zeros((100, 40), bool)
    mask[:, 18:22] = True
    crop, support, xy = square_person_crop(rgb, mask, size=224, margin=0)
    assert crop.size == (224, 224)
    assert support[0].any() and support[-1].any()
    assert xy.shape == (256, 2)
    assert xy[0, 1] < xy[-1, 1]
    assert xy[:, 0].min() < 0  # Square padding is represented, never sampled as valid.


def test_extract_raw_features_uses_same_source_coordinates_and_frozen_cpu_outputs(tmp_path):
    from src.pose_control.qwen21_pro.features import FrozenSourceFeatureExtractor
    from test_qwen21_pro_preparation import fixture_assets
    record, cached, assets = fixture_assets(tmp_path, dual=True)
    class Processor:
        def __call__(self, images, return_tensors, **kwargs):
            assert images.size == (224, 224)
            assert kwargs["do_resize"] is False and kwargs["do_center_crop"] is False
            return {"pixel_values": torch.zeros(1, 3, 224, 224)}
    class Dino(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
        def forward(self, pixel_values):
            assert not torch.is_grad_enabled()
            return type("Output", (), {"last_hidden_state": torch.ones(1, 257, 1536) * self.weight})()
    dino = Dino()
    extractor = FrozenSourceFeatureExtractor(dino, Processor(), model_signature={"weights": "fixture"})
    with torch.inference_mode():
        pack = extractor.extract(record, cached, project_root=tmp_path, assets=assets)
    assert pack.tokens.shape == (1, 2, 256, 1600)
    assert pack.tokens[..., :64].max() <= 1
    assert pack.tokens[..., 64:].max() == 1
    assert pack.valid.any()
    assert not pack.face_identity_valid.any()
    assert not pack.tokens.is_inference()
    assert not pack.tokens.requires_grad and not dino.weight.requires_grad
    assert pack.person_ids == (1, 0)


def test_raw_pack_rejects_trainable_or_wrong_width():
    from src.pose_control.qwen21_pro.features import RawSourceFeatures
    with pytest.raises(ValueError):
        RawSourceFeatures((0,), torch.ones(1, 1, 3, 64), torch.ones(1, 1, 3, dtype=torch.bool),
                          torch.zeros(1, 1, 512), torch.zeros(1, 1, dtype=torch.bool), "hash", {}).validate()


def test_effect_pack_is_bound_to_original_source_bytes_and_accepts_real_config(tmp_path):
    from src.pose_control.qwen21_pro.features import FrozenSourceFeatureExtractor
    from src.pose_control.qwen21_pro.preparation import prepare_task_condition
    from src.pose_control.qwen21_pro.contracts import ProAdapterConfig
    from test_qwen21_pro_preparation import fixture_assets
    record, cached, assets = fixture_assets(tmp_path)
    class Processor:
        def __call__(self, **kwargs):
            return {"pixel_values": torch.zeros(1, 3, 224, 224)}
    class Dino(torch.nn.Module):
        def forward(self, **kwargs):
            return type("Output", (), {"last_hidden_state": torch.ones(1, 257, 1536)})()
    extractor = FrozenSourceFeatureExtractor(Dino(), Processor(), model_signature={"dino": "fixture"})
    pack = extractor.extract(record, cached, project_root=tmp_path, assets=assets)
    result = prepare_task_condition(record, cached, project_root=tmp_path, config=ProAdapterConfig(source_dim=1600),
                                    raw_source_features=pack, assets=assets)
    assert result.source_tokens.shape[-1] == 1600
    # Source-only fingerprint is not changed by an unreadable target appearance path.
    record["target_rgb"] = "other_missing.jpg"
    prepare_task_condition(record, cached, project_root=tmp_path, config=ProAdapterConfig(source_dim=1600),
                           raw_source_features=pack, assets=assets)
    Image.new("RGB", (16, 32), (20, 30, 40)).save(assets["source_rgb"])
    with pytest.raises(ValueError, match="fingerprint"):
        prepare_task_condition(record, cached, project_root=tmp_path, config=ProAdapterConfig(source_dim=1600),
                               raw_source_features=pack, assets=assets)


def test_locked_original_source_drift_is_rejected_before_vae_dino_mixing(tmp_path):
    from src.pose_control.qwen21_pro.preparation import prepare_task_condition
    from src.pose_control.qwen21_pro.cache import file_sha256
    from src.pose_control.qwen21_pro.contracts import ProAdapterConfig
    from test_qwen21_pro_preparation import fixture_assets, bind_fixture_cache
    record, cached, assets = fixture_assets(tmp_path)
    source = str(assets["source_rgb"])
    record["metadata"].update(original_source_rgb=source, original_condition_assets={source: file_sha256(source)})
    cached = bind_fixture_cache(tmp_path, record, dict(cached), name="declared_source_union_cache")
    Image.new("RGB", (16, 32), (1, 2, 3)).save(source)
    with pytest.raises(ValueError, match="locked source"):
        prepare_task_condition(record, cached, project_root=tmp_path, config=ProAdapterConfig(), assets=assets)


def test_extractor_rejects_bare_or_mutated_source_cache_before_models_run(tmp_path):
    from src.pose_control.qwen21_pro.features import FrozenSourceFeatureExtractor
    from test_qwen21_pro_preparation import fixture_assets
    record, cached, assets = fixture_assets(tmp_path)
    extractor = FrozenSourceFeatureExtractor(torch.nn.Identity(), None, model_signature={"fixture": "unused"})
    with pytest.raises(ValueError, match="verified|BoundUnionCondition"):
        extractor.extract(record, dict(cached), project_root=tmp_path, assets=assets)
    cached["reference_latents"][:, :16] = 9
    with pytest.raises(ValueError, match="binding"):
        extractor.extract(record, cached, project_root=tmp_path, assets=assets)


def test_raw_feature_fingerprint_stays_source_only_across_verified_pose_requests(tmp_path):
    import copy
    from src.pose_control.qwen21_pro.features import FrozenSourceFeatureExtractor
    from test_qwen21_pro_preparation import fixture_assets, bind_fixture_cache
    record, cached, assets = fixture_assets(tmp_path)
    class Processor:
        def __call__(self, **kwargs):
            return {"pixel_values": torch.zeros(1, 3, 224, 224)}
    class Dino(torch.nn.Module):
        def forward(self, **kwargs):
            return type("Output", (), {"last_hidden_state": torch.ones(1, 257, 1536)})()
    extractor = FrozenSourceFeatureExtractor(Dino(), Processor(), model_signature={"dino": "fixture"})
    before = extractor.extract(record, cached, project_root=tmp_path, assets=assets)
    another_record = copy.deepcopy(record)
    another_record.update(sample_id="another_request", prompt="a different requested pose")
    another_pose = tmp_path / "different_pose.png"
    Image.new("RGB", (32, 32), (50, 40, 30)).save(another_pose)
    another_record["reference_pose_images"] = [str(another_pose)]
    payload = copy.deepcopy(dict(cached))
    payload["reference_latents"][:, 16:] = 999
    another_bound = bind_fixture_cache(tmp_path, another_record, payload, name="another_pose_cache")
    assert cached.attestation["condition_fingerprint"] != another_bound.attestation["condition_fingerprint"]
    after = extractor.extract(another_record, another_bound, project_root=tmp_path, assets=assets)
    assert before.fingerprint == after.fingerprint
    assert torch.equal(before.tokens, after.tokens)


def test_verified_source_mesh_rois_append_native_source_patches_and_hash_camera(tmp_path):
    import copy
    from src.pose_control.qwen21_pro.features import FrozenSourceFeatureExtractor
    from src.pose_control.qwen21_pro.cache import file_sha256
    from src.pose_control.qwen21_pro.geometry_preparation import validate_source_geometry
    from test_qwen21_pro_preparation import fixture_assets
    record, cached, assets = fixture_assets(tmp_path, dual=True)
    camera = tmp_path / 'own_source_camera.npz'; camera.write_bytes(b'camera-v1')
    assets['source_geometry'] = dict(person_ids=list(record['person_order']),
        rois=[{'face': [9, 1, 16, 12], 'left_hand': [9, 12, 16, 22]}, {'face': [0, 1, 7, 12]}],
        asset_files={'source_camera': str(camera)}, asset_hashes={'source_camera': file_sha256(camera)},
        policy={'min_support_pixels': 8, 'landmarks': 'head_mesh_bounds_not_face_landmarks'},
        source_rgb_sha256=file_sha256(assets['source_rgb']), source_image_wh=[16, 32],
        code_signature={'test_producer': 'explicit-fixture'})
    class Processor:
        def __call__(self, **kwargs): return {'pixel_values': torch.zeros(1, 3, 224, 224)}
    class Dino(torch.nn.Module):
        def forward(self, **kwargs): return type('Out', (), {'last_hidden_state': torch.ones(1, 257, 1536)})()
    extractor = FrozenSourceFeatureExtractor(Dino(), Processor(), model_signature={'dino': 'fixture'})
    value = extractor.extract(record, cached, project_root=tmp_path, assets=assets)
    assert value.tokens.shape[2] == 768
    assert len(value.provenance['coverage'][0]['crops']) == 3
    assert value.provenance['coverage'][0]['crops'][1]['kind'] == 'face'
    camera.write_bytes(b'camera-v2')
    with pytest.raises(ValueError, match='source geometry|source.*SHA|fingerprint'):
        value.validate_binding(record, cached['reference_latents'][:, :16], assets, cached.producer_signature)
