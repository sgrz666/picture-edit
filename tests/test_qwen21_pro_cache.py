import pytest
import torch


def _raw_payload(tokens=None):
    return dict(person_ids=(0,), tokens=torch.ones(1, 1, 2, 1600) if tokens is None else tokens,
                valid=torch.ones(1, 1, 2, dtype=torch.bool), face_identity=torch.zeros(1, 1, 512),
                face_identity_valid=torch.zeros(1, 1, dtype=torch.bool), fingerprint="source", provenance={})


def test_cache_verifies_config_staleness_corruption_and_refuses_overwrite(tmp_path):
    from src.pose_control.qwen21_pro.cache import ProCache
    cache = ProCache(tmp_path, namespace="raw_source")
    payload = _raw_payload()
    meta = cache.put("sample", payload, "a" * 64, signature="sig")
    assert torch.equal(cache.get("sample", expected_hash="a" * 64, signature="sig")["tokens"], payload["tokens"])
    with pytest.raises(ValueError, match="signature"):
        cache.get("sample", expected_hash="a" * 64, signature="other")
    with pytest.raises(ValueError, match="fingerprint"):
        cache.get("sample", expected_hash="b" * 64, signature="sig")
    with pytest.raises(FileExistsError):
        cache.put("sample", _raw_payload(torch.zeros(1, 1, 2, 1600)), "b" * 64, signature="sig")
    with open(meta["path"], "ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="payload"):
        cache.get("sample", expected_hash="a" * 64, signature="sig")


@pytest.mark.parametrize("folder", ["datasets", "models", "conditions", "supervision"])
def test_cache_cannot_write_protected_or_legacy_namespaces(tmp_path, folder):
    from src.pose_control.qwen21_pro.cache import ProCache
    with pytest.raises(ValueError):
        ProCache(tmp_path / folder, namespace="raw_source")


def test_source_fingerprint_changes_only_with_source_and_encoder(tmp_path):
    from src.pose_control.qwen21_pro.cache import source_feature_fingerprint, typed_pack_fingerprint
    from src.pose_control.qwen21_pro.contracts import ProAdapterConfig
    source, geometry = tmp_path / "source.bin", tmp_path / "geometry.bin"
    source.write_bytes(b"source1")
    geometry.write_bytes(b"geom1")
    kwargs = dict(source_assets={"source": source}, model_signature={"dino": "sha"},
                  person_order=[1, 0], source_transform={"axis": [1., 2.]}, source_latent=torch.ones(1, 4, 64))
    a = source_feature_fingerprint(**kwargs)
    typed_args = dict(source_fingerprint=a, geometry_assets={"geometry": geometry}, condition={"person_order": [1, 0]},
                      target_hw=(2, 2), config=ProAdapterConfig(), geometry_codec={"version": 1})
    ta = typed_pack_fingerprint(**typed_args)
    geometry.write_bytes(b"geom2")
    assert a == source_feature_fingerprint(**kwargs)
    assert ta != typed_pack_fingerprint(**typed_args)
    source.write_bytes(b"source2")
    assert a != source_feature_fingerprint(**kwargs)


def test_cache_converts_inference_tensors_to_ordinary_cpu(tmp_path):
    from src.pose_control.qwen21_pro.cache import ProCache
    cache = ProCache(tmp_path, namespace="raw_source")
    with torch.inference_mode():
        value = torch.ones(1, 1, 2, 1600)
        cache.put("a", _raw_payload(value), "c" * 64, signature="sig")
        restored = cache.get("a", expected_hash="c" * 64, signature="sig")
    assert not restored["tokens"].is_inference()


def test_cache_rejects_trainable_or_resampler_outputs(tmp_path):
    from src.pose_control.qwen21_pro.cache import ProCache
    cache = ProCache(tmp_path, namespace="raw_source")
    with pytest.raises(ValueError, match="frozen"):
        cache.put("a", {"tokens": torch.ones(1, requires_grad=True)}, "d" * 64, signature="sig")
    with pytest.raises(ValueError, match="resampler"):
        cache.put("b", {"resampled_tokens": torch.ones(1)}, "e" * 64, signature="sig")


def test_typed_fingerprint_ignores_target_appearance_but_binds_config_and_pose(tmp_path):
    from src.pose_control.qwen21_pro.cache import typed_pack_fingerprint
    from src.pose_control.qwen21_pro.contracts import ProAdapterConfig
    pose = tmp_path / "pose.bin"
    pose.write_bytes(b"pose1")
    condition = {"dataset": "hi4d", "person_order": [1, 0], "reference_pose_images": [str(pose)],
                 "target_rgb": "unreadable", "metadata": {"original_target_sha256": "ignored"}}
    kwargs = dict(source_fingerprint="source", geometry_assets={}, condition=condition,
                  target_hw=(4, 4), config=ProAdapterConfig(), geometry_codec={"version": 1})
    before = typed_pack_fingerprint(**kwargs)
    condition["target_rgb"] = "another_unreadable"
    condition["metadata"]["original_target_sha256"] = "another"
    assert before == typed_pack_fingerprint(**kwargs)
    assert before != typed_pack_fingerprint(**{**kwargs, "config": ProAdapterConfig(source_dim=1600)})
    pose.write_bytes(b"pose2")
    assert before != typed_pack_fingerprint(**kwargs)


def test_raw_namespace_rejects_detached_nonraw_tokens(tmp_path):
    from src.pose_control.qwen21_pro.cache import ProCache
    cache = ProCache(tmp_path, namespace="raw_source")
    with pytest.raises(ValueError, match="raw"):
        cache.put("hidden_trainable_output", {"tokens": torch.zeros(1, 1, 256, 512)}, "f" * 64, signature="sig")


def test_typed_cache_roundtrip_requires_matching_real_config(tmp_path):
    from src.pose_control.qwen21_pro.cache import ProCache
    from src.pose_control.qwen21_pro.contracts import ProAdapterConfig, TaskCondition
    from src.pose_control.qwen21_pro.preparation import prepare_task_condition
    from test_qwen21_pro_preparation import fixture_assets
    record, cached, assets = fixture_assets(tmp_path)
    config = ProAdapterConfig()
    condition = prepare_task_condition(record, cached, project_root=tmp_path, config=config, assets=assets)
    cache = ProCache(tmp_path / "cache", namespace="typed_condition")
    cache.put("a", condition, "a" * 64, signature=config.signature(), config=config)
    result = TaskCondition.from_dict(cache.get("a", expected_hash="a" * 64, signature=config.signature(), config=config))
    result.validate(config)
    with pytest.raises(ValueError, match="config|signature"):
        cache.get("a", expected_hash="a" * 64, signature=config.signature(), config=ProAdapterConfig(width=256))


def test_geometry_asset_roles_bind_camera_id_and_part_codec(tmp_path):
    from src.pose_control.qwen21_pro.cache import typed_pack_fingerprint
    from src.pose_control.qwen21_pro.preparation import geometry_asset_roles, GEOMETRY_CODEC
    from src.pose_control.qwen21_pro.contracts import ProAdapterConfig
    geometry = tmp_path / "geometry.bin"
    geometry.write_bytes(b"same bytes")
    assets = {"geometry": geometry, "camera_id": 4, "people": [{"part_codec": "v6_14_plus1_bg0"}]}
    def fingerprint():
        return typed_pack_fingerprint(source_fingerprint="source", geometry_assets=geometry_asset_roles(assets),
                   condition={"person_order": [1, 0]}, target_hw=(4, 4), config=ProAdapterConfig(), geometry_codec=GEOMETRY_CODEC)
    before = fingerprint()
    assets["camera_id"] = 76
    assert before != fingerprint()
    before = fingerprint()
    assets["people"][0]["part_codec"] = "v6_14_zero_based_invalid_minus1"
    assert before != fingerprint()


def test_bound_loader_rejects_wrong_producer_and_corrupted_old_payload(tmp_path):
    from src.pose_control.qwen21_pro.cache import load_bound_condition
    from src.pose_control.qwen21_union.cache import UnionConditionCache
    from test_qwen21_pro_preparation import fixture_assets, FIXTURE_MODEL_SIGNATURE, FIXTURE_VAE_CONFIG
    record, _, _ = fixture_assets(tmp_path)
    root = tmp_path / "union_cache"
    with pytest.raises(ValueError, match="fingerprint"):
        load_bound_condition(record, root, model_signature={"producer": "wrong"},
                              vae_config=FIXTURE_VAE_CONFIG, dtype="torch.float32")
    meta = UnionConditionCache(root).metadata(record["sample_id"])
    with open(meta["path"], "ab") as stream:
        stream.write(b"bad_payload")
    with pytest.raises(ValueError, match="payload|fingerprint"):
        load_bound_condition(record, root, model_signature=FIXTURE_MODEL_SIGNATURE,
                              vae_config=FIXTURE_VAE_CONFIG, dtype="torch.float32")


@pytest.mark.parametrize("change", ["sample", "prepared_source", "source_tensor", "other_tensor", "record_prompt"])
def test_bound_condition_rejects_loaded_or_record_mutation(tmp_path, change):
    from src.pose_control.qwen21_pro.preparation import prepare_task_condition
    from src.pose_control.qwen21_pro.contracts import ProAdapterConfig
    from test_qwen21_pro_preparation import fixture_assets
    record, cached, assets = fixture_assets(tmp_path)
    if change == "sample":
        record["sample_id"] = "different-sample"
    elif change == "prepared_source":
        from PIL import Image
        Image.new("RGB", (32, 32), (9, 9, 9)).save(record["source_rgb"])
    elif change == "source_tensor":
        cached["reference_latents"][:, :16] = 9
    elif change == "other_tensor":
        cached["reference_latents"][:, 16:] = 9
    else:
        record["prompt"] = "another request"
    with pytest.raises(ValueError, match="binding"):
        prepare_task_condition(record, cached, project_root=tmp_path, config=ProAdapterConfig(), assets=assets)


def test_verified_condition_factory_does_not_accept_a_payload_to_stamp():
    from src.pose_control.qwen21_pro.cache import BoundUnionCondition
    with pytest.raises((TypeError, ValueError)):
        BoundUnionCondition({"reference_latents": torch.full((1, 16, 64), 9.)})


def test_verified_cache_from_other_sample_cannot_be_spliced_into_current_source(tmp_path):
    import copy
    from src.pose_control.qwen21_pro.preparation import prepare_task_condition
    from src.pose_control.qwen21_pro.contracts import ProAdapterConfig
    from test_qwen21_pro_preparation import fixture_assets, bind_fixture_cache
    record, cached, assets = fixture_assets(tmp_path)
    another_record = copy.deepcopy(record)
    another_record["sample_id"] = "another-sample"
    another_payload = copy.deepcopy(dict(cached))
    another_payload["reference_latents"][:, :16] = 9
    another_bound = bind_fixture_cache(tmp_path, another_record, another_payload, name="another_union_cache")
    with pytest.raises(ValueError, match="binding"):
        prepare_task_condition(record, another_bound, project_root=tmp_path, config=ProAdapterConfig(), assets=assets)


@pytest.mark.parametrize("dependency", ["src/data/qwen21_union_dataset.py", "src/pose_control/qwen21_union/contracts.py"])
def test_shared_keypoint_and_transform_source_changes_invalidate_both_fingerprints(monkeypatch, dependency):
    from src.pose_control.qwen21_pro import cache
    from src.pose_control.qwen21_pro.contracts import ProAdapterConfig
    source_kwargs = dict(source_assets={}, model_signature={"weights": "fixture"}, person_order=[0],
                         source_transform={"ratio": [1., 1.]}, source_latent=torch.ones(1, 4, 64))
    typed_kwargs = dict(source_fingerprint="fixed_source", geometry_assets={}, condition={"person_order": [0]},
                        target_hw=(2, 2), config=ProAdapterConfig(), geometry_codec={"version": 1})
    source_before = cache.source_feature_fingerprint(**source_kwargs)
    typed_before = cache.typed_pack_fingerprint(**typed_kwargs)
    original = cache.file_sha256
    def changed_dependency(path):
        from pathlib import Path
        return "changed-shared-algorithm" if Path(path).as_posix().endswith(dependency) else original(path)
    monkeypatch.setattr(cache, "file_sha256", changed_dependency)
    assert source_before != cache.source_feature_fingerprint(**source_kwargs)
    assert typed_before != cache.typed_pack_fingerprint(**typed_kwargs)
    assert dependency in cache.code_signature()
