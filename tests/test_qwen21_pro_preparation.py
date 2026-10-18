"""真实编码、小型磁盘资产与源/目标隔离的 CPU 契约。"""
import copy
import numpy as np
import pytest
import torch
from PIL import Image

from src.pose_control.qwen21_pro.contracts import ProAdapterConfig


FIXTURE_MODEL_SIGNATURE = {"source_vae_weights": "fixture-frozen-producer", "producer": "unit-test"}
FIXTURE_VAE_CONFIG = {"z_dim": 64, "latents_mean": [0.] * 64, "latents_std": [1.] * 64}


def bind_fixture_cache(tmp_path, record, payload, name="union_cache"):
    from src.pose_control.qwen21_union.cache import UnionConditionCache, expected_condition_fingerprint
    from src.pose_control.qwen21_pro.cache import load_bound_condition
    root = tmp_path / name
    dtype = str(payload["reference_latents"].dtype)
    fingerprint = expected_condition_fingerprint(record, FIXTURE_MODEL_SIGNATURE, FIXTURE_VAE_CONFIG, dtype)
    UnionConditionCache(root).put(record["sample_id"], payload, fingerprint)
    return load_bound_condition(record, root, model_signature=FIXTURE_MODEL_SIGNATURE,
                                vae_config=FIXTURE_VAE_CONFIG, dtype=dtype)


def fixture_assets(tmp_path, dual=False):
    transform = dict(original_wh=[16, 32], resized_wh=[16, 32], offset_xy=[8, 0], canvas_wh=[32, 32])
    rgb = tmp_path / "source_original.png"
    Image.new("RGB", (16, 32), (200, 60, 30)).save(rgb)
    prepared_rgb = tmp_path / "source.png"
    Image.new("RGB", (32, 32), (200, 60, 30)).save(prepared_rgb)
    a = np.zeros((32, 16), np.uint8)
    a[:, :8] = 255
    b = 255 - a
    masks = []
    for name, value in (("A", a), ("B", b)):
        path = tmp_path / f"mask_{name}.png"
        Image.fromarray(value).save(path)
        masks.append(path)
    order = [1, 0] if dual else [0]
    pose_paths = []
    for slot in range(len(order)):
        path = tmp_path / f"pose_{slot}.png"
        Image.new("RGB", (32, 32), (slot, 0, 0)).save(path)
        pose_paths.append(str(path))
    record = dict(sample_id="fixture", dataset="hi4d" if dual else "iper", person_order=order,
                  source_rgb=str(prepared_rgb), source_transform=transform, target_transform=transform,
                  reference_pose_images=pose_paths, control_image=pose_paths[0], prompt="fixture pose request",
                  target_rgb="unreadable/target_rgb.jpg", metadata={"original_target_sha256": "DO_NOT_COPY",
                                                                 "original_source_rgb": str(rgb)})
    latent = torch.cat([torch.ones(1, 16, 64)] + [torch.full((1, 16, 64), 500.)] * len(order), 1)
    cached = dict(reference_latents=latent, target_hw=(4, 4), canvas_hw=(32, 32),
                  person_order=order, img_shapes=[[(1, 4, 4)] * (len(order) + 2)],
                  transforms={k: transform for k in ("source_transform", "target_transform")})
    assets = dict(source_rgb=rgb, source_masks=[masks[i] for i in order],
                  source_keypoints=[None] * len(order), target_keypoints=[None] * len(order))
    if dual:
        geometry = tmp_path / "geometry.npz"
        owner = np.where(a > 0, 1, 2).astype(np.int64)
        np.savez(geometry, depth_scene=np.where(a > 0, 2., 4.), depth_A=np.full(a.shape, 2.),
                 depth_B=np.full(a.shape, 4.), semantic_A=a > 0, semantic_B=b > 0,
                 front_owner=owner, contact_A=np.zeros(a.shape), contact_B=np.zeros(a.shape),
                 normal_scene=np.ones((*a.shape, 3)))
        assets["geometry"] = geometry
    else:
        assets["people"] = [{"mask": masks[0]}]
    return record, bind_fixture_cache(tmp_path, record, cached), assets


def test_hi4d_raw_order_maps_to_slots_and_pose_latents_never_enter_bank(tmp_path):
    from src.pose_control.qwen21_pro.preparation import prepare_task_condition
    record, cached, assets = fixture_assets(tmp_path, dual=True)
    result = prepare_task_condition(record, cached, project_root=tmp_path, config=ProAdapterConfig(), assets=assets)
    assert result.person_ids == (1, 0)
    assert result.source_tokens.max() == 1
    assert result.owner[0, 0].reshape(4, 4)[:, 2].min() == 1
    assert result.owner[0, 1].reshape(4, 4)[:, 1].min() == 1
    assert result.owner.sum(1).max() <= 1
    assert not result.spatial_valid["normal"].any()
    assert not result.spatial_valid["part"].any()
    assert (result.spatial["part"] == -1).all()
    assert result.spatial_valid["contact"].all()
    assert result.spatial["contact"].sum() == 0
    assert result.relations.valid.shape == (1, 0)
    assert "DO_NOT_COPY" not in str(result.provenance)
    result.validate(ProAdapterConfig())


@pytest.mark.parametrize("mutation", ["order", "grid", "source_grid", "transform"])
def test_locked_cache_binding_mismatch_rejected(tmp_path, mutation):
    from src.pose_control.qwen21_pro.preparation import prepare_task_condition
    record, cached, assets = fixture_assets(tmp_path, dual=True)
    cached = copy.deepcopy(dict(cached))
    if mutation == "order":
        cached["person_order"] = [0, 1]
    elif mutation == "grid":
        cached["target_hw"] = (2, 8)
    elif mutation == "source_grid":
        cached["img_shapes"][0][0] = (1, 2, 8)
    else:
        cached["transforms"]["source_transform"]["offset_xy"] = [0, 0]
    cached = bind_fixture_cache(tmp_path, record, cached, name="mutated_union_cache")
    with pytest.raises(ValueError):
        prepare_task_condition(record, cached, project_root=tmp_path, config=ProAdapterConfig(), assets=assets)


def test_missing_source_mask_disables_person_not_borrow_geometry(tmp_path):
    from src.pose_control.qwen21_pro.preparation import prepare_task_condition
    record, cached, assets = fixture_assets(tmp_path, dual=True)
    assets["source_masks"][0] = None
    result = prepare_task_condition(record, cached, project_root=tmp_path, config=ProAdapterConfig(), assets=assets)
    assert result.person_valid.tolist() == [[False, True]]
    assert not result.source_token_valid[:, 0].any()
    assert result.owner[:, 0].sum() == 0
    assert not any(v[:, 0].any() for v in result.spatial_valid.values())


def test_normal_decode_camera_cv_and_background_zero(tmp_path):
    from src.pose_control.qwen21_pro.preparation import decode_renderer_normal
    rgb = np.array([[[128, 128, 255], [255, 0, 0]]], dtype=np.uint8)
    normal = decode_renderer_normal(rgb, np.array([[1, 0]], bool))
    assert normal[0, 0, 2] < -.999
    assert np.linalg.norm(normal[0, 0]) == pytest.approx(1)
    assert not normal[0, 1].any()


def test_transform_uses_actual_axis_ratios_and_stretched_part_full_fov():
    from src.pose_control.qwen21_pro.preparation import map_points, transform_map
    transform = dict(original_wh=[604, 1080], resized_wh=[286, 512], offset_xy=[113, 0], canvas_wh=[512, 512])
    points = np.array([[0., 0.], [603., 1079.]])
    assert np.allclose(map_points(map_points(points, transform), transform, inverse=True), points)
    full_fov = np.ones((512, 512), dtype=np.int64)
    canvas = transform_map(full_fov, transform, nearest=True, fill=-1)
    assert (canvas[:, 113:399] == 1).all()
    assert (canvas[:, :113] == -1).all()


def test_effect_profile_cannot_silently_fall_back_to_vae(tmp_path):
    from src.pose_control.qwen21_pro.preparation import prepare_task_condition
    record, cached, assets = fixture_assets(tmp_path)
    with pytest.raises(ValueError, match="1600|raw source"):
        prepare_task_condition(record, cached, project_root=tmp_path, config=ProAdapterConfig(source_dim=1600), assets=assets)


def test_target_rgb_substitution_has_no_condition_effect(tmp_path):
    from src.pose_control.qwen21_pro.preparation import prepare_task_condition
    record, cached, assets = fixture_assets(tmp_path, dual=True)
    a = prepare_task_condition(record, cached, project_root=tmp_path, config=ProAdapterConfig(), assets=assets)
    record["target_rgb"] = str(tmp_path / "still_missing.png")
    record["metadata"]["original_target_sha256"] = "another"
    b = prepare_task_condition(record, cached, project_root=tmp_path, config=ProAdapterConfig(), assets=assets)
    assert a.source_fingerprint == b.source_fingerprint
    for key in a.spatial:
        assert torch.equal(a.spatial[key], b.spatial[key])
    assert torch.equal(a.source_tokens, b.source_tokens)


def test_contact_relations_use_slot_endpoints_and_explicit_camera_extent_codec(tmp_path):
    from src.pose_control.qwen21_pro.preparation import build_contact_relations
    verts = np.array([[[0., 0., 1.], [0., 2., 1.]], [[.1, 0., 1.], [.1, 2., 1.]]])
    mesh = tmp_path / "mesh.npz"
    np.savez(mesh, verts=verts, contact=np.ones((2, 2)))
    camera = tmp_path / "cameras.npz"
    np.savez(camera, ids=[4], extrinsics=[np.column_stack([np.eye(3), np.zeros(3)])])
    labels = tmp_path / "parts.npz"
    np.savez(labels, parts=np.array([0, 1], np.int64))
    assets = dict(contact_mesh=mesh, contact_vertex_parts=labels, camera=camera, camera_id=4)
    result, meta = build_contact_relations(assets, (1, 0), torch.ones(1, 2, dtype=torch.bool))
    assert meta["valid"] and "extent" in meta["codec"]
    assert result.valid.all() and result.valid.shape == (1, 2)
    assert result.src_person.tolist() == [[1, 1]]
    assert result.dst_person.tolist() == [[0, 0]]
    assert result.relative[0, 0, 0] == pytest.approx(.05)
    assert result.relative[0, 0, 3] == pytest.approx(.05)


@pytest.mark.parametrize("raw_values", [(50., 6335.), (.01, .1)])
def test_raw_contact_positive_annotation_values_are_membership_not_probability(tmp_path, raw_values):
    from src.pose_control.qwen21_pro.preparation import build_contact_relations
    mesh, camera, parts = (tmp_path / name for name in ("mesh.npz", "cam.npz", "parts.npz"))
    vertices = np.array([[[0., 0., 1.], [0., 2., 1.]], [[.1, 0., 1.], [.1, 2., 1.]]])
    np.savez(mesh, verts=vertices, contact=np.array([[raw_values[0], 0], [raw_values[1], 0]], np.float64))
    np.savez(camera, ids=[4], extrinsics=[np.column_stack([np.eye(3), np.zeros(3)])])
    np.savez(parts, parts=np.array([4, 7], np.int64))
    assets = dict(contact_mesh=mesh, camera=camera, camera_id=4, contact_vertex_parts=parts)
    result, meta = build_contact_relations(assets, (1, 0), torch.ones(1, 2, dtype=torch.bool))
    assert result.valid.tolist() == [[True]]
    assert result.src_person.tolist() == [[1]] and result.dst_person.tolist() == [[0]]
    assert result.src_part.tolist() == [[4]] and result.dst_part.tolist() == [[4]]
    assert result.confidence.tolist() == [[1.]]
    assert meta["flag_threshold"] == 0
    assert meta["flagged_vertices"] == [1, 1]
    assert meta["raw_contact_codec"] == "finite_nonnegative_annotation_membership_gt0"
    assert meta["confidence_codec"] == "annotated_pair=1;numeric_annotation_is_not_probability"


def test_hi4d_geometry_soft_contact_is_independent_of_raw_annotation_codec(tmp_path):
    from src.pose_control.qwen21_pro.preparation import prepare_task_condition
    record, cached, assets = fixture_assets(tmp_path, dual=True)
    with np.load(assets["geometry"], allow_pickle=False) as geometry:
        values = {key: geometry[key].copy() for key in geometry.files}
    values["contact_A"] = np.full((32, 16), .25, np.float32)
    values["contact_B"] = np.full((32, 16), .75, np.float32)
    soft_geometry = tmp_path / "soft_geometry.npz"
    np.savez(soft_geometry, **values)
    assets["geometry"] = soft_geometry
    result = prepare_task_condition(record, cached, project_root=tmp_path, config=ProAdapterConfig(), assets=assets)
    assert result.spatial["contact"][0, 0].max() == pytest.approx(.75)
    assert result.spatial["contact"][0, 1].max() == pytest.approx(.25)
    assert result.spatial_valid["contact"].all()


def test_soft_contact_values_and_grayscale_visual_depth_remain_typed(tmp_path):
    from src.pose_control.qwen21_pro.preparation import prepare_task_condition
    record, cached, assets = fixture_assets(tmp_path)
    record["dataset"] = "tiktok"
    cached = bind_fixture_cache(tmp_path, record, dict(cached), name="tiktok_union_cache")
    contact, depth, part = (tmp_path / name for name in ("contact.npy", "depth.png", "part.png"))
    np.save(contact, np.full((32, 16), .4, np.float32))
    Image.fromarray(np.tile(np.arange(16, dtype=np.uint8), (32, 1))).save(depth)
    Image.fromarray(np.full((32, 16), 3, np.uint8)).save(part)
    assets["people"][0].update(contact=contact, depth=depth, depth_kind="visual_normalized",
                                part=part, part_codec="v6_14_plus1_bg0")
    result = prepare_task_condition(record, cached, project_root=tmp_path, config=ProAdapterConfig(), assets=assets)
    assert result.spatial["contact"].max() == pytest.approx(.4)
    assert result.spatial_valid["depth"].all()
    assert not result.provenance["geometry"]["depth_metric"]
    assert result.spatial["part"].max() == 2


def test_tiktok_inverse_depth_visual_codec_preserves_near_to_far_direction(tmp_path):
    from src.pose_control.qwen21_pro.preparation import prepare_task_condition
    record, cached, assets = fixture_assets(tmp_path)
    record["dataset"] = "tiktok"
    cached = bind_fixture_cache(tmp_path, record, dict(cached), name="tiktok_union_cache")
    depth = tmp_path / "depth.png"
    values = np.zeros((32, 16), np.uint8)
    values[:16] = 255  # producer: near=P1 white, far=P99 value1.
    values[16:] = 1
    Image.fromarray(values).save(depth)
    assets["people"][0].update(depth=depth, depth_kind="visual_normalized")
    result = prepare_task_condition(record, cached, project_root=tmp_path, config=ProAdapterConfig(), assets=assets)
    assert result.spatial["depth"][0, 0, 0, 0, 1] == 0
    assert result.spatial["depth"][0, 0, 0, 3, 1] == 1
    assert result.provenance["geometry"]["depth_convention"] == "near0_far1;foreground_mask_separate;background0"


def test_face_hand_roi_uses_numeric_keypoints_not_pose_rgb(tmp_path):
    from src.pose_control.qwen21_pro.preparation import prepare_task_condition
    record, cached, assets = fixture_assets(tmp_path)
    points = np.zeros((134, 2), np.float32)
    scores = np.zeros(134, np.float32)
    for block, y in ((slice(24, 92), 7), (slice(92, 113), 16), (slice(113, 134), 25)):
        n = len(points[block])
        points[block] = np.column_stack([np.linspace(1, 7, n), y + np.sin(np.arange(n)) * 2])
        scores[block] = 1
    kp = tmp_path / "points.npz"
    np.savez(kp, keypoints_134_px=points, scores_134=scores, image_width=16, image_height=32)
    assets["target_keypoints"] = [kp]
    result = prepare_task_condition(record, cached, project_root=tmp_path, config=ProAdapterConfig(), assets=assets)
    assert result.face_mask.sum() > 0
    assert (result.hand_mask.sum(-1) > 0).all()
    assert (result.face_mask <= result.person_mask).all()


def test_zero_raw_contact_is_available_but_unknown_parts_are_not(tmp_path):
    from src.pose_control.qwen21_pro.preparation import build_contact_relations
    mesh, camera, parts = (tmp_path / name for name in ("mesh.npz", "cam.npz", "parts.npz"))
    np.savez(mesh, verts=np.array([[[0., 0., 1.], [0., 2., 1.]]] * 2), contact=np.zeros((2, 2)))
    np.savez(camera, ids=[4], extrinsics=[np.column_stack([np.eye(3), np.zeros(3)])])
    np.savez(parts, parts=np.array([0, 1], np.int64))
    assets = dict(contact_mesh=mesh, camera=camera, camera_id=4, contact_vertex_parts=parts)
    result, meta = build_contact_relations(assets, (1, 0), torch.ones(1, 2, dtype=torch.bool))
    assert meta["valid"] and meta["reason"] == "valid_zero_contact"
    assert result.valid.shape == (1, 0)
    assets.pop("contact_vertex_parts")
    _, missing = build_contact_relations(assets, (1, 0), torch.ones(1, 2, dtype=torch.bool))
    assert not missing["valid"]


def test_bare_cache_from_another_sample_cannot_be_mixed_with_locked_source(tmp_path):
    from src.pose_control.qwen21_pro.preparation import prepare_task_condition
    record, cached, assets = fixture_assets(tmp_path)
    bare = {key: copy.deepcopy(value) for key, value in cached.items()}
    bare["reference_latents"][:, :16] = 9
    with pytest.raises(ValueError, match="bound|verified|binding"):
        prepare_task_condition(record, bare, project_root=tmp_path, config=ProAdapterConfig(), assets=assets)


def test_typed_pack_carries_exact_verified_legacy_condition_attestation(tmp_path):
    from src.pose_control.qwen21_pro.preparation import prepare_task_condition
    record, cached, assets = fixture_assets(tmp_path)
    result = prepare_task_condition(record, cached, project_root=tmp_path, config=ProAdapterConfig(), assets=assets)
    keys = ("sample_id", "condition_fingerprint", "payload_sha256", "prepared_source_sha256", "source_latent_sha256")
    assert result.provenance["cache_binding"] == {key: cached.attestation[key] for key in keys}
