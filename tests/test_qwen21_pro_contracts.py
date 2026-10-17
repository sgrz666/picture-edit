"""人物任务条件的布局、有效性和无监督泄漏契约。"""
import importlib
import importlib.util
from dataclasses import replace

import pytest
import torch


def api():
    assert importlib.util.find_spec("src.pose_control.qwen21_pro") is not None, "Pro adapter package is missing"
    return importlib.import_module("src.pose_control.qwen21_pro")


def make_condition(batch=1, people=2):
    module = api()
    b, p, n = batch, people, 4
    masks = torch.ones(b, p, n)
    owner = torch.zeros_like(masks)
    owner[:, 0, :2] = 1
    if p == 2:
        owner[:, 1, 2:] = 1
    else:
        owner[:] = 1
    relations = module.ContactRelations(
        src_person=torch.zeros(b, 0, dtype=torch.long),
        src_part=torch.zeros(b, 0, dtype=torch.long),
        dst_person=torch.zeros(b, 0, dtype=torch.long),
        dst_part=torch.zeros(b, 0, dtype=torch.long),
        relative=torch.zeros(b, 0, 6), confidence=torch.zeros(b, 0),
        valid=torch.zeros(b, 0, dtype=torch.bool),
    )
    return module.TaskCondition(
        target_hw=(2, 2), person_ids=tuple(range(p)),
        spatial={"normal": torch.randn(b, p, 3, 2, 2), "depth": torch.ones(b, p, 1, 2, 2),
                 "mask": torch.ones(b, p, 1, 2, 2), "visibility": torch.ones(b, p, 1, 2, 2),
                 "contact": torch.zeros(b, p, 1, 2, 2), "part": torch.ones(b, p, 1, 2, 2, dtype=torch.long)},
        spatial_valid={k: torch.full((b, p), k != "contact", dtype=torch.bool) for k in ("normal", "depth", "mask", "visibility", "contact", "part")},
        person_mask=masks, owner=owner, face_mask=torch.full_like(masks, .1),
        hand_mask=torch.full((b, p, 2, n), .05), person_valid=torch.ones(b, p, dtype=torch.bool),
        source_tokens=torch.randn(b, p, 3, 64), source_token_valid=torch.ones(b, p, 3, dtype=torch.bool),
        face_identity=torch.randn(b, p, 512), face_identity_valid=torch.ones(b, p, dtype=torch.bool),
        relations=relations, source_fingerprint="source-only-sha256", provenance={"target_geometry": {"codec": "normal"}},
    )


def without_people(pack):
    return replace(pack, person_valid=torch.zeros_like(pack.person_valid), owner=torch.zeros_like(pack.owner),
                   source_token_valid=torch.zeros_like(pack.source_token_valid), face_identity_valid=torch.zeros_like(pack.face_identity_valid),
                   spatial_valid={k: torch.zeros_like(v) for k, v in pack.spatial_valid.items()})


def test_config_dimensions_stages_and_json_signature():
    module = api()
    cfg = module.ProAdapterConfig(hidden_dim=32, width=16, heads=4)
    assert module.ProAdapterConfig.from_dict(cfg.to_dict()) == cfg
    assert cfg.signature() == module.ProAdapterConfig.from_dict(cfg.to_dict()).signature()
    import json
    json.dumps(cfg.to_dict())
    for kwargs in ({"width": 15}, {"stage_indices": (0, 0)}, {"stage_indices": (16,)}, {"source_dim": 0}):
        with pytest.raises(ValueError):
            module.ProAdapterConfig(**kwargs)


def test_condition_roundtrip_device_dtype_and_cfg_repeat():
    pack = make_condition(batch=2)
    pack.validate()
    restored = api().TaskCondition.from_dict(pack.to_dict())
    assert restored.layout_signature() == pack.layout_signature()
    converted = restored.to("cpu", torch.bfloat16)
    assert converted.source_tokens.dtype == torch.bfloat16
    assert converted.spatial["part"].dtype == torch.long
    assert converted.person_valid.dtype == torch.bool
    repeated = pack.repeat_batch(2)
    assert repeated.source_tokens.shape[0] == 4
    assert torch.equal(repeated.source_tokens[:2], pack.source_tokens)
    assert torch.equal(repeated.source_tokens[2:], pack.source_tokens)
    assert repeated.relations.relative.shape == (4, 0, 6)
    repeated.validate()


@pytest.mark.parametrize("mutation", [
    lambda c: replace(c, target_hw=(1, 2)),
    lambda c: replace(c, owner=torch.ones_like(c.owner)),
    lambda c: replace(c, spatial=dict(c.spatial, normal=c.spatial["normal"][:, :, :2])),
    lambda c: replace(c, source_tokens=c.source_tokens[:, :1]),
    lambda c: replace(c, source_token_valid=c.source_token_valid.float()),
    lambda c: replace(c, spatial=dict(c.spatial, part=torch.full_like(c.spatial["part"], 128))),
    lambda c: replace(c, face_identity=c.face_identity * float("nan")),
])
def test_rejects_layout_dtype_owner_range_part_and_nonfinite(mutation):
    with pytest.raises(ValueError):
        mutation(make_condition()).validate()


@pytest.mark.parametrize("key", ["target_rgb", "target_latent", "target_identity", "target_image", "gt_rgb"])
def test_recursive_metadata_cannot_carry_supervision(key):
    pack = make_condition()
    with pytest.raises(ValueError, match="supervision|metadata|provenance"):
        replace(pack, provenance={"nested": [{key: "hidden-path"}]}).validate()


def test_relations_validate_actual_batch_and_valid_endpoints_only():
    pack = make_condition()
    module = api()
    relation = module.ContactRelations(
        src_person=torch.tensor([[0]]), src_part=torch.tensor([[1]]),
        dst_person=torch.tensor([[1]]), dst_part=torch.tensor([[2]]),
        relative=torch.zeros(1, 1, 6), confidence=torch.ones(1, 1), valid=torch.ones(1, 1, dtype=torch.bool),
    )
    replace(pack, relations=relation).validate()
    with pytest.raises(ValueError):
        replace(pack, relations=replace(relation, dst_person=torch.tensor([[2]]))).validate()
    with pytest.raises(ValueError):
        replace(pack, relations=relation.repeat_batch(2)).validate()
    with pytest.raises(ValueError, match="endpoint.*valid person"):
        replace(without_people(pack), relations=relation).validate()
    # Unknown endpoints are legal placeholders only when the relation is invalid.
    replace(pack, relations=replace(relation, src_person=torch.tensor([[-1]]), valid=torch.zeros(1, 1, dtype=torch.bool))).validate()


def test_layout_signature_tracks_binding_and_actual_source_layout():
    pack = make_condition()
    assert replace(pack, person_ids=(9, 8)).layout_signature() != pack.layout_signature()
    assert replace(pack, source_tokens=pack.source_tokens[:, :, :1], source_token_valid=pack.source_token_valid[:, :, :1]).layout_signature() != pack.layout_signature()
    assert replace(pack, source_tokens=pack.source_tokens + 1).layout_signature() == pack.layout_signature()


def test_source_dimension_is_configurable_and_empty_bank_is_legal():
    pack = make_condition()
    cfg = api().ProAdapterConfig(source_dim=1600)
    replace(pack, source_tokens=torch.randn(1, 2, 3, 1600)).validate(cfg)
    replace(pack, source_tokens=pack.source_tokens[:, :, :0], source_token_valid=pack.source_token_valid[:, :, :0]).validate()


@pytest.mark.parametrize("missing", ["all", "normal", "depth", "mask", "visibility", "contact", "part"])
def test_requires_all_six_modalities_with_explicit_invalid_placeholders(missing):
    pack = make_condition()
    spatial = {} if missing == "all" else {k: v for k, v in pack.spatial.items() if k != missing}
    valid = {k: pack.spatial_valid[k] for k in spatial}
    with pytest.raises(ValueError, match="modalit"):
        replace(pack, spatial=spatial, spatial_valid=valid).validate()
    placeholder = replace(pack, spatial=dict(pack.spatial, normal=torch.zeros_like(pack.spatial["normal"])),
                          spatial_valid=dict(pack.spatial_valid, normal=torch.zeros_like(pack.person_valid)))
    placeholder.validate()


@pytest.mark.parametrize("field", ["owner", "face_identity_valid", "source_token_valid", "spatial_valid"])
def test_invalid_person_cannot_enable_source_or_geometry(field):
    pack = without_people(make_condition())
    pack.validate()
    if field == "spatial_valid":
        mutation = {field: dict(pack.spatial_valid, normal=torch.ones_like(pack.person_valid))}
    else:
        value = getattr(pack, field).clone()
        value[:, 0] = True if value.dtype == torch.bool else .1
        mutation = {field: value}
    with pytest.raises(ValueError, match="invalid person"):
        replace(pack, **mutation).validate()


@pytest.mark.parametrize("restriction", ["person_mask", "zero_visibility", "soft_visibility", "invalid_visibility"])
def test_owner_is_bounded_by_amodal_mask_and_valid_visibility(restriction):
    pack = make_condition()
    if restriction == "person_mask":
        pack = replace(pack, person_mask=pack.person_mask * .5)
    elif restriction == "invalid_visibility":
        pack = replace(pack, spatial_valid=dict(pack.spatial_valid, visibility=torch.zeros_like(pack.person_valid)))
    else:
        visibility = pack.spatial["visibility"] * (0 if restriction == "zero_visibility" else .5)
        pack = replace(pack, spatial=dict(pack.spatial, visibility=visibility))
    with pytest.raises(ValueError, match="owner"):
        pack.validate()


def test_soft_visible_owner_and_unavailable_visibility_have_valid_representations():
    pack = make_condition()
    soft = replace(pack, owner=pack.owner * .25, spatial=dict(pack.spatial, visibility=pack.spatial["visibility"] * .5))
    soft.validate()
    assert torch.equal(soft.visible_owner(), soft.owner)
    unknown = replace(pack, owner=torch.zeros_like(pack.owner), spatial_valid=dict(pack.spatial_valid, visibility=torch.zeros_like(pack.person_valid)))
    unknown.validate()
    assert torch.equal(unknown.visible_owner(), torch.zeros_like(pack.owner))


@pytest.mark.parametrize("metadata", [
    {"target_id": "forbidden"}, {"identity_target": "forbidden"}, {"target_arcface": [1, 2]},
    {"geometry": {"target_embedding": [1, 2]}}, {"availability": {"target_appearance": True}},
    {"source": {"target_id": "forbidden"}}, {"nested": [{"appearance_target": "forbidden"}]},
    {"target": {"rgb": "forbidden"}}, {"embedding": [1, 2]}, {"arcface": "forbidden"},
    {"source": {"identity_target": "forbidden"}}, {"geometry": {"latent": "forbidden"}},
])
def test_metadata_rejects_target_aliases_and_appearance_outside_source(metadata):
    with pytest.raises(ValueError, match="supervision|metadata|provenance"):
        replace(make_condition(), provenance=metadata).validate()


def test_metadata_allows_geometry_provenance_and_typed_source_availability():
    metadata = {
        "target_geometry": {"filepath": "/assets/requested.npz", "camera": {"basis": "CV"}},
        "ground_truth_geometry": {"filepath": "/assets/paired_geometry.npz", "normal_basis": "unknown"},
        "source_filepath": "/assets/source.png", "source": {"identity": {"arcface": "source-model", "embedding_hash": "source-hash"}},
        "model_hash": "model-hash", "geometry_codec": "normal-v1", "normal_basis": "camera_cv",
        "person_order": [0, 1], "asset_hashes": {"source": "source-hash", "geometry": "geometry-hash"},
        "availability": {"normal": {"valid": False, "reason": "unknown normal axis"}, "face_identity": False},
    }
    replace(make_condition(), provenance=metadata).validate()


def test_metadata_is_json_data_and_availability_cannot_hide_features():
    for metadata in ({"notes": torch.zeros(512)}, {"availability": {"face_identity": [1., 2.]}}, {"source": {"target_rgb": "forbidden"}}):
        with pytest.raises(ValueError, match="metadata|provenance|supervision"):
            replace(make_condition(), provenance=metadata).validate()


@pytest.mark.parametrize("availability", [[.1, .2], {"normal": [.1, .2]}, {"normal": "embedding-data"},
                                           {"normal": {"valid": False, "reason": "missing", "extra": "embedding-data"}}])
def test_availability_requires_mapping_of_typed_validity_records(availability):
    with pytest.raises(ValueError, match="availability"):
        replace(make_condition(), provenance={"availability": availability}).validate()


def test_availability_accepts_normal_and_face_identity_bool_or_validity_record():
    for availability in ({"normal": {"valid": False, "reason": "missing"}, "face_identity": True},
                         {"normal": False, "face_identity": {"available": True, "reason": "source ArcFace extracted"}}):
        replace(make_condition(), provenance={"availability": availability}).validate()
