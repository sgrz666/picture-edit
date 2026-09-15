from __future__ import annotations

import copy
import importlib
import importlib.util

import pytest
import torch


def _api():
    conditions = importlib.import_module("src.pose_control.v6.conditions")
    injector_module = importlib.import_module("src.pose_control.v6.condition_injector")
    return (
        conditions.ContactRelationBatch,
        conditions.TaskType,
        injector_module.SMPLXConditionInjector,
    )


def _person(batch_size: int = 2, image_size: int = 32, *, offset: float = 0.0) -> dict:
    mask = torch.zeros(batch_size, 1, image_size, image_size)
    mask[..., 4:-4, 5:-5] = 1
    normal = torch.zeros(batch_size, 3, image_size, image_size)
    normal[:, 2] = mask[:, 0]
    pose = torch.zeros(batch_size, 25, image_size, image_size)
    pose[:, 0, image_size // 2, image_size // 2] = 1
    part = torch.zeros(batch_size, 14, image_size, image_size)
    part[:, 1] = mask[:, 0]
    global_params = torch.randn(batch_size, 26, generator=torch.Generator().manual_seed(9))
    global_params = global_params + offset
    return {
        "normal": normal,
        "pose_heatmap": pose,
        "part_onehot": part,
        "smplx_global": global_params,
        "human_mask": mask,
    }


def _relations(batch_size: int = 2):
    ContactRelationBatch, _, _ = _api()
    valid = torch.zeros(batch_size, 8, dtype=torch.bool)
    valid[:, :2] = True
    return ContactRelationBatch(
        src_person=torch.zeros(batch_size, 8, dtype=torch.long),
        src_part=torch.ones(batch_size, 8, dtype=torch.long) * 4,
        dst_person=torch.ones(batch_size, 8, dtype=torch.long),
        dst_part=torch.ones(batch_size, 8, dtype=torch.long) * 7,
        contact_type=torch.ones(batch_size, 8, dtype=torch.long),
        distance=torch.linspace(0, 1, 8).reshape(1, 8, 1).expand(batch_size, -1, -1),
        valid_mask=valid,
    )


def test_condition_injector_module_is_available() -> None:
    assert importlib.util.find_spec("src.pose_control.v6.condition_injector") is not None


def test_single_person_bundle_has_stable_shapes() -> None:
    _, TaskType, Injector = _api()
    person = _person()
    injector = Injector().eval()
    bundle = injector(
        normal_a=person["normal"],
        pose_heatmap_a=person["pose_heatmap"],
        part_onehot_a=person["part_onehot"],
        smplx_global_a=person["smplx_global"],
        human_mask_a=person["human_mask"],
        task_id=torch.full((2,), int(TaskType.SINGLE)),
    )

    assert bundle.person_a_spatial.shape == (2, 256, 4, 4)
    assert bundle.person_a_mask.shape == (2, 1, 4, 4)
    assert bundle.person_a_global_tokens.shape == (2, 4, 256)
    assert bundle.task_token.shape == (2, 1, 256)
    assert bundle.person_b_spatial is None
    assert bundle.person_b_global_tokens is None
    assert bundle.relative_tokens is None
    assert bundle.contact_spatial is None
    assert bundle.contact_tokens is None
    assert bundle.contact_mask is None
    assert bundle.person_count.tolist() == [1, 1]
    assert bundle.person_valid.tolist() == [[True, False], [True, False]]


def test_mixed_dual_batch_masks_invalid_person_and_interaction_conditions() -> None:
    _, TaskType, Injector = _api()
    a = _person(offset=0.0)
    b = _person(offset=1.0)
    b_valid = torch.tensor([True, False])
    relations = _relations()
    relations.valid_mask[1] = False
    bundle = Injector().eval()(
        normal_a=a["normal"],
        pose_heatmap_a=a["pose_heatmap"],
        part_onehot_a=a["part_onehot"],
        smplx_global_a=a["smplx_global"],
        human_mask_a=a["human_mask"],
        normal_b=b["normal"],
        pose_heatmap_b=b["pose_heatmap"],
        part_onehot_b=b["part_onehot"],
        smplx_global_b=b["smplx_global"],
        human_mask_b=b["human_mask"],
        person_b_valid=b_valid,
        relative_geometry=torch.randn(2, 11),
        contact_raster=torch.ones(2, 2, 32, 32),
        contact_relations=relations,
        task_id=torch.tensor([int(TaskType.HANDSHAKE), int(TaskType.SINGLE)]),
    )

    assert bundle.person_count.tolist() == [2, 1]
    assert bundle.person_b_spatial.shape == (2, 256, 4, 4)
    assert torch.count_nonzero(bundle.person_b_spatial[1]) == 0
    assert torch.count_nonzero(bundle.person_b_global_tokens[1]) == 0
    assert torch.count_nonzero(bundle.relative_tokens[1]) == 0
    assert torch.count_nonzero(bundle.contact_spatial[1]) == 0
    assert bundle.contact_mask.tolist() == [
        [True, True, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False],
    ]
    assert torch.count_nonzero(bundle.contact_tokens[~bundle.contact_mask]) == 0


def test_human_mask_blocks_background_changes() -> None:
    _, TaskType, Injector = _api()
    person = _person(batch_size=1)
    changed = copy.deepcopy(person)
    outside = 1 - changed["human_mask"]
    changed["normal"] = changed["normal"] + outside
    injector = Injector().eval()

    def encode(values: dict):
        return injector(
            normal_a=values["normal"],
            pose_heatmap_a=values["pose_heatmap"],
            part_onehot_a=values["part_onehot"],
            smplx_global_a=values["smplx_global"],
            human_mask_a=values["human_mask"],
            task_id=torch.tensor([int(TaskType.SINGLE)]),
        ).person_a_spatial

    torch.testing.assert_close(encode(person), encode(changed), atol=0, rtol=0)


def test_depth_is_ignored_when_disabled_and_jointly_normalized_when_enabled() -> None:
    _, TaskType, Injector = _api()
    a = _person(batch_size=1)
    b = _person(batch_size=1, offset=1.0)
    depth_a = a["human_mask"].clone()
    depth_b = b["human_mask"].clone() * 3
    common = dict(
        normal_a=a["normal"], pose_heatmap_a=a["pose_heatmap"],
        part_onehot_a=a["part_onehot"], smplx_global_a=a["smplx_global"],
        human_mask_a=a["human_mask"], normal_b=b["normal"],
        pose_heatmap_b=b["pose_heatmap"], part_onehot_b=b["part_onehot"],
        smplx_global_b=b["smplx_global"], human_mask_b=b["human_mask"],
        task_id=torch.tensor([int(TaskType.DUAL)]),
    )

    disabled = Injector(use_depth=False).eval()
    without_depth = disabled(**common)
    with_depth = disabled(**common, depth_a=depth_a, depth_b=depth_b)
    torch.testing.assert_close(
        without_depth.person_a_spatial, with_depth.person_a_spatial, atol=0, rtol=0
    )

    enabled = Injector(use_depth=True).eval()
    encoded = enabled(**common, depth_a=depth_a, depth_b=depth_b)
    assert not torch.allclose(encoded.person_a_spatial, encoded.person_b_spatial)


def test_joint_depth_normalization_excludes_a_missing_modality() -> None:
    geometry = importlib.import_module("src.pose_control.v6.geometry_encoder")
    mask_a = torch.ones(1, 1, 2, 2)
    mask_b = torch.ones(1, 1, 2, 2)
    depth_b = torch.tensor([[[[2.0, 4.0], [6.0, 8.0]]]])

    normalized_a, normalized_b = geometry.normalize_scene_depth(
        None,
        depth_b,
        mask_a,
        mask_b,
        torch.tensor([True]),
    )

    assert normalized_a is None
    torch.testing.assert_close(
        normalized_b,
        torch.tensor([[[[-1.0, -1.0 / 3.0], [1.0 / 3.0, 1.0]]]]),
    )


def test_invalid_contracts_fail_loudly() -> None:
    _, TaskType, Injector = _api()
    person = _person(batch_size=1)
    injector = Injector()
    base = dict(
        normal_a=person["normal"],
        pose_heatmap_a=person["pose_heatmap"],
        part_onehot_a=person["part_onehot"],
        smplx_global_a=person["smplx_global"],
        task_id=torch.tensor([int(TaskType.SINGLE)]),
    )
    with pytest.raises(ValueError, match="all be provided"):
        injector(**base, normal_b=person["normal"])

    invalid_part = person["part_onehot"].clone()
    invalid_part[:, 0] = 1
    with pytest.raises(ValueError, match="one-hot"):
        injector(**{**base, "part_onehot_a": invalid_part})

    with pytest.raises(ValueError, match="divisible by 8"):
        injector(
            **{
                **base,
                "normal_a": person["normal"][..., :-1, :-1],
                "pose_heatmap_a": person["pose_heatmap"][..., :-1, :-1],
                "part_onehot_a": person["part_onehot"][..., :-1, :-1],
            }
        )

    with pytest.raises(ValueError, match="fixed public contract"):
        Injector(token_dim=128)


def test_contact_padding_and_global_task_sensitivity() -> None:
    _, TaskType, Injector = _api()
    a = _person(batch_size=1)
    b = _person(batch_size=1, offset=1.0)
    relations = _relations(batch_size=1)
    injector = Injector().eval()
    common = dict(
        normal_a=a["normal"], pose_heatmap_a=a["pose_heatmap"],
        part_onehot_a=a["part_onehot"], human_mask_a=a["human_mask"],
        normal_b=b["normal"], pose_heatmap_b=b["pose_heatmap"],
        part_onehot_b=b["part_onehot"], smplx_global_b=b["smplx_global"],
        human_mask_b=b["human_mask"], contact_relations=relations,
    )
    first = injector(
        **common,
        smplx_global_a=a["smplx_global"],
        task_id=torch.tensor([int(TaskType.HANDSHAKE)]),
    )
    second = injector(
        **common,
        smplx_global_a=a["smplx_global"] + 2,
        task_id=torch.tensor([int(TaskType.HUG)]),
    )
    assert not torch.allclose(first.person_a_global_tokens, second.person_a_global_tokens)
    assert not torch.allclose(first.task_token, second.task_token)
    assert first.contact_mask.tolist() == [[True, True, False, False, False, False, False, False]]
    assert torch.count_nonzero(first.contact_tokens[:, 2:]) == 0


def test_contact_padding_may_use_negative_sentinel_indices() -> None:
    _, TaskType, Injector = _api()
    a = _person(batch_size=1)
    b = _person(batch_size=1, offset=1.0)
    relations = _relations(batch_size=1)
    invalid = ~relations.valid_mask
    for name in ("src_person", "src_part", "dst_person", "dst_part", "contact_type"):
        getattr(relations, name)[invalid] = -1

    bundle = Injector().eval()(
        normal_a=a["normal"],
        pose_heatmap_a=a["pose_heatmap"],
        part_onehot_a=a["part_onehot"],
        smplx_global_a=a["smplx_global"],
        human_mask_a=a["human_mask"],
        normal_b=b["normal"],
        pose_heatmap_b=b["pose_heatmap"],
        part_onehot_b=b["part_onehot"],
        smplx_global_b=b["smplx_global"],
        human_mask_b=b["human_mask"],
        contact_relations=relations,
        task_id=torch.tensor([int(TaskType.DUAL)]),
    )

    assert bundle.contact_mask.tolist() == [
        [True, True, False, False, False, False, False, False]
    ]
    assert torch.count_nonzero(bundle.contact_tokens[:, 2:]) == 0


def test_champ_backend_is_opt_in_and_runs_without_weights_or_video_dependencies() -> None:
    _, TaskType, Injector = _api()
    person = _person(batch_size=1)
    injector = Injector(
        use_depth=True,
        normal_backend="champ",
        depth_backend="champ",
    ).eval()
    bundle = injector(
        normal_a=person["normal"],
        pose_heatmap_a=person["pose_heatmap"],
        part_onehot_a=person["part_onehot"],
        smplx_global_a=person["smplx_global"],
        human_mask_a=person["human_mask"],
        depth_a=person["human_mask"].clone(),
        task_id=torch.tensor([int(TaskType.SINGLE)]),
    )
    assert bundle.person_a_spatial.shape == (1, 256, 4, 4)
    assert injector.spatial_encoder.normal_stem.encoder.__class__.__module__.startswith(
        "third_party.champ_guidance"
    )
