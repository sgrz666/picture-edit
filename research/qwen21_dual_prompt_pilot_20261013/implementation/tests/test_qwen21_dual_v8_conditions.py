"""Public generation contract: fixtures opt into a smaller resolution."""
from dataclasses import FrozenInstanceError, fields, replace

import pytest
import torch

from src.pose_control.qwen21_full.dual.conditions import (
    ControlStrengths, DualGenerationInput, GeometryCondition, SourceCondition,
)


def make_input(*, with_b=True, contact=True, size=32):
    torch.manual_seed(27)
    valid = torch.tensor([[True, with_b]])
    uids = (("alice", "bob" if with_b else None),)
    masks = torch.zeros(1, 2, 1, size, size)
    masks[:, 0, :, 4:-4, 2:size//2+4] = 1
    if with_b:
        masks[:, 1, :, 4:-4, size//2-4:-2] = 1
    owner = masks.clone()
    owner[:, 1] *= 1 - owner[:, 0]
    part = torch.zeros(1, 2, 14, size, size)
    part[:, :, 1:2] = masks
    part_valid = torch.zeros(1, 2, 14, dtype=torch.bool)
    part_valid[:, :, 1] = valid
    positions = torch.zeros(1, 2, 14, 3)
    positions[0, 0, 1] = torch.tensor([0., 0., 2.])
    if with_b:
        positions[0, 1, 1] = torch.tensor([.1, 0., 3.])
    active_contact = with_b and contact
    endpoints = torch.tensor([[[1, 15]]]) if active_contact else torch.empty(1, 0, 2, dtype=torch.long)
    features = torch.randn(1, int(active_contact), 15)
    contact_map = torch.zeros(1, 1, size, size)
    if active_contact:
        contact_map[..., size//2-2:size//2+2, size//2-2:size//2+2] = 1
    geometry = GeometryCondition(
        normal=torch.randn(1, 2, 3, size, size).clamp(-1, 1) * masks,
        depth=masks * torch.tensor([2., 3.])[None, :, None, None, None],
        part=part, pose=torch.rand(1, 2, 25, size, size) * masks,
        amodal_mask=masks, visible_owner=owner,
        global32=torch.randn(1, 2, 32) * valid[..., None],
        relative14=torch.randn(1, 2, 14) if with_b else torch.zeros(1, 2, 14),
        part_positions=positions,
        part_orientations=part_valid[...,None].float()*torch.tensor([0.,1.,0.]),
        part_joints=positions.clone(), part_valid=part_valid,
        part_confidence=part_valid.float(), contact_endpoints=endpoints,
        contact_features=features, contact_valid=torch.ones(1, int(active_contact), dtype=torch.bool),
        contact_map=contact_map, person_valid=valid, person_uids=uids,
        geometry_hash="fixture", allow_test_resolution=True,
    )
    latent_masks = torch.nn.functional.interpolate(masks.flatten(0, 1), size=(4, 4), mode="area").reshape(1, 2, 1, 4, 4)
    source = SourceCondition(
        rgb=torch.rand(1, 3, size, size), masks=masks, person_uids=uids,
        person_latents=torch.randn(1, 2, 64, 4, 4) * valid[..., None, None, None],
        latent_masks=latent_masks,
        dino_patches=torch.randn(1, 2, 6, 1536) * valid[..., None, None],
        dino_valid=valid[..., None].expand(1, 2, 6).clone(),
        allow_test_resolution=True,
    )
    return DualGenerationInput(source, geometry, ("two people",), (19,))


def test_contract_is_frozen_and_never_accepts_ground_truth():
    value = make_input()
    with pytest.raises(FrozenInstanceError):
        value.source.version = "other"
    with pytest.raises(TypeError):
        DualGenerationInput(value.source, value.geometry, value.prompts, value.seeds, target_rgb=value.source.rgb)
    assert {f.name for f in fields(DualGenerationInput)} == {"source", "geometry", "prompts", "seeds"}


def test_to_preserves_discrete_dtypes_and_fixture_opt_in():
    value = make_input().to("cpu", torch.float64)
    assert value.source.rgb.dtype == torch.float64
    assert value.geometry.contact_endpoints.dtype == torch.long
    assert value.geometry.part_valid.dtype == torch.bool
    assert value.source.dino_valid.dtype == torch.bool
    assert value.geometry.geometry_hash == "fixture"
    assert value.validate() is value


def test_production_resolution_requires_512_and_fixture_is_explicit():
    value = make_input()
    with pytest.raises(ValueError, match="512"):
        replace(value.source, allow_test_resolution=False)
    with pytest.raises(ValueError, match="512"):
        replace(value.geometry, allow_test_resolution=False)


@pytest.mark.parametrize("case", ["uid_mismatch", "duplicate_uid", "empty_mask", "invalid_source", "invalid_geometry", "nonfinite", "same_person_edge", "invalid_node_edge"])
def test_rejects_unsafe_conditions(case):
    value = make_input(with_b=case not in {"invalid_source", "invalid_geometry"})
    with pytest.raises(ValueError):
        if case == "uid_mismatch":
            source = replace(value.source, person_uids=(("alice", "carol"),), allow_test_resolution=True)
            replace(value, source=source)
        elif case == "duplicate_uid":
            replace(value.source, person_uids=(("alice", "alice"),), allow_test_resolution=True)
        elif case == "empty_mask":
            replace(value.source, masks=torch.zeros_like(value.source.masks), allow_test_resolution=True)
        elif case == "invalid_source":
            replace(value.source, person_latents=torch.ones_like(value.source.person_latents), allow_test_resolution=True)
        elif case == "invalid_geometry":
            replace(value.geometry, global32=torch.ones_like(value.geometry.global32), allow_test_resolution=True)
        elif case == "nonfinite":
            replace(value.geometry, depth=torch.full_like(value.geometry.depth, float("nan")), allow_test_resolution=True)
        elif case == "same_person_edge":
            replace(value.geometry, contact_endpoints=torch.tensor([[[1, 1]]]), allow_test_resolution=True)
        else:
            replace(value.geometry, contact_endpoints=torch.tensor([[[2, 15]]]), allow_test_resolution=True)


def test_strengths_validate_and_disabled():
    assert ControlStrengths(0, 0, 0).disabled
    assert not ControlStrengths().disabled
    with pytest.raises(ValueError):
        ControlStrengths(geometry=-1)
    with pytest.raises(ValueError):
        ControlStrengths(max_ratio=0)
