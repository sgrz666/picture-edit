from __future__ import annotations

from dataclasses import fields, replace

import pytest
import torch
import torch.nn as nn

from src.pose_control.v6.detail import (
    DetailReferenceBatch,
    DetailRegion,
    DetailAppearanceEncoder,
    DetailSpatialEncoder,
    FaceHandDetailPreparer,
    DetailTokenBinder,
    FaceHandDetailCondition,
    canonicalize_left_hand_keypoints,
    canonicalize_left_hand_pose,
    restore_left_hand_pose,
    restore_left_hand_keypoints,
)
from src.pose_control.v6.token_encoders import PersonTokenBinder


def make_condition(batch_size: int = 2) -> FaceHandDetailCondition:
    face = torch.rand(batch_size, 2, 68, 3)
    hands = torch.rand(batch_size, 2, 2, 21, 3)
    boxes = torch.tensor(
        [[0.10, 0.10, 0.35, 0.40], [0.05, 0.45, 0.25, 0.80], [0.70, 0.45, 0.90, 0.80]]
    ).expand(batch_size, 2, -1, -1).clone()
    valid = torch.ones(batch_size, 2, 3, dtype=torch.bool)
    if batch_size > 1:
        valid[1, 1, 1:] = False
    return FaceHandDetailCondition(
        face_keypoints=face,
        hand_keypoints=hands,
        smplx_detail=torch.randn(batch_size, 2, 103),
        source_boxes=boxes.clone(),
        target_boxes=boxes.clone(),
        region_valid=valid,
        source_indices=torch.tensor([[2, 5], [9, 10]])[:batch_size],
    )


def make_references(batch_size: int = 2, reference_count: int = 2) -> DetailReferenceBatch:
    return DetailReferenceBatch(
        images=torch.randn(batch_size, 2, 3, reference_count, 3, 224, 224),
        reference_valid=torch.ones(batch_size, 2, 3, reference_count, dtype=torch.bool),
    )


def test_detail_condition_contract_to_and_index_select_preserve_discrete_dtypes() -> None:
    condition = make_condition().validate()
    assert DetailRegion.FACE == 0
    assert DetailRegion.LEFT_HAND == 1
    assert DetailRegion.RIGHT_HAND == 2

    converted = condition.to(dtype=torch.float64)
    assert converted.face_keypoints.dtype == torch.float64
    assert converted.smplx_detail.dtype == torch.float64
    assert converted.region_valid.dtype == torch.bool
    assert converted.source_indices.dtype == torch.long
    moved = converted.to(device=torch.device("cpu"))
    assert all(getattr(moved, item.name).device.type == "cpu" for item in fields(moved))
    assert moved.region_valid.dtype == torch.bool
    assert moved.source_indices.dtype == torch.long

    selected = converted.index_select(torch.tensor([1, 0, 1]))
    assert selected.face_keypoints.shape == (3, 2, 68, 3)
    assert selected.source_indices.tolist() == [[9, 10], [2, 5], [9, 10]]


def test_reference_contract_to_and_index_select() -> None:
    references = make_references().validate()
    converted = references.to(dtype=torch.float64)
    assert converted.images.dtype == torch.float64
    assert converted.reference_valid.dtype == torch.bool
    moved = converted.to(device=torch.device("cpu"))
    assert all(getattr(moved, item.name).device.type == "cpu" for item in fields(moved))
    assert moved.reference_valid.dtype == torch.bool
    selected = converted.index_select(torch.tensor([1, 0]))
    torch.testing.assert_close(selected.images[0], converted.images[1])


@pytest.mark.parametrize(
    "changed, message",
    [
        (lambda value: replace(value, face_keypoints=value.face_keypoints[..., :2]), "face_keypoints"),
        (lambda value: replace(value, hand_keypoints=value.hand_keypoints.double()), "floating tensors"),
        (lambda value: replace(value, smplx_detail=value.smplx_detail.fill_(float("nan"))), "finite"),
        (lambda value: replace(value, target_boxes=value.target_boxes.add(1.0)), "normalized"),
        (
            lambda value: replace(
                value,
                source_boxes=value.source_boxes.clone().index_put_(
                    (torch.tensor([0]), torch.tensor([0]), torch.tensor([0]), torch.tensor([2])),
                    torch.tensor([0.0]),
                ),
            ),
            "xyxy",
        ),
        (lambda value: replace(value, face_keypoints=value.face_keypoints.add(2.0)), "normalized"),
        (lambda value: replace(value, source_indices=value.source_indices.float()), "int64"),
    ],
)
def test_detail_condition_rejects_invalid_inputs(changed, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        changed(make_condition()).validate()


def test_reference_contract_rejects_invalid_count_shape_and_mask_dtype() -> None:
    with pytest.raises(ValueError, match="between 1 and 3"):
        make_references(reference_count=4).validate()
    with pytest.raises(ValueError, match="between 1 and 3"):
        make_references(reference_count=0).validate()
    references = make_references()
    with pytest.raises(ValueError, match="reference_valid"):
        replace(references, reference_valid=references.reference_valid.float()).validate()


def test_left_hand_canonicalization_has_explicit_inverse_and_is_an_involution() -> None:
    keypoints = torch.rand(2, 21, 3)
    canonical = canonicalize_left_hand_keypoints(keypoints)
    restored = restore_left_hand_keypoints(canonical)
    torch.testing.assert_close(restored, keypoints)
    torch.testing.assert_close(
        canonicalize_left_hand_keypoints(canonical), keypoints
    )
    torch.testing.assert_close(canonical[..., 0], 1.0 - keypoints[..., 0])
    torch.testing.assert_close(canonical[..., 1:], keypoints[..., 1:])


def test_person_binding_validates_source_range_and_uniqueness_without_state_key_changes() -> None:
    binder = PersonTokenBinder(token_dim=4)
    assert set(binder.state_dict()) == {
        "slot_embedding.weight",
        "source_embedding.weight",
    }
    valid = torch.tensor([[True, True], [True, False]])
    binding = binder.binding_for(torch.tensor([[2, 5], [15, -99]]), valid)
    assert binding.shape == (2, 2, 4)
    with pytest.raises(ValueError, match=r"\[0, 15\]"):
        binder.binding_for(torch.tensor([[2, 16]]), torch.ones(1, 2, dtype=torch.bool))
    oversized_binder = PersonTokenBinder(token_dim=4, max_sources=32)
    with pytest.raises(ValueError, match=r"\[0, 15\]"):
        oversized_binder.binding_for(
            torch.tensor([[2, 16]]), torch.ones(1, 2, dtype=torch.bool)
        )
    with pytest.raises(ValueError, match="unique"):
        binder.binding_for(torch.tensor([[3, 3]]), torch.ones(1, 2, dtype=torch.bool))


def test_detail_token_binding_order_and_invalid_region_zeroing() -> None:
    person_binder = PersonTokenBinder(token_dim=4)
    binder = DetailTokenBinder(token_dim=4)
    with torch.no_grad():
        person_binder.slot_embedding.weight.zero_()
        person_binder.slot_embedding.weight[:, 0] = torch.tensor([100.0, 200.0])
        person_binder.source_embedding.weight.zero_()
        person_binder.source_embedding.weight[:, 0] = torch.arange(16) * 10.0
        binder.region_embedding.weight.zero_()
        binder.region_embedding.weight[:, 0] = torch.tensor([1.0, 2.0, 3.0])
        binder.side_embedding.weight.zero_()
        binder.side_embedding.weight[:, 0] = torch.tensor([0.0, 10.0, 20.0])

    region_tokens = torch.zeros(1, 2, 3, 8, 4)
    region_valid = torch.tensor([[[True, True, True], [True, False, True]]])
    person_binding = person_binder.binding_for(
        torch.tensor([[2, 5]]), region_valid.any(dim=-1)
    )
    tokens, mask = binder(region_tokens, region_valid, person_binding)
    assert tokens.shape == (1, 48, 4)
    assert mask.shape == (1, 48)
    expected_groups = torch.tensor([121.0, 132.0, 143.0, 251.0, 0.0, 273.0])
    torch.testing.assert_close(tokens[0, ::8, 0], expected_groups)
    assert mask.reshape(1, 2, 3, 8)[0, 1, 1].sum() == 0
    assert torch.count_nonzero(tokens.reshape(1, 2, 3, 8, 4)[0, 1, 1]) == 0


def test_axis_angle_left_hand_canonicalization_is_an_involution() -> None:
    pose = torch.randn(2, 45)
    canonical = canonicalize_left_hand_pose(pose)
    torch.testing.assert_close(restore_left_hand_pose(canonical), pose)
    torch.testing.assert_close(canonicalize_left_hand_pose(canonical), pose)
    triplets = pose.reshape(2, 15, 3)
    canonical_triplets = canonical.reshape(2, 15, 3)
    torch.testing.assert_close(canonical_triplets[..., 0], triplets[..., 0])
    torch.testing.assert_close(canonical_triplets[..., 1:], -triplets[..., 1:])


def test_mirrored_left_and_right_hands_share_pre_binding_region_features() -> None:
    condition = make_condition(batch_size=1)
    right_keypoints = torch.rand(2, 21, 3)
    right_pose = torch.randn(2, 45)
    condition.hand_keypoints[:, :, 1] = right_keypoints
    condition.hand_keypoints[:, :, 0] = canonicalize_left_hand_keypoints(
        right_keypoints
    )
    condition.smplx_detail[:, :, 58:103] = right_pose
    condition.smplx_detail[:, :, 13:58] = canonicalize_left_hand_pose(right_pose)

    features = DetailSpatialEncoder(hidden_dim=16).region_features(condition)
    torch.testing.assert_close(features[:, :, 1], features[:, :, 2])


def test_constant_reference_images_have_finite_backward_gradients() -> None:
    images = torch.ones(1, 2, 3, 1, 3, 224, 224, requires_grad=True)
    valid = torch.ones(1, 2, 3, 1, dtype=torch.bool)
    valid[:, 1, 2] = False
    references = DetailReferenceBatch(images=images, reference_valid=valid)
    encoder = DetailAppearanceEncoder(token_dim=4, hidden_dim=8)
    encoder(references).square().sum().backward()

    assert images.grad is not None and torch.isfinite(images.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in encoder.parameters()
    )


def test_masked_nan_reference_features_are_zeroed_before_projection() -> None:
    references = make_references(batch_size=1, reference_count=2)
    references.reference_valid[:, 0, 0] = False
    features = torch.randn(1, 2, 3, 2, 6)
    features[:, 0, 0] = float("nan")
    features.requires_grad_()
    encoder = DetailAppearanceEncoder(token_dim=4, hidden_dim=8)
    output = encoder(references, reference_features=features)
    output.square().sum().backward()

    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in encoder.parameters()
    )
    invalid = torch.randn(1, 2, 3, 2, 6)
    invalid[0, 0, 1, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        encoder(references, reference_features=invalid)


def test_common_owner_registers_person_binding_weights_only_once() -> None:
    class Owner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.person_binder = PersonTokenBinder(token_dim=4)
            self.detail_preparer = FaceHandDetailPreparer(
                token_dim=4, hidden_dim=8
            )

    owner = Owner()
    keys = tuple(owner.state_dict())
    assert keys.count("person_binder.slot_embedding.weight") == 1
    assert keys.count("person_binder.source_embedding.weight") == 1
    assert not any("detail_preparer" in key and "person_binder" in key for key in keys)
