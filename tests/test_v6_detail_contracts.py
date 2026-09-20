from __future__ import annotations

from dataclasses import fields, replace

import pytest
import torch

from src.pose_control.v6.detail import (
    DetailReferenceBatch,
    DetailRegion,
    DetailTokenBinder,
    FaceHandDetailCondition,
    canonicalize_left_hand_keypoints,
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
    binder = DetailTokenBinder(token_dim=4, person_binder=person_binder)
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
    tokens, mask = binder(region_tokens, region_valid, torch.tensor([[2, 5]]))
    assert tokens.shape == (1, 48, 4)
    assert mask.shape == (1, 48)
    expected_groups = torch.tensor([121.0, 132.0, 143.0, 251.0, 0.0, 273.0])
    torch.testing.assert_close(tokens[0, ::8, 0], expected_groups)
    assert mask.reshape(1, 2, 3, 8)[0, 1, 1].sum() == 0
    assert torch.count_nonzero(tokens.reshape(1, 2, 3, 8, 4)[0, 1, 1]) == 0
