from __future__ import annotations

from dataclasses import fields, replace

import pytest
import torch

from src.pose_control.v6.detail import (
    DetailReferenceBatch,
    FaceHandDetailCondition,
    FaceHandDetailPreparer,
)
from src.pose_control.v6.token_encoders import PersonTokenBinder


def make_inputs() -> tuple[FaceHandDetailCondition, DetailReferenceBatch]:
    batch_size = 2
    face = torch.rand(batch_size, 2, 68, 3)
    hands = torch.rand(batch_size, 2, 2, 21, 3)
    boxes = torch.tensor(
        [[0.05, 0.05, 0.95, 0.95], [0.05, 0.05, 0.95, 0.95], [0.05, 0.05, 0.95, 0.95]]
    ).expand(batch_size, 2, -1, -1).clone()
    valid = torch.tensor(
        [
            [[True, True, True], [True, True, True]],
            [[True, False, True], [False, False, False]],
        ]
    )
    condition = FaceHandDetailCondition(
        face_keypoints=face,
        hand_keypoints=hands,
        smplx_detail=torch.randn(batch_size, 2, 103),
        source_boxes=boxes.clone(),
        target_boxes=boxes.clone(),
        region_valid=valid,
        source_indices=torch.tensor([[0, 1], [3, -1]]),
    )
    reference_valid = torch.ones(batch_size, 2, 3, 2, dtype=torch.bool)
    reference_valid[0, 0, 0] = False
    references = DetailReferenceBatch(
        images=torch.randn(batch_size, 2, 3, 2, 3, 224, 224),
        reference_valid=reference_valid,
    )
    return condition, references


def build_preparer(token_dim: int = 12) -> FaceHandDetailPreparer:
    return FaceHandDetailPreparer(
        token_dim=token_dim,
        hidden_dim=24,
    ).eval()


def prepare_detail(
    preparer: FaceHandDetailPreparer,
    condition: FaceHandDetailCondition,
    references: DetailReferenceBatch,
    latent_spatial_size: tuple[int, int],
    token_spatial_size: tuple[int, int],
    source_latents: torch.Tensor | None = None,
):
    person_binder = PersonTokenBinder(token_dim=12)
    person_binding = person_binder.binding_for(
        condition.source_indices, condition.person_valid
    )
    return preparer(
        condition,
        references,
        latent_spatial_size,
        token_spatial_size,
        source_latents,
        person_binding=person_binding,
    )


def test_preparer_emits_exact_shapes_masks_and_zero_optional_canvas() -> None:
    condition, references = make_inputs()
    prepared = prepare_detail(
        build_preparer(), condition, references, (6, 5), (3, 4)
    )
    assert prepared.detail_condition.shape == (2, 144, 6, 5)
    assert prepared.detail_tokens.shape == (2, 48, 12)
    assert prepared.detail_token_mask.shape == (2, 48)
    assert prepared.region_masks.shape == (2, 2, 3, 3, 4)
    assert prepared.region_valid.shape == (2, 2, 3)
    assert prepared.detail_valid.tolist() == [True, True]
    assert torch.count_nonzero(prepared.detail_condition[:, :16]) == 0
    assert torch.count_nonzero(prepared.detail_condition[:, 16:]) > 0
    assert all(
        getattr(prepared, item.name).device == condition.device
        for item in fields(prepared)
    )
    assert prepared.region_masks.min() >= 0
    assert prepared.region_masks.max() <= 1

    expected_mask = condition.region_valid[..., None].expand(-1, -1, -1, 8).reshape(2, 48)
    assert torch.equal(prepared.detail_token_mask, expected_mask)
    invalid_tokens = prepared.detail_tokens[~prepared.detail_token_mask]
    assert torch.count_nonzero(invalid_tokens) == 0
    invalid_masks = prepared.region_masks[~condition.region_valid]
    assert torch.count_nonzero(invalid_masks) == 0


def test_missing_references_use_null_appearance_but_target_invalid_tokens_stay_zero() -> None:
    condition, references = make_inputs()
    preparer = build_preparer()
    prepared = prepare_detail(preparer, condition, references, (4, 4), (4, 4))
    face_tokens = prepared.detail_tokens.reshape(2, 2, 3, 8, 12)[0, 0, 0]
    assert torch.count_nonzero(face_tokens) > 0
    invalid = prepared.detail_tokens.reshape(2, 2, 3, 8, 12)[1, 0, 1]
    assert torch.count_nonzero(invalid) == 0


def test_source_canvas_is_gated_and_overlap_does_not_additively_amplify() -> None:
    condition, references = make_inputs()
    index = torch.tensor([0])
    condition = condition.index_select(index)
    references = references.index_select(index)
    source_boxes = torch.tensor(
        [[
            [[0.05, 0.05, 0.40, 0.40], [0.60, 0.05, 0.95, 0.40], [0.05, 0.60, 0.40, 0.95]],
            [[0.60, 0.60, 0.95, 0.95], [0.30, 0.05, 0.65, 0.40], [0.30, 0.60, 0.65, 0.95]],
        ]],
        requires_grad=True,
    )
    target_boxes = torch.tensor(
        [[
            [[0.10, 0.10, 0.80, 0.80], [0.20, 0.15, 0.90, 0.85], [0.15, 0.20, 0.85, 0.90]],
            [[0.10, 0.15, 0.80, 0.85], [0.20, 0.10, 0.90, 0.80], [0.15, 0.15, 0.85, 0.85]],
        ]],
        requires_grad=True,
    )
    condition = replace(
        condition, source_boxes=source_boxes, target_boxes=target_boxes
    )
    source_latents = torch.linspace(0, 1, 16 * 8 * 8).reshape(
        1, 16, 8, 8
    ).requires_grad_()
    prepared = prepare_detail(
        build_preparer(), condition, references, (8, 8), (4, 4), source_latents
    )
    canvas = prepared.detail_condition[:, :16]
    assert torch.count_nonzero(canvas) > 0
    assert canvas.min() >= 0
    assert canvas.max() <= source_latents.max() + 1e-6

    gradients = torch.autograd.grad(
        canvas.square().mean(), (source_latents, source_boxes, target_boxes)
    )
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(torch.count_nonzero(gradient) > 0 for gradient in gradients)


def test_prepared_to_index_select_and_cfg_expansion_preserve_order() -> None:
    condition, references = make_inputs()
    prepared = prepare_detail(
        build_preparer(), condition, references, (4, 5), (2, 3)
    )
    converted = prepared.to(dtype=torch.float64)
    assert converted.detail_condition.dtype == torch.float64
    assert converted.detail_tokens.dtype == torch.float64
    assert converted.detail_token_mask.dtype == torch.bool
    assert converted.region_valid.dtype == torch.bool
    moved = converted.to(device=torch.device("cpu"))
    assert all(
        getattr(moved, item.name).device.type == "cpu" for item in fields(moved)
    )
    assert moved.detail_token_mask.dtype == torch.bool
    assert moved.region_valid.dtype == torch.bool
    assert moved.detail_valid.dtype == torch.bool
    selected = converted.index_select(torch.tensor([1, 0]))
    torch.testing.assert_close(selected.detail_condition[0], converted.detail_condition[1])

    expanded = converted.expand_to_batch(4)
    assert expanded.detail_condition.shape[0] == 4
    torch.testing.assert_close(expanded.detail_condition[:2], converted.detail_condition)
    torch.testing.assert_close(expanded.detail_condition[2:], converted.detail_condition)
    assert expanded.detail_valid.tolist() == [True, True, True, True]


def test_empty_sample_is_exactly_zero_and_marked_invalid() -> None:
    condition, references = make_inputs()
    condition.region_valid[1] = False
    prepared = prepare_detail(
        build_preparer(), condition, references, (4, 4), (2, 2)
    )
    assert not prepared.detail_valid[1]
    assert torch.count_nonzero(prepared.detail_condition[1]) == 0
    assert torch.count_nonzero(prepared.detail_tokens[1]) == 0
    assert torch.count_nonzero(prepared.region_masks[1]) == 0


def test_zero_sized_prepared_batches_are_rejected_before_expansion() -> None:
    condition, references = make_inputs()
    prepared = prepare_detail(
        build_preparer(), condition, references, (4, 4), (2, 2)
    )
    empty = prepared.index_select(torch.empty(0, dtype=torch.long))
    with pytest.raises(ValueError, match="non-empty"):
        empty.validate()
    with pytest.raises(ValueError, match="non-empty"):
        empty.expand_to_batch(2)
