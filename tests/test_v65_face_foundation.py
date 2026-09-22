from __future__ import annotations

from dataclasses import fields, replace
import importlib.util
import sys
import types

import pytest
import torch


if importlib.util.find_spec("diffusers") is None:
    diffusers = types.ModuleType("diffusers")
    diffusers_models = types.ModuleType("diffusers.models")

    class _SD3ControlNetModel(torch.nn.Module):
        pass

    diffusers_models.SD3ControlNetModel = _SD3ControlNetModel
    diffusers.models = diffusers_models
    sys.modules["diffusers"] = diffusers
    sys.modules["diffusers.models"] = diffusers_models

from src.pose_control.v6.face import (
    FaceConditioningPreparer,
    FaceContentEncoder,
    FaceFineCondition,
    FaceGeometryTokenizer,
    FaceReferenceFeatures,
    FaceSpatialEncoder,
    PreparedFaceConditioning,
    Resampler,
    dwpose68_to_face72,
    normalize_face_landmarks_to_roi,
    render_face_landmark_heatmaps,
    smplx137_to_face72,
)


def make_condition(batch_size: int = 2) -> FaceFineCondition:
    boxes = torch.tensor(
        [
            [[0.10, 0.15, 0.40, 0.55], [0.55, 0.20, 0.90, 0.70]],
            [[0.05, 0.10, 0.35, 0.50], [0.60, 0.25, 0.95, 0.75]],
        ],
        dtype=torch.float32,
    )[:batch_size]
    landmarks = torch.rand(batch_size, 2, 72, 3)
    landmarks[..., :2] = (
        boxes[..., None, :2]
        + landmarks[..., :2]
        * (boxes[..., None, 2:] - boxes[..., None, :2])
    )
    valid = torch.ones(batch_size, 2, dtype=torch.bool)
    if batch_size > 1:
        valid[1, 1] = False
    return FaceFineCondition(
        landmarks=landmarks,
        jaw_pose=torch.randn(batch_size, 2, 3),
        expression=torch.randn(batch_size, 2, 10),
        source_boxes=boxes.clone(),
        target_boxes=boxes.clone(),
        face_valid=valid,
        source_indices=torch.tensor([[2, 5], [9, 10]], dtype=torch.long)[:batch_size],
    )


def make_references(
    batch_size: int = 2,
    reference_count: int = 2,
) -> FaceReferenceFeatures:
    valid = torch.ones(batch_size, 2, reference_count, dtype=torch.bool)
    if batch_size > 1:
        valid[1, 1] = False
    return FaceReferenceFeatures(
        arcface=torch.randn(batch_size, 2, reference_count, 512),
        dino_patches=torch.randn(batch_size, 2, reference_count, 256, 1536),
        reference_valid=valid,
    )


def make_prepared(batch_size: int = 2) -> PreparedFaceConditioning:
    valid = torch.tensor([[True, True], [True, False]])[:batch_size]
    target_boxes = torch.tensor(
        [
            [[0.10, 0.15, 0.40, 0.55], [0.55, 0.20, 0.90, 0.70]],
            [[0.05, 0.10, 0.35, 0.50], [0.00, 0.00, 0.00, 0.00]],
        ]
    )[:batch_size]
    mask = valid.to(torch.float32)
    return PreparedFaceConditioning(
        spatial_features=torch.randn(batch_size, 2, 512, 16, 16)
        * mask[..., None, None, None],
        identity_tokens=torch.randn(batch_size, 2, 4, 512) * mask[..., None, None],
        texture_tokens=torch.randn(batch_size, 2, 8, 512) * mask[..., None, None],
        texture_delta=torch.randn(batch_size, 2, 4, 512) * mask[..., None, None],
        geometry_tokens=torch.randn(batch_size, 2, 4, 512) * mask[..., None, None],
        face_masks=torch.rand(batch_size, 2, 128, 128) * mask[..., None, None],
        target_boxes=target_boxes,
        face_valid=valid,
    )


def test_face_condition_contract_preserves_discrete_dtypes_and_validates_binding() -> None:
    condition = make_condition().validate()
    converted = condition.to(dtype=torch.float64)
    assert converted.landmarks.dtype == torch.float64
    assert converted.target_boxes.dtype == torch.float64
    assert converted.face_valid.dtype == torch.bool
    assert converted.source_indices.dtype == torch.long
    assert all(getattr(converted, item.name).device.type == "cpu" for item in fields(converted))

    selected = converted.index_select(torch.tensor([1, 0, 1]))
    assert selected.landmarks.shape == (3, 2, 72, 3)
    assert selected.source_indices.tolist() == [[9, 10], [2, 5], [9, 10]]
    expanded = condition.expand_to_batch(4)
    torch.testing.assert_close(expanded.landmarks[:2], condition.landmarks)
    torch.testing.assert_close(expanded.landmarks[2:], condition.landmarks)

    duplicate = replace(condition, source_indices=torch.tensor([[2, 2], [9, 10]]))
    with pytest.raises(ValueError, match="unique"):
        duplicate.validate()
    out_of_range = replace(condition, source_indices=torch.tensor([[2, 16], [9, 10]]))
    with pytest.raises(ValueError, match=r"\[0, 15\]"):
        out_of_range.validate()


def test_reference_contract_enforces_one_to_three_references_and_dtype_device() -> None:
    references = make_references().validate()
    converted = references.to(dtype=torch.float64)
    assert converted.arcface.dtype == torch.float64
    assert converted.dino_patches.dtype == torch.float64
    assert converted.reference_valid.dtype == torch.bool
    selected = converted.index_select(torch.tensor([1, 0]))
    torch.testing.assert_close(selected.arcface[0], converted.arcface[1])

    with pytest.raises(ValueError, match="between 1 and 3"):
        make_references(reference_count=0).validate()
    with pytest.raises(ValueError, match="between 1 and 3"):
        make_references(reference_count=4).validate()
    with pytest.raises(ValueError, match="reference_valid"):
        replace(references, reference_valid=references.reference_valid.float()).validate()


def test_prepared_contract_cfg_expansion_keeps_target_box_order_and_invalid_zero() -> None:
    prepared = make_prepared().validate()
    converted = prepared.to(dtype=torch.float64)
    assert converted.spatial_features.dtype == torch.float64
    assert converted.target_boxes.dtype == torch.float64
    assert converted.face_valid.dtype == torch.bool

    expanded = prepared.expand_to_batch(4)
    assert expanded.face_valid.tolist() == [
        [True, True],
        [True, False],
        [True, True],
        [True, False],
    ]
    torch.testing.assert_close(expanded.target_boxes[:2], prepared.target_boxes)
    torch.testing.assert_close(expanded.target_boxes[2:], prepared.target_boxes)
    assert torch.count_nonzero(expanded.target_boxes[1, 1]) == 0

    bad = replace(prepared, identity_tokens=prepared.identity_tokens.clone())
    bad.identity_tokens[1, 1, 0, 0] = 1
    with pytest.raises(ValueError, match="invalid"):
        bad.validate()


def test_landmark_conversions_and_roi_normalization_are_correct() -> None:
    smplx = torch.arange(137 * 3, dtype=torch.float32).reshape(137, 3)
    torch.testing.assert_close(smplx137_to_face72(smplx), smplx[65:137])

    dwpose = torch.rand(2, 68, 3)
    face = dwpose68_to_face72(dwpose)
    assert face.shape == (2, 72, 3)
    assert torch.count_nonzero(face[:, :4]) == 0
    torch.testing.assert_close(face[:, 4:], dwpose)

    local = torch.rand(2, 72, 3)
    local[..., 2] = torch.rand(2, 72)
    boxes = torch.tensor([[10.0, 20.0, 30.0, 60.0], [5.0, 7.0, 25.0, 47.0]])
    global_points = local.clone()
    global_points[..., :2] = (
        boxes[:, None, :2]
        + local[..., :2] * (boxes[:, None, 2:] - boxes[:, None, :2])
    )
    normalized = normalize_face_landmarks_to_roi(global_points, boxes)
    torch.testing.assert_close(normalized, local)

    scale = 3.25
    translation = torch.tensor([101.0, -37.0])
    transformed_points = global_points.clone()
    transformed_points[..., :2] = global_points[..., :2] * scale + translation
    transformed_boxes = boxes * scale + translation.repeat(2)
    transformed = normalize_face_landmarks_to_roi(transformed_points, transformed_boxes)
    torch.testing.assert_close(transformed, normalized, atol=2e-6, rtol=2e-6)


def test_geometry_tokenizer_zeros_invalid_people_and_their_gradients() -> None:
    landmarks = torch.rand(1, 2, 72, 3, requires_grad=True)
    jaw = torch.randn(1, 2, 3, requires_grad=True)
    expression = torch.randn(1, 2, 10, requires_grad=True)
    valid = torch.tensor([[True, False]])
    tokens = FaceGeometryTokenizer()(landmarks, jaw, expression, valid)
    assert tokens.shape == (1, 2, 4, 512)
    assert torch.count_nonzero(tokens[0, 1]) == 0
    tokens.sum().backward()
    assert torch.count_nonzero(landmarks.grad[0, 1]) == 0
    assert torch.count_nonzero(jaw.grad[0, 1]) == 0
    assert torch.count_nonzero(expression.grad[0, 1]) == 0


def test_resampler_outputs_eight_tokens_and_honors_context_mask() -> None:
    torch.manual_seed(7)
    model = Resampler(depth=1)
    context = torch.randn(2, 12, 1536)
    context_mask = torch.tensor([[True] * 7 + [False] * 5] * 2)
    changed = context.clone()
    changed[:, 7:] = torch.randn_like(changed[:, 7:]) * 1000
    first = model(context, context_mask=context_mask)
    second = model(changed, context_mask=context_mask)
    assert first.shape == (2, 8, 512)
    torch.testing.assert_close(first, second, atol=1e-5, rtol=1e-5)


def test_content_encoder_masked_pooling_zero_init_delta_and_missing_reference_gradients() -> None:
    torch.manual_seed(11)
    encoder = FaceContentEncoder(resampler_depth=1, perceiver_depth=1).eval()
    identity_linears = [
        module
        for module in encoder.identity_projection.modules()
        if isinstance(module, torch.nn.Linear)
    ]
    assert [(layer.in_features, layer.out_features) for layer in identity_linears] == [
        (512, 1024),
        (1024, 4 * 512),
    ]
    assert isinstance(encoder.identity_norm, torch.nn.LayerNorm)
    assert not hasattr(encoder, "identity_query_bias")
    references = make_references(batch_size=1, reference_count=2)
    references.reference_valid[..., 1] = False
    changed = FaceReferenceFeatures(
        arcface=references.arcface.clone(),
        dino_patches=references.dino_patches.clone(),
        reference_valid=references.reference_valid.clone(),
    )
    changed.arcface[..., 1, :] = 10_000
    changed.dino_patches[..., 1, :, :] = -10_000
    identity, texture, delta = encoder(references)
    changed_identity, changed_texture, changed_delta = encoder(changed)
    assert identity.shape == (1, 2, 4, 512)
    assert texture.shape == (1, 2, 8, 512)
    assert delta.shape == (1, 2, 4, 512)
    torch.testing.assert_close(identity, changed_identity)
    torch.testing.assert_close(texture, changed_texture)
    assert torch.count_nonzero(delta) == 0
    assert torch.count_nonzero(changed_delta) == 0
    direct_delta = encoder.face_perceiver(identity, texture)
    assert direct_delta.shape == (1, 2, 4, 512)
    assert torch.count_nonzero(direct_delta) == 0

    arcface = torch.randn(1, 2, 1, 512, requires_grad=True)
    dino = torch.randn(1, 2, 1, 256, 1536, requires_grad=True)
    missing = FaceReferenceFeatures(
        arcface=arcface,
        dino_patches=dino,
        reference_valid=torch.zeros(1, 2, 1, dtype=torch.bool),
    )
    outputs = encoder(missing)
    assert all(torch.count_nonzero(value) == 0 for value in outputs)
    sum(value.sum() for value in outputs).backward()
    assert torch.count_nonzero(arcface.grad) == 0
    assert torch.count_nonzero(dino.grad) == 0


def test_spatial_heatmaps_masks_and_encoder_reach_16_by_16() -> None:
    landmarks = torch.rand(1, 2, 72, 3)
    landmarks[..., 2] = torch.rand(1, 2, 72)
    valid = torch.tensor([[True, False]])
    heatmaps, masks = render_face_landmark_heatmaps(landmarks, valid)
    assert heatmaps.shape == (1, 2, 72, 128, 128)
    assert masks.shape == (1, 2, 128, 128)
    assert torch.count_nonzero(heatmaps[0, 1]) == 0
    assert torch.count_nonzero(masks[0, 1]) == 0
    assert torch.all((masks >= 0) & (masks <= 1))

    encoder = FaceSpatialEncoder()
    convolutions = [
        module for module in encoder.modules() if isinstance(module, torch.nn.Conv2d)
    ]
    assert len(convolutions) == 8
    assert sum(module.stride == (2, 2) for module in convolutions) == 3
    spatial = encoder(heatmaps, masks, valid)
    assert spatial.shape == (1, 2, 512, 16, 16)
    assert torch.count_nonzero(spatial[0, 1]) == 0
    assert torch.count_nonzero(encoder.output.weight) > 0


def test_preparer_combines_all_face_features_and_masks_invalid_target_boxes() -> None:
    condition = make_condition(batch_size=1)
    condition.face_valid[0, 1] = False
    references = make_references(batch_size=1, reference_count=1)
    preparer = FaceConditioningPreparer(
        resampler_depth=1,
        perceiver_depth=1,
    ).eval()
    prepared = preparer(condition, references).validate()
    assert prepared.spatial_features.shape == (1, 2, 512, 16, 16)
    assert prepared.identity_tokens.shape == (1, 2, 4, 512)
    assert prepared.texture_tokens.shape == (1, 2, 8, 512)
    assert prepared.texture_delta.shape == (1, 2, 4, 512)
    assert prepared.geometry_tokens.shape == (1, 2, 4, 512)
    assert prepared.face_masks.shape == (1, 2, 128, 128)
    assert prepared.target_boxes.shape == (1, 2, 4)
    assert torch.count_nonzero(prepared.spatial_features[0, 1]) == 0
    assert torch.count_nonzero(prepared.identity_tokens[0, 1]) == 0
    assert torch.count_nonzero(prepared.geometry_tokens[0, 1]) == 0
    assert torch.count_nonzero(prepared.face_masks[0, 1]) == 0
    assert torch.count_nonzero(prepared.target_boxes[0, 1]) == 0


def test_preparer_accepts_fp16_reference_cache_with_fp32_geometry() -> None:
    condition = make_condition(batch_size=1)
    references = make_references(batch_size=1, reference_count=1).to(
        dtype=torch.float16
    )
    prepared = FaceConditioningPreparer(
        resampler_depth=1,
        perceiver_depth=1,
    ).eval()(condition, references)
    assert prepared.spatial_features.dtype == torch.float32
    assert prepared.identity_tokens.dtype == torch.float32


def test_source_and_person_slot_bindings_are_local_and_invalid_safe() -> None:
    torch.manual_seed(23)
    condition = make_condition(batch_size=1)
    references = make_references(batch_size=1, reference_count=1)
    preparer = FaceConditioningPreparer(
        resampler_depth=1,
        perceiver_depth=1,
    ).eval()
    baseline = preparer(condition, references)
    changed_condition = replace(
        condition,
        source_indices=torch.tensor([[3, 5]], dtype=torch.long),
    )
    changed = preparer(changed_condition, references)
    for name in ("identity_tokens", "texture_tokens", "geometry_tokens"):
        before = getattr(baseline, name)
        after = getattr(changed, name)
        assert not torch.equal(before[:, 0], after[:, 0])
        torch.testing.assert_close(before[:, 1], after[:, 1])
    torch.testing.assert_close(baseline.texture_delta, changed.texture_delta)

    # Make both people's raw inputs identical: slot 0/1 must still remain distinct.
    identical_condition = replace(
        condition,
        landmarks=condition.landmarks[:, :1].expand(-1, 2, -1, -1).clone(),
        jaw_pose=condition.jaw_pose[:, :1].expand(-1, 2, -1).clone(),
        expression=condition.expression[:, :1].expand(-1, 2, -1).clone(),
        source_boxes=condition.source_boxes[:, :1].expand(-1, 2, -1).clone(),
        target_boxes=condition.target_boxes[:, :1].expand(-1, 2, -1).clone(),
    )
    identical_references = FaceReferenceFeatures(
        arcface=references.arcface[:, :1].expand(-1, 2, -1, -1).clone(),
        dino_patches=references.dino_patches[:, :1]
        .expand(-1, 2, -1, -1, -1)
        .clone(),
        reference_valid=references.reference_valid[:, :1].expand(-1, 2, -1).clone(),
    )
    with torch.no_grad():
        preparer.source_index_embedding.weight.zero_()
        preparer.person_slot_embedding.weight[0].zero_()
        preparer.person_slot_embedding.weight[1].fill_(0.25)
    slot_bound = preparer(identical_condition, identical_references)
    assert not torch.equal(slot_bound.identity_tokens[:, 0], slot_bound.identity_tokens[:, 1])
    assert not torch.equal(slot_bound.texture_tokens[:, 0], slot_bound.texture_tokens[:, 1])
    assert not torch.equal(slot_bound.geometry_tokens[:, 0], slot_bound.geometry_tokens[:, 1])

    invalid_condition = replace(
        condition,
        face_valid=torch.tensor([[True, False]]),
        source_indices=torch.tensor([[2, 999]], dtype=torch.long),
    )
    invalid = preparer(invalid_condition, references)
    assert torch.count_nonzero(invalid.identity_tokens[:, 1]) == 0
    assert torch.count_nonzero(invalid.texture_tokens[:, 1]) == 0
    assert torch.count_nonzero(invalid.geometry_tokens[:, 1]) == 0
