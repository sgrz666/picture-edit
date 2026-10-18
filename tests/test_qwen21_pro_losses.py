"""区域监督按内容面积归一化，固定几何裁剪保留生成端梯度。"""
import importlib
from dataclasses import replace

import pytest
import torch
from torch import nn

from test_qwen21_pro_contracts import make_condition
from src.pose_control.qwen21_pro import ProAdapterConfig


def api():
    return importlib.import_module("src.pose_control.qwen21_pro.losses")


def regions():
    pack = make_condition(people=1)
    return replace(pack, face_mask=torch.tensor([[[1., 0., 0., 0.]]]),
                   hand_mask=torch.tensor([[[[1., 1., 1., 0.], [0., 0., 0., 0.]]]]),
                   spatial=dict(pack.spatial, contact=torch.tensor([[[[[1., 1.], [0., 0.]]]]])),
                   spatial_valid=dict(pack.spatial_valid, contact=torch.ones(1, 1, dtype=torch.bool)))


def test_exclusive_priority_and_valid_content_area():
    masks = api().regional_masks(regions(), torch.tensor([[1., .5, 1., 0.]]))
    expected = {"face": [1., 0., 0., 0.], "contact": [0., .5, 0., 0.],
                "hand": [0., 0., 1., 0.], "body": [0., 0., 0., 0.]}
    for name, values in expected.items():
        torch.testing.assert_close(masks[name], torch.tensor([values]))
    assert (sum(masks.values()) <= 1).all()


def test_soft_support_is_not_squared_and_invalid_person_is_excluded():
    pack = regions()
    pack = replace(pack, owner=pack.owner * .5, person_mask=pack.person_mask * .5,
                   spatial=dict(pack.spatial, visibility=pack.spatial["visibility"] * .5))
    masks = api().regional_masks(pack, torch.full((1, 4), .5))
    torch.testing.assert_close(sum(masks.values()), torch.full((1, 4), .5))
    pack = replace(pack, person_valid=torch.zeros_like(pack.person_valid), owner=torch.zeros_like(pack.owner),
                   source_token_valid=torch.zeros_like(pack.source_token_valid),
                   face_identity_valid=torch.zeros_like(pack.face_identity_valid),
                   spatial_valid={k: torch.zeros_like(v) for k, v in pack.spatial_valid.items()})
    assert sum(api().regional_masks(pack, torch.ones(1, 4)).values()).sum() == 0


def test_weighted_flow_loss_normalization_zero_area_and_coverage():
    prediction = torch.tensor([[[1., 1.], [2., 2.], [3., 3.], [99., 99.]]], requires_grad=True)
    result, metrics = api().regional_flow_loss(prediction, torch.zeros_like(prediction),
        torch.tensor([[1., .5, 1., 0.]]), regions(),
        config=api().RegionalLossConfig(lambda_global=1., lambda_face=2., lambda_contact=3., lambda_hand=4., lambda_body=9.))
    expected_global = (1 + .5 * 4 + 9) / 2.5
    assert result.item() == pytest.approx(expected_global + 2 * 1 + 3 * 4 + 4 * 9)
    assert metrics["coverage"]["contact"] == pytest.approx(.5 / 2.5)
    assert metrics["area"]["body"] == 0 and metrics["losses"]["body"] == 0
    result.backward()
    assert prediction.grad[:, 3].abs().sum() == 0


def test_spatial_validity_and_actual_source_dimension():
    pack = regions()
    pack = replace(pack, source_tokens=torch.randn(1, 1, 3, 1600),
                   spatial_valid=dict(pack.spatial_valid, contact=torch.zeros(1, 1, dtype=torch.bool)))
    masks = api().regional_masks(pack, torch.ones(1, 4), config=ProAdapterConfig(source_dim=1600))
    assert masks["contact"].sum() == 0
    with pytest.raises(ValueError, match="source_tokens"):
        api().regional_masks(pack, torch.ones(1, 4))


@pytest.mark.parametrize("bad", [-1., float("nan")])
def test_invalid_loss_weights_rejected(bad):
    with pytest.raises(ValueError):
        api().RegionalLossConfig(lambda_face=bad)


def test_fixed_geometry_feature_encoder_is_frozen_but_input_is_differentiable():
    encoder = nn.Conv2d(3, 4, 1)
    generated = torch.randn(1, 3, 8, 8, requires_grad=True)
    target = torch.zeros_like(generated, requires_grad=True)
    boxes = torch.tensor([[0., .0, .0, .5, .5]])
    loss = api().fixed_geometry_feature_loss(encoder, generated, target, boxes, output_size=(4, 4))
    loss.backward()
    assert generated.grad.abs().sum() > 0
    assert target.grad is None and all(not p.requires_grad and p.grad is None for p in encoder.parameters())
    assert generated.grad[:, :, 5:, 5:].abs().sum() == 0
    with pytest.raises(ValueError, match="fixed geometry"):
        api().fixed_geometry_feature_loss(encoder, generated, target, boxes.requires_grad_())


def test_feature_encoder_dtype_is_explicit_and_does_not_round_or_convert_storage():
    encoder = nn.Conv2d(3, 4, 1).bfloat16()
    image = torch.randn(1, 3, 8, 8, requires_grad=True)
    with pytest.raises(ValueError, match="FP32"):
        api().fixed_geometry_feature_loss(encoder, image, image.detach(), torch.tensor([[0., 0., 0., 1., 1.]]))
    assert encoder.weight.dtype == torch.bfloat16
