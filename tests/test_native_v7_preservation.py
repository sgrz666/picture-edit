"""CPU numerical and gradient contracts for explicit Source preservation."""
import pytest
import torch
from torch.nn import functional as F


def test_preservation_config_roundtrips_single_or_both_enabled_objectives():
    from src.pose_control.qwen21_full.native_v7.preservation import PreservationConfig

    for raw in ({'source_identity': .2}, {'background': .1}, {'source_identity': .2, 'background': .1}):
        config = PreservationConfig.from_dict(raw)
        assert PreservationConfig.from_dict(config.to_dict()) == config
        assert config.to_dict() == {'source_identity': raw.get('source_identity', 0.),
                                    'background': raw.get('background', 0.)}


@pytest.mark.parametrize('raw', [
    {}, {'source_identity': 0., 'background': 0.}, {'source_identity': -1.},
    {'background': float('nan')}, {'background': float('inf')},
    {'source_identity': True}, {'source_identity': '0.2'},
    {'source_identity': .2, 'unknown': 1.}, None, [],
])
def test_preservation_config_rejects_disabled_or_malformed_values(raw):
    from src.pose_control.qwen21_full.native_v7.preservation import PreservationConfig

    with pytest.raises(ValueError):
        PreservationConfig.from_dict(raw)


def test_default_preservation_config_cannot_silently_enable_nothing():
    from src.pose_control.qwen21_full.native_v7.preservation import PreservationConfig

    with pytest.raises(ValueError):
        PreservationConfig()


def test_source_identity_attracts_source_when_gt_points_elsewhere():
    from src.pose_control.qwen21_full.native_v7.preservation import source_identity_loss

    generated = torch.tensor([[[0., 1.], [0., 0.]]], requires_grad=True)
    source = torch.tensor([[[1., 0.], [0., 0.]]], requires_grad=True)
    valid = torch.tensor([[True, False]])
    loss, count = source_identity_loss(generated, source, valid)
    assert count == 1 and loss.item() == pytest.approx(1.)
    loss.backward()
    updated = generated.detach() - .1 * generated.grad
    assert F.cosine_similarity(updated[0, 0], source[0, 0], dim=0) > 0
    assert F.cosine_similarity(updated[0, 0], generated.detach()[0, 0], dim=0) < 1
    assert source.grad is None
    assert not generated.grad[0, 1].count_nonzero()


def test_source_identity_respects_person_mapping_and_absolute_cosine():
    from src.pose_control.qwen21_full.native_v7.preservation import source_identity_loss

    generated = torch.eye(2)[None].requires_grad_()
    source = generated.detach().clone()
    valid = torch.ones(1, 2, dtype=torch.bool)
    same, count = source_identity_loss(generated, source, valid)
    swapped, _ = source_identity_loss(generated, source.flip(1), valid)
    opposite, _ = source_identity_loss(generated, -source, valid)
    assert count == 2 and same.item() == pytest.approx(0.)
    assert swapped.item() == pytest.approx(1.)
    assert opposite.item() == pytest.approx(2.)


def test_source_identity_averages_all_valid_faces_including_singletons():
    from src.pose_control.qwen21_full.native_v7.preservation import source_identity_loss

    generated = torch.tensor([[[1., 0.], [0., 1.]], [[1., 0.], [1., 0.]]])
    source = torch.tensor([[[1., 0.], [1., 0.]], [[0., 1.], [-1., 0.]]])
    loss, count = source_identity_loss(generated, source, torch.tensor([[True, False], [True, True]]))
    assert count == 3 and loss.item() == pytest.approx(1.)


def test_empty_source_identity_is_differentiable_zero():
    from src.pose_control.qwen21_full.native_v7.preservation import source_identity_loss

    generated = torch.randn(2, 2, 4, requires_grad=True)
    source = torch.randn_like(generated, requires_grad=True)
    loss, count = source_identity_loss(generated, source, torch.zeros(2, 2, dtype=torch.bool))
    assert count == 0 and loss.item() == 0. and loss.requires_grad
    loss.backward()
    assert torch.equal(generated.grad, torch.zeros_like(generated))
    assert source.grad is None


@pytest.mark.parametrize('field,value', [
    ('generated', torch.ones(1, 2, dtype=torch.float32)),
    ('generated', torch.ones(1, 2, 0)),
    ('generated', torch.ones(1, 2, 3, dtype=torch.int64)),
    ('source', torch.ones(1, 3, 3)),
    ('source', torch.ones(1, 2, 3, dtype=torch.int64)),
    ('source', torch.full((1, 2, 3), float('inf'))),
    ('generated', torch.full((1, 2, 3), float('nan'))),
    ('valid', torch.ones(1, 2)),
    ('valid', torch.ones(2, dtype=torch.bool)),
    ('source', torch.empty(1, 2, 3, device='meta')),
    ('valid', torch.empty(1, 2, dtype=torch.bool, device='meta')),
    ('generated', None),
])
def test_source_identity_rejects_invalid_tensor_contracts(field, value):
    from src.pose_control.qwen21_full.native_v7.preservation import source_identity_loss

    arguments = {'generated': torch.ones(1, 2, 3), 'source': torch.ones(1, 2, 3),
                 'valid': torch.ones(1, 2, dtype=torch.bool)}
    arguments[field] = value
    with pytest.raises(ValueError):
        source_identity_loss(**arguments)


@pytest.mark.parametrize('sigma_value', [0., .2, .8, 1.])
def test_background_uses_x0_at_each_sigma_without_dividing_by_sigma(sigma_value):
    from src.pose_control.qwen21_full.native_v7.preservation import source_background_loss

    prediction = torch.tensor([[[2., -3.], [4., 2.]]], requires_grad=True)
    source = torch.tensor([[[10., 20.], [30., 40.]]], requires_grad=True)
    residual = torch.tensor([[[1., 3.], [2., 4.]]])
    noisy = source.detach() + sigma_value * prediction.detach() + residual
    loss, area = source_background_loss(prediction, noisy, torch.tensor([sigma_value]), source,
                                        torch.ones(1, 2), torch.ones(1, 2))
    assert loss.item() == pytest.approx(7.5)
    assert area.item() == pytest.approx(2.)
    loss.backward()
    assert torch.allclose(prediction.grad, -2 * sigma_value * residual / 4)
    assert source.grad is None


def test_background_excludes_foreground_and_padding_without_double_coverage():
    from src.pose_control.qwen21_full.native_v7.preservation import source_background_loss

    prediction = torch.zeros(1, 4, 1, requires_grad=True)
    noisy = torch.tensor([[[2.], [4.], [100.], [100.]]])
    support = torch.tensor([[.25, 1., 0., 0.]], requires_grad=True)
    coverage = torch.tensor([[.25, 1., 1., 0.]], requires_grad=True)
    source = torch.zeros_like(prediction, requires_grad=True)
    loss, area = source_background_loss(prediction, noisy, torch.ones(1), source, support, coverage)
    assert loss.item() == pytest.approx(13.6)
    assert area.item() == pytest.approx(1.25) and not area.requires_grad
    loss.backward()
    assert prediction.grad[0, :2].abs().sum() > 0
    assert not prediction.grad[0, 2:].count_nonzero()
    assert source.grad is None and support.grad is None and coverage.grad is None


def test_background_averages_valid_images_instead_of_token_area():
    from src.pose_control.qwen21_full.native_v7.preservation import source_background_loss

    prediction = torch.zeros(3, 3, 2, requires_grad=True)
    noisy = torch.tensor([[[1., 1.]] * 3, [[3., 3.]] * 3, [[100., 100.]] * 3])
    support = torch.tensor([[1., 1., 1.], [.25, 0., 0.], [0., 0., 0.]])
    loss, area = source_background_loss(prediction, noisy, torch.ones(3),
                                        torch.zeros_like(prediction), support, torch.ones(3, 3))
    assert loss.item() == pytest.approx(5.)
    assert area.item() == pytest.approx(3.25)
    loss.backward()
    assert not prediction.grad[2].count_nonzero()


def test_empty_background_support_is_differentiable_zero():
    from src.pose_control.qwen21_full.native_v7.preservation import source_background_loss

    prediction = torch.randn(1, 3, 64, requires_grad=True)
    source = torch.randn_like(prediction, requires_grad=True)
    support = torch.zeros(1, 3, requires_grad=True)
    loss, area = source_background_loss(prediction, torch.zeros_like(prediction), torch.tensor([.2]),
                                        source, support, torch.ones(1, 3))
    assert loss.item() == 0. and loss.requires_grad and area.item() == 0.
    loss.backward()
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))
    assert source.grad is None and support.grad is None


def test_background_accepts_bfloat_prediction_with_float32_frozen_cache():
    from src.pose_control.qwen21_full.native_v7.preservation import source_background_loss

    prediction = torch.ones(1, 2, 64, dtype=torch.bfloat16, requires_grad=True)
    loss, area = source_background_loss(prediction, torch.zeros(1, 2, 64), torch.tensor([.5]),
                                        torch.zeros(1, 2, 64), torch.ones(1, 2), torch.ones(1, 2))
    assert loss.dtype == torch.float32 and loss.item() == pytest.approx(.25)
    assert area.item() == pytest.approx(2.)
    loss.backward()
    assert prediction.grad.abs().sum() > 0


def test_background_rejects_support_exceeding_coverage_before_fp32_conversion():
    from src.pose_control.qwen21_full.native_v7.preservation import source_background_loss

    tokens = torch.zeros(1, 1, 1)
    with pytest.raises(ValueError, match='support.*coverage'):
        source_background_loss(tokens, tokens, torch.ones(1), tokens,
                               torch.tensor([[.5 + 1e-12]], dtype=torch.float64),
                               torch.tensor([[.5]], dtype=torch.float64))


@pytest.mark.parametrize('field,value', [
    ('prediction', torch.ones(1, 3)),
    ('prediction', torch.ones(1, 3, 0)),
    ('prediction', torch.ones(1, 3, 2, dtype=torch.int64)),
    ('prediction', torch.full((1, 3, 2), float('nan'))),
    ('noisy', torch.ones(1, 3, 1)),
    ('noisy', torch.ones(1, 3, 2, dtype=torch.int64)),
    ('source_latent', torch.full((1, 3, 2), float('inf'))),
    ('source_latent', torch.empty(1, 3, 2, device='meta')),
    ('sigma', torch.tensor([[.2]])),
    ('sigma', torch.tensor([1])),
    ('sigma', torch.tensor([float('nan')])),
    ('sigma', torch.tensor([-.1])),
    ('sigma', torch.tensor([1.1])),
    ('support', torch.ones(1, 3, 1)),
    ('support', torch.ones(1, 3, dtype=torch.bool)),
    ('support', torch.tensor([[1., -.1, 1.]])),
    ('support', torch.tensor([[1., 1.1, 1.]])),
    ('support', torch.tensor([[1., float('nan'), 1.]])),
    ('coverage', torch.tensor([[1., .5, 1.]])),
    ('coverage', torch.tensor([[1., 1.1, 1.]])),
    ('coverage', torch.empty(1, 3, device='meta')),
    ('prediction', None),
])
def test_background_rejects_invalid_tensor_contracts(field, value):
    from src.pose_control.qwen21_full.native_v7.preservation import source_background_loss

    arguments = {'prediction': torch.ones(1, 3, 2), 'noisy': torch.ones(1, 3, 2),
                 'sigma': torch.tensor([.2]), 'source_latent': torch.ones(1, 3, 2),
                 'support': torch.ones(1, 3), 'coverage': torch.ones(1, 3)}
    arguments[field] = value
    with pytest.raises(ValueError):
        source_background_loss(**arguments)
