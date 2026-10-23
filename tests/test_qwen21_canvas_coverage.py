"""Canvas coverage must reach the native body residual as a fraction, not a flag.

A padded portrait has boundary tokens that cover only part of the real image. These tests pin
the two behaviours the remediation plan asks for: fully invalid padding tokens receive no
residual and no gradient, and a half-covered boundary token is damped by exactly that fraction.
"""
from __future__ import annotations

import importlib

import pytest
import torch
from torch import nn

from src.pose_control.qwen21_full.native_geometry import NativeGeometryFeatures


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def features(seed=21, *, dim=32, requires_grad=False, hw=(2, 2)):
    generator = torch.Generator().manual_seed(seed)
    count = hw[0] * hw[1]
    return NativeGeometryFeatures(
        torch.randn(1, count, dim, generator=generator).requires_grad_(requires_grad),
        torch.zeros(1, count, dim), hw, torch.tensor([[True, False]]))


def adapter():
    module = importlib.import_module("src.pose_control.qwen21_full.native_body")
    torch.manual_seed(10)
    return module.Qwen21NativeBody(
        module.NativeBodyConfig(hidden_dim=16, geometry_dim=32, width=8))


def inject(model, hidden, feats, valid_mask=None, *, prepared=None):
    return model(hidden, prepared=model.prepare_geometry(feats) if prepared is None else prepared,
                 layer=4, sigma=torch.tensor([.5]),
                 target_mask=torch.ones(hidden.shape[1], dtype=torch.bool),
                 valid_mask=valid_mask)


def initialize_heads(model):
    """Fresh Linear heads default to near-zero weights, which would hide every assertion."""
    with torch.no_grad():
        for head in model.output_heads:
            nn.init.normal_(head.weight, std=.04)
    return model


def test_boolean_canvas_mask_still_suppresses_exactly_as_before():
    model = initialize_heads(adapter())
    hidden, feats = torch.randn(1, 4, 16), features()
    flagged = torch.tensor([[True, False, True, False]])
    output = inject(model, hidden, feats, flagged)
    torch.testing.assert_close(output[:, [1, 3]], hidden[:, [1, 3]], rtol=0, atol=0)
    assert not torch.equal(output[:, [0, 2]], hidden[:, [0, 2]])


def test_fractional_coverage_scales_the_residual_by_exactly_that_area():
    model = initialize_heads(adapter())
    hidden = torch.randn(1, 4, 16)
    feats = features()
    unmasked = inject(model, hidden, feats)
    assert not torch.equal(unmasked, hidden), "heads are live, so an unmasked token must change"
    # Dyadic weights keep the scaling exact; the unmasked reference is reconstructed by
    # subtraction, so only the 0.0 and 1.0 lanes can be asserted bit-exactly.
    coverage = torch.tensor([[1.0, 0.5, 0.25, 0.0]])
    damped = inject(model, hidden, feats, coverage)
    torch.testing.assert_close(damped - hidden, coverage[..., None] * (unmasked - hidden),
                               rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(damped[:, 3], hidden[:, 3], rtol=0, atol=0)
    torch.testing.assert_close(damped[:, 0], unmasked[:, 0], rtol=0, atol=0)


def test_invalid_tokens_get_no_residual_but_still_leak_gradient_through_the_local_conv():
    """Measured behaviour, not the wish list.

    The coverage mask writes no residual to an invalid token, yet prepare_geometry() runs a local
    convolution over the token grid, so a padding token still reaches its neighbours and still
    receives gradient. Suppressing that means masking the conv input and then the Attention slots
    -- a semantics change for already-trained weights, which the plan sequences *after* this
    measurement ("先最小化 Padding...如果仍有明显边界影响，再实现 Attention 级别的屏蔽").
    """
    model = initialize_heads(adapter())
    hidden = torch.randn(1, 4, 16)
    feats = features(requires_grad=True)
    coverage = torch.tensor([[1.0, 0.5, 0.25, 0.0]])
    output = inject(model, hidden, feats, coverage)
    torch.testing.assert_close(output[:, 3], hidden[:, 3], rtol=0, atol=0)
    gradient = torch.autograd.grad(output.sum(), feats.geometry)[0]
    assert torch.count_nonzero(gradient[0, 3]) > 0, "conv leak disappeared; update this test and NATIVE_BODY"
    leaked = float(gradient[0, 3].abs().mean())
    valid = float(gradient[0, :3].abs().mean())
    print(f"padding-token gradient {leaked:.3e} vs valid tokens {valid:.3e} (ratio {leaked / valid:.3f})")
    assert leaked < valid


def test_out_of_range_coverage_is_refused_rather_than_clamped():
    model, hidden, feats = adapter(), torch.randn(1, 4, 16), features()
    with pytest.raises(ValueError, match="0..1 area coverage"):
        inject(model, hidden, feats, torch.full((1, 4), 2.0))
    with pytest.raises(ValueError, match="0..1 area coverage"):
        inject(model, hidden, feats, torch.tensor([[-0.5, 1.0, 1.0, 1.0]]))
    with pytest.raises(ValueError, match="0..1 area coverage"):
        inject(model, hidden, feats, torch.full((1, 4), float("nan")))
    with pytest.raises(ValueError, match="0..1 area coverage"):
        inject(model, hidden, feats, torch.rand(4, 1))
    with pytest.raises(ValueError, match="0..1 area coverage"):
        inject(model, hidden, feats, torch.randint(0, 2, (1, 4)))


def test_boundary_token_coverage_of_a_padded_portrait_is_a_half_not_a_one():
    from src.pose_control.qwen21_full.face_coordinates import valid_image_tokens

    mask = torch.zeros(1, 1, 64, 32)
    mask[0, 0, :56, :] = 1  # 8 padding rows on a 64x32 canvas -> the last token row is half covered
    batch = {"valid_image_mask": mask, "tgt_image": torch.zeros(1, 3, 64, 32)}
    coverage = valid_image_tokens(batch, (4, 2), "cpu")
    assert coverage.shape == (1, 8, 1)
    assert coverage.flatten()[:6].tolist() == [1.0] * 6
    assert coverage.flatten()[6:].tolist() == [0.5, 0.5]


def test_the_body_loss_excludes_padding_tokens_exactly_and_downweights_boundaries():
    """Plan §8 'Padding Loss': a fully invalid token must not be able to move the loss."""
    from src.data.letterbox import valid_body_velocity_loss

    generator = torch.Generator().manual_seed(5)
    squared = torch.rand(1, 8, 10, generator=generator) * 5.0
    human = torch.zeros(1, 8, 1)
    human[0, 0] = 1.0
    coverage = torch.tensor([[[1.0], [1.0], [0.5], [0.0], [0.0], [0.0], [0.0], [0.0]]])
    reference = valid_body_velocity_loss(squared, human, coverage)
    assert torch.isfinite(reference) and float(reference) > 0.0

    erased = squared.clone()
    erased[:, 3:] = 0.0  # the five fully-padding tokens, plus the boundary one at half weight
    torch.testing.assert_close(valid_body_velocity_loss(erased, human, coverage), reference,
                               rtol=0, atol=1e-7)
    boundary = squared.clone()
    boundary[:, 2] = 0.0  # only the half-covered token changes
    assert float(valid_body_velocity_loss(boundary, human, coverage)) < float(reference)
    with pytest.raises(ValueError, match="no valid image tokens"):
        valid_body_velocity_loss(squared, human, torch.zeros(1, 8, 1))


def test_integer_grid_may_not_be_read_as_a_square_when_the_canvas_is_a_portrait():
    from src.pose_control.qwen21_full.face_coordinates import valid_image_tokens

    batch = {"valid_image_mask": torch.ones(1, 1, 64, 32), "tgt_image": torch.zeros(1, 3, 64, 32)}
    with pytest.raises(ValueError, match="real canvas grid"):
        valid_image_tokens(batch, 4, "cpu")
    # A genuinely square canvas keeps working through the integer spelling.
    square = {"valid_image_mask": torch.ones(1, 1, 64, 64), "tgt_image": torch.zeros(1, 3, 64, 64)}
    assert valid_image_tokens(square, 4, "cpu").shape == (1, 16, 1)


COVERAGE = torch.tensor([[1.0, 0.5, 0.25, 0.0]])


def gated(model, hidden, feats, coverage=COVERAGE):
    prepared = model.prepare_geometry(feats, conv_input_valid=(coverage != 0))
    return model(hidden, prepared=prepared, layer=4, sigma=torch.tensor([.5]),
                 target_mask=torch.ones(hidden.shape[1], dtype=torch.bool), valid_mask=coverage)


def test_gating_the_conv_input_makes_the_padding_gradient_exactly_zero():
    """方案 §8「完全无效 Token 的梯度为 0」: damping the residual is not enough.

    The convolution mixes neighbours, so an unmasked padding token still receives gradient from the
    loss on its valid neighbours. Zeroing its contribution before the conv is what zeroes that.
    """
    model = initialize_heads(adapter())
    hidden, feats = torch.randn(1, 4, 16), features(requires_grad=True)
    output = gated(model, hidden, feats)
    torch.testing.assert_close(output[:, 3], hidden[:, 3], rtol=0, atol=0)
    gradient = torch.autograd.grad(output.sum(), feats.geometry)[0]
    assert torch.count_nonzero(gradient[0, 3]) == 0, "padding token still receives gradient"
    for row in range(3):
        assert torch.count_nonzero(gradient[0, row]) > 0, f"valid token {row} stopped learning"


def test_gating_removes_the_leak_that_the_unmasked_path_measured():
    model = initialize_heads(adapter())
    hidden, feats = torch.randn(1, 4, 16), features(requires_grad=True)
    ungated = inject(model, hidden, feats, COVERAGE)
    gated_out = gated(model, hidden, feats)
    assert not torch.equal(gated_out[:, :3], ungated[:, :3]), \
        "a padding neighbour that no longer leaks must change its neighbours"
    torch.testing.assert_close(gated_out[:, 3], ungated[:, 3], rtol=0, atol=0)


def test_an_all_valid_gate_is_bit_identical_to_no_gate_at_all():
    # Control: the gate must not perturb numerics when nothing is padding.
    model = initialize_heads(adapter())
    feats = features()
    plain = model.prepare_geometry(feats)
    opened = model.prepare_geometry(feats, conv_input_valid=torch.ones(1, 4, dtype=torch.bool))
    torch.testing.assert_close(plain.tokens, opened.tokens, rtol=0, atol=0)


def test_gate_shape_and_finateness_are_checked_before_it_is_used():
    model, feats = adapter(), features()
    with pytest.raises(ValueError, match="conv_input_valid"):
        model.prepare_geometry(feats, conv_input_valid=torch.ones(4, dtype=torch.bool))
    with pytest.raises(ValueError, match="conv_input_valid"):
        model.prepare_geometry(feats, conv_input_valid=torch.full((1, 4), float("nan")))
    with pytest.raises(ValueError, match="conv_input_valid"):
        model.prepare_geometry(feats, conv_input_valid=torch.full((1, 4), 2.0))
    # The geometry itself is untouched by the check, so the ungated path still runs.
    assert model.prepare_geometry(feats, conv_input_valid=None).tokens.shape == (1, 4, 8)
