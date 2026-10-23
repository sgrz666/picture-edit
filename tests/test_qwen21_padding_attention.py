"""Tests for the opt-in padding-key exclusion and the token-coverage helpers."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from src.data.canvas_contract import aspect_preserving, native_pad
from src.pose_control.qwen21_full.padding_attention import (
    install_padding_key_mask, padding_attention_active, padding_key_masked)


class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, hidden_states=None, *, key_valid=None, attention_mask=None, **kwargs):
        self.seen = key_valid
        return hidden_states


class FakeModel(nn.Module):
    def __init__(self, blocks=3):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(FakeBlock() for _ in range(blocks))


def drive(model, *, tokens, batch=1, attention_mask=None):
    """Call every block the way the backbone does, with an all-valid key mask."""
    key_valid = torch.ones(batch, tokens + 5, dtype=torch.bool)
    for block in model.transformer_blocks:
        block(key_valid=key_valid, attention_mask=attention_mask)
    return key_valid


# ---- token coverage on the contract ----------------------------------------------------------

def test_token_coverage_counts_real_image_area_per_block():
    spec = native_pad((400, 225))
    coverage = spec.token_valid_coverage()
    assert coverage.shape == (26, 16)
    padding = spec.padding_token_mask()
    assert int(padding.sum()) == 41, int(padding.sum())
    assert coverage.min().item() == 0.0
    assert coverage.max().item() == 1.0
    # Column 15 spans x=240..255 while the frame ends at x=225, so that whole column is padding.
    assert bool(padding[:, 15].all())
    # Row 25 spans y=400..415 and the frame ends at y=400, so that whole row is padding too.
    assert bool(padding[25, :].all())
    # 26 padding columns + 16 padding rows, corner counted once.
    assert int(padding.sum()) == 26 + 16 - 1
    # The token straddling the right edge holds 1 real column of 16, so it must stay attendable.
    assert coverage[24, 14].item() == pytest.approx(16 * 1 / 256.0)
    assert coverage[0, 0].item() == 1.0


def test_a_tiktok_portrait_has_no_fully_invalid_tokens_only_partial_ones():
    spec = native_pad((1080, 604))
    coverage = spec.token_valid_coverage()
    assert coverage.shape == (68, 38)
    assert int(spec.padding_token_mask().sum()) == 0, "8px margins never fill a whole 16px block"
    assert coverage[67, 37].item() == pytest.approx((604 - 592) * (1080 - 1072) / 256.0)
    assert float(coverage[:67, :37].min()) == 1.0


def test_budget_canvases_report_their_centred_padding():
    spec = aspect_preserving((1080, 604), max_pixels=256 * 256)
    coverage = spec.token_valid_coverage()
    assert coverage.shape == tuple(spec.token_hw)
    assert coverage.max().item() == 1.0
    assert int(spec.padding_token_mask().sum()) == int((coverage == 0).sum())
    # Centred padding leaves a partial band on both edges, and those tokens are not "nothing".
    assert 0.0 < float(coverage.min()) or int(spec.padding_token_mask().sum()) > 0
    interior = coverage[1:-1, 1:-1]
    assert float(interior.min()) == 1.0, "only the outer ring may be partially covered"


# ---- the exclusion hook ------------------------------------------------------------------------

def test_disabled_by_default_and_flips_only_padding_keys_when_enabled():
    model = FakeModel()
    spec = native_pad((400, 225))
    tokens = spec.token_count
    install_padding_key_mask(model)

    drive(model, tokens=tokens)
    assert model.transformer_blocks[0].seen.all(), "no binding means the mask is untouched"
    assert not padding_attention_active(model)

    with padding_key_masked(model, spec, allow_semantic_change=True) as info:
        assert info["padding_tokens"] == 41 and info["valid_tokens"] == 375
        key_valid = drive(model, tokens=tokens)
    assert not padding_attention_active(model)

    seen = model.transformer_blocks[0].seen
    tail = seen[:, -tokens:].reshape(26, 16)
    assert bool((tail == spec.padding_token_mask().__invert__()).all())
    assert int(tail.sum()) == 375
    assert bool(seen[:, :5].all()), "the prefix (text and reference image) must be untouched"
    assert key_valid is not seen, "the caller's tensor must not be mutated in place"


def test_every_block_is_wrapped_not_just_the_first():
    model = FakeModel(blocks=4)
    install_padding_key_mask(model)
    with padding_key_masked(model, native_pad((400, 225)), allow_semantic_change=True):
        drive(model, tokens=native_pad((400, 225)).token_count)
    for block in model.transformer_blocks:
        assert not block.seen[:, -416:].all(), block


def test_refuses_without_an_explicit_semantics_acknowledgement():
    model = FakeModel()
    install_padding_key_mask(model)
    with pytest.raises(ValueError, match="allow_semantic_change"):
        with padding_key_masked(model, native_pad((400, 225))):
            pass


def test_refuses_a_blockmask_path_it_does_not_own():
    model = FakeModel()
    install_padding_key_mask(model)
    with padding_key_masked(model, native_pad((400, 225)), allow_semantic_change=True):
        with pytest.raises(RuntimeError, match="BlockMask"):
            drive(model, tokens=native_pad((400, 225)).token_count, attention_mask=object())


def test_refuses_a_batch_it_cannot_attribute_to_one_canvas():
    model = FakeModel()
    install_padding_key_mask(model)
    with padding_key_masked(model, native_pad((400, 225)), allow_semantic_change=True):
        with pytest.raises(ValueError, match="batch size 1"):
            drive(model, tokens=native_pad((400, 225)).token_count, batch=2)


def test_refuses_double_install_and_nesting():
    model = FakeModel()
    report = install_padding_key_mask(model)
    assert report["blocks_wrapped"] == 3 and report["default"] == "disabled"
    with pytest.raises(ValueError, match="already installed"):
        install_padding_key_mask(model)
    spec = native_pad((400, 225))
    with padding_key_masked(model, spec, allow_semantic_change=True):
        with pytest.raises(RuntimeError, match="not reentrant"):
            with padding_key_masked(model, spec, allow_semantic_change=True):
                pass


def test_restores_the_binding_even_when_the_body_raises():
    model = FakeModel()
    install_padding_key_mask(model)
    with pytest.raises(KeyError):
        with padding_key_masked(model, native_pad((400, 225)), allow_semantic_change=True):
            raise KeyError("boom")
    assert not padding_attention_active(model)
