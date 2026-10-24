"""The conv-input gate must work through the real installed Native Body, not just the module.

`prepare_geometry(conv_input_valid=...)` is unit-tested; this pins the two links the training entry
actually uses: `bind_native_body(..., conv_input_gate=True)` reaching the geometry stack inside a
`QwenImage21Transformer2DModel`, and the resulting gradient at a fully-invalid canvas token.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from src.pose_control.qwen21_full.native_body import NativeBodyConfig, Qwen21NativeBody
from src.pose_control.qwen21_full.native_geometry import NativeGeometryFeatures
from src.pose_control.qwen21_full.native_integration import bind_native_body, install_native_body

GRID = (2, 4)
TOKENS = GRID[0] * GRID[1]
GEOMETRY_DIM = 32
LATENT_DIM = 64  # what the transformer is fed; inner_dim is what Native Body injects into


def build():
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel

    torch.manual_seed(7)
    model = QwenImage21Transformer2DModel(
        in_channels=64, out_channels=64, num_layers=32, num_attention_heads=2,
        attention_head_dim=8, context_in_dim=8, axes_dims_rope=(2, 2, 4), mlp_ratio=2,
        causal_condition=True).eval()
    body = Qwen21NativeBody(NativeBodyConfig(hidden_dim=model.inner_dim,
                                             geometry_dim=GEOMETRY_DIM, width=8))
    install_native_body(model, body)
    with torch.no_grad():
        for head in body.output_heads:
            nn.init.normal_(head.weight, std=.03)
    return model, body


def features():
    generator = torch.Generator().manual_seed(11)
    return NativeGeometryFeatures(
        torch.randn(1, TOKENS, GEOMETRY_DIM, generator=generator).requires_grad_(True),
        torch.zeros(1, TOKENS, GEOMETRY_DIM), GRID, torch.tensor([[True, False]]))


def inputs():
    # img_mask is per four-token slot: 3 text positions + 2 target slots; the reference image is
    # (1,2,2)=4 tokens and the target canvas is the (2,4) grid = 8 tokens.
    img_mask = torch.tensor([[True, False, False, True, True]])
    return {"hidden_states": torch.randn(1, 4 + TOKENS, LATENT_DIM),
            "encoder_hidden_states": torch.randn(1, 3, 8),
            "encoder_hidden_states_mask": torch.ones(1, 3, dtype=torch.bool),
            "timestep": torch.tensor([.5]),
            "img_shapes": [[(1, 2, 2), (1, 2, 4)]], "img_mask": img_mask, "return_dict": False}


def padding_token_gradient(model, feats, coverage, *, gate: bool):
    with bind_native_body(model, feats, valid_mask=coverage, conv_input_gate=gate):
        out = model(**inputs())[0]
    return torch.autograd.grad(out[:, 4:].sum(), feats.geometry)[0]  # target slots only


def test_gate_reaches_the_installed_transformer():
    model, _ = build()
    feats = features()
    coverage = torch.ones(1, TOKENS)
    coverage[0, -1] = 0.0  # a 16x16 block holding no real image area
    with torch.enable_grad():
        off = padding_token_gradient(model, feats, coverage, gate=False)
    feats = features()
    with torch.enable_grad():
        on = padding_token_gradient(model, feats, coverage, gate=True)
    assert torch.count_nonzero(off[0, -1]) > 0, "the ungated path must still show the leak"
    assert torch.count_nonzero(on[0, -1]) == 0, "gate did not reach the geometry stack"
    assert torch.count_nonzero(on[0, :-1]) > 0, "valid tokens must still receive gradient"


def test_gate_needs_a_declared_mask_and_is_off_by_default():
    model, _ = build()
    feats = features()
    values = inputs()
    with bind_native_body(model, feats, conv_input_gate=True):
        with pytest.raises(ValueError, match="conv_input_gate needs a valid_mask"):
            model(**values)
    # Same binding without the gate must run, proving the default is unchanged behaviour.
    with torch.no_grad(), bind_native_body(model, feats):
        out = model(**values)[0]
    assert out.ndim == 3 and out.shape[0] == 1 and out.shape[-1] == LATENT_DIM
