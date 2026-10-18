"""Dual control must bind both source people and leave native prefix intact."""
import torch
from torch import nn

from src.pose_control.qwen21.body_adapter import Qwen21BodyAdapter, Qwen21ControlContext
from src.pose_control.qwen21.dual_adapter import Qwen21DualAdapter, Qwen21DualState


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([nn.Identity() for _ in range(32)])

    def forward(self, hidden_states, timestep, img_shapes):
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
        return hidden_states


def test_dual_source_slots_and_missing_person_mask():
    adapter = Qwen21DualAdapter(width=16, hidden_size=8)
    latent = torch.zeros(1, 4, 64)
    latent[:, 0, 0] = 1
    latent[:, 3, 1] = 2
    masks = torch.zeros(1, 2, 1, 32, 32)
    masks[:, 0, :, :16, :16] = 1
    masks[:, 1, :, 16:, 16:] = 1
    tokens = adapter._source_tokens(latent, masks)
    assert tokens.shape == (1, 2, 8, 16)
    assert not torch.allclose(tokens[:, 0], tokens[:, 1])
    with torch.no_grad():
        reversed_tokens = adapter._source_tokens(latent, masks.flip(1))
    torch.testing.assert_close(tokens[:, 0], reversed_tokens[:, 1])
    torch.testing.assert_close(tokens[:, 1], reversed_tokens[:, 0])
    masks[:, 1] = 0
    try:
        adapter._source_tokens(latent, masks)
    except ValueError as error:
        assert "mask is empty" in str(error)
    else:
        raise AssertionError("empty B source mask was accepted")


def test_dual_interaction_zero_for_invalid_b_and_prefix_unchanged():
    adapter = Qwen21DualAdapter(width=16, hidden_size=8)
    for head in adapter.interaction_heads.values():
        nn.init.constant_(head.weight, 0.01)
    state = Qwen21DualState(
        geometry=torch.ones(1, 4, 16), interaction=torch.ones(1, 4, 16),
        appearance=torch.zeros(1, 4, 16), target_mask=torch.ones(1, 4, 1),
        interaction_valid=torch.tensor([False]),
    )
    hidden = torch.randn(1, 8, 8)
    model = TinyTransformer()
    sigma = torch.tensor([0.5])
    shapes = [[(1, 2, 2), (1, 2, 2)]]
    with Qwen21ControlContext(model, adapter, state, strength=0):
        zero = model(hidden_states=hidden, timestep=sigma, img_shapes=shapes)
    torch.testing.assert_close(zero, hidden, rtol=0, atol=0)
    with Qwen21ControlContext(model, adapter, state):
        controlled = model(hidden_states=hidden, timestep=sigma, img_shapes=shapes)
    torch.testing.assert_close(controlled[:, :4], hidden[:, :4], rtol=0, atol=0)
    with Qwen21ControlContext(model, adapter, state.geometry_only()):
        geometry_only = model(hidden_states=hidden, timestep=sigma, img_shapes=shapes)
    torch.testing.assert_close(controlled, geometry_only)
    state.interaction_valid[:] = True
    with Qwen21ControlContext(model, adapter, state):
        active = model(hidden_states=hidden, timestep=sigma, img_shapes=shapes)
    assert not torch.allclose(active[:, -4:], geometry_only[:, -4:])


def test_existing_single_residual_is_unchanged():
    original = Qwen21BodyAdapter(width=16, hidden_size=8)
    dual = Qwen21DualAdapter(width=16, hidden_size=8)
    dual.load_state_dict(original.state_dict(), strict=False)
    hidden = torch.randn(1, 4, 8)
    geometry = torch.randn(1, 4, 16)
    sigma = torch.tensor([0.37])
    for layer in (3, 8, 13, 18, 23, 28):
        torch.testing.assert_close(dual.residual(layer, hidden, geometry, sigma),
                                   original.residual(layer, hidden, geometry, sigma), rtol=0, atol=0)
