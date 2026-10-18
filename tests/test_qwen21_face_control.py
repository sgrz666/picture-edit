"""Face branch preserves original Body behavior and edits only target face tokens."""

import torch
from torch import nn

from src.pose_control.qwen21.body_adapter import Qwen21ControlContext
from src.pose_control.qwen21.face_adapter import Qwen21FaceAdapter


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(nn.Identity() for _ in range(32))

    def forward(self, hidden_states, timestep, img_shapes):
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class UnusedBody(nn.Module):
    def residual(self, *args):
        raise AssertionError('Body branch should be disabled in this test')


class ConstantBody(nn.Module):
    def residual(self, layer_index, target, geometry, sigma):
        return torch.ones_like(target) * .01


def test_face_zero_init_target_only_roi_and_gradients():
    face = Qwen21FaceAdapter(width=16, hidden_size=8)
    state = face.prepare(torch.ones(1, 512), torch.full((1, 3, 128, 128), 128, dtype=torch.uint8),
                         torch.tensor([[.3, .3, .7, .7]]), torch.tensor([True]), token_hw=(4, 4))
    model = TinyTransformer()
    hidden = torch.randn(1, 48, 8)
    sigma = torch.tensor([.5])
    shapes = [[(1, 4, 4)] * 3]  # two references and one target
    native = model(hidden_states=hidden, timestep=sigma, img_shapes=shapes)
    with Qwen21ControlContext(model, UnusedBody(), torch.zeros(1, 16, 1),
                              strength=0, face_adapter=face, face_state=state, face_strength=1):
        zero = model(hidden_states=hidden, timestep=sigma, img_shapes=shapes)
    torch.testing.assert_close(zero, native, atol=0, rtol=0)
    with torch.no_grad():
        face.zero_heads['3'].weight.fill_(.01)
    with Qwen21ControlContext(model, UnusedBody(), torch.zeros(1, 16, 1),
                              strength=0, face_adapter=face, face_state=state, face_strength=1):
        controlled = model(hidden_states=hidden, timestep=sigma, img_shapes=shapes)
    torch.testing.assert_close(controlled[:, :32], native[:, :32], atol=0, rtol=0)
    delta = controlled[:, 32:] - native[:, 32:]
    assert torch.count_nonzero(delta[state.mask.squeeze(-1) == 0]) == 0
    assert torch.count_nonzero(delta[state.mask.squeeze(-1) != 0]) > 0
    controlled[:, 32:].sum().backward()
    assert face.zero_heads['3'].weight.grad is not None


def test_invalid_face_falls_back_to_body():
    face = Qwen21FaceAdapter(width=16, hidden_size=8)
    state = face.prepare(torch.ones(1, 512), torch.zeros(1, 3, 128, 128, dtype=torch.uint8),
                         torch.tensor([[.3, .3, .7, .7]]), torch.tensor([False]), token_hw=(4, 4))
    assert torch.count_nonzero(state.mask) == 0
    hidden = torch.randn(1, 16, 8)
    delta = face.residual(3, hidden, state, torch.tensor([.5]))
    assert torch.count_nonzero(delta) == 0


def test_1024_face_mask_has_4096_target_tokens_and_zero_outside_roi():
    face = Qwen21FaceAdapter(width=16, hidden_size=8)
    state = face.prepare(torch.ones(1, 512), torch.full((1, 3, 128, 128), 128, dtype=torch.uint8),
                         torch.tensor([[.3, .3, .7, .7]]), torch.tensor([True]), token_hw=(64, 64))
    assert state.mask.shape == (1, 4096, 1)
    with torch.no_grad():
        face.zero_heads['3'].weight.fill_(.01)
    delta = face.residual(3, torch.randn(1, 4096, 8), state, torch.tensor([.5]))
    assert torch.count_nonzero(delta[state.mask.squeeze(-1) == 0]) == 0
    assert torch.count_nonzero(delta[state.mask.squeeze(-1) != 0]) > 0


def test_trained_face_switch_off_is_exact_body():
    face = Qwen21FaceAdapter(width=16, hidden_size=8)
    with torch.no_grad():
        face.zero_heads['3'].weight.fill_(.01)
    state = face.prepare(torch.ones(1, 512), torch.full((1, 3, 128, 128), 128, dtype=torch.uint8),
                         torch.tensor([[.3, .3, .7, .7]]), torch.tensor([True]), token_hw=(4, 4))
    model, body = TinyTransformer(), ConstantBody()
    hidden, sigma, shapes = torch.randn(1, 32, 8), torch.tensor([.5]), [[(1, 4, 4)] * 2]
    with Qwen21ControlContext(model, body, torch.zeros(1, 16, 1)):
        baseline = model(hidden_states=hidden, timestep=sigma, img_shapes=shapes)
    with Qwen21ControlContext(model, body, torch.zeros(1, 16, 1),
                              face_adapter=face, face_state=state, face_strength=0):
        switched_off = model(hidden_states=hidden, timestep=sigma, img_shapes=shapes)
    torch.testing.assert_close(switched_off, baseline, atol=0, rtol=0)
