"""The control hook may change target tokens but must preserve prefix tokens."""

import torch
from torch import nn

from src.pose_control.qwen21.body_adapter import LAYER_INDICES, Qwen21ControlContext
from src.pose_control.qwen21.runtime import Qwen21IPERCache, transformer_inputs
from PIL import Image


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([nn.Identity() for _ in range(32)])
        self.frozen_weight = nn.Parameter(torch.tensor(2.0), requires_grad=False)

    def forward(self, hidden_states, timestep, img_shapes):
        hidden_states = hidden_states * self.frozen_weight
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class TinyAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.0))

    def residual(self, layer_index, target_hidden, geometry, sigma):
        assert layer_index in LAYER_INDICES
        return self.scale * geometry.expand_as(target_hidden)


def test_zero_control_matches_native_and_nonzero_control_changes_only_target():
    model = TinyTransformer()
    adapter = TinyAdapter()
    hidden = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    sigma = torch.tensor([0.5])
    shapes = [[(1, 1, 3), (1, 1, 3)]]
    native = model(hidden_states=hidden, timestep=sigma, img_shapes=shapes)
    with Qwen21ControlContext(model, adapter, torch.ones(1, 3, 1)) as control:
        zero = model(hidden_states=hidden, timestep=sigma, img_shapes=shapes)
    torch.testing.assert_close(zero, native, rtol=0, atol=0)
    assert all(value == 1 for value in control.call_counts.values())

    adapter.scale.data.fill_(0.25)
    with Qwen21ControlContext(model, adapter, torch.ones(1, 3, 1)):
        changed = model(hidden_states=hidden, timestep=sigma, img_shapes=shapes)
        changed[:, -3:].sum().backward()
    torch.testing.assert_close(changed[:, :3], native[:, :3], rtol=0, atol=0)
    torch.testing.assert_close(changed[:, -3:] - native[:, -3:], torch.full((1, 3, 4), 1.5))
    # KV-cache inference can call the block with only the target segment.
    target_only = hidden[:, -3:]
    with Qwen21ControlContext(model, adapter, torch.ones(1, 3, 1)):
        cached = model(hidden_states=target_only, timestep=sigma, img_shapes=shapes)
    torch.testing.assert_close(cached - model(hidden_states=target_only, timestep=sigma, img_shapes=shapes), torch.full((1, 3, 4), 1.5))
    assert adapter.scale.grad is not None and adapter.scale.grad.abs() > 0
    assert model.frozen_weight.grad is None
    assert not getattr(model, "_pic_edit_qwen21_control_active")


def test_1024_transformer_input_uses_64_square_target_grid():
    pair = {
        'resolution': 1024,
        'source_latent': torch.zeros(1, 4096, 4),
        'prompt_embeds': torch.zeros(1, 2, 4),
        'prompt_mask': torch.ones(1, 2, dtype=torch.bool),
        'img_mask': torch.ones(1, 2048, dtype=torch.bool),
    }
    result = transformer_inputs(pair, torch.zeros(1, 4096, 4), torch.tensor([.5]))
    assert result['img_shapes'] == [[(1, 64, 64), (1, 64, 64)]]
    assert result['hidden_states'].shape == (1, 8192, 4)


def test_512_transformer_input_keeps_32_square_target_grid():
    pair = {
        'resolution': 512,
        'source_latent': torch.zeros(1, 1024, 4),
        'prompt_embeds': torch.zeros(1, 2, 4),
        'prompt_mask': torch.ones(1, 2, dtype=torch.bool),
        'img_mask': torch.ones(1, 512, dtype=torch.bool),
    }
    result = transformer_inputs(pair, torch.zeros(1, 1024, 4), torch.tensor([.5]))
    assert result['img_shapes'] == [[(1, 32, 32), (1, 32, 32)]]
    assert result['hidden_states'].shape == (1, 2048, 4)


def test_latent_cache_keys_are_resolution_specific(tmp_path):
    image = Image.new('RGB', (8, 8), 'white')
    small = Qwen21IPERCache(tmp_path, resolution=512)
    large = Qwen21IPERCache(tmp_path, resolution=1024)
    assert small._path(image, 'source') != large._path(image, 'source')
