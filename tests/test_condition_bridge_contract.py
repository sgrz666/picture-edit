from __future__ import annotations

import importlib

import torch

from src.pose_control.v6.reasoning import InternalControlState


def _make_state(batch_size: int = 2) -> InternalControlState:
    person_valid = torch.tensor([[True, True], [True, False]])[:batch_size]
    person_count = person_valid.sum(dim=1).long()
    interaction_valid = person_count == 2
    person_tokens = torch.randn(batch_size, 2, 16, 32)
    person_mask = torch.ones(batch_size, 2, 16, dtype=torch.bool)
    person_tokens[:, 1] *= person_valid[:, 1, None, None]
    person_mask[:, 1] &= person_valid[:, 1, None]
    interaction = torch.randn(batch_size, 32, 4, 4)
    interaction *= interaction_valid[:, None, None, None]
    interaction_high = torch.randn(batch_size, 32, 8, 8)
    interaction_high *= interaction_valid[:, None, None, None]
    return InternalControlState(
        geometry_feature=torch.randn(batch_size, 32, 4, 4),
        geometry_highres=torch.randn(batch_size, 32, 8, 8),
        interaction_feature=interaction,
        interaction_highres=interaction_high,
        person_tokens=person_tokens,
        person_token_mask=person_mask,
        person_valid=person_valid,
        person_count=person_count,
        interaction_valid=interaction_valid,
    ).validate()


def test_reasoner_bridge_projects_state_and_pools_four_tokens_per_person() -> None:
    bridge_module = importlib.import_module("src.pose_control.v6.condition_bridge")
    assert hasattr(bridge_module, "ReasonerControlBridge")
    assert not hasattr(bridge_module, "ConditionBundleBridge")
    bridge = bridge_module.ReasonerControlBridge(
        reasoning_dim=32, geometry_channels=16, token_dim=24
    )
    output = bridge(_make_state())
    assert output.scene_feature.shape == (2, 16, 4, 4)
    assert output.geometry_tokens.shape == (2, 2, 4, 24)
    assert output.geometry_token_mask.shape == (2, 2, 4)
    assert torch.count_nonzero(output.geometry_tokens[1, 1]) == 0
    assert not output.geometry_token_mask[1, 1].any()


def test_interaction_projection_has_no_bias_and_single_rows_ignore_interaction() -> None:
    bridge_module = importlib.import_module("src.pose_control.v6.condition_bridge")
    bridge = bridge_module.ReasonerControlBridge(
        reasoning_dim=32, geometry_channels=16, token_dim=24
    ).eval()
    assert bridge.interaction_projection.bias is None
    state = _make_state()
    first = bridge(state)
    changed = _make_state()
    changed.geometry_feature.copy_(state.geometry_feature)
    changed.person_tokens.copy_(state.person_tokens)
    changed.person_token_mask.copy_(state.person_token_mask)
    changed.interaction_feature[1].normal_(mean=100, std=20)
    second = bridge(changed)
    torch.testing.assert_close(first.scene_feature[1], second.scene_feature[1])


def test_mask_aware_pooling_ignores_invalid_spatial_tokens() -> None:
    bridge_module = importlib.import_module("src.pose_control.v6.condition_bridge")
    bridge = bridge_module.ReasonerControlBridge(
        reasoning_dim=32, geometry_channels=16, token_dim=24
    ).eval()
    state = _make_state(batch_size=1)
    state.person_token_mask[:, :, 1::2] = False
    first = bridge(state)
    changed = _make_state(batch_size=1)
    changed.geometry_feature.copy_(state.geometry_feature)
    changed.interaction_feature.copy_(state.interaction_feature)
    changed.person_token_mask.copy_(state.person_token_mask)
    changed.person_tokens.copy_(state.person_tokens)
    changed.person_tokens[:, :, 1::2].normal_(mean=100, std=10)
    second = bridge(changed)
    torch.testing.assert_close(first.geometry_tokens, second.geometry_tokens)
