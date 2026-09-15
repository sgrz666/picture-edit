from __future__ import annotations

import importlib
import importlib.util

import torch


def _make_bundle(batch_size: int = 2):
    conditions = importlib.import_module("src.pose_control.v6.conditions")
    return conditions.ConditionBundle(
        person_a_spatial=torch.randn(batch_size, 256, 4, 4),
        person_b_spatial=torch.randn(batch_size, 256, 4, 4),
        person_a_mask=torch.ones(batch_size, 1, 4, 4),
        person_b_mask=torch.ones(batch_size, 1, 4, 4),
        person_a_global_tokens=torch.randn(batch_size, 4, 256),
        person_b_global_tokens=torch.randn(batch_size, 4, 256),
        relative_tokens=torch.randn(batch_size, 2, 256),
        contact_spatial=torch.randn(batch_size, 128, 4, 4),
        contact_tokens=torch.randn(batch_size, 8, 256),
        contact_mask=torch.tensor([[True] * 8, [False] * 8]),
        task_token=torch.randn(batch_size, 1, 256),
        person_valid=torch.tensor([[True, True], [True, False]]),
        person_count=torch.tensor([2, 1]),
    )


def test_condition_bundle_bridge_module_is_available() -> None:
    assert importlib.util.find_spec("src.pose_control.v6.condition_bridge") is not None
    conditions = importlib.import_module("src.pose_control.v6.conditions")
    assert hasattr(conditions, "AdapterIdentityCondition")


def test_bridge_projects_bundle_without_reopening_raw_conditions() -> None:
    bridge_module = importlib.import_module("src.pose_control.v6.condition_bridge")
    bundle = _make_bundle()
    bridge = bridge_module.ConditionBundleBridge(geometry_channels=16, token_dim=32)
    output = bridge(bundle)

    assert output.person_features.shape == (2, 2, 16, 4, 4)
    assert output.person_masks.shape == (2, 2, 1, 4, 4)
    assert output.global_tokens.shape == (2, 2, 4, 32)
    assert output.relative_tokens.shape == (2, 2, 32)
    assert output.contact_spatial.shape == (2, 16, 4, 4)
    assert output.contact_tokens.shape == (2, 8, 32)
    assert output.task_token.shape == (2, 1, 32)
    assert torch.count_nonzero(output.person_features[1, 1]) == 0
    assert torch.count_nonzero(output.relative_tokens[1]) == 0
    assert torch.count_nonzero(output.contact_tokens[1]) == 0


def test_identity_condition_is_separate_and_validated_against_bundle() -> None:
    conditions = importlib.import_module("src.pose_control.v6.conditions")
    bundle = _make_bundle()
    identity = conditions.AdapterIdentityCondition(
        source_person_latents=torch.randn(2, 2, 16, 8, 8),
        source_indices=torch.tensor([[0, 1], [0, 1]]),
    )
    assert identity.validate(bundle) is identity
    selected = identity.index_select(torch.tensor([1]))
    assert selected.source_person_latents.shape == (1, 2, 16, 8, 8)
    assert selected.source_indices.shape == (1, 2)
