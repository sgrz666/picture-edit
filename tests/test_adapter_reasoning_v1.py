from __future__ import annotations

import copy
import importlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from src.pose_control.v6.conditions import ConditionBundle


def _api():
    module = importlib.import_module("src.pose_control.v6.reasoning")
    return (
        module.AdapterReasoningConfig,
        module.InternalControlState,
        module.SMPLXAdapterReasoner,
    )


def _make_bundle(
    counts: tuple[int, ...] = (1, 2),
    *,
    size: int = 8,
    seed: int = 101,
    with_contact: bool = True,
) -> ConditionBundle:
    batch_size = len(counts)
    generator = torch.Generator().manual_seed(seed)
    person_valid = torch.tensor([[True, count == 2] for count in counts])

    mask_a = torch.zeros(batch_size, 1, size, size)
    mask_b = torch.zeros_like(mask_a)
    mask_a[..., 1:-1, 1 : size // 2 + 1] = 1
    mask_b[..., 1:-1, size // 2 - 1 : -1] = 1
    mask_b = mask_b * person_valid[:, 1, None, None, None]

    spatial_a = torch.randn(batch_size, 256, size, size, generator=generator) * mask_a
    spatial_b = torch.randn(batch_size, 256, size, size, generator=generator) * mask_b
    global_a = torch.randn(batch_size, 4, 256, generator=generator)
    global_b = torch.randn(batch_size, 4, 256, generator=generator)
    global_b = global_b * person_valid[:, 1, None, None]
    relative = torch.randn(batch_size, 2, 256, generator=generator)
    relative = relative * person_valid[:, 1, None, None]

    contact_spatial = torch.zeros(batch_size, 128, size, size)
    contact_tokens = torch.zeros(batch_size, 8, 256)
    contact_mask = torch.zeros(batch_size, 8, dtype=torch.bool)
    if with_contact:
        for row, count in enumerate(counts):
            if count == 2:
                contact_spatial[row, :, size // 2, size // 2] = 1
                contact_tokens[row, :2] = torch.randn(2, 256, generator=generator)
                contact_mask[row, :2] = True

    return ConditionBundle(
        person_a_spatial=spatial_a,
        person_b_spatial=spatial_b,
        person_a_mask=mask_a,
        person_b_mask=mask_b,
        person_a_global_tokens=global_a,
        person_b_global_tokens=global_b,
        relative_tokens=relative,
        contact_spatial=contact_spatial,
        contact_tokens=contact_tokens,
        contact_mask=contact_mask,
        task_token=torch.randn(batch_size, 1, 256, generator=generator),
        person_valid=person_valid,
        person_count=torch.tensor(counts, dtype=torch.long),
    )


def _build_reasoner(*, dropout: float = 0.0):
    AdapterReasoningConfig, _, SMPLXAdapterReasoner = _api()
    config = AdapterReasoningConfig(
        hidden_dim=32,
        global_attention_heads=4,
        cross_person_heads=4,
        contact_attention_heads=4,
        ffn_ratio=2,
        dropout=dropout,
    )
    return SMPLXAdapterReasoner(config).eval()


def test_reasoning_public_contract_is_available() -> None:
    assert importlib.util.find_spec("src.pose_control.v6.reasoning") is not None
    AdapterReasoningConfig, InternalControlState, SMPLXAdapterReasoner = _api()
    config = AdapterReasoningConfig()
    assert config.condition_dim == 256
    assert config.hidden_dim == 512
    assert config.reasoning_downsample_factor == 2
    assert config.global_attention_heads == 8
    assert config.cross_person_layers == 2
    assert config.contact_gate_init == -4.0
    assert InternalControlState is not None
    assert SMPLXAdapterReasoner is not None


@pytest.mark.parametrize(
    ("counts", "size", "low_size"),
    [((1,), 8, 4), ((2,), 8, 4), ((1, 2, 1), 4, 2)],
)
def test_single_dual_and_mixed_outputs_have_stable_shapes(
    counts: tuple[int, ...], size: int, low_size: int
) -> None:
    reasoner = _build_reasoner()
    state = reasoner(_make_bundle(counts, size=size))
    batch_size = len(counts)
    assert state.geometry_feature.shape == (batch_size, 32, low_size, low_size)
    assert state.geometry_highres.shape == (batch_size, 32, size, size)
    assert state.interaction_feature.shape == (batch_size, 32, low_size, low_size)
    assert state.interaction_highres.shape == (batch_size, 32, size, size)
    assert state.person_tokens.shape == (batch_size, 2, low_size * low_size, 32)
    assert state.person_token_mask.shape == (batch_size, 2, low_size * low_size)
    assert state.person_valid.tolist() == [[True, count == 2] for count in counts]
    assert state.person_count.tolist() == list(counts)
    assert state.interaction_valid.tolist() == [count == 2 for count in counts]
    assert state.validate() is state

    for row, count in enumerate(counts):
        if count == 1:
            assert torch.count_nonzero(state.person_tokens[row, 1]) == 0
            assert not state.person_token_mask[row, 1].any()
            assert torch.count_nonzero(state.interaction_feature[row]) == 0
            assert torch.count_nonzero(state.interaction_highres[row]) == 0


def test_reasoner_rejects_odd_condition_resolution() -> None:
    reasoner = _build_reasoner()
    with pytest.raises(ValueError, match="even"):
        reasoner(_make_odd_bundle())


def _make_odd_bundle() -> ConditionBundle:
    bundle = _make_bundle((1,), size=8)
    spatial_fields = (
        "person_a_spatial",
        "person_b_spatial",
        "person_a_mask",
        "person_b_mask",
        "contact_spatial",
    )
    for name in spatial_fields:
        value = getattr(bundle, name)
        setattr(bundle, name, value[..., :7, :7])
    return bundle


def test_reasoner_masks_background_and_rejects_empty_valid_person() -> None:
    torch.manual_seed(3)
    reasoner = _build_reasoner()
    bundle = _make_bundle((1,))
    changed = copy.deepcopy(bundle)
    background = ~bundle.person_a_mask.bool()
    changed.person_a_spatial = torch.where(
        background.expand_as(changed.person_a_spatial),
        torch.randn_like(changed.person_a_spatial) * 100,
        changed.person_a_spatial,
    )
    first = reasoner(bundle)
    second = reasoner(changed)
    torch.testing.assert_close(first.geometry_feature, second.geometry_feature)
    torch.testing.assert_close(first.person_tokens, second.person_tokens)

    invalid = copy.deepcopy(bundle)
    invalid.person_a_mask.zero_()
    with pytest.raises(ValueError, match="empty mask"):
        reasoner(invalid)


def test_shared_person_reasoner_uses_global_task_and_role_binding() -> None:
    torch.manual_seed(5)
    reasoner = _build_reasoner()
    assert hasattr(reasoner, "person_reasoner")
    assert not hasattr(reasoner, "person_a_reasoner")
    assert not hasattr(reasoner, "person_b_reasoner")

    bundle = _make_bundle((2,))
    bundle.person_b_spatial.copy_(bundle.person_a_spatial)
    bundle.person_b_mask.copy_(bundle.person_a_mask)
    bundle.person_b_global_tokens.copy_(bundle.person_a_global_tokens)
    state = reasoner(bundle)
    assert not torch.allclose(state.person_tokens[:, 0], state.person_tokens[:, 1])

    changed_global = copy.deepcopy(bundle)
    changed_global.person_a_global_tokens.add_(2)
    changed_task = copy.deepcopy(bundle)
    changed_task.task_token.sub_(3)
    assert not torch.allclose(state.person_tokens, reasoner(changed_global).person_tokens)
    assert not torch.allclose(state.person_tokens, reasoner(changed_task).person_tokens)


def test_dual_person_and_relative_changes_do_not_affect_single_rows() -> None:
    torch.manual_seed(7)
    reasoner = _build_reasoner()
    bundle = _make_bundle((1, 2))
    changed = copy.deepcopy(bundle)
    changed.person_b_spatial.add_(50)
    changed.person_b_global_tokens.sub_(20)
    changed.relative_tokens.mul_(-4)
    first = reasoner(bundle)
    second = reasoner(changed)
    torch.testing.assert_close(first.geometry_feature[0], second.geometry_feature[0])
    torch.testing.assert_close(first.person_tokens[0], second.person_tokens[0])
    assert not torch.allclose(first.geometry_feature[1], second.geometry_feature[1])
    assert not torch.allclose(first.person_tokens[1], second.person_tokens[1])


def test_contact_gate_and_relation_mask_behavior() -> None:
    torch.manual_seed(11)
    reasoner = _build_reasoner()
    assert torch.allclose(
        reasoner.contact_reasoner.contact_gate.sigmoid(),
        torch.full((2,), torch.sigmoid(torch.tensor(-4.0))),
    )
    empty = _make_bundle((2,), with_contact=False)
    padded = copy.deepcopy(empty)
    padded.contact_tokens.normal_(mean=100, std=10)
    first = reasoner(empty)
    second = reasoner(padded)
    torch.testing.assert_close(first.geometry_feature, second.geometry_feature)
    torch.testing.assert_close(first.interaction_feature, second.interaction_feature)
    assert torch.isfinite(first.interaction_feature).all()

    active = _make_bundle((2,), with_contact=True)
    active_state = reasoner(active)
    assert not torch.allclose(first.interaction_feature, active_state.interaction_feature)


def test_relation_padding_does_not_couple_batch_rows() -> None:
    torch.manual_seed(13)
    reasoner = _build_reasoner()
    bundle = _make_bundle((2, 2))
    bundle.contact_mask[0, 1:] = False
    bundle.contact_tokens[0, 1:].normal_(mean=200, std=20)
    batched = reasoner(bundle).index_select(torch.tensor([0]))
    standalone = reasoner(bundle.index_select(torch.tensor([0])))
    torch.testing.assert_close(
        batched.geometry_feature, standalone.geometry_feature, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        batched.interaction_feature, standalone.interaction_feature, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        batched.person_tokens, standalone.person_tokens, atol=1e-5, rtol=1e-5
    )


def test_internal_control_state_to_and_index_select_preserve_discrete_dtypes() -> None:
    state = _build_reasoner()(_make_bundle((1, 2)))
    converted = state.to(torch.float16)
    assert converted.geometry_feature.dtype == torch.float16
    assert converted.person_tokens.dtype == torch.float16
    assert converted.person_token_mask.dtype == torch.bool
    assert converted.person_valid.dtype == torch.bool
    assert converted.person_count.dtype == torch.long
    assert converted.interaction_valid.dtype == torch.bool
    selected = converted.index_select(torch.tensor([1]))
    assert selected.geometry_feature.shape[0] == 1
    assert selected.person_count.tolist() == [2]
    assert selected.validate() is selected


def test_reasoning_smoke_and_v62_config_are_declared() -> None:
    from scripts.smoke_adapter_reasoner import default_report_name

    assert default_report_name("single") == "adapter_reasoner_smoke_single.json"
    assert default_report_name("dual") == "adapter_reasoner_smoke_dual.json"
    assert default_report_name("mixed") == "adapter_reasoner_smoke_mixed.json"
    config_path = Path(__file__).parents[1] / "configs" / "adapter_v6_architecture.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["version"] == "6.2-adapter-reasoning-v1"
    assert payload["reasoning"]["hidden_dim"] == 512
    assert payload["reasoning"]["downsample_factor"] == 2
    assert payload["reasoning"]["cross_person_layers"] == 2
    assert payload["reasoning"]["contact_gate_init"] == -4.0
