"""消融只改变模型条件，sealed 输入、监督和共享初始化保持原值。"""
from dataclasses import replace
import importlib

import pytest
import torch

from test_qwen21_pro_contracts import make_condition
from src.pose_control.qwen21_pro.adapter import ProTaskAdapter
from src.pose_control.qwen21_pro.cache import _payload_digest
from src.pose_control.qwen21_pro.contracts import ProAdapterConfig


def api():
    try:
        return importlib.import_module("src.pose_control.qwen21_pro.ablations")
    except ModuleNotFoundError:
        pytest.fail("Pro retrained ablation implementation is missing")


def config(stages=(0, 4, 8, 12)):
    return ProAdapterConfig(hidden_dim=16, width=8, heads=2, stage_indices=stages)


@pytest.mark.parametrize("arm,disabled", [("no_dense", {"normal", "depth", "part"}),
                                         ("no_contact", {"contact"}), ("no_appearance", set())])
def test_forward_masks_only_requested_conditions_without_mutating_sealed_pack(arm, disabled):
    module = api()
    adapter = module.AblatedTaskAdapter(config(), arm=arm)
    pack = make_condition()
    original = _payload_digest(pack.to_dict())
    observed = []
    handle = adapter.geometry_encoder.register_forward_pre_hook(lambda _, args: observed.append(args[0]))
    adapter.forward_layer(0, torch.randn(1, 6, 16), torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.bool), pack, torch.ones(1))
    handle.remove()
    transformed = observed[0]
    assert _payload_digest(pack.to_dict()) == original
    for key in disabled:
        assert not transformed.spatial_valid[key].any()
        assert torch.count_nonzero(transformed.spatial[key]) == 0
    for key in {"mask", "visibility"}:
        assert torch.equal(transformed.spatial[key], pack.spatial[key])
        assert torch.equal(transformed.spatial_valid[key], pack.spatial_valid[key])
    assert torch.equal(transformed.owner, pack.owner)
    assert torch.equal(transformed.face_mask, pack.face_mask)
    assert torch.equal(transformed.hand_mask, pack.hand_mask)
    if arm == "no_appearance":
        assert not transformed.source_token_valid.any()
        assert not transformed.face_identity_valid.any()
    else:
        assert torch.equal(transformed.source_token_valid, pack.source_token_valid)
    transformed.validate(config())


def test_full_is_numerically_equal_and_upgrade_preserves_parameter_objects():
    module = api()
    model = torch.nn.Module()
    model.task_adapter = ProTaskAdapter(config())
    ids = {name: id(value) for name, value in model.named_parameters()}
    plain = ProTaskAdapter(config())
    module.configure_ablation(model, arm="full4", seed=3407)
    plain.load_state_dict({k: v for k, v in model.task_adapter.state_dict().items() if k != "_extra_state"}
                          | {"_extra_state": plain.get_extra_state()})
    with torch.no_grad():
        for name in model.task_adapter.output_heads:
            model.task_adapter.output_heads[name].weight.fill_(.1)
            plain.output_heads[name].weight.fill_(.1)
    args = (0, torch.randn(1, 4, 16), torch.ones(4, dtype=torch.bool), make_condition(), torch.ones(1))
    assert torch.equal(model.task_adapter.forward_layer(*args), plain.forward_layer(*args))
    assert ids == {name: id(value) for name, value in model.named_parameters()}


def test_arms_share_every_common_initial_tensor_and_extra_heads_are_zero():
    module = api()
    inventories = {}
    for arm in module.ARMS:
        cfg = replace(config(), stage_indices=module.arm_stages(arm))
        model = torch.nn.Module()
        model.task_adapter = ProTaskAdapter(cfg)
        module.configure_ablation(model, arm=arm, seed=3407)
        inventories[arm] = module.tensor_inventory(model.task_adapter)
        for head in model.task_adapter.output_heads.values():
            assert torch.count_nonzero(head.weight) == torch.count_nonzero(head.bias) == 0
    full = inventories["full4"]
    for arm, current in inventories.items():
        assert all(current[name] == value for name, value in full.items())


def test_wrong_arm_checkpoint_and_runtime_mutation_are_rejected():
    module = api()
    full = module.AblatedTaskAdapter(config(), arm="full4")
    other = module.AblatedTaskAdapter(config(), arm="no_dense")
    with pytest.raises(ValueError, match="arm"):
        other.load_state_dict(full.state_dict())
    with pytest.raises((AttributeError, ValueError), match="immutable|arm"):
        other.arm = "full4"
    with pytest.raises(ValueError, match="stage"):
        module.AblatedTaskAdapter(config(), arm="full8")
