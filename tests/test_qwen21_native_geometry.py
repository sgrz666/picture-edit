from __future__ import annotations

import copy
import importlib
import importlib.util
from dataclasses import replace

import pytest
import torch

from src.pose_control.qwen21_full.deepgen_adapter import UnifiedSMPLXAdapterV6
from src.pose_control.qwen21_full.qwen_adapter import Qwen21FullAdapter
from src.pose_control.qwen21_full.reasoning import AdapterReasoningConfig


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def api():
    name = "src.pose_control.qwen21_full.native_geometry"
    assert importlib.util.find_spec(name) is not None, "native geometry endpoint is missing"
    return importlib.import_module(name).Qwen21NativeGeometry


def full_model(*, depth=False):
    torch.manual_seed(17)
    legacy = UnifiedSMPLXAdapterV6.build_tiny(condition_use_depth=depth)
    return Qwen21FullAdapter(legacy, qwen_hidden_size=16, qwen_text_dim=24, pooled_input_dim=20).eval()


def raw_inputs(*, dual=False, width=64):
    generator = torch.Generator().manual_seed(31)
    values = dict(
        normal_a=torch.rand(1, 3, 32, width, generator=generator) * 2 - 1,
        pose_heatmap_a=torch.rand(1, 25, 32, width, generator=generator),
        part_onehot_a=torch.zeros(1, 14, 32, width),
        human_mask_a=torch.ones(1, 1, 32, width),
        smplx_global_a=torch.randn(1, 26, generator=generator),
        task_id=torch.tensor([int(dual)]),
    )
    values["part_onehot_a"][:, 1] = 1
    if dual:
        for key in ("normal", "pose_heatmap", "part_onehot", "human_mask", "smplx_global"):
            values[f"{key}_b"] = values[f"{key}_a"].clone()
        values["normal_b"].neg_()
    return values


def payload(full):
    return {"architecture": "qwen21_full_v1", "adapter_state_dict": full.state_dict()}


def assert_legacy_projection_parity(full, state, features):
    captured = {}
    handles = []
    for name in ("geometry", "interaction"):
        def capture(_module, args, key=name):
            captured[key] = args[0]
        handles.append(getattr(full.legacy.condition_bridge, name + "_projection").register_forward_pre_hook(capture))
    try:
        old = full.legacy.condition_bridge(state)
    finally:
        for handle in handles:
            handle.remove()
    # Match NCHW storage before comparing Conv2d results: channels-last kernels
    # can round differently even when the unprojected features are bitwise equal.
    geometry = features.geometry.transpose(1, 2).reshape_as(state.geometry_feature).contiguous()
    interaction = features.interaction.transpose(1, 2).reshape_as(state.interaction_feature).contiguous()
    torch.testing.assert_close(geometry, captured["geometry"], rtol=0, atol=0)
    torch.testing.assert_close(interaction, captured["interaction"], rtol=0, atol=0)
    torch.testing.assert_close(full.legacy.condition_bridge.geometry_projection(geometry), old.geometry_scene, rtol=0, atol=0)
    torch.testing.assert_close(full.legacy.condition_bridge.interaction_projection(interaction), old.interaction_scene, rtol=0, atol=0)


@pytest.mark.parametrize("dual,width", [(False, 32), (False, 64), (True, 64)])
def test_frontend_reuses_learned_features_before_old_projection(dual, width):
    full = full_model()
    native = api().from_full_adapter(full).eval()
    inputs = raw_inputs(dual=dual, width=width)
    with torch.no_grad():
        bundle = full.condition_injector(**inputs)
        state = full.legacy.reasoner(bundle)
        features = native(native.condition_injector(**inputs), target_token_hw=(2, width // 16))
    assert features.geometry.shape == (1, width // 8, 32)
    assert features.target_token_hw == (2, width // 16)
    assert torch.equal(features.person_valid, bundle.person_valid)
    assert_legacy_projection_parity(full, state, features)
    if not dual:
        assert torch.count_nonzero(features.interaction) == 0


def test_copied_frontend_is_independent_and_has_no_control_core():
    full = full_model()
    native = api().from_full_adapter(full)
    old_weight = full.condition_injector.spatial_encoder.normal_stem.stages[0].weight
    new_weight = native.condition_injector.spatial_encoder.normal_stem.stages[0].weight
    before = old_weight.detach().clone()
    with torch.no_grad():
        new_weight.add_(1)
    torch.testing.assert_close(old_weight, before, rtol=0, atol=0)
    assert old_weight.data_ptr() != new_weight.data_ptr()
    assert not hasattr(native, "legacy")
    assert not hasattr(native, "control_core")
    assert not any("1536" in name or "control_core" in name or "output_bridge" in name for name, _ in native.named_parameters())


def test_all_four_feature_paths_receive_gradients():
    full = full_model()
    native = api().from_full_adapter(full)
    state = full.legacy.reasoner(full.condition_injector(**raw_inputs(dual=True)))
    names = ("geometry_feature", "geometry_highres", "interaction_feature", "interaction_highres")
    leaves = {name: getattr(state, name).detach().clone().requires_grad_() for name in names}
    state = replace(state, **leaves)
    features = native.project_state(state, target_token_hw=(2, 4))
    (features.geometry.square().mean() + features.interaction.square().mean()).backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0 for value in leaves.values())


def test_raw_conditions_still_train_the_existing_encoder_and_reasoner():
    native = api().from_full_adapter(full_model()).train()
    features = native(native.condition_injector(**raw_inputs()), target_token_hw=(2, 4))
    features.geometry.square().mean().backward()
    for module in (native.condition_injector.spatial_encoder.pose_stem, native.reasoner.person_reasoner, native.geometry_high_downsample):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


@pytest.mark.parametrize("hw", [(4, 2), (2, 3), (0, 4), (True, 4), (2,), (2.0, 4)])
def test_grid_errors_never_trigger_silent_resize(hw):
    native = api().from_full_adapter(full_model())
    bundle = native.condition_injector(**raw_inputs())
    with pytest.raises(ValueError, match="grid|target_token_hw"):
        native(bundle, target_token_hw=hw)


def test_nonfinite_features_are_rejected():
    native = api().from_full_adapter(full_model())
    state = native.reasoner(native.condition_injector(**raw_inputs()))
    state.geometry_feature[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        native.project_state(state, target_token_hw=(2, 4))


def test_checkpoint_extraction_is_complete_and_does_not_instantiate_legacy():
    full = full_model()
    native = api().from_full_checkpoint(payload(full), reasoning_config=full.legacy.reasoner.config)
    inputs = raw_inputs()
    native.eval()
    expected = api().from_full_adapter(full).eval()
    actual_features = native(native.condition_injector(**inputs), target_token_hw=(2, 4))
    expected_features = expected(expected.condition_injector(**inputs), target_token_hw=(2, 4))
    torch.testing.assert_close(actual_features.geometry, expected_features.geometry, rtol=0, atol=0)
    assert native.inheritance_report["loaded_tensor_count"] > 0
    assert native.inheritance_report["ignored_key_count"] > 0
    assert native.inheritance_report["source_architecture"] == "qwen21_full_v1"


@pytest.mark.parametrize("mutation", ["missing", "extra", "shape", "nan", "architecture", "depth"])
def test_checkpoint_incompatibility_is_explicit(mutation):
    full = full_model(depth=mutation == "depth")
    value = payload(full)
    state = dict(value["adapter_state_dict"])
    key = "legacy.condition_bridge.geometry_high_downsample.weight"
    if mutation == "missing":
        state.pop(key)
    elif mutation == "extra":
        state["legacy.reasoner.unexpected.weight"] = torch.zeros(1)
    elif mutation == "shape":
        state[key] = state[key][:1]
    elif mutation == "nan":
        state[key] = state[key].clone().fill_(float("nan"))
    elif mutation == "architecture":
        value["architecture"] = "qwen21_union_v1"
    value["adapter_state_dict"] = state
    with pytest.raises(ValueError):
        api().from_full_checkpoint(value, reasoning_config=full.legacy.reasoner.config)


def test_depth_enabled_full_is_not_silently_migrated():
    with pytest.raises(ValueError, match="depth|Depth"):
        api().from_full_adapter(full_model(depth=True))


def test_native_state_has_version_and_rejects_a_different_configuration():
    model = api().from_full_adapter(full_model())
    state = copy.deepcopy(model.state_dict())
    assert state["_extra_state"]["schema"] == "qwen21_full_native_geometry_v1"
    clone = api()(reasoning_config=model.reasoner.config)
    clone.load_state_dict(state, strict=True)
    state["_extra_state"]["reasoning_config"]["dropout"] = .5
    with pytest.raises(ValueError, match="config"):
        clone.load_state_dict(state, strict=True)


def test_default_width_matches_qwen_geometry_contract():
    with torch.device("meta"):
        model = api()()
    assert model.reasoner.config == AdapterReasoningConfig()
    assert model.geometry_high_downsample.out_channels == 512
    assert model.condition_injector.spatial_encoder.fusion[0].in_channels == 384


def test_checkpoint_import_really_materializes_cpu_weights_inside_meta_context():
    full = full_model()
    constructor = api()
    with torch.device("meta"):
        native = constructor.from_full_checkpoint(payload(full), reasoning_config=full.legacy.reasoner.config)
    assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in native.parameters())
    torch.testing.assert_close(native.geometry_high_downsample.weight, full.legacy.condition_bridge.geometry_high_downsample.weight)


def test_mixed_batch_keeps_interaction_on_dual_rows_only():
    full = full_model()
    native = api().from_full_adapter(full).eval()
    values = {key: torch.cat((value, value), dim=0) for key, value in raw_inputs(dual=True).items()}
    values["person_b_valid"] = torch.tensor([True, False])
    values["task_id"] = torch.tensor([1, 0])
    bundle = full.condition_injector(**values)
    state = full.legacy.reasoner(bundle)
    features = native(bundle, target_token_hw=(2, 4))
    assert features.person_valid.tolist() == [[True, True], [True, False]]
    assert torch.count_nonzero(features.interaction[1]) == 0
    assert torch.count_nonzero(features.interaction[0]) > 0
    assert_legacy_projection_parity(full, state, features)


def audit_api():
    name = "scripts.audit.audit_qwen21_native_geometry"
    assert importlib.util.find_spec(name) is not None, "native geometry audit is missing"
    return importlib.import_module(name)


def test_audit_checks_checkpoint_read_only_and_refuses_output_overwrite(tmp_path):
    full = full_model()
    checkpoint = tmp_path / "full.pt"
    torch.save(payload(full), checkpoint)
    before = checkpoint.read_bytes()
    report = audit_api().audit_checkpoint(checkpoint, reasoning_config=full.legacy.reasoner.config)
    assert report["passed"]
    assert len(report["cases"]) == 3
    assert all(case["geometry_max_abs"] == 0 and case["interaction_max_abs"] == 0 for case in report["cases"])
    assert all(case["geometry_fused_max_abs"] == 0 and case["interaction_fused_max_abs"] == 0 for case in report["cases"])
    assert checkpoint.read_bytes() == before
    output = tmp_path / "audit.json"
    audit_api().write_report(report, output)
    with pytest.raises(FileExistsError):
        audit_api().write_report(report, output)
