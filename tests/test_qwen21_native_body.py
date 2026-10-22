from __future__ import annotations

import copy
import importlib
import importlib.util
from functools import partial

import pytest
import torch
from torch import nn

from src.pose_control.qwen21_full.native_geometry import NativeGeometryFeatures


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def body_api():
    name = "src.pose_control.qwen21_full.native_body"
    assert importlib.util.find_spec(name) is not None, "native body module is missing"
    return importlib.import_module(name)


def integration_api():
    name = "src.pose_control.qwen21_full.native_integration"
    assert importlib.util.find_spec(name) is not None, "native body integration is missing"
    return importlib.import_module(name)


def features(seed=21, *, dim=32, requires_grad=False, hw=(2, 2)):
    generator = torch.Generator().manual_seed(seed)
    n = hw[0] * hw[1]
    return NativeGeometryFeatures(
        torch.randn(1, n, dim, generator=generator).requires_grad_(requires_grad),
        torch.zeros(1, n, dim), hw, torch.tensor([[True, False]]),
    )


def adapter():
    module = body_api()
    torch.manual_seed(10)
    return module.Qwen21NativeBody(module.NativeBodyConfig(hidden_dim=16, geometry_dim=32, width=8))


def nonzero_heads(model):
    with torch.no_grad():
        for head in model.output_heads:
            nn.init.normal_(head.weight, std=.04)


def test_zero_output_is_exact_including_signed_zero_and_keeps_prefix():
    model = adapter()
    hidden = torch.randn(1, 7, 16)
    hidden[0, 4, 0] = -0.0
    mask = torch.tensor([False, False, False, True, True, True, True])
    output = model(hidden, prepared=model.prepare_geometry(features()), layer=0, sigma=torch.tensor([.5]), target_mask=mask)
    torch.testing.assert_close(output, hidden, rtol=0, atol=0)
    assert torch.equal(torch.signbit(output), torch.signbit(hidden))


def test_head_learns_first_then_query_geometry_and_frontend_receive_gradients():
    model = adapter()
    optimizer = torch.optim.SGD(model.parameters(), lr=.2)
    hidden = torch.randn(1, 4, 16)
    mask = torch.ones(4, dtype=torch.bool)
    value = features(requires_grad=True)
    output = model(hidden, prepared=model.prepare_geometry(value), layer=0, sigma=torch.tensor([.4]), target_mask=mask)
    (output - 1).square().mean().backward()
    assert model.output_heads[0].weight.grad.abs().sum() > 0
    assert model.query_projection.weight.grad is None or model.query_projection.weight.grad.count_nonzero() == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    value = features(requires_grad=True)
    output = model(hidden, prepared=model.prepare_geometry(value), layer=0, sigma=torch.tensor([.4]), target_mask=mask)
    (output - 1).square().mean().backward()
    assert model.query_projection.weight.grad.abs().sum() > 0
    assert model.geometry_projection.weight.grad.abs().sum() > 0
    assert value.geometry.grad.abs().sum() > 0


def test_current_hidden_geometry_layer_and_sigma_are_independent_inputs():
    model = adapter()
    nonzero_heads(model)
    hidden = torch.randn(1, 4, 16)
    mask = torch.ones(4, dtype=torch.bool)
    def residual(h, f, layer=0, sigma=.5):
        return model(h, prepared=model.prepare_geometry(f), layer=layer, sigma=torch.tensor([sigma]), target_mask=mask) - h
    original = residual(hidden, features())
    assert not torch.allclose(original, residual(hidden.flip(-1), features()))
    assert not torch.allclose(original, residual(hidden, features(seed=22)))
    assert not torch.allclose(original, residual(hidden, features(), layer=1))
    assert not torch.allclose(original, residual(hidden, features(), sigma=.2))


def test_canvas_mask_suppresses_only_requested_target_updates():
    model = adapter()
    nonzero_heads(model)
    hidden = torch.randn(1, 6, 16)
    mask = torch.tensor([False, False, True, True, True, True])
    valid = torch.tensor([[True, False, True, False]])
    output = model(hidden, prepared=model.prepare_geometry(features()), layer=4, sigma=torch.tensor([.5]), target_mask=mask, valid_mask=valid)
    torch.testing.assert_close(output[:, :2], hidden[:, :2], rtol=0, atol=0)
    torch.testing.assert_close(output[:, [3, 5]], hidden[:, [3, 5]], rtol=0, atol=0)
    assert not torch.equal(output[:, [2, 4]], hidden[:, [2, 4]])


def test_bf16_hidden_keeps_trainable_branch_fp32():
    model = adapter()
    hidden = torch.randn(1, 4, 16).bfloat16()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(hidden, prepared=model.prepare_geometry(features()), layer=0, sigma=torch.tensor([.5]), target_mask=torch.ones(4, dtype=torch.bool))
    assert output.dtype == torch.bfloat16
    assert all(p.dtype == torch.float32 for p in model.parameters())
    model.bfloat16()
    with pytest.raises(ValueError, match="FP32"):
        model.prepare_geometry(features())


@pytest.mark.parametrize("options", [{"hidden_dim": 0}, {"width": True}, {"num_groups": 33}, {"injection_layers": (0, 0)}, {"injection_layers": (32,)}])
def test_bad_architecture_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        body_api().NativeBodyConfig(**options)


def qwen(seed=4):
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel
    torch.manual_seed(seed)
    return QwenImage21Transformer2DModel(
        in_channels=64, out_channels=64, num_layers=32, num_attention_heads=2,
        attention_head_dim=8, context_in_dim=8, axes_dims_rope=(2, 2, 4), mlp_ratio=2,
    ).eval()


def inputs(text_length=3, references=1, target_hw=(2, 2)):
    # Actual VLM slots: each reference reserves one slot -> four image tokens.
    text_length = max(text_length, references)
    target_tokens = target_hw[0] * target_hw[1]
    img_mask = torch.zeros(1, text_length + target_tokens // 4, dtype=torch.bool)
    img_mask[:, :references] = True
    img_mask[:, -target_tokens // 4:] = True
    return dict(
        hidden_states=torch.randn(1, 4 * references + target_tokens, 64),
        encoder_hidden_states=torch.randn(1, text_length, 8),
        encoder_hidden_states_mask=torch.ones(1, text_length, dtype=torch.bool),
        timestep=torch.tensor([.5]), img_shapes=[[(1, 2, 2)] * references + [(1, *target_hw)]],
        img_mask=img_mask, return_dict=False,
    )


def installed(seed=4):
    model = qwen(seed)
    integration_api().install_native_body(model, adapter())
    return model


@pytest.mark.parametrize("references", [0, 1, 2])
def test_real_qwen_blocks_preserve_zero_and_disabled_outputs(references):
    base = qwen()
    data = inputs(references=references)
    with torch.no_grad():
        expected = base(**data)[0]
        integration_api().install_native_body(base, adapter())
        zero = base(**data, native_geometry=features())[0]
        disabled = base(**data, native_enabled=False)[0]
    torch.testing.assert_close(zero, expected, rtol=0, atol=0)
    torch.testing.assert_close(disabled, expected, rtol=0, atol=0)


def test_nonzero_real_qwen_control_changes_target_and_preserves_causal_prefix():
    model = installed()
    data = inputs()
    with torch.no_grad():
        expected = model(**data, native_enabled=False)[0]
        nonzero_heads(model.native_body)
        actual = model(**data, native_geometry=features())[0]
    torch.testing.assert_close(actual[:, :-4], expected[:, :-4], rtol=0, atol=0)
    assert not torch.equal(actual[:, -4:], expected[:, -4:])


def test_install_preserves_original_parameter_objects_keys_and_freezes_them():
    model = qwen()
    before = dict(model.named_parameters())
    keys = set(model.state_dict())
    report = integration_api().install_native_body(model, adapter())
    after = dict(model.named_parameters())
    assert keys <= set(model.state_dict())
    assert all(after[key] is value and not value.requires_grad for key, value in before.items())
    assert report["injection_layers"] == list(range(32))
    assert all(p.requires_grad for p in model.native_body.parameters())
    with pytest.raises(ValueError, match="installed"):
        integration_api().install_native_body(model, adapter())


def test_checkpoint_recompute_after_two_forwards_uses_the_correct_conditions():
    reference, checked = installed(), installed()
    nonzero_heads(reference.native_body)
    checked.native_body.load_state_dict(reference.native_body.state_dict())
    checked.enable_gradient_checkpointing()
    original_checkpoint = checked._gradient_checkpointing_func
    data_a, data_b = inputs(3), inputs(5, references=2, target_hw=(2, 4))
    data_b["timestep"] = torch.tensor([.2])
    a_ref, b_ref = features(41, requires_grad=True), features(42, requires_grad=True, hw=(2, 4))
    a_chk, b_chk = features(41, requires_grad=True), features(42, requires_grad=True, hw=(2, 4))
    outputs_ref = [reference(**data_a, native_geometry=a_ref)[0], reference(**data_b, native_geometry=b_ref)[0]]
    outputs_chk = [checked(**data_a, native_geometry=a_chk)[0], checked(**data_b, native_geometry=b_chk)[0]]
    assert checked._gradient_checkpointing_func is original_checkpoint
    assert checked._native_body_execution.get() is None
    for expected, actual in zip(outputs_ref, outputs_chk):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    sum(output[:, -value.geometry.shape[1]:].square().mean() for output, value in zip(outputs_ref, (a_ref, b_ref))).backward()
    sum(output[:, -value.geometry.shape[1]:].square().mean() for output, value in zip(outputs_chk, (a_chk, b_chk))).backward()
    for (name, expected), (_, actual) in zip(reference.native_body.named_parameters(), checked.native_body.named_parameters()):
        if expected.grad is None:
            assert actual.grad is None, name
        else:
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-7, msg=name)
    torch.testing.assert_close(a_chk.geometry.grad, a_ref.geometry.grad, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(b_chk.geometry.grad, b_ref.geometry.grad, rtol=1e-5, atol=1e-7)
    assert all(p.grad is None for name, p in checked.named_parameters() if not name.startswith("native_body."))
    assert all(head.weight.grad is not None and head.weight.grad.abs().sum() > 0 for head in checked.native_body.output_heads)


def test_runtime_rejects_missing_conditions_bad_layout_and_cache_without_leaking_state():
    model = installed()
    data = inputs()
    with pytest.raises(ValueError, match="geometry"):
        model(**data)
    bad = dict(data, img_mask=torch.zeros_like(data["img_mask"]))
    with pytest.raises(ValueError, match="mask|slot"):
        model(**bad, native_geometry=features())
    with pytest.raises(ValueError, match="cache"):
        model(**dict(data, kv_cache=object(), kv_cache_mode="cached"), native_geometry=features())
    assert model._native_body_execution.get() is None
    model(**data, native_geometry=features())


def test_reentrant_custom_checkpointing_is_rejected_before_it_silently_loses_gradients():
    from torch.utils.checkpoint import checkpoint
    model = installed()
    custom = partial(checkpoint, use_reentrant=True)
    model.enable_gradient_checkpointing(gradient_checkpointing_func=custom)
    with pytest.raises(ValueError, match="non-reentrant|standard"):
        model(**inputs(), native_geometry=features())
    assert model._gradient_checkpointing_func is custom
    assert model._native_body_execution.get() is None


def test_real_bf16_qwen_has_exact_zero_equivalence_with_fp32_adapter():
    model = qwen().bfloat16()
    data = inputs()
    data["hidden_states"] = data["hidden_states"].bfloat16()
    data["encoder_hidden_states"] = data["encoder_hidden_states"].bfloat16()
    with torch.no_grad():
        expected = model(**data)[0]
        integration_api().install_native_body(model, adapter())
        actual = model(**data, native_geometry=features())[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(p.dtype == torch.float32 for p in model.native_body.parameters())


def test_failure_inside_a_block_restores_checkpoint_helper_and_execution(monkeypatch):
    model = installed()
    model.enable_gradient_checkpointing()
    checkpoint = model._gradient_checkpointing_func
    def fail(*args, **kwargs):
        raise RuntimeError("deliberate block failure")
    with monkeypatch.context() as patch:
        patch.setattr(model.native_body, "forward", fail)
        with pytest.raises(RuntimeError, match="deliberate"):
            model(**inputs(), native_geometry=features())
    assert model._gradient_checkpointing_func is checkpoint
    assert model._native_body_execution.get() is None
    model(**inputs(), native_geometry=features())


def test_native_adapter_state_is_versioned_and_configuration_is_checked():
    model = adapter()
    state = copy.deepcopy(model.state_dict())
    assert state["_extra_state"]["schema"] == "qwen21_full_native_body_v1"
    state["_extra_state"]["config"]["injection_layers"] = [0]
    with pytest.raises(ValueError, match="config"):
        model.load_state_dict(state)


def test_pipeline_condition_binding_cleans_up_and_checkpoint_keeps_the_bound_sample():
    model = installed()
    nonzero_heads(model.native_body)
    model.enable_gradient_checkpointing()
    first, second = features(91, requires_grad=True), features(92, requires_grad=True)
    with integration_api().bind_native_body(model, first):
        output_a = model(**inputs(3))[0]
    assert model._native_body_binding.get() is None
    with integration_api().bind_native_body(model, second):
        output_b = model(**inputs(5, references=2))[0]
    assert model._native_body_binding.get() is None
    (output_a[:, -4:].square().mean() + output_b[:, -4:].square().mean()).backward()
    assert first.geometry.grad is not None and first.geometry.grad.abs().sum() > 0
    assert second.geometry.grad is not None and second.geometry.grad.abs().sum() > 0
    with pytest.raises(RuntimeError, match="deliberate"):
        with integration_api().bind_native_body(model, first):
            raise RuntimeError("deliberate pipeline failure")
    assert model._native_body_binding.get() is None


def test_cpu_audit_covers_real_qwen_structure_and_checkpoint_gradients():
    name = "scripts.audit.audit_qwen21_native_body"
    assert importlib.util.find_spec(name) is not None, "native body audit is missing"
    report = importlib.import_module(name).audit_native_body()
    assert report["passed"]
    assert report["num_qwen_layers"] == 32
    assert report["zero_max_abs"] == 0
    assert report["prefix_max_abs"] == 0
    assert report["checkpoint_gradients_match"]
    assert report["all_output_heads_have_gradients"]
    assert report["base_parameters_frozen"]


def test_gpu_runtime_audit_rejects_cpu_before_attempting_to_load_weights():
    name = "scripts.audit.verify_qwen21_native_body_runtime"
    assert importlib.util.find_spec(name) is not None, "native body runtime audit is missing"
    with pytest.raises(ValueError, match="CUDA"):
        importlib.import_module(name).verify_runtime(base_model="missing", full_checkpoint="missing", device="cpu")
