"""冻结 Linear CPU 权重的输入反传、重计算与 LoRA 更新。"""
import copy
import importlib
import os
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.pose_control.qwen21_union.weights import ControlLoRALinear


def api():
    return importlib.import_module("src.pose_control.qwen21_pro.offload")


def model():
    net = nn.Module()
    net.base = nn.Linear(4, 4).requires_grad_(False)
    net.norm = nn.LayerNorm(4).requires_grad_(False)
    net.task_adapter = nn.Linear(4, 4)
    return net


@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4), (2, 2, 3, 4)])
def test_same_objects_keys_dtype_and_native_input_gradient(shape):
    net = model().double()
    original = copy.deepcopy(net.base)
    parameters = {n: id(p) for n, p in net.named_parameters()}
    keys = tuple(net.state_dict())
    report = api().place_training_model(net, "cpu", memory_mode="frozen_linear_cpu")
    assert report["offloaded_linear_count"] == 1
    assert tuple(net.state_dict()) == keys
    assert {n: id(p) for n, p in net.named_parameters()} == parameters
    x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    output, expected = net.base(x), original(other)
    grad = torch.randn_like(output)
    output.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, other.grad, rtol=0, atol=0)
    assert net.base.weight.grad is None


def test_nonreentrant_recompute_and_lora_optimizer_update_match_resident():
    torch.manual_seed(9)
    layer = ControlLoRALinear(nn.Linear(4, 4).requires_grad_(False), rank=2, alpha=2)
    with torch.no_grad():
        layer.lora_B.fill_(.1)
    net = nn.Module()
    net.control_img_in = layer
    resident = copy.deepcopy(net)
    api().place_training_model(net, "cpu", memory_mode="frozen_linear_cpu")
    x = torch.randn(2, 3, 4)
    for item in (net, resident):
        optimizer = torch.optim.AdamW([p for p in item.parameters() if p.requires_grad], lr=.01)
        hidden = x.clone().requires_grad_()
        loss = checkpoint(item.control_img_in, hidden, use_reentrant=False).square().mean()
        loss.backward()
        optimizer.step()
        item.saved_input_grad = hidden.grad
    torch.testing.assert_close(net.saved_input_grad, resident.saved_input_grad, rtol=0, atol=0)
    for actual, expected in zip(net.parameters(), resident.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_offload_rejects_hooks_and_unknown_mode_before_mutation():
    net = model()
    net.base._hf_hook = object()
    with pytest.raises(ValueError, match="hook"):
        api().place_training_model(net, "cpu", memory_mode="frozen_linear_cpu")
    assert type(net.base) is nn.Linear
    with pytest.raises(ValueError, match="memory_mode"):
        api().place_training_model(model(), "cpu", memory_mode="video_group")


def test_cpu_weight_saved_for_backward_and_no_accidental_task_offload():
    net = model()
    net.base.to(dtype=torch.bfloat16)
    api().place_training_model(net, "cpu", memory_mode="frozen_linear_cpu")
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda x: saved.append(x) or x, lambda x: x):
        net.base(torch.randn(2, 4, dtype=torch.bfloat16, requires_grad=True)).float().sum().backward()
    assert any(x is net.base.weight for x in saved)
    assert type(net.task_adapter) is nn.Linear
    assert net.base.weight.dtype == torch.bfloat16
    report = api().audit_memory_policy(net)
    assert report["memory_mode"] == "frozen_linear_cpu"
    assert report["offloaded_dtype_counts"] == {"torch.bfloat16": 1}


def test_upstream_auto_device_hook_registry_rejects_selective_placement_before_mutation():
    net = model()
    net._auto_device_hooks = [net.base.register_forward_pre_hook(lambda *args: None)]
    before = copy.deepcopy(net.state_dict())
    identities = {name: id(parameter) for name, parameter in net.named_parameters()}
    with pytest.raises(ValueError, match="hook"):
        api().place_training_model(net, "cpu", memory_mode="frozen_linear_cpu")
    assert type(net.base) is nn.Linear
    assert {name: id(parameter) for name, parameter in net.named_parameters()} == identities
    for name, tensor in before.items():
        torch.testing.assert_close(net.state_dict()[name], tensor, rtol=0, atol=0)


SOURCE = Path(os.environ.get("QWEN21_UNION_SOURCE", "/data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1/source/VideoX-Fun"))


@pytest.mark.skipif(not torch.cuda.is_available() or not SOURCE.is_dir()
                    or os.environ.get("QWEN21_PRO_CUDA_TESTS") != "1",
                    reason="requires CUDA, pinned VideoX-Fun and QWEN21_PRO_CUDA_TESTS=1")
@pytest.mark.parametrize("with_lora", [False, True])
def test_fixed_native_cuda_two_updates_resident_offload_forward_backward_optimizer_parity(with_lora):
    from src.pose_control.qwen21_union.pipeline import _upstream, scoped_transformer_class
    from src.pose_control.qwen21_union.weights import install_control_lora, control_lora_targets
    from src.pose_control.qwen21_pro.integration import install_pro_adapter
    from src.pose_control.qwen21_pro.contracts import ProAdapterConfig
    from src.pose_control.qwen21_pro import training
    from src.pose_control.qwen21_union.training import TrainingSchedule
    from test_qwen21_pro_training import pair
    training.configure_determinism()
    _, models, _, _ = _upstream(SOURCE)
    cls = scoped_transformer_class(models.QwenImage21ControlTransformer2DModel)
    targets = control_lora_targets() if with_lora else ()
    def construct(mode):
        torch.manual_seed(21)
        net = cls(num_layers=32, num_attention_heads=4, attention_head_dim=8, context_in_dim=8,
                  axes_dims_rope=(2, 2, 4), control_in_dim=129).to(dtype=torch.bfloat16)
        with torch.no_grad():
            for block in net.control_blocks:
                block.after_proj.weight.normal_(std=.02)
        if with_lora:
            install_control_lora(net, expected_parameters=None)
        install_pro_adapter(net, ProAdapterConfig(hidden_dim=32, width=16, heads=4, source_dim=1600))
        original = {n: id(p) for n, p in net.named_parameters()}
        api().place_training_model(net, "cuda", memory_mode=mode)
        assert {n: id(p) for n, p in net.named_parameters()} == original
        net.gradient_checkpointing = True
        net.train()
        return net
    resident, offloaded = construct("resident"), construct("frozen_linear_cpu")
    assert next(offloaded.parameters()).device.type == "cpu"
    optimizers = [training.make_optimizer(n, lr=.01, control_lora_targets=targets) for n in (resident, offloaded)]
    frozen = [training.frozen_parameter_digest(n) for n in (resident, offloaded)]
    condition, supervision, pack = pair(1600)
    schedule = TrainingSchedule(torch.tensor([500.]), torch.tensor([.5]), 0., 4)
    noise = torch.ones_like(supervision["target_latent"], device="cuda")
    for step in range(2):
        losses, gradients = [], []
        for net, optimizer in zip((resident, offloaded), optimizers):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = training.flow_loss(net, condition, supervision, 0, noise, schedule, "cuda", task_condition=pack)
            losses.append(loss.detach())
            loss.backward()
            gradients.append({n: p.grad.detach().clone() for n, p in net.named_parameters() if p.grad is not None})
            assert net.task_adapter.output_heads["0"].weight.grad.abs().sum() > 0
            if step:
                assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.task_adapter.geometry_encoder.parameters())
                assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.task_adapter.banks.parameters())
            training.audit_trainable_parameters(net, control_lora_targets=targets)
            assert all(p.dtype == torch.float32 and p.device.type == "cuda" for p in net.parameters() if p.requires_grad)
            optimizer.step()
        torch.testing.assert_close(losses[0], losses[1], rtol=1e-5, atol=1e-6)
        assert gradients[0].keys() == gradients[1].keys()
        for name in gradients[0]:
            torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=2e-4, atol=2e-5)
        for left, right in zip(resident.parameters(), offloaded.parameters()):
            torch.testing.assert_close(left.cpu(), right.cpu(), rtol=2e-4, atol=2e-5)
    assert [training.frozen_parameter_digest(n) for n in (resident, offloaded)] == frozen
