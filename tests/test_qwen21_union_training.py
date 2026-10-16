"""Flow scheduler, gradient continuity, and deterministic sampling contracts."""
import importlib
import importlib.util

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from torch import nn
from types import MethodType
import os
import subprocess
import sys

from test_qwen21_union_lora import make_transformer, weights_module


def training_module():
    name = "src.pose_control.qwen21_union.training"
    assert importlib.util.find_spec(name) is not None, "flow training implementation is missing"
    return importlib.import_module(name)


def scheduler():
    return FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)


def cached_pair():
    condition = {
        "reference_latents": torch.randn(1, 4, 64),
        "control_latents": torch.randn(1, 4, 64),
        "prompt_embeds": torch.randn(1, 3, 4096),
        "prompt_mask": torch.tensor([[True, True, False]]),
        "img_mask": torch.tensor([[False, True, False, True]]),
        "img_shapes": [[(1, 2, 2), (1, 2, 2)]],
        "target_hw": (2, 2),
    }
    supervision = {"target_latent": torch.randn(1, 4, 64),
                   "valid_target_tokens": torch.tensor([[1., .5, 0., 1.]])}
    return condition, supervision


class TinyFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.structure = make_transformer(width=4)
        weights_module().install_control_lora(self.structure, expected_parameters=None)
        self.base = nn.Linear(4, 64)
        self.base.requires_grad_(False)
        self.last_inputs = None

    def forward(self, **inputs):
        self.last_inputs = inputs
        hints = self.structure.control_img_in(inputs["control_context"])
        # Frozen downstream operations must remain in the autograd graph.
        prediction = self.base(torch.tanh(hints))
        return (prediction,)


def test_flow_schedule_uses_target_resolution_and_paired_time_sigma():
    module = training_module()
    s = scheduler()
    schedule = module.make_training_schedule(s, target_tokens=1024)
    assert len(schedule.timesteps) == len(schedule.sigmas) == 1000
    assert torch.equal(schedule.timesteps, s.timesteps)
    assert torch.equal(schedule.sigmas, s.sigmas[:-1])
    assert not torch.equal(schedule.sigmas, torch.linspace(1., .001, 1000))
    assert schedule.shift == pytest.approx(.63)
    torch.testing.assert_close(schedule.timesteps / 1000, schedule.sigmas)
    with pytest.raises(ValueError, match="target"):
        module.make_training_schedule(s, target_tokens=0)


def test_flow_loss_only_noises_target_and_uses_masked_content_mean():
    module = training_module()
    condition, supervision = cached_pair()
    noise = torch.randn_like(supervision["target_latent"])
    model = TinyFlow()
    s = scheduler()
    schedule = module.make_training_schedule(s, 4)
    loss = module.flow_loss(model, condition, supervision, 250, noise, schedule, "cpu")
    inputs = model.last_inputs
    sigma = schedule.sigmas[250]
    noisy = (1 - sigma) * supervision["target_latent"] + sigma * noise
    assert torch.equal(inputs["hidden_states"][:, :4], condition["reference_latents"])
    torch.testing.assert_close(inputs["hidden_states"][:, -4:], noisy)
    torch.testing.assert_close(inputs["timestep"], schedule.timesteps[250:251] / 1000)
    assert inputs["control_context"].shape == (1, 8, 129)
    assert inputs["control_context"][:, :4].count_nonzero() == 0
    assert inputs["control_context"][:, -4:, 64:].count_nonzero() == 0
    prediction = model(**inputs)[0][:, -4:]
    square = (prediction.float() - (noise - supervision["target_latent"]).float()).square()
    mask = supervision["valid_target_tokens"].unsqueeze(-1)
    expected = (square * mask).sum() / (mask.sum() * 64)
    torch.testing.assert_close(loss, expected)


def test_flow_gradients_cross_frozen_base_and_b_then_a_receive_updates():
    module = training_module()
    model = TinyFlow()
    condition, supervision = cached_pair()
    noise = torch.randn_like(supervision["target_latent"])
    frozen_before = weights_module().frozen_parameter_digest(model)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    schedule = module.make_training_schedule(scheduler(), 4)
    module.flow_loss(model, condition, supervision, 500, noise, schedule, "cpu").backward()
    layer = model.structure.control_img_in
    assert torch.isfinite(layer.lora_B.grad).all() and layer.lora_B.grad.abs().sum() > 0
    assert layer.lora_A.grad.count_nonzero() == 0
    optim.step()
    optim.zero_grad(set_to_none=True)
    module.flow_loss(model, condition, supervision, 500, noise, schedule, "cpu").backward()
    assert torch.isfinite(layer.lora_A.grad).all() and layer.lora_A.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert weights_module().frozen_parameter_digest(model) == frozen_before


def test_flow_rejects_out_of_range_mask_and_empty_mask():
    module = training_module()
    condition, supervision = cached_pair()
    schedule = module.make_training_schedule(scheduler(), 4)
    supervision["valid_target_tokens"].fill_(2)
    with pytest.raises(ValueError, match="coverage"):
        module.flow_loss(TinyFlow(), condition, supervision, 0, torch.zeros(1, 4, 64), schedule, "cpu")
    supervision["valid_target_tokens"].zero_()
    with pytest.raises(ValueError, match="content"):
        module.flow_loss(TinyFlow(), condition, supervision, 0, torch.zeros(1, 4, 64), schedule, "cpu")


def test_trajectory_is_stable_balanced_and_streams_are_independent():
    module = training_module()
    order = module.balanced_sample_order({"iper": ["i2", "i1"], "tiktok": ["t1", "t2"],
                                         "hi4d": ["h1", "h2"]})
    assert [name for name, _ in order] == ["iper", "tiktok", "hi4d"] * 2
    assert len(set(order)) == 6
    assert module.balanced_sample_order({"iper": ["i1", "i2"], "tiktok": ["t2", "t1"],
                                         "hi4d": ["h2", "h1"]}) == order
    seed = module.deterministic_seed(3407, 1, 2, "sample", "noise")
    assert 0 <= seed < 2**63
    assert seed == module.deterministic_seed(3407, 1, 2, "sample", "noise")
    assert seed != module.deterministic_seed(3407, 1, 2, "sample", "timestep")
    with pytest.raises(ValueError, match="equal"):
        module.balanced_sample_order({"iper": ["i"], "tiktok": [], "hi4d": ["h"]})


def test_optimizer_warmup_uses_fp32_state_and_reaches_constant_lr():
    module = training_module()
    model = make_transformer()
    weights_module().install_control_lora(model, expected_parameters=None)
    optimizer = module.make_optimizer(model, expected_parameters=None)
    lr = module.make_lr_scheduler(optimizer, warmup_updates=2)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-5)
    loss = model.control_img_in(torch.ones(1, 129)).sum()
    loss.backward()
    optimizer.step()
    lr.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
    for state in optimizer.state.values():
        assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32


def test_training_draw_does_not_consume_global_rng_or_depend_on_validation():
    module = training_module()
    before = torch.get_rng_state().clone()
    one = module.training_draw(train_seed=3407, update=2, microstep=1, sample_id="sample",
                               target_shape=(1, 4, 64))
    assert torch.equal(before, torch.get_rng_state())
    torch.randn(17)
    two = module.training_draw(train_seed=3407, update=2, microstep=1, sample_id="sample",
                               target_shape=(1, 4, 64))
    assert one["index"] == two["index"]
    assert torch.equal(one["noise"], two["noise"])


def test_cached_inference_tensors_are_cloned_before_training_backward():
    module = training_module()
    with torch.inference_mode():
        condition, supervision = cached_pair()
    loss = module.flow_loss(TinyFlow(), condition, supervision, 250, torch.ones(1, 4, 64),
                            module.make_training_schedule(scheduler(), 4), "cpu")
    loss.backward()
    assert torch.isfinite(loss)


def driver_model():
    model = make_transformer()
    weights_module().install_control_lora(model, expected_parameters=None)
    def forward(self, **inputs):
        hints = torch.tanh(self.control_img_in(inputs["control_context"]))
        return (self.transformer_blocks[0](hints).repeat(1, 1, 16),)
    model.forward = MethodType(forward, model)
    return model


def test_bounded_training_driver_stops_at_boundary_and_resumes_same_trajectory():
    module = training_module()
    order = module.balanced_sample_order({"iper": ["i1", "i2"], "tiktok": ["t1", "t2"],
                                         "hi4d": ["h1", "h2"]})
    pair = cached_pair()
    model = driver_model()
    optimizer = module.make_optimizer(model, expected_parameters=None)
    lr = module.make_lr_scheduler(optimizer)
    events, microevents = [], []
    result = module.run_training_updates(model, optimizer=optimizer, lr_scheduler=lr,
        scheduler=scheduler(), sample_order=order, load_sample=lambda sample_id: pair,
        updates=2, device="cpu", on_update=events.append, on_microstep=microevents.append,
        stop_requested=lambda: True, expected_parameters=None)
    assert result["cursor"].optimizer_step == 1
    assert result["cursor"].microstep == result["cursor"].next_sample_position == 3
    assert len(microevents) == 3 and len(events) == 1
    resumed = module.run_training_updates(model, optimizer=optimizer, lr_scheduler=lr,
        scheduler=scheduler(), sample_order=order, load_sample=lambda sample_id: pair,
        updates=2, cursor=result["cursor"], device="cpu", expected_parameters=None)
    assert resumed["cursor"].optimizer_step == 2
    assert [item["sample_id"] for item in result["trajectory"] + resumed["trajectory"]] == [sample_id for _, sample_id in order]


def test_training_driver_rejects_budget_overrun_and_preserves_frozen_weights():
    module = training_module()
    pair = cached_pair()
    order = (("iper", "i1"), ("tiktok", "t1"), ("hi4d", "h1"))
    model = driver_model()
    optimizer = module.make_optimizer(model, expected_parameters=None)
    lr = module.make_lr_scheduler(optimizer)
    with pytest.raises(ValueError, match="budget"):
        module.run_training_updates(model, optimizer=optimizer, lr_scheduler=lr,
            scheduler=scheduler(), sample_order=order, load_sample=lambda sample_id: pair,
            updates=2, device="cpu", expected_parameters=None)
    before = weights_module().frozen_parameter_digest(model)
    result = module.run_training_updates(model, optimizer=optimizer, lr_scheduler=lr,
        scheduler=scheduler(), sample_order=order, load_sample=lambda sample_id: pair,
        updates=1, device="cpu", expected_parameters=None)
    assert result["cursor"].optimizer_step == 1
    assert result["updates"][0]["gradient_norm"] > 0
    assert weights_module().frozen_parameter_digest(model) == before


def test_configure_determinism_sets_real_torch_flags_before_cuda_and_is_idempotent():
    module = training_module()
    assert hasattr(module, "configure_determinism"), "strict CUDA determinism configuration is missing"
    code = "\n".join([
        "import os, torch",
        "from src.pose_control.qwen21_union.training import configure_determinism",
        "assert not torch.cuda.is_initialized()",
        "first = configure_determinism()",
        "second = configure_determinism()",
        "assert first == second",
        "assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'",
        "assert torch.are_deterministic_algorithms_enabled()",
        "assert not torch.is_deterministic_algorithms_warn_only_enabled()",
        "assert torch.backends.cudnn.deterministic",
        "assert not torch.backends.cudnn.benchmark",
        "assert not torch.cuda.is_initialized()",
    ])
    environment = dict(os.environ)
    environment.pop("CUBLAS_WORKSPACE_CONFIG", None)
    result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("existing", [None, ":16:8", "bad-value"])
def test_configure_determinism_rejects_already_initialized_cuda_with_wrong_config(monkeypatch, existing):
    module = training_module()
    assert hasattr(module, "configure_determinism"), "strict CUDA determinism configuration is missing"
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    if existing is None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    else:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", existing)
    with pytest.raises(RuntimeError, match="before CUDA"):
        module.configure_determinism()
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == existing
