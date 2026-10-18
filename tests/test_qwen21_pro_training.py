"""Pro 冻结审计、原生输入与确定性更新边界。"""
import copy
import importlib
from dataclasses import replace

import pytest
import torch
from torch import nn

from test_qwen21_pro_contracts import make_condition
from src.pose_control.qwen21_pro import ProAdapterConfig, ProTaskAdapter
from src.pose_control.qwen21_union.weights import ControlLoRALinear
from src.pose_control.qwen21_union.training import TrainingSchedule


def api():
    return importlib.import_module("src.pose_control.qwen21_pro.training")


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class TinyPro(nn.Module):
    def __init__(self, source_dim=64):
        super().__init__()
        self.base = nn.Linear(64, 8).requires_grad_(False)
        self.output = nn.Linear(8, 64).requires_grad_(False)
        self.task_adapter = ProTaskAdapter(ProAdapterConfig(hidden_dim=8, width=8, heads=2,
                                                          stage_indices=(0,), source_dim=source_dim))
        self.seen = None

    def forward(self, hidden_states, *, task_condition, timestep, **kwargs):
        self.seen = {"hidden_states": hidden_states.detach().clone(), "task_condition": task_condition,
                     "timestep": timestep.detach().clone(), **kwargs}
        hidden = self.base(hidden_states)
        target = torch.zeros(hidden.shape[1], dtype=torch.bool, device=hidden.device)
        target[-task_condition.target_hw[0] * task_condition.target_hw[1]:] = True
        hidden = self.task_adapter.forward_layer(0, hidden, target, task_condition, timestep)
        return (self.output(hidden),)


class Scheduler:
    config = {"use_dynamic_shifting": True}

    def set_timesteps(self, sigmas, device, mu):
        self.sigmas = torch.cat([torch.as_tensor(sigmas).float(), torch.zeros(1)])
        self.timesteps = self.sigmas[:-1] * 1000


def pair(source_dim=64):
    generator = torch.Generator().manual_seed(81)
    condition = {"reference_latents": torch.randn(1, 4, 64, generator=generator),
                 "control_latents": torch.randn(1, 4, 64, generator=generator),
                 "prompt_embeds": torch.randn(1, 3, 8, generator=generator),
                 "prompt_mask": torch.tensor([[True, True, False]]),
                 "img_mask": torch.tensor([[False, True, False, True]]),
                 "img_shapes": [[(1, 2, 2), (1, 2, 2)]], "target_hw": (2, 2)}
    supervision = {"target_latent": torch.randn(1, 4, 64, generator=generator),
                   "valid_target_tokens": torch.tensor([[1., .5, 0., 1.]])}
    with torch.random.fork_rng():
        torch.manual_seed(81)
        pack = make_condition(people=1)
    pack = replace(pack, source_tokens=torch.randn(1, 1, 3, source_dim, generator=generator))
    return condition, supervision, pack


def run(model, updates=4, **kwargs):
    optimizer = kwargs.pop("optimizer", None) or api().make_optimizer(model, lr=.01)
    lr = kwargs.pop("lr_scheduler", None) or api().make_lr_scheduler(optimizer, warmup_updates=2)
    item = pair(model.task_adapter.config.source_dim)
    order = tuple((dataset, f"{dataset}_{i}") for i in range(4) for dataset in ("iper", "tiktok", "hi4d"))
    result = api().run_training_updates(model, optimizer=optimizer, lr_scheduler=lr, scheduler=Scheduler(),
        sample_order=order, load_sample=lambda _: item, updates=updates, device="cpu",
        require_verified_conditions=False, **kwargs)
    return result, optimizer, lr


def test_audit_task_only_and_explicit_exact_control_lora_inventory():
    model = TinyPro()
    report = api().audit_trainable_parameters(model)
    assert report["trainable_names"] == tuple(f"task_adapter.{n}" for n, _ in model.task_adapter.named_parameters())
    model.control_img_in = ControlLoRALinear(nn.Linear(129, 8), rank=2, alpha=2)
    with pytest.raises(ValueError, match="LoRA"):
        api().audit_trainable_parameters(model)
    report = api().audit_trainable_parameters(model, control_lora_targets=("control_img_in",))
    assert report["control_lora"][0] == {"name": "control_img_in", "rank": 2, "alpha": 2., "enabled": True}
    model.control_img_in.enabled = False
    with pytest.raises(ValueError, match="enabled"):
        api().audit_trainable_parameters(model, control_lora_targets=("control_img_in",))


@pytest.mark.parametrize("mutation", ["extra", "missing", "disabled", "bf16", "foreign_lora"])
def test_audit_rejects_unexpected_or_disabled_training(mutation):
    model = TinyPro()
    if mutation == "extra":
        model.output.weight.requires_grad_(True)
    elif mutation == "missing":
        model.task_adapter.query_projection.weight.requires_grad_(False)
    elif mutation == "disabled":
        model.task_adapter.enabled = False
    elif mutation == "bf16":
        model.task_adapter.query_projection.weight.data = model.task_adapter.query_projection.weight.data.bfloat16()
    else:
        model.foreign = ControlLoRALinear(nn.Linear(8, 8), rank=2, alpha=2)
    with pytest.raises(ValueError):
        api().make_optimizer(model)


def test_flow_loss_passes_one_actual1600_pack_and_only_noises_target():
    model = TinyPro(source_dim=1600)
    condition, supervision, pack = pair(1600)
    noise = torch.ones_like(supervision["target_latent"])
    schedule = TrainingSchedule(torch.tensor([500.]), torch.tensor([.5]), 0., 4)
    loss, metrics = api().flow_loss(model, condition, supervision, 0, noise, schedule, "cpu",
                                  task_condition=pack, return_metrics=True)
    assert model.seen["task_condition"] is pack
    torch.testing.assert_close(model.seen["hidden_states"][:, :4], condition["reference_latents"])
    torch.testing.assert_close(model.seen["hidden_states"][:, -4:], .5 * supervision["target_latent"] + .5 * noise)
    assert "target_latent" not in model.seen and "valid_target_tokens" not in model.seen
    assert "coverage" in metrics
    loss.backward()
    assert model.task_adapter.output_heads["0"].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)


def test_runner_boundary_stop_balanced_cursors_and_head_upstream_logs():
    torch.manual_seed(8)
    model = TinyPro()
    frozen = api().frozen_parameter_digest(model)
    seen = []
    result, _, _ = run(model, on_update=seen.append, stop_requested=lambda: True)
    assert result["cursor"].optimizer_step == 1 and result["cursor"].microstep == 3
    assert result["cursor"].noise_draw_cursor == result["cursor"].timestep_draw_cursor == 3
    assert [e["dataset"] for e in result["trajectory"]] == ["iper", "tiktok", "hi4d"]
    assert seen[0]["gradient_norms"]["head"] > 0
    assert "upstream" in seen[0]["gradient_norms"]
    assert "coverage" in result["trajectory"][0]
    assert api().frozen_parameter_digest(model) == frozen
    assert all(p.grad is None for p in model.parameters())


def test_runner_deterministic_and_nonfinite_clears_partial_gradients():
    torch.manual_seed(90)
    first = TinyPro()
    second = copy.deepcopy(first)
    result_a, _, _ = run(first, updates=2)
    result_b, _, _ = run(second, updates=2)
    assert result_a["trajectory"] == result_b["trajectory"]
    for a, b in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    condition, supervision, pack = pair()
    supervision["target_latent"][0, 0, 0] = float("nan")
    opt = api().make_optimizer(first)
    with pytest.raises(ValueError, match="finite"):
        api().run_training_updates(first, optimizer=opt, lr_scheduler=api().make_lr_scheduler(opt),
            scheduler=Scheduler(), sample_order=[("iper", "a"), ("tiktok", "b"), ("hi4d", "c")],
            load_sample=lambda _: (condition, supervision, pack), updates=1, device="cpu", require_verified_conditions=False)
    assert all(p.grad is None for p in first.parameters())


def test_real_runner_rejects_unverified_legacy_mapping_by_default():
    model = TinyPro()
    opt = api().make_optimizer(model)
    with pytest.raises(ValueError, match="verified"):
        api().run_training_updates(model, optimizer=opt, lr_scheduler=api().make_lr_scheduler(opt), scheduler=Scheduler(),
            sample_order=[("iper", "a"), ("tiktok", "b"), ("hi4d", "c")],
            load_sample=lambda _: pair(), updates=1, device="cpu")


def test_refiner_inpaint_context_changes_frozen_prediction_and_rejects_partial_payload():
    condition, supervision, _ = pair()
    noisy = supervision["target_latent"]
    condition.update(keep_mask=torch.full((1, 4, 1), .25), inpaint_latents=torch.full((1, 4, 64), 3.))
    inputs = api().transformer_inputs(condition, noisy, torch.ones(1))
    # A frozen native projection is sensitive to both genuine inpaint operands.
    projection = nn.Linear(129, 1, bias=False).requires_grad_(False)
    with torch.no_grad():
        projection.weight.zero_()
        projection.weight[:, 64] = 2.
        projection.weight[:, 65:] = 1. / 64
    output = projection(inputs["control_context"])
    torch.testing.assert_close(output[:, :4], torch.zeros(1, 4, 1))
    torch.testing.assert_close(output[:, -4:], torch.full((1, 4, 1), 3.5))
    condition.pop("keep_mask")
    with pytest.raises(ValueError, match="together"):
        api().transformer_inputs(condition, noisy, torch.ones(1))


def test_verified_training_rejects_same_shape_pack_with_another_sample_binding(tmp_path):
    from test_qwen21_pro_preparation import fixture_assets
    from src.pose_control.qwen21_pro.preparation import prepare_task_condition
    record, condition, assets = fixture_assets(tmp_path)
    pack = prepare_task_condition(record, condition, project_root=tmp_path, config=ProAdapterConfig(), assets=assets)
    api()._verify_sample(condition, pack, "fixture")
    changed = replace(pack, provenance={**pack.provenance, "cache_binding": {
        **pack.provenance.get("cache_binding", {}), "condition_fingerprint": "wrong-same-shape"}})
    with pytest.raises(ValueError, match="binding"):
        api()._verify_sample(condition, changed, "fixture")


def test_runner_allows_explicit_custom_typed_verifier_and_rejects_bare_dict_attestation():
    model = TinyPro()
    opt = api().make_optimizer(model)
    order = [("iper", "a"), ("tiktok", "b"), ("hi4d", "c")]
    def validator(condition, pack, sample_id):
        return api().SampleAttestation(sample_id, "c" * 64, "d" * 64, pack.source_fingerprint,
                                       pack.target_hw, pack.person_ids)
    result = api().run_training_updates(model, optimizer=opt, lr_scheduler=api().make_lr_scheduler(opt),
        scheduler=Scheduler(), sample_order=order, load_sample=lambda _: pair(), updates=1, device="cpu",
        sample_validator=validator)
    assert result["cursor"].optimizer_step == 1
    with pytest.raises(ValueError, match="SampleAttestation"):
        api().run_training_updates(model, optimizer=opt, lr_scheduler=api().make_lr_scheduler(opt),
            scheduler=Scheduler(), sample_order=order, load_sample=lambda _: pair(), updates=1, device="cpu",
            sample_validator=lambda *args: {"verified": True})


def test_variable_target_grids_use_distinct_schedule_and_keep_balanced_order():
    small = pair()
    condition, supervision, pack = pair()
    pack = replace(pack, target_hw=(2, 4), spatial={k: v.repeat(1, 1, 1, 1, 2) for k, v in pack.spatial.items()},
        person_mask=pack.person_mask.repeat(1, 1, 2), owner=pack.owner.repeat(1, 1, 2),
        face_mask=pack.face_mask.repeat(1, 1, 2), hand_mask=pack.hand_mask.repeat(1, 1, 1, 2))
    condition = {**condition, "target_hw": (2, 4), "control_latents": condition["control_latents"].repeat(1, 2, 1),
        "img_shapes": [[(1, 2, 2), (1, 2, 4)]], "img_mask": torch.tensor([[False, True, False, True, True]])}
    supervision = {k: v.repeat(1, 2, 1) if v.ndim == 3 else v.repeat(1, 2) for k, v in supervision.items()}
    model, seen = TinyPro(), []
    opt = api().make_optimizer(model)
    result = api().run_training_updates(model, optimizer=opt, lr_scheduler=api().make_lr_scheduler(opt),
        scheduler=Scheduler(), sample_order=[("iper", "a"), ("tiktok", "b"), ("hi4d", "c")], updates=1,
        load_sample=lambda name: (condition, supervision, pack) if name == "b" else small, device="cpu",
        require_verified_conditions=False, on_microstep=seen.append)
    assert [entry["sample_id"] for entry in seen] == ["a", "b", "c"]
    assert result["cursor"].microstep == 3


@pytest.mark.parametrize("entrypoint", ["run_updates", "flow_loss"])
def test_training_rejects_snapshot_restore_hooks_added_after_placement_before_forward(entrypoint):
    from src.pose_control.qwen21_pro.offload import place_training_model
    model = TinyPro()
    place_training_model(model, "cpu", memory_mode="resident")
    optimizer = api().make_optimizer(model)
    lr = api().make_lr_scheduler(optimizer)
    condition, supervision, pack = pair()
    head = model.task_adapter.output_heads["0"]
    snapshot = {name: value.detach().clone() for name, value in head.named_parameters()}
    called = []
    def restore_snapshot(module, args):
        called.append(True)
        with torch.no_grad():
            for name, parameter in module.named_parameters():
                parameter.copy_(snapshot[name])
    model._auto_device_hooks = [head.register_forward_pre_hook(restore_snapshot)]
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    with pytest.raises(ValueError, match="hook"):
        if entrypoint == "run_updates":
            api().run_training_updates(model, optimizer=optimizer, lr_scheduler=lr, scheduler=Scheduler(),
                sample_order=[("iper", "a"), ("tiktok", "b"), ("hi4d", "c")], updates=1,
                load_sample=lambda _: (condition, supervision, pack), device="cpu", require_verified_conditions=False)
        else:
            schedule = TrainingSchedule(torch.tensor([500.]), torch.tensor([.5]), 0., 4)
            api().flow_loss(model, condition, supervision, 0, torch.ones_like(supervision["target_latent"]),
                            schedule, "cpu", task_condition=pack)
    assert called == [] and model.seen is None
    assert optimizer.state_dict()["state"] == {}
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
