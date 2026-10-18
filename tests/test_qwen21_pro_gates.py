"""独立固定18 loss定义与有界训练状态，CPU不能冒充GPU通过。"""
import importlib
import json

import pytest
import torch

from test_qwen21_pro_training import TinyPro, Scheduler, pair
from src.pose_control.qwen21_pro import training


def api():
    try:
        return importlib.import_module("src.pose_control.qwen21_pro.gates")
    except ModuleNotFoundError:
        pytest.fail("Independent Pro gates are missing")


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


ORDER = tuple((d, f"{d}{i}") for i in range(2) for d in ("iper", "tiktok", "hi4d"))


def test_fixed18_uses_global_content_mse_not_regional_total_and_seed43():
    module = api()
    model = TinyPro()
    sample = pair()
    report = module.evaluate_fixed_overfit_loss(model, ORDER, lambda _: sample, Scheduler(), device="cpu")
    assert len(report["entries"]) == 18
    assert report["indices"] == [250, 500, 750]
    assert report["noise_seed"] == 43
    assert report["noise_dtype"] == "torch.bfloat16"
    noise = torch.randn(sample[1]["target_latent"].shape,
                        generator=torch.Generator().manual_seed(43), dtype=torch.bfloat16)
    _, metrics = training.flow_loss(model, sample[0], sample[1], 250, noise,
                                    Scheduler(), "cpu", task_condition=sample[2], return_metrics=True)
    assert report["entries"][0]["global_content_mse"] == metrics["losses"]["global"]
    assert report["global_content_mse"] == pytest.approx(sum(e["global_content_mse"] for e in report["entries"]) / 18)
    assert any(e["regional_total"] > e["global_content_mse"] for e in report["entries"])
    repeated = module.evaluate_fixed_overfit_loss(model, ORDER, lambda _: sample, Scheduler(), device="cpu")
    assert repeated == report
    assert model.training


def test_overfit_gate_needs_completed200_cuda_and_twenty_percent():
    module = api()
    valid = dict(completed_updates=200, planned_updates=200, device="cuda", initial_global=10.,
                 final_global=8., fixed_evaluations=18, frozen_unchanged=True, cursor_complete=True,
                 source_verified=True, stopped_early=False)
    assert module.overfit_gate(**valid)["passed"]
    assert not module.overfit_gate(**{**valid, "final_global": 8.22132})["passed"]
    for changes in ({"device": "cpu"}, {"completed_updates": 199}, {"stopped_early": True},
                    {"fixed_evaluations": 17}, {"cursor_complete": False}):
        assert not module.overfit_gate(**{**valid, **changes})["passed"]


def test_safe_stop_saves_boundary_and_never_passes_quality(tmp_path):
    module = api()
    model, sample = TinyPro(), pair()
    metadata = {"config": model.task_adapter.config.to_dict(), "scope": "joint", "memory_mode": "resident",
                "control_lora_targets": [], **{k: "fixture" for k in ("task_signature", "source_signature",
                "cache_signature", "code_signature", "data_signature", "trajectory_signature")}}
    result = module.run_pilot(model, scheduler=Scheduler(), load_sample=lambda _: sample,
        sample_order=ORDER, metadata=metadata, run_dir=tmp_path / "stopped", profile="overfit",
        device="cpu", stop_requested=lambda: True, require_verified_conditions=False)
    assert result["cursor"]["optimizer_step"] == 1
    assert result["status"] == "incomplete" and not result["passed"]
    assert (tmp_path / "stopped/checkpoint_000001/checkpoint.json").is_file()
    assert (tmp_path / "stopped/checkpoint_000000/checkpoint.json").is_file()
    lines = (tmp_path / "stopped/microsteps.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["active_parameters"] > 0
    assert result["frozen_unchanged"]


def test_native_zero_probe_detects_nonzero_initial_heads_and_checks_prefix():
    module = api()
    class Toggle(TinyPro):
        def forward(self, *args, task_enabled=True, **kwargs):
            previous = self.task_adapter.enabled
            self.task_adapter.enabled = task_enabled
            try:
                return super().forward(*args, **kwargs)
            finally:
                self.task_adapter.enabled = previous
    model, sample = Toggle(), pair()
    inputs, base = module.probe_zero_equivalence(model, sample, device="cpu")
    assert module.verify_reference_prefix(model, inputs, sample[2], base)
    with torch.no_grad():
        model.task_adapter.output_heads["0"].weight.fill_(.2)
    with pytest.raises(ValueError, match="zero"):
        module.probe_zero_equivalence(model, sample, device="cpu")


def test_smoke_exact_is_four_continuous_equals_two_plus_two_on_real_torch(tmp_path):
    module = api()
    model, sample = TinyPro(), pair()
    metadata = {"config": model.task_adapter.config.to_dict(), "scope": "joint", "memory_mode": "resident",
                "control_lora_targets": [], **{k: "fixture" for k in ("task_signature", "source_signature",
                "cache_signature", "code_signature", "data_signature", "trajectory_signature")}}
    report = module.run_pilot(model, scheduler=Scheduler(), load_sample=lambda _: sample,
        sample_order=ORDER, metadata=metadata, run_dir=tmp_path / "exact", profile="smoke_exact",
        device="cpu", require_verified_conditions=False)
    assert report["exact_resume"] is True
    assert not report["passed"] and not report["engineering_passed"]


@pytest.mark.parametrize("missing", [None, "heads", "geometry", "contacts", "appearance", "identity"])
def test_complete_engineering_gate_requires_every_full4_branch(missing):
    module = api()
    branches = {name: 1. for name in ("heads", "geometry", "contacts", "appearance", "identity")}
    if missing:
        branches[missing] = 0.
    decision = module.engineering_gate(complete=True, device="cuda", frozen_unchanged=True,
        prefix_unchanged=True, source_verified=True, zero_equivalence=True, profile="overfit",
        exact_resume=None, arm="full4", branch_maxima=branches)
    assert decision["passed"] is (missing is None)


def test_loss_metric_pass_alone_cannot_mark_pilot_passed(tmp_path, monkeypatch):
    module = api()
    model, sample = TinyPro(), pair()
    metadata = {"config": model.task_adapter.config.to_dict(), "scope": "joint", "memory_mode": "resident",
                "control_lora_targets": [], **{k: "fixture" for k in ("task_signature", "source_signature",
                "cache_signature", "code_signature", "data_signature", "trajectory_signature")}}
    monkeypatch.setattr(module, "overfit_gate", lambda **kwargs: {"passed": True, "relative_global_drop": .2})
    report = module.run_pilot(model, scheduler=Scheduler(), load_sample=lambda _: sample,
        sample_order=ORDER, metadata=metadata, run_dir=tmp_path / "metric_only", profile="smoke",
        device="cpu", require_verified_conditions=False)
    assert not report["engineering_passed"]
    assert not report["passed"]
    assert report["overfit_metric_gate"]["passed"]
    saved = json.loads((tmp_path / "metric_only/checkpoint_000002/checkpoint.json").read_text())
    assert report["final_checkpoint"]["cursor"] == saved["cursor"]
    assert report["final_checkpoint"]["state_sha256"] == saved["state_sha256"]
    assert report["final_checkpoint"]["selection_role"] == "engineering_probe"
