"""GPU gate orchestration is validated with actual CPU flow/optimizer work."""
import importlib
import importlib.util
import json

import pytest
import torch

from test_qwen21_union_checkpoint import metadata
from test_qwen21_union_training import cached_pair, driver_model, scheduler


def gates_module():
    name = "src.pose_control.qwen21_union.gates"
    assert importlib.util.find_spec(name) is not None, "smoke/overfit gate implementation is missing"
    return importlib.import_module(name)


def six_samples():
    pairs = {dataset + str(index): cached_pair() for index in range(2) for dataset in ("iper", "tiktok", "hi4d")}
    order = tuple((dataset, dataset + str(index)) for index in range(2) for dataset in ("iper", "tiktok", "hi4d"))
    return order, pairs.__getitem__


def test_fixed_overfit_loss_evaluates_exact_18_positions_without_rng_drift():
    module = gates_module()
    model = driver_model()
    order, load_sample = six_samples()
    before = torch.get_rng_state().clone()
    one = module.evaluate_fixed_overfit_loss(model, order, load_sample, scheduler(), device="cpu")
    assert torch.equal(before, torch.get_rng_state())
    two = module.evaluate_fixed_overfit_loss(model, order, load_sample, scheduler(), device="cpu")
    assert one == two
    assert len(one["losses"]) == 18
    assert {entry["index"] for entry in one["losses"]} == {250, 500, 750}
    assert {entry["noise_seed"] for entry in one["losses"]} == {43}
    assert model.training


def test_real_driver_resume_gate_four_vs_two_plus_two(tmp_path):
    module = gates_module()
    model = driver_model()
    order, load_sample = six_samples()
    result = module.check_resume_equivalence(model, order, load_sample, scheduler(), tmp_path,
        metadata(), device="cpu", expected_parameters=None)
    assert result["passed"]
    assert result["continuous_updates"] == 4
    assert result["resumed_updates"] == [2, 2]
    assert result["microsteps"] == 12
    assert result["max_lora_abs_error"] == 0
    assert (tmp_path / "resume_checkpoint_000002" / "checkpoint.json").is_file()
    starts = json.loads((tmp_path / "resume_start_states.json").read_text())
    assert starts["continuous"] == starts["split_first"]
    assert starts["saved_step2"] == starts["restored_step2"]
    comparison = json.loads((tmp_path / "resume_state_comparison.json").read_text())
    assert comparison["lora"]["passed"]
    assert comparison["optimizer"]["passed"]
    assert (tmp_path / "resumed_checkpoint_000004" / "checkpoint.json").is_file()


def test_overfit_core_reports_before_after_and_saves_full_checkpoint(tmp_path):
    module = gates_module()
    order, load_sample = six_samples()
    result = module.train_overfit_cached(driver_model(), order, load_sample, scheduler(), tmp_path,
        metadata(), updates=2, device="cpu", expected_parameters=None)
    assert result["optimizer_step"] == 2
    assert result["evaluated_positions"] == 18
    assert result["gate_pass"] == (result["relative_loss_drop"] >= .2)
    assert result["loss_before"] > 0 and result["loss_after"] > 0
    assert (tmp_path / "checkpoint_000002" / "checkpoint.json").is_file()


def test_gate_requires_exact_six_balanced_unique_train_samples(tmp_path):
    module = gates_module()
    data = tmp_path / "data"
    data.mkdir()
    rows = [{"sample_id": f"{dataset}{index}", "dataset": dataset, "split": "train"}
            for dataset in ("iper", "tiktok", "hi4d") for index in range(2)]
    for name in ("overfit6_conditions.jsonl", "overfit6_supervision.jsonl"):
        (data / name).write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    order = module.overfit_sample_order(tmp_path)
    assert len(order) == 6
    assert [dataset for dataset, _ in order] == ["iper", "tiktok", "hi4d"] * 2
    rows[0]["split"] = "val"
    (data / "overfit6_conditions.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="train"):
        module.overfit_sample_order(tmp_path)


def test_state_comparison_reports_all_differences_before_asserting():
    module = gates_module()
    assert hasattr(module, "state_comparison_report"), "full CPU comparison diagnostics are missing"
    left = {"first": torch.zeros(3), "later": [torch.ones(2), {"step": 4}]}
    right = {"first": torch.tensor([0., .01, 0.]), "later": [torch.tensor([1., 1.2]), {"step": 4}]}
    result = module.state_comparison_report(left, right)
    assert not result["passed"]
    assert result["compared_tensors"] == 2
    assert result["mismatched_tensors"] == 2
    assert result["mismatched_elements"] == 2
    assert result["max_abs_error"] == pytest.approx(.2)
    assert result["worst_tensor"] == "state.later.0"
    assert {entry["path"] for entry in result["tensor_differences"]} == {"state.first", "state.later.0"}
    json.dumps(result, allow_nan=False)


def test_failed_resume_keeps_final_split_checkpoint_and_full_comparison(tmp_path):
    module = gates_module()
    model = driver_model()
    order, original_loader = six_samples()
    calls = {"count": 0}
    def changed_loader(sample_id):
        calls["count"] += 1
        condition, supervision = original_loader(sample_id)
        if calls["count"] > 12:
            supervision = dict(supervision, target_latent=supervision["target_latent"] + .25)
        return condition, supervision
    with pytest.raises(AssertionError, match="resume"):
        module.check_resume_equivalence(model, order, changed_loader, scheduler(), tmp_path,
            metadata(), device="cpu", expected_parameters=None)
    assert (tmp_path / "resumed_checkpoint_000004" / "state.pt").is_file()
    comparison = json.loads((tmp_path / "resume_state_comparison.json").read_text())
    assert not comparison["passed"]
    assert comparison["lora"]["mismatched_tensors"] > 0
    assert comparison["lora"]["max_abs_error"] > 1e-6
