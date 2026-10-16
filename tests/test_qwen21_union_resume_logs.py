"""Resume logs inherit only the checkpoint's complete, contiguous prefix."""
import copy
import hashlib
import importlib
import importlib.util
import json

import pytest
import torch

from test_qwen21_union_checkpoint import metadata
from test_qwen21_union_training import cached_pair, driver_model, scheduler, training_module
from test_qwen21_union_lora import weights_module


def resume_module():
    name = "src.pose_control.qwen21_union.resume_logs"
    assert importlib.util.find_spec(name) is not None, "resume log prefix implementation is missing"
    return importlib.import_module(name)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def append(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value) + "\n")


@pytest.fixture
def completed_parent(tmp_path):
    from src.pose_control.qwen21_union.checkpoint import save_checkpoint
    parent = tmp_path / "parent"
    parent.mkdir()
    model = driver_model()
    initial = copy.deepcopy(model.state_dict())
    pair = cached_pair()
    order = tuple((dataset, dataset + str(index)) for index in range(4)
                  for dataset in ("iper", "tiktok", "hi4d"))
    module = training_module()
    optimizer = module.make_optimizer(model, expected_parameters=None)
    lr = module.make_lr_scheduler(optimizer)
    saved = {}
    def on_update(event):
        append(parent / "updates.jsonl", {key: value for key, value in event.items() if key != "cursor"})
        if event["optimizer_step"] == 2:
            saved["cursor"] = event["cursor"]
            saved["checkpoint"] = save_checkpoint(parent / "checkpoint_000002", model, optimizer, lr,
                                                   metadata=metadata(), cursor=event["cursor"])
    module.run_training_updates(model, optimizer=optimizer, lr_scheduler=lr, scheduler=scheduler(),
        sample_order=order, load_sample=lambda sid: pair, updates=4, device="cpu", expected_parameters=None,
        on_microstep=lambda event: append(parent / "microsteps.jsonl", event), on_update=on_update)
    return dict(parent=parent, model=model, initial=initial, pair=pair, order=order, **saved)


def test_four_updates_vs_resume_two_inherits_prefix_and_appends_global_history(completed_parent, tmp_path):
    module = resume_module()
    from src.pose_control.qwen21_union.checkpoint import load_checkpoint
    case = completed_parent
    destination = tmp_path / "resumed"
    destination.mkdir()
    original = {name: (case["parent"] / name).read_bytes() for name in ("microsteps.jsonl", "updates.jsonl")}
    provenance = module.prepare_resume_logs(destination, case["checkpoint"], case["cursor"])
    assert provenance["parent_checkpoint"] == str(case["checkpoint"].resolve())
    assert provenance["state_sha256"] == sha((case["checkpoint"] / "state.pt").read_bytes())
    assert (destination / "resume_log_provenance.json").is_file()
    for name, count in (("microsteps.jsonl", 6), ("updates.jsonl", 2)):
        prefix = b"".join(original[name].splitlines(keepends=True)[:count])
        assert (destination / name).read_bytes() == prefix
        assert provenance["logs"][name]["parent_log_sha256"] == sha(original[name])
        assert provenance["logs"][name]["selected_prefix_sha256"] == sha(prefix)
        assert provenance["logs"][name]["selected_lines"] == count
        assert (case["parent"] / name).read_bytes() == original[name]
    model = driver_model()
    model.load_state_dict(case["initial"])
    optimizer = training_module().make_optimizer(model, expected_parameters=None)
    lr = training_module().make_lr_scheduler(optimizer)
    cursor = load_checkpoint(case["checkpoint"], model, optimizer, lr, expected_metadata=metadata())
    training_module().run_training_updates(model, optimizer=optimizer, lr_scheduler=lr, scheduler=scheduler(),
        sample_order=case["order"], load_sample=lambda sid: case["pair"], updates=4, cursor=cursor,
        device="cpu", expected_parameters=None,
        on_microstep=lambda event: append(destination / "microsteps.jsonl", event),
        on_update=lambda event: append(destination / "updates.jsonl", {key: value for key, value in event.items() if key != "cursor"}))
    for name, count in (("microsteps.jsonl", 12), ("updates.jsonl", 4)):
        rows = [json.loads(line) for line in (destination / name).read_text().splitlines()]
        continuous = [json.loads(line) for line in original[name].splitlines()]
        assert len(rows) == count
        keys = ("optimizer_step", "loss", "gradient_norm", "lr") if name == "updates.jsonl" else ("position", "update", "microstep", "sample_id", "index", "noise_seed", "loss")
        assert [{key: row[key] for key in keys} for row in rows] == [{key: row[key] for key in keys} for row in continuous]
    for key, value in weights_module().lora_state_dict(case["model"]).items():
        torch.testing.assert_close(value, weights_module().lora_state_dict(model)[key], rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["missing", "short", "partial_line", "bad_json", "gap_micro", "gap_update"])
def test_resume_logs_refuse_missing_truncated_or_discontinuous_prefix(completed_parent, tmp_path, kind):
    module = resume_module()
    case = completed_parent
    target = case["parent"] / ("updates.jsonl" if kind == "gap_update" else "microsteps.jsonl")
    lines = target.read_bytes().splitlines(keepends=True)
    if kind == "missing":
        # Use a never-created parent log path by changing only a dedicated fixture directory.
        alternate = tmp_path / "missing_parent"
        alternate.mkdir()
        checkpoint = alternate / "checkpoint_000002"
        checkpoint.mkdir()
        for name in ("checkpoint.json", "state.pt"):
            (checkpoint / name).write_bytes((case["checkpoint"] / name).read_bytes())
    else:
        checkpoint = case["checkpoint"]
        if kind == "short":
            target.write_bytes(b"".join(lines[:5]))
        elif kind == "partial_line":
            target.write_bytes(b"".join(lines[:5]) + lines[5].rstrip(b"\r\n"))
        elif kind == "bad_json":
            target.write_bytes(b"".join(lines[:5]) + b"{broken}\n")
        else:
            record = json.loads(lines[1])
            record["optimizer_step" if kind == "gap_update" else "position"] += 1
            lines[1] = (json.dumps(record) + "\n").encode()
            target.write_bytes(b"".join(lines))
    destination = tmp_path / "resumed"
    destination.mkdir()
    with pytest.raises((ValueError, FileNotFoundError)):
        module.prepare_resume_logs(destination, checkpoint, case["cursor"])
    assert not any(destination.iterdir()), "all source validation must precede destination writes"


@pytest.mark.parametrize("name", ["microsteps.jsonl", "updates.jsonl", "resume_log_provenance.json"])
def test_resume_logs_refuse_overwrite_before_creating_any_files(completed_parent, tmp_path, name):
    module = resume_module()
    destination = tmp_path / "resumed"
    destination.mkdir()
    (destination / name).write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        module.prepare_resume_logs(destination, completed_parent["checkpoint"], completed_parent["cursor"])
    assert sorted(path.name for path in destination.iterdir()) == [name]
    assert (destination / name).read_text() == "keep"


def test_resume_logs_verify_checkpoint_hash_and_cursor(completed_parent, tmp_path):
    module = resume_module()
    case = completed_parent
    destination = tmp_path / "resumed"
    destination.mkdir()
    with (case["checkpoint"] / "state.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        module.prepare_resume_logs(destination, case["checkpoint"], case["cursor"])
    assert not any(destination.iterdir())


def test_step_zero_allows_absent_parent_logs_and_creates_empty_history(tmp_path):
    module = resume_module()
    from src.pose_control.qwen21_union.checkpoint import TrainingCursor, save_checkpoint
    parent = tmp_path / "parent"
    parent.mkdir()
    model = driver_model()
    optimizer = training_module().make_optimizer(model, expected_parameters=None)
    lr = training_module().make_lr_scheduler(optimizer)
    checkpoint = save_checkpoint(parent / "checkpoint_000000", model, optimizer, lr,
                                 metadata=metadata(), cursor=TrainingCursor())
    destination = tmp_path / "resumed"
    destination.mkdir()
    provenance = module.prepare_resume_logs(destination, checkpoint, TrainingCursor())
    assert (destination / "microsteps.jsonl").read_bytes() == b""
    assert (destination / "updates.jsonl").read_bytes() == b""
    assert provenance["logs"]["microsteps.jsonl"]["parent_log_sha256"] is None


def test_later_malformed_parent_rows_are_ignored_after_checkpoint_prefix(completed_parent, tmp_path):
    module = resume_module()
    case = completed_parent
    for name in ("microsteps.jsonl", "updates.jsonl"):
        with (case["parent"] / name).open("ab") as stream:
            stream.write(b"later incomplete JSON")
    destination = tmp_path / "resumed"
    destination.mkdir()
    result = module.prepare_resume_logs(destination, case["checkpoint"], case["cursor"])
    assert result["logs"]["microsteps.jsonl"]["selected_lines"] == 6
    assert result["logs"]["updates.jsonl"]["selected_lines"] == 2


def test_resume_logs_refuse_manifest_cursor_mismatch(completed_parent, tmp_path):
    module = resume_module()
    from src.pose_control.qwen21_union.checkpoint import TrainingCursor
    destination = tmp_path / "resumed"
    destination.mkdir()
    wrong = TrainingCursor(optimizer_step=1, microstep=3, next_sample_position=3,
                           loss_log_cursor=3, gradient_log_cursor=1)
    with pytest.raises(ValueError, match="cursor"):
        module.prepare_resume_logs(destination, completed_parent["checkpoint"], wrong)
    assert not any(destination.iterdir())


def test_resume_logs_require_a_run_outside_parent_run(completed_parent):
    module = resume_module()
    destination = completed_parent["parent"] / "nested_resume"
    destination.mkdir()
    with pytest.raises(ValueError, match="separate"):
        module.prepare_resume_logs(destination, completed_parent["checkpoint"], completed_parent["cursor"])
    assert not any(destination.iterdir())
