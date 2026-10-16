import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest


def execution():
    path = Path(__file__).resolve().parents[1] / "experiments/qwen21_union_dataset_adaptation_v1/scripts/execute.py"
    spec = importlib.util.spec_from_file_location("union_execution_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failed_phase_never_starts_later_gpu_work(tmp_path, monkeypatch):
    module = execution()
    calls = []
    def fail(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=7)
    monkeypatch.setattr(module.subprocess, "run", fail)
    monkeypatch.setattr(sys, "argv", ["execute.py", "--workspace", str(tmp_path), "--name", "bounded"])
    with pytest.raises(RuntimeError, match="audit failed"):
        module.main()
    assert len(calls) == 1
    events = [json.loads(line) for line in (tmp_path / "queues/bounded/events.jsonl").read_text().splitlines()]
    assert events[-1]["status"] == "stopped"
    assert not any(event.get("stage") == "overfit" for event in events)


def test_gate_failure_is_not_a_successful_process_exit(tmp_path):
    module = execution()
    gate = tmp_path / "runs/overfit/gate.json"
    gate.parent.mkdir(parents=True)
    gate.write_text(json.dumps(dict(kind="overfit", gate_pass=False)))
    with pytest.raises(RuntimeError, match="gate failed"):
        module.read_gate(tmp_path, "overfit", "overfit")
