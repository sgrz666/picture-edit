from __future__ import annotations

import json
import sys
from pathlib import Path
import pytest

from src.pose_control.qwen21_full.prompt_study.protocol import ARMS
from src.pose_control.qwen21_full.prompt_study.results import eval_jobs
import scripts.run.run_qwen21_pei2i_study as runner


def test_fixed_generation_budget_and_no_duplicate_jobs():
    jobs = eval_jobs(include_final_test=True)
    assert len(jobs) == 3696
    keys = {
        (j["arm"], j["step"], j["split"], j["pair_index"], j["seed"])
        for j in jobs
    }
    assert len(keys) == len(jobs)
    assert len(eval_jobs(include_final_test=False)) == 2928
    for arm in ARMS:
        assert sum(j["arm"] == arm for j in jobs) == 462


def test_runner_finaltest_gate_rejects_incomplete_training(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--run", str(tmp_path), "--phase", "finaltest"],
    )
    with pytest.raises(RuntimeError, match="Final test requires all 8 arms"):
        runner.main()


def test_runner_test_split_rewrite_rejects_incomplete_training(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--run", str(tmp_path), "--phase", "rewrite", "--splits", "test"],
    )
    with pytest.raises(RuntimeError, match="Cannot process test"):
        runner.main()


def test_runner_dry_run_executes_cleanly(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--run", str(tmp_path), "--phase", "initial", "--dry-run"],
    )
    runner.main()
    out = capsys.readouterr().out
    assert "[DRY-RUN]" in out
    assert "initial" in out


def test_runner_dispatches_initial_phase_correctly(tmp_path: Path, monkeypatch):
    invoked_cmds = []

    def mock_invoke(argv, dry_run=False):
        invoked_cmds.append(list(argv))

    monkeypatch.setattr(runner, "invoke", mock_invoke)
    monkeypatch.setattr(runner, 'require_preflight', lambda *a, **k: None)
    monkeypatch.setattr(runner, 'locked_assets', lambda _: {'model': {'path': '/locked/model'}, 'initial_checkpoint': {'path': '/locked/init'}, 'head_lora': {'path': '/locked/head', 'sha256': 'head'}})
    eval_calls = []
    monkeypatch.setattr(runner, 'dispatch_eval', lambda *a, **k: eval_calls.append(a))
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--run", str(tmp_path), "--phase", "initial", "--arms", "qwen_p0"],
    )
    runner.main()

    # Step 0 train must be dispatched with stop-after-step 0
    assert len(invoked_cmds) > 0
    train_cmd = invoked_cmds[0]
    assert "--stop-after-step" in train_cmd
    assert train_cmd[train_cmd.index("--stop-after-step") + 1] == "0"
    assert "--arm" in train_cmd
    assert train_cmd[train_cmd.index("--arm") + 1] == "qwen_p0"

    # Subsequent commands should be eval jobs for step 0
    assert {args[3] for args in eval_calls} == {'train', 'val', 'diagnostic'}
    assert train_cmd[train_cmd.index('--model') + 1] == '/locked/model'
    assert train_cmd[train_cmd.index('--init-checkpoint') + 1] == '/locked/init'
    # Must NOT include test split in initial phase!
    assert not any(args[3] == 'test' for args in eval_calls)


def test_runner_dispatches_train_and_finaltest(tmp_path: Path, monkeypatch):
    # Set up all 8 arms completed summaries for finaltest
    for arm in ARMS:
        arm_dir = tmp_path / f"training/{arm}"
        arm_dir.mkdir(parents=True, exist_ok=True)
        (arm_dir / "summary.json").write_text(
            json.dumps({"completed": True, "steps": 1024}),
            encoding="utf-8",
        )

    invoked_cmds = []

    def mock_invoke(argv, dry_run=False):
        invoked_cmds.append(list(argv))

    monkeypatch.setattr(runner, "invoke", mock_invoke)
    monkeypatch.setattr(runner, 'check_eight_arms_completed', lambda _: True)
    monkeypatch.setattr(runner, 'require_preflight', lambda *a, **k: None)
    eval_calls = []
    monkeypatch.setattr(runner, 'dispatch_eval', lambda *a, **k: eval_calls.append(a))
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--run", str(tmp_path), "--phase", "finaltest", "--arms", "qwen_p0"],
    )
    runner.main()

    # finaltest should dispatch eval jobs for test split at steps 0 and 1024
    assert len(eval_calls) == 4
    assert {(args[2], args[3], args[5]) for args in eval_calls} == {(step, 'test', seed) for step in (0, 1024) for seed in (42, 43)}
