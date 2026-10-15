from __future__ import annotations

import sys
from pathlib import Path
import pytest

from src.pose_control.qwen21_full.prompt_study.sampling import (
    make_order,
    noise_seed,
    validate_trajectory,
)
import scripts.train.train_qwen21_joint_pilot as train_pilot


def test_resume_at_epoch_boundary_uses_locked_permutation():
    order = make_order()
    saved = {"step": 512, "sample_order": order[:512]}
    rows = [
        dict(step=u + 1, pair_index=order[u], noise_seed=noise_seed(u))
        for u in range(512)
    ]
    validate_trajectory(saved, rows, order)
    assert order[511] == 511
    assert order[512] == make_order()[512]
    rows[-1]["noise_seed"] += 1
    with pytest.raises(ValueError, match="trajectory"):
        validate_trajectory(saved, rows, order)


def test_argument_parsing_study_vs_legacy(monkeypatch):
    base_args = [
        "train.py",
        "--train-pairs", "train.jsonl",
        "--val-pairs", "val.jsonl",
        "--features", "features",
        "--face-supervision", "face_sup.json",
        "--cache", "cache",
        "--output", "output",
        "--init-checkpoint", "init.pt",
        "--gradient-only",
    ]

    # Legacy: steps 1024 should fail
    monkeypatch.setattr(sys, "argv", base_args + ["--steps", "1024"])
    with pytest.raises(SystemExit):
        train_pilot.arguments()

    # Legacy: steps 256 succeeds
    monkeypatch.setattr(sys, "argv", base_args + ["--steps", "256"])
    args = train_pilot.arguments()
    assert args.steps == 256

    # Study: steps 256 should fail
    study_args = base_args + [
        "--study-lock", "lock.json",
        "--arm", "qwen_p0",
        "--prompt-manifest", "prompts.jsonl",
        "--resolution", "native",
        "--align-mode", "native",
        "--noise-offset", "256",
        "--seed", "42",
    ]
    monkeypatch.setattr(sys, "argv", study_args + ["--steps", "256"])
    with pytest.raises(SystemExit):
        train_pilot.arguments()

    # Study: steps 1024 succeeds and supports stop-after-step
    monkeypatch.setattr(
        sys,
        "argv",
        study_args + ["--steps", "1024", "--stop-after-step", "128"],
    )
    args = train_pilot.arguments()
    assert args.steps == 1024
    assert args.stop_after_step == 128
    assert args.arm == "qwen_p0"
