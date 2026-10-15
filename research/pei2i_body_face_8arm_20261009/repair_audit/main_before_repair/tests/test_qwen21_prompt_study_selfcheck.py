from __future__ import annotations

import copy
from pathlib import Path
import pytest
import torch

from src.pose_control.qwen21_full.prompt_study.results import compare_private_continuations
import scripts.run.run_qwen21_pei2i_study as runner


def test_compare_private_continuations():
    weight = torch.tensor([1.0, 2.0, 3.0])
    state = {
        "step": 4,
        "sample_order": [0, 1, 2, 3],
        "joint_signature": {"model": "test"},
        "joint_initialization_sha256": "init_sha",
        "adapter_state_dict": {"fc.weight": weight},
        "optimizer_state_dict": {"state": {}},
    }
    rows = [
        {
            "step": i + 1,
            "pair_index": i,
            "pair_id": f"p{i}",
            "noise_seed": 256 + 42 + i,
            "prompt_sha256": "p_sha",
            "sigma": 0.5,
            "face_active": True,
            "loss": 1.0,
            "body_flow": 0.8,
            "face_flow": 0.2,
            "identity": 0.0,
            "perceptual": 0.0,
        }
        for i in range(4)
    ]

    left_state = copy.deepcopy(state)
    right_state = copy.deepcopy(state)
    left_rows = copy.deepcopy(rows)
    right_rows = copy.deepcopy(rows)

    res = compare_private_continuations(left_state, right_state, left_rows, right_rows)
    assert res["adapter"]["max_abs"] == pytest.approx(0.0)
    assert res["errors"] == []

    # If steps != 4, raises ValueError
    bad_state = copy.deepcopy(state)
    bad_state["step"] = 3
    with pytest.raises(ValueError, match="Private continuation must end at four updates"):
        compare_private_continuations(bad_state, right_state, left_rows, right_rows)


def test_run_selfcheck_fails_on_missing_lock(tmp_path: Path):
    with pytest.raises(RuntimeError, match="Selfcheck failed"):
        runner.run_selfcheck(tmp_path)
