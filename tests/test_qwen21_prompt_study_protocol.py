from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.pose_control.qwen21_full.prompt_study.protocol import (
    ARMS,
    PROTOCOLS,
    SCHEMA,
    P0_PROMPT,
    PROMPT_ROLES,
    INIT_SHA256,
    TRAIN_SHA256,
    VAL_SHA256,
    validate_arm,
    sha256_file,
    text_sha256,
    atomic_json,
    validate_study_config,
)
from src.pose_control.qwen21_full.prompt_study.sampling import (
    make_order,
    noise_seed,
    warmup_scale,
    validate_trajectory,
)


def test_eight_arms_and_two_complete_epochs():
    assert ARMS == tuple(
        f"{base}_{p}" for base in ("qwen", "head") for p in ("p0", "pg", "pb", "pbg")
    )
    order = make_order()
    assert len(order) == 1024
    assert order[:512] == list(range(512))
    assert sorted(order[512:]) == list(range(512))
    assert make_order() == order
    assert (noise_seed(0), noise_seed(1023)) == (298, 1321)


def test_invalid_arm_rejected_before_loading_models():
    with pytest.raises(ValueError, match="arm"):
        validate_arm("head_punknown")
    base, proto = validate_arm("qwen_pbg")
    assert base == "qwen"
    assert proto == "pbg"
    base, proto = validate_arm("head_p0")
    assert base == "head"
    assert proto == "p0"


def test_protocol_constants_integrity():
    assert SCHEMA == "qwen21_pei2i_body_face_study_v1"
    assert PROTOCOLS == ("p0", "pg", "pb", "pbg")
    assert len(P0_PROMPT) > 20
    assert PROMPT_ROLES["p0"] == ()
    assert "source_A" in PROMPT_ROLES["pg"]
    assert "target_B" not in PROMPT_ROLES["pg"]
    assert "target_B" in PROMPT_ROLES["pb"]
    assert len(INIT_SHA256) == 64
    assert len(TRAIN_SHA256) == 64
    assert len(VAL_SHA256) == 64


def test_warmup_scale():
    assert warmup_scale(0) == pytest.approx(1 / 20)
    assert warmup_scale(18) == pytest.approx(19 / 20)
    assert warmup_scale(19) == pytest.approx(1.0)
    assert warmup_scale(100) == pytest.approx(1.0)


def test_noise_seed_bounds():
    with pytest.raises(ValueError, match="update_index"):
        noise_seed(-1)
    with pytest.raises(ValueError, match="update_index"):
        noise_seed(1024)


def test_validate_trajectory_valid_and_corrupt():
    order = make_order()
    saved = {"step": 10, "sample_order": order[:10]}
    metrics = [
        {"step": i + 1, "pair_index": order[i], "noise_seed": noise_seed(i)}
        for i in range(10)
    ]
    validate_trajectory(saved, metrics, order)

    corrupt_metrics = [dict(m) for m in metrics]
    corrupt_metrics[5]["noise_seed"] += 1
    with pytest.raises(ValueError, match="Resume trajectory differs"):
        validate_trajectory(saved, corrupt_metrics, order)


def test_atomic_json_and_sha256(tmp_path: Path):
    target = tmp_path / "subdir" / "test.json"
    data = {"hello": "world", "numbers": [1, 2, 3]}
    atomic_json(target, data)
    assert target.is_file()
    assert json.loads(target.read_text(encoding="utf-8")) == data
    h1 = sha256_file(target)
    h2 = text_sha256(target.read_bytes().decode("utf-8"))
    assert h1 == h2


def test_validate_study_config():
    valid = {
        "schema": SCHEMA,
        "max_updates": 1024,
        "eval_every": 128,
        "resolution": "native",
        "align_mode": "native",
        "with_hand": False,
        "hand_active": False,
        "use_depth": False,
        "arms": ARMS,
    }
    validate_study_config(valid)

    bad = dict(valid, max_updates=256)
    with pytest.raises(ValueError, match="max_updates"):
        validate_study_config(bad)

    bad_hand = dict(valid, with_hand=True)
    with pytest.raises(ValueError, match="Hand"):
        validate_study_config(bad_hand)


def test_final_gate_rejects_summaries_without_training_evidence(tmp_path):
    from src.pose_control.qwen21_full.prompt_study.protocol import require_eight_arms_completed
    for arm in ARMS:
        atomic_json(tmp_path / 'training' / arm / 'summary.json', dict(completed=True, steps=1024))
    with pytest.raises(RuntimeError):
        require_eight_arms_completed(tmp_path)


def test_binding_rejects_head_weights_on_plain_qwen(tmp_path):
    from src.pose_control.qwen21_full.prompt_study.protocol import validate_study_binding
    lock = {'artifacts': {'model': {'path': str(tmp_path)}, 'head_lora': {'sha256': 'a' * 64}}}
    with pytest.raises(ValueError, match='LoRA'):
        validate_study_binding(lock, 'qwen_pg', 'a' * 64, tmp_path)
    with pytest.raises(ValueError, match='Head'):
        validate_study_binding(lock, 'head_pg', None, tmp_path)


def test_directory_artifact_detects_changed_weight(tmp_path):
    from src.pose_control.qwen21_full.prompt_study.protocol import directory_artifact, verify_artifact
    p = tmp_path / 'model.safetensors'
    p.write_bytes(b'original')
    artifact = directory_artifact(tmp_path)
    p.write_bytes(b'changed')
    with pytest.raises(ValueError, match='artifact'):
        verify_artifact(artifact)
