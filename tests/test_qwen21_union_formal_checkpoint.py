import hashlib
import json
import pytest


def checkpoint(tmp_path, monkeypatch, *, step=512, protocol=None):
    from src.pose_control.qwen21_union import runner
    binding = {"base_fingerprint": "base", "union_fingerprint": "union", "initial_lora_sha256": "initial"}
    monkeypatch.setattr(runner, "training_binding", lambda work: binding)
    monkeypatch.setattr(runner, "formal_trajectory_sha256", lambda work: "trajectory")
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "state.pt").write_bytes(b"signed-state")
    metadata = dict(**binding, scope="joint", training_protocol=protocol or runner.formal_training_protocol(), trajectory_sha256="trajectory")
    cursor = dict(optimizer_step=step, accumulation_steps=3, microstep=step * 3,
                  next_sample_position=step * 3, loss_log_cursor=step * 3, gradient_log_cursor=step)
    manifest = dict(format_version="qwen21_union_training_checkpoint_v1", metadata=metadata, cursor=cursor,
                    state_sha256=hashlib.sha256(b"signed-state").hexdigest())
    (root / "checkpoint.json").write_text(json.dumps(manifest))
    return runner, root


@pytest.mark.parametrize("step", [0, 128, 256])
def test_formal_inference_refuses_nonfinal_weights(tmp_path, monkeypatch, step):
    runner, path = checkpoint(tmp_path, monkeypatch, step=step)
    with pytest.raises(ValueError, match="512"):
        runner.validate_formal_inference_checkpoint(path, "joint", tmp_path)


def test_formal_inference_rejects_overfit_protocol_and_corruption(tmp_path, monkeypatch):
    runner, path = checkpoint(tmp_path, monkeypatch, protocol={"kind": "overfit6"})
    with pytest.raises(ValueError, match="protocol"):
        runner.validate_formal_inference_checkpoint(path, "joint", tmp_path)


def test_formal_checkpoint_verifies_payload_bytes_and_scope(tmp_path, monkeypatch):
    runner, path = checkpoint(tmp_path, monkeypatch)
    assert runner.validate_formal_inference_checkpoint(path, "joint", tmp_path)["cursor"]["optimizer_step"] == 512
    with pytest.raises(ValueError, match="scope"):
        runner.validate_formal_inference_checkpoint(path, "target", tmp_path)
    (path / "state.pt").write_bytes(b"replaced")
    with pytest.raises(ValueError, match="checksum"):
        runner.validate_formal_inference_checkpoint(path, "joint", tmp_path)
