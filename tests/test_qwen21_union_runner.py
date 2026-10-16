from pathlib import Path
import pytest


def test_run_directory_refuses_overwrite(tmp_path):
    from src.pose_control.qwen21_union.runner import new_run_directory
    first = new_run_directory(tmp_path, "smoke_20261016")
    (first / "evidence.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        new_run_directory(tmp_path, "smoke_20261016")
    assert (first / "evidence.txt").read_text() == "keep"


def test_run_name_cannot_escape_workspace(tmp_path):
    from src.pose_control.qwen21_union.runner import new_run_directory
    for bad in ("../outside", "/absolute", "a/b", "a\\b", ".", ""):
        with pytest.raises(ValueError):
            new_run_directory(tmp_path, bad)


def test_report_does_not_claim_unfinished_training_completed(tmp_path):
    from src.pose_control.qwen21_union.runner import summarize_progress
    summary = summarize_progress(tmp_path)
    assert summary["completed_training_arms"] == []
    assert summary["test_images"] == 0
    assert summary["complete"] is False


def test_progress_deduplicates_outputs_and_checks_actual_files(tmp_path):
    import json
    from src.pose_control.qwen21_union.runner import summarize_progress
    image = tmp_path / "one.png"
    image.write_bytes(b"evidence")
    import hashlib
    row = dict(sample_id="one", dataset="iper", arm="native", seed=42, steps=40, out=str(image), image_sha256=hashlib.sha256(image.read_bytes()).hexdigest())
    for name in ("first", "duplicate"):
        run = tmp_path / "runs" / name
        run.mkdir(parents=True)
        (run / "summary.json").write_text(json.dumps(dict(kind="infer", split="test", produced=1024)))
        (run / "records.jsonl").write_text(json.dumps(row) + "\n")
    assert summarize_progress(tmp_path)["test_images"] == 1
    image.write_bytes(b"replaced")
    assert summarize_progress(tmp_path)["test_images"] == 0


def test_inference_protocol_rejects_typos_and_incomplete_formal_matrix():
    from src.pose_control.qwen21_union.runner import validate_inference_protocol
    with pytest.raises(ValueError, match="arm"):
        validate_inference_protocol("native,unoin_frozen", "42,43", "val", 40, None)
    with pytest.raises(ValueError, match="formal"):
        validate_inference_protocol("native", "42", "test", 40, None)
    arms, seeds = validate_inference_protocol("native,union_frozen,union_lora_joint,union_lora_target", "42,43", "test", 40, None)
    assert len(arms) == 4 and seeds == [42, 43]
    assert validate_inference_protocol("union_zero", "42", "val", 2, 2) == (["union_zero"], [42])


def test_missing_lora_checkpoint_fails_before_data_or_model_loading():
    from types import SimpleNamespace
    from src.pose_control.qwen21_union.runner import command_infer
    args = SimpleNamespace(arms="union_lora_joint", seeds="42", split="val", steps=40, limit=2,
                           joint_checkpoint=None, target_checkpoint=None)
    with pytest.raises(ValueError, match="explicit checkpoint"):
        command_infer(args)
