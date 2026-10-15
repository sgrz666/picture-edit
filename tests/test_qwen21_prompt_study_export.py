from __future__ import annotations

import json
from pathlib import Path
import tarfile
import pytest

from src.pose_control.qwen21_full.prompt_study.results import (
    build_export_manifest,
    package_export,
    verify_report_relative_links,
)


def test_export_includes_allowed_and_excludes_prohibited(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    # Create allowed directories and files
    (run_dir / "data").mkdir()
    (run_dir / "data" / "lock.json").write_text("{}", encoding="utf-8")

    (run_dir / "prompts").mkdir()
    (run_dir / "prompts" / "p0.jsonl").write_text('{"prompt": "test"}\n', encoding="utf-8")
    (run_dir / "prompts" / "input_assets").mkdir()
    (run_dir / "prompts" / "input_assets" / "in.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    (run_dir / "training" / "qwen_p0").mkdir(parents=True)
    (run_dir / "training" / "qwen_p0" / "metrics.jsonl").write_text('{"step": 1}\n', encoding="utf-8")
    (run_dir / "training" / "qwen_p0" / "checkpoint.pt").write_bytes(b"MODEL_WEIGHTS")
    (run_dir / "training" / "qwen_p0" / "model.safetensors").write_bytes(b"SAFETENSORS")

    (run_dir / "cache").mkdir()
    (run_dir / "cache" / "latent.pt").write_bytes(b"CACHED_LATENTS")
    (run_dir / "cache" / "info.json").write_text('{"cache": true}\n', encoding="utf-8")

    (run_dir / "evaluation" / "qwen_p0_step_0" / "images").mkdir(parents=True)
    out_img = run_dir / "evaluation" / "qwen_p0_step_0" / "images" / "out.png"
    out_img.write_bytes(b"\x89PNG\r\n\x1a\n")

    # Report with valid relative links
    (run_dir / "report").mkdir()
    (run_dir / "report" / "index.html").write_text(
        '<html><body><a href="../evaluation/qwen_p0_step_0/images/out.png">Link</a>'
        '<img src="../prompts/input_assets/in.png"></body></html>',
        encoding="utf-8",
    )

    # Test manifest
    manifest = build_export_manifest(run_dir)
    paths = {m["path"] for m in manifest}

    assert "data/lock.json" in paths
    assert "prompts/p0.jsonl" in paths
    assert "prompts/input_assets/in.png" in paths
    assert "training/qwen_p0/metrics.jsonl" in paths
    assert "evaluation/qwen_p0_step_0/images/out.png" in paths
    assert "report/index.html" in paths

    # Must NOT contain .pt, .safetensors, or cache
    for p in paths:
        assert not p.endswith(".pt")
        assert not p.endswith(".safetensors")
        assert "cache" not in p.split("/")

    # Run packaging
    res = package_export(run_dir)
    tar_path = Path(res["tar_path"])
    assert tar_path.is_file()

    # Verify tar members
    with tarfile.open(tar_path, "r:gz") as tar:
        names = tar.getnames()
        assert "data/lock.json" in names
        assert "files.json" in names
        assert "report/index.html" in names
        for name in names:
            assert not name.startswith("/")
            assert not name.startswith("..")
            assert not name.startswith("staging/")
            assert not name.endswith(".pt")
            assert not name.endswith(".safetensors")
            assert "cache" not in name.split("/")


def test_export_fails_on_broken_relative_links(tmp_path: Path):
    run_dir = tmp_path / "broken_run"
    run_dir.mkdir()
    (run_dir / "data").mkdir()
    (run_dir / "data" / "lock.json").write_text("{}", encoding="utf-8")
    (run_dir / "report").mkdir()
    (run_dir / "report" / "index.html").write_text(
        '<html><body><a href="missing_nonexistent_image.png">Broken</a></body></html>',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Report links verification failed"):
        package_export(run_dir)
