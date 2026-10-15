from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.pose_control.qwen21_full.prompt_study.protocol import P0_PROMPT
from src.pose_control.qwen21_full.prompt_study.prompts import (
    request_roles,
    parse_answer,
    compile_prompt,
    make_p0_record,
    PromptBank,
)


def test_pg_never_requests_B_or_manual_caption():
    row = dict(
        source_rgb="a.png",
        target_rgb="SECRET_B.png",
        target_depth_vis="d.png",
        target_normal="n.png",
        filtered_dwpose="p.png",
        action_description="SECRET_ACTION",
    )
    req = request_roles(row, "pg")
    assert req == [
        ("source_A", "a.png"),
        ("target_depth", "d.png"),
        ("target_normal", "n.png"),
        ("target_dwpose", "p.png"),
    ]
    assert "SECRET" not in str(req)


def test_compiled_text_has_no_unprovided_image_tokens():
    raw = "<image1>: Raise both arms. Follow <image2>. Preserve clothing."
    result = compile_prompt(raw, ("source_A", "target_B"))
    assert "<image" not in result and "Raise both arms" in result
    with pytest.raises(ValueError, match="image"):
        compile_prompt("Use <image9>", ("source_A", "target_B"))
    assert (
        parse_answer('thinking</think>{"rewritten_prompt":"Raise both arms"}')[
            "rewritten_prompt"
        ]
        == "Raise both arms"
    )


def test_parse_answer_fences_and_empty():
    fenced = "```json\n{\"rewritten_prompt\": \"Bend forward slightly.\"}\n```"
    assert parse_answer(fenced)["rewritten_prompt"] == "Bend forward slightly."

    with pytest.raises(ValueError, match="Missing nonempty"):
        parse_answer('{"rewritten_prompt": ""}')


def test_prompt_bank_loads_valid_and_rejects_invalid(tmp_path: Path):
    manifest = tmp_path / "train.jsonl"
    r1 = make_p0_record("pair_0", "train")
    r2 = make_p0_record("pair_1", "train")
    manifest.write_text(
        json.dumps(r1) + "\n" + json.dumps(r2) + "\n",
        encoding="utf-8",
    )

    bank = PromptBank(manifest, "p0", ["pair_0", "pair_1"])
    assert bank.prompt_for("pair_0") == P0_PROMPT
    assert bank.prompt_for("pair_1") == P0_PROMPT

    # Reject incomplete set
    with pytest.raises(ValueError, match="cover the locked split"):
        PromptBank(manifest, "p0", ["pair_0", "pair_1", "pair_2"])

    # Reject corrupt hash
    corrupt = make_p0_record("pair_0", "train")
    corrupt["compiled_prompt_sha256"] = "wrong_hash"
    bad_manifest = tmp_path / "bad.jsonl"
    bad_manifest.write_text(json.dumps(corrupt) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Prompt hash differs"):
        PromptBank(bad_manifest, "p0", ["pair_0"])


def test_parse_answer_with_surrounding_conversational_text():
    raw = (
        "Here is the result based on your instructions:\n"
        "```json\n"
        '{"rewritten_prompt": "Cross both arms over the chest."}\n'
        "```\n"
        "Let me know if you need any adjustments!"
    )
    assert parse_answer(raw)["rewritten_prompt"] == "Cross both arms over the chest."


def test_compile_prompt_handles_image_naming_variants():
    raw = "Keep identity from Image_1. Change pose to match <image_2>."
    compiled = compile_prompt(raw, ("source_A", "target_B"))
    assert "Image_1" not in compiled
    assert "<image_2>" not in compiled
    assert "the source image" in compiled
    assert "the target pose described in this instruction" in compiled


def test_request_roles_geometry_fallback(tmp_path: Path):
    geom_dir = tmp_path / "geometry/pair_99"
    geom_dir.mkdir(parents=True, exist_ok=True)
    filt_p = geom_dir / "dwpose_filtered.png"
    filt_p.write_bytes(b"PNG")

    row = {
        "pair_id": "pair_99",
        "source_rgb": "a.png",
        "target_depth_vis": "d.png",
        "target_normal": "n.png",
    }
    # Should find filtered dwpose under geometry directory
    roles = request_roles(row, "pg", assets_dir=tmp_path / "geometry")
    assert any(role == "target_dwpose" and path == str(filt_p) for role, path in roles)

