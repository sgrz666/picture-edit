from __future__ import annotations

import json
from pathlib import Path
from PIL import Image
import pytest

from scripts.reports.build_qwen21_pei2i_study_report import (
    text_block,
    build_study_report,
)
from src.pose_control.qwen21_full.prompt_study.protocol import ARMS, PROTOCOLS
from src.pose_control.qwen21_full.prompt_study.results import verify_report_relative_links


def test_raw_prompt_is_html_escaped():
    block = text_block("<script>alert(1)</script>")
    assert "<script>" not in block
    assert "&lt;script&gt;" in block
    quote_block = text_block('pair "test" & <other>')
    assert '"' not in quote_block
    assert "&quot;test&quot;" in quote_block
    assert "&amp;" in quote_block


def test_build_study_report_with_fixtures(tmp_path: Path):
    run_dir = tmp_path / "study_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir = run_dir / "data"
    prompts_dir = run_dir / "prompts"
    eval_dir = run_dir / "evaluation"
    training_dir = run_dir / "training"
    data_dir.mkdir(exist_ok=True)
    prompts_dir.mkdir(exist_ok=True)
    eval_dir.mkdir(exist_ok=True)
    training_dir.mkdir(exist_ok=True)

    # 1. Lock file
    (data_dir / "lock.json").write_text(json.dumps({"schema": "pei2i_prompt_study_v1", "arms": ARMS}), encoding="utf-8")

    # 2. Prompts
    for proto in PROTOCOLS:
        lines = []
        for i in range(2):
            lines.append(json.dumps({
                "pair_id": f"pair_{i}",
                "protocol": proto,
                "instruction": f"Instruction <test> {proto} for pair_{i}",
                "raw_answer": f"PE Answer with <script>alert({i})</script>",
                "compiled_prompt": f"A dancer performing {proto} <prompt>",
                "compiled_prompt_sha256": f"sha_{proto}_{i}",
            }))
        (prompts_dir / f"{proto}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 3. Create mock images for reference and outputs
    dummy_img = Image.new("RGB", (64, 64), color="blue")

    # 4. Evaluation runs: 8 arms, 2 steps (0, 1024), 2 seeds (42, 43)
    for arm in ARMS:
        for step in (0, 1024):
            for seed in (42, 43):
                edir = eval_dir / f"{arm}_step_{step}_seed_{seed}"
                edir.mkdir(parents=True, exist_ok=True)
                img_dir = edir / "images"
                img_dir.mkdir(exist_ok=True)

                # Save sample images
                src_p = img_dir / "src.png"
                tgt_p = img_dir / "tgt.png"
                dummy_img.save(src_p)
                dummy_img.save(tgt_p)

                rows = []
                scores = []
                for idx in range(2):
                    pair_id = f"pair_{idx}"
                    out_p = img_dir / f"out_{idx}.png"
                    dummy_img.save(out_p)

                    # Mark one case as failed on arm head_pbg step 1024 idx 1
                    is_fail = (arm == "head_pbg" and step == 1024 and idx == 1)
                    status = "failed" if is_fail else "complete"

                    row = {
                        "pair_index": idx,
                        "pair_id": pair_id,
                        "metadata": {"pair_id": pair_id, "source_appearance": f"group_{idx % 2}"},
                        "status": status,
                        "source": "images/src.png",
                        "target": "images/tgt.png",
                        "image": f"images/out_{idx}.png" if status == "complete" else None,
                        "failure": "CUDA Out of memory error" if is_fail else None,
                        "target_face_box": [0.2, 0.2, 0.5, 0.5],
                    }
                    rows.append(row)

                    score = {
                        "pair_index": idx,
                        "status": status,
                        "identity_A": 0.85 if status == "complete" else None,
                        "pose_pck_002": 0.92 if status == "complete" else None,
                        "bg_l1_A": 0.04 if status == "complete" else None,
                    }
                    scores.append(score)

                # Write files
                (edir / "config.json").write_text(json.dumps({
                    "arm": arm,
                    "study_step": step,
                    "study_split": "val",
                    "seed": seed,
                }), encoding="utf-8")
                (edir / "per_sample.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
                (edir / "scores.json").write_text(json.dumps({
                    "metric_resolution": "per_sample_native",
                    "per_sample": scores,
                }), encoding="utf-8")

    # 5. Build report
    report_html = build_study_report(run_dir)
    assert report_html.is_file()
    content = report_html.read_text(encoding="utf-8")

    # Verify Chinese titles and sections
    assert "TikTok PE-I2I" in content
    assert "实验配置与架构隔离" in content
    assert "配对对比分析" in content
    assert "八组样本并排对比" in content
    assert "异常与失败归档" in content

    # Verify script escaping
    assert "<script>" not in content
    assert "&lt;script&gt;" in content

    # Verify partial progress labeled as "进行中"
    assert "进行中 (In Progress)" in content

    # Verify relative links in HTML all point to existing files
    missing_links = verify_report_relative_links(report_html, run_dir)
    assert missing_links == [], f"Broken relative links found: {missing_links}"

    # Verify summary.json generated
    summary_json = report_html.parent / "summary.json"
    assert summary_json.is_file()
    summ = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summ["failed_cases_count"] > 0  # the mock failure was captured
    assert summ["is_completed"] is False


def test_build_study_report_with_subdirectory_prompts_and_intermediate_steps(tmp_path: Path):
    run_dir = tmp_path / "study_run_sub"
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir = run_dir / "data"
    prompts_dir = run_dir / "prompts"
    eval_dir = run_dir / "evaluation"
    data_dir.mkdir(exist_ok=True)
    prompts_dir.mkdir(exist_ok=True)
    eval_dir.mkdir(exist_ok=True)

    # 1. Lock file
    (data_dir / "lock.json").write_text(json.dumps({"schema": "pei2i_prompt_study_v1", "arms": ARMS}), encoding="utf-8")

    # 2. Prompts in subdirectory structure: prompts/<proto>/<split>.jsonl
    for proto in PROTOCOLS:
        p_sub = prompts_dir / proto
        p_sub.mkdir(exist_ok=True)
        lines = [
            json.dumps({
                "pair_id": "pair_0",
                "protocol": proto,
                "instruction": f"Instruction {proto}",
                "rewritten_prompt": f"PE Raw Answer {proto}",
                "compiled_prompt": f"A dancer performing {proto}",
                "compiled_prompt_sha256": f"sha_{proto}",
            })
        ]
        (p_sub / "train.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 3. Intermediate step evaluation: step 128
    dummy_img = Image.new("RGB", (64, 64), color="red")
    edir = eval_dir / "qwen_p0_step_128_seed_42"
    edir.mkdir(parents=True, exist_ok=True)
    img_dir = edir / "images"
    img_dir.mkdir(exist_ok=True)
    dummy_img.save(img_dir / "src.png")
    dummy_img.save(img_dir / "tgt.png")
    dummy_img.save(img_dir / "out.png")

    (edir / "config.json").write_text(json.dumps({
        "arm": "qwen_p0",
        "study_step": 128,
        "study_split": "train",
        "seed": 42,
    }), encoding="utf-8")
    (edir / "per_sample.jsonl").write_text(json.dumps({
        "pair_index": 0,
        "pair_id": "pair_0",
        "status": "complete",
        "source": "images/src.png",
        "target": "images/tgt.png",
        "image": "images/out.png",
    }) + "\n", encoding="utf-8")
    (edir / "scores.json").write_text(json.dumps({"per_sample": []}), encoding="utf-8")

    report_html = build_study_report(run_dir)
    assert report_html.is_file()
    content = report_html.read_text(encoding="utf-8")
    # Verify Step 128 header was dynamically rendered
    assert "步数 Step 128" in content
    # Verify PE prompt was discovered from subdirectory
    summary_json = report_html.parent / "summary.json"
    summ = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summ["pe_success"] == 4  # all 4 protocols found

