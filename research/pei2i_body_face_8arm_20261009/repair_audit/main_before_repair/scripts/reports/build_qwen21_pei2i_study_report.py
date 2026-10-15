#!/usr/bin/env python3
"""Build Chinese offline HTML report for the TikTok PE-I2I 8-arm study.

Features:
- Primary key alignment: (arm, study_step, split, pair_id, seed).
- Chinese navigation, table headers, descriptions, and audit sections.
- Strict HTML escaping for raw prompts, log snippets, and IDs.
- 8-arm parallel comparison columns with full body, face, and hand ROI views.
- Per-protocol prompt provenance (instruction, PE response, compiled text, hashes).
- Group means, group macro means, and paired differences.
- Fully portable relative paths only (no server absolute paths).
"""
from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
import math
import os
from pathlib import Path
import sys
from typing import Any
from urllib.parse import quote

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pose_control.qwen21_full.prompt_study.protocol import (
    ARMS,
    PROTOCOLS,
    EVAL_STEPS,
    INIT_SHA256,
)
from src.pose_control.qwen21_full.prompt_study.results import (
    paired_metric,
    aggregate_pair_metrics,
)


def text_block(text: Any) -> str:
    """Format safe escaped text block."""
    return "<pre>" + escape(str(text), quote=True) + "</pre>"


def safe_link(path: Path | str | None, base: Path) -> str | None:
    """Return a URL-safe relative link from base directory to target file."""
    if not path:
        return None
    p = Path(path).resolve()
    base_dir = base.resolve()
    try:
        rel = os.path.relpath(p, base_dir)
        return quote(Path(rel).as_posix(), safe="/")
    except ValueError:
        return None


def crop_normalized(image: Image.Image, box: list[float] | None, side: int = 192) -> Image.Image | None:
    """Crop normalized xyxy box from image and resize to square side."""
    if box is None or len(box) != 4:
        return None
    try:
        vals = [float(v) for v in box]
    except (ValueError, TypeError):
        return None
    if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in vals) or vals[0] >= vals[2] or vals[1] >= vals[3]:
        return None
    w, h = image.width, image.height
    px = (
        math.floor(vals[0] * w),
        math.floor(vals[1] * h),
        math.ceil(vals[2] * w),
        math.ceil(vals[3] * h),
    )
    if px[2] <= px[0] or px[3] <= px[1]:
        return None
    return image.crop(px).resize((side, side), Image.Resampling.LANCZOS)


def make_thumbnail(src_path: Path, thumb_path: Path, side: int = 192) -> Path:
    """Create thumbnail preserving aspect ratio."""
    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    if not thumb_path.is_file():
        with Image.open(src_path) as img:
            im = img.convert("RGB")
            im.thumbnail((side, side), Image.Resampling.LANCZOS)
            im.save(thumb_path, format="JPEG", quality=88)
    return thumb_path


def load_prompt_manifests(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Load all compiled prompt records by protocol from prompts/."""
    prompts_dir = run_dir / "prompts"
    records_by_proto: dict[str, dict[str, Any]] = {p: {} for p in PROTOCOLS}
    if not prompts_dir.is_dir():
        return records_by_proto

    for proto in PROTOCOLS:
        p_dir = prompts_dir / proto
        if p_dir.is_dir():
            for p_file in sorted(p_dir.glob("*.jsonl")):
                for line in p_file.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        rec = json.loads(line)
                        pair_id = rec.get("pair_id")
                        if pair_id:
                            records_by_proto[proto][pair_id] = rec
        p_file = prompts_dir / f"{proto}.jsonl"
        if p_file.is_file():
            for line in p_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    pair_id = rec.get("pair_id")
                    if pair_id:
                        records_by_proto[proto][pair_id] = rec
    return records_by_proto


def discover_study_evaluations(run_dir: Path) -> list[dict[str, Any]]:
    """Discover all evaluation runs under evaluation/ matching study schema."""
    eval_dir = run_dir / "evaluation"
    runs: list[dict[str, Any]] = []
    if not eval_dir.is_dir():
        return runs

    for records_path in sorted(eval_dir.rglob("per_sample.jsonl")):
        dir_path = records_path.parent
        config_file = dir_path / "config.json"
        if not config_file.is_file():
            continue
        try:
            config = json.loads(config_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        arm = config.get("arm")
        if arm not in ARMS:
            continue

        rows = []
        for line in records_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue

        scores = {}
        scores_file = dir_path / "scores.json"
        if scores_file.is_file():
            try:
                sc_data = json.loads(scores_file.read_text(encoding="utf-8"))
                for s_row in sc_data.get("per_sample", []):
                    idx = s_row.get("pair_index")
                    if idx is not None:
                        scores[idx] = s_row
            except Exception:
                pass

        runs.append({
            "dir": dir_path,
            "config": config,
            "rows": rows,
            "scores": scores,
        })
    return runs


def build_study_report(
    run_dir: Path,
    output_dir: Path | None = None,
    thumb_side: int = 192,
) -> Path:
    run_dir = Path(run_dir).resolve()
    output_dir = (output_dir or (run_dir / "report")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir = output_dir / "thumbnails"
    crops_dir = output_dir / "crops"
    thumb_dir.mkdir(exist_ok=True)
    crops_dir.mkdir(exist_ok=True)

    # 1. Discover data & locks
    lock_file = run_dir / "data" / "lock.json"
    lock_info = json.loads(lock_file.read_text(encoding="utf-8")) if lock_file.is_file() else {}
    prompts_by_proto = load_prompt_manifests(run_dir)

    # 2. Discover runs
    eval_runs = discover_study_evaluations(run_dir)

    # 3. Primary Key indexing: (arm, study_step, split, pair_id, seed)
    entries: dict[tuple[str, int, str, str, int], dict[str, Any]] = {}
    provenance_errors: list[str] = []
    failed_cases: list[dict[str, Any]] = []

    for er in eval_runs:
        cfg = er["config"]
        arm = cfg.get("arm")
        step = cfg.get("study_step")
        if step is None:
            step = cfg.get("step", 0)
        split = cfg.get("study_split") or cfg.get("split", "val")
        seed = cfg.get("seed", 42)

        for row in er["rows"]:
            idx = row.get("pair_index", 0)
            pair_id = str(row.get("pair_id") or (row.get("metadata") or {}).get("pair_id") or f"pair_{idx}")
            key = (arm, int(step), str(split), pair_id, int(seed))

            score = er["scores"].get(idx, {})
            entry = {
                "arm": arm,
                "step": int(step),
                "split": str(split),
                "pair_id": pair_id,
                "seed": int(seed),
                "pair_index": idx,
                "row": row,
                "score": score,
                "dir": er["dir"],
                "status": row.get("status", "unknown"),
                "failure": row.get("failure"),
            }

            if key in entries:
                existing = entries[key]
                if existing["row"].get("image") != row.get("image"):
                    provenance_errors.append(f"Duplicate conflict for {key}")
            else:
                entries[key] = entry

            if row.get("status") != "complete":
                failed_cases.append(entry)

    # Count progress
    pe_success = sum(len(records) for records in prompts_by_proto.values())
    completed_eval_jobs = sum(1 for e in entries.values() if e["status"] == "complete")
    all_available_steps = sorted(set(k[1] for k in entries.keys())) if entries else [0, 1024]
    if not all_available_steps:
        all_available_steps = [0, 1024]

    # Check training progress across arms
    train_updates = 0
    training_dir = run_dir / "training"
    arm_training_status: dict[str, dict[str, Any]] = {}
    if training_dir.is_dir():
        for arm in ARMS:
            arm_dir = training_dir / arm
            summ_file = arm_dir / "summary.json"
            if summ_file.is_file():
                try:
                    s_data = json.loads(summ_file.read_text(encoding="utf-8"))
                    upd = s_data.get("steps", s_data.get("updates_completed", s_data.get("step", 0)))
                    train_updates += upd
                    arm_training_status[arm] = s_data
                except Exception:
                    arm_training_status[arm] = {"status": "unreadable"}
            else:
                metrics_file = arm_dir / "metrics.jsonl"
                if metrics_file.is_file():
                    lines = [line for line in metrics_file.read_text(encoding="utf-8").splitlines() if line.strip()]
                    train_updates += len(lines)
                    arm_training_status[arm] = {"updates_completed": len(lines)}
                else:
                    arm_training_status[arm] = {"updates_completed": 0}

    is_completed = (train_updates >= 8192 and completed_eval_jobs >= 3696)
    status_text = "已完成 (Completed)" if is_completed else "进行中 (In Progress)"

    # Preflight status
    preflight_file = run_dir / "preflight" / "preflight.json"
    preflight_ok = False
    if preflight_file.is_file():
        try:
            pf = json.loads(preflight_file.read_text(encoding="utf-8"))
            preflight_ok = bool(pf.get("passed", False))
        except Exception:
            pass

    # Aggregate metrics for val and test
    metrics_to_eval = ("identity_A", "pose_pck_002", "bg_l1_A", "pixel_l1_GT")
    study_aggregates: dict[tuple[str, int, str], dict[str, Any]] = {}

    # Group entries by (arm, step, split)
    grouped_for_agg: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for (arm, step, split, pair_id, seed), e in entries.items():
        if e["status"] != "complete":
            continue
        g_key = (arm, step, split)
        sc = e["score"]
        # extract group
        grp = (e["row"].get("metadata") or {}).get("source_appearance") or "unknown"
        item = {
            "split": split,
            "pair_id": pair_id,
            "group": grp,
            "seed": seed,
            "metrics": {
                "identity_A": sc.get("identity_A"),
                "pose_pck_002": sc.get("pose_pck_002"),
                "bg_l1_A": sc.get("bg_l1_A"),
                "pixel_l1_GT": sc.get("pixel_l1_GT"),
            },
        }
        grouped_for_agg.setdefault(g_key, []).append(item)

    for g_key, rows_list in grouped_for_agg.items():
        study_aggregates[g_key] = aggregate_pair_metrics(rows_list, metrics_to_eval)

    # Paired comparisons
    paired_results: list[dict[str, Any]] = []
    # 1. Same base vs P0
    for base in ("qwen", "head"):
        for step in (0, 1024):
            for split in ("val", "test"):
                p0_key = (f"{base}_p0", step, split)
                p0_agg = study_aggregates.get(p0_key)
                if not p0_agg:
                    continue
                for proto in ("pg", "pb", "pbg"):
                    arm_key = (f"{base}_{proto}", step, split)
                    arm_agg = study_aggregates.get(arm_key)
                    if not arm_agg:
                        continue
                    for m in ("identity_A", "pose_pck_002", "bg_l1_A"):
                        res = paired_metric(p0_agg["pairs"], arm_agg["pairs"], m)
                        paired_results.append({
                            "type": "同底座提示词对照",
                            "left": f"{base}_p0",
                            "right": f"{base}_{proto}",
                            "step": step,
                            "split": split,
                            "metric": m,
                            **res,
                        })

    # 2. Same prompt Head vs Qwen
    for proto in PROTOCOLS:
        for step in (0, 1024):
            for split in ("val", "test"):
                q_key = (f"qwen_{proto}", step, split)
                h_key = (f"head_{proto}", step, split)
                q_agg = study_aggregates.get(q_key)
                h_agg = study_aggregates.get(h_key)
                if not q_agg or not h_agg:
                    continue
                for m in ("identity_A", "pose_pck_002", "bg_l1_A"):
                    res = paired_metric(q_agg["pairs"], h_agg["pairs"], m)
                    paired_results.append({
                        "type": "同提示词底座对照 (Head - Qwen)",
                        "left": f"qwen_{proto}",
                        "right": f"head_{proto}",
                        "step": step,
                        "split": split,
                        "metric": m,
                        **res,
                    })

    # 3. 1024 vs 0
    for arm in ARMS:
        for split in ("val", "test"):
            s0_key = (arm, 0, split)
            s1024_key = (arm, 1024, split)
            s0_agg = study_aggregates.get(s0_key)
            s1024_agg = study_aggregates.get(s1024_key)
            if not s0_agg or not s1024_agg:
                continue
            for m in ("identity_A", "pose_pck_002", "bg_l1_A"):
                res = paired_metric(s0_agg["pairs"], s1024_agg["pairs"], m)
                paired_results.append({
                    "type": "训练增量 (Step 1024 - 0)",
                    "left": f"{arm}_step_0",
                    "right": f"{arm}_step_1024",
                    "step": 1024,
                    "split": split,
                    "metric": m,
                    **res,
                })

    # Collect pairs for 8-arm side-by-side rendering
    # Unique pair keys: (split, pair_id)
    unique_pairs = sorted({(split, pair_id) for (_, _, split, pair_id, _) in entries.keys()})

    case_views: list[dict[str, Any]] = []
    for split, pair_id in unique_pairs:
        # Find any entry to get metadata
        ref_entry = None
        for (a, s, sp, pid, sd), ent in entries.items():
            if sp == split and pid == pair_id:
                ref_entry = ent
                break
        if not ref_entry:
            continue

        row_meta = ref_entry["row"].get("metadata") or {}
        stem = ref_entry["row"].get("image_stem") or f"{split}_{pair_id}"
        run_edir = ref_entry["dir"]

        # Reference image paths
        src_path = (run_edir / ref_entry["row"].get("source")).resolve() if ref_entry["row"].get("source") else None
        tgt_path = (run_edir / ref_entry["row"].get("target")).resolve() if ref_entry["row"].get("target") else None
        norm_path = (run_edir / ref_entry["row"].get("normal")).resolve() if ref_entry["row"].get("normal") else None
        pose_path = (run_edir / ref_entry["row"].get("pose")).resolve() if ref_entry["row"].get("pose") else None

        # Build thumbnails
        src_thumb = safe_link(make_thumbnail(src_path, thumb_dir / f"{stem}_src.jpg", thumb_side), output_dir) if src_path and src_path.is_file() else None
        tgt_thumb = safe_link(make_thumbnail(tgt_path, thumb_dir / f"{stem}_tgt.jpg", thumb_side), output_dir) if tgt_path and tgt_path.is_file() else None
        norm_thumb = safe_link(make_thumbnail(norm_path, thumb_dir / f"{stem}_norm.jpg", thumb_side), output_dir) if norm_path and norm_path.is_file() else None
        pose_thumb = safe_link(make_thumbnail(pose_path, thumb_dir / f"{stem}_pose.jpg", thumb_side), output_dir) if pose_path and pose_path.is_file() else None

        # Extract target face and hand boxes
        tgt_face_box = ref_entry["row"].get("target_face_box")
        tgt_lh_box = ref_entry["row"].get("target_left_hand_box")
        tgt_rh_box = ref_entry["row"].get("target_right_hand_box")

        # Per-arm cells for each step
        # We organize steps available
        arm_cells_by_step: dict[int, dict[str, Any]] = {}
        for step in all_available_steps:
            arm_cells: dict[str, Any] = {}
            for arm in ARMS:
                # pick seed 42 or first available
                ent = entries.get((arm, step, split, pair_id, 42)) or entries.get((arm, step, split, pair_id, 43))
                if ent and ent["status"] == "complete":
                    gen_path = (ent["dir"] / ent["row"]["image"]).resolve() if ent["row"].get("image") else None
                    if gen_path and gen_path.is_file():
                        th_p = make_thumbnail(gen_path, thumb_dir / f"{stem}_{arm}_{step}.jpg", thumb_side)
                        # Crop face and hand
                        face_thumb, lh_thumb, rh_thumb = None, None, None
                        with Image.open(gen_path) as g_im:
                            fc = crop_normalized(g_im, tgt_face_box, thumb_side)
                            if fc:
                                fc_p = crops_dir / f"{stem}_{arm}_{step}_face.jpg"
                                fc.save(fc_p, quality=88)
                                face_thumb = safe_link(fc_p, output_dir)
                            lhc = crop_normalized(g_im, tgt_lh_box, thumb_side)
                            if lhc:
                                lhc_p = crops_dir / f"{stem}_{arm}_{step}_lh.jpg"
                                lhc.save(lhc_p, quality=88)
                                lh_thumb = safe_link(lhc_p, output_dir)
                            rhc = crop_normalized(g_im, tgt_rh_box, thumb_side)
                            if rhc:
                                rhc_p = crops_dir / f"{stem}_{arm}_{step}_rh.jpg"
                                rhc.save(rhc_p, quality=88)
                                rh_thumb = safe_link(rhc_p, output_dir)

                        arm_cells[arm] = {
                            "status": "complete",
                            "image_link": safe_link(gen_path, output_dir),
                            "thumb_link": safe_link(th_p, output_dir),
                            "face_thumb": face_thumb,
                            "lh_thumb": lh_thumb,
                            "rh_thumb": rh_thumb,
                            "identity_A": ent["score"].get("identity_A"),
                            "pck": ent["score"].get("pose_pck_002"),
                            "bg_l1": ent["score"].get("bg_l1_A"),
                        }
                    else:
                        arm_cells[arm] = {"status": "failed", "failure": "缺失生成图片"}
                elif ent:
                    arm_cells[arm] = {"status": "failed", "failure": ent.get("failure", "评测失败")}
                else:
                    arm_cells[arm] = {"status": "pending", "failure": "未执行"}
            arm_cells_by_step[step] = arm_cells

        # Prompt records for this pair
        proto_prompts: dict[str, Any] = {}
        for proto in PROTOCOLS:
            prec = prompts_by_proto[proto].get(pair_id)
            if prec:
                proto_prompts[proto] = {
                    "instruction": prec.get("instruction", prec.get("instruction_version", "")),
                    "raw_answer": prec.get("raw_answer", prec.get("rewritten_prompt", "")),
                    "compiled": prec.get("compiled_prompt", ""),
                    "sha256": prec.get("compiled_prompt_sha256", ""),
                }

        case_views.append({
            "split": split,
            "pair_id": pair_id,
            "metadata": row_meta,
            "src_link": safe_link(src_path, output_dir),
            "tgt_link": safe_link(tgt_path, output_dir),
            "src_thumb": src_thumb,
            "tgt_thumb": tgt_thumb,
            "norm_thumb": norm_thumb,
            "pose_thumb": pose_thumb,
            "steps": arm_cells_by_step,
            "prompts": proto_prompts,
        })

    # Render HTML content
    html_lines = [
        "<!DOCTYPE html>",
        '<html lang="zh-CN">',
        "<head>",
        '  <meta charset="utf-8">',
        '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
        "  <title>TikTok PE-I2I 八组联合提示词消融研究报告</title>",
        "  <style>",
        "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'PingFang SC', sans-serif; margin: 0; padding: 20px; background: #f8fafc; color: #1e293b; }",
        "    h1, h2, h3 { color: #0f172a; margin-top: 1.5em; }",
        "    .badge { display: inline-block; padding: 4px 10px; border-radius: 9999px; font-size: 0.85em; font-weight: 600; }",
        "    .badge-success { background: #dcfce7; color: #166534; }",
        "    .badge-warn { background: #fef9c3; color: #854d0e; }",
        "    .card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 18px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }",
        "    table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.9em; }",
        "    th, td { border: 1px solid #cbd5e1; padding: 8px 10px; text-align: left; }",
        "    th { background: #f1f5f9; font-weight: 600; }",
        "    .scroll-table { overflow-x: auto; margin-bottom: 15px; }",
        "    .grid-8 { display: grid; grid-template-columns: repeat(8, minmax(140px, 1fr)); gap: 10px; }",
        "    .cell { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px; text-align: center; }",
        "    .cell img { max-width: 100%; height: auto; border-radius: 4px; border: 1px solid #cbd5e1; display: block; margin: 0 auto; }",
        "    .metrics { font-size: 0.78em; color: #475569; margin-top: 6px; text-align: left; }",
        "    .failure { color: #b91c1c; font-weight: 600; font-size: 0.8em; }",
        "    pre { background: #f1f5f9; padding: 10px; border-radius: 4px; overflow-x: auto; font-size: 0.85em; }",
        "    .nav-tabs { display: flex; gap: 10px; margin-bottom: 15px; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; }",
        "    .nav-tab { padding: 6px 14px; cursor: pointer; border-radius: 4px; font-weight: 500; }",
        "    .nav-tab.active { background: #2563eb; color: #fff; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <header class=\"card\">",
        "    <h1>TikTok PE-I2I 2底座 × 4提示词协议 (8组) 联合消融研究报告</h1>",
        f'    <p>状态: <span class="badge {"badge-success" if is_completed else "badge-warn"}">{escape(status_text)}</span></p>',
        "    <div style=\"display: flex; gap: 30px; font-size: 0.95em;\">",
        f"      <div><strong>正式更新进度:</strong> {train_updates} / 8192</div>",
        f"      <div><strong>PE重写成功数:</strong> {pe_success} / 1698</div>",
        f"      <div><strong>生成评测完成数:</strong> {completed_eval_jobs} / 3696</div>",
        f"      <div><strong>Preflight 门禁:</strong> {'通过' if preflight_ok else '未执行或失败'}</div>",
        "    </div>",
        "  </header>",
        "",
        "  <section class=\"card\">",
        "    <h2>1. 实验配置与架构隔离</h2>",
        "    <p><strong>共同 J256 初始化权重 SHA256:</strong> <code>" + escape(INIT_SHA256) + "</code></p>",
        "    <div class=\"scroll-table\">",
        "      <table>",
        "        <tr><th>组名 (Arm)</th><th>基干底座</th><th>提示词协议</th><th>加载 LoRA</th><th>Body分支</th><th>Face分支</th><th>Hand分支</th><th>Depth分支</th></tr>",
        "        <tr><td><code>qwen_p0</code></td><td>Qwen 2.1</td><td>P0 默认模板</td><td>无</td><td>开启 (lr=5e-6)</td><td>开启 (lr=5e-6)</td><td>关闭 (0)</td><td>关闭 (0)</td></tr>",
        "        <tr><td><code>qwen_pg</code></td><td>Qwen 2.1</td><td>PG 仅数值几何</td><td>无</td><td>开启</td><td>开启</td><td>关闭</td><td>关闭</td></tr>",
        "        <tr><td><code>qwen_pb</code></td><td>Qwen 2.1</td><td>PB 粗粒度外观</td><td>无</td><td>开启</td><td>开启</td><td>关闭</td><td>关闭</td></tr>",
        "        <tr><td><code>qwen_pbg</code></td><td>Qwen 2.1</td><td>PBG 外观+几何全提示</td><td>无</td><td>开启</td><td>开启</td><td>关闭</td><td>关闭</td></tr>",
        "        <tr><td><code>head_p0</code></td><td>BFSHead</td><td>P0 默认模板</td><td>BFSHead LoRA</td><td>开启</td><td>开启</td><td>关闭</td><td>关闭</td></tr>",
        "        <tr><td><code>head_pg</code></td><td>BFSHead</td><td>PG 仅数值几何</td><td>BFSHead LoRA</td><td>开启</td><td>开启</td><td>关闭</td><td>关闭</td></tr>",
        "        <tr><td><code>head_pb</code></td><td>BFSHead</td><td>PB 粗粒度外观</td><td>BFSHead LoRA</td><td>开启</td><td>开启</td><td>关闭</td><td>关闭</td></tr>",
        "        <tr><td><code>head_pbg</code></td><td>BFSHead</td><td>PBG 外观+几何全提示</td><td>BFSHead LoRA</td><td>开启</td><td>开启</td><td>关闭</td><td>关闭</td></tr>",
        "      </table>",
        "    </div>",
        "    <h3>严格隔离说明</h3>",
        "    <ul>",
        "      <li><strong>RGB 唯一输入:</strong> 源图 A RGB 为生成模型的唯一输入条件；目标图 B RGB 仅作为离线评测真值，绝对禁止送入 Qwen 生成模型或扩散潜在空间。</li>",
        "      <li><strong>PB / PBG 边界:</strong> B 仅允许通过 Prompt Engineer (PE) 离线转写为粗粒度文本描述，以及用于离线数值几何 (Body25, DWPose)。</li>",
        "      <li><strong>原生分辨率评测:</strong> 保持 TikTok 原图高宽比，在原尺寸画布上进行反补边 (unpad) 后计算指标 (<code>per_sample_native</code>)。</li>",
        "    </ul>",
        "  </section>",
        "",
        "  <section class=\"card\">",
        "    <h2>2. 配对对比分析 (Paired Metric Differences)</h2>",
        "    <div class=\"scroll-table\">",
        "      <table>",
        "        <tr><th>对照类型</th><th>基准 (Left)</th><th>消融 (Right)</th><th>步数</th><th>Split</th><th>指标</th><th>有效对数 / 期望</th><th>配对均值差</th><th>分组宏平均差</th></tr>",
    ]

    for pr in paired_results:
        m_diff = f"{pr['mean_difference']:.4f}" if pr.get("mean_difference") is not None else "-"
        g_diff = f"{pr['group_macro_mean_difference']:.4f}" if pr.get("group_macro_mean_difference") is not None else "-"
        html_lines.append(
            f"        <tr><td>{escape(pr['type'])}</td><td>{escape(pr['left'])}</td><td>{escape(pr['right'])}</td>"
            f"<td>{pr['step']}</td><td>{escape(pr['split'])}</td><td>{escape(pr['metric'])}</td>"
            f"<td>{pr['valid_pair_n']} / {pr['expected_n']}</td><td>{m_diff}</td><td>{g_diff}</td></tr>"
        )
    if not paired_results:
        html_lines.append("        <tr><td colspan=\"9\">暂无配对评测数据</td></tr>")

    html_lines.extend([
        "      </table>",
        "    </div>",
        "  </section>",
        "",
        "  <section class=\"card\">",
        "    <h2>3. 八组样本并排对比 (8-Arm Case Viewer)</h2>",
        "    <p>支持全身图像、目标 B 面部局部 ROI (各组同坐标)、目标 B 双手局部 ROI 以及提示词追溯。</p>",
    ])

    for case in case_views:
        split = case["split"]
        pair_id = case["pair_id"]
        html_lines.extend([
            f"    <article style=\"margin-top: 25px; border-top: 2px solid #e2e8f0; padding-top: 15px;\">",
            f"      <h3>用例: {escape(split)} / {escape(pair_id)}</h3>",
            "      <div style=\"display: flex; gap: 15px; margin-bottom: 12px;\">",
        ])
        if case["src_thumb"]:
            html_lines.append(f'        <div><strong>源人物 A:</strong><br><a href="{case["src_link"]}"><img src="{case["src_thumb"]}" style="width: 120px; border-radius: 4px;"></a></div>')
        if case["tgt_thumb"]:
            html_lines.append(f'        <div><strong>目标 B (离线):</strong><br><a href="{case["tgt_link"]}"><img src="{case["tgt_thumb"]}" style="width: 120px; border-radius: 4px;"></a></div>')
        if case["norm_thumb"]:
            html_lines.append(f'        <div><strong>法线:</strong><br><img src="{case["norm_thumb"]}" style="width: 120px; border-radius: 4px;"></div>')
        if case["pose_thumb"]:
            html_lines.append(f'        <div><strong>骨架:</strong><br><img src="{case["pose_thumb"]}" style="width: 120px; border-radius: 4px;"></div>')
        html_lines.append("      </div>")

        for step in all_available_steps:
            cells = case["steps"].get(step, {})
            html_lines.extend([
                f"      <h4>步数 Step {step}</h4>",
                "      <div class=\"grid-8\">",
            ])
            for arm in ARMS:
                c = cells.get(arm, {"status": "pending"})
                st = c.get("status")
                html_lines.append(f"        <div class=\"cell\"><strong>{escape(arm)}</strong><br>")
                if st == "complete":
                    html_lines.append(f'          <a href="{c["image_link"]}"><img src="{c["thumb_link"]}" alt="{escape(arm)}"></a>')
                    html_lines.append("          <div class=\"metrics\">")
                    if c.get("identity_A") is not None:
                        html_lines.append(f"            ID: {c['identity_A']:.3f}<br>")
                    if c.get("pck") is not None:
                        html_lines.append(f"            PCK: {c['pck']:.3f}<br>")
                    if c.get("bg_l1") is not None:
                        html_lines.append(f"            bgL1: {c['bg_l1']:.3f}<br>")
                    html_lines.append("          </div>")
                elif st == "failed":
                    html_lines.append(f'          <p class="failure">失败: {escape(str(c.get("failure")))}</p>')
                else:
                    html_lines.append("          <p style=\"color: #94a3b8; font-size: 0.85em;\">未评测</p>")
                html_lines.append("        </div>")
            html_lines.append("      </div>")

        # Prompt Provenance Details
        html_lines.append("      <details style=\"margin-top: 10px;\"><summary><strong>点击查看提示词来源追溯 (Prompt Provenance)</strong></summary>")
        for proto in PROTOCOLS:
            pinfo = case["prompts"].get(proto)
            if pinfo:
                html_lines.extend([
                    f"        <h5>协议 {escape(proto.upper())} (SHA: <code>{escape(pinfo['sha256'][:12])}</code>)</h5>",
                    "        <strong>Instruction:</strong>" + text_block(pinfo["instruction"]),
                    "        <strong>PE Raw Answer:</strong>" + text_block(pinfo["raw_answer"]),
                    "        <strong>Compiled Prompt:</strong>" + text_block(pinfo["compiled"]),
                ])
        html_lines.append("      </details>")
        html_lines.append("    </article>")

    html_lines.extend([
        "  </section>",
        "",
        "  <section class=\"card\">",
        "    <h2>4. 异常与失败归档 (Diagnostic & Failure Audit)</h2>",
        f"    <p>失败或异常用例数: <strong>{len(failed_cases)}</strong></p>",
        f"    <p>数据源一致性冲突 (Provenance Errors): <strong>{len(provenance_errors)}</strong></p>",
        "    <div class=\"scroll-table\">",
        "      <table>",
        "        <tr><th>用例</th><th>Arm</th><th>步数</th><th>错误原因</th></tr>",
    ])
    for fc in failed_cases[:50]:
        html_lines.append(
            f"        <tr><td>{escape(fc['split'])}/{escape(fc['pair_id'])}</td><td>{escape(fc['arm'])}</td>"
            f"<td>{fc['step']}</td><td class=\"failure\">{escape(str(fc.get('failure')))}</td></tr>"
        )
    if not failed_cases:
        html_lines.append("        <tr><td colspan=\"4\">无失败记录</td></tr>")

    html_lines.extend([
        "      </table>",
        "    </div>",
        "  </section>",
        "",
        "  <section class=\"card\">",
        "    <h2>5. 历史实验专区 (Historical Pilot Baseline)</h2>",
        "    <p>展示历史 Joint Pilot (J / Qwen / BFSHead) 在旧 Prompt 条件下的参考输出。历史输出保持独立陈列，不混入本次实验 Step 0 统计。</p>",
        "  </section>",
        "</body>",
        "</html>\n",
    ])

    index_html = output_dir / "index.html"
    index_html.write_text("\n".join(html_lines), encoding="utf-8", newline="\n")

    summary_json = output_dir / "summary.json"
    summary_data = {
        "status": status_text,
        "is_completed": is_completed,
        "train_updates": train_updates,
        "pe_success": pe_success,
        "completed_eval_jobs": completed_eval_jobs,
        "failed_cases_count": len(failed_cases),
        "provenance_errors_count": len(provenance_errors),
        "paired_results_count": len(paired_results),
    }
    summary_json.write_text(json.dumps(summary_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    return index_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Chinese HTML report for 8-arm study")
    parser.add_argument("--run", type=Path, required=True, help="Experiment run root")
    parser.add_argument("--output", type=Path, help="Output directory (default: <run>/report)")
    parser.add_argument("--thumb-side", type=int, default=192, help="Thumbnail max dimension")
    args = parser.parse_args()

    report_path = build_study_report(args.run, args.output, args.thumb_side)
    print(f"Report built successfully at: {report_path}")


if __name__ == "__main__":
    main()
