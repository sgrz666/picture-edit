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
import shutil
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
    sha256_file,
)
from src.pose_control.qwen21_full.prompt_study.results import (
    paired_metric,
    aggregate_pair_metrics,
    eval_jobs,
    validate_evaluation_artifacts,
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


def render_prompt_evidence(rec: dict[str, Any], run: Path, output: Path, thumbs: Path, side: int) -> list[str]:
    """Link every retained PE request, including samples outside generation monitors."""
    lines = [f'<details><summary>{escape(rec["split"])} / {escape(rec["pair_id"])} / {escape(rec["protocol"].upper())}</summary>']
    if rec.get('raw_path'):
        raw_path = run / rec['raw_path']
        lines.append(f'<p><a href="{safe_link(raw_path, output)}">PE 原始请求与回答</a>（保留完整思考、输入说明、采样参数与哈希）</p>')
    lines.append('<strong>实际编译提示词</strong>' + text_block(rec['compiled_prompt']))
    for item in rec.get('inputs', []):
        asset = run / item['asset_relative_path']
        thumb = make_thumbnail(asset, thumbs / f"pe_{item['sha256'][:24]}.jpg", side)
        lines.append(f'<div>实际 PE 输入：{escape(item["role"])} <code>{escape(item["sha256"])}</code><a href="{safe_link(asset, output)}"><img width="120" src="{safe_link(thumb, output)}" alt="{escape(item["role"])}"></a></div>')
    lines.append('</details>')
    return lines


def build_metric_curves(aggregates: dict, output: Path) -> list[tuple[str, Path]]:
    """Export standard plots of video-group macro means with missing points retained."""
    plots = []
    colors = ('#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#9467BD', '#333333')
    for split in ('val', 'test'):
        steps = EVAL_STEPS if split == 'val' else (0, 1024)
        for metric, label in (('identity_A', '身份'), ('pose_pck_002', '动作 PCK'), ('bg_l1_A', '背景 L1')):
            series = {arm: [aggregates.get((arm, step, split), {}).get('group_macro_means', {}).get(metric) for step in steps] for arm in ARMS}
            if not any(value is not None for values in series.values() for value in values):
                continue
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib import pyplot as plt
            figure, axis = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
            for (arm, values), color in zip(series.items(), colors):
                axis.plot(steps, [float('nan') if v is None else v for v in values], marker='o', markersize=4, linewidth=1.5, color=color, label=arm)
            axis.set_xlabel('Committed update')
            axis.set_ylabel(metric + ' (video group macro mean)')
            axis.set_xticks(steps)
            axis.grid(alpha=.2)
            axis.legend(fontsize=8, ncol=4, loc='upper center', bbox_to_anchor=(.5, 1.16))
            path = output / 'charts' / f'{split}_{metric}.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=150)
            plt.close(figure)
            plots.append((f'{"验证" if split == "val" else "独立测试"} {label}', path))
    return plots


def committed_training_metrics(run: Path, arm: str) -> tuple[int, list[dict]]:
    """Read the validated latest checkpoint's metric prefix; retain trailing attempts."""
    from scripts.run.run_qwen21_pei2i_study import checkpoint_step
    directory = run / 'training' / arm
    numbered = sorted(directory.glob('checkpoint_step_*.pt'))
    candidates = numbered[-1:] + ([directory / 'checkpoint_last.pt'] if (directory / 'checkpoint_last.pt').is_file() else [])
    if not candidates:
        return 0, []
    step = max(checkpoint_step(path, run, arm) for path in candidates)
    rows = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()][:step]
    if len(rows) != step or any(row.get('step') != i + 1 for i, row in enumerate(rows)):
        raise RuntimeError('Committed loss trajectory differs from checkpoint')
    if any(not isinstance(row.get('loss'), (int, float)) or not math.isfinite(row['loss']) for row in rows):
        raise RuntimeError('Committed training loss is not finite')
    return step, rows


def build_training_loss_curves(trajectories: dict[str, list[dict]], output: Path) -> list[tuple[str, Path]]:
    """Export raw committed total and available component losses without smoothing."""
    plots = []
    for key, label in (('loss', '训练总损失'), ('body_flow', 'Body flow 损失'), ('face_flow', 'Face flow 损失'), ('identity', '身份辅助损失'), ('perceptual', '感知辅助损失')):
        series = {arm: [(row['step'], row[key]) for row in rows if isinstance(row.get(key), (int, float)) and math.isfinite(row[key])] for arm, rows in trajectories.items()}
        if not any(series.values()):
            continue
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        figure, axis = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
        for arm, values in series.items():
            if values:
                axis.plot([step for step, _ in values], [value for _, value in values], linewidth=.9, alpha=.8, label=arm)
        axis.set_xlabel('Committed update')
        axis.set_ylabel(key)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8, ncol=4)
        path = output / 'charts' / f'training_{key}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=150)
        plt.close(figure)
        plots.append((label, path))
    return plots


def read_preflight_evidence(run: Path) -> dict[str, Any]:
    """Summarize the actual lock-bound selfcheck, eight gradients and private probe."""
    evidence: dict[str, Any] = {'passed': False, 'selfcheck': False, 'gradients': {}, 'resume': False, 'errors': []}
    try:
        lock_sha = sha256_file(run / 'data/lock.json')
        selfcheck = json.loads((run / 'preflight/selfcheck.json').read_text(encoding='utf-8'))
        mandatory = {key: value for key, value in selfcheck.get('checks', {}).items() if key != 'eight_arms_completed'}
        evidence['selfcheck'] = (selfcheck.get('schema') == 'qwen21_study_selfcheck_v1' and selfcheck.get('passed') is True and bool(mandatory) and all(value is True for value in mandatory.values()) and selfcheck.get('study_lock_sha256') == lock_sha)
        pg_signature = None
        for arm in ARMS:
            record = json.loads((run / f'preflight/{arm}/gradient_only.json').read_text(encoding='utf-8'))
            signature = record.get('joint_signature', {})
            evidence['gradients'][arm] = all(record.get(key) is True for key in ('passed', 'frozen_transformer_unchanged', 'frozen_adapter_unchanged')) and signature.get('study_lock_sha256') == lock_sha and signature.get('arm') == arm
            if arm == 'qwen_pg':
                pg_signature = signature
        probe = json.loads((run / 'preflight/resume_check/result.json').read_text(encoding='utf-8'))
        evidence['resume'] = probe.get('mandatory_exact_gate', {}).get('passed') is True and probe.get('private_optimizer_updates') == 10 and probe.get('numerical_review_required') is False and probe.get('study_lock_sha256') == lock_sha and probe.get('joint_signature') == pg_signature
        evidence['passed'] = evidence['selfcheck'] and all(evidence['gradients'].values()) and evidence['resume']
    except (OSError, ValueError, KeyError) as exc:
        evidence['errors'].append(str(exc))
    return evidence


def historical_references(run: Path, output: Path, thumbs: Path, side: int) -> list[str]:
    """Copy explicit SHA-bound old-pilot images, preserving independent metadata."""
    manifest = run / 'data/historical_references.json'
    lines = ['<section class="card"><h2>历史实验专区</h2><p>历史记录保留旧提示词、预算与实际加载条件，独立展示，不补入本次 step 0 或任何正式统计。</p>']
    if not manifest.is_file():
        return lines + ['<p>尚未提供 data/historical_references.json；历史图像待导入。</p></section>']
    snapshot: dict[str, Any] = {'schema': 'qwen21_study_historical_references_v1', 'source_manifest_sha256': sha256_file(manifest), 'records': [], 'errors': []}
    destination = output / 'history'
    destination.mkdir(parents=True, exist_ok=True)
    try:
        payload = json.loads(manifest.read_text(encoding='utf-8'))
        if payload.get('schema') != snapshot['schema'] or not isinstance(payload.get('records'), list):
            raise ValueError('历史清单 schema/records 无效')
        for record in payload['records']:
            try:
                if any(key not in record for key in ('id', 'label', 'prompt', 'budget', 'loading', 'metadata', 'images')) or not all(isinstance(record[key], dict) for key in ('budget', 'loading', 'metadata')):
                    raise ValueError('历史记录缺少实际提示词、预算或加载 metadata')
                copied = dict(record, images=[])
                for item in record['images']:
                    source = Path(item['path'])
                    source = source if source.is_absolute() else run / source
                    if source.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.webp') or sha256_file(source) != item['sha256']:
                        raise ValueError(f'历史图片校验失败：{item["path"]}')
                    with Image.open(source) as image:
                        image.verify()
                    target = destination / (item['sha256'] + source.suffix.lower())
                    if not target.is_file():
                        shutil.copyfile(source, target)
                    if sha256_file(target) != item['sha256']:
                        raise ValueError(f'历史图片校验失败：复制文件 {target.name}')
                    copied['images'].append(dict(item, original_path=item['path'], path='history/' + target.name))
                snapshot['records'].append(copied)
                lines.append('<article><h3>' + escape(str(record['label'])) + '</h3><strong>实际旧提示词</strong>' + text_block(record['prompt']))
                lines.append('<strong>原预算与加载条件</strong>' + text_block({key: record[key] for key in ('budget', 'loading', 'metadata')}))
                for item in copied['images']:
                    target = output / item['path']
                    thumb = make_thumbnail(target, thumbs / f'history_{item["sha256"][:24]}.jpg', side)
                    lines.append(f'<figure><figcaption>{escape(str(item.get("pair_id", "")))}</figcaption><a href="{safe_link(target, output)}"><img width="192" src="{safe_link(thumb, output)}" alt="历史原始输出"></a><code>{escape(item["sha256"])}</code></figure>')
                lines.append('</article>')
            except (OSError, ValueError, KeyError, TypeError) as exc:
                snapshot['errors'].append(str(exc))
                lines.append('<p class="failure">历史图片校验失败：' + escape(str(exc)) + '</p>')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        snapshot['errors'].append(str(exc))
        lines.append('<p class="failure">历史清单校验失败：' + escape(str(exc)) + '</p>')
    (destination / 'references.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines.append(f'<p><a href="{safe_link(destination / "references.json", output)}">历史快照与校验记录</a></p></section>')
    return lines


def summarize_detection_counts(entries: dict) -> list[dict[str, Any]]:
    """Keep detector failures separate by arm, checkpoint, split and inference seed."""
    groups = {}
    for entry in entries.values():
        if entry['status'] != 'complete':
            continue
        key = tuple(entry[name] for name in ('arm', 'step', 'split', 'seed'))
        if key not in groups:
            groups[key] = dict(zip(('arm', 'step', 'split', 'seed'), key), generated_images=0, output_no_face=0, output_multiple_faces=0, output_detection_unknown=0, source_no_face=0, source_multiple_faces=0, source_detection_unknown=0, identity_invalid=0, pose_failures={})
        group, score = groups[key], entry['score']
        group['generated_images'] += 1
        for actor, field in (('output', 'face_count'), ('source', 'source_face_count')):
            count = score.get(field)
            if count == 0:
                group[actor + '_no_face'] += 1
            elif isinstance(count, int) and count > 1:
                group[actor + '_multiple_faces'] += 1
            elif count != 1:
                group[actor + '_detection_unknown'] += 1
        identity = score.get('identity_A')
        if not isinstance(identity, (int, float)) or isinstance(identity, bool) or not math.isfinite(identity):
            group['identity_invalid'] += 1
        if score.get('pose_failure'):
            reason = str(score['pose_failure'])
            group['pose_failures'][reason] = group['pose_failures'].get(reason, 0) + 1
    return [groups[key] for key in sorted(groups)]


def load_prompt_manifests(run_dir: Path, errors: list[str] | None = None) -> dict[str, dict[Any, Any]]:
    """Load all compiled prompt records by protocol from prompts/."""
    prompts_dir = run_dir / "prompts"
    records_by_proto: dict[str, dict[str, Any]] = {p: {} for p in PROTOCOLS}
    if not prompts_dir.is_dir():
        return records_by_proto

    from src.pose_control.qwen21_full.prompt_study.prompts import validate_prompt_record
    for proto in PROTOCOLS:
        p_dir = prompts_dir / proto
        if p_dir.is_dir():
            for p_file in sorted(p_dir.glob("*.jsonl")):
                for line in p_file.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            if rec.get('split') != p_file.stem or rec.get('protocol') != proto or p_file.stem not in ('train', 'val', 'test', 'diagnostic'):
                                raise ValueError('prompt split/protocol differs')
                            validate_prompt_record(rec, run_dir=run_dir)
                            key = (rec['split'], rec['pair_id'])
                            if key in records_by_proto[proto]:
                                raise ValueError('duplicate prompt')
                            records_by_proto[proto][key] = rec
                        except (ValueError, KeyError, OSError) as exc:
                            if errors is not None:
                                errors.append(f'{p_file.relative_to(run_dir)}: {exc}')
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

        validation_errors = []
        try:
            validated = validate_evaluation_artifacts(dir_path, run_dir, require_scores=True, allow_score_failures=True)
            scores = validated['scores']
        except (ValueError, KeyError, OSError, TypeError) as exc:
            validation_errors.append(str(exc))
        runs.append({
            "dir": dir_path,
            "config": config,
            "rows": rows,
            "scores": scores,
            "validation_errors": validation_errors,
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
    prompt_errors: list[str] = []
    prompts_by_proto = load_prompt_manifests(run_dir, prompt_errors)
    locked_pairs = {}
    for split, filename in {'train': 'train512_pairs.jsonl', 'val': 'val24_pairs.jsonl', 'test': 'test24_pairs.jsonl', 'diagnostic': 'diagnostic6_pairs.jsonl'}.items():
        path = run_dir / 'data' / filename
        if path.is_file():
            locked_pairs[split] = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

    # 2. Discover runs
    eval_runs = discover_study_evaluations(run_dir)

    # 3. Primary Key indexing: (arm, study_step, split, pair_id, seed)
    entries: dict[tuple[str, int, str, str, int], dict[str, Any]] = {}
    provenance_errors: list[str] = prompt_errors
    failed_cases: list[dict[str, Any]] = []

    for er in eval_runs:
        provenance_errors.extend(f"{er['dir'].relative_to(run_dir)}: {e}" for e in er['validation_errors'])
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
                "status": 'provenance_error' if er['validation_errors'] else ('scoring_error' if score.get('score_failure') else row.get("status", "unknown")),
                "failure": '; '.join(er['validation_errors']) if er['validation_errors'] else (score.get('score_failure') or row.get("failure")),
            }

            if key in entries:
                existing = entries[key]
                provenance_errors.append(f"Duplicate conflict for {key}")
                existing['status'] = 'provenance_error'
                existing['failure'] = '重复任务记录，请检查保留的全部尝试'
                failed_cases.append(existing)
            else:
                entries[key] = entry

            if entry['status'] != "complete":
                failed_cases.append(entry)

    detection_audit = summarize_detection_counts(entries)
    # Count progress
    pe_success = sum(len(records) for proto, records in prompts_by_proto.items() if proto != 'p0')
    completed_eval_jobs = sum(1 for e in entries.values() if e["status"] == "complete")
    scoring_errors_count = sum(1 for e in entries.values() if e['status'] == 'scoring_error')
    all_available_steps = list(EVAL_STEPS)
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
                    if any(s_data.get(k) is not True for k in ('frozen_transformer_unchanged', 'frozen_adapter_unchanged')):
                        raise ValueError('缺少冻结检查证据')
                    from scripts.run.run_qwen21_pei2i_study import checkpoint_step
                    if checkpoint_step(arm_dir / f'checkpoint_step_{upd:06d}.pt', run_dir, arm) != upd:
                        raise ValueError('checkpoint 更新数不匹配')
                    train_updates += upd
                    arm_training_status[arm] = s_data
                except Exception as exc:
                    arm_training_status[arm] = {"status": "unreadable"}
                    provenance_errors.append(f'training/{arm}: {exc}')
            else:
                metrics_file = arm_dir / "metrics.jsonl"
                if metrics_file.is_file():
                    lines = [line for line in metrics_file.read_text(encoding="utf-8").splitlines() if line.strip()]
                    arm_training_status[arm] = {"updates_completed": 0, 'uncommitted_metrics_rows': len(lines)}
                else:
                    arm_training_status[arm] = {"updates_completed": 0}

    training_complete = False
    try:
        from src.pose_control.qwen21_full.prompt_study.protocol import require_eight_arms_completed
        require_eight_arms_completed(run_dir)
        training_complete = True
    except (ValueError, RuntimeError, OSError):
        pass
    preflight_evidence = read_preflight_evidence(run_dir)
    preflight_ok = preflight_evidence['passed']

    # Aggregate metrics for val and test
    metrics_to_eval = ("identity_A", "pose_pck_002", "bg_l1_A", "pixel_l1_GT")
    study_aggregates: dict[tuple[str, int, str], dict[str, Any]] = {}

    # Group entries by (arm, step, split)
    grouped_for_agg: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for (arm, step, split, pair_id, seed), e in entries.items():
        if e["status"] not in ("complete", "failed"):
            continue
        g_key = (arm, step, split)
        sc = e["score"] if e['status'] == 'complete' else {}
        # extract group
        grp = (e["row"].get("metadata") or {}).get("source_video_group") or "unknown"
        item = {
            "split": split,
            "pair_id": pair_id,
            "group": grp,
            'source_video_group': grp,
            "seed": seed,
            'status': e['status'],
            "metrics": {
                "identity_A": sc.get("identity_A"),
                "pose_pck_002": sc.get("pose_pck_002"),
                "bg_l1_A": sc.get("bg_l1_A"),
                "pixel_l1_GT": sc.get("pixel_l1_GT"),
            },
        }
        grouped_for_agg.setdefault(g_key, []).append(item)

    schedule = eval_jobs(include_final_test=True)
    for arm in ARMS:
        for step in EVAL_STEPS:
            for split, pairs in locked_pairs.items():
                indices = sorted({j['pair_index'] for j in schedule if (j['arm'], j['step'], j['split']) == (arm, step, split)})
                expected_pairs = [dict(split=split, pair_id=pairs[i]['pair_id'], source_video_group=pairs[i].get('source_video_group', 'unknown')) for i in indices if i < len(pairs)]
                if expected_pairs:
                    grouped_for_agg.setdefault((arm, step, split), [])
                    seeds = tuple(sorted({j['seed'] for j in schedule if (j['arm'], j['step'], j['split']) == (arm, step, split)}))
                    study_aggregates[(arm, step, split)] = aggregate_pair_metrics(grouped_for_agg[(arm, step, split)], metrics_to_eval, expected_pairs=expected_pairs, expected_seeds=seeds)
    for g_key, rows_list in grouped_for_agg.items():
        if g_key not in study_aggregates:
            study_aggregates[g_key] = aggregate_pair_metrics(rows_list, metrics_to_eval)

    # Paired comparisons
    curves = build_metric_curves(study_aggregates, output_dir)
    loss_trajectories = {}
    for arm in ARMS:
        try:
            _, rows = committed_training_metrics(run_dir, arm)
            if rows:
                loss_trajectories[arm] = rows
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            provenance_errors.append(f'training loss/{arm}: {exc}')
    loss_curves = build_training_loss_curves(loss_trajectories, output_dir)
    curve_html = ['<section class="card"><h2>身份、动作与背景曲线</h2><p>预定种子的尝试齐全后，在每个样本内求有效指标均值，再按原始视频组宏平均。缺少预定任务的样本保持缺失，曲线留空档；无脸或无效检测保留 None 并单列计数。推理种子与同视频样本不视作独立训练重复。</p>']
    for label, path in curves:
        curve_html.append(f'<figure><figcaption>{label}</figcaption><a href="{safe_link(path, output_dir)}"><img style="max-width:100%" src="{safe_link(path, output_dir)}" alt="{label}"></a></figure>')
    if not curves:
        curve_html.append('<p>尚无满足完整种子与评分证据的有效曲线点。</p>')
    curve_html.append('</section>')
    curve_html.append('<details class="card"><summary>各指标有效样本、种子与原视频分组均值</summary><table><tr><th>组名</th><th>步数</th><th>数据分组</th><th>指标</th><th>有效样本 / 期望</th><th>缺失种子任务</th><th>无效种子指标</th><th>原视频组均值</th></tr>')
    for (arm, step, split), aggregate in sorted(study_aggregates.items()):
        if split not in ('val', 'test'):
            continue
        for metric in ('identity_A', 'pose_pck_002', 'bg_l1_A'):
            missing = sum(row['missing_seed_count'] for row in aggregate['pairs'])
            invalid = sum(row['invalid_seed_counts'][metric] for row in aggregate['pairs'])
            groups = {group: values.get(metric) for group, values in aggregate['group_means'].items()}
            curve_html.append(f'<tr><td>{escape(arm)}</td><td>{step}</td><td>{escape(split)}</td><td>{escape(metric)}</td><td>{aggregate["valid_counts"][metric]} / {aggregate["expected_n"]}</td><td>{missing}</td><td>{invalid}</td><td>{escape(json.dumps(groups, ensure_ascii=False))}</td></tr>')
    curve_html.append('</table></details>')
    is_completed = (training_complete and preflight_ok and train_updates == 8192 and completed_eval_jobs == 3696 and pe_success == 1698 and not provenance_errors)
    status_text = "已完成 (Completed)" if is_completed else "进行中 (In Progress)"
    conditions = {
        'schema': 'qwen21_study_report_conditions_v1', 'arms': list(ARMS),
        'status': status_text, 'common_adapter_initialization_sha256': INIT_SHA256,
        'additional_updates_per_arm': 1024, 'qwen_reference_rgb': ['A'],
        'test_backpropagation': False, 'rgb_resolution': 'original_pixels_pad16_unpad',
        'pe_inputs': {'p0': [], 'pg': ['A', 'B depth', 'B normal', 'B filtered DWPose'],
                      'pb': ['A', 'B'], 'pbg': ['A', 'B', 'B depth', 'B normal', 'B filtered DWPose']},
        'training_branches': ['Body', 'Face'], 'disabled_branches': ['Hand', 'Depth', 'Interaction', 'shared Detail'],
        'body_group_lrs': [2.5e-5, 1.25e-5, 2.5e-6], 'face_group_lrs': [5e-5, 2.5e-5, 5e-6],
        'optimizer': {'name': 'AdamW', 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': .01,
                      'warmup_updates': 20, 'gradient_clip': 1., 'batch_size': 1},
        'loss_weights': {'flow': 1., 'face_roi': .5, 'identity': .1, 'perceptual': .05},
        'inference': {'steps': 30, 'cfg': 1., 'body': 1., 'face': 1., 'hand': 0., 'seeds': [42, 43]},
        'data_lock': lock_info,
        'actual_training_configuration_files': {arm: {
            'path': safe_link(run_dir / f'training/{arm}/config.json', output_dir),
            'sha256': sha256_file(run_dir / f'training/{arm}/config.json')}
            for arm in ARMS if (run_dir / f'training/{arm}/config.json').is_file()},
        'limitation': 'Shared J256 was previously trained on plain Qwen; PB/PBG expose target RGB to PE; BFS Head uses A-only RGB in this study.',
    }
    (output_dir / 'conditions.json').write_text(json.dumps(conditions, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    curve_html.append('<section class="card"><h2>训练损失曲线</h2><p>仅展示通过 checkpoint、样本顺序、噪声种子与提示词校验的已提交更新。原始损失不平滑，断点后的未提交日志不计入。</p>')
    for label, path in loss_curves:
        curve_html.append(f'<figure><figcaption>{label}</figcaption><a href="{safe_link(path, output_dir)}"><img style="max-width:100%" src="{safe_link(path, output_dir)}" alt="{label}"></a></figure>')
    if not loss_curves:
        curve_html.append('<p>尚无通过 checkpoint 校验的训练损失记录。</p>')
    curve_html.append('</section>')
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
    unique_pairs = {(split, pair_id, seed) for (_, _, split, pair_id, seed) in entries.keys()}
    for job in schedule:
        pairs = locked_pairs.get(job['split'], [])
        if job['pair_index'] < len(pairs):
            unique_pairs.add((job['split'], pairs[job['pair_index']]['pair_id'], job['seed']))
    unique_pairs = sorted(unique_pairs)

    case_views: list[dict[str, Any]] = []
    for split, pair_id, case_seed in unique_pairs:
        # Find any entry to get metadata
        ref_entry = None
        for (a, s, sp, pid, sd), ent in entries.items():
            if sp == split and pid == pair_id:
                ref_entry = ent
                break
        if not ref_entry:
            reference = {}
            for proto in ('pbg', 'pb', 'pg'):
                record = prompts_by_proto[proto].get((split, pair_id))
                if record:
                    reference.update({item['role']: item['asset_relative_path'] for item in record['inputs']})
            pair = next((r for r in locked_pairs.get(split, []) if r['pair_id'] == pair_id), {})
            ref_entry = {'row': {'metadata': pair, 'source': reference.get('source_A'), 'target': reference.get('target_B'), 'normal': reference.get('target_normal'), 'pose': reference.get('target_dwpose')}, 'dir': run_dir}

        row_meta = ref_entry["row"].get("metadata") or {}
        stem = hashlib.sha256(f'{split}:{pair_id}:{case_seed}'.encode()).hexdigest()[:24]
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
                ent = entries.get((arm, step, split, pair_id, case_seed))
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
            prec = prompts_by_proto[proto].get((split, pair_id))
            if prec:
                raw = {}
                if prec.get('raw_path'):
                    raw = json.loads((run_dir / prec['raw_path']).read_text(encoding='utf-8'))
                proto_prompts[proto] = {
                    "instruction": raw.get('instruction', 'P0 固定文本，无 PE 请求'),
                    "raw_answer": raw.get('raw_output', 'P0 无 PE 回答'),
                    "compiled": prec.get("compiled_prompt", ""),
                    "sha256": prec.get("compiled_prompt_sha256", ""),
                    'inputs': prec.get('inputs', []),
                    'record': prec,
                }

        case_views.append({
            "split": split,
            "pair_id": pair_id,
            'seed': case_seed,
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
        '    [hidden] { display: none !important; } .filters { display:flex;flex-wrap:wrap;gap:12px;position:sticky;top:0;background:white;padding:12px;z-index:5; } .grid-8 { overflow-x:auto; } .roi { display:block; } body[data-view="full"] .roi { display:none; } body[data-view="face"] .whole, body[data-view="face"] .roi:not(.face), body[data-view="left"] .whole, body[data-view="left"] .roi:not(.left), body[data-view="right"] .whole, body[data-view="right"] .roi:not(.right) { display:none; }',
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
        '<details class="card"><summary>实际 Preflight 证据</summary>' + text_block(preflight_evidence) + '</details>',
        "",
        "  <section class=\"card\">",
        "    <h2>1. 实验配置与架构隔离</h2>",
        "    <p><strong>共同 J256 初始化权重 SHA256:</strong> <code>" + escape(INIT_SHA256) + "</code></p>",
        "    <div class=\"scroll-table\">",
        "      <table>",
        "        <tr><th>组名 (Arm)</th><th>基干底座</th><th>提示词协议</th><th>加载 LoRA</th><th>Body分支</th><th>Face分支</th><th>Hand分支</th><th>Depth分支</th></tr>",
        "        <tr><td><code>qwen_p0</code></td><td>Qwen 2.1</td><td>P0 历史固定短提示词</td><td>无</td><td>开启</td><td>开启</td><td>关闭 (0)</td><td>关闭 (0)</td></tr>",
        "        <tr><td><code>qwen_pg</code></td><td>Qwen 2.1</td><td>PG：PE 看 A＋depth＋normal＋DWPose</td><td>无</td><td>开启</td><td>开启</td><td>关闭</td><td>关闭</td></tr>",
        "        <tr><td><code>qwen_pb</code></td><td>Qwen 2.1</td><td>PB：PE 看 A＋B，转写目标姿态</td><td>无</td><td>开启</td><td>开启</td><td>关闭</td><td>关闭</td></tr>",
        "        <tr><td><code>qwen_pbg</code></td><td>Qwen 2.1</td><td>PBG：PE 看 A＋B＋depth＋normal＋DWPose</td><td>无</td><td>开启</td><td>开启</td><td>关闭</td><td>关闭</td></tr>",
        "        <tr><td><code>head_p0</code></td><td>Qwen＋BFSHead</td><td>P0 历史固定短提示词</td><td>冻结，强度 1.0，从研究 step 0 加载</td><td>开启</td><td>开启</td><td>关闭</td><td>关闭</td></tr>",
        "        <tr><td><code>head_pg</code></td><td>Qwen＋BFSHead</td><td>PG：PE 看 A＋depth＋normal＋DWPose</td><td>冻结，强度 1.0，从研究 step 0 加载</td><td>开启</td><td>开启</td><td>关闭</td><td>关闭</td></tr>",
        "        <tr><td><code>head_pb</code></td><td>Qwen＋BFSHead</td><td>PB：PE 看 A＋B，转写目标姿态</td><td>冻结，强度 1.0，从研究 step 0 加载</td><td>开启</td><td>开启</td><td>关闭</td><td>关闭</td></tr>",
        "        <tr><td><code>head_pbg</code></td><td>Qwen＋BFSHead</td><td>PBG：PE 看 A＋B＋depth＋normal＋DWPose</td><td>冻结，强度 1.0，从研究 step 0 加载</td><td>开启</td><td>开启</td><td>关闭</td><td>关闭</td></tr>",
        "      </table>",
        "    </div>",
        '<p>以上为锁定的计划条件；实际训练配置文件、SHA 和完整数据锁见 <a href="conditions.json">实验条件 JSON</a>。各组均从 J256 开始，创建新优化器并额外训练 1024 步。Body 参数组学习率：2.5e-5 / 1.25e-5 / 2.5e-6；Face：5e-5 / 2.5e-5 / 5e-6；warmup 20 步。FP32 Adapter 与优化器状态，BF16 主干计算，batch=1。</p>',
        '<p>Body 接收 normal、Body25、part、26 维 SMPL-X global 与 mask；Face 独立接收目标面部几何和 A 身份。Depth、DWPose 本轮用于 PE 和核对图，不表示已启用相应 Adapter 分支。服装设计、颜色和纹理由 A 保持，褶皱与姿态阴影允许适应新动作。</p>',
        '<p>推理 30 步、CFG=1，Body=1、Face=1、Hand=0；每 128 步保留检查点与监测推理。完整验证每 256 步使用 seed 42/43。独立测试在全部八组训练完成后仅推理，不进行反向传播。</p>',
        "    <h3>严格隔离说明</h3>",
        '<details><summary>实际底座加载记录</summary>',
    ]
    for arm in ARMS:
        loading_file = run_dir / f'training/{arm}/lora_loading.json'
        if not loading_file.is_file():
            loading_file = run_dir / f'preflight/{arm}/lora_loading.json'
        html_lines.append(f'<h4>{escape(arm)}</h4>')
        if loading_file.is_file():
            html_lines.append(text_block(loading_file.read_text(encoding='utf-8')))
        else:
            html_lines.append('<p>实际加载记录尚未生成</p>')
    html_lines.extend([
        '</details>',
        "    <ul>",
        "      <li><strong>RGB 唯一输入:</strong> 源图 A RGB 为 Qwen 生成模型的唯一 RGB 输入；目标 B 可用于 PB/PBG 的 PE 重写、数值几何、尺寸与离线评分。验证/测试 B 不进入 Qwen 参考条件或目标 latent 缓存。</li>",
        "      <li><strong>PB / PBG 边界:</strong> B RGB 只给 PE 观察并转写可见姿态；训练 B latent 是监督目标，不能与图像参考混淆。测试 B 可用于几何和离线评分，不进入 Qwen 参考条件。</li>",
        "      <li><strong>原生分辨率评测:</strong> 保持 TikTok 原图高宽比，在原尺寸画布上进行反补边 (unpad) 后计算指标 (<code>per_sample_native</code>)。</li>",
        "    </ul>",
        "  </section>",
        "",
        "  <section class=\"card\">",
        "    <h2>2. 配对对比分析 (Paired Metric Differences)</h2>",
        "    <div class=\"scroll-table\">",
        "      <table>",
        "        <tr><th>对照类型</th><th>基准 (Left)</th><th>消融 (Right)</th><th>步数</th><th>Split</th><th>指标</th><th>有效对数 / 期望</th><th>配对均值差</th><th>分组宏平均差</th></tr>",
    ])

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
        *curve_html,
        "    <h2>3. 八组样本并排对比 (8-Arm Case Viewer)</h2>",
        "    <p>支持全身图像、目标 B 面部局部 ROI (各组同坐标)、目标 B 双手局部 ROI 以及提示词追溯。</p>",
        '<div class="filters"><label>数据分组 <select id="filter-split"><option value="">全部</option><option value="train">训练</option><option value="val">验证</option><option value="test">独立测试</option><option value="diagnostic">跨人物诊断</option></select></label><label>步数 <select id="filter-step"><option value="">全部</option>' + ''.join(f'<option>{step}</option>' for step in EVAL_STEPS) + '</select></label><label>随机种子 <select id="filter-seed"><option value="">全部</option><option>42</option><option>43</option></select></label><label>样本 <input id="filter-pair" type="search" placeholder="输入样本编号"></label><label>图像 <select id="filter-view"><option value="all">全身与局部</option><option value="full">全身</option><option value="face">面部</option><option value="left">左手</option><option value="right">右手</option></select></label></div>',
    ])

    for case in case_views:
        split = case["split"]
        pair_id = case["pair_id"]
        html_lines.extend([
            f'    <article class="case" data-split="{escape(split, quote=True)}" data-pair="{escape(pair_id, quote=True)}" data-seed="{case["seed"]}" style="margin-top:25px;border-top:2px solid #e2e8f0;padding-top:15px;">',
            f"      <h3>用例: {escape(split)} / {escape(pair_id)} / seed {case['seed']}</h3>",
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
        reference_row = next((e['row'] for e in entries.values() if e['split'] == split and e['pair_id'] == pair_id), {})
        reference_dir = next((e['dir'] for e in entries.values() if e['split'] == split and e['pair_id'] == pair_id), None)
        if reference_dir:
            html_lines.append('<details><summary>实际条件与 Body25 数值</summary>')
            for field, label in (('conditioning_normal', '实际法线'), ('conditioning_body25_preview', '实际 Body25 热图'), ('conditioning_depth', 'PE 深度'), ('conditioning_dwpose_original', '原始 DWPose'), ('conditioning_dwpose_filtered', '过滤 DWPose'), ('conditioning_body25', '实际 Body25 数值'), ('conditioning_dwpose_json', 'DWPose 原始数值')):
                value = reference_row.get(field)
                if value and (reference_dir / value).is_file():
                    path = reference_dir / value
                    html_lines.append(f'<a href="{safe_link(path, output_dir)}">{label}</a> ')
                    if path.suffix.lower() == '.png':
                        thumb = make_thumbnail(path, thumb_dir / f'condition_{hashlib.sha256(str(path).encode()).hexdigest()[:24]}.jpg', thumb_side)
                        html_lines.append(f'<img width="120" src="{safe_link(thumb, output_dir)}" alt="{label}">')
            html_lines.append('</details>')

        for step in all_available_steps:
            cells = case["steps"].get(step, {})
            html_lines.extend([
                f'<section class="case-step" data-step="{step}">',
                f"      <h4>步数 Step {step}</h4>",
                "      <div class=\"grid-8\">",
            ])
            for arm in ARMS:
                c = cells.get(arm, {"status": "pending"})
                st = c.get("status")
                html_lines.append(f"        <div class=\"cell\"><strong>{escape(arm)}</strong><br>")
                if st == "complete":
                    html_lines.append(f'          <a class="whole" href="{c["image_link"]}"><img src="{c["thumb_link"]}" alt="{escape(arm)}"></a>')
                    html_lines.append("          <div class=\"metrics\">")
                    if c.get("identity_A") is not None:
                        html_lines.append(f"            ID: {c['identity_A']:.3f}<br>")
                    if c.get("pck") is not None:
                        html_lines.append(f"            PCK: {c['pck']:.3f}<br>")
                    if c.get("bg_l1") is not None:
                        html_lines.append(f"            bgL1: {c['bg_l1']:.3f}<br>")
                    html_lines.append("          </div>")
                    for label, field, kind in (('目标 B 面部 ROI', 'face_thumb', 'face'), ('左手 ROI', 'lh_thumb', 'left'), ('右手 ROI', 'rh_thumb', 'right')):
                        if c.get(field):
                            html_lines.append(f'<div class="roi {kind}">{label}<img src="{c[field]}" alt="{label}"></div>')
                elif st == "failed":
                    html_lines.append(f'          <p class="failure">失败: {escape(str(c.get("failure")))}</p>')
                else:
                    html_lines.append("          <p style=\"color: #94a3b8; font-size: 0.85em;\">未评测</p>")
                html_lines.append("        </div>")
            html_lines.append("      </div>")
            html_lines.append('</section>')

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
                    '        <strong>完整来源记录:</strong>' + text_block(json.dumps(pinfo['record'], ensure_ascii=False, indent=2)),
                ])
                for item in pinfo['inputs']:
                    asset = run_dir / item['asset_relative_path']
                    thumb = make_thumbnail(asset, thumb_dir / f"pe_{proto}_{hashlib.sha256(str(asset).encode()).hexdigest()[:16]}.jpg", thumb_side)
                    html_lines.append(f'<div>实际 PE 输入: {escape(item["role"])} ({escape(item["sha256"][:12])})<a href="{safe_link(asset, output_dir)}"><img width="120" src="{safe_link(thumb, output_dir)}" alt="{escape(item["role"])}"></a></div>')
        html_lines.append("      </details>")
        html_lines.append("    </article>")

    html_lines.extend([
        "  </section>",
        '<section class="card"><h2>全部正式提示词与 PE 输入审查</h2><p>包含未进入训练监控图像的样本。P0 不计入 PE 重写成功数。</p>',
    ])
    for proto in PROTOCOLS:
        for rec in prompts_by_proto[proto].values():
            html_lines.extend(render_prompt_evidence(rec, run_dir, output_dir, thumb_dir, thumb_side))
    html_lines.append('</section><section class="card"><h2>开发 QA8 提示词审查</h2><p>开发记录独立展示，正式统计不重复计数。</p>')
    from src.pose_control.qwen21_full.prompt_study.prompts import validate_prompt_record
    for proto in ('pg', 'pb', 'pbg'):
        qa_file = run_dir / f'prompts/qa8/{proto}/train.jsonl'
        if qa_file.is_file():
            for line in qa_file.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    try:
                        rec = json.loads(line)
                        validate_prompt_record(rec, run_dir=run_dir)
                        html_lines.extend(render_prompt_evidence(rec, run_dir, output_dir, thumb_dir, thumb_side))
                    except (ValueError, KeyError, OSError) as exc:
                        html_lines.append('<p class="failure">开发记录校验失败：' + escape(str(exc)) + '</p>')
    html_lines.extend([
        '</section>',
        "",
        "  <section class=\"card\">",
        "    <h2>4. 异常与失败归档 (Diagnostic & Failure Audit)</h2>",
        f"    <p>失败或异常用例数: <strong>{len(failed_cases)}</strong></p>",
        f"    <p>评分程序错误数（图片保留，排除出完成数与指标）: <strong>{scoring_errors_count}</strong></p>",
        f"    <p>数据源一致性冲突 (Provenance Errors): <strong>{len(provenance_errors)}</strong></p>",
        '<details><summary>按推理种子列出的无脸、多脸与动作检测失败</summary><table><tr><th>组名 / 步数 / 数据分组 / 种子</th><th>生成图数</th><th>生成图无脸 / 多脸 / 未评分</th><th>A 无脸 / 多脸 / 未评分</th><th>无效身份指标</th><th>动作检测失败原因</th></tr>',
        *[f'<tr><td>{escape(row["arm"])} / {row["step"]} / {escape(row["split"])} / {row["seed"]}</td><td>{row["generated_images"]}</td><td>{row["output_no_face"]} / {row["output_multiple_faces"]} / {row["output_detection_unknown"]}</td><td>{row["source_no_face"]} / {row["source_multiple_faces"]} / {row["source_detection_unknown"]}</td><td>{row["identity_invalid"]}</td><td>{escape(json.dumps(row["pose_failures"], ensure_ascii=False))}</td></tr>' for row in detection_audit],
        '</table></details>',
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
        *historical_references(run_dir, output_dir, thumb_dir, thumb_side),
        "</body>",
        "</html>\n",
    ])
    html_lines.insert(-3, '<script type="text/javascript">(() => { const value=id=>document.getElementById(id).value; function filter(){document.body.dataset.view=value("filter-view"); document.querySelectorAll(".case").forEach(el=>{el.hidden=(value("filter-split") && el.dataset.split!==value("filter-split")) || (value("filter-seed") && el.dataset.seed!==value("filter-seed")) || !el.dataset.pair.toLowerCase().includes(value("filter-pair").toLowerCase());});document.querySelectorAll(".case-step").forEach(el=>{el.hidden=!!value("filter-step") && el.dataset.step!==value("filter-step");});} ["filter-split","filter-step","filter-seed","filter-pair","filter-view"].forEach(id=>document.getElementById(id).addEventListener("input",filter)); filter(); })();</script>')

    index_html = output_dir / "index.html"
    index_html.write_text("\n".join(html_lines), encoding="utf-8", newline="\n")

    summary_json = output_dir / "summary.json"
    summary_data = {
        "status": status_text,
        "is_completed": is_completed,
        "train_updates": train_updates,
        "pe_success": pe_success,
        "completed_eval_jobs": completed_eval_jobs,
        "scoring_errors_count": scoring_errors_count,
        "failed_cases_count": len(failed_cases),
        "provenance_errors_count": len(provenance_errors),
        "paired_results_count": len(paired_results),
        'paired_results': paired_results,
        'provenance_errors': provenance_errors,
        'eval_pending': 3696 - len(entries),
        'p0_records': len(prompts_by_proto['p0']),
        'preflight_passed': preflight_ok,
        'preflight_evidence': preflight_evidence,
        'metric_aggregates': [{'arm': arm, 'step': step, 'split': split, **aggregate} for (arm, step, split), aggregate in study_aggregates.items()],
        'detection_audit': detection_audit,
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
