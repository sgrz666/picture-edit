from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .protocol import (
    ARMS,
    EVAL_STEPS,
    FULL_VAL_STEPS,
    TRAIN_MONITOR,
    VAL_MONITOR,
)


def eval_jobs(*, include_final_test: bool = False) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []

    def append(arm: str, step: int, split: str, indices: Any, seeds: tuple[int, ...]) -> None:
        for seed in seeds:
            for index in indices:
                jobs.append(dict(arm=arm, step=step, split=split, pair_index=index, seed=seed))

    for arm in ARMS:
        for step in EVAL_STEPS:
            append(arm, step, "train", TRAIN_MONITOR, (42,))
            if step in FULL_VAL_STEPS:
                append(arm, step, "val", range(24), (42, 43))
            else:
                append(arm, step, "val", VAL_MONITOR, (42,))
            if step in (0, 512, 1024):
                append(arm, step, "diagnostic", range(6), (42,))
        if include_final_test:
            for step in (0, 1024):
                append(arm, step, "test", range(24), (42, 43))
    return jobs


def validate_retained_artifacts(directory: Path, row: dict[str, Any]) -> None:
    """Verify written reference bytes even for failed generation attempts."""
    from .protocol import sha256_file
    directory = Path(directory).resolve()
    artifacts = row.get('artifact_sha256')
    if not isinstance(artifacts, dict):
        raise ValueError('Evaluation exported reference hashes missing')
    for key in ('source', 'target', 'source_mask', 'target_mask'):
        if row.get(key) and key not in artifacts:
            raise ValueError(f'Evaluation exported reference hash missing: {key}')
    for key, digest in artifacts.items():
        value = row.get(key)
        if not isinstance(value, str):
            raise ValueError(f'Evaluation artifact path missing: {key}')
        artifact = (directory / value).resolve()
        if not artifact.is_relative_to(directory) or not artifact.is_file() or sha256_file(artifact) != digest:
            raise ValueError(f'Evaluation exported artifact changed: {key}')
    if row.get('status') == 'complete' and (not row.get('source') or not row.get('target')):
        raise ValueError('Evaluation original A/B reference images missing')


def validate_evaluation_artifacts(directory: Path, run_dir: Path, *, expected: dict[str, Any] | None = None, require_scores: bool = False, allow_score_failures: bool = False) -> dict[str, Any]:
    """Validate retained attempts against current locked split/text and actual PNG bytes."""
    from PIL import Image
    from .protocol import SCHEMA, INIT_SHA256, sha256_file, load_study_lock, validate_study_binding
    from .prompts import PromptBank
    directory, run_dir = Path(directory), Path(run_dir)
    lock = load_study_lock(run_dir / 'data/lock.json', verify_assets=False)
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    if config.get('study_schema') != SCHEMA:
        raise ValueError('Evaluation study provenance is missing')
    for name, value in (expected or {}).items():
        if config.get(name) != value:
            raise ValueError(f'Evaluation configuration differs: {name}')
    arm, split = config['arm'], config['study_split']
    if arm not in ARMS or split not in ('train', 'val', 'test', 'diagnostic'):
        raise ValueError('Unsupported evaluation arm/split')
    proto = arm.split('_', 1)[1]
    validate_study_binding(lock, arm, config.get('lora_sha256'), config.get('model', ''))
    if config.get('resolution') != 'native' or config.get('align_mode') != 'native' or config.get('steps') != 30 or config.get('generation_rgb_inputs') != ['Source A'] or config.get('with_hand') is not False:
        raise ValueError('Evaluation native/source-only/no-Hand configuration differs')
    checkpoint_metadata = json.loads((directory / 'checkpoint_metadata.json').read_text(encoding='utf-8'))
    signature = checkpoint_metadata['joint_signature']
    if checkpoint_metadata.get('study_schema') != SCHEMA or checkpoint_metadata.get('joint_initialization_sha256') != INIT_SHA256 or checkpoint_metadata.get('arm') != arm or checkpoint_metadata.get('protocol') != proto or checkpoint_metadata.get('max_updates') != 1024:
        raise ValueError('Evaluation common initialization/arm/protocol/budget differs')
    if checkpoint_metadata.get('step') != config['study_step'] or signature.get('arm') != arm or signature.get('study_lock_sha256') != config.get('lock_sha') or signature.get('max_updates') != 1024:
        raise ValueError('Evaluation checkpoint metadata differs')
    validate_study_binding(lock, arm, signature.get('lora_sha256'), signature.get('model', ''))
    loading = json.loads((directory / 'lora_loading.json').read_text(encoding='utf-8'))
    if loading.get('file_sha256') != config.get('lora_sha256') or loading.get('checkpoint_sha256') != config.get('checkpoint_sha256'):
        raise ValueError('Actual loaded checkpoint/LoRA provenance differs')
    for name in ('checkpoint_sha256', 'model_config_sha256'):
        value = config.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f'Missing model/checkpoint provenance: {name}')
    filename = {'train': 'train512_pairs.jsonl', 'val': 'val24_pairs.jsonl', 'test': 'test24_pairs.jsonl', 'diagnostic': 'diagnostic6_pairs.jsonl'}[split]
    pairs_file = run_dir / 'data' / filename
    prompt_file = run_dir / f'prompts/{proto}/{split}.jsonl'
    for name, path in (('lock_sha', run_dir / 'data/lock.json'), ('pairs_sha256', pairs_file), ('prompt_manifest_sha256', prompt_file)):
        if config.get(name) != sha256_file(path):
            raise ValueError(f'Evaluation provenance hash differs: {name}')
    pairs = [json.loads(line) for line in pairs_file.read_text(encoding='utf-8').splitlines() if line.strip()]
    bank = PromptBank(prompt_file, proto, [r['pair_id'] for r in pairs])
    if signature.get('prompt_manifest_sha256') != sha256_file(run_dir / f'prompts/{proto}/train.jsonl') or signature.get('instruction_sha256') != sha256_file(run_dir / 'prompts/instruction_v1.json'):
        raise ValueError('Evaluation checkpoint training prompt/template differs')
    indices = config['indices']
    if len(set(indices)) != len(indices) or any(not isinstance(i, int) or not 0 <= i < len(pairs) for i in indices):
        raise ValueError('Duplicate/out-of-range evaluation indices')
    jobs = {(j['arm'], j['step'], j['split'], j['pair_index'], j['seed']) for j in eval_jobs(include_final_test=True)}
    if any((arm, config['study_step'], split, i, config['seed']) not in jobs for i in indices):
        raise ValueError('Evaluation is outside the locked schedule')
    rows = [json.loads(line) for line in (directory / 'per_sample.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(rows) != len(indices) or {r.get('pair_index') for r in rows} != set(indices):
        raise ValueError('Incomplete or duplicate evaluation attempts; preserve directory and investigate')
    for row in rows:
        pair_id = pairs[row['pair_index']]['pair_id']
        if row.get('pair_id') != pair_id or row.get('arm') != arm or row.get('study_step') != config['study_step'] or row.get('seed') != config['seed'] or row.get('study_split') != split:
            raise ValueError('Evaluation row key differs from locked job')
        if row.get('status') not in ('complete', 'failed'):
            raise ValueError('Evaluation attempt status invalid')
        validate_retained_artifacts(directory, row)
        if row['status'] == 'failed':
            continue
        record = bank.records[pair_id]
        if row.get('prompt_sha256') != record['compiled_prompt_sha256'] or row.get('prompt') != record['compiled_prompt'] or row.get('initialization') != 'fresh_gaussian':
            raise ValueError('Evaluation actual prompt/initialization differs')
        image = (directory / row['image']).resolve()
        if not image.is_relative_to(directory.resolve()) or image.suffix.lower() != '.png' or sha256_file(image) != row.get('image_sha256'):
            raise ValueError('Evaluation PNG missing or changed')
        with Image.open(image) as im:
            if [im.width, im.height] != row.get('image_size') or im.format != 'PNG':
                raise ValueError('Evaluation PNG dimensions differ')
            im.verify()
    scores: dict[int, Any] = {}
    scores_path = directory / 'scores.json'
    if scores_path.is_file():
        data = json.loads(scores_path.read_text(encoding='utf-8'))
        if data.get('schema') != 'qwen21_joint_pilot_scores_v2' or data.get('metric_resolution') != 'per_sample_native':
            raise ValueError('Scoring resolution provenance missing')
        expected_scoring = lock.get('scoring_signature_sha256')
        if not isinstance(expected_scoring, str) or len(expected_scoring) != 64 or data.get('scoring_signature_sha256') != expected_scoring:
            raise ValueError('Scoring model/code/version provenance differs')
        if data.get('config_sha256') != sha256_file(directory / 'config.json') or data.get('per_sample_sha256') != sha256_file(directory / 'per_sample.jsonl'):
            raise ValueError('Offline scoring snapshot differs from retained evaluation')
        items = data.get('per_sample', [])
        if len(items) != len(rows) or {s.get('pair_index') for s in items} != set(indices):
            raise ValueError('Incomplete or duplicate offline score records')
        scores = {s['pair_index']: s for s in items}
        for row in rows:
            score = scores[row['pair_index']]
            for key in ('pair_id', 'arm', 'protocol', 'study_step', 'study_split', 'seed', 'prompt_sha256', 'image_sha256', 'image_size', 'status'):
                if score.get(key) != row.get(key):
                    raise ValueError(f'Offline score row provenance differs: {key}')
            if score.get('score_failure') and not allow_score_failures:
                raise ValueError(f"Offline scoring failed for {row['pair_id']}: {score['score_failure']}")
    elif require_scores:
        raise ValueError('Offline scores missing')
    return {'dir': directory, 'config': config, 'rows': rows, 'scores': scores}


def paired_metric(
    left: list[dict[str, Any]], right: list[dict[str, Any]], metric: str
) -> dict[str, Any]:
    right_map = {(r["split"], r["pair_id"]): r for r in right}
    values: list[float] = []
    group_values: dict[str, list[float]] = defaultdict(list)

    def finite(value: Any) -> bool:
        return (
            isinstance(value, (float, int))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )

    for row in left:
        other = right_map.get((row["split"], row["pair_id"]))
        a = row["metrics"].get(metric)
        b = None if other is None else other["metrics"].get(metric)
        if finite(a) and finite(b):
            diff = float(b - a)
            values.append(diff)
            grp = str(row.get("group", "default"))
            group_values[grp].append(diff)

    means = {g: sum(v) / len(v) for g, v in group_values.items()} if group_values else {}
    return {
        "expected_n": len(left),
        "valid_pair_n": len(values),
        "missing_or_invalid_n": len(left) - len(values),
        "mean_difference": sum(values) / len(values) if values else None,
        "group_macro_mean_difference": sum(means.values()) / len(means) if means else None,
        "group_means": means,
    }


def compare_private_continuations(
    left_state: dict[str, Any],
    right_state: dict[str, Any],
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    from scripts.run.run_qwen21_joint_pilot import _state_drift, _loss_drift
    from src.pose_control.qwen21_full.joint_training import assert_exact_state

    errors: list[str] = []
    if (
        left_state["step"] != 4
        or right_state["step"] != 4
        or len(left_rows) != 4
        or len(right_rows) != 4
    ):
        raise ValueError("Private continuation must end at four updates")
    for key in ("step", "sample_order", "joint_signature", "joint_initialization_sha256"):
        assert_exact_state(left_state[key], right_state[key], key)
    assert_exact_state(left_rows[:2], right_rows[:2], "forked_metrics_prefix")
    for a, b in zip(left_rows, right_rows):
        for key in (
            "step",
            "pair_index",
            "pair_id",
            "noise_seed",
            "prompt_sha256",
            "sigma",
            "face_active",
        ):
            assert_exact_state(a[key], b[key], f"metrics.{key}")
    if len(left_rows) != len(right_rows):
        raise ValueError("Continuation metric lengths differ")
    return {
        "adapter": _state_drift(
            left_state["adapter_state_dict"],
            right_state["adapter_state_dict"],
            "adapter",
            errors,
        ),
        "optimizer": _state_drift(
            left_state["optimizer_state_dict"],
            right_state["optimizer_state_dict"],
            "resume.optimizer",
            errors,
        ),
        "losses": _loss_drift(left_rows, right_rows, "losses", errors),
        "errors": errors,
    }


PROHIBITED_EXPORT_EXTENSIONS = {".pt", ".safetensors", ".bin", ".onnx"}
ALLOWED_EXPORT_PREFIXES = ("data", "prompts", "training", "evaluation", "report", "preflight", "history")


def aggregate_pair_metrics(
    rows: list[dict[str, Any]],
    metrics: tuple[str, ...] = ("identity_A", "pose_pck_002", "bg_l1_A", "pixel_l1_GT"),
    *, expected_pairs: list[dict[str, Any]] | None = None, expected_seeds: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    def finite(val: Any) -> bool:
        return isinstance(val, (float, int)) and not isinstance(val, bool) and math.isfinite(val)

    pairs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        pairs[(str(r.get("split", "val")), str(r.get("pair_id", "")))].append(r)
    expected_map = {(str(r['split']), str(r['pair_id'])): r for r in expected_pairs or []}
    for key in expected_map:
        pairs.setdefault(key, [])

    aggregated_rows: list[dict[str, Any]] = []
    group_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    metric_values: dict[str, list[float]] = defaultdict(list)

    for (split, pair_id), items in sorted(pairs.items()):
        metadata = expected_map.get((split, pair_id), items[0] if items else {})
        grp = str(metadata.get("source_video_group") or metadata.get("group", "unknown"))
        pair_metrics: dict[str, float | None] = {}
        seed_counts: dict[str, int] = {}
        attempted_seeds = {it.get('seed') for it in items}
        invalid_seed_counts: dict[str, int] = {}
        for m in metrics:
            vals = [
                it["metrics"][m]
                for it in items
                if "metrics" in it and m in it["metrics"] and finite(it["metrics"][m])
            ]
            valid_seeds = {it.get('seed') for it in items if finite(it.get('metrics', {}).get(m))}
            seed_counts[m] = len(valid_seeds)
            invalid_seed_counts[m] = len(attempted_seeds - valid_seeds)
            complete = expected_seeds is None or attempted_seeds == set(expected_seeds)
            if vals and complete:
                mean_v = sum(vals) / len(vals)
                pair_metrics[m] = mean_v
                group_values[grp][m].append(mean_v)
                metric_values[m].append(mean_v)
            else:
                pair_metrics[m] = None
        aggregated_rows.append({
            "split": split,
            "pair_id": pair_id,
            "group": grp,
            "metrics": pair_metrics,
            "raw_seeds": [it.get("seed") for it in items],
            'valid_seed_counts': seed_counts,
            'expected_seeds': list(expected_seeds) if expected_seeds is not None else None,
            'invalid_seed_counts': invalid_seed_counts,
            'missing_seed_count': len(set(expected_seeds or ()) - attempted_seeds),
            'seed_details': [{'seed': it.get('seed'), 'status': it.get('status', 'complete'), 'metrics': it.get('metrics', {})} for it in items],
        })

    group_means: dict[str, dict[str, float]] = defaultdict(dict)
    for g, m_dict in group_values.items():
        for m, vals in m_dict.items():
            if vals:
                group_means[g][m] = sum(vals) / len(vals)

    group_macro_means: dict[str, float | None] = {}
    overall_means: dict[str, float | None] = {}
    valid_counts: dict[str, int] = {}
    for m in metrics:
        g_vals = [group_means[g][m] for g in group_means if m in group_means[g]]
        group_macro_means[m] = sum(g_vals) / len(g_vals) if g_vals else None
        overall_means[m] = sum(metric_values[m]) / len(metric_values[m]) if metric_values[m] else None
        valid_counts[m] = len(metric_values[m])

    return {
        "pairs": aggregated_rows,
        "group_means": dict(group_means),
        "group_macro_means": group_macro_means,
        "overall_means": overall_means,
        "valid_counts": valid_counts,
        "expected_n": len(pairs),
        "missing_or_invalid_counts": {m: len(pairs) - valid_counts[m] for m in metrics},
    }


def verify_report_relative_links(report_html_path: Path, run_dir: Path) -> list[str]:
    """Parse report/index.html and verify every href/src relative link points to an existing file."""
    import re
    from urllib.parse import unquote

    if not report_html_path.is_file():
        raise FileNotFoundError(f"Report file not found: {report_html_path}")

    content = report_html_path.read_text(encoding="utf-8")
    report_dir = report_html_path.parent
    run_dir = run_dir.resolve()

    missing_links: list[str] = []
    pattern = re.compile(r'(?:href|src)=["\']([^"\']+)["\']', re.IGNORECASE)
    for match in pattern.finditer(content):
        raw_url = match.group(1).strip()
        if not raw_url or raw_url.startswith(("#", "http://", "https://", "data:", "javascript:")):
            continue
        cleaned = unquote(raw_url.split("?")[0].split("#")[0])
        target = (report_dir / cleaned).resolve()
        if not target.is_relative_to(run_dir):
            missing_links.append(f"Escapes run directory: {raw_url}")
            continue
        if not target.exists():
            missing_links.append(f"Missing referenced file: {raw_url} (resolved: {target})")

    return missing_links


def build_export_manifest(run_dir: Path) -> list[dict[str, Any]]:
    """Collect exportable files excluding .pt, .safetensors, .bin, .onnx, and cache/."""
    from .protocol import sha256_file

    run_dir = run_dir.resolve()
    manifest: list[dict[str, Any]] = []

    for prefix in ALLOWED_EXPORT_PREFIXES:
        pdir = run_dir / prefix
        if not pdir.is_dir():
            continue
        for path in sorted(pdir.rglob("*")):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != run_dir and parent.is_relative_to(run_dir)):
                raise ValueError(f"Export symlink is not permitted: {path}")
            if not resolved.is_relative_to(run_dir):
                raise ValueError(f"File {path} escapes run directory {run_dir}")
            rel = path.relative_to(run_dir)
            rel_parts = [p.lower() for p in rel.parts]
            if any(p in ("cache", ".cache", "__pycache__") for p in rel_parts):
                continue
            if path.suffix.lower() in PROHIBITED_EXPORT_EXTENSIONS:
                continue

            manifest.append({
                "path": rel.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    return manifest


def package_export(run_dir: Path, output_tar: Path | None = None) -> dict[str, Any]:
    """Package verified exportable files into export/results.tar.gz and write export/files.json."""
    import shutil
    import tarfile

    run_dir = run_dir.resolve()
    export_dir = run_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    import uuid
    from .protocol import atomic_json, sha256_file
    staging_dir = export_dir / f"staging_{uuid.uuid4().hex}"

    # Validate the copied snapshot, including links to files excluded from export.
    files = build_export_manifest(run_dir)
    needed = sum(entry['bytes'] for entry in files)
    if shutil.disk_usage(export_dir).free < 2 * needed + 1024 * 1024:
        raise RuntimeError('Insufficient free space for export staging and archive')
    staging_dir.mkdir(parents=True)
    atomic_json(export_dir / 'status.json', {'status': 'staging', 'staging_dir': str(staging_dir), 'files_count': len(files)})
    files_json = export_dir / "files.json"
    files_json.write_text(json.dumps(files, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    for entry in files:
        rel = Path(entry["path"])
        src = run_dir / rel
        dst = staging_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if dst.stat().st_size != entry['bytes'] or sha256_file(dst) != entry['sha256']:
            raise RuntimeError(f"Export snapshot changed while copying: {entry['path']}")

    report_html = staging_dir / 'report/index.html'
    if report_html.is_file():
        missing = verify_report_relative_links(report_html, staging_dir)
        if missing:
            atomic_json(export_dir / 'status.json', {'status': 'failed', 'staging_dir': str(staging_dir), 'errors': missing})
            raise RuntimeError(f"Report links verification failed in staged export: {missing[:5]}")

    # Copy files.json into staging as well
    shutil.copy2(files_json, staging_dir / "files.json")

    # 4. Create tar.gz
    tar_path = output_tar or (export_dir / "results.tar.gz")
    tar_path = Path(tar_path)
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_tar = tar_path.with_name(tar_path.name + '.tmp')
    with tarfile.open(temporary_tar, "w:gz") as tar:
        for entry in files:
            rel = entry["path"]
            tar.add(staging_dir / rel, arcname=rel)
        tar.add(staging_dir / "files.json", arcname="files.json")
    temporary_tar.replace(tar_path)
    atomic_json(export_dir / 'status.json', {'status': 'complete', 'tar_path': str(tar_path), 'files_count': len(files), 'total_bytes': needed, 'archive_sha256': sha256_file(tar_path), 'staging_dir': str(staging_dir)})

    return {
        "tar_path": str(tar_path),
        "files_count": len(files),
        "total_bytes": sum(f["bytes"] for f in files),
        "files_json": str(files_json),
    }
