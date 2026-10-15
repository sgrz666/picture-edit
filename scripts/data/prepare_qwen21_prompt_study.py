#!/usr/bin/env python3
"""Prepare and lock dataset splits, geometry assets, and metadata for the PE-I2I study.

Performs deterministic test24 selection, video group disjointness audit,
DWPose confidence-filtering rendering, and lock.json generation.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pose_control.qwen21_full.prompt_study.protocol import (
    INIT_SHA256,
    TRAIN_SHA256,
    VAL_SHA256,
    atomic_json,
    sha256_file,
    text_sha256,
    DATA_LOCK_SCHEMA, SCHEMA, ARMS, MODEL_REVISION, directory_artifact, load_study_lock,
)
from src.pose_control.qwen21_full.prompt_study.sampling import make_order
from src.pose_control.qwen21_full.prompt_study.pairs import (
    select_test24,
    audit_video_groups,
    normalize_diagnostic_row,
    TEST_GROUPS,
)
from src.pose_control.qwen21_full.prompt_study.geometry import render_dwpose

DEFAULT_FACE_RUN = Path('/home/shangguanrz/nihaochen/qwen21_full/tiktok_face_v1_20261004/data')
DEFAULT_MODEL = Path('/home/shangguanrz/project/pic-edit/models/Qwen-Image-2.1')
DEFAULT_INIT = Path('/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/training/J/checkpoint_step_000256.pt')
DEFAULT_HEAD = Path('/home/shangguanrz/models/bfs/bfs_head_v1.1_qwen_2.1.safetensors')
DEFAULT_PE = Path('/data/shangguanrz/pic-edit-models/Qwen-Image-2.1-PE-I2I')


def candidate_failure_reasons(row: dict[str, Any], bundles: dict[str, Any]) -> list[str]:
    """Eligibility depends only on input assets, never generated quality or Face visibility."""
    from PIL import Image
    import torch
    reasons = []
    fields = ('source_rgb', 'target_rgb', 'source_person_mask', 'target_person_mask',
              'target_normal', 'target_depth_vis', 'target_dwpose', 'target_dwpose_json')
    for field in fields:
        if not row.get(field) or not Path(row[field]).is_file():
            reasons.append(f'missing_{field}')
    if reasons:
        return reasons
    try:
        with Image.open(row['target_rgb']) as image:
            target_size = image.size
            image.verify()
        for field in ('source_rgb', 'source_person_mask', 'target_person_mask', 'target_normal', 'target_depth_vis', 'target_dwpose'):
            with Image.open(row[field]) as image:
                size = image.size
                image.verify()
            if field.startswith('target') and size != target_size:
                reasons.append(f'{field}_native_size_differs')
        dwpose = json.loads(Path(row['target_dwpose_json']).read_text())
        if (dwpose['image_width'], dwpose['image_height']) != target_size:
            reasons.append('dwpose_json_native_size_differs')
        render_dwpose(dwpose)  # validates134 points, confidence, native canvas
        seq, stem = str(row['sequence_id']), str(row['target_stem'])
        root = Path(row['target_normal']).parent.parent
        part = root / 'part_512' / f'{stem}.png'
        with Image.open(part) as image:
            import numpy as np
            labels = np.asarray(image)
            if labels.ndim != 2 or labels.min() < 0 or labels.max() > 14:
                reasons.append('invalid_part_labels')
        bundle_path = root / 'v6_conditions.pt'
        if str(bundle_path) not in bundles:
            bundles[str(bundle_path)] = torch.load(bundle_path, map_location='cpu', weights_only=False, mmap=True)
        bundle = bundles[str(bundle_path)]
        index = bundle['stem_to_idx'][stem]
        points, vector = bundle['kps_25_coords'][index], bundle['smplx_global'][index]
        if points.shape != (25, 3) or vector.shape != (26,) or not torch.isfinite(points).all() or not torch.isfinite(vector).all():
            reasons.append('invalid_body_condition_shape_or_values')
        else:
            good = (points[:, 2] > .05) & ((points[:, :2] >= 0) & (points[:, :2] <= 1)).all(-1)
            if int(good.sum()) < 8:
                reasons.append('fewer_than_8_visible_body_joints')
    except (OSError, KeyError, ValueError, RuntimeError) as exc:
        reasons.append(f'invalid_asset:{type(exc).__name__}:{exc}')
    return reasons


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def prepare_data_lock(
    source_pilot: Path,
    test_candidates_path: Path,
    diagnostics_path: Path,
    output_dir: Path,
    face_source_run: Path | None = None,
    model: Path = DEFAULT_MODEL, initial_checkpoint: Path = DEFAULT_INIT,
    head_lora: Path = DEFAULT_HEAD, pe_model: Path = DEFAULT_PE,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    data_dir = output_dir if output_dir.name == "data" else output_dir / "data"
    lock_path = data_dir / 'lock.json'
    if lock_path.is_file():
        existing = load_study_lock(lock_path)
        for key, actual in (('model', model), ('initial_checkpoint', initial_checkpoint),
                            ('head_lora', head_lora), ('pe_model', pe_model)):
            if Path(existing['artifacts'][key]['path']).resolve() != Path(actual).resolve():
                raise ValueError(f'Existing lock uses a different {key}; preserve the run')
        return existing
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy train512 and val24 from source pilot
    src_train = source_pilot / "train512_pairs.jsonl"
    src_val = source_pilot / "val24_pairs.jsonl"
    if not src_train.is_file() or not src_val.is_file():
        # Maybe source_pilot is experiment root, check data subdir
        if (source_pilot / "data/train512_pairs.jsonl").is_file():
            src_train = source_pilot / "data/train512_pairs.jsonl"
            src_val = source_pilot / "data/val24_pairs.jsonl"
        else:
            raise FileNotFoundError(f"Missing train512/val24 under {source_pilot}")

    dst_train = data_dir / "train512_pairs.jsonl"
    dst_val = data_dir / "val24_pairs.jsonl"
    shutil.copyfile(src_train, dst_train)
    shutil.copyfile(src_val, dst_val)

    train_sha = sha256_file(dst_train)
    val_sha = sha256_file(dst_val)
    if train_sha != TRAIN_SHA256:
        raise ValueError(f"train512 hash mismatch: expected {TRAIN_SHA256}, got {train_sha}")
    if val_sha != VAL_SHA256:
        raise ValueError(f"val24 hash mismatch: expected {VAL_SHA256}, got {val_sha}")

    train_rows = read_jsonl(dst_train)
    val_rows = read_jsonl(dst_val)

    # 2. Select test24
    test_candidates = read_jsonl(test_candidates_path)
    rejected_candidates = []
    eligible_candidates = []
    bundles: dict[str, Any] = {}

    for row in test_candidates:
        # Check basic asset presence if accessible
        group = row.get("source_video_group")
        if group not in TEST_GROUPS:
            rejected_candidates.append({"pair_id": row.get("pair_id"), "reason": f"group {group} not in test groups"})
            continue
        reasons = candidate_failure_reasons(row, bundles)
        if reasons:
            rejected_candidates.append({'pair_id': row.get('pair_id'), 'reasons': reasons})
        else:
            eligible_candidates.append(row)

    test_rows = select_test24(eligible_candidates)
    dst_test = data_dir / "test24_pairs.jsonl"
    write_jsonl(dst_test, test_rows)
    test_sha = sha256_file(dst_test)

    dst_rejected = data_dir / "rejected_test_candidates.jsonl"
    write_jsonl(dst_rejected, rejected_candidates)

    # 3. Normalize diagnostic rows
    raw_diag_all = read_jsonl(diagnostics_path)
    by_id = {r['pair_id']: r for r in raw_diag_all if r.get('pair_id') in {f'tiktok_{i:02d}' for i in range(6)}}
    if len(by_id) != 6:
        raise ValueError('Missing required tiktok_00..05 cross-person diagnostic rows')
    raw_diag = [by_id[f'tiktok_{i:02d}'] for i in range(6)]
    diag_rows = [normalize_diagnostic_row(r) for r in raw_diag]
    dst_diag = data_dir / "diagnostic6_pairs.jsonl"
    write_jsonl(dst_diag, diag_rows)
    diag_sha = sha256_file(dst_diag)

    # 4. Audit video groups
    face_source_run = Path(face_source_run or DEFAULT_FACE_RUN)
    face_manifest = face_source_run / 'train_pairs.jsonl'
    face_rows = read_jsonl(face_manifest)
    face_groups = {r['source_video_group'] for r in face_rows}
    group_audit = audit_video_groups(train_rows, val_rows, test_rows, face_groups)

    # 5. Build geometry previews for rows with available dwpose_json
    geom_dir = data_dir / "geometry"
    assets_records = []
    all_rows = [("train", r) for r in train_rows] + [("val", r) for r in val_rows] + [("test", r) for r in test_rows] + [("diagnostic", r) for r in diag_rows]
    for split_name, row in all_rows:
        pid = row["pair_id"]
        dwpose_json = row.get("target_dwpose_json")
        if not dwpose_json or not Path(dwpose_json).is_file():
            raise FileNotFoundError(f'Missing actual DWPose JSON for {pid}')
        if dwpose_json:
            pair_geom = geom_dir / pid
            pair_geom.mkdir(parents=True, exist_ok=True)
            rec = json.loads(Path(dwpose_json).read_text(encoding="utf-8"))
            filt_img, diag = render_dwpose(rec)
            filt_path = pair_geom / "dwpose_filtered.png"
            filt_img.save(filt_path)
            orig_png = row.get("target_dwpose_png") or row.get("target_dwpose")
            if orig_png and Path(orig_png).is_file():
                shutil.copyfile(orig_png, pair_geom / "dwpose_original.png")
            assets_records.append({
                "pair_id": pid,
                "split": split_name,
                "filtered_dwpose": str(filt_path),
                "diagnostics": diag,
                "filtered_sha256": sha256_file(filt_path),
                "original_dwpose": str(pair_geom / 'dwpose_original.png'),
                "original_sha256": sha256_file(pair_geom / 'dwpose_original.png'),
                "dwpose_json_sha256": sha256_file(dwpose_json),
                "input_hashes": {k: sha256_file(row[k]) for k in ('source_rgb', 'target_rgb', 'target_normal', 'target_depth_vis')},
            })

    from src.pose_control.qwen21_full.prompt_study.source_assets import add_source_asset_hashes
    assets_records = add_source_asset_hashes(assets_records, all_rows)
    write_jsonl(data_dir / "assets.jsonl", assets_records)

    # 6. Build and write lock.json
    lock_payload = {
        "schema": DATA_LOCK_SCHEMA,
        "config": dict(schema=SCHEMA, max_updates=1024, eval_every=128, resolution='native',
                       align_mode='native', with_hand=False, hand_active=False, use_depth=False, arms=ARMS),
        "epoch_order": make_order(),
        "source_assets_manifest": {"path": "assets.jsonl", "sha256": sha256_file(data_dir / 'assets.jsonl'), "count": len(assets_records)},
        "splits": {
            "train": {"path": "train512_pairs.jsonl", "count": len(train_rows), "sha256": train_sha},
            "val": {"path": "val24_pairs.jsonl", "count": len(val_rows), "sha256": val_sha},
            "test": {"path": "test24_pairs.jsonl", "count": len(test_rows), "sha256": test_sha},
            "diagnostic": {"path": "diagnostic6_pairs.jsonl", "count": len(diag_rows), "sha256": diag_sha},
        },
        "video_group_audit": group_audit,
        "initial_checkpoint_sha256": INIT_SHA256,
        "historical_face_train": {'path': str(face_manifest), 'sha256': sha256_file(face_manifest), 'count': len(face_rows)},
        "artifacts": {'model': directory_artifact(model, MODEL_REVISION),
                      'initial_checkpoint': {'path': str(initial_checkpoint.resolve()), 'sha256': sha256_file(initial_checkpoint)},
                      'head_lora': {'path': str(head_lora.resolve()), 'sha256': sha256_file(head_lora)},
                      'pe_model': directory_artifact(pe_model)},
        "geometry_assets": [{'path': str((geom_dir / r['pair_id'] / name).relative_to(data_dir)),
                              'sha256': sha256_file(geom_dir / r['pair_id'] / name)}
                             for r in assets_records for name in ('dwpose_filtered.png', 'dwpose_original.png')],
    }
    lock_path = data_dir / "lock.json"
    from src.pose_control.qwen21_full.prompt_study.measurement import measurement_artifacts, scoring_signature
    lock_payload['measurement_artifacts'] = measurement_artifacts(ROOT / 'models', code_root=ROOT)
    lock_payload['scoring_signature_sha256'] = scoring_signature(lock_payload['measurement_artifacts'])
    if lock_payload['artifacts']['initial_checkpoint']['sha256'] != INIT_SHA256:
        raise ValueError('Common initialization bytes differ from approved J256')
    if lock_path.is_file() and json.loads(lock_path.read_text()) != lock_payload:
        raise ValueError('Existing experiment lock differs; preserve it and use a new run directory')
    atomic_json(lock_path, lock_payload)
    return lock_payload


def verify_data_lock(data_dir: Path) -> bool:
    load_study_lock(Path(data_dir) / 'lock.json')
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or verify PE-I2I prompt study data lock")
    parser.add_argument("--source-pilot", type=Path, help="Directory containing original pilot train512 and val24")
    parser.add_argument("--face-source-run", type=Path, help="Face run source directory")
    parser.add_argument("--test-candidates", type=Path, help="Manifest for test candidates")
    parser.add_argument("--diagnostics", type=Path, help="Manifest for diagnostic pairs")
    parser.add_argument("--output", type=Path, help="Output data directory")
    parser.add_argument("--verify", action="store_true", help="Verify existing data lock")
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--init-checkpoint', type=Path, default=DEFAULT_INIT)
    parser.add_argument('--head-lora', type=Path, default=DEFAULT_HEAD)
    parser.add_argument('--pe-model', type=Path, default=DEFAULT_PE)
    args = parser.parse_args()

    if args.verify:
        target_dir = args.output if args.output else Path("data")
        verify_data_lock(target_dir)
        print("Data lock verification succeeded.")
    else:
        if not all([args.source_pilot, args.test_candidates, args.diagnostics, args.output]):
            parser.error("Preparing lock requires --source-pilot, --test-candidates, --diagnostics, --output")
        prepare_data_lock(
            args.source_pilot,
            args.test_candidates,
            args.diagnostics,
            args.output,
            args.face_source_run,
            args.model, args.init_checkpoint, args.head_lora, args.pe_model,
        )
        print("Data lock prepared successfully.")


if __name__ == "__main__":
    main()
