#!/usr/bin/env python3
"""Unified experiment runner and phase dispatcher for the 8-arm PE-I2I study.

Manages phases: lock, rewrite-qa, rewrite, faces, cache, selfcheck,
preflight, initial, train, finaltest, report, export.
Enforces process isolation, dependency checks, and the final test gate.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARMS = tuple(f"{base}_{p}" for base in ("qwen", "head") for p in ("p0", "pg", "pb", "pbg"))

PHASES = (
    "lock",
    "rewrite-qa",
    "rewrite",
    "faces",
    "cache",
    "selfcheck",
    "preflight",
    "initial",
    "train",
    "finaltest",
    "report",
    "export",
)

DEFAULT_INIT_CHECKPOINT = "/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/training/J/checkpoint_step_000256.pt"
DEFAULT_HEAD_LORA = "/home/shangguanrz/models/bfs/bfs_head_v1.1_qwen_2.1.safetensors"


def invoke(argv: list[str], dry_run: bool = False) -> None:
    print("RUN:", " ".join(argv), flush=True)
    if dry_run:
        return
    subprocess.run(argv, check=True, cwd=ROOT)


def check_eight_arms_completed(run_dir: Path) -> bool:
    from src.pose_control.qwen21_full.prompt_study.protocol import require_eight_arms_completed
    try:
        require_eight_arms_completed(run_dir)
        return True
    except (RuntimeError, ValueError, OSError):
        return False


def run_selfcheck(run_dir: Path) -> dict[str, Any]:
    from src.pose_control.qwen21_full.prompt_study.protocol import P0_PROMPT, PROTOCOLS, INIT_SHA256, TRAIN_SHA256, VAL_SHA256, atomic_json, load_study_lock, sha256_file
    from src.pose_control.qwen21_full.prompt_study.prompts import PromptBank, request_roles
    from src.pose_control.qwen21_full.prompt_study.sampling import make_order, noise_seed, validate_trajectory
    import ast
    run_dir = Path(run_dir)
    data_dir = run_dir / "data"
    preflight_dir = run_dir / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)

    checks: dict[str, Any] = {}

    # 1. P0 Prompt identical to legacy
    tree = ast.parse((ROOT / 'src/pose_control/qwen21_full/qwen_runtime.py').read_text(encoding='utf-8'))
    legacy = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PROMPT' for t in n.targets))
    checks["p0_prompt_byte_match"] = P0_PROMPT == legacy
    errors: dict[str, str] = {}
    try:
        load_study_lock(data_dir / 'lock.json')
        checks['lock_actual_assets_and_data'] = True
    except Exception as exc:
        checks['lock_actual_assets_and_data'] = False
        errors['lock'] = str(exc)

    # 2. Lock file exists and valid
    lock_path = data_dir / "lock.json"
    checks["lock_exists"] = lock_path.is_file()
    if lock_path.is_file():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        checks["train_sha_valid"] = lock.get('splits', {}).get('train', {}).get('sha256') == TRAIN_SHA256
        checks["val_sha_valid"] = lock.get('splits', {}).get('val', {}).get('sha256') == VAL_SHA256
        checks["initial_checkpoint_sha_valid"] = lock.get("initial_checkpoint_sha256") == INIT_SHA256

        # Check trajectory
        order = lock.get("epoch_order")
        checks["order_length_1024"] = len(order) == 1024 if order else False
        checks["order_epoch0_identity"] = order[:512] == list(range(512)) if order else False
        checks["order_matches_make_order"] = order == make_order() if order else False

    # 3. Check PromptBanks for train, val, diagnostic
    prompts_dir = run_dir / "prompts"
    prompt_checks: dict[str, bool] = {}
    for proto in PROTOCOLS:
        for split in ("train", "val", "diagnostic"):
            p = prompts_dir / proto / f"{split}.jsonl"
            try:
                pairs = [json.loads(line) for line in pair_path(run_dir, split).read_text(encoding='utf-8').splitlines() if line.strip()]
                bank = PromptBank(p, proto, [r['pair_id'] for r in pairs])
                for pair in pairs:
                    rec = bank.records[pair['pair_id']]
                    for item, (role, source) in zip(rec['inputs'], request_roles(pair, proto, assets_dir=data_dir / 'geometry', run_dir=run_dir)):
                        if item['role'] != role or Path(item['path']).resolve() != Path(source).resolve() or item['sha256'] != sha256_file(source):
                            raise ValueError('PE input does not bind to locked pair')
                prompt_checks[f"{proto}_{split}"] = True
            except Exception as exc:
                prompt_checks[f"{proto}_{split}"] = False
                errors[f'prompt_{proto}_{split}'] = str(exc)
    checks["prompts_ready"] = all(prompt_checks.values()) if prompt_checks else False
    try:
        order = make_order()
        for step in (511, 512, 513):
            rows = [dict(step=u+1, pair_index=order[u], noise_seed=noise_seed(u)) for u in range(step)]
            validate_trajectory(dict(step=step, sample_order=order[:step]), rows, order)
        checks['trajectory_512_boundary'] = True
    except Exception as exc:
        checks['trajectory_512_boundary'] = False
        errors['trajectory'] = str(exc)
    try:
        from scripts.data.prepare_qwen21_prompt_study_faces import validate_face_assets
        validate_face_assets(run_dir)
        checks['face_assets_actual_files'] = True
    except Exception as exc:
        checks['face_assets_actual_files'] = False
        errors['faces'] = str(exc)
    try:
        from scripts.precompute_qwen21_prompt_study_cache import validate_cache_manifest
        for proto in PROTOCOLS:
            validate_cache_manifest(run_dir, proto, ('train', 'val', 'diagnostic'))
        checks['cache_actual_files'] = True
    except Exception as exc:
        checks['cache_actual_files'] = False
        errors['cache'] = str(exc)
    try:
        from scripts.precompute_qwen21_pei2i_prompts import validate_qa8_review
        validate_qa8_review(run_dir)
        checks['qa8_inspection_review'] = True
    except Exception as exc:
        checks['qa8_inspection_review'] = False
        errors['qa8_review'] = str(exc)

    # 4. Final test gate status
    checks["eight_arms_completed"] = check_eight_arms_completed(run_dir)

    passed = all(
        v is True
        for k, v in checks.items()
        if k != "eight_arms_completed"  # gate checked separately
    )

    report = {
        "schema": "qwen21_study_selfcheck_v1",
        "passed": passed,
        "checks": checks,
        "errors": errors,
        'study_lock_sha256': sha256_file(lock_path) if lock_path.is_file() else None,
    }
    atomic_json(preflight_dir / "selfcheck.json", report)
    if not passed:
        raise RuntimeError(f"Selfcheck failed: {checks}")
    return report


def pair_path(run: Path, split: str) -> Path:
    return run / 'data' / {'train': 'train512_pairs.jsonl', 'val': 'val24_pairs.jsonl', 'test': 'test24_pairs.jsonl', 'diagnostic': 'diagnostic6_pairs.jsonl'}[split]


def locked_assets(run: Path) -> dict[str, Any]:
    from src.pose_control.qwen21_full.prompt_study.protocol import load_study_lock
    return load_study_lock(run / 'data/lock.json')['artifacts']


def validate_gradient_artifact(run: Path, arm: str, report: dict[str, Any]) -> None:
    from src.pose_control.qwen21_full.prompt_study.protocol import load_study_lock, validate_study_binding, sha256_file, INIT_SHA256
    from src.pose_control.qwen21_full.prompt_study.prompts import PromptBank
    from src.pose_control.qwen21_full.data_runtime import study_code_signature, study_asset_signature, study_prompt_signature
    from src.pose_control.qwen21_full.joint_training import validate_group_gradients
    try:
        lock = load_study_lock(run / 'data/lock.json', verify_assets=False)
        signature = report['joint_signature']
        proto = arm.split('_', 1)[1]
        pairs = [json.loads(line) for line in pair_path(run, 'train').read_text(encoding='utf-8').splitlines() if line.strip()]
        bank = PromptBank(run / f'prompts/{proto}/train.jsonl', proto, [r['pair_id'] for r in pairs])
        expected = dict(arm=arm, protocol=proto, study_lock_sha256=sha256_file(run / 'data/lock.json'), prompt_manifest_sha256=sha256_file(bank.path), max_updates=1024, initial_history_steps=256, instruction_sha256=sha256_file(run / 'prompts/instruction_v1.json'), model_manifest_sha256=lock['artifacts']['model']['manifest_sha256'], trainer_code_sha256=sha256_file(ROOT / 'scripts/train/train_qwen21_joint_pilot.py'), joint_code_sha256=sha256_file(ROOT / 'src/pose_control/qwen21_full/joint_training.py'))
        expected.update(study_code_signature())
        expected.update(study_asset_signature(run / 'data/face_cache', run / 'data'))
        expected.update(study_prompt_signature(bank))
        if any(signature.get(k) != v for k, v in expected.items()) or report.get('initialization_sha256') != INIT_SHA256:
            raise ValueError('Gradient signature/init differs from current locked data/text/code/assets')
        validate_study_binding(lock, arm, signature.get('lora_sha256'), signature.get('model', ''))
        if not report.get('low_noise'):
            raise ValueError('Missing low-noise gradient evidence')
        for row in report['low_noise']:
            validate_group_gradients(row['group_grad_norms'], branches=('body', 'face'))
            if row.get('face_active') is not True:
                raise ValueError('Low-noise Face gate did not activate')
        validate_group_gradients(report['high_noise']['group_grad_norms'], branches=('body',))
        if report['high_noise'].get('face_active') is not False:
            raise ValueError('High-noise Face gate did not close')
    except (ValueError, OSError, KeyError, RuntimeError, AssertionError) as exc:
        raise RuntimeError(f'Invalid gradient preflight for {arm}: {exc}') from exc


def require_preflight(run: Path, *, gradient: bool = True, resume: bool = True) -> None:
    path = run / 'preflight/selfcheck.json'
    if not path.is_file() or json.loads(path.read_text(encoding='utf-8')).get('passed') is not True:
        raise RuntimeError('Passed selfcheck is required before dispatch')
    run_selfcheck(run)
    if gradient:
        for arm in ARMS:
            path = run / f'preflight/{arm}/gradient_only.json'
            report = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
            if any(report.get(k) is not True for k in ('passed', 'frozen_transformer_unchanged', 'frozen_adapter_unchanged')):
                raise RuntimeError(f'Passed gradient preflight required: {arm}')
            validate_gradient_artifact(run, arm, report)
    if resume:
        path = run / 'preflight/resume_check/result.json'
        report = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
        if report.get('mandatory_exact_gate', {}).get('passed') is not True or report.get('private_optimizer_updates') != 10 or report.get('numerical_review_required') is True:
            raise RuntimeError('Passed private resume check required; inspect preflight/resume_check/result.json')
        from src.pose_control.qwen21_full.prompt_study.protocol import sha256_file
        current_signature = json.loads((run / 'preflight/qwen_pg/gradient_only.json').read_text(encoding='utf-8'))['joint_signature']
        if report.get('study_lock_sha256') != sha256_file(run / 'data/lock.json') or report.get('joint_signature') != current_signature:
            raise RuntimeError('Private resume check signature differs from current lock/gradient preflight')


def checkpoint_step(path: Path, run: Path, arm: str) -> int:
    import torch
    from src.pose_control.qwen21_full.prompt_study.protocol import sha256_file, load_study_lock, validate_study_binding, INIT_SHA256, SCHEMA
    from src.pose_control.qwen21_full.prompt_study.prompts import PromptBank
    from src.pose_control.qwen21_full.prompt_study.sampling import make_order, validate_trajectory
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    signature = state.get('joint_signature', {})
    lock = load_study_lock(run / 'data/lock.json', verify_assets=False)
    validate_study_binding(lock, arm, signature.get('lora_sha256'), signature.get('model', ''))
    if state.get('study_schema') != SCHEMA or state.get('joint_initialization_sha256') != INIT_SHA256 or state.get('arm') != arm or signature.get('arm') != arm or signature.get('study_lock_sha256') != sha256_file(run / 'data/lock.json') or signature.get('prompt_manifest_sha256') != sha256_file(run / f'prompts/{arm.split("_", 1)[1]}/train.jsonl') or state.get('max_updates') != 1024:
        raise RuntimeError(f'Checkpoint signature differs: {path}')
    for key, artifact in (('instruction_sha256', run / 'prompts/instruction_v1.json'), ('face_assets_sha256', run / 'data/face_assets.json')):
        if signature.get(key) != sha256_file(artifact):
            raise RuntimeError(f'Checkpoint {key} differs: {path}')
    rows = [json.loads(line) for line in (path.parent / 'metrics.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    validate_trajectory(state, rows[:state['step']], make_order())
    pairs = [json.loads(line) for line in pair_path(run, 'train').read_text(encoding='utf-8').splitlines() if line.strip()]
    proto = arm.split('_', 1)[1]
    bank = PromptBank(run / f'prompts/{proto}/train.jsonl', proto, [r['pair_id'] for r in pairs])
    for u, row in enumerate(rows[:state['step']]):
        pair_id = pairs[lock['epoch_order'][u]]['pair_id']
        if row.get('pair_id') != pair_id or row.get('prompt_sha256') != bank.records[pair_id]['compiled_prompt_sha256']:
            raise RuntimeError(f'Checkpoint committed pair/prompt differs: {path}')
    return int(state['step'])


def dispatch_eval(
    run_dir: Path,
    arm: str,
    step: int,
    split: str,
    indices: list[int] | range,
    seed: int,
    dry_run: bool = False,
) -> None:
    from src.pose_control.qwen21_full.prompt_study.protocol import validate_arm, sha256_file
    from src.pose_control.qwen21_full.prompt_study.results import validate_evaluation_artifacts
    base, protocol = validate_arm(arm)
    assets = locked_assets(run_dir)
    ckpt = run_dir / f"training/{arm}/checkpoint_step_{step:06d}.pt"
    out_dir = run_dir / f"evaluation/{arm}/step_{step:06d}/{split}/seed_{seed}"
    pair_file = run_dir / "data" / (
        f"{split}512_pairs.jsonl" if split == "train" else (
            f"{split}24_pairs.jsonl" if split in ("val", "test") else (
                f"{split}6_pairs.jsonl" if split == "diagnostic" else f"{split}_pairs.jsonl"
            )
        )
    )
    prompt_file = run_dir / f"prompts/{protocol}/{split}.jsonl"
    expected = dict(arm=arm, study_step=step, study_split=split, seed=seed, indices=list(indices), checkpoint_sha256=sha256_file(ckpt), pairs_sha256=sha256_file(pair_file), prompt_manifest_sha256=sha256_file(prompt_file), lock_sha=sha256_file(run_dir / 'data/lock.json'), model=assets['model']['path'], lora_sha256=assets['head_lora']['sha256'] if base == 'head' else None)
    if out_dir.is_dir() and any(out_dir.iterdir()):
        validate_evaluation_artifacts(out_dir, run_dir, expected=expected)
        if not (out_dir / 'scores.json').is_file():
            invoke([sys.executable, str(ROOT / 'scripts/eval/eval_qwen21_joint_pilot.py'), '--score-only', '--output', str(out_dir)])
        validate_evaluation_artifacts(out_dir, run_dir, expected=expected, require_scores=True)
        return
    indices_str = ",".join(str(i) for i in indices)
    argv = [
        sys.executable,
        str(ROOT / "scripts/eval/eval_qwen21_joint_pilot.py"),
        "--study-lock",
        str(run_dir / "data/lock.json"),
        "--model", assets['model']['path'],
        "--checkpoint",
        str(ckpt),
        "--pairs",
        str(pair_file),
        "--features",
        str(run_dir / "data/face_cache"),
        "--cache",
        str(run_dir / f"cache/{protocol}"),
        "--output",
        str(out_dir),
        "--arm",
        arm,
        "--prompt-manifest",
        str(prompt_file),
        "--study-step",
        str(step),
        "--study-split",
        split,
        "--indices",
        indices_str,
        "--seed",
        str(seed),
        "--resolution",
        "native",
        "--align-mode",
        "native",
        "--steps",
        "30",
        "--variant",
        "BFSHead" if base == "head" else "Qwen",
        "--stage",
        "final",
    ]
    if base == "head":
        argv.extend(["--lora", assets['head_lora']['path']])
    invoke(argv, dry_run=dry_run)
    validate_evaluation_artifacts(out_dir, run_dir, expected=expected)
    invoke([sys.executable, str(ROOT / 'scripts/eval/eval_qwen21_joint_pilot.py'), '--score-only', '--output', str(out_dir)])
    validate_evaluation_artifacts(out_dir, run_dir, expected=expected, require_scores=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="PE-I2I study master runner")
    parser.add_argument("--run", type=Path, required=True, help="Experiment root")
    parser.add_argument("--phase", required=True, choices=PHASES, help="Execution phase")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "diagnostic"])
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--max-updates", type=int, default=1024, choices=[1024])
    parser.add_argument("--eval-every", type=int, default=128, choices=[128])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run = Path(args.run)
    if args.dry_run:
        print(f"[DRY-RUN] Phase: {args.phase}")
        print(f"[DRY-RUN] Arms: {args.arms}")
        print(f"[DRY-RUN] Splits: {args.splits}")
        print(f"[DRY-RUN] Run dir: {run}")
        print('[DRY-RUN] Fixed budget: 462/arm, 3696 total, 2928 before test')
        print(f'[DRY-RUN] Dependencies: lock={(run / "data/lock.json").is_file()}, selfcheck={(run / "preflight/selfcheck.json").is_file()}, resume={(run / "preflight/resume_check/result.json").is_file()}')
        print('[DRY-RUN] No files or processes created')
        preview = [sys.executable, str(Path(__file__).resolve()), '--run', str(run), '--phase', args.phase, '--arms', *args.arms, '--splits', *args.splits, '--max-updates', '1024', '--eval-every', '128']
        if args.resume:
            preview.append('--resume')
        print('[DRY-RUN] Dispatcher argv:', json.dumps(preview))
        print('[DRY-RUN] Child argv resolves model/checkpoint paths from the verified lock at execution')
        return

    from src.pose_control.qwen21_full.prompt_study.protocol import validate_arm, FULL_VAL_STEPS, TRAIN_MONITOR, VAL_MONITOR
    for arm in args.arms:
        validate_arm(arm)
    if args.phase in ('initial', 'train'):
        require_preflight(run)
    elif args.phase == 'preflight':
        require_preflight(run, gradient=False, resume=False)

    if args.phase == "selfcheck":
        res = run_selfcheck(run)
        print("Selfcheck passed:", res["passed"])
        return

    if args.phase == "report":
        from scripts.reports.build_qwen21_pei2i_study_report import build_study_report
        rep = build_study_report(run)
        print("Report built successfully at:", rep)
        return

    if args.phase == "export":
        from src.pose_control.qwen21_full.prompt_study.results import package_export
        exp = package_export(run)
        print(f"Export packaged: {exp['files_count']} files ({exp['total_bytes']} bytes) -> {exp['tar_path']}")
        return

    if args.phase == "lock":
        lock_script = ROOT / "scripts/data/prepare_qwen21_prompt_study.py"
        src_pilot = os.environ.get("STUDY_SOURCE_PILOT", "/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/data")
        test_cands = os.environ.get("STUDY_TEST_CANDIDATES", "/data/shangguanrz/pic-edit-datasets/TikTokDataset/TikTok_v3/manifests/pairs_strict_v1/test_pairs.jsonl")
        diags = os.environ.get("STUDY_DIAGNOSTICS", '/home/shangguanrz/nihaochen/qwen21_full/native_value_audit_20261008/data/pairs.jsonl')
        argv = [
            sys.executable,
            str(lock_script),
            "--source-pilot",
            src_pilot,
            "--test-candidates",
            test_cands,
            "--diagnostics",
            diags,
            "--output",
            str(run / "data"),
        ]
        invoke(argv, dry_run=args.dry_run)
        return

    if args.phase == "rewrite-qa":
        locked_assets(run)
        pe_script = ROOT / "scripts/precompute_qwen21_pei2i_prompts.py"
        argv = [
            sys.executable,
            str(pe_script),
            "--run",
            str(run),
            "--qa8",
        ]
        invoke(argv, dry_run=args.dry_run)
        return

    if args.phase in ("rewrite", "cache") and "test" in args.splits:
        if not check_eight_arms_completed(run):
            raise RuntimeError("Cannot process test prompts/cache until all 8 arms have completed 1024 updates")
    if args.phase in ('rewrite', 'rewrite-qa', 'faces', 'cache'):
        locked_assets(run)

    if args.phase == "rewrite":
        pe_script = ROOT / "scripts/precompute_qwen21_pei2i_prompts.py"
        argv = [
            sys.executable,
            str(pe_script),
            "--run",
            str(run),
            "--splits",
            *args.splits,
            "--protocols",
            "p0",
            "pg",
            "pb",
            "pbg",
        ]
        invoke(argv, dry_run=args.dry_run)
        return

    if args.phase == "faces":
        faces_script = ROOT / "scripts/data/prepare_qwen21_prompt_study_faces.py"
        argv = [
            sys.executable,
            str(faces_script),
            "--run",
            str(run),
            "--splits",
            'train', 'val', 'diagnostic', 'test',
        ]
        invoke(argv, dry_run=args.dry_run)
        return

    if args.phase == "cache":
        cache_script = ROOT / "scripts/precompute_qwen21_prompt_study_cache.py"
        argv = [
            sys.executable,
            str(cache_script),
            "--run",
            str(run),
            '--model', locked_assets(run)['model']['path'],
            "--splits",
            *args.splits,
            "--protocols",
            "p0",
            "pg",
            "pb",
            "pbg",
        ]
        invoke(argv, dry_run=args.dry_run)
        return

    if args.phase == "preflight":
        assets = locked_assets(run)
        train_script = ROOT / "scripts/train/train_qwen21_joint_pilot.py"
        for arm in args.arms:
            base, protocol = validate_arm(arm)
            preflight_out = run / f"preflight/{arm}"
            preflight_out.mkdir(parents=True, exist_ok=True)
            argv = [
                sys.executable,
                str(train_script),
                "--study-lock",
                str(run / "data/lock.json"),
                '--model', assets['model']['path'],
                "--arm",
                arm,
                "--train-pairs",
                str(run / "data/train512_pairs.jsonl"),
                "--val-pairs",
                str(run / "data/val24_pairs.jsonl"),
                "--prompt-manifest",
                str(run / f"prompts/{protocol}/train.jsonl"),
                "--features",
                str(run / "data/face_cache"),
                "--face-supervision",
                str(run / "data/face_supervision_native.json"),
                "--cache",
                str(run / f"cache/{protocol}"),
                "--output",
                str(preflight_out),
                "--steps",
                "1024",
                "--resolution",
                "native",
                "--align-mode",
                "native",
                "--noise-offset",
                "256",
                "--gradient-only",
                "--init-checkpoint",
                assets['initial_checkpoint']['path'],
            ]
            if base == "head":
                argv.extend(["--lora", assets['head_lora']['path']])
            invoke(argv, dry_run=args.dry_run)
        require_preflight(run, resume=False)
        invoke([sys.executable, str(ROOT / 'scripts/run/check_qwen21_prompt_study_resume.py'), '--run', str(run)])
        return

    if args.phase == "initial":
        assets = locked_assets(run)
        train_script = ROOT / "scripts/train/train_qwen21_joint_pilot.py"
        for arm in args.arms:
            base, protocol = validate_arm(arm)
            arm_train_out = run / f"training/{arm}"
            arm_train_out.mkdir(parents=True, exist_ok=True)
            argv_init = [
                sys.executable,
                str(train_script),
                "--study-lock",
                str(run / "data/lock.json"),
                '--model', assets['model']['path'],
                "--arm",
                arm,
                "--train-pairs",
                str(run / "data/train512_pairs.jsonl"),
                "--val-pairs",
                str(run / "data/val24_pairs.jsonl"),
                "--prompt-manifest",
                str(run / f"prompts/{protocol}/train.jsonl"),
                "--features",
                str(run / "data/face_cache"),
                "--face-supervision",
                str(run / "data/face_supervision_native.json"),
                "--cache",
                str(run / f"cache/{protocol}"),
                "--output",
                str(arm_train_out),
                "--steps",
                "1024",
                "--stop-after-step",
                "0",
                "--resolution",
                "native",
                "--align-mode",
                "native",
                "--noise-offset",
                "256",
                "--gradient-report",
                str(run / f"preflight/{arm}/gradient_only.json"),
                "--init-checkpoint",
                assets['initial_checkpoint']['path'],
            ]
            if base == "head":
                argv_init.extend(["--lora", assets['head_lora']['path']])
            initial_checkpoint = arm_train_out / 'checkpoint_step_000000.pt'
            if initial_checkpoint.is_file():
                if checkpoint_step(initial_checkpoint, run, arm) != 0:
                    raise RuntimeError('Invalid initial checkpoint')
            elif any(arm_train_out.iterdir()):
                raise RuntimeError(f'Incomplete initial output preserved: {arm_train_out}')
            else:
                invoke(argv_init, dry_run=args.dry_run)

            # Step 0 evaluations: train monitor, val, diagnostic
            dispatch_eval(run, arm, 0, "train", list(TRAIN_MONITOR), 42, dry_run=args.dry_run)
            for s in (42, 43):
                dispatch_eval(run, arm, 0, "val", list(range(24)), s, dry_run=args.dry_run)
            dispatch_eval(run, arm, 0, "diagnostic", list(range(6)), 42, dry_run=args.dry_run)
        return

    if args.phase == "train":
        if not args.resume:
            raise RuntimeError('Train requires --resume from validated initial checkpoint')
        assets = locked_assets(run)
        train_script = ROOT / "scripts/train/train_qwen21_joint_pilot.py"
        for arm in args.arms:
            base, protocol = validate_arm(arm)
            arm_train_out = run / f"training/{arm}"
            arm_train_out.mkdir(parents=True, exist_ok=True)

            for target_step in range(args.eval_every, args.max_updates + 1, args.eval_every):
                ckpt = arm_train_out / f"checkpoint_step_{target_step:06d}.pt"
                prev_ckpt = arm_train_out / f"checkpoint_step_{target_step - args.eval_every:06d}.pt"
                last_ckpt = arm_train_out / "checkpoint_last.pt"

                if not ckpt.is_file():
                    argv_train = [
                        sys.executable,
                        str(train_script),
                        "--study-lock",
                        str(run / "data/lock.json"),
                        '--model', assets['model']['path'],
                        "--arm",
                        arm,
                        "--train-pairs",
                        str(run / "data/train512_pairs.jsonl"),
                        "--val-pairs",
                        str(run / "data/val24_pairs.jsonl"),
                        "--prompt-manifest",
                        str(run / f"prompts/{protocol}/train.jsonl"),
                        "--features",
                        str(run / "data/face_cache"),
                        "--face-supervision",
                        str(run / "data/face_supervision_native.json"),
                        "--cache",
                        str(run / f"cache/{protocol}"),
                        "--output",
                        str(arm_train_out),
                        "--steps",
                        str(args.max_updates),
                        "--stop-after-step",
                        str(target_step),
                        "--resolution",
                        "native",
                        "--align-mode",
                        "native",
                        "--noise-offset",
                        "256",
                        "--gradient-report",
                        str(run / f"preflight/{arm}/gradient_only.json"),
                    ]
                    candidates = list(arm_train_out.glob('checkpoint_step_*.pt'))
                    if last_ckpt.is_file():
                        candidates.append(last_ckpt)
                    validated = [(checkpoint_step(p, run, arm), p) for p in candidates]
                    if any(step >= target_step for step, _ in validated):
                        raise RuntimeError(f'Missing scheduled checkpoint {target_step} for {arm}; later committed state exists. Preserve artifacts and repair the missing boundary before resuming.')
                    previous = max((v for v in validated if v[0] < target_step), key=lambda v: v[0], default=None)
                    if previous is None:
                        raise RuntimeError(f'Missing initial checkpoint for {arm}')
                    argv_train.extend(['--resume', str(previous[1])])

                    if base == "head":
                        argv_train.extend(["--lora", assets['head_lora']['path']])

                    invoke(argv_train, dry_run=args.dry_run)
                elif checkpoint_step(ckpt, run, arm) != target_step:
                    raise RuntimeError('Checkpoint step differs')

                # Step evaluations after each chunk
                dispatch_eval(run, arm, target_step, "train", list(TRAIN_MONITOR), 42, dry_run=args.dry_run)
                if target_step in FULL_VAL_STEPS:
                    for s in (42, 43):
                        dispatch_eval(run, arm, target_step, "val", list(range(24)), s, dry_run=args.dry_run)
                else:
                    dispatch_eval(run, arm, target_step, "val", list(VAL_MONITOR), 42, dry_run=args.dry_run)

                if target_step in (512, 1024):
                    dispatch_eval(run, arm, target_step, "diagnostic", list(range(6)), 42, dry_run=args.dry_run)
                invoke([sys.executable, str(ROOT / 'scripts/reports/build_qwen21_pei2i_study_report.py'), '--run', str(run)])
        return

    if args.phase == "finaltest":
        if not check_eight_arms_completed(run):
            raise RuntimeError("Final test requires all 8 arms to complete 1024 steps with completed=True")
        print("Final test gate passed. Proceeding with test24 evaluation.")
        require_preflight(run)

        for arm in args.arms:
            for step in (0, 1024):
                for s in (42, 43):
                    dispatch_eval(run, arm, step, "test", list(range(24)), s, dry_run=args.dry_run)
        return


if __name__ == "__main__":
    main()
