#!/usr/bin/env python3
"""Run ten private qwen_pg updates; require exact restoration and report CUDA drift."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.run.run_qwen21_pei2i_study import invoke, locked_assets
from src.pose_control.qwen21_full.prompt_study.protocol import atomic_json, sha256_file

CONTINUATIONS = ('resume', 'repeat1', 'repeat2')


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def private_argv(run: Path, name: str, stop: int, *, resume: bool = False):
    assets = locked_assets(run)
    output = run / f'preflight/resume_check/{name}'
    argv = [sys.executable, str(ROOT / 'scripts/train/train_qwen21_joint_pilot.py'), '--study-lock', str(run / 'data/lock.json'), '--arm', 'qwen_pg', '--model', assets['model']['path'], '--train-pairs', str(run / 'data/train512_pairs.jsonl'), '--val-pairs', str(run / 'data/val24_pairs.jsonl'), '--prompt-manifest', str(run / 'prompts/pg/train.jsonl'), '--features', str(run / 'data/face_cache'), '--face-supervision', str(run / 'data/face_supervision_native.json'), '--cache', str(run / 'cache/pg'), '--output', str(output), '--steps', '1024', '--stop-after-step', str(stop), '--resolution', 'native', '--align-mode', 'native', '--noise-offset', '256', '--gradient-report', str(run / 'preflight/qwen_pg/gradient_only.json')]
    argv += ['--resume', str(output / 'checkpoint_step_000002.pt')] if resume else ['--init-checkpoint', assets['initial_checkpoint']['path']]
    return argv


def fork_step2(run: Path):
    import torch
    folder = run / 'preflight/resume_check'
    source = folder / 'direct/checkpoint_step_000002.pt'
    state = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    prefix = (folder / 'direct/metrics.jsonl').read_bytes()
    logged = rows(folder / 'direct/metrics.jsonl')
    if state['step'] != 2 or state['sample_order'] != [0, 1] or len(logged) != 2 or [r['step'] for r in logged] != [1, 2]:
        raise ValueError('Shared private checkpoint must commit exactly two updates')
    source_sha = sha256_file(source)
    for name in CONTINUATIONS:
        destination = folder / name
        destination.mkdir()
        for filename in ('checkpoint_step_000002.pt', 'checkpoint_last.pt'):
            shutil.copyfile(source, destination / filename)
            if sha256_file(destination / filename) != source_sha:
                raise RuntimeError('Fork checkpoint bytes differ')
        (destination / 'metrics.jsonl').write_bytes(prefix)
        shutil.copyfile(folder / 'direct/run_manifest.json', destination / 'run_manifest.json')
        atomic_json(destination / 'fork_manifest.json', {'checkpoint_step': 2, 'checkpoint_sha256': source_sha, 'metrics_prefix_sha256': hashlib.sha256(prefix).hexdigest(), 'joint_signature': state['joint_signature']})


def compare_resume(run: Path):
    import torch
    from src.pose_control.qwen21_full.joint_training import assert_exact_state
    from src.pose_control.qwen21_full.prompt_study.results import compare_private_continuations
    from scripts.run.run_qwen21_joint_pilot import _RNG_KEYS, _DRIFT_KEYS, _LOSS_KEYS
    folder = run / 'preflight/resume_check'
    errors, states, records, manifests = [], {}, {}, {}
    shared_path = folder / 'direct/checkpoint_step_000002.pt'
    shared_sha = sha256_file(shared_path)
    shared = torch.load(shared_path, map_location='cpu', weights_only=False, mmap=True)
    for name in ('direct', *CONTINUATIONS):
        try:
            states[name] = torch.load(folder / name / 'checkpoint_last.pt', map_location='cpu', weights_only=False, mmap=True)
            records[name] = rows(folder / name / 'metrics.jsonl')
            manifests[name] = json.loads((folder / name / 'run_manifest.json').read_text(encoding='utf-8'))
            summary = json.loads((folder / name / 'summary.json').read_text(encoding='utf-8'))
            if summary.get('steps') != 4 or summary.get('max_updates') != 1024 or any(summary.get(k) is not True for k in ('frozen_transformer_unchanged', 'frozen_adapter_unchanged')):
                raise ValueError('Private summary must commit four updates under total budget 1024 with frozen modules')
            if len(records[name]) != 4 or [r['step'] for r in records[name]] != [1, 2, 3, 4] or [r['pair_index'] for r in records[name]] != [0, 1, 2, 3]:
                raise ValueError('Private trajectory differs')
            for row in records[name]:
                if row['noise_seed'] != 298 + row['step'] - 1 or any(not isinstance(row.get(k), (int, float)) or not math.isfinite(row[k]) for k in _LOSS_KEYS):
                    raise ValueError('Private noise or finite losses differ')
            proof = json.loads((folder / name / 'resume_state_step_000002.json').read_text(encoding='utf-8'))
            if any(proof.get(k) is not True for k in ('adapter_restored_exact', 'optimizer_restored_exact', 'rng_restored_exact')) or proof.get('checkpoint_step') != 2 or proof.get('checkpoint_sha256') != shared_sha:
                raise ValueError('Exact Adapter/optimizer/RNG restore proof failed')
            assert_exact_state(shared['joint_signature'], proof['joint_signature'], f'{name}.restore.signature')
            if name != 'direct':
                fork = json.loads((folder / name / 'fork_manifest.json').read_text(encoding='utf-8'))
                prefix = b''.join((folder / name / 'metrics.jsonl').read_bytes().splitlines(keepends=True)[:2])
                if fork.get('checkpoint_sha256') != shared_sha or fork.get('metrics_prefix_sha256') != hashlib.sha256(prefix).hexdigest() or sha256_file(folder / name / 'checkpoint_step_000002.pt') != shared_sha:
                    raise ValueError('Shared step2 bytes/prefix differ')
            for key in (*_RNG_KEYS, 'arm'):
                assert_exact_state(states['direct'][key], states[name][key], f'{name}.{key}')
            assert_exact_state(manifests['direct'], manifests[name], f'{name}.forked_manifest')
        except (ValueError, OSError, KeyError, RuntimeError, AssertionError) as exc:
            errors.append(f'{name}: {exc}')
    comparisons = {}
    for a, b in (('direct', 'resume'), ('direct', 'repeat1'), ('direct', 'repeat2'), ('repeat1', 'repeat2')):
        try:
            comparison = compare_private_continuations(states[a], states[b], records[a], records[b])
            comparisons[f'{a}_vs_{b}'] = comparison
            errors.extend(comparison['errors'])
        except (ValueError, OSError, KeyError, RuntimeError, AssertionError) as exc:
            errors.append(f'{a}_vs_{b}: {exc}')
    controls = ('direct_vs_repeat1', 'direct_vs_repeat2', 'repeat1_vs_repeat2')
    outside, envelope = [], {}
    if all(key in comparisons for key in (*controls, 'direct_vs_resume')):
        for category in ('adapter', 'optimizer'):
            envelope[category] = {k: max(comparisons[n][category][k] for n in controls) for k in _DRIFT_KEYS}
        envelope['losses'] = {loss: {k: max(comparisons[n]['losses'][loss][k] for n in controls) for k in _DRIFT_KEYS} for loss in _LOSS_KEYS}
        for category in ('adapter', 'optimizer', 'losses'):
            groups = [(category, comparisons['direct_vs_resume'][category], envelope[category])] if category != 'losses' else [(f'losses.{loss}', comparisons['direct_vs_resume']['losses'][loss], envelope['losses'][loss]) for loss in _LOSS_KEYS]
            for label, measured, bound in groups:
                for key in _DRIFT_KEYS:
                    if measured[key] > math.nextafter(bound[key], math.inf):
                        outside.append({'path': f'{label}.{key}', 'measured': measured[key], 'observed_max': bound[key]})
    else:
        errors.append('Missing private continuation comparisons')
    updates = len(records.get('direct', [])) + sum(max(0, len(records.get(n, [])) - 2) for n in CONTINUATIONS)
    if updates != 10:
        errors.append('Private optimizer update budget differs from ten')
    report = {'schema': 'qwen21_study_resume_check_v1', 'passed': not errors, 'mandatory_exact_gate': {'passed': not errors, 'errors': errors}, 'private_optimizer_updates': updates, 'scheduled_private_optimizer_updates': 10, 'shared_step2_checkpoint_sha256': shared_sha, 'study_lock_sha256': sha256_file(run / 'data/lock.json'), 'joint_signature': shared['joint_signature'], 'numerical_review_required': bool(outside), 'numerical_diagnostics': {'comparisons': comparisons, 'observed_envelope': envelope, 'outside_envelope': outside, 'interpretation': 'Two independent repeats provide a descriptive drift range, not a confidence bound or proof of reproducibility.'}}
    atomic_json(folder / 'result.json', report)
    return report


def run_resume_check(run: Path):
    run = Path(run)
    folder = run / 'preflight/resume_check'
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError('Private resume check already has artifacts; preserve and inspect them before starting a new check')
    folder.mkdir(parents=True, exist_ok=True)
    invoke(private_argv(run, 'direct', 2))
    fork_step2(run)
    invoke(private_argv(run, 'direct', 4, resume=True))
    for name in CONTINUATIONS:
        invoke(private_argv(run, name, 4, resume=True))
    return compare_resume(run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    result = run_resume_check(args.run)
    print(json.dumps(result, ensure_ascii=False))
    if not result['mandatory_exact_gate']['passed'] or result['numerical_review_required']:
        raise RuntimeError('Private resume check needs investigation; inspect retained result.json')


if __name__ == '__main__':
    main()
