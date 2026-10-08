#!/usr/bin/env python3
"""Run a bounded pilot phase. Inspect saved images before approving a mid-stage continuation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
RUN = Path('/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008')
OLD = Path('/home/shangguanrz/nihaochen/qwen21_full/tiktok_face_v1_20261004')
INITIAL = OLD / 'training/checkpoint_step_003000.pt'
LORAS = {'Qwen': None, 'BFSBody': Path('/home/shangguanrz/models/bfs/bfs_body_swap_v1.0_qwen_2.1.safetensors'),
         'BFSHead': Path('/home/shangguanrz/models/bfs/bfs_head_v1.1_qwen_2.1.safetensors')}


def invoke(script, values):
    command = [sys.executable, str(ROOT / script), *map(str, values)]
    print(json.dumps({'command': command}), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def train(run, arm, steps, *, gradient=False, resume=False, private=None):
    common = arm == 'J'
    base = LORAS.get(arm)
    directory = run / ('preflight' if gradient else 'training') / arm
    if private:
        directory = run / 'preflight/resume_check' / private
    values = ['--train-pairs', run / 'data' / ('common256_pairs.jsonl' if common else 'branch256_pairs.jsonl'),
              '--val-pairs', run / 'data/val24_pairs.jsonl', '--features', OLD / 'data/face_cache',
              '--face-supervision', OLD / 'data/face_supervision.json', '--cache', OLD / 'native_cache_tiktok',
              '--output', directory, '--steps', steps, '--noise-offset', 0 if common else 256]
    if base:
        values += ['--lora', base]
    if gradient:
        values += ['--gradient-only']
    else:
        values += ['--gradient-report', run / 'preflight' / arm / 'gradient_only.json']
    if private:
        values += ['--save-every', 2]
    values += ['--resume', directory / 'checkpoint_last.pt'] if resume else [
        '--init-checkpoint', INITIAL if common else run / 'training/J/checkpoint_step_000256.pt']
    invoke('scripts/train/train_qwen21_joint_pilot.py', values)


def evaluate(run, arm, checkpoint, stage, *, monitor=False, strengths=None):
    lock = json.loads((run / 'data/lock.json').read_text())
    base = LORAS.get(arm)
    for dataset in ['tiktok', 'iper']:
        output = run / 'evaluation' / f'{arm}_{stage}_{dataset}'
        values = ['--checkpoint', checkpoint, '--pairs', run / 'data' / ('val24_pairs.jsonl' if dataset == 'tiktok' else 'iper12_pairs.jsonl'),
                  '--dataset-type', 'tiktok_v3' if dataset == 'tiktok' else 'iper',
                  '--align-mode', 'letterbox' if dataset == 'tiktok' else 'resize',
                  '--features', OLD / 'data/face_cache',
                  '--cache', OLD / f'native_cache_{dataset}', '--output', output,
                  '--label', f'{arm}_{stage}_{dataset}', '--variant', arm, '--stage', stage,
                  '--resolution', 512, '--seed', 42, '--steps', 30]
        if base:
            values += ['--lora', base]
        if monitor:
            values += ['--indices', ','.join(map(str, lock[f'monitor_{dataset}_indices']))]
        if strengths is not None:
            if dataset == 'iper':
                continue
            values += ['--indices', '0,7,14,22', '--geometry-strength', strengths[0], '--face-strength', strengths[1]]
        invoke('scripts/eval/eval_qwen21_joint_pilot.py', values)
        invoke('scripts/eval/eval_qwen21_joint_pilot.py', ['--score-only', '--output', output])
    invoke('scripts/reports/build_qwen21_joint_pilot_report.py', ['--run', run])


_CONTINUATIONS = ('resume', 'repeat1', 'repeat2')
_RNG_KEYS = ('torch_rng_state', 'cuda_rng_state', 'python_rng_state', 'numpy_rng_state')
_CHECKPOINT_METADATA = ('architecture', 'model_revision', 'branch', 'resolution', 'step',
                        'sample_order', 'joint_signature', 'joint_initialization_sha256')
_TRAJECTORY_KEYS = ('step', 'pair_index', 'pair_id', 'noise_seed', 'sigma', 'face_active',
                    'auxiliary_requested', 'auxiliary_used', 'auxiliary_eligible_faces',
                    'source_stem', 'target_stem')
_LOSS_KEYS = ('loss', 'body_flow', 'face_flow', 'identity', 'perceptual')
_MANIFEST_KEYS = ('joint_signature', 'controls', 'loss', 'optimizer_parameters', 'parameter_dtype')
_DRIFT_KEYS = ('max_abs', 'mean_abs', 'relative_l2')


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json_write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def _exact_equal(left, right):
    import numpy as np
    import torch
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_exact_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_exact_equal(a, b) for a, b in zip(left, right))
    return left == right


def _metrics(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fork_resume_checkpoint(run):
    """Fork the exact direct step2 bytes/log prefix without redoing updates 1/2."""
    import torch
    folder = run / 'preflight/resume_check'
    source = folder / 'direct/checkpoint_step_000002.pt'
    state = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    rows = _metrics(folder / 'direct/metrics.jsonl')
    if state['step'] != 2 or state['sample_order'] != [0, 1] or len(rows) != 4:
        raise ValueError('Direct trajectory must save step2 and complete four updates before forking')
    if [row['step'] for row in rows] != [1, 2, 3, 4] or [row['pair_index'] for row in rows] != [0, 1, 2, 3]:
        raise ValueError('Direct metric sequence does not match the checkpoint trajectory')
    destinations = [folder / name for name in _CONTINUATIONS]
    if any(path.exists() for path in destinations):
        raise FileExistsError('Resume-check continuations already exist; archive the previous check first')
    lines = (folder / 'direct/metrics.jsonl').read_bytes().splitlines(keepends=True)
    if len(lines) != 4:
        raise ValueError('Direct metrics must contain exactly four nonblank JSONL lines')
    prefix = b''.join(lines[:2])
    source_sha = _sha256(source)
    for destination in destinations:
        destination.mkdir()
        methods = {}
        for name in ('checkpoint_step_000002.pt', 'checkpoint_last.pt'):
            target = destination / name
            try:
                os.link(source, target)
                methods[name] = 'hardlink'
            except OSError:
                shutil.copy2(source, target)
                methods[name] = 'copy'
            if _sha256(target) != source_sha:
                raise RuntimeError(f'Forked checkpoint bytes differ: {target}')
        (destination / 'metrics.jsonl').write_bytes(prefix)
        shutil.copy2(folder / 'direct/run_manifest.json', destination / 'run_manifest.json')
        _json_write(destination / 'fork_manifest.json', {
            'checkpoint_step': 2, 'source_checkpoint': str(source.resolve()),
            'checkpoint_sha256': source_sha, 'metrics_prefix_sha256': hashlib.sha256(prefix).hexdigest(),
            'joint_signature': state['joint_signature'], 'fork_methods': methods,
        })


def _state_drift(left, right, label, errors):
    """Measure all floating tensors; structural/scalar mismatches stay hard errors."""
    import torch
    total = dict(elements=0, tensors=0, absolute=0., square=0., left_square=0., right_square=0., maximum=0.)
    per_tensor = {}

    def visit(a, b, path):
        if isinstance(a, torch.Tensor):
            if not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
                errors.append(f'{label}.{path}: tensor shape/dtype differs')
                return
            # Adam stores its discrete update counter in a floating tensor.
            if not a.is_floating_point() or (label.endswith('.optimizer') and path.endswith('.step')):
                if not _exact_equal(a, b):
                    errors.append(f'{label}.{path}: discrete tensor differs')
                return
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                errors.append(f'{label}.{path}: nonfinite tensor')
                return
            x, y = a.double(), b.double()
            difference = (x - y).abs()
            count = a.numel()
            absolute, square = float(difference.sum()), float(difference.square().sum())
            left_square, right_square = float(x.square().sum()), float(y.square().sum())
            maximum = float(difference.max()) if count else 0.
            denominator = math.sqrt(max(left_square, right_square))
            per_tensor[path] = {'max_abs': maximum, 'mean_abs': absolute / count if count else 0.,
                                'relative_l2': math.sqrt(square) / denominator if denominator else 0.}
            total['elements'] += count
            total['tensors'] += 1
            for key, value in [('absolute', absolute), ('square', square),
                               ('left_square', left_square), ('right_square', right_square)]:
                total[key] += value
            total['maximum'] = max(total['maximum'], maximum)
        elif isinstance(a, dict):
            if not isinstance(b, dict) or a.keys() != b.keys():
                errors.append(f'{label}.{path}: mapping keys differ')
                return
            for key in a:
                visit(a[key], b[key], f'{path}.{key}' if path else str(key))
        elif isinstance(a, (tuple, list)):
            if type(a) is not type(b) or len(a) != len(b):
                errors.append(f'{label}.{path}: sequence differs')
                return
            for index, (x, y) in enumerate(zip(a, b)):
                visit(x, y, f'{path}[{index}]')
        elif not _exact_equal(a, b):
            errors.append(f'{label}.{path}: scalar metadata differs')

    visit(left, right, '')
    denominator = math.sqrt(max(total['left_square'], total['right_square']))
    return {'max_abs': total['maximum'],
            'mean_abs': total['absolute'] / total['elements'] if total['elements'] else 0.,
            'relative_l2': math.sqrt(total['square']) / denominator if denominator else 0.,
            'tensor_count': total['tensors'], 'element_count': total['elements'], 'per_tensor': per_tensor,
            'exact': _exact_equal(left, right)}


def _loss_drift(left, right, label, errors):
    result = {}
    for key in _LOSS_KEYS:
        pairs = []
        for a, b in zip(left[2:], right[2:]):
            x, y = a.get(key), b.get(key)
            if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) for value in (x, y)):
                errors.append(f'{label}.{key}: missing or nonfinite continuation loss')
                continue
            pairs.append((float(x), float(y)))
        differences = [abs(x - y) for x, y in pairs]
        denominator = math.sqrt(max(sum(x*x for x, _ in pairs), sum(y*y for _, y in pairs)))
        result[key] = {'max_abs': max(differences, default=0.),
                       'mean_abs': sum(differences) / len(differences) if differences else 0.,
                       'relative_l2': math.sqrt(sum(value*value for value in differences)) / denominator if denominator else 0.,
                       'steps': [{'step': index + 3, 'left': x, 'right': y, 'abs_delta': abs(x-y)}
                                 for index, (x, y) in enumerate(pairs)]}
    return result


def compare_resume(run):
    """Hard-check restoration, and describe floating drift using independent repeats.

    The observed two-repeat envelope is descriptive, not a confidence bound or
    proof of reproducibility. Outside-envelope drift requests parent assessment
    separately from the exact-restore/trajectory/finite-output hard gate.
    """
    import torch
    torch.set_num_threads(4)
    folder = run / 'preflight/resume_check'
    names = ('direct', *_CONTINUATIONS)
    errors, states, records, manifests = [], {}, {}, {}
    for name in names:
        try:
            states[name] = torch.load(folder / name / 'checkpoint_last.pt', map_location='cpu', weights_only=False, mmap=True)
            records[name] = _metrics(folder / name / 'metrics.jsonl')
            manifests[name] = json.loads((folder / name / 'run_manifest.json').read_text())
            summary = json.loads((folder / name / 'summary.json').read_text())
            if summary.get('completed') is not True or summary.get('steps') != 4 or any(
                    summary.get(key) is not True for key in ('frozen_transformer_unchanged', 'frozen_adapter_unchanged')):
                errors.append(f'{name}: completion/frozen summary failed')
            if len(records[name]) != 4 or [row.get('step') for row in records[name]] != [1, 2, 3, 4]:
                errors.append(f'{name}: expected exactly four ordered metric rows')
            if states[name].get('step') != 4 or states[name].get('sample_order') != [0, 1, 2, 3]:
                errors.append(f'{name}: final checkpoint step/sample_order differs')
            for row in records[name]:
                for key in _LOSS_KEYS:
                    value = row.get(key)
                    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                        errors.append(f'{name}.step{row.get("step")}.{key}: missing/nonfinite loss')
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            errors.append(f'{name}: missing/invalid artifacts: {error}')

    source_sha = None
    try:
        source = folder / 'direct/checkpoint_step_000002.pt'
        source_sha = _sha256(source)
        source_state = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
        if source_state.get('step') != 2 or source_state.get('sample_order') != [0, 1]:
            errors.append('direct: shared checkpoint is not the exact step2 trajectory')
        for name in _CONTINUATIONS:
            proof = json.loads((folder / name / 'resume_state_step_000002.json').read_text())
            fork = json.loads((folder / name / 'fork_manifest.json').read_text())
            signature = states['direct']['joint_signature']
            if any(proof.get(key) is not True for key in ('adapter_restored_exact', 'optimizer_restored_exact', 'rng_restored_exact')):
                errors.append(f'{name}: exact Adapter/optimizer/RNG restoration proof failed')
            for artifact in (proof, fork):
                if artifact.get('checkpoint_step') != 2 or artifact.get('checkpoint_sha256') != source_sha or not _exact_equal(artifact.get('joint_signature'), signature):
                    errors.append(f'{name}: source checkpoint/signature proof differs')
            if _sha256(folder / name / 'checkpoint_step_000002.pt') != source_sha:
                errors.append(f'{name}: saved shared step2 checkpoint bytes differ')
            resumed_manifest = json.loads((folder / name / 'resume_manifest.json').read_text())
            for key in _MANIFEST_KEYS:
                if key not in manifests['direct'] or not _exact_equal(manifests['direct'][key], resumed_manifest.get(key)):
                    errors.append(f'{name}: resumed manifest {key} differs')
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        errors.append(f'shared step2/proofs: missing/invalid artifacts: {error}')

    if 'direct' in states and 'direct' in records:
        for name in _CONTINUATIONS:
            if name not in states or name not in records:
                continue
            for key in (*_CHECKPOINT_METADATA, *_RNG_KEYS):
                if key not in states['direct'] or not _exact_equal(states['direct'][key], states[name].get(key)):
                    errors.append(f'{name}: final {key} differs')
            if not _exact_equal(records['direct'][:2], records[name][:2]):
                errors.append(f'{name}: copied first-two-row prefix differs')
            for index, (left, right) in enumerate(zip(records['direct'], records[name])):
                for key in _TRAJECTORY_KEYS:
                    if key not in left or not _exact_equal(left[key], right.get(key)):
                        errors.append(f'{name}.step{index + 1}: trajectory {key} differs')
            for key in _MANIFEST_KEYS:
                if key not in manifests.get(name, {}) or not _exact_equal(manifests['direct'].get(key), manifests[name][key]):
                    errors.append(f'{name}: forked manifest {key} differs')

    comparisons = {}
    for left_name, right_name in [('direct', 'resume'), ('direct', 'repeat1'), ('direct', 'repeat2'), ('repeat1', 'repeat2')]:
        if any(name not in states or name not in records for name in (left_name, right_name)):
            continue
        label = f'{left_name}_vs_{right_name}'
        try:
            comparisons[label] = {
                'adapter': _state_drift(states[left_name]['adapter_state_dict'], states[right_name]['adapter_state_dict'], label + '.adapter', errors),
                'optimizer': _state_drift(states[left_name]['optimizer_state_dict'], states[right_name]['optimizer_state_dict'], label + '.optimizer', errors),
                'losses': _loss_drift(records[left_name], records[right_name], label + '.losses', errors),
            }
        except (KeyError, RuntimeError) as error:
            errors.append(f'{label}: missing/invalid numerical state: {error}')

    controls = ('direct_vs_repeat1', 'direct_vs_repeat2', 'repeat1_vs_repeat2')
    envelope, outside = {}, []
    if all(name in comparisons for name in (*controls, 'direct_vs_resume')):
        for category in ('adapter', 'optimizer'):
            envelope[category] = {key: max(comparisons[name][category][key] for name in controls) for key in _DRIFT_KEYS}
        envelope['losses'] = {loss: {key: max(comparisons[name]['losses'][loss][key] for name in controls)
                                    for key in _DRIFT_KEYS} for loss in _LOSS_KEYS}
        for category in ('adapter', 'optimizer', 'losses'):
            groups = [(category, comparisons['direct_vs_resume'][category], envelope[category])] if category != 'losses' else [
                (f'losses.{loss}', comparisons['direct_vs_resume']['losses'][loss], envelope['losses'][loss]) for loss in _LOSS_KEYS]
            for path, measurement, bounds in groups:
                for key in _DRIFT_KEYS:
                    # Exactly one double-precision ULP handles arithmetic rounding;
                    # no fixed absolute/relative allclose threshold is introduced.
                    if measurement[key] > math.nextafter(bounds[key], math.inf):
                        outside.append({'path': f'{path}.{key}', 'measured': measurement[key], 'observed_max': bounds[key]})
    exact = bool(comparisons) and len(comparisons) == 4 and all(
        row['adapter']['exact'] and row['optimizer']['exact'] and all(stats['max_abs'] == 0 for stats in row['losses'].values())
        for row in comparisons.values())
    numerical_review = bool(outside)
    report = {
        'passed': not errors, 'review_required': bool(errors) or numerical_review,
        'numerical_review_required': numerical_review, 'updates_per_trajectory': 4,
        'private_optimizer_updates': len(records.get('direct', [])) + sum(max(0, len(records.get(name, [])) - 2) for name in _CONTINUATIONS),
        'scheduled_private_optimizer_updates': 10, 'private_update_budget': {'direct': 4, 'resume': 2, 'repeat1': 2, 'repeat2': 2},
        'shared_step2_checkpoint_sha256': source_sha,
        'mandatory_exact_gate': {'passed': not errors, 'errors': errors, 'requires': [
            'exact shared step2/fork prefix', 'exact Adapter/optimizer/RNG restoration before update',
            'exact checkpoint metadata and discrete sample/noise/sigma trajectory', 'exact final RNG',
            'matching J controls/signature and frozen summaries', 'finite tensor/loss outputs']},
        'comparisons': comparisons, 'observed_envelope': envelope,
        'exact_parameters_optimizer_and_losses': exact,
        'numerical_diagnostics': {'within_observed_envelope': not outside, 'outside_envelope': outside,
            'reference_comparisons': list(controls), 'comparison_slack': 'one float64 ULP of each observed bound',
            'relative_l2_definition': 'norm(left-right) / max(norm(left), norm(right)); zero if both are zero',
            'loss_steps': [3, 4], 'interpretation': 'Descriptive envelope from two independent continuation repeats; not a confidence bound. '
                'Outside-envelope values require parent assessment before formal training, but do not fail the exact-restore hard gate. '
                'CUDA warn-only determinism remains partial (Flash/grid-sample/upsample/pooling reductions may vary).'},
    }
    _json_write(folder / 'result.json', report)
    return report


def run_resume_check(run):
    train(run, 'J', 4, private='direct')
    fork_resume_checkpoint(run)
    for name in _CONTINUATIONS:
        train(run, 'J', 4, private=name, resume=True)
    return compare_resume(run)


def require_mid_review(run, arm):
    path = run / 'quality_reviews' / f'{arm}_mid.json'
    review = json.loads(path.read_text())
    if review.get('reviewed_cases') != 12 or not 0 <= review.get('new_severe_failures', 12) < 3 or review.get('continue') is not True:
        raise ValueError(f'Image quality gate did not pass: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--phase', required=True, choices=['initial', 'J-preflight', 'resume-check', 'J-mid', 'J-final',
        'diagnostics', *[f'{arm}-{stage}' for arm in LORAS for stage in ['preflight', 'hotload', 'mid', 'final']]])
    args = parser.parse_args()
    run = args.run
    hold = run / 'data/preprocessing_review_required.json'
    if hold.is_file():
        parser.error('Pilot stopped: original pixel dimensions are required. The existing runner uses fixed 512 inputs. '
                     f'Read the handoff and implement/validate native resolution before starting a new run. Hold: {hold}')
    if args.phase == 'initial':
        evaluate(run, 'init', INITIAL, 'initial')
    elif args.phase == 'resume-check':
        result = run_resume_check(run)
        print(json.dumps({'resume_check_passed': result['passed'], 'numerical_review_required': result['numerical_review_required'],
                          'private_optimizer_updates': result['private_optimizer_updates']}), flush=True)
        if not result['passed']:
            raise RuntimeError('Resume hard gate failed; inspect preflight/resume_check/result.json')
    elif args.phase == 'diagnostics':
        for arm, strength in [('JBodyOnly', (1, 0)), ('JFaceOnly', (0, 1))]:
            evaluate(run, arm, run / 'training/J/checkpoint_step_000256.pt', 'common', strengths=strength)
    else:
        arm, stage = args.phase.split('-')
        if stage == 'preflight':
            train(run, arm, 256, gradient=True)
        elif stage == 'hotload':
            evaluate(run, arm, run / 'training/J/checkpoint_step_000256.pt', 'hotload', monitor=True)
        elif stage == 'mid':
            if arm == 'J' and not json.loads((run / 'preflight/resume_check/result.json').read_text()).get('passed'):
                raise ValueError('Resume check required before formal training')
            train(run, arm, 128)
            evaluate(run, arm, run / 'training' / arm / 'checkpoint_step_000128.pt', 'mid', monitor=True)
        else:
            require_mid_review(run, arm)
            train(run, arm, 256, resume=True)
            evaluate(run, arm, run / 'training' / arm / 'checkpoint_step_000256.pt', 'common' if arm == 'J' else 'final')


if __name__ == '__main__':
    main()
