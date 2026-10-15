#!/usr/bin/env python3
"""Collect bounded-run results and verify the explicit single-case criteria."""
import csv
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from scripts.run_qwen21_face_single_sample import OUT, sha


def read(name):
    return json.loads((OUT / name).read_text())


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def csv_write(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    decision = read('training_decision.json')
    visual = read('visual_review.json')
    body = read('body_stage_metrics.json')
    face = read('face_stage_metrics.json') if decision['face_started'] else []
    reference = body[-1]['metrics']
    criteria = []
    for row in face:
        review = visual['face_steps'][str(row['added_step'])]
        metric = row['metrics']
        source_id = metric['identity_cosine_reliable']
        reference_id = reference['identity_cosine_reliable']
        checks = {
            'no_obvious_artifact': review['no_obvious_artifact'],
            'manual_face_preservation_improved': review['face_preservation_improved'],
            'reliable_identity_improved': source_id is not None and reference_id is not None and source_id > reference_id,
            'pose_drop_within_002': metric['pose_pck_002'] is not None and metric['pose_pck_002'] >= reference['pose_pck_002'] - .02,
            'background_increase_within_1': metric['background_mae_source'] <= reference['background_mae_source'] + 1.,
        }
        criteria.append({'added_face_steps': row['added_step'], 'checks': checks,
                         'passed': row['added_step'] > 0 and all(checks.values()),
                         'metrics': metric, 'manual_review': review})
    if face:
        left = np.asarray(Image.open(body[-1]['image']))
        right = np.asarray(Image.open(face[0]['image']))
        zero = {'pixels_equal': bool(np.array_equal(left, right)),
                'max_pixel_abs': int(np.abs(left.astype('int16')-right.astype('int16')).max())}
        if not zero['pixels_equal']:
            raise AssertionError('Face step 0 must reproduce frozen Body pixels')
    else:
        zero = None
    passed = [r for r in criteria if r['passed']]
    result = {'body_gate_passed': decision['body_gate_passed'],
              'face_single_case_success': bool(passed), 'passing_face_steps': [r['added_face_steps'] for r in passed],
              'face_initial_vs_frozen_body': zero, 'criteria': criteria,
              '1024_inference_required': bool(passed),
              'final_test_set_used': False, 'default_checkpoint_replaced': False,
              'conclusion_scope': 'one explicitly selected training pair; not generalization'}
    write('final_acceptance.json', result)

    metrics = []
    for group, rows in (('diagnosis', read('diagnosis_metrics.json')), ('body_training', body), ('face_training', face)):
        for row in rows:
            metrics.append({'group': group, 'case': row.get('case', ''),
                            'added_training_steps': row.get('added_step', ''),
                            'image': row['image'], 'sampling_steps': 30, 'seed': 42,
                            **row['metrics'], 'identity_effective_n': int(row['metrics']['identity_cosine_reliable'] is not None),
                            'seconds': row['seconds'], 'peak_cuda_gib': row['peak_cuda_gib'],
                            'inference_concurrent_with_training': row.get('concurrent_training', False)})
    csv_write(OUT / 'metrics.csv', metrics)
    gradients = []
    resources = {}
    for branch in ('body', 'face'):
        directory = OUT / f'{branch}_train'
        if not (directory / 'summary.json').exists():
            continue
        summary = json.loads((directory / 'summary.json').read_text())
        history = [json.loads(line) for line in (directory / 'train_metrics.jsonl').read_text().splitlines()]
        if len(history) != summary['optimizer_steps'] or any(not np.isfinite(r['loss']) or r['grad_norm'] <= 0 for r in history):
            raise AssertionError(f'{branch}: invalid update count, nonfinite loss or interrupted gradient')
        resources[branch] = {key: summary[key] for key in ('optimizer_steps', 'training_pair_uses', 'seconds', 'peak_cuda_gib',
                            'nonzero_gradient_steps', 'frozen_parameters_unchanged', 'base_versions_unchanged')}
        resources[branch]['zero_schedule_sigma_resamples'] = sum(r.get('zero_schedule_sigma_resamples', 0) for r in history)
        for row in history:
            gradients.append({'branch': branch, **{key: row.get(key) for key in ('step', 'sigma', 'loss', 'flow', 'identity',
                              'perceptual', 'outside', 'grad_norm', 'elapsed_seconds', 'peak_cuda_gib')},
                              **{f'gradient_{key}': value for key, value in row['group_grad_norms'].items()}})
    csv_write(OUT / 'training_gradients.csv', gradients)
    resources['complete_inference'] = [{key: row[key] for key in ('group', 'case', 'added_training_steps', 'seconds',
                                      'peak_cuda_gib', 'inference_concurrent_with_training')} for row in metrics]
    write('resource_summary.json', resources)

    protection = read('original_protection.json')
    original = [{'path': path, 'expected_sha256': value, 'actual_sha256': sha(path)} for path, value in protection.items()]
    if any(row['actual_sha256'] != row['expected_sha256'] for row in original):
        raise AssertionError('Original checkpoint changed')
    checkpoint_files = sorted((OUT / 'body_train').glob('checkpoint_step_*.pt')) + sorted((OUT / 'face_train').glob('checkpoint_step_*.pt'))
    checkpoints = [{'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size} for path in checkpoint_files]
    code = [ROOT / path for path in (
        'train_iper_qwen21_full.py', 'eval_iper_qwen21_full.py', 'src/pose_control/qwen21_full/execution.py',
        'src/pose_control/qwen21_full/qwen_adapter.py', 'scripts/run_qwen21_face_single_sample.py',
        'scripts/verify_qwen21_face_single_sample.py', 'scripts/verify_qwen21_full_resume.py',
        'scripts/probe_qwen21_face_single_sample.py', 'scripts/write_qwen21_face_single_sample_report.py',
        'scripts/finalize_qwen21_face_single_sample.py')]
    write('delivery_manifest.json', {'original_checkpoints': original, 'milestone_checkpoints': checkpoints,
          'code': [{'path': str(path), 'sha256': sha(path)} for path in code],
          'fixed_sample_sha256': sha(OUT / 'single_pair.jsonl'), 'resource_summary': resources,
          'acceptance': result})
    print(json.dumps({'face_single_case_success': bool(passed), 'checkpoints': len(checkpoints),
                      'complete_generation_records': len(metrics)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
