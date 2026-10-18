#!/usr/bin/env python3
"""Bounded Face stability diagnosis; no training and no final-test samples."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RUN_ROOT = Path('/home/shangguanrz/nihaochen/qwen21_full')
DEFAULT_OUTPUT = RUN_ROOT / 'face_multisample_v1/diagnostics'
PREVIOUS = RUN_ROOT / 'face_validation_20261003'
EXPECTED_SHA = {
    'body': 'aa36b338805e14cc6d6a87ffbf8f74522512e56b02011af6426b1244c360fb71',
    'roi': '3638228bd0358e9d5ef6bc438f437d7a85a235d708c67c1d3c4986c59e3704f4',
    'aligned': '31e4a6b4a678e34a0c0af55e9a7f4d10249b43b8faee5fc96294ecb55174b363',
}


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def write_pairs(path, rows):
    text = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
    if path.exists() and path.read_text() != text:
        raise ValueError(f'Fixed diagnostic pair manifest changed: {path}')
    path.write_text(text)


def build_jobs():
    jobs = []
    for variant in ('roi', 'aligned'):
        for strength in (.25, .5):
            jobs.append(dict(name=f'{variant}_strength_{strength:g}', category='strength',
                             variant=variant, pairs='validation_pairs.jsonl', mode='body_face',
                             face_strength=strength, face_group_gains=[1.] * 6, stages=[], expected_images=4))
        for group in range(6):
            gains = [1.] * 6
            gains[group] = 0.
            jobs.append(dict(name=f'{variant}_disable_group_{group}', category='group',
                             variant=variant, pairs='failure_pairs.jsonl', mode='body_face',
                             face_strength=1., face_group_gains=gains, stages=[], expected_images=2))
        jobs.append(dict(name=f'{variant}_trajectory', category='trajectory',
                         variant=variant, pairs='failure_pairs.jsonl', mode='body_face',
                         face_strength=1., face_group_gains=[1.] * 6, stages=[10, 20, 25, 30], expected_images=2))
    jobs.append(dict(name='body_trajectory', category='trajectory', variant='body',
                     pairs='failure_pairs.jsonl', mode='body', face_strength=0.,
                     face_group_gains=[1.] * 6, stages=[10, 20, 25, 30], expected_images=2))
    assert sum(job['expected_images'] for job in jobs) == 46
    return jobs


def prepare_plan(output, previous):
    output.mkdir(parents=True, exist_ok=True)
    historical = read(previous / 'manifest.json')
    fixed = previous / 'fixed_pairs.jsonl'
    if sha(fixed) != historical['fixed_pairs_sha256']:
        raise AssertionError('Original six-case manifest checksum changed')
    rows = lines(fixed)
    validation = [row for row in rows if row['validation_set'] == 'validation']
    if [row['pair_id'] for row in validation] != ['val_000', 'val_001', 'val_002', 'val_003']:
        raise AssertionError('Expected the four original validation cases in their fixed order')
    failure = [row for row in validation if row['pair_id'] in ('val_001', 'val_002')]
    checkpoints = historical['checkpoints']
    for name, expected in EXPECTED_SHA.items():
        if checkpoints[name]['sha256'] != expected or sha(checkpoints[name]['path']) != expected:
            raise AssertionError(f'Fixed diagnostic checkpoint changed: {name}')
    write_pairs(output / 'validation_pairs.jsonl', validation)
    write_pairs(output / 'failure_pairs.jsonl', failure)
    plan = dict(schema='qwen21_full_face_stability_v1', checkpoints=checkpoints,
                previous_validation=str(previous), resolution=512, steps=30, seed=42,
                true_cfg_scale=1., progress_mode='sigma', use_kv_cache=False,
                face_scatter_grid='checkpoint', added_face_reference_image=False,
                prompt=historical['prompt'], model_revision=historical['model_revision'],
                expected_new_final_images=46, expected_stage_predictions=24,
                final_50_test_used=False, training_performed=False,
                face_ablation='Apply gains to scheduled Face injection; retain all six Face computations.',
                stage_prediction='Pre-update latent minus physical scheduler sigma times same-step velocity.',
                jobs=build_jobs())
    existing = output / 'plan.json'
    if existing.exists() and read(existing) != plan:
        raise ValueError('Existing diagnostic plan differs; use another output directory')
    write_json(existing, plan)
    print(json.dumps({'phase': 'prepare', 'jobs': len(plan['jobs']), 'maximum_final_images': 46,
                      'training': False, 'plan': str(existing)}), flush=True)
    return plan


def gpu_checks(output, plan):
    """Forward-only engineering gates, including the original leakage checks."""
    import torch
    from diffusers import QwenImage21Pipeline
    from scripts.run_qwen21_weight_inheritance import source_pair, verify_gpu
    from scripts.verify_qwen21_full_runtime import parity
    from src.pose_control.qwen21_full.execution import (
        MODEL, FaceConditionStore, QwenFullDataset, Qwen21IPERCache, batch_one,
        load_adapter_checkpoint, new_adapter, prepare,
    )
    from src.pose_control.qwen21_full.qwen_adapter import Qwen21FullControlContext
    from src.pose_control.qwen21_full.qwen_runtime import dataset_image_to_pil, transformer_inputs

    torch.set_num_threads(4)
    torch.manual_seed(42)
    pipe = QwenImage21Pipeline.from_pretrained(str(MODEL), dtype=torch.bfloat16).to('cuda')
    for module in (pipe.transformer, pipe.vae, pipe.text_encoder):
        module.eval().requires_grad_(False)
    adapter = new_adapter('cuda').eval().requires_grad_(False)
    dataset = QwenFullDataset(output / 'failure_pairs.jsonl', resolution=512, dataset_type='iper')
    store = FaceConditionStore(RUN_ROOT / 'face_cache')
    cache = Qwen21IPERCache(RUN_ROOT / 'native_cache', resolution=512, prompt=plan['prompt'])
    reports = []
    with torch.no_grad():
        for variant in ('roi', 'aligned'):
            load_adapter_checkpoint(adapter, plan['checkpoints'][variant]['path'])
            base = verify_gpu(pipe, adapter, cache, dataset, store, variant,
                              face_reference_token_hw=(32, 32))
            batch = batch_one(dataset, 1)
            pair = source_pair(pipe, cache, dataset_image_to_pil(batch['src_image'][0]))
            face = prepare(adapter, batch, pair, [dataset.samples[1]], store, with_face=True)
            noise = torch.randn((1, 1024, 64), device='cuda', dtype=pair['source_latent'].dtype,
                                generator=torch.Generator(device='cuda').manual_seed(42))
            inputs = transformer_inputs(pair, noise, noise.new_tensor([.2]))

            def predict(**kwargs):
                with Qwen21FullControlContext(pipe.transformer, adapter, face, hand_strength=0,
                                             hand_mode='off', **kwargs) as context:
                    value = pipe.transformer(**inputs)[0][:, -1024:].clone()
                return value, context

            with torch.autocast('cuda', dtype=torch.bfloat16):
                body, _ = predict(face_strength=0)
                groups_off, context = predict(face_group_gains=[0.] * 6, detailed_face_diagnostics=True)
                all_off = parity(body, groups_off, .001)
                if not all_off['passed'] or len(context.injection_diagnostics) != 32:
                    raise AssertionError('All Face groups off does not reproduce Body or lacks layer diagnostics')
                saved = adapter.output_bridges['face'].weight.detach().clone()
                try:
                    adapter.output_bridges['face'].weight.zero_()
                    zero, _ = predict()
                    zero_bridge = parity(body, zero, .001)
                finally:
                    adapter.output_bridges['face'].weight.copy_(saved)
                if not zero_bridge['passed']:
                    raise AssertionError('Zero Face output interface does not reproduce Body')
            base.update(all_face_groups_off_vs_body=all_off, zero_face_interface_vs_body=zero_bridge,
                        all_layers_recorded=32, check_cases=['val_001', 'val_002'])
            reports.append(base)
    result = dict(passed=True, forward_only=True, training_performed=False, checks=reports)
    write_json(output / 'gpu_correctness.json', result)
    print(json.dumps({'phase': 'checks', 'passed': True, 'output': str(output / 'gpu_correctness.json')}), flush=True)


def collect_index(output, plan):
    index = []
    for job in plan['jobs']:
        records = output / 'generations' / job['name'] / 'per_sample.jsonl'
        if not records.exists():
            continue
        rows = lines(records)
        if len(rows) != job['expected_images']:
            raise AssertionError(f'Wrong diagnostic image count: {job["name"]}')
        for row in rows:
            if not Path(row['image']).exists() or row['control_forwards'] != 30:
                raise AssertionError('Missing final image or incomplete denoising')
            if any(count != 30 for count in row['block_calls'].values()):
                raise AssertionError('Missing one of the 32 layer injections')
            if len(row['injection_diagnostics']) != 32 * 30:
                raise AssertionError('Missing detailed actual-injection diagnostics')
            if sorted(item['sampling_step'] for item in row.get('stage_predictions', [])) != job['stages']:
                raise AssertionError('Missing requested pre-update stage predictions')
            index.append({**row, 'diagnostic_job': job['name'], 'category': job['category'],
                          'variant': job['variant'], 'face_strength': job['face_strength'],
                          'face_group_gains': job['face_group_gains']})
    write_json(output / 'generation_index.json', index)
    text = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in index)
    (output / 'per_sample.jsonl').write_text(text)
    write_json(output / 'generation_acceptance.json', dict(
        complete=len(index) == 46, completed_final_images=len(index), planned_maximum=46,
        stage_predictions=sum(len(row.get('stage_predictions', [])) for row in index),
        training_performed=False, final_50_test_used=False))
    return index


def generate(output, plan, category):
    checks = output / 'gpu_correctness.json'
    if not checks.exists() or not read(checks)['passed']:
        raise RuntimeError('Run --phase checks before diagnosis generation')
    for job in plan['jobs']:
        if category != 'all' and job['category'] != category:
            continue
        command = [sys.executable, str(ROOT / 'eval_iper_qwen21_full.py'),
                   '--pairs', str(output / job['pairs']), '--dataset-type', 'iper',
                   '--body-checkpoint', plan['checkpoints']['body']['path'],
                   '--face-checkpoint', plan['checkpoints'][job['variant'] if job['variant'] != 'body' else 'roi']['path'],
                   '--modes', job['mode'], '--resolution', '512', '--steps', '30', '--seed', '42',
                   '--prompt', plan['prompt'], '--progress-mode', 'sigma', '--no-use-kv-cache',
                   '--face-scatter-grid', 'checkpoint', '--face-strength', str(job['face_strength']),
                   '--face-group-gains', *map(str, job['face_group_gains']), '--detailed-face-diagnostics',
                   '--output', str(output / 'generations' / job['name'])]
        if job['stages']:
            command.extend(['--diagnostic-stage-steps', *map(str, job['stages'])])
        print(json.dumps({'phase': 'generate', 'job': job['name'], 'expected_images': job['expected_images']}), flush=True)
        with (output / f'{job["name"]}.log').open('a') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    index = collect_index(output, plan)
    for name, expected in EXPECTED_SHA.items():
        if sha(plan['checkpoints'][name]['path']) != expected:
            raise AssertionError('Diagnosis modified an original checkpoint')
    print(json.dumps({'phase': 'generate', 'completed_images': len(index), 'maximum': 46}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'checks', 'generate', 'index', 'all'), default='prepare')
    parser.add_argument('--category', choices=('all', 'strength', 'group', 'trajectory'), default='all')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--previous', type=Path, default=PREVIOUS)
    args = parser.parse_args()
    plan = prepare_plan(args.output, args.previous)
    if args.phase in ('checks', 'all'):
        gpu_checks(args.output, plan)
    if args.phase in ('generate', 'all'):
        generate(args.output, plan, args.category)
    if args.phase == 'index':
        collect_index(args.output, plan)


if __name__ == '__main__':
    main()
