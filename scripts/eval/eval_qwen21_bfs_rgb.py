#!/usr/bin/env python3
"""Inference-only, four-case BFS audit with locked A RGB + B RGB references."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import time

import torch
from PIL import Image

from src.data.letterbox import Letterbox
from scripts.eval.eval_qwen21_native_value import (
    RUN as PREVIOUS, append_record, json_write, read_rows, read_rgb, sha256,
)
from scripts.eval.eval_qwen21_bfs_skeleton import (
    MODEL, LORAS, EXPECTED_SHA, VARIANTS, versions, digest_text, sample,
)
from scripts.eval.qwen21_native_value_prompts import build_prompt
from src.pose_control.qwen21_full.frozen_bfs_lora import (
    load_frozen_bfs, _convert_state, _verify_loaded, frozen_transformer_digest,
)

DEFAULT = Path('/home/shangguanrz/nihaochen/qwen21_full/bfs_rgb_direct_20261010')
PAIR_IDS = ('iper_00', 'iper_01', 'tiktok_00', 'tiktok_01')
RESOLUTION = 1024
PROTOCOL = dict(resolution=1024, steps=30, seed=42, true_cfg_scale=1.,
                use_kv_cache=False, control_forwards=0, target_rgb_in_native_references=True)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def prepare_rgb(pair, directory):
    """Prepare only the two authorized RGB references; never read face/guide assets."""
    images, refs, alignment = [], [], {}
    folder = Path(directory) / 'inputs' / pair['pair_id']
    folder.mkdir(parents=True, exist_ok=True)
    for i, (role, side) in enumerate((('A', 'source'), ('B', 'target')), 1):
        original = read_rgb(pair[f'{side}_rgb'])
        transform = Letterbox.from_size(original.size, RESOLUTION)
        image = transform.image(original, fill=(128, 128, 128))
        path = folder / f'{role}.png'
        image.save(path)
        images.append(image)
        refs.append(dict(index=i, role=role, path=str(path), sha256=sha256(path)))
        alignment[side] = transform.metadata()
    return images, refs, alignment


def reference_signature(refs):
    if [(r['index'], r['role']) for r in refs] != [(1, 'A'), (2, 'B')]:
        raise ValueError('References must be exactly A RGB, then B RGB')
    for ref in refs:
        if sha256(ref['path']) != ref['sha256']:
            raise ValueError('Reference file checksum differs')
    return [(r['index'], r['role'], r['sha256']) for r in refs]


def validate_native(saved, prompt, refs, runtime):
    expected = {**PROTOCOL, 'lora_loaded': False}
    if any(saved.get(k) != v for k, v in expected.items()):
        raise ValueError('Native A+B protocol differs')
    if saved['prompt'] != prompt or saved['prompt_sha256'] != digest_text(prompt):
        raise ValueError('Native A+B detailed prompt differs')
    if saved.get('runtime_versions', runtime) != runtime:
        raise ValueError('Native runtime differs')
    if reference_signature(saved['references']) != reference_signature(refs):
        raise ValueError('Native A/B reference content differs')
    with Image.open(saved['image']) as image:
        if image.size != (RESOLUTION, RESOLUTION):
            raise ValueError('Native output has wrong size')
    if sha256(saved['image']) != saved['image_sha256']:
        raise ValueError('Native output checksum differs')


def validate_saved(saved, prompt, refs, variant, lock_sha):
    expected = {**PROTOCOL, 'variant': variant, 'method': variant, 'runtime_versions': versions(),
                'lora_loaded': variant != 'native', 'lora_strength': 0. if variant == 'native' else 1.,
                'lora_sha256': None if variant == 'native' else EXPECTED_SHA[variant],
                'lock_sha256': lock_sha}
    if any(saved.get(k) != v for k, v in expected.items()):
        raise ValueError('Resume protocol/weights/runtime differs')
    if saved['prompt'] != prompt or saved['prompt_sha256'] != digest_text(prompt):
        raise ValueError('Resume prompt differs')
    if reference_signature(saved['references']) != reference_signature(refs):
        raise ValueError('Resume A/B references differ')
    if sha256(saved['image']) != saved['image_sha256']:
        raise ValueError('Resume output checksum differs')
    with Image.open(saved['image']) as image:
        if image.size != (RESOLUTION, RESOLUTION):
            raise ValueError('Resume output dimensions differ')


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    by_id = {r['pair_id']: r for r in read_rows(PREVIOUS / 'data/pairs.jsonl')}
    pairs = [by_id[pid] for pid in PAIR_IDS]
    prompts = json.loads((PREVIOUS / 'prompts/locked_prompts.json').read_text())
    locked = {p['pair_id']: prompts[p['pair_id']]['native_ab'] for p in pairs}
    dev = read_rows(PREVIOUS / 'data/dev_pairs.jsonl')[:1]
    dev_prompts = {p['pair_id']: build_prompt(p, 'native_ab') for p in dev}
    protected = {str(p): sha256(p) for p in LORAS.values()}
    for variant, path in LORAS.items():
        if protected[str(path)] != EXPECTED_SHA[variant]:
            raise ValueError('BFS checkpoint changed')
    for p in [*pairs, *dev]:
        for field in ('source_rgb', 'target_rgb'):
            protected[p[field]] = sha256(p[field])
    formal_hashes = {sha256(p[f]) for p in pairs for f in ('source_rgb', 'target_rgb')}
    if any(sha256(p[f]) in formal_hashes for p in dev for f in ('source_rgb', 'target_rgb')):
        raise ValueError('Independent development RGB overlaps formal input')
    from diffusers import QwenImage21Pipeline
    from src.pose_control.qwen21_full import frozen_bfs_lora
    payload = {'architecture': 'bfs_rgb_direct_v1', **PROTOCOL, 'variants': list(VARIANTS),
               'reference_roles': ['A', 'B'], 'pair_ids': list(PAIR_IDS), 'versions': versions(),
               'model': str(MODEL.resolve()), 'model_revision': '790c92633540aa0cb11d9abf19eb46d861714758',
               'align_mode': 'letterbox_v1', 'training': False, 'project_adapters': False,
               'second_edit': False, 'lora_strength': 1., 'protected_files': protected,
               'prompt_file_sha256': sha256(PREVIOUS / 'prompts/locked_prompts.json'),
               'pair_file_sha256': sha256(PREVIOUS / 'data/pairs.jsonl'),
               'runtime_sources': {str(Path(p)): sha256(p) for p in [
                   __file__, inspect.getfile(sample), inspect.getfile(QwenImage21Pipeline), frozen_bfs_lora.__file__]}}
    lock_path = root / 'experiment_lock.json'
    if lock_path.exists() and json.loads(lock_path.read_text()) != payload:
        raise ValueError('Existing experiment lock differs')
    json_write(lock_path, payload)
    for name, value in [('pairs.json', pairs), ('prompts.json', locked), ('dev_pairs.json', dev),
                        ('dev_prompts.json', dev_prompts)]:
        target = root / 'data' / name
        if target.exists() and json.loads(target.read_text()) != value:
            raise ValueError(f'Locked input changed: {name}')
        json_write(target, value)
    old_dir = PREVIOUS / 'generations/sampling_1024_30'
    config = json.loads((old_dir / 'config.json').read_text())
    old = {r['pair_id']: r for r in [json.loads(s) for s in (old_dir / 'per_sample.jsonl').read_text().splitlines()]
           if r['method'] == 'native_ab'}
    validations = []
    for p in pairs:
        _, refs, _ = prepare_rgb(p, root / 'generations')
        row = old[p['pair_id']]
        reasons = []
        try:
            if config['versions'] != versions() or any(config.get(k) != v for k, v in {
                'model_revision': payload['model_revision'], 'align_mode': 'letterbox_v1',
                'steps': 30, 'seed': 42, 'true_cfg_scale': 1., 'use_kv_cache': False}.items()):
                raise ValueError('Historical runtime/sampler configuration differs')
            validate_native(row, locked[p['pair_id']], refs, versions())
        except (ValueError, KeyError, FileNotFoundError) as error:
            reasons.append(str(error))
        reused = not reasons
        if reused:
            result = {**row, 'method': 'native', 'variant': 'native', 'historical_method': 'native_ab',
                      'historical_record_path': str(old_dir / 'per_sample.jsonl'),
                      'historical_record_sha256': sha256(old_dir / 'per_sample.jsonl'),
                      'cost_measurement': 'historical', 'lora_strength': 0., 'lora_sha256': None,
                      'runtime_versions': versions(), 'lock_sha256': sha256(lock_path)}
            append_record(root / 'generations.jsonl', result)
        validations.append({'pair_id': p['pair_id'], 'reused': reused, 'rejection_reasons': reasons,
                            'prompt_sha256': digest_text(locked[p['pair_id']]), 'references': refs,
                            'historical_image': row['image'], 'historical_image_sha256': row['image_sha256']})
    json_write(root / 'data/baseline_validation.json', validations)
    json_write(root / 'preparation.json', {'finished_utc': timestamp(), 'native_reused': sum(v['reused'] for v in validations),
                                          'formal_count': 12, 'new_lora_count': 8})
    print(json.dumps({'prepared': True, 'native_reused': sum(v['reused'] for v in validations)}, ensure_ascii=False), flush=True)


def preflight(pipe, variant, root):
    pair = json.loads((root / 'data/dev_pairs.json').read_text())[0]
    prompt = json.loads((root / 'data/dev_prompts.json').read_text())[pair['pair_id']]
    refs, records, _ = prepare_rgb(pair, root / 'development' / variant)
    captured = {}
    class Captured(Exception):
        pass
    def capture(module, args, kwargs):
        captured.update(args=args, kwargs=kwargs)
        raise Captured()
    hook = pipe.transformer.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        try:
            sample(pipe, prompt, refs, RESOLUTION, witness=False)
        except Captured:
            pass
    finally:
        hook.remove()
    if not captured:
        raise AssertionError('No captured A+B Transformer input')
    def predict():
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            output = pipe.transformer(*captured['args'], **captured['kwargs'])
        value = output[0] if isinstance(output, (list, tuple)) else output.sample
        if not torch.isfinite(value).all():
            raise FloatingPointError('Nonfinite A+B prediction')
        return value.detach().float().cpu()
    native = predict()
    report = load_frozen_bfs(pipe, LORAS[variant], adapter_name=variant, strength=0.)
    zero = predict()
    error = float((native - zero).abs().max())
    if error > 1e-3:
        raise AssertionError(f'Zero-strength prediction differs: {error}')
    pipe.set_adapters([variant], adapter_weights=[1.])
    pipe.transformer.eval().requires_grad_(False)
    state, _ = _convert_state(LORAS[variant], pipe.transformer)
    report.update(_verify_loaded(pipe.transformer, state, variant, 1.), strength=1.)
    live = predict()
    impact = float((native - live).abs().max())
    if impact == 0:
        raise AssertionError('BFS has no prediction impact')
    value = {'passed': True, 'pair_id': pair['pair_id'], 'resolution': RESOLUTION,
             'formal_ranking': False, 'zero_strength_max_abs': error,
             'strength_one_max_abs_change': impact, 'all_predictions_finite': True,
             'reference_roles': ['A', 'B'], 'references': records, 'prompt': prompt,
             'project_adapter_calls': 0, 'loading': report}
    json_write(root / 'development' / variant / 'preflight.json', value)
    json_write(root / 'loading' / f'{variant}.json', report)
    del captured, native, zero, live, state
    torch.cuda.empty_cache()
    return report


def run(args):
    root, variant = args.output, args.variant
    lock_path = root / 'experiment_lock.json'
    lock = json.loads(lock_path.read_text())
    if lock['versions'] != versions():
        raise ValueError('Runtime changed after preparation')
    for path, expected in {**lock['runtime_sources'], **lock['protected_files']}.items():
        if sha256(path) != expected:
            raise ValueError(f'Locked source/input/checkpoint changed: {path}')
    pairs = json.loads((root / 'data/pairs.json').read_text())
    prompts = json.loads((root / 'data/prompts.json').read_text())
    index = root / 'generations.jsonl'
    previous = {(r['pair_id'], r['method']): r for r in [json.loads(s) for s in index.read_text().splitlines()]} if index.exists() else {}
    pending = []
    for pair in pairs:
        _, refs, _ = prepare_rgb(pair, root / 'generations')
        saved = previous.get((pair['pair_id'], variant))
        if saved:
            validate_saved(saved, prompts[pair['pair_id']], refs, variant, sha256(lock_path))
        else:
            pending.append(pair)
    if not pending:
        print(json.dumps({'variant': variant, 'resume_verified': len(pairs), 'gpu_loaded': False}), flush=True)
        return
    torch.set_num_threads(4)
    torch.manual_seed(42)
    from diffusers import QwenImage21Pipeline
    pipe = QwenImage21Pipeline.from_pretrained(str(MODEL), dtype=torch.bfloat16).to('cuda')
    for module in (pipe.transformer, pipe.text_encoder, pipe.vae):
        module.eval().requires_grad_(False)
    pipe.set_progress_bar_config(disable=True)
    report = load_frozen_bfs(pipe, None) if variant == 'native' else preflight(pipe, variant, root)
    before = frozen_transformer_digest(pipe.transformer)
    for pair in pending:
        refs, reference_records, alignment = prepare_rgb(pair, root / 'generations')
        reference_signature(reference_records)
        if getattr(pipe.transformer, '_pic_edit_qwen21_control_active', False):
            raise AssertionError('Unexpected project control hook')
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        output, witness = sample(pipe, prompts[pair['pair_id']], refs, RESOLUTION)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        rgba = output.images[0].convert('RGBA')
        image = Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba).convert('RGB')
        if image.size != (RESOLUTION, RESOLUTION):
            raise AssertionError('Unexpected output dimensions')
        path = root / 'generations/images' / f'{pair["pair_id"]}_{variant}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        row = {**PROTOCOL, 'pair_id': pair['pair_id'], 'dataset': pair['dataset'], 'method': variant,
               'variant': variant, 'prompt': prompts[pair['pair_id']], 'prompt_sha256': digest_text(prompts[pair['pair_id']]),
               'references': reference_records, 'alignment': alignment,
               'source_rgb': reference_records[0]['path'], 'target_rgb': reference_records[1]['path'],
               'source_original_rgb': pair['source_rgb'], 'target_original_rgb': pair['target_rgb'],
               'image': str(path), 'image_sha256': sha256(path), 'seconds': seconds,
               'peak_cuda_gib': torch.cuda.max_memory_allocated() / 1024**3, 'cost_measurement': 'current',
               'lora_loaded': variant != 'native', 'lora_strength': 0. if variant == 'native' else 1.,
               'lora_sha256': report['file_sha256'], 'lora_path': report['path'],
               'runtime_versions': versions(), 'lock_sha256': sha256(lock_path),
               'correctness_observation': witness, 'metadata': pair, 'block_calls': {},
               'created_utc': timestamp()}
        validate_saved(row, row['prompt'], reference_records, variant, sha256(lock_path))
        append_record(index, row)
        status = {'variant': variant, 'last_pair': pair['pair_id'], 'seconds': seconds,
                  'peak_cuda_gib': row['peak_cuda_gib'], 'updated_utc': timestamp()}
        json_write(root / 'status.json', status)
        print(json.dumps(status), flush=True)
    if frozen_transformer_digest(pipe.transformer) != before:
        raise AssertionError('LoRA values changed during inference')
    if any(p.requires_grad or p.grad is not None for m in (pipe.transformer, pipe.text_encoder, pipe.vae) for p in m.parameters()):
        raise AssertionError('Parameters not fully frozen')
    json_write(root / 'loading' / f'{variant}_completed.json', {
        'passed': True, 'all_parameters_frozen': True, 'lora_digest_unchanged': before,
        'project_adapter_calls': 0, 'no_optimizer_constructed': True,
        'reference_roles': ['A', 'B'], 'loading': report})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--variant', choices=VARIANTS, default='native')
    args = parser.parse_args()
    prepare(args.output) if args.prepare else run(args)
