#!/usr/bin/env python3
"""Inference-only BFS comparison with the locked native A+skeleton protocol.

Run --prepare once, then one fresh process per variant. No project control
Adapter is constructed. Resume checks images, prompts, references and runtime.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from scripts.eval.eval_qwen21_native_value import (
    RUN as PREVIOUS, ROOT as PROJECT, SamplingWitness, append_record, json_write,
    read_rows, references_for_method, save_inputs, sha256,
)
from scripts.eval.qwen21_native_value_prompts import build_prompt
from src.pose_control.qwen21_full.frozen_bfs_lora import (
    load_frozen_bfs, _convert_state, _verify_loaded, frozen_transformer_digest,
)

DEFAULT = Path('/home/shangguanrz/nihaochen/qwen21_full/bfs_skeleton_direct_20261010')
MODEL = PROJECT / 'models/Qwen-Image-2.1'
LORAS = {
    'bfs_body': Path('/home/shangguanrz/models/bfs/bfs_body_swap_v1.0_qwen_2.1.safetensors'),
    'bfs_head': Path('/home/shangguanrz/models/bfs/bfs_head_v1.1_qwen_2.1.safetensors'),
}
EXPECTED_SHA = {
    'bfs_body': '7dc0a53aba4dbc70c204f498936a619d226559da11011f748c389189af46b664',
    'bfs_head': 'd1d748d5601077f3b6d05766f6823510e901970d916afa404a97e85dc92fa88e',
}
VARIANTS = ('native', 'bfs_body', 'bfs_head')
HIGHRES = ('iper_00', 'iper_01', 'tiktok_00', 'tiktok_01')


def versions():
    return {'python': sys.version.split()[0], 'torch': torch.__version__,
            'diffusers': importlib.metadata.version('diffusers'),
            'transformers': importlib.metadata.version('transformers'), 'torch_cuda': torch.version.cuda}


def digest_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def prepare(root):
    pairs = read_rows(PREVIOUS / 'data/pairs.jsonl')
    assert len(pairs) == 12 and sum(p['dataset'] == 'iper' for p in pairs) == 6
    prompts = json.loads((PREVIOUS / 'prompts/locked_prompts.json').read_text())
    locked = {p['pair_id']: prompts[p['pair_id']]['native_skeleton'] for p in pairs}
    root.mkdir(parents=True, exist_ok=True)
    baseline, validations = [], []
    fingerprints = {str(path): sha256(path) for path in LORAS.values()}
    for name, path in LORAS.items():
        if fingerprints[str(path)] != EXPECTED_SHA[name]:
            raise ValueError(f'BFS source checksum differs: {name}')
    for res in (512, 1024):
        directory = root / f'generations/sampling_{res}_30'
        selected = [p for p in pairs if res == 512 or p['pair_id'] in HIGHRES]
        olddir = PREVIOUS / f'generations/sampling_{res}_30'
        config = json.loads((olddir / 'config.json').read_text())
        if config['versions'] != versions():
            raise ValueError('Historical baseline runtime differs; regenerate native before reuse')
        if any(config[k] != v for k, v in dict(steps=30, seed=42, true_cfg_scale=1., use_kv_cache=False,
                                                align_mode='letterbox_v1', model_revision='790c92633540aa0cb11d9abf19eb46d861714758').items()):
            raise ValueError('Historical sampling protocol differs')
        old = {r['pair_id']: r for r in map(json.loads, (olddir / 'per_sample.jsonl').read_text().splitlines())
               if r['method'] == 'native_skeleton'}
        for pair in selected:
            prepared = save_inputs(pair, res, directory)
            images, paths, alignment, crop = prepared
            _, refs = references_for_method('native_skeleton', images, paths)
            for ref in refs:
                ref['sha256'] = sha256(ref['path'])
            row = old[pair['pair_id']]
            if row['prompt'] != locked[pair['pair_id']] or row['prompt_sha256'] != digest_text(row['prompt']):
                raise ValueError('Historical baseline prompt differs')
            if [(r['index'], r['role'], r['sha256']) for r in row['references']] != [(r['index'], r['role'], r['sha256']) for r in refs]:
                raise ValueError('Historical baseline prepared inputs differ')
            if sha256(row['image']) != row['image_sha256'] or Image.open(row['image']).size != (res, res):
                raise ValueError('Historical baseline image changed')
            if row['control_forwards'] != 0 or row['lora_loaded'] or row['target_rgb_in_native_references']:
                raise ValueError('Historical baseline is not native A+skeleton')
            for ref in row['references']:
                if sha256(ref['path']) != ref['sha256']:
                    raise ValueError('Historical reference file changed')
            for field in ('source_rgb', 'target_rgb', 'target_dwpose'):
                fingerprints[pair[field]] = sha256(pair[field])
            baseline.append({**row, 'method': 'native', 'historical_method': row['method'],
                             'historical_record_path': str(olddir / 'per_sample.jsonl'),
                             'historical_record_sha256': sha256(olddir / 'per_sample.jsonl'),
                             'cost_measurement': 'historical', 'lora_strength': 0.,
                             'runtime_versions': versions(), 'reference_checksums_revalidated': True})
            validations.append({'pair_id': pair['pair_id'], 'resolution': res, 'references': refs,
                                'alignment': alignment, 'baseline_image_sha256': row['image_sha256'],
                                'prompt_sha256': row['prompt_sha256'], 'baseline_reused': True})
    dev = read_rows(PREVIOUS / 'data/dev_pairs.jsonl')[:1]
    # One already-independent development case is shared by the two LoRA checks.
    devprompts = {p['pair_id']: build_prompt(p, 'native_skeleton', 'structured') for p in dev}
    formal_hashes = {sha256(p[f]) for p in pairs for f in ('source_rgb', 'target_rgb')}
    if any(sha256(p[f]) in formal_hashes for p in dev for f in ('source_rgb', 'target_rgb')):
        raise ValueError('Development/formal RGB overlap')
    payload = {'architecture': 'bfs_skeleton_direct_v1', 'versions': versions(),
               'model': str(MODEL.resolve()), 'model_revision': '790c92633540aa0cb11d9abf19eb46d861714758',
               'steps': 30, 'seed': 42, 'true_cfg_scale': 1., 'use_kv_cache': False,
               'strength': 1., 'reference_roles': ['A', 'B_skeleton'], 'training': False,
               'project_adapters': False, 'highres_ids': list(HIGHRES), 'pair_ids': [p['pair_id'] for p in pairs],
               'input_files_sha256': fingerprints, 'prior_manifest_sha256': sha256(PREVIOUS / 'data/pairs.jsonl'),
               'prior_prompts_sha256': sha256(PREVIOUS / 'prompts/locked_prompts.json'),
               'sampling_implementation_sha256': sha256(inspect.getfile(__import__('diffusers').QwenImage21Pipeline)),
               'historical_sampling_implementation_note': 'Same installed runtime and identical sampling call; historical config did not store pipeline-source SHA.',
               'runtime_source_sha256': {str(p): sha256(p) for p in [Path(__file__), PROJECT / 'src/pose_control/qwen21_full/frozen_bfs_lora.py']}}
    lock = root / 'experiment_lock.json'
    if lock.exists() and json.loads(lock.read_text()) != payload:
        raise ValueError('Experiment lock differs; do not overwrite')
    json_write(lock, payload)
    for name, data in [('pairs.json', pairs), ('prompts.json', locked), ('dev_pairs.json', dev),
                       ('dev_prompts.json', devprompts), ('baseline_validation.json', validations)]:
        json_write(root / 'data' / name, data)
    for row in baseline:
        append_record(root / 'generations.jsonl', row)
    print(json.dumps({'prepared': True, 'reused_native': len(baseline), 'new_formal_expected': 32}), flush=True)


def validate_saved(saved, expected, refs):
    for key in ('variant', 'resolution', 'prompt', 'steps', 'seed', 'true_cfg_scale', 'use_kv_cache',
                'lora_sha256', 'lora_strength', 'lock_sha256', 'runtime_versions'):
        if saved.get(key) != expected.get(key):
            raise ValueError(f'Resume differs: {key}')
    if saved.get('prompt_sha256') != digest_text(expected['prompt']):
        raise ValueError('Resume prompt checksum differs')
    if saved['references'] != refs or any(sha256(r['path']) != r['sha256'] for r in refs):
        raise ValueError('Resume references differ')
    if sha256(saved['image']) != saved['image_sha256'] or Image.open(saved['image']).size != (saved['resolution'],)*2:
        raise ValueError('Resume image changed')
    if saved['control_forwards'] != 0:
        raise ValueError('Unexpected project Adapter')


def validate_native_reuse(saved, prompt, refs, resolution, runtime):
    expected = dict(resolution=resolution, prompt=prompt, steps=30, seed=42,
                    true_cfg_scale=1., use_kv_cache=False, control_forwards=0,
                    lora_loaded=False, target_rgb_in_native_references=False, runtime_versions=runtime)
    if any(saved.get(k) != v for k,v in expected.items()):
        raise ValueError('Historical native protocol differs')
    if saved['prompt_sha256'] != digest_text(prompt):
        raise ValueError('Historical native prompt checksum differs')
    signature = lambda records: [(r['index'],r['role'],r['sha256']) for r in records]
    if signature(saved['references']) != signature(refs):
        raise ValueError('Historical native reference content differs')
    for ref in [*saved['references'], *refs]:
        if sha256(ref['path']) != ref['sha256']:
            raise ValueError('Historical native reference file changed')
    if sha256(saved['image']) != saved['image_sha256'] or Image.open(saved['image']).size != (resolution,)*2:
        raise ValueError('Historical native output changed')


def sample(pipe, prompt, refs, res, witness=True):
    kwargs = dict(prompt=prompt, image=refs, width=res, height=res, output_resolution=res,
                  num_inference_steps=30, true_cfg_scale=1., use_kv_cache=False,
                  generator=torch.Generator(device='cuda').manual_seed(42))
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        if not witness:
            return pipe(**kwargs)
        observer = SamplingWitness(pipe.transformer, None, 30, res // 16)
        with observer:
            result = pipe(**kwargs)
    return result, observer.record()


def preflight(pipe, variant, root):
    """Compare exactly the same first-step two-reference Transformer input."""
    pair = json.loads((root / 'data/dev_pairs.json').read_text())[0]
    prompt = json.loads((root / 'data/dev_prompts.json').read_text())[pair['pair_id']]
    directory = root / 'development' / variant
    images, paths, _, _ = save_inputs(pair, 512, directory)
    refs, _ = references_for_method('native_skeleton', images, paths)
    captured = {}
    class Captured(Exception):
        pass
    def capture(module, args, kwargs):
        captured['args'], captured['kwargs'] = args, kwargs
        raise Captured()
    hook = pipe.transformer.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        try:
            sample(pipe, prompt, refs, 512, witness=False)
        except Captured:
            pass
    finally:
        hook.remove()
    if not captured:
        raise AssertionError('Did not capture native Transformer input')
    def predict():
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            output = pipe.transformer(*captured['args'], **captured['kwargs'])
        value = output[0] if isinstance(output, (tuple, list)) else output.sample
        if not torch.isfinite(value).all():
            raise FloatingPointError('Nonfinite prediction')
        return value.detach().float().cpu()
    native = predict()
    report = load_frozen_bfs(pipe, LORAS[variant], adapter_name=variant, strength=0.)
    zero = predict()
    error = float((native - zero).abs().max())
    if error > 1e-3:
        raise AssertionError(f'Zero-strength prediction error: {error}')
    pipe.set_adapters([variant], adapter_weights=[1.])
    # PEFT set_adapter re-enables LoRA requires_grad even for an inference arm.
    pipe.transformer.eval().requires_grad_(False)
    state, _ = _convert_state(LORAS[variant], pipe.transformer)
    report.update(_verify_loaded(pipe.transformer, state, variant, 1.), strength=1.)
    live = predict()
    impact = float((native - live).abs().max())
    if impact == 0:
        raise AssertionError('LoRA had no prediction effect')
    del captured, native, zero, live, state
    result, observation = sample(pipe, prompt, refs, 512)
    directory.mkdir(parents=True, exist_ok=True)
    result.images[0].convert('RGB').save(directory / 'complete_sample.png')
    json_write(directory / 'preflight.json', {'passed': True, 'pair_id': pair['pair_id'],
        'formal_ranking': False, 'zero_strength_max_abs': error, 'strength_one_max_abs_change': impact,
        'all_predictions_finite': True, 'project_adapter_calls': 0, 'observation': observation,
        'prompt': prompt, 'reference_roles': ['A', 'B_skeleton'], 'loading': report,
        'complete_image_sha256': sha256(directory / 'complete_sample.png')})
    json_write(root / 'loading' / f'{variant}.json', report)
    return report


def run(args):
    root = args.output
    lock = json.loads((root / 'experiment_lock.json').read_text())
    source_matches = lock['runtime_source_sha256'][str(Path(__file__))] == sha256(__file__)
    if not source_matches and (root / 'post_run_code_revision.json').exists():
        # Keep the exact source used for all 32 images; this declared revision
        # only adds read-only native resume. Unknown later source edits fail.
        revision = json.loads((root / 'post_run_code_revision.json').read_text())
        source_matches = (revision['current_sha256'] == sha256(__file__)
                          and revision['executed_sha256'] == lock['runtime_source_sha256'][str(Path(__file__))]
                          and sha256(revision['executed_snapshot']) == revision['executed_sha256'])
    if lock['versions'] != versions() or not source_matches:
        raise ValueError('Locked runtime/entry changed')
    for path, digest in lock['input_files_sha256'].items():
        if sha256(path) != digest:
            raise ValueError(f'Locked input changed: {path}')
    pairs = json.loads((root / 'data/pairs.json').read_text())
    prompts = json.loads((root / 'data/prompts.json').read_text())
    if args.resolution == 1024:
        pairs = [p for p in pairs if p['pair_id'] in HIGHRES]
    directory = root / f'generations/sampling_{args.resolution}_30'
    index = root / 'generations.jsonl'
    old = [json.loads(l) for l in index.read_text().splitlines()] if index.exists() else []
    existing = {(r['pair_id'], r['resolution'], r['method']): r for r in old}
    if args.variant == 'native' and all((p['pair_id'],args.resolution,'native') in existing for p in pairs):
        for pair in pairs:
            images,paths,_,_ = save_inputs(pair,args.resolution,directory)
            _,refs = references_for_method('native_skeleton',images,paths)
            for ref in refs: ref['sha256'] = sha256(ref['path'])
            previous = existing[pair['pair_id'],args.resolution,'native']
            validate_native_reuse(previous,prompts[pair['pair_id']],refs,args.resolution,versions())
        print(json.dumps({'native_reuse_validated':len(pairs),'resolution':args.resolution,
                          'gpu_loaded':False}),flush=True)
        return
    torch.set_num_threads(4)
    torch.manual_seed(42)
    from diffusers import QwenImage21Pipeline
    pipe = QwenImage21Pipeline.from_pretrained(str(MODEL), dtype=torch.bfloat16).to('cuda')
    for module in (pipe.transformer, pipe.text_encoder, pipe.vae):
        module.eval().requires_grad_(False)
    pipe.set_progress_bar_config(disable=True)
    if args.variant == 'native':
        report = load_frozen_bfs(pipe, None)
    elif args.resolution == 512:
        report = preflight(pipe, args.variant, root)
    else:
        if not json.loads((root / 'development' / args.variant / 'preflight.json').read_text())['passed']:
            raise AssertionError('Missing 512 preflight')
        report = load_frozen_bfs(pipe, LORAS[args.variant], adapter_name=args.variant, strength=1.)
    before = frozen_transformer_digest(pipe.transformer)
    for pair in pairs:
        images, paths, alignment, _ = save_inputs(pair, args.resolution, directory)
        refs, records = references_for_method('native_skeleton', images, paths)
        for ref in records:
            ref['sha256'] = sha256(ref['path'])
        expected = dict(variant=args.variant, resolution=args.resolution, prompt=prompts[pair['pair_id']],
                        steps=30, seed=42, true_cfg_scale=1., use_kv_cache=False,
                        lora_sha256=report['file_sha256'], lora_strength=0. if args.variant == 'native' else 1.,
                        lock_sha256=sha256(root / 'experiment_lock.json'), runtime_versions=versions())
        previous = existing.get((pair['pair_id'], args.resolution, args.variant))
        if previous:
            if previous.get('cost_measurement') == 'historical' and args.variant == 'native':
                validate_native_reuse(previous,expected['prompt'],records,args.resolution,versions())
                continue
            validate_saved(previous, expected, records)
            continue
        if getattr(pipe.transformer, '_pic_edit_qwen21_control_active', False):
            raise AssertionError('Unexpected custom Adapter hooks')
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result, observation = sample(pipe, expected['prompt'], refs, args.resolution)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        image = result.images[0].convert('RGBA')
        image = Image.alpha_composite(Image.new('RGBA', image.size, 'white'), image).convert('RGB')
        if image.size != (args.resolution, args.resolution):
            raise AssertionError('Wrong output size')
        path = directory / 'images' / f'{pair["pair_id"]}_{args.variant}.png'
        path.parent.mkdir(parents=True, exist_ok=True); image.save(path)
        row = {**expected, 'pair_id': pair['pair_id'], 'dataset': pair['dataset'], 'method': args.variant,
               'image': str(path), 'image_sha256': sha256(path), 'prompt_sha256': digest_text(expected['prompt']),
               'source_rgb': paths['A'], 'target_rgb': paths['B'],
               'source_original_rgb': pair['source_rgb'], 'target_original_rgb': pair['target_rgb'],
               'references': records, 'alignment': alignment, 'metadata': pair, 'seconds': seconds,
               'peak_cuda_gib': torch.cuda.max_memory_allocated()/1024**3,
               'cost_measurement': 'current', 'control_forwards': 0, 'block_calls': {},
               'lora_loaded': args.variant != 'native', 'lora_path': report['path'],
               'correctness_observation': observation, 'target_rgb_in_native_references': False}
        append_record(index, row)
        json_write(root / 'status.json', {'variant': args.variant, 'resolution': args.resolution,
                                        'last_pair': pair['pair_id'], 'last_seconds': seconds})
        print(json.dumps({'generated': pair['pair_id'], 'variant': args.variant, 'res': args.resolution,
                          'seconds': seconds, 'peak_cuda_gib': row['peak_cuda_gib']}), flush=True)
    if before != frozen_transformer_digest(pipe.transformer):
        raise AssertionError('Frozen LoRA changed during inference')
    if any(p.requires_grad or p.grad is not None for m in (pipe.transformer, pipe.text_encoder, pipe.vae) for p in m.parameters()):
        raise AssertionError('Inference changed frozen/gradient state')
    json_write(root / 'loading' / f'{args.variant}_{args.resolution}_completed.json',
               {'passed': True, 'all_parameters_frozen': True, 'lora_digest_unchanged': before,
                'project_adapter_calls': 0, 'no_optimizer_constructed': True, 'loading': report})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--variant', choices=VARIANTS, default='native')
    parser.add_argument('--resolution', type=int, choices=(512,1024), default=512)
    args = parser.parse_args()
    prepare(args.output) if args.prepare else run(args)
