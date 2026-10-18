#!/usr/bin/env python3
"""Verify and generate the fixed 60-image inheritance comparison, without training."""
from __future__ import annotations

# Project root for the categorized entrypoint.
from pathlib import Path as _EntryPath
import sys as _entry_sys
_entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))


from dataclasses import replace
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from diffusers import QwenImage21Pipeline
from eval_iper_qwen21_full import opaque
from scripts.audit_qwen21_weight_compatibility import file_sha
from scripts.verify_qwen21_full_runtime import empty_face, compare_prepared, parity
from src.pose_control.qwen21_full.execution import (
    MODEL, RUN_ROOT, FaceConditionStore, QwenFullDataset, Qwen21IPERCache,
    batch_one, face_mask, json_write, load_adapter_checkpoint, new_adapter, prepare, seed_all,
)
from src.pose_control.qwen21_full.qwen_adapter import Qwen21FullControlContext
from src.pose_control.qwen21_full.qwen_runtime import PROMPT, MODEL_REVISION, dataset_image_to_pil, transformer_inputs

OUT = RUN_ROOT / 'weight_inheritance'
CHECKPOINTS = {'baseline': RUN_ROOT / 'face_512_100/checkpoint_last.pt',
               'body512': OUT / 'inherit_body512.pt', 'body1024': OUT / 'inherit_body1024.pt'}


def source_pair(pipe, cache, source):
    entry = cache.source(pipe, source)
    tokens = (cache.resolution // 16) ** 2
    return {'source_latent': entry['reference_latent'].to('cuda'),
            'prompt_embeds': entry['prompt_embeds'].to('cuda'),
            'prompt_mask': entry['prompt_mask'].to('cuda'),
            'img_mask': torch.cat((entry['image_mask'], torch.ones(1, tokens // 4, dtype=torch.bool)), 1).to('cuda'),
            'resolution': cache.resolution}


@torch.no_grad()
def verify_gpu(pipe, adapter, cache, dataset, store, candidate, *, face_reference_token_hw=None):
    resolution = cache.resolution
    grid = resolution // 16
    batch = batch_one(dataset, 0)
    pair = source_pair(pipe, cache, dataset_image_to_pil(batch['src_image'][0]))
    noise = torch.randn((1, grid * grid, 64), device='cuda', dtype=pair['source_latent'].dtype,
                        generator=torch.Generator(device='cuda').manual_seed(42))
    sigma = noise.new_tensor([.2])
    inputs = transformer_inputs(pair, noise, sigma)
    body = prepare(adapter, batch, pair, [dataset.samples[0]], store, with_face=False)
    face = prepare(adapter, batch, pair, [dataset.samples[0]], store, with_face=True)
    altered = dict(batch, tgt_image=-batch['tgt_image'])
    changed = prepare(adapter, altered, dict(pair, target_latent=torch.full_like(noise, float('nan'))),
                      [dataset.samples[0]], store, with_face=True)
    leakage = compare_prepared(face, changed)
    if not leakage['exactly_equal']:
        raise AssertionError('Target RGB or supervised latent entered generation conditions')

    def predict(prepared, *, inspect_prefix=False, **strengths):
        handles, snapshots, checked = [], {}, []
        context = Qwen21FullControlContext(pipe.transformer, adapter, prepared,
                    target_token_hw=(grid, grid), hand_strength=0, hand_mode='off',
                    face_reference_token_hw=face_reference_token_hw, **strengths)
        try:
            if inspect_prefix:
                for layer, block in enumerate(pipe.transformer.transformer_blocks):
                    def before(module, args, output, index=layer):
                        prefix = output.shape[1] - grid * grid
                        snapshots[index] = output[:, :prefix].clone()
                    handles.append(block.register_forward_hook(before))
            with context:
                if inspect_prefix:
                    for layer, block in enumerate(pipe.transformer.transformer_blocks):
                        def after(module, args, output, index=layer):
                            reference = snapshots.pop(index)
                            if not torch.equal(reference, output[:, :reference.shape[1]]):
                                raise AssertionError(f'Injection changed prefix at layer {index}')
                            checked.append(index)
                        handles.append(block.register_forward_hook(after))
                result = pipe.transformer(**inputs)[0][:, -grid * grid:].clone()
            if not bool(torch.isfinite(result).all()):
                raise AssertionError('Nonfinite controlled prediction')
            if inspect_prefix and (checked != list(range(32)) or context.call_counts != {i: 1 for i in range(32)}):
                raise AssertionError('Missing block injection verification')
            return result, checked
        finally:
            for handle in handles:
                handle.remove()

    with torch.autocast('cuda', dtype=torch.bfloat16):
        native = pipe.transformer(**inputs)[0][:, -grid * grid:]
        disabled, _ = predict(face, geometry_strength=0, interaction_strength=0, detail_strength=0, face_strength=0)
        body_prediction, _ = predict(body, face_strength=0)
        off, _ = predict(face, face_strength=0)
        invalid, _ = predict(replace(face, face=empty_face(face.face)).validate(), face_strength=1)
        _, checked = predict(face, face_strength=1, inspect_prefix=True)
        full = adapter.build_residuals(noisy_target=noise, prepared=face,
                    prompt_embeds=pair['prompt_embeds'], prompt_mask=pair['prompt_mask'], sigma=sigma,
                    target_token_hw=(grid, grid), face_strength=1, hand_strength=0, hand_mode='off',
                    face_reference_token_hw=face_reference_token_hw)
        support = face_mask(face, grid).squeeze(-1) > 0
        if full.raw_residuals.face is None:
            raise AssertionError('Face did not run in the valid reference check')
        projected = [adapter.output_bridges['face'](value) for value in full.raw_residuals.face]
        outside = max(float(value[~support].abs().max()) if (~support).any() else 0. for value in projected)
    comparisons = {'all_control_off_vs_native': parity(native, disabled, .001),
                   'face_off_vs_body': parity(body_prediction, off, .001),
                   'invalid_face_vs_body': parity(body_prediction, invalid, .001)}
    passed = all(value['passed'] for value in comparisons.values()) and outside == 0
    report = {'candidate': candidate, 'resolution': resolution, 'passed': passed,
              'face_reference_token_hw': face_reference_token_hw,
              'comparisons': comparisons, 'target_rgb_leakage_check': leakage,
              'prefix_unchanged_at_injection_layers': checked,
              'face_residual_outside_roi_max_abs': outside, 'target_tokens': grid * grid,
              'backward_or_training_performed': False}
    if not passed:
        raise AssertionError(report)
    return report


def check_sample_boundaries():
    from src.pose_control.qwen21_full.data_runtime import read_pairs, pair_metadata
    def identifiers(path):
        return {(m['source_appearance'], m['source_stem'], m['target_appearance'], m['target_stem'])
                for row in read_pairs(path) for m in [pair_metadata(row)]}
    val = identifiers(RUN_ROOT / 'val_pairs.jsonl')
    cross = identifiers(RUN_ROOT / 'cross_person_pairs.jsonl')
    train = identifiers(RUN_ROOT / 'train_pairs.jsonl')
    final = identifiers(ROOT / 'datasets/iPER/iper_sampled_6src64tgt/splits/baseline_50_eval_pairs.jsonl')
    if len(val) != 4 or len(cross) != 2 or val & train or (val | cross) & final:
        raise AssertionError('Fixed sample boundaries failed')
    return {'validation_pairs': 4, 'cross_person_pairs': 2, 'final_test_pairs_used': False,
            'validation_training_pair_overlap': 0}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = {'checkpoints': {key: {'path': str(path), 'sha256': file_sha(path)} for key, path in CHECKPOINTS.items()},
              'pairs': {name: file_sha(RUN_ROOT / f'{name}_pairs.jsonl') for name in ('val', 'cross_person')},
              'steps': 30, 'seed': 42, 'true_cfg_scale': 1.0, 'prompt': PROMPT,
              'model': str(MODEL), 'model_revision': MODEL_REVISION, 'use_kv_cache': True,
              'face_reference_image_added': False, 'training_performed': False,
              'sample_boundaries': check_sample_boundaries(), 'expected_images': 60}
    manifest = OUT / 'generation_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError('Existing comparison config differs')
    json_write(manifest, config)
    seed_all(42)
    torch.set_num_threads(4)
    pipe = QwenImage21Pipeline.from_pretrained(str(MODEL), dtype=torch.bfloat16).to('cuda')
    for model in (pipe.transformer, pipe.text_encoder, pipe.vae):
        model.eval().requires_grad_(False)
    base_versions = [parameter._version for model in (pipe.transformer, pipe.text_encoder, pipe.vae)
                     for parameter in model.parameters()]
    pipe.set_progress_bar_config(disable=True)
    adapter = new_adapter('cuda').eval().requires_grad_(False)
    store = FaceConditionStore(RUN_ROOT / 'face_cache')
    acceptance_path = OUT / 'runtime_checks.json'
    acceptance = json.loads(acceptance_path.read_text()) if acceptance_path.exists() else {}
    records = OUT / 'generations.jsonl'
    rows = [json.loads(line) for line in records.read_text().splitlines()] if records.exists() else []
    completed = {(row['set'], row['resolution'], row['candidate'], row['mode'], row['pair_index']): row for row in rows}
    for resolution in (512, 1024):
        cache = Qwen21IPERCache(OUT / 'native_cache', resolution=resolution, prompt=PROMPT)
        validation = QwenFullDataset(RUN_ROOT / 'val_pairs.jsonl', resolution=resolution)
        datasets = [('validation', validation)]
        if resolution == 1024:
            datasets.append(('cross_person', QwenFullDataset(RUN_ROOT / 'cross_person_pairs.jsonl', resolution=resolution)))
        for candidate, checkpoint in CHECKPOINTS.items():
            load_adapter_checkpoint(adapter, checkpoint)
            adapter.eval().requires_grad_(False)
            versions = [p._version for p in adapter.parameters()]
            key = f'{candidate}_{resolution}'
            acceptance[key] = verify_gpu(pipe, adapter, cache, validation, store, candidate)
            json_write(acceptance_path, acceptance)
            print(json.dumps({'runtime_check': key, 'passed': True}), flush=True)
            for set_name, dataset in datasets:
                for mode in ('body', 'body_face'):
                    for index in range(len(dataset)):
                        identifier = (set_name, resolution, candidate, mode, index)
                        old = completed.get(identifier)
                        if old and Path(old['image']).exists() and file_sha(Path(old['image'])) == old['image_sha256']:
                            continue
                        batch = batch_one(dataset, index)
                        meta = dataset[index]['pair_metadata']
                        directory = OUT / f'{set_name}_{resolution}' / candidate
                        directory.mkdir(parents=True, exist_ok=True)
                        stem = f'{index:03d}_{meta["source_appearance"]}_{meta["source_stem"]}_{meta["target_stem"]}'
                        source = dataset_image_to_pil(batch['src_image'][0])
                        target = dataset_image_to_pil(batch['tgt_image'][0])
                        source_path, target_path = directory / f'{stem}_source.png', directory / f'{stem}_target.png'
                        source.save(source_path)
                        target.save(target_path)
                        with torch.no_grad():
                            pair = source_pair(pipe, cache, source)
                            prepared = prepare(adapter, batch, pair, [meta], store, with_face=mode == 'body_face')
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        started = time.perf_counter()
                        def finite_output(module, args, output):
                            tensor = output[0]
                            if not bool(torch.isfinite(tensor).all()):
                                raise AssertionError('Nonfinite prediction during complete sampling')
                        handle = pipe.transformer.register_forward_hook(finite_output)
                        try:
                            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                                with Qwen21FullControlContext(pipe.transformer, adapter, prepared,
                                        face_strength=1 if mode == 'body_face' else 0,
                                        hand_strength=0, hand_mode='off', progress_mode='step', num_inference_steps=30) as context:
                                    result = pipe(prompt=PROMPT, image=[source], width=resolution, height=resolution,
                                                  output_resolution=resolution, num_inference_steps=30,
                                                  true_cfg_scale=1.0, use_kv_cache=True,
                                                  generator=torch.Generator(device='cuda').manual_seed(42))
                        finally:
                            handle.remove()
                        torch.cuda.synchronize()
                        seconds = time.perf_counter() - started
                        image = opaque(result.images[0])
                        if image.size != (resolution, resolution) or context.forward_count != 30 or any(v != 30 for v in context.call_counts.values()):
                            raise AssertionError('Incomplete sampling or wrong image resolution')
                        path = directory / f'{stem}_{mode}.png'
                        image.save(path)
                        row = {'set': set_name, 'resolution': resolution, 'candidate': candidate, 'mode': mode,
                               'pair_index': index, 'metadata': meta, 'image': str(path), 'image_sha256': file_sha(path),
                               'source': str(source_path), 'target': str(target_path),
                               'seconds': seconds, 'peak_cuda_gib': torch.cuda.max_memory_allocated() / 1024**3,
                               'checkpoint_sha256': config['checkpoints'][candidate]['sha256'],
                               'control_forwards': context.forward_count, 'block_calls': context.call_counts,
                               'sampling_schedule': context.step_diagnostics,
                               'last_control_diagnostics': context.output.diagnostics,
                               'face_valid': bool(prepared.face.face_valid.any()) if prepared.face is not None else None}
                        with records.open('a') as stream:
                            stream.write(json.dumps(row) + '\n')
                        completed[identifier] = row
                        print(json.dumps({'completed': len(completed), 'expected': 60, 'case': identifier,
                                          'seconds': seconds, 'peak_cuda_gib': row['peak_cuda_gib']}), flush=True)
                        del result, pair, prepared
            if [p._version for p in adapter.parameters()] != versions:
                raise AssertionError('Inference modified adapter parameters')
    if len(completed) != 60 or len(acceptance) != 6:
        raise AssertionError('Incomplete fixed comparison')
    after = [parameter._version for model in (pipe.transformer, pipe.text_encoder, pipe.vae) for parameter in model.parameters()]
    if after != base_versions:
        raise AssertionError('Inference modified base parameters')
    for candidate, checkpoint in CHECKPOINTS.items():
        if file_sha(checkpoint) != config['checkpoints'][candidate]['sha256']:
            raise AssertionError('Checkpoint changed during inference')
    json_write(OUT / 'generation_summary.json', {'passed': True, 'images': 60, 'runtime_cases': 6,
               'training_performed': False, 'base_and_adapter_parameters_unchanged': True,
               'final_test_set_used': False, 'all_predictions_finite': True})


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        OUT.mkdir(parents=True, exist_ok=True)
        json_write(OUT / 'generation_failure.json', {'type': type(error).__name__, 'error': str(error),
                   'traceback': traceback.format_exc(), 'training_performed': False})
        raise
