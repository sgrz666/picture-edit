#!/usr/bin/env python3
"""Verify committed pilot artifacts, frozen identities and measured resource bounds."""
import argparse
import hashlib
import math
from pathlib import Path

from core import read_json, read_jsonl, sha256, verify_row_assets, write_json
from runtime import open_cache


def audit(config, tag, report_name):
    root = Path(config['workspace_root'])
    events = read_jsonl(root / 'output' / f'campaign_{tag}' / 'events.jsonl')
    if events[-1]['event'] != 'campaign_completed':
        raise ValueError('Campaign did not complete')
    command = next(e['command'] for e in events if e['event'] == 'started')
    data, cache = [Path(command[command.index('--' + k) + 1]) for k in ('data', 'cache')]
    index = open_cache(config, data, cache)
    for name in ('pilot_train.jsonl', 'pilot_val.jsonl', 'diagnostics.jsonl'):
        for row in read_jsonl(data / name):
            verify_row_assets(row)
    source = root.parents[1] / 'source/diffusers'
    provenance = read_json(source / 'provenance.json')
    blobs = {}
    for record in provenance['upstream_git_blobs']:
        body = (source / record['file']).read_bytes()
        actual = hashlib.sha1(f'blob {len(body)}\0'.encode() + body).hexdigest()
        if actual != record['git_blob_sha1'] or len(body) != record['bytes']:
            raise ValueError(f'Upstream snapshot changed: {record["file"]}')
        blobs[record['file']] = actual
    for split, expected in read_json(data / 'index.json')['source_manifests'].items():
        if sha256(Path(config['dataset_root']) / split / 'pairs.jsonl') != expected:
            raise ValueError(f'Original dataset manifest changed: {split}')
    for record in index['entries'].values():
        if sha256(cache / record['file']) != record['sha256']:
            raise ValueError('Cache tensor changed')
    training = root / 'output' / f'pilot_{tag}'
    updates = [row for path in sorted(training.glob('updates_*.jsonl')) for row in read_jsonl(path)]
    if [r['step'] for r in updates] != list(range(1, 301)):
        raise ValueError('Expected 300 consecutive real updates')
    if not all(math.isfinite(r['loss']) and math.isfinite(r['gradient_norm']) and r['gradient_norm'] > 0 for r in updates):
        raise ValueError('Non-finite or zero-gradient update')
    trainable = read_json(training / 'trainable.json')
    if not trainable['base_frozen'] or not trainable['optimizer_only_lora']:
        raise ValueError('Trainable parameter boundary failed')
    checkpoints = {}
    for step in (100, 200, 300):
        path = training / f'checkpoint-{step:06d}'
        record = read_json(path / 'checkpoint.json')
        if sha256(path / 'training_state.pt') != record['training_state_sha256']:
            raise ValueError('Training state changed')
        if sha256(path / 'pytorch_lora_weights.safetensors') != record['lora_sha256']:
            raise ValueError('Saved LoRA changed')
        checkpoints[step] = record
    resume = read_json(training / 'resume_000100.json')
    if not resume['lora_exact'] or not resume['rng_restored'] or resume['optimizer_states'] != trainable['trainable_tensors']:
        raise ValueError('Resume verification failed')
    inference = root / 'output' / f'comparison_{tag}'
    reload = read_json(inference / 'native_lora_reload.json')
    if not reload['exact_after_dtype_cast'] or reload['adapter_sha256'] != checkpoints[300]['lora_sha256']:
        raise ValueError('Native LoRA reload failed')
    outputs = read_jsonl(inference / 'outputs.jsonl')
    if len(outputs) != 40 or len({(r['method'], r['case']) for r in outputs}) != 40:
        raise ValueError('Expected 40 unique comparison/diagnostic outputs')
    for record in outputs:
        if sha256(record['image']) != record['image_sha256']:
            raise ValueError('Generated image changed')
    report = root / 'report' / f'pilot_report_{tag}'
    resources = read_json(report / 'resource_summary.json')
    if any(r['error'] or r['violation'] or r['min_gpu_free_mib'] < config['min_free_gpu_mib']
           or r['min_ram_available_mib'] < config['min_available_ram_mib'] for r in resources):
        raise ValueError('Resource bounds were violated')
    result = {'pass': True, 'real_updates': len(updates), 'generated_images': len(outputs),
              'cache_samples_verified': len(index['entries']), 'upstream_blobs': blobs,
              'trainable_parameters': trainable['trainable_parameters'], 'resume': resume,
              'lora_reload': reload, 'checkpoints': checkpoints, 'versions': index['versions'],
              'min_gpu_free_mib_all_stages': min(r['min_gpu_free_mib'] for r in resources),
              'min_available_ram_mib_all_stages': min(r['min_ram_available_mib'] for r in resources),
              'metrics_sha256': sha256(report / 'metrics.json'), 'original_manifests_unchanged': True}
    write_json(report / report_name, result)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--report-name', required=True)
    args = parser.parse_args()
    audit(read_json(args.config), args.tag, args.report_name)
