#!/usr/bin/env python3
"""Build the 50-pair val subset and an isolated config for Base Qwen inference.

New artifacts only. The source experiment data and configs are read-only.
"""
import json
import sys
from pathlib import Path

OLD = Path('/data/shangguanrz/LoRA/experiments/bilibili-dwpose-edit-lora')
NEW = Path('/data/shangguanrz/LoRA/experiments/bilibili-pose-id-grpo/runs/20261112_v1')
BASE = NEW / 'base_qwen_eval_20261114_v1'
OLD_DATA = OLD / 'data/data_20261111_v1'
NEW_DATA = NEW / 'data_v2'
OLD_CONFIG = OLD / 'configs/full.json'


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows),
                          encoding='utf-8')


def main():
    if BASE.exists():
        raise SystemExit(f'refusing to overwrite existing {BASE}')
    new_rows = read_jsonl(NEW_DATA / 'val.jsonl')
    old_rows = read_jsonl(OLD_DATA / 'val.jsonl')
    if len(new_rows) != 50:
        raise SystemExit(f'expected 50 new val pairs, got {len(new_rows)}')
    old_by_id = {row['pair_id']: row for row in old_rows}
    missing = [row['pair_id'] for row in new_rows if row['pair_id'] not in old_by_id]
    if missing:
        raise SystemExit(f'missing old cache rows: {len(missing)}')
    selected = [old_by_id[row['pair_id']] for row in new_rows]
    if len({row['pair_id'] for row in selected}) != 50:
        raise SystemExit('duplicate pair ids after selection')

    data_dir = BASE / 'data_subset'
    config_dir = BASE / 'configs'
    data_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)

    write_jsonl(data_dir / 'pilot_val.jsonl', selected)
    write_jsonl(data_dir / 'pilot_train.jsonl', [])
    write_jsonl(data_dir / 'diagnostics.jsonl', [])
    write_json(data_dir / 'index.json', {
        'schema': 'base_qwen_subset_v1',
        'source': str(NEW_DATA / 'val.jsonl'),
        'pairs': 50,
        'note': '50 fixed validation pairs for Base Qwen reference only',
    })

    config = json.loads(OLD_CONFIG.read_text(encoding='utf-8'))
    config.update({
        'workspace_root': str(BASE),
        'cache_manifests': ['pilot_train.jsonl', 'pilot_val.jsonl'],
        'validation_manifest': 'pilot_val.jsonl',
        'expected_train_pairs': 50,
        'inference_steps': 30,
        'true_cfg_scale': 1.0,
        'seed': 42,
        'canvas_wh': [384, 672],
        'prompt': selected[0]['prompt'],
        'note': 'Base Qwen reference inference for the 50 fixed validation pairs; no adapter, no training.',
    })
    write_json(config_dir / 'base_infer.json', config)

    clips = {}
    id_pairs = 0
    for row in new_rows:
        clips[row['clip_id']] = clips.get(row['clip_id'], 0) + 1
        ctx = row.get('reward_context')
        if isinstance(ctx, str):
            ctx = json.loads(ctx)
        if ctx and ctx.get('id_applicable'):
            id_pairs += 1
    print(json.dumps({'base': str(BASE), 'pairs': 50, 'clips': clips, 'id_applicable': id_pairs,
                      'config': str(config_dir / 'base_infer.json')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    sys.exit(main())
