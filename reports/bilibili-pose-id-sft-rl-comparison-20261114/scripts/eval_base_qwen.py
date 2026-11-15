#!/usr/bin/env python3
"""Evaluate the 50 Base Qwen images with the same CPU evaluator as the RL report."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

RUNTIME = Path('/data/shangguanrz/LoRA/experiments/bilibili-pose-id-grpo/source/runtime_20261112_v9')
sys.path.insert(0, str(RUNTIME))

from src.rl.evaluation import save_evaluation  # noqa: E402

NEW = Path('/data/shangguanrz/LoRA/experiments/bilibili-pose-id-grpo/runs/20261112_v1')
BASE = NEW / 'base_qwen_eval_20261114_v1'
INFER = BASE / 'output/base_infer_20261114_v2'
DEST = BASE / 'eval_base_qwen_20261114_v1'
BASELINE = NEW / 'training/main/pose_id/evaluation/update_000000/val'
MODELS = '/home/shangguanrz/project/pic-edit/models'


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    if DEST.exists():
        raise SystemExit(f'refusing to overwrite {DEST}')
    val_rows = read_jsonl(NEW / 'data_v2/val.jsonl')
    records = [json.loads(path.read_text(encoding='utf-8')) for path in sorted(INFER.glob('baseline_normal_*.json'))]
    if len(records) != 50:
        raise SystemExit(f'expected 50 base outputs, got {len(records)}')
    by_pair = {}
    for record in records:
        pair_id = record['original_row']['pair_id']
        if pair_id in by_pair:
            raise SystemExit(f'duplicate pair {pair_id}')
        by_pair[pair_id] = Path(record['image'])
    missing = [row['pair_id'] for row in val_rows if row['pair_id'] not in by_pair]
    if missing:
        raise SystemExit(f'missing base outputs: {len(missing)}')
    samples = []
    for row in val_rows:
        image = Image.open(by_pair[row['pair_id']]).convert('RGB')
        if image.size != (384, 672):
            raise SystemExit(f'wrong base image size {image.size} for {row["pair_id"]}')
        samples.append(SimpleNamespace(image=image, metadata=json.dumps(row, ensure_ascii=False)))
    summary = save_evaluation(samples, DEST, MODELS, baseline_directory=BASELINE)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
