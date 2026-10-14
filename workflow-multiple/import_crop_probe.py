#!/usr/bin/env python3
"""Does crop geometry explain the control-group bias in the target-import metric?

Measured fact this probe was asked about: per-body ``target_minus_source`` sits at +0.084 / +0.109 on
same-identity pairs - where "more like B" cannot mean anything, because A and B are the same two
people in the same clothes - while the cross pairs that are allowed to flag land at -0.023. A control
group that out-signals the experiment means the statistic is measuring something other than identity.

The candidate explanation is the crop: reference crops are cut from Hi4D silhouette mask boxes,
generated crops from detector boxes, and both only get 2% padding, so the two sides of one comparison
carry different amounts of background. This script re-crops both sides into a square (an opt-in shape,
so nothing stored changes) and reports the same statistic under both crop geometries, grouped by pair
kind. It writes only its own json under logs_batch/ and never touches a run's artifacts.

Usage: import_crop_probe.py [--runs a b c] [--limit N] [--all]
"""
import argparse
import json
import statistics
from pathlib import Path

from PIL import Image

from evaluate_dual import source_boxes_canvas, target_person_boxes
from src.pose_edit.appearance import crop_person, judge_import, load_embedder
from src.pose_edit.common import read_json

ROOT = Path(__file__).resolve().parent
SHAPES = ('tight', 'square')


def run_paths(run_dir):
    geometry = read_json(run_dir / 'geometry' / 'dual_geometry.json')
    pair_dir = Path(geometry['pair_dir'])
    if not pair_dir.is_absolute():
        pair_dir = (ROOT / pair_dir).resolve()
    return pair_dir, read_json(pair_dir / 'pair.json')


def bodies_by_shape(run_dir, pair, preprocess, embed, shapes=SHAPES):
    with Image.open(run_dir / 'inputs' / 'A_reference.png') as image:
        reference = image.convert('RGB')
    with Image.open(run_dir / 'generation' / 'final_image.png') as image:
        generated = image.convert('RGB')
    with Image.open(Path(pair_dir_for(pair)) / pair['B']['image']) as image:
        target = image.convert('RGB')
    evaluation = read_json(run_dir / 'evaluation' / 'eval_dual.json')
    boxes = [one['box_xyxy'] for one in evaluation.get('detected_boxes', [])]
    out = {}
    for shape in shapes:
        source = [crop_person(reference, box, shape=shape) for box in
            source_boxes_canvas(pair, preprocess)]
        targets = [crop_person(target, box, shape=shape) for box in target_person_boxes(pair)]
        bodies = [crop_person(generated, box, shape=shape) for box in boxes]
        if not (all(source) and all(targets) and bodies and all(bodies)):
            out[shape] = None
            continue
        out[shape] = judge_import(embed, source, targets, bodies)
    return out


def pair_dir_for(pair):
    for base in ('pairs', 'pairs_cross'):
        candidate = ROOT / base / pair['name']
        if (candidate / 'pair.json').is_file():
            return candidate
    raise FileNotFoundError(f"no pair directory for {pair['name']}")


def collect(values):
    diffs = [body['target_minus_source'] for block in values if block for body in block['bodies']]
    flagged = sum(1 for block in values if block
        for body in block['bodies'] if body['looks_more_like_target'])
    return diffs, flagged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='*', default=None)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--out', type=Path, default=ROOT / 'logs_batch' / 'import_crop_probe.json')
    args = parser.parse_args()

    config = read_json(args.config)
    embed, error = load_embedder(config['dinov2_repo'])
    if embed is None:
        raise SystemExit(f'DINOv2 unavailable: {error}')

    if args.runs:
        names = args.runs
    else:
        names = sorted(p.parent.parent.name for p in (ROOT / 'runs').glob('*/evaluation/eval_dual.json'))
    if args.limit:
        names = names[:args.limit]

    groups = {}
    for name in names:
        run_dir = ROOT / 'runs' / name
        if not (run_dir / 'generation' / 'final_image.png').is_file():
            continue
        pair_dir, pair = run_paths(run_dir)
        if not (pair_dir / pair['B']['image']).is_file():
            continue
        preprocess = read_json(run_dir / 'inputs' / 'source_preprocess.json')['A']
        blocks = bodies_by_shape(run_dir, pair, preprocess, embed)
        kind = 'cross' if pair.get('kind') != 'same_identity' else 'same'
        for shape, block in blocks.items():
            groups.setdefault((kind, shape), []).append(block)
        print(f'read {name}', flush=True)

    report = {}
    print(f'\n{"kind":6} {"crop":7} {"bodies":>7} {"median":>8} {"mean":>8} {"share>0.02":>11}')
    for (kind, shape), values in sorted(groups.items()):
        diffs, flagged = collect(values)
        if not diffs:
            continue
        report[f'{kind}/{shape}'] = {'bodies': len(diffs), 'median': statistics.median(diffs),
            'mean': sum(diffs) / len(diffs), 'flagged': flagged,
            'share_flagged': flagged / len(diffs)}
        row = report[f'{kind}/{shape}']
        print(f'{kind:6} {shape:7} {row["bodies"]:7d} {row["median"]:+8.3f} {row["mean"]:+8.3f} '
            f'{row["share_flagged"]:10.0%}')
    print('\nreading: the control group is `same` - if its median does not move toward 0 under the '
        'square crop, the bias is not crop geometry and the flag stays withdrawn. If it does move, '
        'the metric still needs the cross-pair value reported as a difference against the control.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'written {args.out}')


if __name__ == '__main__':
    main()
