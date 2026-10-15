#!/usr/bin/env python3
"""Build blind pairwise annotation packets for the pose-transfer workflows.

    python prepare.py --pair-dir ../../workflow-multiple/runs \
        --a-arm skeleton_only.b_canvas --b-arm with_b.b_canvas \
        --image generation/final_image.png --out bundle
    python prepare.py --selfcheck bundle

Only runs that exist on **both** arms are used, so every question is a real paired comparison.
The arm shown on the left is decided by a recorded seed and never written to index.json;
truth.json is for the tally step only and must not be handed to annotators.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def arm_of(run_dir, pattern):
    """Match arms like `<pair>.<arm>` or `<pair>_<arm>` against the directory name."""
    name = Path(run_dir).name
    return pattern if pattern in name else None


def collect_runs(root, image, wanted=None):
    packets = {}
    for manifest in sorted(Path(root).rglob('run.json')):
        run_dir = manifest.parent
        picture = run_dir / image
        if not picture.is_file():
            continue
        manifest_data = read_json(manifest) or {}
        settings = manifest_data.get('settings') or {}
        packets[run_dir.name] = {
            'run': run_dir.name, 'dir': str(run_dir), 'image': str(picture),
            'image_sha256': hashlib.sha256(picture.read_bytes()).hexdigest(),
            'pair': (Path(manifest_data.get('A_original') or '').parent.name or run_dir.name),
            'resolution': settings.get('resolution'), 'seed': settings.get('seed'),
            'steps': settings.get('steps'),
            'status': manifest_data.get('status'),
        }
    if wanted:
        packets = {key: value for key, value in packets.items() if any(w in key for w in wanted)}
    return packets


def split_run(name):
    """Run names are `<pair>.<arm>.<framing>[.fit]`; legacy names have no dot and no arm."""
    head, _, tail = name.partition('.')
    return head, tail


def choose(candidates, pattern):
    """Exact arm suffix wins; an unnamed runner-up is fine; three of them is not a question."""
    exact = [item for item in candidates if item['tail'] == pattern]
    if exact:
        return exact[0], []
    if len(candidates) == 1:
        return candidates[0], []
    return None, sorted(item['tail'] for item in candidates)


def build(root, a_pattern, b_pattern, image, out, seed, limit):
    packets = collect_runs(root, image)
    by_pair = {}
    for name, item in packets.items():
        pair, tail = split_run(name)
        item['tail'] = tail
        by_pair.setdefault(pair, []).append(item)

    questions, ambiguous = [], []
    for pair in sorted(by_pair):
        a_side, a_extra = choose([item for item in by_pair[pair] if item['tail'].startswith(a_pattern)],
            a_pattern)
        b_side, b_extra = choose([item for item in by_pair[pair] if item['tail'].startswith(b_pattern)],
            b_pattern)
        if a_extra or b_extra:
            ambiguous.append(f'{pair}: a={a_extra or "-"} b={b_extra or "-"}')
        if a_side and b_side:
            questions.append({'key': pair, 'a': a_side, 'b': b_side})
    rng = random.Random(seed)
    rng.shuffle(questions)
    if limit:
        questions = questions[:limit]

    out.mkdir(parents=True, exist_ok=True)
    (out / 'img').mkdir(exist_ok=True)
    index, truth = [], []
    for number, question in enumerate(questions, 1):
        show_a_first = rng.random() < 0.5
        left, right = (question['a'], question['b']) if show_a_first else (question['b'], question['a'])
        files = {}
        for side, item in (('left', left), ('right', right)):
            suffix = item['image_sha256'][:12]
            filename = f'q{number:03d}_{side}_{suffix}.png'
            target = out / 'img' / filename
            if not target.is_file() and not target.is_symlink():
                try:
                    target.symlink_to(Path(item['image']).resolve())
                except OSError:
                    shutil.copy2(item['image'], target)
            files[side] = filename
        index.append({'id': f'q{number:03d}', 'pair': question['key'],
            'left': files['left'], 'right': files['right'],
            'left_sha256': left['image_sha256'], 'right_sha256': right['image_sha256']})
        truth.append({'id': f'q{number:03d}', 'pair': question['key'],
            'reference': a_pattern, 'candidate': b_pattern,
            'left_arm': a_pattern if show_a_first else b_pattern,
            'right_arm': b_pattern if show_a_first else a_pattern,
            'runs': {'a': question['a']['dir'], 'b': question['b']['dir'],
                'a_sha256': question['a']['image_sha256'],
                'b_sha256': question['b']['image_sha256']}})

    (out / 'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'questions.js').write_text(
        'window.QUESTIONS = ' + json.dumps(index, ensure_ascii=False) + ';\n', encoding='utf-8')
    (out / 'truth.json').write_text(json.dumps({'seed': seed, 'reference_arm': a_pattern,
        'candidate_arm': b_pattern, 'items': truth}, ensure_ascii=False, indent=2), encoding='utf-8')
    unpaired = sorted(set(by_pair) - {question['key'] for question in questions})
    print(f'BUNDLE {out} questions={len(questions)} unpaired_skipped={len(unpaired)} seed={seed}')
    if ambiguous:
        print('AMBIGUOUS narrow --a-arm/--b-arm to one of the listed suffixes: '
              + '; '.join(ambiguous))
    if unpaired:
        print('UNPAIRED (only one arm has an image, not usable as a question): '
              + ', '.join(unpaired))
    return questions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair-dir', type=Path, required=False, help='directory holding run dirs')
    parser.add_argument('--a-arm', default='skeleton_only', help='reference arm substring')
    parser.add_argument('--b-arm', default='with_b', help='candidate arm substring')
    parser.add_argument('--image', default='generation/final_image.png')
    parser.add_argument('--out', type=Path, default=Path('bundle'))
    parser.add_argument('--seed', type=int, default=20261014)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--selfcheck', type=Path, default=None,
        help='validate an existing bundle without copying images')
    args = parser.parse_args()

    if args.selfcheck:
        index = json.loads((args.selfcheck / 'index.json').read_text(encoding='utf-8'))
        truth = json.loads((args.selfcheck / 'truth.json').read_text(encoding='utf-8'))['items']
        left_arms = [item['left_arm'] for item in truth]
        diff = abs(left_arms.count(args.a_arm) - left_arms.count(args.b_arm))
        spread = max(2, len(truth) ** 0.5)  # a fair coin on 12 questions lands near 6+-3.5
        print(f'SELFCHECK questions={len(index)} truth={len(truth)} '
              f'left_arm_balance={left_arms.count(args.a_arm)}/{left_arms.count(args.b_arm)} '
              f'imbalanced={diff > spread} (tolerance +-{spread:.1f})')
        missing = [item['id'] for item in index
            if not (args.selfcheck / 'img' / item['left']).is_file()
            or not (args.selfcheck / 'img' / item['right']).is_file()]
        print(f'missing_images={missing or "none"}')
        return 0 if not missing else 1
    if not args.pair_dir:
        raise SystemExit('either --pair-dir or --selfcheck is required')
    build(args.pair_dir, args.a_arm, args.b_arm, args.image, args.out, args.seed, args.limit)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
