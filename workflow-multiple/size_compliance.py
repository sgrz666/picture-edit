#!/usr/bin/env python3
"""Size compliance: does the generator draw each person at the torso the guide asked for?

``eval_dual.json`` already stores, per detected person, the torso it was asked to draw
(``requested_torso_px``, from the guide geometry) and the torso measured back out of the generated
image (``detected_torso_px``). Their ratio is the generator's obedience to the requested scale, and
it is the measurement that decides whether a sizing complaint belongs to the generator or to the
geometry it was handed. Reading only those two fields costs no inference, so this runs while a
batch holds the GPU.

Grouping follows the run name convention ``<pair>.<arm>.<framing>[.fit]``; anything not matching is
skipped rather than guessed at.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def group_of(run_name):
    """(arm, framing, guide) from a run name, or None when the name is not a matrix cell."""
    parts = run_name.split('.')
    if len(parts) < 3:
        return None
    guide = 'fit' if len(parts) > 3 and parts[3] == 'fit' else 'dwpose'
    return (parts[1], parts[2], guide)


def compliance_groups(runs_root):
    """detected / requested torso per generated person, keyed by arm / framing / guide."""
    groups = defaultdict(list)
    for path in sorted(Path(runs_root).glob('*/evaluation/eval_dual.json')):
        key = group_of(path.parent.parent.name)
        if key is None:
            continue
        for person in json.loads(path.read_text()).get('persons', []):
            requested, detected = person.get('requested_torso_px'), person.get('detected_torso_px')
            if requested and detected:
                groups[key].append(float(detected) / float(requested))
    return groups


def median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2 if ordered else None


def summarise(groups):
    summary = {}
    for key, raw in sorted(groups.items()):
        values = sorted(raw)
        summary['/'.join(key)] = {'people': len(values), 'median': median(values),
            'p25': values[len(values) // 4], 'p75': values[(3 * len(values)) // 4],
            'min': values[0], 'max': values[-1],
            'within_5_percent': sum(1 for value in values if abs(value - 1.0) <= 0.05) / len(values),
            'within_10_percent': sum(1 for value in values if abs(value - 1.0) <= 0.10) / len(values)}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT / 'runs')
    parser.add_argument('--out', type=Path, default=ROOT / 'logs_batch' / 'size_compliance.json')
    args = parser.parse_args()

    summary = summarise(compliance_groups(args.runs))
    for group, block in summary.items():
        print(f'SIZE_COMPLIANCE {group:34s} people={block["people"]:3d} median={block["median"]:.3f} '
            f'IQR={block["p25"]:.3f}-{block["p75"]:.3f} '
            f'range={block["min"]:.3f}-{block["max"]:.3f} '
            f'within_5pct={block["within_5_percent"]:.2f} within_10pct={block["within_10_percent"]:.2f}')
    if not summary:
        print(f'SIZE_COMPLIANCE nothing to read under {args.runs}')
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'written {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
