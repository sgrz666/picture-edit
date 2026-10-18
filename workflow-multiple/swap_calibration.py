#!/usr/bin/env python3
"""Calibrate the swap threshold against the observed margin distribution.

`target_import` taught the lesson twice now: a flag is only as good as the control behind it. The
swap rule is "appearance pairing disagrees with pose pairing by more than 0.02 per person". That
number was chosen to keep float noise from being reported as a swap, not measured. This prints the
margin distribution for flagged and unflagged runs so the README can say where 0.02 sits: below a
gap, or inside a continuum.
"""
import json
import glob
from pathlib import Path

import numpy as np

rows = []
for path in sorted(glob.glob('runs/*/evaluation/eval_dual.json')):
    name = Path(path).parent.parent.name
    parts = name.split('.')
    if len(parts) < 3:
        continue
    j = json.load(open(path, encoding='utf-8'))
    a = j.get('appearance') or {}
    matrix = a.get('similarity_matrix')
    if not matrix or len(matrix) < 2:
        continue
    m = np.array(matrix, dtype=float)
    if m.shape[0] != m.shape[1]:
        continue
    diagonal = np.mean([m[i, i] for i in range(m.shape[0])])
    off = [m[i, k] for i in range(m.shape[0]) for k in range(m.shape[1]) if k != i]
    if not off:
        continue
    rows.append((name, a.get('identity_swap_suspected') == 1, diagonal, float(np.mean(off)),
        a.get('swap_margin_per_person'), float(np.mean(off) - diagonal)))

print(f'{len(rows)} runs carry a square similarity matrix\n')
flagged = [r for r in rows if r[1]]
clean = [r for r in rows if not r[1]]
for label, group in (('flagged as swap', flagged), ('not flagged', clean)):
    margins = sorted(r[5] for r in group)
    stored = sorted(r[4] for r in group if r[4] is not None)
    if not margins:
        print(f'{label:14} none')
        continue
    print(f'{label:14} n={len(margins):2}  off-diag minus diag: min={margins[0]:+.3f} '
        f'median={margins[len(margins)//2]:+.3f} max={margins[-1]:+.3f}')
    if stored:
        print(f'{"":14} stored swap_margin_per_person: min={stored[0]:.3f} '
            f'median={stored[len(stored)//2]:.3f} max={stored[-1]:.3f}')

all_margins = sorted(r[5] for r in rows)
print('\nfull distribution of (off-diagonal mean - diagonal mean), ascending:')
print('  ' + ' '.join(f'{value:+.3f}' for value in all_margins))
above = [value for value in all_margins if value > 0.02]
print(f'\nthreshold 0.02: {len(above)} of {len(all_margins)} runs sit above it '
    f'({len(flagged)} flagged)')
print('the stored per-person margin is the disagreement after the appearance-optimal assignment, '
    'so compare the two lists rather than reading one as the truth')
print('\nnearest cases to the boundary (decisions that would flip if 0.02 moved):')
near = sorted(rows, key=lambda r: abs((r[4] or 0.0) - 0.02))[:6]
for name, flag, diagonal, off, stored, margin in near:
    print(f'  {name[:56]:56} flag={flag!s:5} stored_margin='
        f'{"-" if stored is None else f"{stored:.3f}"} off_minus_diag={margin:+.3f}')
