"""Explicit, ordered native references for the supplementary factorial audit."""
from __future__ import annotations

import json
from pathlib import Path

COMBINATION_REFERENCES = {
    'native_skeleton': ('A', 'B_skeleton'),
    'native_ab_skeleton': ('A', 'B', 'B_skeleton'),
    'native_skeleton_normal': ('A', 'B_skeleton', 'B_normal'),
    'native_skeleton_depth': ('A', 'B_skeleton', 'B_depth'),
    'native_skeleton_normal_depth': ('A', 'B_skeleton', 'B_normal', 'B_depth'),
    'native_ab_skeleton_normal': ('A', 'B', 'B_skeleton', 'B_normal'),
    'native_ab_skeleton_depth': ('A', 'B', 'B_skeleton', 'B_depth'),
    'native_ab_skeleton_normal_depth': ('A', 'B', 'B_skeleton', 'B_normal', 'B_depth'),
}
COMBINATION_METHODS = tuple(COMBINATION_REFERENCES)
NEW_METHODS = COMBINATION_METHODS[1:]
COMBINATION_LABELS = {
    'native_skeleton': '原生 A＋骨架',
    'native_ab_skeleton': '原生 A＋B＋骨架',
    'native_skeleton_normal': '原生 A＋骨架＋法线',
    'native_skeleton_depth': '原生 A＋骨架＋深度',
    'native_skeleton_normal_depth': '原生 A＋骨架＋法线＋深度',
    'native_ab_skeleton_normal': '原生 A＋B＋骨架＋法线',
    'native_ab_skeleton_depth': '原生 A＋B＋骨架＋深度',
    'native_ab_skeleton_normal_depth': '原生 A＋B＋骨架＋法线＋深度',
}
ROLE_DESCRIPTIONS = {
    'A': 'the source photograph of person A and the scene to edit. A is the ONLY identity, clothing, body proportions, background, camera view and lighting to preserve.',
    'B': 'a photograph of a DIFFERENT person B, used ONLY for the target body articulation, hand articulation and head orientation. Do not transfer B\'s face, hair, skin appearance, clothing, body shape or background.',
    'B_skeleton': 'the colored skeleton guide for the target pose. Dots and lines encode joints and limb directions, not a person\'s appearance or output colors. Use it to guide joint placement and the overall limb arrangement.',
    'B_normal': 'the surface-normal guide for the SAME target pose. Its colors encode local surface orientation and shape, NOT skin color, clothing, lighting or output texture. Use it only as auxiliary three-dimensional geometric information; generate A\'s natural skin and clothing.',
    'B_depth': 'the depth guide for the SAME target pose. Its grayscale encodes relative surface distance and front/back relationships, NOT skin tone, clothing, lighting or output texture. Use it only as auxiliary three-dimensional geometric information.',
}


def combination_prompt(baseline: str, method: str) -> str:
    """Replace only IMAGE ROLES; keep all three detailed content blocks exact."""
    if method == 'native_skeleton':
        return baseline
    blocks = baseline.split('\n\n')
    if len(blocks) != 4 or not blocks[0].startswith('IMAGE ROLES:\n'):
        raise ValueError('Expected the locked four-block structured prompt')
    if [x.split(':', 1)[0] for x in blocks[1:]] != ['TARGET POSE', 'APPEARANCE AND SCENE', 'OUTPUT']:
        raise ValueError('Locked content block labels differ')
    recipe = COMBINATION_REFERENCES[method]
    roles = ' '.join(f'Image {i + 1} is {ROLE_DESCRIPTIONS[role]}' for i, role in enumerate(recipe))
    return '\n\n'.join(['IMAGE ROLES:\n' + roles, *blocks[1:]])


def output_key(row):
    return row['pair_id'], int(row['resolution']), int(row['steps']), row['method']


def protocol_expected(path):
    value = json.loads(Path(path).read_text())
    if value.get('architecture') != 'native_condition_combinations_v1':
        raise ValueError('Unsupported supplementary protocol')
    keys = [output_key(row) for row in value['expected_outputs']]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate protocol output keys')
    return set(keys)


def require_protocol_outputs(path, rows):
    actual = [output_key(row) for row in rows]
    if len(actual) != len(set(actual)):
        raise ValueError('Duplicate output keys')
    expected = protocol_expected(path)
    missing, extra = expected - set(actual), set(actual) - expected
    if missing or extra:
        raise ValueError(f'Protocol coverage mismatch: missing={sorted(missing)} extra={sorted(extra)}')
    return expected


def factor_edges():
    """Twelve comparisons, each changing exactly one of B/normal/depth."""
    results = []
    for left, a in COMBINATION_REFERENCES.items():
        for right, b in COMBINATION_REFERENCES.items():
            added = set(b) - set(a)
            if set(a) <= set(b) and len(added) == 1:
                role = next(iter(added))
                if role in ('B', 'B_normal', 'B_depth'):
                    results.append((role, left, right))
    return tuple(results)
