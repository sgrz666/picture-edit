"""Declared inference ablations; never change checkpoint architecture."""
from dataclasses import asdict
import re

VARIANTS = ('full', 'body_interaction', 'no_source_detail', 'body_only', 'no_hand', 'no_face')
DIAGNOSTICS = ('none', 'photo_prompt', 'monochrome_guides')
PHOTO_SUFFIX = (' Images 3, 4 and 5 are pose-only geometric references. Produce a natural photograph. '
                'Do not draw skeleton lines, colored contact dots, diagram marks, overlays or text on the output.')


def control_records(variant):
    from src.pose_control.qwen21_full.native_v5.contracts import NativeControlOptionsV5
    from src.pose_control.qwen21_full.native_dual_depth import DualControlOptions
    if variant not in VARIANTS:
        raise ValueError('Unknown Native variant: ' + variant)
    controls = asdict(NativeControlOptionsV5())
    base = asdict(DualControlOptions())
    if variant in ('body_interaction', 'body_only'):
        controls.update(face=False, hand=False, detail=False)
    if variant == 'body_only':
        controls['interaction'] = False
        base.update(interaction=False, raster=False, part_contact=False)
    if variant == 'no_source_detail':
        controls['detail'] = False
    if variant == 'no_hand':
        controls['hand'] = False
    if variant == 'no_face':
        controls['face'] = False
    return controls, base


def generation_key(mode, variant='full', tag=None):
    if variant not in VARIANTS:
        raise ValueError('Unknown Native variant: ' + variant)
    if mode != 'native_v5' and variant != 'full':
        raise ValueError('Native variants require --mode native_v5')
    tag = tag or (variant if variant != 'full' else None)
    if tag is not None and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', tag):
        raise ValueError('Use a simple --generation-tag without path separators')
    return mode if tag is None else mode + '__' + tag
