"""Auditable detailed prompts for the inference-only Qwen value comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from scripts.eval.qwen21_condition_combinations import NEW_METHODS, combination_prompt

METHODS = (
    'native_ab', 'native_ab_crop', 'native_skeleton', 'native_geometry',
    'simple_body', 'full_body', 'full_body_face', 'full_body_native_face',
    'simple_body1024',
)
LEGACY_METHODS = METHODS
METHODS = LEGACY_METHODS + NEW_METHODS


def build_prompt(row: dict, method: str, variant: str = 'structured') -> str:
    if method not in METHODS or variant not in ('structured', 'narrative'):
        raise ValueError('Unknown comparison method or prompt variant')
    if method in NEW_METHODS:
        if variant != 'structured':
            raise ValueError('Supplementary combinations use the locked structured template')
        return combination_prompt(build_prompt(row, 'native_skeleton', 'structured'), method)
    action = row['action_description'].strip()
    preserve = row['source_preserve_description'].strip()
    if not action or not preserve or '[' in action or '[' in preserve:
        raise ValueError('Prompts require a visually checked action and A appearance description')
    if method == 'full_body_native_face':
        return (
            'Edit image 1, which is the already posed result photograph, to restore the facial '
            'identity of person A shown in image 2. Image 2 is a face crop of A from the original '
            'source photograph and is an identity reference only, not a pose or lighting reference. '
            f'Person A: {preserve} '
            'Make the face recognizable as A, retaining A\'s facial structure, skin appearance, '
            'hairline and glasses if present, with natural skin texture and coherent eyes, nose and mouth. '
            'Preserve the head position, head rotation, gaze direction and expression already present '
            'in image 1. Do not copy the head rotation or expression of the cropped reference. '
            'Keep image 1\'s entire body pose, body proportions, clothing, footwear, background, '
            'camera framing and lighting. Blend any face changes naturally into the existing head '
            'and scene without colored patches or visible seams. Return one edited photograph, '
            'not a collage; do not add another person, text or objects.'
        )
    roles = {
        'native_ab': (
            'Image 1 is the source photograph of person A and the scene to edit. '
            'Image 2 shows a different person B and is ONLY a reference for the target pose and head orientation.'
        ),
        'native_ab_crop': (
            'Image 1 is the source photograph of person A and the scene to edit. '
            'Image 2 shows a different person B and is ONLY a reference for the target pose and head orientation. '
            'Image 3 is a crop of the SAME person A\'s face from image 1. Use image 3 only to preserve A\'s '
            'identity and facial appearance, not to copy A\'s original head orientation.'
        ),
        'native_skeleton': (
            'Image 1 is the source photograph of person A and the scene to edit. '
            'Image 2 is a colored skeleton guide for the target pose. Its dots and lines represent '
            'body joints and limb directions, not a person\'s appearance or the colors of the output.'
        ),
        'native_geometry': (
            'Image 1 is the source photograph of person A and the scene to edit. '
            'Image 2 is the surface-normal guide and image 3 is the depth guide for the target pose. '
            'These two guides describe body articulation and orientation only. Their colors are '
            'geometric encodings, not skin colors, clothing or a background to reproduce.'
        ),
    }.get(method, (
        'Image 1 is the source photograph of person A and the scene to edit. '
        'The target body and head geometry is supplied through the control branch; the requested '
        'articulation is also described below. Person A is the only identity to generate.'
    ))
    retain = (
        f'Retain person A from image 1: {preserve} '
        'Keep A\'s recognizable facial features, hairstyle, skin appearance, glasses if present, '
        'clothing, footwear, height and body proportions. Adapt the requested articulation to A\'s '
        'own proportions rather than replacing A with the reference person\'s body shape. '
        'Preserve image 1\'s background, camera view and lighting. Do not transfer any reference '
        'person\'s identity, hairstyle, clothing, body shape, objects or background.'
    )
    pose = (
        f'Repose A to the following target articulation: {action} '
        'Follow the visible torso, arm and leg arrangement and the target head direction, rather '
        'than keeping A\'s original pose. When the description says image-left or image-right, '
        'it refers to the viewer\'s sides of the pose reference.'
    )
    result = (
        'Return a single natural edited photograph of A in the original scene. Generate exactly '
        'one person, not the reference person or a second copy. Do not output a collage or the '
        'condition guides. Keep occlusions anatomically plausible, avoid invented limbs and '
        'colored patches, and reconstruct newly uncovered areas consistently with the source scene.'
    )
    if variant == 'structured':
        return '\n\n'.join(('IMAGE ROLES:\n' + roles, 'TARGET POSE:\n' + pose,
                            'APPEARANCE AND SCENE:\n' + retain, 'OUTPUT:\n' + result))
    return ' '.join((roles, pose, retain, result))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--variant', choices=('structured', 'narrative', 'both'), default='structured')
    p.add_argument('--methods', nargs='+', choices=METHODS, default=list(LEGACY_METHODS))
    a = p.parse_args()
    rows = [json.loads(line) for line in a.pairs.read_text().splitlines() if line.strip()]
    variants = ('structured', 'narrative') if a.variant == 'both' else (a.variant,)
    prompts = {v: {row['pair_id']: {m: build_prompt(row, m, v) for m in a.methods} for row in rows}
               for v in variants}
    value = prompts if a.variant == 'both' else prompts[a.variant]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists() and json.loads(a.output.read_text()) != value:
        raise ValueError('Refusing to overwrite different, potentially locked prompts')
    a.output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'pairs': len(rows), 'variants': list(variants), 'output': str(a.output)}))


if __name__ == '__main__':
    main()
