"""The four-image role contract for the two-person workflow.

The prompt engineer always sees all four images: the source photo, the target interaction
photo, and one skeletal guide per target person. What the *generator* sees is the experiment:

* ``with_b``        - the generator sees the same four images, so no reference rewriting is
  needed and the target photograph is available as an appearance risk.
* ``skeleton_only`` - the generator sees three images (source and the two guides). B is
  dropped, so every clause that leaned on it has to be rewritten into explicit body words
  and the guide numbers shift down by one.
"""
from __future__ import annotations

import re

from src.pose_edit.prompt_contract import (IMAGE_REFERENCE, PE_SAMPLING, parse_pe_answer,
    referenced_images, validate_generator_prompt)

PE_ROLES = ('A_reference', 'B_reference', 'skeleton_1', 'skeleton_2')
GENERATOR_ROLES = {'with_b': PE_ROLES,
    'skeleton_only': ('A_reference', 'skeleton_1', 'skeleton_2')}
# PE image number -> generator image number, per arm.
ROLE_MAP = {'with_b': {1: 1, 2: 2, 3: 3, 4: 4}, 'skeleton_only': {1: 1, 3: 2, 4: 3}}


def generator_allowed(arm):
    """The image numbers the generator of this arm can actually see."""
    return set(ROLE_MAP[arm].values())


def citation_rule(arm):
    """One sentence stating which PE numbers the answer may cite.

    PE always numbers images as it received them; only ``compile_prompt`` renumbers for the
    editor. Letting the model renumber itself was what made a correct answer look like a
    contract violation.
    """
    if arm not in ROLE_MAP:
        raise ValueError(f'unknown generator arm {arm!r}')
    if arm == 'with_b':
        return ('In rewritten_prompt you may cite <image1>, <image2>, <image3> and <image4> '
            'freely.')
    return ('In rewritten_prompt you may cite ONLY <image1> (the source photograph), <image3> '
        'and <image4> (the two skeletal guides). Do NOT cite <image2> anywhere in '
        'rewritten_prompt: state each body action, the distance between the two people and '
        'where they touch in explicit words. Keep the numbers exactly as you received them; the '
        'workflow renumbers the guides for the editor afterwards.')


DUAL_ROLE_SYSTEM_SUFFIX = (
    '\n\nWORKFLOW-SPECIFIC OUTPUT REFERENCE CONTRACT. This takes precedence over the general '
    'multi-image reference rules above for the final rewritten_prompt only. You receive four '
    'images and must number them as you received them: <image1> the source photograph of two '
    'people, <image2> the target photograph of two people interacting, <image3> and <image4> a '
    'skeletal guide of each person in <image2> respectively. Never renumber images yourself; the '
    'workflow rewrites the numbers for the editor afterwards. The user instruction states which '
    'of these four numbers you may cite in the final rewritten_prompt. Anything you would else '
    'take from a number you may not cite must be written as explicit words about bodies, limbs, '
    'head direction, gaze, distance and contact. Keep the official three-field JSON contract '
    'with wh_ratio="" and ratio_follow="<image1>". Never ask for a third person, never copy the '
    'target people\'s faces, hair, clothing or background, and never ask to draw the skeleton.'
)


def pe_instruction(arm, prop_policy='source_only'):
    if prop_policy != 'source_only':
        raise ValueError('This workflow supports source-only appearance and objects')
    if arm not in ROLE_MAP:
        raise ValueError(f'unknown generator arm {arm!r}')
    visible = citation_rule(arm)
    guide_note = ('<image3> and <image4> are 2D projections of an estimated 3D body for each '
        'person in <image2>, already placed in that photograph\'s frame. Black regions in them '
        'are not a scene, and their colours carry no meaning. Use them for joint arrangement; '
        'the target photograph may clarify head direction and hands where a guide is ambiguous.'
        if arm == 'with_b' else
        '<image3> and <image4> are 2D projections of an estimated 3D body for each person in '
        'the target photograph. Black regions in them are not a scene and their colours carry '
        'no meaning. Describe the interaction in words because the target photograph will not '
        'reach the editor.')
    return (
        'Write a precise English editing instruction for two-person interaction transfer. '
        '<image1> is the source photograph with exactly two people: retain both people\'s '
        'identity, face, hairstyle, body proportions, clothing design, colours, material, '
        'accessories, shoes and the photographic scene, including other objects already there. '
        '<image2> is the target photograph of two people interacting: use ONLY their body '
        'actions, the arrangement between them, their relative distance and height, contact and '
        'leaning, head orientation and facial expression. Never copy its identities, hair, '
        'clothes, jewellery, colours, background or held objects. '
        + guide_note + ' '
        'The edit changes pose and interpersonal arrangement only. Keep exactly two people, one '
        'per source person; do not add, remove or merge anyone. Preserve objects already in the '
        'source and do not introduce target-only objects or a replacement for them. '
        'Organise the instruction around source appearance preservation, the explicit action of '
        'each of the two people, how they relate to each other, the skeletal guides, the '
        'original photographic style, and the actual edit. Describe limbs using viewer '
        'image-left/image-right only; do not assign anatomical left/right labels, which can '
        'conflict with image directions. Do not enumerate the target clothing or other target '
        'appearance in negative clauses. ' + visible + ' '
        'In rewritten_prompt, number images exactly as the editor will receive them. Do not '
        'render skeleton lines or their colours. Keep the source setting and overall '
        'illumination, allowing natural cloth folds and pose-dependent shadows. Return the '
        'official JSON contract with rewritten_prompt, wh_ratio="", ratio_follow="<image1>". '
        'Do not change the canvas or introduce a new style.'
    )


def compile_prompt(text, arm):
    """Renumber references for the arm and refuse anything the editor cannot see."""
    if arm not in ROLE_MAP:
        raise ValueError(f'unknown generator arm {arm!r}')
    mapping = ROLE_MAP[arm]

    def replace(match):
        index = int(match.group(1) or match.group(2))
        if index not in mapping:
            if arm == 'skeleton_only' and index == 2:
                raise ValueError('The prompt refers to the target photograph, which the '
                    'skeleton_only generator cannot see; describe each person\'s body action, '
                    'their distance and contact in explicit words instead')
            raise ValueError(f'Unknown PE image reference {index}')
        return f'<image{mapping[index]}>'

    compiled = IMAGE_REFERENCE.sub(replace, text).strip()
    if not compiled or re.search(r'<image(?![1-4]>)[^>]*>', compiled, re.I):
        raise ValueError('Unresolved image reference')
    contract = ('Edit exactly the two people in <image1> into the requested interaction. '
        '<image1> supplies all identity, hairstyle, face, body proportions, outfit, '
        'accessories and scene appearance for both of them. '
        + ('<image2> is the target photograph and is shown for the interaction only; copy no '
            'appearance from it. ' if arm == 'with_b' else '')
        + ('The remaining images are colored skeletal pose guides, one per person: use their '
            'joint positions, never their line colours as appearance. '
            if arm == 'with_b' else
            '<image2> and <image3> are colored skeletal pose guides, one per person: use their '
            'joint positions, never their line colours as appearance. ')
        + 'Do not add a third person or any target-only object. Preserve objects already in the '
        'source. ')
    result = contract + compiled + (
        ' Preserve both source people, their clothing and the original setting. Render a natural '
        'photograph without skeleton lines, diagrams or pose-guide colors.')
    return validate_generator_prompt(result, allowed=generator_allowed(arm))


__all__ = ['PE_SAMPLING', 'PE_ROLES', 'GENERATOR_ROLES', 'DUAL_ROLE_SYSTEM_SUFFIX',
    'parse_pe_answer', 'pe_instruction', 'compile_prompt', 'generator_allowed',
    'referenced_images', 'validate_generator_prompt']
