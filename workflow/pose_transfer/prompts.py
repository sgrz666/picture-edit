from __future__ import annotations

import json
import re

PE_SAMPLING = {'max_new_tokens': 24000, 'do_sample': True, 'temperature': 1.0,
    'top_p': 0.95, 'top_k': 20}
IMAGE_REFERENCE = re.compile(r'<image[_\s]*(\d+)>|\bimage[_\s]*(\d+)\b', re.I)

PE_ROLE_SYSTEM_SUFFIX = (
    '\n\nWORKFLOW-SPECIFIC OUTPUT REFERENCE CONTRACT. This takes precedence over '
    'the general multi-image reference rules above for the final rewritten_prompt only. '
    'You receive three images and may refer to all three in your internal reasoning. '
    'The downstream generator will receive only the source and skeleton. Therefore the '
    'final rewritten_prompt MUST NOT contain <image2> or any reference to the target '
    'photograph, even in preservation or negative object clauses. Translate everything '
    'needed from the target photograph into explicit body, head, gaze, and hand action '
    'in words. Refer only to the source as <image1> and the skeleton as <image3>. '
    'The workflow compiler will map <image3> to downstream <image2>. Keep the official '
    'three-field JSON contract, wh_ratio="", ratio_follow="<image1>". '
    'Exclude target-only props in plain words, without citing their source image.'
)


def pe_instruction(prop_policy='source_only'):
    if prop_policy != 'source_only':
        raise ValueError('This workflow supports source-only appearance and objects')
    return (
        'Write a precise English editing instruction for single-person pose transfer. '
        '<image1> is the source photograph: retain its person identity, face, hairstyle, glasses, '
        'clothing design, colors, material, accessories, shoes and photographic scene. '
        '<image2> is the target photograph: use ONLY its body action, head orientation, '
        'observable hand configuration and facial expression. Never copy its identity, hair, '
        'clothes, jewelry, colors, background or objects. '
        '<image3> is a DWPose-style 2D projection of an estimated 3D SMPL-X body, '
        'already scaled and positioned for the source canvas; black regions are not a scene. '
        'Use this guide for the visible body joint arrangement. B RGB may clarify head direction '
        'and hands when the estimated guide is inaccurate; do not invent invisible details. '
        'The task transfers pose ONLY. The target basketball must NOT appear. Describe the '
        'raised/turned hand and its orientation. Do not introduce any target-only object or '
        'a replacement for it. Preserve objects already in the source; if the source has '
        'no held objects, keep the transferred hand empty. '
        'Organize the rewritten instruction around source appearance preservation, explicit '
        'target body/head/hand action, skeletal reference, original photographic style, and '
        'the actual edit. Avoid guessing ethnicity, age, brands or hidden anatomy. '
        'Describe arm and hand placement using viewer image-left/image-right only; '
        'do not assign anatomical left/right labels, which can conflict with image directions. '
        'Do not enumerate the target clothes or other target appearance in negative clauses; '
        'preserve source appearance without bringing those target details into the instruction. '
        'The downstream editor receives ONLY the source and skeleton, so state the desired '
        'action explicitly in words. In rewritten_prompt, refer to the source as <image1> and '
        'the skeleton as <image3>; do NOT refer to the target photograph as <image2>. '
        'Do not render skeleton lines or their colors. Keep the source setting and overall '
        'illumination, allowing natural cloth folds and pose-dependent shadows. '
        'Return the official JSON contract with rewritten_prompt, wh_ratio="", '
        'ratio_follow="<image1>". Do not change the canvas or introduce a new style.'
    )


def parse_pe_answer(raw):
    answer = raw.split('</think>')[-1].strip()
    if answer.startswith('```'):
        answer = re.sub(r'^```(?:json)?\s*|\s*```$', '', answer, flags=re.I)
    start = answer.find('{')
    if start < 0:
        raise ValueError('PE returned no JSON answer')
    payload = answer[start:]
    result, end = json.JSONDecoder().raw_decode(payload)
    if payload[end:].strip() or not isinstance(result, dict):
        raise ValueError('PE returned malformed JSON or trailing content')
    text = result.get('rewritten_prompt')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('PE returned an empty rewritten_prompt')
    if result.get('ratio_follow') != '<image1>' or result.get('wh_ratio', '') != '':
        raise ValueError('Pose transfer must preserve the source image canvas')
    return result


def validate_generator_prompt(text):
    """Reject references the two-image generator cannot resolve.

    The generator only ever sees the source and the pose guide, so a prompt that names a
    third image, or that names no image at all, is a contract break rather than a style
    choice. This guards both compiled PE output and a hand-supplied --prompt-file.
    """
    indices = {int(match.group(1) or match.group(2))
        for match in IMAGE_REFERENCE.finditer(text)}
    unknown = sorted(index for index in indices if index not in (1, 2))
    if unknown:
        raise ValueError(f'Generator prompt refers to image{unknown[0]}, but only the source '
            '(<image1>) and the pose guide (<image2>) exist downstream')
    if not indices:
        raise ValueError('Generator prompt refers to neither the source nor the pose guide')
    return text


def compile_prompt(text, prop_policy='source_only'):
    if prop_policy != 'source_only':
        raise ValueError('Unsupported object policy')

    def replace(match):
        index = int(match.group(1) or match.group(2))
        if index == 1:
            return '<image1>'
        if index == 2:
            raise ValueError('PE prompt refers to B RGB, which the generator cannot see; '
                'describe the target body/head/hand action explicitly instead')
        if index == 3:
            return '<image2>'
        raise ValueError(f'Unknown PE image reference {index}')

    compiled = IMAGE_REFERENCE.sub(replace, text).strip()
    if not compiled or re.search(r'<image(?![12]>)[^>]*>', compiled, re.I):
        raise ValueError('Unresolved image reference')
    contract = (
        'Edit the single person in <image1> into the desired pose. '
        '<image1> supplies all identity, hairstyle, face, outfit, accessories and scene appearance. '
        '<image2> is only a colored skeletal pose guide; use its joint positions, '
        'never its line colors as appearance. '
        'Do not add the basketball or any other target-only object. '
        'Preserve objects already in the source. '
    )
    return validate_generator_prompt(contract + compiled + (
        ' Preserve the source person and their original clothing and setting. '
        'Render a natural photograph without skeleton lines, diagrams or pose-guide colors.'
    ))
