"""Who is on the left, and does the prompt agree with the skeleton fed as <image3>.

The Hi4D dual pipeline names the two people by tracked identity (A/B). The native geometry
cache keeps that order -- slot 0 is identity A everywhere (`native_hybrid_assets.py` stacks
`person_mask_A` then `person_mask_B`). Colour words in the prompts are *not* evidence of that order:
the guide renderer paints a 12-colour limb palette, so red/blue/white describes a limb, never a
person (verified on pixels -- see `IDENTITY_COLOUR` below).
The prompt bank, however, addresses the two people as "person 1 = the person on the left in
<image1>". Those two conventions coincide only when identity A happens to stand on the left,
which is 3,586 of 6,600 pairs; for the other 3,014 the prompt's Person-1 description and the
skeleton fed into <image3> belong to different people.

This module is the single place that decides the order and checks that a condition's text and
its images agree, so neither the assembler nor a future rewrite run can silently reintroduce
the divergence.
"""

from __future__ import annotations

import re

import numpy as np
from PIL import Image

# Below this horizontal separation the phrase "the person on the left" does not identify a
# person, so a condition built from such a frame is refused rather than guessed.
# The cut has to be relative, not absolute: the rewrite worker measures the 512x512 preprocessed
# silhouettes, while the assembler measures Hi4D's raw seg masks (measured 940x1280 on pair12).
# 20 pixels is 3.9% of one width and 2.1% of the other, so the two layers disagreed about exactly
# the borderline pairs -- the bank could flag a pair `side_ambiguous` that the assembler then
# considered resolvable. 20 px on the 512 reference is the cut the bank already applied, so this
# fraction keeps every existing bank verdict and only tightens the raw-scale site to match it.
MIN_RESOLVABLE_GAP_PX = 20.0
MIN_RESOLVABLE_GAP_FRAC = MIN_RESOLVABLE_GAP_PX / 512.0

# Identity -> colour attributed by the bank's own prose. Do NOT read this as what the renderer
# draws: measured pixels show `render_poseguide` paints a 12-colour limb palette on every guide
# (the fed <image2> carries red 4699 / blue 2401 / pale 2299 px on one pilot pair), so a colour
# word in a prompt describes a limb at best and never a person. The table exists only to phrase
# the advisory that quotes the colour the text chose.
IDENTITY_COLOUR = {'A': 'red', 'B': 'blue'}
COLOUR_IDENTITY = {'red': 'A', 'blue': 'B'}

SIDE_PHRASE = re.compile(r'\bon the (left|right)\b'
                         r'|\bthe (left|right) (?:person|individual|figure|man|woman)\b'
                         r'|\bleft person\b|\bright person\b', re.I)
IMAGE3_SPAN = 140

# The prompts describe person 1 first and person 2 second, so the writer's own binding is the
# stretch of text between a person-1/<image3> marker and the next person-2/<image4> marker.
# A fixed character window reached past that boundary and captured person 2's "on the right
# side", which made 843 of 1,208 records look self-contradictory.
PERSON1_MARK = re.compile(r'<image[\s_]*3>|\bperson[\s_]*1\b', re.I)
PERSON2_MARK = re.compile(r'<image[\s_]*4>|\bperson[\s_]*2\b', re.I)


def _side_of(phrase) -> str:
    side = next((g for g in phrase.groups() if g), None)
    if side is None:
        side = 'left' if 'left' in phrase.group(0).lower() else 'right'
    return side.lower()


def prompt_side_claim(prompt: str) -> str:
    """'left' / 'right' / 'none' / 'contradictory' -- what the text says <image3> is.

    For each person-1 marker only the first side phrase inside its own zone counts: taking
    every side word in the sentence would call almost everything contradictory, and taking the
    nearest word inside a fixed window still crossed into the person-2 description.
    """
    text = prompt or ''
    sides = set()
    for ref in PERSON1_MARK.finditer(text):
        following = PERSON2_MARK.search(text, ref.end())
        end = following.start() if following else ref.start() + 2 * IMAGE3_SPAN
        zone = text[ref.start():max(ref.end(), end)]
        phrase = SIDE_PHRASE.search(zone)
        if phrase:
            sides.add(_side_of(phrase))
    if not sides:
        return 'none'
    if len(sides) > 1:
        return 'contradictory'
    return sides.pop()


# The colour has to modify the word "skeleton" directly: an earlier window-based version also
# caught "red-haired woman in <image1>, skeleton in <image3>". Superseded by SLOT_COLOUR_CLAIM,
# which is the only colour vocabulary the judges, the gate and the purge share.
PIXEL_NUMBER = re.compile(r'\b\d+(?:\.\d+)?\s*-?\s*(?:pixels?|px)\b', re.I)
HARD_NEGATION = re.compile(
    r'\bno physical contact\b|\bno hug\b|\bno touch(?:ing)?\b|\bno hold\b'
    r'|\bremain separated\b|\bwithout any contact\b|\bnot touching\b', re.I)
TARGET_IMAGE_REF = re.compile(
    r'target photograph|target reference|reference photograph|ground[ -]?truth'
    r'|unseen target|target image|fifth image|<image[\s_]*5>|\bimage[\s_]*5\b', re.I)


class PersonOrderError(ValueError):
    """Raised when the left/right order cannot be established or the condition is inconsistent."""


def mask_geometry(path):
    """(mean x of the silhouette, image width) or (None, width) when empty/unreadable.

    Hi4D carries masks in two shapes in this tree: the manifests store a plain path string, the
    geometry cache stores `{'path': ..., 'sha256': ...}`. Accepting both keeps a caller from
    opening a dict's repr as a filename.
    """
    if path is None:
        return None, None
    if isinstance(path, dict):
        path = path.get('path')
        if path is None:
            return None, None
    array = np.asarray(Image.open(str(path)).convert('L'))
    _rows, cols = np.nonzero(array > 10)
    if cols.size == 0:
        return None, int(array.shape[1])
    return float(cols.mean()), int(array.shape[1])


def mask_centroid_x(path) -> float | None:
    """Mean x of a person silhouette, or None when the mask is empty or unreadable."""
    return mask_geometry(path)[0]


def measure_source_order(source_mask_A, source_mask_B, *, min_gap: float | None = None) -> dict:
    """Which tracked person stands on the left in the source photograph.

    The separability test is a fraction of the measured frame width, so a caller pointing at
    preprocessed 512-wide masks and a caller pointing at raw 940-wide masks decide the same way.
    Passing `min_gap` keeps the old absolute-pixel behaviour for callers that deliberately want it.
    """
    cx_a, width_a = mask_geometry(source_mask_A)
    cx_b, width_b = mask_geometry(source_mask_B)
    if cx_a is None or cx_b is None:
        return {'resolvable': False, 'reason': 'empty_or_missing_source_mask',
                'cx_A': cx_a, 'cx_B': cx_b, 'gap': None, 'left_actor': None}
    gap = abs(cx_a - cx_b)
    width = max(width_a or 0, width_b or 0) or 1
    frac = gap / float(width)
    limit = MIN_RESOLVABLE_GAP_FRAC if min_gap is None else min_gap / float(width)
    if frac < limit:
        return {'resolvable': False,
                'reason': f'source_centroids_within_{100 * limit:.1f}pct_of_{width}px',
                'cx_A': round(cx_a, 1), 'cx_B': round(cx_b, 1), 'gap': round(gap, 1),
                'gap_frac': round(frac, 4), 'mask_width': width, 'left_actor': None}
    return {'resolvable': True, 'reason': None,
            'cx_A': round(cx_a, 1), 'cx_B': round(cx_b, 1), 'gap': round(gap, 1),
            'gap_frac': round(frac, 4), 'mask_width': width,
            'left_actor': 'A' if cx_a < cx_b else 'B'}


def resolve_slot_identities(bank_record: dict, source_mask_A, source_mask_B,
                            *, min_gap: float | None = None) -> dict:
    """Decide which tracked identity goes into <image3> / <image4>, refusing bad frames.

    `<image3>` must carry the source-left person because that is who the prompt calls
    person 1. The bank's own `left_actor` field is re-measured here instead of trusted: a
    stale or hand-edited field must not be able to redefine the order.
    """
    measured = measure_source_order(source_mask_A, source_mask_B, min_gap=min_gap)
    if not measured['resolvable']:
        raise PersonOrderError(
            f"{bank_record.get('pair_id', '?')}: left/right unresolvable "
            f"({measured['reason']}, gap={measured['gap']})")
    recorded = bank_record.get('left_actor')
    if recorded is None:
        raise PersonOrderError(
            f"{bank_record.get('pair_id', '?')}: bank record has no left_actor; "
            "the prompt's person 1 = source-left convention cannot be honoured")
    if recorded != measured['left_actor']:
        raise PersonOrderError(
            f"{bank_record.get('pair_id', '?')}: recorded left_actor={recorded} but the source "
            f"masks say {measured['left_actor']} (cx A={measured['cx_A']} B={measured['cx_B']})")
    right = 'B' if measured['left_actor'] == 'A' else 'A'
    return {'slot0_identity': measured['left_actor'], 'slot1_identity': right,
            'evidence': measured}


def prompt_colour_claim(prompt: str) -> str | None:
    """Identity implied by a colour word attached to `<image3>`, if the text says exactly one.

    Built on :func:`slot_colour_claims` so the narrow red/blue vocabulary cannot drift apart from
    the full one the gate and the purge use -- two answers to "is there a colour claim" measured
    with different word lists is how a gate ends up half-blind.
    """
    implied = {COLOUR_IDENTITY[colour]
               for colour, index in slot_colour_claims(prompt)
               if index == '3' and colour in COLOUR_IDENTITY}
    if len(implied) != 1:
        # "person 1 follows the red skeleton, person 2 the blue one" is ordinary prose, not an
        # error, and must not take down a caller that is only collecting witnesses.
        return None
    return implied.pop()


# `<image3>`/`<image4>` are rendered single-colour, so calling one red or blue states something
# false about the image the generator actually receives. Colour words about the *combined*
# <image2> are true (identity A is drawn red, B blue) and must not be matched here -- hence the
# requirement that the colour and the slot reference sit in the same sentence.
# Any colour adjective attached to a skeleton reference in ANY of the fed images.
#
# This used to stop at <image3>/<image4>, on the assumption that the combined <image2> is drawn
# "A red, B blue" so a colour word there was true. Measured pixels disprove that: the guide the
# assembler feeds as <image2> contains red AND blue AND pale ink in every pair
# (pilot pair 000082: red 4699 px, blue 2401 px, white-ish 2299 px), because `render_poseguide`
# paints a 12-colour OpenPose limb palette -- the colour encodes WHICH LIMB, not WHO. The isolated
# guides are the same way (person_0: red 1944 / blue 1099 / white 1212), so "the blue skeleton in
# <image2>", "the white skeleton in <image3>" and friends all mis-describe the image the generator
# is handed, and none of them can identify a person.
SLOT_COLOUR = r'(?:red|blue|green|yellow|orange|cyan|magenta|pink|purple|white|black|teal|grey|gray)'
SLOT_COLOUR_CLAIM = re.compile(
    # The colour must be adjacent to the word "skeleton": the loose `[^.]{0,60}` form let
    # "the red-haired woman, skeleton in <image3>" count as a colour claim, which a test forbids.
    r'\b(?:' + SLOT_COLOUR + r')[ -](?:colou?red\s+)?skeletons?\b[^.]{0,40}?<image[\s_]*([234])>'
    r'|<image[\s_]*([234])>[^.]{0,60}?skeletons?\b[^.]{0,30}?\b' + SLOT_COLOUR + r'\b'
    # appositive form the first two branches missed: "source <image3>, red skeleton"
    r'|<image[\s_]*([234])>[\s,;]+' + SLOT_COLOUR + r'\s+skeletons?\b', re.I)


def slot_colour_claims(prompt: str) -> list:
    """[(colour, '2'|'3'|'4')] -- colour words attached to a fed skeleton image.

    Read from the matched span rather than from capture-group positions: the three branches put the
    colour and the reference in different group slots, so indexing groups would silently mis-read.
    """
    claims = []
    for match in SLOT_COLOUR_CLAIM.finditer(prompt or ''):
        span = match.group(0)
        colour = re.search(r'\b' + SLOT_COLOUR + r'\b', span, re.I)
        index = re.search(r'<image[\s_]*([234])>', span, re.I)
        if colour and index:
            claims.append((colour.group(0).lower(), index.group(1)))
    return claims


def slot_contract_problems(prompt: str) -> list:
    """Hard failures for a bank record: text that is false about the images it will feed."""
    return [f'<image{index}> is called the {colour} skeleton, but every fed skeleton image is drawn '
            f'with a multi-colour limb palette, so the sentence describes something not present in '
            f'that image and cannot identify a person'
            for colour, index in sorted(set(slot_colour_claims(prompt)))]


def verify_condition(prompt: str, slot_identities: dict) -> list[str]:
    """Hard failures only: reasons this condition's images contradict its text.

    The one invariant that must never break is that `<image3>` carries the source-left person,
    because the prompt addresses the pair by side. Wording that is merely ambiguous or that
    names a skeleton colour goes to :func:`advisories` instead, so a gate can block on the
    invariant without stalling on prose quality.
    """
    evidence = slot_identities.get('evidence') or {}
    left_actor = evidence.get('left_actor')
    if left_actor is None:
        raise PersonOrderError('slot_identities carries no measured evidence.left_actor')
    slot0 = slot_identities['slot0_identity']

    problems = []
    if slot0 != left_actor:
        problems.append(
            f'<image3> carries identity {slot0} but the source-left person is {left_actor}; '
            'the prompt addresses the two people by side')
    if prompt_side_claim(prompt) == 'right' and slot0 == left_actor:
        problems.append(f'prompt says <image3> is the right person, but slot 0 holds the '
                        f'source-left identity {slot0}')
    return problems


def advisories(prompt: str, slot_identities: dict) -> list[str]:
    """Text-quality witnesses that do not invalidate the condition's ordering."""
    notes = []
    claim = prompt_side_claim(prompt)
    if claim == 'contradictory':
        notes.append('prompt calls <image3> both the left and the right person')
    colour = prompt_colour_claim(prompt)
    if colour is not None:
        # the isolated skeleton renders are single-colour; a colour here is read off <image2>
        notes.append(f'prompt names the <image3> skeleton {IDENTITY_COLOUR[colour]} '
                     f'(slot 0 holds identity {slot_identities["slot0_identity"]})')
    notes.extend(text_contract_problems(prompt))
    return notes


def text_contract_problems(prompt: str) -> list[str]:
    """Text-side contract breaks that are independent of image order."""
    problems = []
    if TARGET_IMAGE_REF.search(prompt or ''):
        problems.append('prompt refers to the target photograph the generator never receives')
    return problems


def contamination(prompt: str) -> dict:
    """Non-blocking text-quality witnesses; kept separate so a gate can choose its own policy."""
    return {
        'pixel_numbers': len(PIXEL_NUMBER.findall(prompt or '')),
        'hard_negations': len(HARD_NEGATION.findall(prompt or '')),
        'target_image_refs': len(TARGET_IMAGE_REF.findall(prompt or '')),
        'colour_claims': prompt_colour_claim(prompt) is not None,
    }
