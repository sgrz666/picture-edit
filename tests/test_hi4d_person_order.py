"""The prompt addresses the pair by side, so <image3> must carry the source-left person.

Regression coverage for the defect found in the Hi4D bank on 2026-10-28: the rewrite pipeline
fed the source-left actor's skeleton as <image3> and wrote "person 1 on the left in <image1>"
into every prompt, while the condition assembler fed native geometry slot 0 -- always tracked
identity A -- into that same position. 3,014 of 6,600 pairs were therefore trained on a prompt
that described one person and showed another, and the bank's own audit reported 0% inverted
because it compared `left_actor` against the same masks that produced it.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pose_control.qwen21_full.native_v5.person_order import (  # noqa: E402
    PersonOrderError, advisories, contamination, measure_source_order, prompt_colour_claim,
    prompt_side_claim, resolve_slot_identities, slot_colour_claims, slot_contract_problems,
    text_contract_problems, verify_condition)

PILOT_CONDITION_DIR = Path(
    '/data/shangguanrz/pic-edit-experiments/qwen21_hi4d_dual/'
    'hi4d-fixedpair-v5_20261025_v1/data/native_v5_v1')

PROMPT_CLAIMING_LEFT = ('Edit the two existing people in <image1>. <image2> defines the combined '
                        'dual-person spatial layout, while <image3> guides person 1 on the left in '
                        '<image1> and <image4> guides person 2 on the right in <image1>.')


def write_mask(path: Path, centre_x: int, width: int = 40) -> None:
    array = np.zeros((512, 512), dtype=np.uint8)
    array[:, max(0, centre_x - width // 2):centre_x + width // 2] = 255
    Image.fromarray(array).save(path)


@pytest.fixture
def masks(tmp_path):
    """A on the left (x=150), B on the right (x=360)."""
    a = tmp_path / 'A.png'
    b = tmp_path / 'B.png'
    write_mask(a, 150)
    write_mask(b, 360)
    return a, b


def test_measure_source_order_finds_the_left_person(masks):
    measured = measure_source_order(*masks)
    assert measured['resolvable'] is True
    assert measured['left_actor'] == 'A'
    assert measured['gap'] > 100


def test_overlapping_silhouettes_are_refused_not_guessed(tmp_path):
    a = tmp_path / 'A.png'
    b = tmp_path / 'B.png'
    write_mask(a, 256, width=30)
    write_mask(b, 262, width=30)
    measured = measure_source_order(a, b)
    assert measured['resolvable'] is False
    assert 'within' in measured['reason']
    with pytest.raises(PersonOrderError):
        resolve_slot_identities({'pair_id': 'p', 'left_actor': 'A'}, a, b)


def test_stale_left_actor_field_is_rejected(masks):
    """A hand-edited or stale field must not be able to redefine the order."""
    with pytest.raises(PersonOrderError, match='recorded left_actor=B'):
        resolve_slot_identities({'pair_id': 'p', 'left_actor': 'B'}, *masks)


def test_missing_left_actor_field_is_rejected(masks):
    with pytest.raises(PersonOrderError, match='no left_actor'):
        resolve_slot_identities({'pair_id': 'p'}, *masks)


def test_correct_order_passes(masks):
    ids = resolve_slot_identities({'pair_id': 'p', 'left_actor': 'A'}, *masks)
    assert ids == {'slot0_identity': 'A', 'slot1_identity': 'B', 'evidence': ids['evidence']}
    assert verify_condition(PROMPT_CLAIMING_LEFT, ids) == []


def test_shipped_defect_is_caught(masks):
    """Identity A in <image3> while B is the source-left person: exactly the bank's failure."""
    ids = resolve_slot_identities({'pair_id': 'p', 'left_actor': 'A'}, *masks)
    # swap the fed order without touching the text, as the old assembler did when A was on the right
    wrong = {'slot0_identity': 'B', 'slot1_identity': 'A', 'evidence': ids['evidence']}
    problems = verify_condition(PROMPT_CLAIMING_LEFT, wrong)
    assert problems and 'source-left person is A' in problems[0]


def test_prompt_contradicting_its_own_order_is_caught(masks):
    ids = resolve_slot_identities({'pair_id': 'p', 'left_actor': 'A'}, *masks)
    # internally consistent but reversed: <image3> called the right person while slot 0 holds
    # the source-left identity -- the shape the bank's 25 self-corrected records take
    reversed_text = ('<image3> guides person 1 on the right in <image1> and <image4> guides '
                     'person 2 on the left in <image1>.')
    assert prompt_side_claim(reversed_text) == 'right'
    assert any('is the right person' in p for p in verify_condition(reversed_text, ids))


def test_skeleton_colour_claim_is_advisory_not_a_order_failure(masks):
    ids = resolve_slot_identities({'pair_id': 'p', 'left_actor': 'A'}, *masks)
    assert prompt_colour_claim('Person 1 matches the blue skeleton in <image3>.') == 'B'
    text = 'Person 1 matches the blue skeleton in <image3>.'
    assert verify_condition(text, ids) == []          # ordering itself is fine
    assert any('blue' in note for note in advisories(text, ids))
    # every skeleton-colour claim is worth a note: the isolated renders are single-colour, so the
    # colour can only have been read off <image2>, which does not name the slots
    assert any('red' in note for note in advisories(
        'Person 1 matches the red skeleton in <image3>.', ids))
    # "red-haired woman ..., skeleton in <image3>" must not count as a colour claim
    assert prompt_colour_claim('the red-haired woman, skeleton in <image3>') is None


def test_prompt_naming_both_sides_is_advisory(masks):
    ids = resolve_slot_identities({'pair_id': 'p', 'left_actor': 'A'}, *masks)
    filler = ' The pair stand on a green screen with lights behind them. ' * 4
    both = f'<image3> guides the left person here.{filler}and <image3> is clearly the right person.'
    assert prompt_side_claim(both) == 'contradictory'
    assert any('both the left and the right' in n for n in advisories(both, ids))
    assert verify_condition(both, ids) == []          # ambiguous prose does not block training


def test_verify_requires_measured_evidence():
    with pytest.raises(PersonOrderError, match='evidence.left_actor'):
        verify_condition(PROMPT_CLAIMING_LEFT, {'slot0_identity': 'A'})


def test_contamination_counts_hyphenated_pixel_values():
    report = contamination('They stand 7.8-pixel apart, 148.8 pixels, then remain separated '
                           'with no physical contact.')
    assert report['pixel_numbers'] == 2
    assert report['hard_negations'] == 2
    assert text_contract_problems('copy the pose from the target reference photograph') != []
    assert text_contract_problems(PROMPT_CLAIMING_LEFT) == []


@pytest.mark.skipif(not PILOT_CONDITION_DIR.is_dir(), reason='pilot conditions live on sg')
def test_condition_gate_rejects_a_contradictory_declaration(tmp_path):
    """End to end: read_condition must refuse a dual condition whose order contradicts its text."""
    from src.pose_control.qwen21_full.native_v5 import conditioning

    sources = sorted(PILOT_CONDITION_DIR.glob('*_qwen.json'))
    if not sources:
        pytest.skip('no assembled pilot conditions')
    spec = json.loads(sources[0].read_text(encoding='utf-8'))
    left = 'A'  # the pilot's two pairs both have identity A on the left
    assert prompt_side_claim(spec['prompt']['text']) in ('left', 'contradictory')
    # The pilot was assembled against the pre-remediation bank, so its recorded bank SHA is
    # stale since the 2026-10-27 regeneration. This test is about the order gate, so re-pin it.
    import hashlib
    bank_path = Path(spec['prompt']['bank']['path'])
    if bank_path.is_file():
        spec['prompt']['bank']['sha256'] = hashlib.sha256(bank_path.read_bytes()).hexdigest()

    def write(declared_slot0, evidence_left):
        mutated = dict(spec)
        mutated['person_order'] = {'slot0_identity': declared_slot0,
                                   'slot1_identity': 'B' if declared_slot0 == 'A' else 'A',
                                   'evidence': {'left_actor': evidence_left}}
        path = tmp_path / f'cond_{declared_slot0}_{evidence_left}.json'
        blob = json.dumps(mutated, ensure_ascii=False)
        path.write_text(blob, encoding='utf-8')
        return {'path': str(path), 'sha256': hashlib.sha256(blob.encode()).hexdigest()}

    conditioning.read_condition(write(left, left))          # consistent -> loads
    with pytest.raises(ValueError, match='contradicts its image order'):
        conditioning.read_condition(write('B' if left == 'A' else 'A', left))


def test_separability_cut_is_scale_relative(tmp_path):
    """20 px is 3.9% of the bank's 512-wide masks but only 2.1% of Hi4D's raw 940-wide masks.

    The rewrite worker flags `side_ambiguous` from the preprocessed masks while the assembler
    measures the raw seg masks, so an absolute cut let the same physical pair be "refused" in one
    layer and "resolvable" in the other. Both must now decide by fraction of width.
    """
    def stripe(name, centre, size, half=20):
        path = tmp_path / f'{name}_{size}.png'
        array = np.zeros((size, size), dtype=np.uint8)
        array[:, max(0, centre - half):centre + half] = 255
        Image.fromarray(array).save(path)
        return path

    # 15/512 = 2.9% and 28/940 = 3.0%: both below the 3.9% cut. The old absolute rule accepted the
    # second one because 28 >= 20, which is precisely the disagreement this test pins down.
    small = measure_source_order(stripe('a', 300, 512), stripe('b', 315, 512))
    raw = measure_source_order(stripe('c', 300, 940), stripe('d', 328, 940))
    assert not small['resolvable'] and not raw['resolvable'], (small, raw)
    assert small['gap'] == 15.0 and raw['gap'] == 28.0     # pixels still reported

    # 40/940 = 4.3% is above the cut, so separation is accepted on the raw scale too.
    assert measure_source_order(stripe('e', 300, 940),
                                stripe('f', 340, 940))['resolvable']

    # Same relative layout at both resolutions must give the same verdict.
    assert (measure_source_order(stripe('g', 200, 512), stripe('h', 215, 512))['resolvable']
            == measure_source_order(stripe('i', 360, 940), stripe('j', 388, 940))['resolvable'])


def test_side_claim_stays_inside_its_own_person_zone():
    """Person 2 being "on the right" must not make the text look self-contradictory.

    The first version of this judge measured distance with a fixed 140-character window, which
    reached past the end of the person-1 description into the person-2 description and reported
    843 of 1,208 bank records as calling <image3> both sides at once.
    """
    prompt = (PROMPT_CLAIMING_LEFT +
              ' Person 1 remains on the left side and Person 2 remains on the right side, '
              'standing in close body contact: Person 1 (referencing <image3>) raises one arm.')
    assert prompt_side_claim(prompt) == 'left'
    ids = {'slot0_identity': 'A', 'slot1_identity': 'B', 'evidence': {'left_actor': 'A'}}
    assert not [a for a in advisories(prompt, ids) if 'both the left and the right' in a]


def test_reversed_side_claim_is_still_a_hard_failure():
    prompt = ('<image3> guides person 1 on the right in <image1> and <image4> guides person 2 '
              'on the left in <image1>.')
    ids = {'slot0_identity': 'A', 'slot1_identity': 'B', 'evidence': {'left_actor': 'A'}}
    assert prompt_side_claim(prompt) == 'right'
    assert verify_condition(prompt, ids)


def test_colour_attached_to_an_isolated_skeleton_is_a_contract_problem():
    prompt = ('Person 1 (the man on the left in <image1>, matching the blue skeleton in <image3>) '
              'stands upright. Person 2 (matching the red skeleton in <image4>) leans back.')
    assert sorted(slot_colour_claims(prompt)) == [('blue', '3'), ('red', '4')]
    assert len(slot_contract_problems(prompt)) == 2
    # Colour words about the combined skeleton are NOT exempt either: the fed <image2> is drawn with
    # a 12-colour limb palette (measured on a pilot pair: red 4699 / blue 2401 / white-ish 2299 px),
    # so a colour identifies a limb and never a person. One sentence yields one claim -- the matcher
    # takes the colour nearest the reference, so do not expect two from one clause.
    combined = ('<image2> shows the pair where the red skeleton is person 1 and the blue '
                'skeleton is person 2.')
    assert slot_colour_claims(combined) == [('red', '2')]
    assert len(slot_contract_problems(combined)) == 1
    clothing = 'Person 1 (the woman in the red shirt in <image1>) stands upright.'
    assert slot_colour_claims(clothing) == []
    # A colour word that is not attached to a skeleton reference stays untouched.
    assert slot_colour_claims('the two people in the blue light of the studio') == []


def test_appositive_colour_after_the_reference_is_a_claim_too():
    """`source <image3>, red skeleton` makes the same false statement as the other two orders."""
    prompt = ('Person 1 (the man on the left, source <image3>, red skeleton) stands upright while '
              'Person 2 matches the skeleton in <image4>.')
    assert ('red', '3') in slot_colour_claims(prompt)
    assert slot_contract_problems(prompt)
    # `red-and-blue skeleton layout of <image2>` is a claim about the fed combined image too, and
    # the combined guide is a multi-colour limb palette, so it is flagged as well. The matcher
    # reports the colour nearest the reference, which here is the one adjacent to "skeleton".
    combined = 'matching the intertwined red-and-blue skeleton layout of <image2>.'
    assert slot_colour_claims(combined) == [('blue', '2')]
