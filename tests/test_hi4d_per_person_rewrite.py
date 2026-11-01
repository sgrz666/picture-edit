"""Contract tests for the per-person Hi4D rewrite (v4).

These pin the two mistakes that made v3 wrong at the population level:
  * the npy files are SMPL-24, not DWPose-18;
  * "Person 1" is the source-left identity, which is `left_actor`, not always tracked A.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = (
    ROOT / 'experiments/4.two man /hi4d_pe_prompt_rewrite_20261020/scripts/pe_rewrite_hi4d_per_person_v4.py',
    ROOT / 'experiments/hi4d_pe_prompt_rewrite_20261020/scripts/pe_rewrite_hi4d_per_person_v4.py',
)
PATH = next((p for p in CANDIDATES if p.is_file()), None)
if PATH is None:
    pytest.skip('v4 rewrite script is not present in this checkout', allow_module_level=True)

spec = importlib.util.spec_from_file_location('hi4d_v4', PATH)
v4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v4)


def make_person(cx=100.0):
    arr = np.zeros((24, 4), dtype=float)
    points = {
        0: (cx, 200), 1: (cx - 15, 198), 2: (cx + 15, 198), 3: (cx, 170),
        4: (cx - 18, 260), 5: (cx + 18, 260), 6: (cx, 145), 7: (cx - 20, 330),
        8: (cx + 20, 330), 9: (cx, 120), 10: (cx - 20, 340), 11: (cx + 20, 340),
        12: (cx, 105), 13: (cx - 10, 112), 14: (cx + 10, 112), 15: (cx, 90),
        16: (cx - 25, 110), 17: (cx + 25, 110), 18: (cx - 40, 150), 19: (cx + 40, 150),
        20: (cx - 45, 180), 21: (cx + 45, 200), 22: (cx - 50, 185), 23: (cx + 50, 205),
    }
    for index, (x, y) in points.items():
        arr[index] = [x, y, 1.0, 1.0]
    return arr


def test_smpl24_index_describes_a_coherent_body():
    arr = make_person()
    facts = v4.person_facts(arr)
    # neck (y=105) to pelvis (y=200) is the torso scale
    assert facts['scale'] == pytest.approx(95.0, abs=0.5)
    assert facts['torso']['lean'] == 'upright'
    assert facts['legs']['stance'] == 'narrow'
    assert facts['legs']['knees'] == 'straight'
    for field in (v4.FIELD_HIGH, v4.FIELD_LOW):
        arm = facts['arms'][field]
        assert arm['visible']
        assert not arm['above_shoulder'] and not arm['below_hip']
    assert facts['arms'][v4.FIELD_HIGH]['wrist_xy'][1] < facts['arms'][v4.FIELD_LOW]['wrist_xy'][1]


def test_dwpose18_reading_would_have_been_incoherent():
    """Guard against silently reverting to the DWPose index table.

    Under SMPL-24 the shoulders are 16/17 (50 px apart here); under the old DWPose table
    2/5 are the hips (30 px apart) and the 'shoulder' width collapses.
    """
    arr = make_person()
    assert abs(arr[16][0] - arr[17][0]) == pytest.approx(50.0)
    # the old table called 4/7 the two wrists; here they are the left knee and left ankle
    assert abs(arr[4][0] - arr[7][0]) < 5
    assert abs(arr[4][1] - arr[7][1]) > 50


def test_person_one_is_left_actor_not_tracked_a():
    a, b = make_person(100), make_person(300)
    as_a = v4.interaction_facts(a, b, 'A')
    as_b = v4.interaction_facts(a, b, 'B')
    assert as_a['crossed'] is False
    assert as_b['crossed'] is True
    assert 'stays on the left' in as_a['layout']
    assert 'is on the right of the frame' in as_b['layout']


def test_contact_is_attributed_to_the_left_actor():
    a, b = make_person(100), make_person(300)
    # put B's left wrist on A's right shoulder
    b[20] = [a[17][0], a[17][1], 1.0, 1.0]
    as_b = v4.interaction_facts(a, b, 'B')
    assert as_b['contact_kind'] == 'hand_to_part'
    assert "Person 1's hand is at Person 2's shoulders" in as_b['contact']
    as_a = v4.interaction_facts(a, b, 'A')
    assert "Person 2's hand is at Person 1's shoulders" in as_a['contact']


def test_part_names_are_coarse_enough_to_stay_true():
    assert v4.part_name(15) == 'head and shoulders'
    assert v4.part_name(12) == 'head and shoulders'
    assert v4.part_name(16) == 'shoulders'
    assert v4.part_name(20) == 'arm'
    assert v4.part_name(4) == 'legs'


def test_numbers_are_stripped_and_legs_get_a_frame():
    assert v4.strip_numbers('leaning at a strong 29-degree tilt') == 'leaning at a strong tilt'
    assert v4.strip_numbers('about 26 px from the midpoint') == 'from the midpoint'
    assert v4.normalize_clause('legs', 'narrow stance with both knees bent') == \
        'set in a narrow stance with both knees bent'
    assert v4.normalize_clause(v4.FIELD_HIGH, 'wrist positioned between the shoulder and the hip') == \
        'positioned between the shoulder and the hip'


def test_arm_claims_that_contradict_measurement_are_rejected():
    facts = v4.person_facts(make_person())
    high = facts['arms'][v4.FIELD_HIGH]
    high['above_shoulder'] = True
    high['above_head'] = False
    assert v4.arm_claim_conflict('hanging below the hip close to the body', high)
    assert not v4.arm_claim_conflict('raised above the shoulder with a nearly straight elbow', high)
    high['reach_ratio'] = 0.05
    assert v4.arm_claim_conflict('extended outward from the body', high)
    high['reach_ratio'] = 0.8
    assert v4.arm_claim_conflict('held close to the body', high)
    high['reach_ratio'] = 0.2
    high['above_shoulder'] = False
    assert v4.arm_claim_conflict('hanging below the hip', high)
    assert v4.arm_claim_conflict('raised above the head', high)
    assert v4.arm_claim_conflict('raised above the shoulder', high)
    assert not v4.arm_claim_conflict('held at waist height', high)


def test_direction_and_stance_claims_are_checked():
    facts = v4.person_facts(make_person())
    facts['torso']['lean'] = 'toward the right of the frame'
    assert v4.direction_conflict('torso', 'leaning toward the left of the frame', facts)
    assert not v4.direction_conflict('torso', 'leaning toward the right of the frame', facts)
    facts['legs']['one_foot_lifted'] = True
    assert v4.leg_conflict('set in a narrow stance with both feet planted', facts['legs'])
    facts['legs']['one_foot_lifted'] = False
    facts['legs']['stance'] = 'narrow'
    assert v4.leg_conflict('set in a wide stance', facts['legs'])
    assert v4.HEAD_ROTATION.search('turned toward the left of the frame')
    assert not v4.HEAD_ROTATION.search('centered over the shoulders')


def test_cross_claims_are_repaired_only_when_not_beyond_the_opposite_shoulder(tmp_path):
    arr = make_person(100)
    facts = v4.person_facts(arr)
    field = next(f for f in (v4.FIELD_HIGH, v4.FIELD_LOW)
                 if facts['arms'][f].get('joint_side') == 'right')
    path = tmp_path / 'a.npy'
    cache = {'npy': str(path), 'facts': facts,
             'clauses': {v4.FIELD_HIGH: 'held at waist height',
                         v4.FIELD_LOW: 'held at waist height'}}
    cache['clauses'][field] = 'crossed toward the opposite side of the body'

    # right wrist at x=90: past the midline (100) but not past the left shoulder (75)
    arr[21] = [90, 190, 1.0, 1.0]
    np.save(path, arr)
    cache['facts'] = v4.person_facts(arr)
    assert not v4.clear_arm_cross(cache, field)
    clauses, repairs = v4.repair_arm_cross(cache)
    assert field in repairs
    assert 'cross' not in clauses[field]

    # right wrist at x=60: 15 px beyond the left shoulder
    arr[21] = [60, 190, 1.0, 1.0]
    np.save(path, arr)
    cache['facts'] = v4.person_facts(arr)
    assert v4.clear_arm_cross(cache, field)
    _clauses, repairs = v4.repair_arm_cross(cache)
    assert field not in repairs


def test_unmeasured_weight_claims_are_stripped():
    assert v4.strip_unmeasured(
        'set in a moderate stance with both knees straight and the body weight shifted toward the left of the frame'
    ) == 'set in a moderate stance with both knees straight'
    assert v4.strip_unmeasured('set in a wide stance and the weight carried on the legs') == \
        'set in a wide stance'


def test_every_fallback_clause_is_a_usable_phrase():
    facts = v4.person_facts(make_person())
    for field in v4.REQUIRED_FIELDS:
        clause = v4.fallback_clause(field, facts)
        assert 3 <= len(clause.split()) <= 28, (field, clause)
        assert not v4.BANNED.search(clause), (field, clause)
        assert not any(ch.isdigit() for ch in clause), (field, clause)


def test_person_sentence_names_the_person_and_both_arms():
    facts = v4.person_facts(make_person())
    clauses = {field: v4.fallback_clause(field, facts) for field in v4.REQUIRED_FIELDS}
    sentence = v4.person_sentence('Person 2', clauses, facts)
    assert sentence.startswith("Person 2's torso is ")
    assert 'the arm whose wrist is higher is ' in sentence
    assert 'the arm whose wrist is lower is ' in sentence
