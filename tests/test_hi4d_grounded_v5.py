"""Regression contracts for official-3D Hi4D pose text and immutable output."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = [ROOT / 'experiments' / family / 'scripts/hi4d_grounded_v5.py'
              for family in ('hi4d_pe_prompt_rewrite_20261020',
                             '4.two man /hi4d_pe_prompt_rewrite_20261020')]


def producer():
    path = next((p for p in CANDIDATES if p.is_file()), CANDIDATES[0])
    assert path.is_file(), 'the grounded v5 generator must exist'
    spec = importlib.util.spec_from_file_location('grounded_v5_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def person(offset=0):
    j = np.zeros((24, 3), dtype=float)
    points = {0: (0, 1, 3), 1: (-.1, 1, 3), 2: (.1, 1, 3),
              4: (-.1, .6, 3), 5: (.1, .6, 3), 7: (-.1, .2, 3),
              8: (.1, .2, 3), 10: (-.1, .1, 3.1), 11: (.1, .1, 3.1),
              12: (0, 1.55, 3), 15: (0, 1.8, 3), 16: (-.2, 1.5, 3),
              17: (.2, 1.5, 3), 18: (-.3, 1.2, 3), 19: (.3, 1.2, 3),
              20: (-.4, .9, 3), 21: (.4, .9, 3)}
    for idx, xyz in points.items():
        j[idx] = xyz
    j[:, 0] += offset
    return j


def test_camera_z_is_not_confidence_and_swapping_actors_is_rejected():
    v5 = producer()
    joints = np.array([person(), person(1)])
    meta = {'R_world_to_camera': np.eye(3).tolist(),
            't_world_to_camera': [0, 0, 0], 'K_512': [[400, 0, 256], [0, 400, 256], [0, 0, 1]]}
    cam = joints.copy()
    cam[:, [3, 6, 9, 13, 14, 22, 23], 2] = 3
    joints = cam.copy()
    uvw = cam @ np.array(meta['K_512']).T
    npys = [np.column_stack((uvw[i, :, :2]/uvw[i, :, 2:3], cam[i, :, 2], np.ones(24))) for i in range(2)]
    result = v5.validate_projection(joints, npys[0], npys[1], meta)
    assert result['A']['depth_max_abs_error_m'] < 1e-8
    with pytest.raises(ValueError, match='projection'):
        v5.validate_projection(joints, npys[1], npys[0], meta)


def test_projected_straight_118_degree_elbow_keeps_its_physical_bend():
    v5 = producer()
    j = person()
    # At the elbow, a 118.7-degree angle has a large depth component invisible in an XY view.
    j[18] = (0, 1.3, 3)
    j[16] = (-.2, 1.3, 3)
    theta = math.radians(61.3)
    j[20] = (.2*math.cos(theta), 1.3, 3+.2*math.sin(theta))
    facts = v5.person_facts(j)
    assert facts['arms']['left']['elbow_angle_deg'] == pytest.approx(118.7, abs=1e-5)
    clauses = v5.person_clauses(facts)
    assert 'bent elbow' in clauses['left_arm']
    assert 'straight elbow' not in clauses['left_arm']


def test_legs_are_independent_and_anatomical_sides_do_not_follow_wrist_height():
    v5 = producer()
    j = person()
    j[4], j[7] = (-.1, .65, 3), (-.1, .65, 3.4)
    j[20, 1], j[21, 1] = 1.8, .8
    facts = v5.person_facts(j)
    clauses = v5.person_clauses(facts)
    assert facts['arms']['left']['joint_ids'] == [16, 18, 20]
    assert facts['arms']['right']['joint_ids'] == [17, 19, 21]
    assert 'bent knee' in clauses['left_leg']
    assert 'nearly straight knee' in clauses['right_leg']
    sentence = v5.person_sentence('Person 2', clauses)
    assert "Person 2's anatomical left arm" in sentence
    assert "Person 2's anatomical right leg" in sentence
    assert 'higher' not in sentence and 'both knees' not in sentence


def test_2d_foot_height_gap_never_claims_ground_contact_or_lift():
    v5 = producer()
    j = person()
    j[7] = (-.1, .2, 2.6)
    j[8] = (.1, .2, 3.5)
    sentence = v5.person_sentence('Person 1', v5.person_clauses(v5.person_facts(j)))
    assert not any(word in sentence.lower() for word in ('lifted', 'planted', 'ground', 'weight', 'facing', 'turned'))


def test_contact_is_only_named_3d_nearness_with_stable_person_identity():
    v5 = producer()
    a, b = person(), person(2)
    b[20] = a[17] + np.array([0, .05, 0])
    near = v5.interaction_facts(np.array([a, b]), 'B')
    assert any(h['person'] == '1' and h['side'] == 'left' and h['other_index'] == 17 for h in near['near_hits'])
    assert "Person 1's anatomical left hand is near Person 2's anatomical right shoulder" in near['text']
    assert 'contact' not in near['text'] and 'touch' not in near['text']


def test_own_side_3d_wrist_does_not_become_crossed_by_camera_projection():
    v5 = producer()
    j = person()
    facts = v5.person_facts(j)
    assert facts['arms']['right']['lateral_band'] == 'own_side'
    assert 'across' not in v5.person_clauses(facts)['right_arm']


def test_existing_output_is_rejected_before_inputs_are_read(tmp_path):
    v5 = producer()
    with pytest.raises(FileExistsError):
        v5.build_bank(tmp_path/'missing_root', tmp_path/'missing_exp', tmp_path)


def test_bent_torso_hand_distance_uses_a_3d_body_segment():
    v5 = producer()
    j = person()
    j[16], j[17], j[12] = (-.2, 1.1, 3.5), (.2, 1.1, 3.5), (0, 1.2, 3.5)
    j[20] = (-.02, 1.08, 3.4)
    facts = v5.person_facts(j)
    assert facts['arms']['left']['distance_to_torso_segment_m'] < .03
    assert facts['arms']['left']['reach_band'] == 'near'


def test_ankle_separation_is_full_3d_and_makes_no_standing_claim():
    v5 = producer()
    j = person()
    j[7], j[8] = (-.1, .2, 3), (-.1, 1.0, 3)
    facts = v5.person_facts(j)
    assert facts['stance']['ankle_separation_m'] == pytest.approx(.8)
    text = v5.person_sentence('Person 1', v5.person_clauses(facts))
    assert "Person 1's ankles are held well apart" in text
    assert 'stance' not in text and 'standing' not in text


def test_bone_centerline_distance_is_not_a_body_surface_text_claim():
    v5 = producer()
    facts = v5.person_facts(person())
    assert 'distance_to_torso_segment_m' in facts['arms']['left']
    for field in ('left_arm', 'right_arm'):
        clause = v5.person_clauses(facts)[field]
        assert all(phrase not in clause for phrase in ('held near the torso',
                   'held away from the torso', 'held well away from the torso'))


def test_joint15_alone_never_authorizes_above_head_height():
    v5 = producer()
    j = person()
    j[20, 1] = 1.9  # Above the internal head joint, with no measured crown.
    facts = v5.person_facts(j)
    assert facts['arms']['left']['above_head'] is False
    assert facts['head_reference']['status'] == 'unavailable'
    assert 'above head height' not in v5.person_clauses(facts)['left_arm']


@pytest.mark.parametrize('side,wrist_y,head_joint_y,crown_y', [
    # Independently measured official raw geometry: dance28 f103 B right hand.
    ('right', 1.797163486480713, 1.7533655166625977, 1.9361920356750488),
    # hug00 f108 A left hand is at the partner's shoulder, below its own crown.
    ('left', 1.6709866523742676, 1.6316046714782715, 1.8127492666244507),
])
def test_real_shoulder_holds_are_not_called_above_head(side, wrist_y, head_joint_y, crown_y):
    v5 = producer()
    j = person()
    j[20 if side == 'left' else 21, 1] = wrist_y
    j[15, 1] = head_joint_y
    facts = v5.person_facts(j, head_crown_y=crown_y)
    assert facts['arms'][side]['above_head'] is False
    assert facts['head_reference']['reference_y_m'] == pytest.approx(crown_y, abs=1e-6)
    assert 'above head height' not in v5.person_clauses(facts)[side+'_arm']


def test_true_overhead_wrist_is_measured_against_mesh_crown():
    v5 = producer()
    j = person()
    j[20, 1] = 2.2
    reference = {'kind': 'smpl_head_mesh_max_world_y', 'part': 'head', 'vertex_count': 1194,
                 'mesh_source': {'path': '/raw/frame.npz', 'sha256': 'a'*64},
                 'segmentation_source': {'path': '/assets/head.json', 'sha256': 'b'*64}}
    facts = v5.person_facts(j, head_crown_y=2.0, head_reference=reference)
    assert facts['arms']['left']['above_head'] is True
    assert 'above head height' in v5.person_clauses(facts)['left_arm']
    assert facts['head_reference']['kind'] == 'smpl_head_mesh_max_world_y'
    assert facts['head_reference']['mesh_source'] == reference['mesh_source']
    assert facts['head_reference']['segmentation_source'] == reference['segmentation_source']


def test_crown_margin_is_strict_and_not_rounded_before_classification():
    v5 = producer()
    j = person()
    j[12, 1] = 1.5
    j[20, 1] = 2.0 + .08*.5
    assert v5.person_facts(j, head_crown_y=2.0)['arms']['left']['above_head'] is False
    j[20, 1] += 1e-7
    assert v5.person_facts(j, head_crown_y=2.0)['arms']['left']['above_head'] is True


@pytest.mark.parametrize('crown', [float('nan'), float('inf')])
def test_invalid_explicit_crown_is_rejected(crown):
    with pytest.raises(ValueError, match='head crown'):
        producer().person_facts(person(), head_crown_y=crown)


def test_head_mesh_crown_uses_only_named_head_vertices_and_fails_closed():
    v5 = producer()
    verts = np.zeros((2, 6890, 3))
    verts[:, 1, 1] = [1.95, 1.8]
    verts[:, 2, 1] = [1.9, 1.85]
    verts[:, 3, 1] = 2.4  # An overhead hand must not become the head reference.
    np.testing.assert_allclose(v5.head_mesh_crowns(verts, [1, 2]), [1.95, 1.85])
    for ids in ([], [6890], [-1], [1.5]):
        with pytest.raises(ValueError, match='head vertices'):
            v5.head_mesh_crowns(verts, ids)
    with pytest.raises(ValueError, match='official vertices'):
        v5.head_mesh_crowns(np.zeros((2, 24, 3)), [1, 2])
