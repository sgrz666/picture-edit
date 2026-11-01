"""Independent verifier regressions; fixtures never call the v5 producer."""
import importlib.util
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
_CANDIDATES = [
    ROOT / 'experiments/hi4d_pe_prompt_rewrite_20261020/scripts/verify_hi4d_grounded_v5.py',
    ROOT / 'experiments/4.two man /hi4d_pe_prompt_rewrite_20261020/scripts/verify_hi4d_grounded_v5.py',
]
SCRIPT = next((p for p in _CANDIDATES if p.is_file()), _CANDIDATES[0])


def verifier():
    assert SCRIPT.is_file(), 'the independent 3D verifier must exist'
    spec = importlib.util.spec_from_file_location('independent_hi4d_v5', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def body():
    j = np.zeros((24, 3), dtype=np.float64)
    j[:, 2] = 3
    for k, xyz in {
        0: (0, .9, 3), 1: (-.1, .9, 3), 2: (.1, .9, 3),
        4: (-.1, .45, 3), 5: (.1, .45, 3), 7: (-.1, 0, 3), 8: (.1, 0, 3),
        12: (0, 1.5, 3), 15: (0, 1.7, 3),
        16: (-.25, 1.4, 3), 17: (.25, 1.4, 3),
        18: (-.25, .9, 3), 19: (.25, .9, 3),
        20: (-.25, .4, 3.9), 21: (.25, .4, 3),
    }.items():
        j[k] = xyz
    return j


def camera():
    return {'R_world_to_camera': np.eye(3).tolist(), 't_world_to_camera': [0, 0, 0],
            'K_512': [[200, 0, 256], [0, 200, 256], [0, 0, 1]]}


def projected(joints):
    k = np.asarray(camera()['K_512'])
    image = joints @ k.T
    return np.column_stack((image[:, :2] / image[:, 2:], joints[:, 2], np.ones(24)))


def test_projected_straight_arm_is_bent_in_official_3d():
    v = verifier()
    geometry = v.recompute_geometry(body())
    assert geometry['elbow_angles']['left'] == pytest.approx(119.0546041, abs=1e-6)
    assert geometry['elbow_angles']['right'] == pytest.approx(180)
    assert geometry['knee_angles'] == {'left': 180., 'right': 180.}


def test_projection_checks_camera_z_instead_of_confidence():
    v = verifier()
    raw = np.stack((body(), body() + [1, 0, 0]))
    arrays = {identity: projected(raw[i]) for i, identity in enumerate(('A', 'B'))}
    assert v.projection_problems(raw, camera(), arrays) == []
    arrays['A'][:, 2] = 1
    assert any('camera_Z' in error for error in v.projection_problems(raw, camera(), arrays))


def test_projection_detects_swapped_tracked_people():
    v = verifier()
    raw = np.stack((body(), body() + [1, 0, 0]))
    arrays = {'A': projected(raw[1]), 'B': projected(raw[0])}
    assert any('A:xy' in error for error in v.projection_problems(raw, camera(), arrays))


def test_cross_body_is_measured_along_the_3d_anatomical_axis():
    v = verifier()
    j = body()
    j[21] = [.37, .95, 3]
    geometry = v.recompute_geometry(j)
    # Historical dance2 error: a right wrist on its own side must not be crossed.
    assert geometry['hand_shoulder_axis']['right'] == pytest.approx(.74)
    assert geometry['hand_shoulder_axis']['left'] == pytest.approx(-.5)


def test_near_hand_does_not_mean_surface_contact():
    v = verifier()
    people = np.stack((body(), body() + [1, 0, 0]))
    people[0, 20] = people[1, 17] + [.1, 0, 0]
    hits = v.recompute_proximity(people)
    assert hits[0]['identity'] == 'A'
    assert hits[0]['side'] == 'left'
    assert hits[0]['other_joint'] == 17
    assert hits[0]['distance_m'] == pytest.approx(.1)
    assert 'contact' not in hits[0]


def test_source_left_b_is_independent_of_cached_identity_labels(tmp_path):
    v = verifier()
    manifest = {}
    for identity, x in [('A', 380), ('B', 120)]:
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[100:400, x - 10:x + 10] = 255
        path = tmp_path / f'{identity}.png'
        Image.fromarray(mask).save(path)
        manifest[f'source_person_mask_{identity}'] = str(path)
    correct = {'left_actor': 'B', 'right_actor': 'A', 'side_ambiguous': False,
               'source_order': {'left_actor': 'B', 'resolvable': True}}
    assert v.source_order_problems(correct, manifest) == []
    wrong = dict(correct, left_actor='A', right_actor='B')
    assert any('source_left' in error for error in v.source_order_problems(wrong, manifest))


def test_camera_metadata_is_pinned_to_official_camera_and_crop():
    v = verifier()
    meta = dict(camera(), camera_id=4, extrinsic_mode='direct',
                original_K=camera()['K_512'], A_original_to_512=np.eye(3).tolist())
    official = {'ids': np.array([4]), 'intrinsics': np.array([meta['original_K']]),
                'extrinsics': np.array([np.column_stack((np.eye(3), [0, 0, 0]))])}
    assert v.camera_problems(meta, official, 4) == []
    meta['R_world_to_camera'][0][0] = -.5
    assert any('official_R' in error for error in v.camera_problems(meta, official, 4))


def test_coverage_cannot_pass_by_repeating_one_valid_row():
    v = verifier()
    expected = {'test': {'test_one', 'test_two'}, 'val': set(), 'train': set()}
    correct = [{'split': 'test', 'pair_id': 'test_one'}, {'split': 'test', 'pair_id': 'test_two'}]
    assert v.coverage_problems(correct, expected) == []
    wrong = [correct[0], correct[0]]
    errors = v.coverage_problems(wrong, expected)
    assert any('duplicate_pair_id' in error for error in errors)
    assert any('missing_pair_ids' in error for error in errors)


def golden_clauses():
    return {
        'torso': 'upright with the shoulders above the hips',
        'left_arm': 'has a bent elbow, with the hand below shoulder height and below waist height, on its own anatomical side',
        'right_arm': 'has a nearly straight elbow, with the hand below shoulder height and below waist height, on its own anatomical side',
        'left_leg': 'has a nearly straight knee', 'right_leg': 'has a nearly straight knee',
        'stance': 'held close together',
    }


def test_all_generated_clauses_match_an_independent_3d_reading():
    v = verifier()
    assert v.expected_clauses(body()) == golden_clauses()
    assert v.description_problems(golden_clauses(), body()) == []


@pytest.mark.parametrize('field,replacement', [
    ('torso', 'upright with the head turned toward the partner'),
    ('left_arm', 'has a nearly straight elbow'),
    ('right_arm', 'has a sharply bent elbow'),
    ('left_leg', 'has a bent knee'),
    ('right_leg', 'has a bent knee'),
    ('stance', 'wide'),
])
def test_closed_grammar_rejects_every_wrong_clause(field, replacement):
    v = verifier()
    description = golden_clauses()
    description[field] = replacement
    assert any(field in error for error in v.description_problems(description, body()))


def test_flipping_anatomical_limb_fields_is_rejected():
    v = verifier()
    description = golden_clauses()
    description['left_arm'], description['right_arm'] = description['right_arm'], description['left_arm']
    assert len(v.description_problems(description, body())) == 2


def test_correct_fields_cannot_hide_a_tampered_final_prompt():
    v = verifier()
    semantics = 'A known complete pose.'
    texts = v.expected_prompt_texts(semantics)
    row = dict(texts, extracted_pose_semantics=semantics, rewritten_prompt=semantics,
               compiled_prompt_sha256=v.text_sha256(texts['compiled_prompt']),
               compiled_prompt_inference_sha256=v.text_sha256(texts['compiled_prompt_inference']))
    assert v.prompt_problems(row, semantics) == []
    row['compiled_prompt_inference'] += ' Person 1 has straight elbows.'
    # Even a matching tampered hash does not rescue unsupported text.
    row['compiled_prompt_inference_sha256'] = v.text_sha256(row['compiled_prompt_inference'])
    assert any('compiled_prompt_inference:text' in error for error in v.prompt_problems(row, semantics))


def test_tampered_source_sha_and_wrong_path_are_rejected(tmp_path):
    v = verifier()
    path = tmp_path / 'official.npz'
    path.write_bytes(b'official immutable source')
    ref = {'path': str(path), 'sha256': v.sha256(path)}
    assert v.reference_problems(ref, path) == []
    assert v.reference_problems(dict(ref, sha256='0' * 64), path) == ['sha256']
    other = tmp_path / 'other.npz'
    other.write_bytes(path.read_bytes())
    assert 'path' in v.reference_problems(dict(ref, path=str(other)), path)


def test_swap_persons_is_rejected_even_if_their_descriptions_are_valid():
    v = verifier()
    people = {'person_1': {'identity': 'B', 'official_actor_index': 1},
              'person_2': {'identity': 'A', 'official_actor_index': 0}}
    assert v.person_binding_problems(people, 'B') == []
    assert len(v.person_binding_problems(people, 'A')) == 2


def test_fact_side_flip_and_angle_tamper_are_rejected():
    v = verifier()
    expected = v.expected_facts(body())
    observed = copy.deepcopy(expected)
    observed['arms']['left']['joint_ids'] = [17, 19, 21]
    observed['arms']['left']['elbow_angle_deg'] = 179
    errors = v.value_problems(observed, expected)
    assert any('joint_ids' in error for error in errors)
    assert any('elbow_angle_deg' in error for error in errors)


def test_interaction_person_numbers_follow_source_left_b():
    v = verifier()
    raw = np.stack((body(), body() + [2, 0, 0]))
    raw[1, 20] = raw[0, 17] + [0, .05, 0]
    expected = v.expected_interaction(raw, 'B')
    assert expected['near_hits'][0]['person'] == '1'
    assert expected['near_hits'][0]['identity'] == 'B'
    assert expected['near_hits'][0]['other_index'] == 17
    assert "Person 1's anatomical left hand is near Person 2's anatomical right shoulder." in expected['text']


def test_forward_bent_torso_uses_3d_segment_distance_not_world_horizontal_proxy():
    v = verifier()
    j = body()
    j[16], j[17] = [-.25, .9, 4], [.25, .9, 4]
    j[20] = [0, .9, 3.5]
    g = v.recompute_geometry(j)
    assert g['hand_torso_segment_m']['left'] == pytest.approx(0, abs=1e-12)
    facts = v.expected_facts(j)
    assert facts['arms']['left']['reach_band'] == 'near'
    assert facts['arms']['left']['nearest_torso_segment_fraction'] == pytest.approx(.5)


def test_vertical_split_uses_full_3d_ankle_distance_and_no_stance_claim():
    v = verifier()
    j = body()
    j[7], j[8] = [-.1, 0, 3], [-.1, 1.2, 3]
    g = v.recompute_geometry(j)
    assert g['ankle_distance_m'] == pytest.approx(1.2)
    assert v.expected_clauses(j)['stance'] == 'held well apart'
    sentence = v.expected_person_text(1, j)
    assert "ankles are held well apart" in sentence
    assert 'stance' not in sentence


def test_torso_segment_distance_is_kept_numeric_and_never_claims_surface_proximity():
    v = verifier()
    sentence = v.expected_person_text(1, body())
    assert 'distance_to_torso_segment_m' in v.expected_facts(body())['arms']['left']
    assert not any(phrase in sentence for phrase in
                   ('held near the torso', 'held away from the torso', 'held well away from the torso'))


def test_crop_affine_must_follow_the_declared_original_pixel_crop():
    v = verifier()
    meta = {'crop': {'x0': 20., 'y0': 30., 'side': 1024., 'scale': .5},
            'A_original_to_512': [[.5, 0, -10], [0, .5, -15], [0, 0, 1]]}
    assert v.crop_problems(meta) == []
    meta['A_original_to_512'][0][2] += 100
    assert any('affine' in e for e in v.crop_problems(meta))


def test_raw_canvas_check_rejects_a_self_consistent_but_wrong_crop(tmp_path):
    v = verifier()
    pixels = np.zeros((512, 512, 3), dtype=np.uint8)
    pixels[100:400, 140:320] = [150, 200, 70]
    raw, processed = tmp_path / 'official.png', tmp_path / 'processed.png'
    Image.fromarray(pixels).save(raw)
    Image.fromarray(pixels).save(processed)
    affine = np.eye(3)
    assert v.raster_problems(raw, processed, affine) == []
    affine[0, 2] = 100
    assert any('pixels' in e for e in v.raster_problems(raw, processed, affine))


def test_legacy_crossing_distance_and_status_are_not_unchecked_telemetry():
    v = verifier()
    arrays = {'A': projected(body()), 'B': projected(body() + [1, 0, 0])}
    manifest = {'pair': 'p', 'action': 'a', 'camera_id': 4,
                'source_frame_id': 1, 'target_frame_id': 2}
    row = {'pair_id': v.manifest_id('test', manifest), 'left_actor': 'A',
           'is_crossed': False, 'status': 'complete',
           'min_keypoint_distance': round(float(np.linalg.norm(arrays['A'][:, None, :2] - arrays['B'][None, :, :2], axis=2).min()), 1)}
    assert v.legacy_metadata_problems(row, manifest, 'test', arrays) == []
    row.update(is_crossed=True, min_keypoint_distance=9999, status='failed')
    errors = v.legacy_metadata_problems(row, manifest, 'test', arrays)
    assert all(any(field in e for e in errors) for field in ('is_crossed', 'min_keypoint_distance', 'status'))


def head_fixture(tmp_path):
    raw_path, seg_path = tmp_path / 'official.npz', tmp_path / 'head.json'
    verts = np.zeros((2, 6890, 3))
    verts[0, :1194, 1] = 2.1
    verts[1, :1194, 1] = 2.3
    np.savez(raw_path, verts=verts)
    seg_path.write_text(json.dumps({'head': list(range(1194))}))
    return raw_path, seg_path


def test_above_head_uses_raw_crown_instead_of_smpl_head_joint(tmp_path):
    v = verifier()
    raw_path, seg_path = head_fixture(tmp_path)
    reference = v.recompute_head_reference(raw_path, seg_path, 0)
    assert reference['reference_y_m'] == pytest.approx(2.1)
    assert reference['vertex_count'] == 1194
    j = body()
    j[20, 1] = 1.9  # Above SMPL joint15=1.7, below the actual crown=2.1.
    assert v.expected_facts(j, reference)['arms']['left']['above_head'] is False
    assert 'above head height' not in v.expected_clauses(j, reference)['left_arm']
    j[20, 1] = 2.2
    assert v.expected_facts(j, reference)['arms']['left']['above_head'] is True


def test_above_crown_gate_uses_unrounded_reference_at_threshold(tmp_path):
    v = verifier()
    raw_path, seg_path = head_fixture(tmp_path)
    with np.load(raw_path) as raw:
        verts = raw['verts'].copy()
    verts[0, 0, 1] = 2.10000049
    np.savez(raw_path, verts=verts)
    reference = v.recompute_head_reference(raw_path, seg_path, 0)
    j = body()
    j[20, 1] = 2.1000002 + .08 * np.linalg.norm(j[12] - j[0])
    assert v.expected_facts(j, reference)['arms']['left']['above_head'] is False


def test_production_head_reference_refuses_missing_vertices_or_head_segment(tmp_path):
    v = verifier()
    raw_path, seg_path = head_fixture(tmp_path)
    np.savez(raw_path, joints_3d=np.zeros((2, 24, 3)))
    with pytest.raises(ValueError, match='verts'):
        v.recompute_head_reference(raw_path, seg_path, 0)
    raw_path, seg_path = head_fixture(tmp_path)
    seg_path.write_text(json.dumps({'head': []}))
    with pytest.raises(ValueError, match='head'):
        v.recompute_head_reference(raw_path, seg_path, 0)


def valid_build_metadata(tmp_path):
    v = verifier()
    root, exp, bank = tmp_path / 'root', tmp_path / 'exp', tmp_path / 'exp/r4'
    code = bank / 'provenance/hi4d_grounded_v5.py'
    code.parent.mkdir(parents=True)
    code.write_text('frozen producer')
    sources, manifests = {}, {}
    for split in ('test', 'val', 'train'):
        for path, dest in ((exp / f'v4_per_person/prompts/{split}.jsonl', sources),
                           (root / f'datasets/Hi4D_pilot_v1/final_manifests/pairs_1500_{split}.jsonl', manifests)):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}\n')
            dest[split] = {'path': str(path), 'sha256': v.sha256(path)}
    prompt = bank / 'prompts/test.jsonl'
    prompt.parent.mkdir()
    prompt.write_text('{}\n')
    rows = [{'left_actor': 'A', 'projected_layout': {'band': 'person_1_left'}, 'interaction': {'near_hits': []}}]
    summary = {'schema': v.SCHEMA, 'status': 'complete', 'output': str(bank),
               'code': {'path': str(code), 'sha256': v.sha256(code)}, 'thresholds': v.THRESHOLDS,
               'source_banks': sources, 'manifests': manifests, 'rows': 1, 'targets': 1, 'person_cache': 2,
               'source_left_counts': {'A': 1}, 'projected_layout_counts': {'person_1_left': 1}, 'near_rows': 0,
               'splits': {'test': {'rows': 1, 'bank': str(prompt), 'sha256': v.sha256(prompt)}}}
    path = bank / 'report/build_summary.json'
    path.parent.mkdir()
    path.write_text(json.dumps(summary))
    marker = {'schema': v.SCHEMA, 'summary_sha256': v.sha256(path),
              'code_sha256': v.sha256(code), 'banks': copy.deepcopy(summary['splits'])}
    return v, root, exp, bank, rows, summary, marker


def test_build_marker_and_summary_metadata_cannot_be_rehashed_into_validity(tmp_path):
    v, root, exp, bank, rows, summary, marker = valid_build_metadata(tmp_path)
    args = (bank, exp, root, rows, 1, 2, {'test': 1})
    assert v.build_metadata_problems(summary, marker, *args) == []
    summary.update(schema='wrong', status='failed', rows=9999)
    path = bank / 'report/build_summary.json'
    path.write_text(json.dumps(summary))
    marker.update(summary_sha256=v.sha256(path), code_sha256='0' * 64)
    errors = v.build_metadata_problems(summary, marker, *args)
    assert all(any(field in e for e in errors) for field in ('schema', 'status', 'rows', 'code_sha256'))


def test_optional_ready_cannot_point_at_an_unrelated_or_failed_audit(tmp_path):
    v, root, exp, bank, rows, summary, marker = valid_build_metadata(tmp_path)
    counts = {'rows': 1, 'targets': 1, 'person_cache': 2, 'test': 1}
    assert v.ready_problems(bank, counts) == []
    audit_path = exp / 'report/audit/summary.json'
    audit_path.parent.mkdir(parents=True)
    audit = {'schema': 'hi4d_grounded_v5_independent_audit_v1', 'status': 'PASS', 'bank': str(bank),
             'counts': {'rows': 1, 'targets': 1, 'persons': 2, 'rows_checked': 1, 'split_rows': {'test': 1}},
             'error_count': 0, 'producer': {'sha256': v.sha256(bank / 'provenance/hi4d_grounded_v5.py')},
             'validator': {'sha256': v.sha256(SCRIPT)},
             'bank_sha256': {'test': v.sha256(bank / 'prompts/test.jsonl')}}
    audit_path.write_text(json.dumps(audit))
    ready = {'schema': 'hi4d_grounded_prompt_release_v1', 'release': bank.name, 'prompt_schema': v.SCHEMA,
             'status': 'READY_DATA_AND_BINDING', 'counts': counts, 'producer_sha256': audit['producer']['sha256'],
             'independent_audit': '../report/audit/summary.json', 'independent_audit_rows': 1, 'independent_audit_errors': 0}
    (bank / 'READY.json').write_text(json.dumps(ready))
    assert v.ready_problems(bank, counts) == []
    ready['status'] = 'NOT_READY'
    (bank / 'READY.json').write_text(json.dumps(ready))
    assert any('status' in e for e in v.ready_problems(bank, counts))
    ready['status'] = 'READY_DATA_AND_BINDING'
    (bank / 'READY.json').write_text(json.dumps(ready))
    audit['bank'] = '/unrelated/bank'
    audit_path.write_text(json.dumps(audit))
    assert any('audit.bank' in e for e in v.ready_problems(bank, counts))
    audit['bank'] = str(bank)
    audit['bank_sha256']['test'] = '0' * 64
    audit_path.write_text(json.dumps(audit))
    assert any('audit.bank_sha256' in e for e in v.ready_problems(bank, counts))
