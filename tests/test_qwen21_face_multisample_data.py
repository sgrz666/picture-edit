"""Fail-closed checks for the experiment's split and sampling contracts."""
from copy import deepcopy

import pytest

from scripts.data.prepare_qwen21_face_multisample import PREFERRED, validate_sets


def example(app, index, role='source_motion'):
    source_number = 0 if role == 'source_standard' else 1 + index % 2
    return {'appearance': app,
            'source': {'stem': f'{role}_{source_number}', 'role': role, 'timestamp_sec': 0., 'rgb_1024': 'source.png'},
            'target': {'stem': f'target_{index}', 'role': 'target', 'timestamp_sec': 5. + index, 'rgb_1024': 'target.png'}}


@pytest.fixture
def manifests():
    body, face = [], []
    for apps in PREFERRED.values():
        for app in apps:
            group = [example(app, i, 'source_standard' if i % 2 == 0 else 'source_motion') for i in range(8)]
            # Two different motion sources are needed in each Face subset.
            group[1]['source']['stem'] = 'source_motion_1'
            group[3]['source']['stem'] = 'source_motion_2'
            body.extend(group)
            face.extend(group[:4])
    val = [example(app, i) for app in ['001_2', '001_5', '006_1', '009_3'] for i in [0]]
    val += [example(app, i) for app in ['012_4', '015_1', '019_3', '024_2'] for i in [0, 1]]
    cross = [{'source_appearance': '001_2', 'target_appearance': '006_1',
              'source': val[0]['source'], 'target': example('006_1', i)['target']} for i in [2, 3]]
    final = [example('001_19', 0)]
    return body, face, val, cross, deepcopy(body), deepcopy(val), final


def test_balanced_manifests_acceptance(manifests):
    assert validate_sets(*manifests)['face_subset_of_body']


def test_final_appearance_leak_is_rejected(manifests):
    manifests[-1][0]['appearance'] = '012_4'
    with pytest.raises(ValueError, match='appearances overlap'):
        validate_sets(*manifests)


def test_validation_cannot_supply_training_pair(manifests):
    manifests[0][0] = deepcopy(manifests[2][0])
    with pytest.raises(ValueError):
        validate_sets(*manifests)


def test_face_pair_outside_body_is_rejected(manifests):
    manifests[1][0] = example('001_12', 999, 'source_standard')
    with pytest.raises(ValueError, match='subset'):
        validate_sets(*manifests)


def test_adjacent_source_target_pair_is_rejected(manifests):
    manifests[0][0]['target']['timestamp_sec'] = 1.
    with pytest.raises(ValueError, match='temporal gap'):
        validate_sets(*manifests)


def test_missing_distinct_reference_is_rejected(manifests):
    for row in manifests[0][:8]:
        if row['source']['role'] == 'source_motion':
            row['source']['stem'] = 'source_motion_single'
    manifests[4][:8] = deepcopy(manifests[0][:8])
    with pytest.raises(ValueError, match='three distinct'):
        validate_sets(*manifests)
