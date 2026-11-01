"""The active Hi4D bank must reach a fresh, SHA-bound consumer closure."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.fixture
def prep(tmp_path, monkeypatch):
    value = module(ROOT / 'scripts/data/prepare_qwen21_fixedpair_v5.py', 'fresh_hi4d_prep')
    monkeypatch.setattr(value, 'DATA_ROOT', tmp_path, raising=False)
    monkeypatch.setattr(value, 'HROOT', tmp_path / 'old_live')
    return value


def bank(tmp_path, rows=None):
    text = 'Person 1 is the person on the left in <image1>. <image3> guides Person 1; <image4> guides Person 2.'
    row = {'schema': 'hi4d_pei2i_q4_plan_a_v5', 'pair_id': 'train_p1',
           'left_actor': 'B', 'right_actor': 'A', 'compiled_prompt_inference': text,
           'compiled_prompt_inference_sha256': hashlib.sha256(text.encode()).hexdigest()}
    path = tmp_path / 'bank.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in (rows or [row])), encoding='utf8')
    return path


def test_lock_requires_an_explicit_fresh_hi4d_root(prep):
    with pytest.raises(ValueError, match='hi4d_root'):
        prep.lock(kind='hi4d')


def test_lock_materializes_actual_v5_text_and_bank(prep, tmp_path):
    path = bank(tmp_path)
    dst = tmp_path / 'fresh_pilot'
    prep.lock(kind='hi4d', hi4d_root=dst, bank=path, pair_ids=['train_p1'], roles=['train'])
    result = json.loads((dst / 'data/selection.json').read_text())
    assert result['pairs'][0]['bank_record']['schema'] == 'hi4d_pei2i_q4_plan_a_v5'
    assert result['pairs'][0]['bank'] == result['bank']
    assert result['bank']['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result['training_ready'] is False
    assert result['frozen_cache_status'] == 'not_encoded'
    assert result['updates'] is None and result['stage_budgets'] is None
    assert 'continuous_stage_authorization' not in result


def test_lock_refuses_old_live_root_before_writing(prep, tmp_path):
    path = bank(tmp_path)
    with pytest.raises(ValueError, match='old|live|existing'):
        prep.lock(kind='hi4d', hi4d_root=prep.HROOT, bank=path, pair_ids=['train_p1'])
    assert not prep.HROOT.exists()


def test_lock_refuses_existing_output_before_writing(prep, tmp_path):
    path = bank(tmp_path)
    dst = tmp_path / 'existing'
    dst.mkdir()
    with pytest.raises(FileExistsError):
        prep.lock(kind='hi4d', hi4d_root=dst, bank=path, pair_ids=['train_p1'])
    assert list(dst.iterdir()) == []


@pytest.mark.parametrize('bad', ['schema', 'sha'])
def test_lock_checks_content_before_creating_root(prep, tmp_path, bad):
    path = bank(tmp_path)
    row = json.loads(path.read_text())
    row['schema' if bad == 'schema' else 'compiled_prompt_inference_sha256'] = 'bad'
    path.write_text(json.dumps(row) + '\n')
    dst = tmp_path / 'rejected'
    with pytest.raises(ValueError, match='schema|fingerprint'):
        prep.lock(kind='hi4d', hi4d_root=dst, bank=path, pair_ids=['train_p1'])
    assert not dst.exists()


def test_lock_rejects_output_outside_persistent_volume(prep, tmp_path):
    path = bank(tmp_path)
    with pytest.raises(ValueError, match='persistent|DATA_ROOT'):
        prep.lock(kind='hi4d', hi4d_root=tmp_path.parent / 'outside', bank=path, pair_ids=['train_p1'])


def test_legacy_bank_requires_explicit_schema(prep, tmp_path):
    path = bank(tmp_path)
    row = json.loads(path.read_text())
    row['schema'] = 'hi4d_pei2i_q4_plan_a_v4'
    path.write_text(json.dumps(row) + '\n')
    prep.lock(kind='hi4d', hi4d_root=tmp_path / 'explicit_legacy', bank=path,
              expected_schema=row['schema'], pair_ids=['train_p1'])


def test_active_bank_defaults_are_v5(prep):
    assert prep.HI4D_SCHEMA == 'hi4d_pei2i_q4_plan_a_v5'
    assert 'v5_grounded_3d_20261030' in str(prep.HI4D_BANK)


def pilot_module():
    paths = [ROOT / 'experiments' / name / 'scripts/build_v5_binding_pilot.py'
             for name in ('hi4d_pe_prompt_rewrite_20261020', '4.two man /hi4d_pe_prompt_rewrite_20261020')]
    path = next((p for p in paths if p.is_file()), None)
    assert path is not None, 'Fresh V5 consumer builder is missing'
    return module(path, 'fresh_hi4d_builder')


def test_representatives_cover_identity_crossing_and_regressions():
    pilot = pilot_module()
    rs = [dict(pair_id='a', left_actor='A', is_crossed=False, action='hug'),
          dict(pair_id='b', left_actor='B', is_crossed=False, action='hug'),
          dict(pair_id='c', left_actor='A', is_crossed=True, action='hug'),
          dict(pair_id='y', left_actor='B', is_crossed=False, action='yoga92'),
          dict(pair_id='d', left_actor='A', is_crossed=False, action='dance2')]
    chosen = pilot.representative_ids(rs)
    assert set(chosen) == {'a', 'b', 'c', 'y', 'd'}


def test_bank_identity_must_equal_source_left():
    pilot = pilot_module()
    row = {'pair_id': 'wrong', 'left_actor': 'B', 'right_actor': 'A',
           'per_person': {'person_1': {'identity': 'A'}, 'person_2': {'identity': 'B'}}}
    with pytest.raises(ValueError, match='identity|source-left'):
        pilot.validate_person_binding(row)


def test_recomputed_outer_hash_cannot_authorize_swapped_guide_identities():
    pilot = pilot_module()
    case = {'prompt': 'text', 'prompt_sha256': 'textsha', 'bank': {'path': 'b', 'sha256': 'banksha'},
            'bank_record': {'left_actor': 'B'}}
    a = {'path': 'A.png', 'sha256': 'A'}
    b = {'path': 'B.png', 'sha256': 'B'}
    condition = {'prompt': {'text': 'text', 'sha256': 'textsha', 'bank': case['bank']},
                 'person_order': {'slot0_identity': 'B', 'slot1_identity': 'A'},
                 'guide_identity_files': {'A': a, 'B': b},
                 'images': [{}, {}, {'file': a}, {'file': b}]}
    with pytest.raises(ValueError, match='guide identity'):
        pilot.verify_binding(condition, case)
    condition['images'][2]['file'], condition['images'][3]['file'] = b, a
    pilot.verify_binding(condition, case)
