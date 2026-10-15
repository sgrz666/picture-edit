from pathlib import Path
import pytest
from scripts.data import prepare_qwen21_prompt_study as prepare


def test_existing_invalid_lock_is_checked_before_any_copy(tmp_path, monkeypatch):
    data = tmp_path / 'run/data'
    data.mkdir(parents=True)
    lock = data / 'lock.json'
    lock.write_text('{}')
    original = data / 'train512_pairs.jsonl'
    original.write_bytes(b'original data')
    def reject(*args, **kwargs):
        raise ValueError('invalid existing lock')
    monkeypatch.setattr(prepare, 'load_study_lock', reject)
    monkeypatch.setattr(prepare.shutil, 'copyfile', lambda *a: pytest.fail('Copied before verification'))
    with pytest.raises(ValueError, match='existing lock'):
        prepare.prepare_data_lock(tmp_path / 'source', tmp_path / 'candidates', tmp_path / 'diag', data)
    assert original.read_bytes() == b'original data'


def test_valid_existing_lock_is_reused_without_reselection(tmp_path, monkeypatch):
    data = tmp_path / 'run/data'
    data.mkdir(parents=True)
    (data / 'lock.json').write_text('{}')
    existing = {'artifacts': {key: {'path': str(path.resolve())} for key, path in (
        ('model', prepare.DEFAULT_MODEL), ('initial_checkpoint', prepare.DEFAULT_INIT),
        ('head_lora', prepare.DEFAULT_HEAD), ('pe_model', prepare.DEFAULT_PE))}}
    monkeypatch.setattr(prepare, 'load_study_lock', lambda *a: existing)
    monkeypatch.setattr(prepare.shutil, 'copyfile', lambda *a: pytest.fail('Existing run was rewritten'))
    assert prepare.prepare_data_lock(tmp_path / 'source', tmp_path / 'candidates', tmp_path / 'diag', data) is existing
