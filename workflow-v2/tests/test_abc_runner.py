"""ABC stage evidence, non-destructive reuse and public CLI contracts."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def module():
    assert (ROOT / 'abc_runner.py').is_file(), 'ABC runner must be implemented'
    return importlib.import_module('abc_runner')


def test_cli_defaults_and_explicit_checkpoint():
    api = module()
    args = api.parse_args(['--source-a', 'a.png', '--source-b', 'b.png',
        '--interaction', 'c.png', '--name', 'abc', '--checkpoint', 'weights.pt'])
    assert args.mode == 'native_v5'
    assert args.role_map == 'A=left,B=right'
    assert str(args.run_root) == '/data/shangguanrz/pic-edit-workflow-v2/runs'
    with pytest.raises(SystemExit):
        api.parse_args(['--source-a', 'a', '--source-b', 'b', '--interaction', 'c'])


def test_cli_rejects_pair_mixing_and_unsafe_names():
    api = module()
    base = ['--source-a', 'a', '--source-b', 'b', '--interaction', 'c',
        '--mode', 'geometry_only']
    with pytest.raises(SystemExit):
        api.parse_args(base + ['--pair', 'old'])
    with pytest.raises(SystemExit):
        api.parse_args(base + ['--name', '../escape'])


def test_mode_independent_shared_stage_reuses_without_worker(tmp_path):
    api = module()
    calls = []

    def worker(request):
        calls.append(request)
        directory = Path(request['output_dir'])
        (directory / 'scene.json').write_text('{"scene":1}', encoding='utf-8')

    request = {'mode': 'geometry_only'}
    one = api.execute_stage(tmp_path, 'geometry', {'inputs': 'same'}, request, worker)
    two = api.execute_stage(tmp_path, 'geometry', {'inputs': 'same'},
        {'mode': 'native_v5'}, lambda request: pytest.fail('must reuse geometry'))
    assert len(calls) == 1
    assert one['outputs'] == two['outputs']


def test_verify_is_read_only_and_never_runs_worker(tmp_path):
    api = module()

    def worker(request):
        (Path(request['output_dir']) / 'scene.json').write_text('{}', encoding='utf-8')

    api.execute_stage(tmp_path, 'geometry', {'source': 'a'}, {}, worker)
    before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in tmp_path.rglob('*') if p.is_file()}
    api.execute_stage(tmp_path, 'geometry', {'source': 'a'}, {},
        lambda request: pytest.fail('verify ran worker'), verify=True)
    assert before == {str(p): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in tmp_path.rglob('*') if p.is_file()}
    with pytest.raises(RuntimeError, match='new --name'):
        api.execute_stage(tmp_path, 'prompt', {}, {}, worker, verify=True)
    assert not (tmp_path / 'prompt').exists()


@pytest.mark.parametrize('mutation', ['artifact', 'identity'])
def test_stale_stage_rejects_without_overwriting(tmp_path, mutation):
    api = module()

    def worker(request):
        (Path(request['output_dir']) / 'image.png').write_bytes(b'original-image')

    api.execute_stage(tmp_path, 'geometry', {'source': 'a'}, {}, worker)
    identity = {'source': 'b' if mutation == 'identity' else 'a'}
    output = tmp_path / 'geometry' / 'image.png'
    if mutation == 'artifact':
        output.write_bytes(b'changed-image')
    before = output.read_bytes()
    with pytest.raises(RuntimeError, match='new --name'):
        api.execute_stage(tmp_path, 'geometry', identity, {}, worker)
    assert output.read_bytes() == before


def test_recursive_model_content_identity_detects_same_size_change(tmp_path):
    api = module()
    nested = tmp_path / 'nested'
    nested.mkdir()
    asset = nested / 'weights.pt'
    asset.write_bytes(b'one')
    first = api.asset_identity(tmp_path)
    asset.write_bytes(b'two')
    assert api.asset_identity(tmp_path) != first


def test_generation_modes_have_separate_artifact_directories(tmp_path):
    api = module()

    def worker(request):
        (Path(request['output_dir']) / 'final_image.png').write_bytes(
            request['mode'].encode())

    for mode in ('qwen_baseline', 'native_v5'):
        api.execute_stage(tmp_path, f'generation/{mode}', {'mode': mode},
            {'mode': mode}, worker)
    assert (tmp_path / 'generation/qwen_baseline/final_image.png').read_bytes() == b'qwen_baseline'
    assert (tmp_path / 'generation/native_v5/final_image.png').read_bytes() == b'native_v5'


def test_new_workflow_has_its_own_entrypoint():
    text = (ROOT / 'run.py').read_text(encoding='utf-8')
    assert 'abc_runner' in text
    assert (ROOT / 'run.sh').is_file()


def test_request_without_scene_never_has_raw_interaction_reference(tmp_path):
    contracts = importlib.import_module('abc_contracts')
    assert contracts.parse_role_map('A=right,B=left') == {'A': 'right', 'B': 'left'}
    with pytest.raises(ValueError):
        contracts.parse_role_map('A=left,B=left')
    scene = {'sources': [{'person_id': 'source_A', 'canvas_image': '/a.png'},
        {'person_id': 'source_B', 'canvas_image': '/b.png'}],
        'guides': {'person_0': '/ga.png', 'person_1': '/gb.png', 'combined': '/gab.png'}}
    assert contracts.reference_paths(scene) == ['/a.png', '/b.png', '/ga.png', '/gb.png', '/gab.png']

def test_failed_generation_retries_in_new_attempt_without_overwriting(tmp_path):
    api=module()
    def failed(request):
        (Path(request['output_dir'])/'partial.json').write_text('old evidence')
        raise RuntimeError('interrupted inference')
    with pytest.raises(RuntimeError,match='interrupted'):
        api.execute_stage(tmp_path,'generation/qwen_baseline',{'code':1},{},failed)
    def success(request):
        (Path(request['output_dir'])/'final_image.png').write_bytes(b'new image')
        (Path(request['output_dir'])/'generation.json').write_text('{}')
    api.execute_stage(tmp_path,'generation/qwen_baseline',{'code':2},{},success)
    assert (tmp_path/'generation/qwen_baseline/partial.json').read_text()=='old evidence'
    assert (tmp_path/'generation/qwen_baseline/final_image.png').read_bytes()==b'new image'
