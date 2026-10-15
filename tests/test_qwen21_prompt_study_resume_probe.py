from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import torch


def test_private_probe_runs_ten_updates_with_exact_shared_step2(tmp_path, monkeypatch):
    script = Path(__file__).parents[1] / 'scripts/run/check_qwen21_prompt_study_resume.py'
    assert script.is_file(), 'mandatory study resume probe is missing'
    probe = importlib.import_module('scripts.run.check_qwen21_prompt_study_resume')
    calls = []
    lock = tmp_path / 'data/lock.json'
    lock.parent.mkdir()
    lock.write_text('{}')
    monkeypatch.setattr(probe, 'locked_assets', lambda _: {'model': {'path': '/model'}, 'initial_checkpoint': {'path': '/init'}, 'head_lora': {'path': '/head'}})
    def invoke(argv):
        calls.append(argv)
        get = lambda flag: argv[argv.index(flag) + 1]
        output = Path(get('--output'))
        output.mkdir(parents=True, exist_ok=True)
        stop = int(get('--stop-after-step'))
        start = 0
        prefix = b''
        if '--resume' in argv:
            source = Path(get('--resume'))
            saved = torch.load(source, weights_only=False)
            start = saved['step']
            prefix = (output / 'metrics.jsonl').read_bytes()
            (output / f'resume_state_step_{start:06d}.json').write_text(json.dumps({'adapter_restored_exact': True, 'optimizer_restored_exact': True, 'rng_restored_exact': True, 'checkpoint_step': start, 'checkpoint_sha256': probe.sha256_file(source), 'joint_signature': saved['joint_signature']}))
        signature = {'arm': 'qwen_pg', 'max_updates': 1024}
        state = {'step': stop, 'sample_order': list(range(stop)), 'joint_signature': signature, 'joint_initialization_sha256': 'init', 'adapter_state_dict': {'weight': torch.tensor([1.0])}, 'optimizer_state_dict': {'state': {}}, 'rng_torch': torch.tensor([42]), 'arm': 'qwen_pg'}
        state.update(torch_rng_state=torch.tensor([42]), cuda_rng_state=[], python_rng_state=(42,), numpy_rng_state=(42,))
        torch.save(state, output / f'checkpoint_step_{stop:06d}.pt')
        torch.save(state, output / 'checkpoint_last.pt')
        rows = [{'step': i + 1, 'pair_index': i, 'pair_id': f'p{i}', 'noise_seed': 298 + i, 'prompt_sha256': 'prompt', 'sigma': .5, 'face_active': True, 'loss': 1., 'body_flow': .8, 'face_flow': .2, 'identity': 0., 'perceptual': 0.} for i in range(start, stop)]
        (output / 'metrics.jsonl').write_bytes(prefix + ''.join(json.dumps(r) + '\n' for r in rows).encode())
        if start == 0:
            (output / 'run_manifest.json').write_text(json.dumps({'joint_signature': signature, 'controls': {'body': 1, 'face': 1}, 'optimizer_parameters': ['weight']}))
        (output / 'summary.json').write_text(json.dumps({'steps': stop, 'completed': False, 'max_updates': 1024, 'frozen_transformer_unchanged': True, 'frozen_adapter_unchanged': True}))
    monkeypatch.setattr(probe, 'invoke', invoke)
    result = probe.run_resume_check(tmp_path)
    assert [Path(c[c.index('--output') + 1]).name for c in calls] == ['direct', 'direct', 'resume', 'repeat1', 'repeat2']
    assert [c[c.index('--stop-after-step') + 1] for c in calls] == ['2', '4', '4', '4', '4']
    assert all(c[c.index('--steps') + 1] == '1024' for c in calls)
    assert result['private_optimizer_updates'] == 10
    assert result['mandatory_exact_gate']['passed'] is True
    folder = tmp_path / 'preflight/resume_check'
    shared = (folder / 'direct/checkpoint_step_000002.pt').read_bytes()
    prefix = b''.join((folder / 'direct/metrics.jsonl').read_bytes().splitlines(keepends=True)[:2])
    for name in ('resume', 'repeat1', 'repeat2'):
        assert (folder / name / 'checkpoint_step_000002.pt').read_bytes() == shared
        assert (folder / name / 'metrics.jsonl').read_bytes().startswith(prefix)
    (folder / 'resume/resume_state_step_000002.json').write_text('{}')
    damaged = probe.compare_resume(tmp_path)
    assert damaged['mandatory_exact_gate']['passed'] is False


def test_private_probe_preserves_existing_incomplete_artifacts(tmp_path):
    script = Path(__file__).parents[1] / 'scripts/run/check_qwen21_prompt_study_resume.py'
    assert script.is_file(), 'mandatory study resume probe is missing'
    probe = importlib.import_module('scripts.run.check_qwen21_prompt_study_resume')
    directory = tmp_path / 'preflight/resume_check/direct'
    directory.mkdir(parents=True)
    (directory / 'failure.txt').write_text('evidence')
    with pytest.raises(FileExistsError):
        probe.run_resume_check(tmp_path)
    assert (directory / 'failure.txt').read_text() == 'evidence'
