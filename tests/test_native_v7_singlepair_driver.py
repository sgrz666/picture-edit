"""CPU-only checks for the private pilot's stage and artifact boundaries."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest


def driver():
    path = Path(__file__).resolve().parents[1] / 'experiments/hi4d_qwen21_dual_20260930/native-v7-singlepair-overfit-20261117/scripts/run_training.py'
    spec = importlib.util.spec_from_file_location('v7_singlepair_driver', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase_targets_are_successful_updates_not_per_stage_restarts():
    configs = [{'stage': {'phase': phase, 'budget_updates': budget}}
               for phase, budget in zip(('S0', 'S1', 'S2', 'S3'), (30, 30, 120, 120))]
    assert driver().phase_targets(configs) == (30, 60, 180, 300)
    configs[2]['stage']['budget_updates'] = 30
    with pytest.raises(ValueError, match='budget'):
        driver().phase_targets(configs)


def completed_metadata():
    parent = {'path': '/data/parent.pt', 'sha256': 'a' * 64}
    config = {'stage': {'phase': 'S1', 'budget_updates': 30}, 'initialization': parent,
              'gradient_report': {'path': '/data/S1-grad.json', 'sha256': 'b' * 64}}
    summary = {'budget_completed': True, 'requested_updates': 30,
               'completed_stage_updates': 30, 'global_updates': 60}
    state = {'config': deepcopy(config), 'config_sha256': 'signature', 'global_update': 60,
             'stage_update': 30, 'parent': parent, 'stage_origin': parent,
             'gradient_report': deepcopy(config['gradient_report'])}
    return config, summary, state, parent


def test_completed_metadata_accepts_exact_parent_and_config():
    config, summary, state, parent = completed_metadata()
    driver().check_completed_stage(summary, state, config, 'signature', 60, parent)


@pytest.mark.parametrize('mutation', ('partial', 'global', 'parent', 'signature', 'phase', 'gradient'))
def test_completed_metadata_rejects_plausible_but_unrelated_checkpoints(mutation):
    config, summary, state, parent = completed_metadata()
    if mutation == 'partial':
        summary['budget_completed'] = False
    elif mutation == 'global':
        state['global_update'] = 30
    elif mutation == 'parent':
        state['parent'] = {'path': '/data/other.pt', 'sha256': 'c' * 64}
    elif mutation == 'signature':
        state['config_sha256'] = 'different'
    elif mutation == 'phase':
        state['config']['stage']['phase'] = 'S0'
    else:
        state['gradient_report']['sha256'] = 'c' * 64
    with pytest.raises(ValueError, match='completed stage'):
        driver().check_completed_stage(summary, state, config, 'signature', 60, parent)


def test_new_segment_selection_excludes_old_and_unrelated_jobs(tmp_path):
    wanted = {'stage': {'phase': 'S1'}, 'initialization': {'path': '/data/parent.pt'}}
    for name, config in (('old', wanted), ('new', wanted), ('foreign', {'stage': {'phase': 'S3'}})):
        segment = tmp_path / name
        segment.mkdir()
        (segment / 'resolved.json').write_text(json.dumps({'config': config}))
        (segment / 'summary.json').write_text(json.dumps({'checkpoint': name}))
    selected, summary = driver().new_stage_summary(tmp_path, {tmp_path / 'old'}, wanted)
    assert selected == tmp_path / 'new'
    assert summary['checkpoint'] == 'new'
    with pytest.raises(ValueError, match='exactly one'):
        driver().new_stage_summary(tmp_path, set(), wanted)


@pytest.mark.parametrize('action', ('preflight', 'train', 'advance-stage'))
def test_vae_tiling_flag_reaches_wrapper_before_shared_cli_boundary(tmp_path, action):
    command = driver().child_command(tmp_path, [action, '--config', 'ready.json'],
                                     resource=tmp_path / 'resources.json', vae_tiling=True)
    boundary = command.index('--')
    assert command[2] == str(tmp_path / 'scripts/gpu_action.py')
    assert '--vae-tiling' in command[:boundary]
    assert command[boundary + 1:] == [action, '--config', 'ready.json']


def test_generation_child_gets_same_explicit_tiling_policy(tmp_path):
    arguments = ['run_training.py', 'generate', '--config', 'ready.json']
    enabled = driver().child_command(tmp_path, arguments, vae_tiling=True)
    disabled = driver().child_command(tmp_path, arguments, vae_tiling=False)
    assert enabled == [sys.executable, '-u', *arguments, '--vae-tiling']
    assert disabled == [sys.executable, '-u', *arguments]


@pytest.mark.parametrize('action', ('preflight', 'train', 'advance-stage'))
def test_hidden_offload_is_independent_of_vae_tiling_and_reaches_each_gpu_action(tmp_path, action):
    command = driver().child_command(tmp_path, [action], resource=tmp_path / 'resources.json', offload_hidden=True)
    boundary = command.index('--')
    assert '--offload-hidden' in command[:boundary]
    assert '--vae-tiling' not in command
    assert command[boundary + 1:] == [action]


def test_generation_receives_independent_offload_policy_for_reporting(tmp_path):
    command = driver().child_command(tmp_path, ['run_training.py', 'generate'],
                                     vae_tiling=True, offload_hidden=True)
    assert command[-2:] == ['--vae-tiling', '--offload-hidden']


@pytest.mark.parametrize('action', ([], ['generate']))
def test_both_cli_parsers_expose_vae_tiling_without_loading_cuda(action):
    path = driver().__file__
    result = subprocess.run([sys.executable, path, *action, '--help'], capture_output=True, text=True, check=True)
    assert '--vae-tiling' in result.stdout
    assert '--offload-hidden' in result.stdout


@pytest.mark.parametrize('action',('preflight','train','advance-stage'))
def test_native_full_gpu_budget_reaches_every_child(tmp_path,action):
    command=driver().child_command(tmp_path,[action],resource=tmp_path/'resource.json',memory_fraction=.9)
    boundary=command.index('--')
    position=command.index('--memory-fraction')
    assert position<boundary and command[position+1]=='0.9'
    assert command[boundary+1:]==[action]


def test_generation_receives_same_full_gpu_budget(tmp_path):
    command=driver().child_command(tmp_path,['run_training.py','generate'],memory_fraction=.9)
    assert command[-2:]==['--memory-fraction','0.9']
