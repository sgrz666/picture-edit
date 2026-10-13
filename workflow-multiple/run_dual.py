#!/usr/bin/env python3
"""One-command two-person interaction transfer; all artifacts stay in workflow-multiple/runs."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dual_geometry import GUIDE_FRAMINGS, produce
from dual_prompts import GENERATOR_ROLES, generator_allowed
from src.pose_edit.common import (atomic_json, file_lock, finish_stage, make_request,
    model_signature, read_json, sha256_file, stage_outputs, stage_reasons, stage_valid)
from src.pose_edit.prompt_contract import PE_SAMPLING, validate_generator_prompt

ROOT = Path(__file__).resolve().parent
STAGES = ('geometry', 'prompt', 'generate')
STAGE_DIRECTORY = {'geometry': 'geometry', 'prompt': 'prompt', 'generate': 'generation'}
STAGE_SCRIPT = {'geometry': None, 'prompt': 'pe_dual_worker.py',
    'generate': 'generate_dual_worker.py'}
SHARED_CODE = ['src/pose_edit/common.py', 'src/pose_edit/geometry.py']
PROMPT_CODE = SHARED_CODE + ['src/pose_edit/prompt_contract.py', 'dual_prompts.py']


def discover_pair(explicit):
    directory = Path(explicit).expanduser().resolve()
    if not (directory / 'pair.json').is_file():
        found = sorted(item.name for item in directory.iterdir()) if directory.is_dir() else []
        raise FileNotFoundError(f'--pair needs a directory holding pair.json; got {directory} '
            f'with {found}')
    return directory


def code_identity(names):
    return {name: sha256_file((REPO_ROOT if name.startswith('src/') else ROOT) / name)
        for name in names}


def pair_inputs(pair_dir):
    pair = read_json(pair_dir / 'pair.json')
    files = {'pair_json': pair_dir / 'pair.json', 'A_original': pair_dir / pair['A']['image'],
        'B_original': pair_dir / pair['B']['image']}
    for item in pair['B']['keypoints']:
        files[f"B_keypoints_{item['slot']}"] = pair_dir / item['file']
    return pair, {name: Path(path) for name, path in files.items()}


def run_worker(python, script, request, request_path, log_path):
    atomic_json(request_path, request)
    print(f'RUN {script}; log={log_path}', flush=True)
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.run([python, '-u', str(ROOT / script), '--request', str(request_path)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if process.returncode:
        tail = '\n'.join(log_path.read_text(encoding='utf-8', errors='replace').splitlines()[-35:])
        raise RuntimeError(f'{script} failed with exit code {process.returncode}.\n{tail}')


def dependency_check(config):
    required = [config['geometry_python'], config['qwen_python'],
        Path(config['pe_model']) / 'system_prompt.txt', config['qwen_model'], config['hi4d_manifest']]
    for name in required:
        if not Path(name).exists():
            raise FileNotFoundError(f'Required asset not found: {name}')
    print('ASSET_PATHS_OK', flush=True)
    subprocess.run([config['qwen_python'], '-c',
        'import torch,transformers,numpy,PIL; from diffusers import QwenImage21Pipeline; '
        'torch.cuda.init(); print("QWEN_CUDA_OK",torch.cuda.get_device_name(0),'
        '"free/total GiB",*[round(x/1024**3,2) for x in torch.cuda.mem_get_info()])'], check=True)


def stage_identity(stage, config, paths, pair_dir):
    """Everything a cached stage must still match; never an absolute path."""
    arm = config['generator_references']
    if stage == 'geometry':
        settings = {key: config[key] for key in
            ('framing', 'resolution', 'keypoint_score_threshold')}
        input_names = ['pair_json', 'A_original', 'B_original', 'B_keypoints_1', 'B_keypoints_2']
        identities = None
        code = SHARED_CODE + ['dual_geometry.py']
    elif stage == 'prompt':
        settings = {key: config[key] for key in
            ('pe_model', 'pe_seed', 'prop_policy', 'generator_references')}
        input_names = ['A_reference', 'B_reference', 'skeleton_1', 'skeleton_2']
        identities = model_signature(config['pe_model'])
        code = PROMPT_CODE + ['pe_dual_worker.py']
    else:
        settings = {key: config[key] for key in ('qwen_model', 'resolution', 'steps', 'seed',
            'true_cfg_scale', 'use_kv_cache', 'adapter', 'lora', 'generator_references')}
        input_names = list(GENERATOR_ROLES[arm]) + ['generation_prompt', 'source_preprocess']
        identities = model_signature(config['qwen_model'])
        code = ['src/pose_edit/common.py', 'src/pose_edit/prompt_contract.py',
            'dual_prompts.py', 'generate_dual_worker.py']
    identity = {'settings': settings,
        'inputs_sha256': {key: sha256_file(paths[key]) for key in input_names},
        'code': code_identity(code)}
    if identities is not None:
        identity['model_identity'] = identities
    if stage == 'prompt':
        identity['sampling'], identity['enable_thinking'] = PE_SAMPLING, True
        identity['generator_allowed'] = sorted(generator_allowed(arm))
    return identity, input_names


def stage_request(stage, config, paths, pair_dir, run_dir):
    identity, input_names = stage_identity(stage, config, paths, pair_dir)
    runtime = {key: paths[key] for key in input_names}
    runtime.update({key: paths[key] for key in ('generation_prompt_provenance',)
        if key in paths})
    return make_request(stage, identity, run_dir, {'run_dir': str(run_dir),
        'pair_dir': str(pair_dir), **runtime})


def select_generation_prompt(run_dir, paths, manual_prompt, verify=False, arm='with_b'):
    """Freeze the text the generator reads without touching the prompt engineer's own output."""
    compiled = Path(paths['pe_compiled_prompt'])
    source = Path(manual_prompt).expanduser().resolve() if manual_prompt else compiled
    text = source.read_text(encoding='utf-8').strip()
    if not text:
        raise ValueError(f'Prompt file is empty: {source}')
    validate_generator_prompt(text, allowed=generator_allowed(arm))
    target = run_dir / 'prompt' / 'generation_prompt.txt'
    provenance = run_dir / 'prompt' / 'generation_prompt.json'
    selected = {'generation_prompt': str(target), 'generation_prompt_provenance': str(provenance)}
    if verify:
        recorded = read_json(provenance) if provenance.is_file() else None
        if not target.is_file() or target.read_text(encoding='utf-8') != text + '\n':
            raise RuntimeError(f'{target} does not hold the selected prompt; rerun without '
                '--verify to materialise it')
        if not recorded or recorded.get('sha256') != sha256_file(target):
            raise RuntimeError(f'{provenance} does not describe the selected prompt')
        return selected
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file() or target.read_text(encoding='utf-8') != text + '\n':
        target.write_text(text + '\n', encoding='utf-8')
    atomic_json(provenance, {'arm': arm, 'source': 'manual' if manual_prompt else 'pe_compiled',
        'source_path': str(source), 'source_sha256': sha256_file(source),
        'generation_prompt': str(target), 'sha256': sha256_file(target)})
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--pair', type=Path, help='Directory holding pair.json')
    parser.add_argument('--name', help='Run directory name, default the pair directory name')
    parser.add_argument('--until', choices=STAGES, default='generate')
    parser.add_argument('--framing', choices=GUIDE_FRAMINGS)
    parser.add_argument('--generator-references', choices=sorted(GENERATOR_ROLES))
    parser.add_argument('--prompt-file', type=Path, help='Generate with this prompt text instead '
        'of the prompt-engineering output; PE stays cached and is never overwritten')
    parser.add_argument('--resolution', type=int)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.force and args.verify:
        parser.error('--force and --verify are mutually exclusive')
    config = read_json(args.config)
    for key, value in (('resolution', args.resolution), ('steps', args.steps), ('seed', args.seed),
            ('framing', args.framing), ('generator_references', args.generator_references)):
        if value is not None:
            config[key] = value
    if config['generator_references'] not in GENERATOR_ROLES:
        parser.error('--generator-references must be with_b or skeleton_only')
    if config['framing'] not in GUIDE_FRAMINGS:
        parser.error('--framing must be b_canvas or source_fit')
    if (config.get('adapter') is not False or config.get('lora') is not False
            or config.get('prop_policy') != 'source_only'):
        parser.error('This workflow is fixed to source-only objects and no adapter/LoRA')
    if config['resolution'] < 256 or config['steps'] < 1:
        parser.error('resolution must be >=256 and steps must be positive')
    if args.check:
        dependency_check(config)
        return
    if not args.pair:
        parser.error('--pair is required: a directory holding pair.json, A_source.png and B_target.png')
    pair_dir = discover_pair(args.pair)
    name = args.name or pair_dir.name
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', name):
        parser.error('--name must be a simple directory name')
    run_dir = ROOT / 'runs' / name
    if args.verify and not run_dir.is_dir():
        raise RuntimeError(f'No existing run to verify: {run_dir}')
    with file_lock(ROOT / '.workflow.lock'):
        completed, checked = [], []
        manifest = {'schema': 'dual_pose_workflow_v1', 'status': 'running',
            'pair_dir': str(pair_dir), 'settings': config, 'requested_until': args.until,
            'prompt_source': str(args.prompt_file) if args.prompt_file else None,
            'stages': completed, 'outputs': {}, 'started_at_unix': time.time()}
        try:
            pair, paths = pair_inputs(pair_dir)
            manifest['pair'] = {'name': pair['name'], 'kind': pair['kind'],
                'manifest': pair['manifest'], 'scoring': pair['scoring']}
            if not args.verify:
                atomic_json(run_dir / 'run.json', manifest)
            for stage in STAGES:
                directory = run_dir / STAGE_DIRECTORY[stage]
                record = directory / 'stage.json'
                request = stage_request(stage, config, paths, pair_dir, run_dir)
                reasons = stage_reasons(record, request, run_dir)
                cached = not args.force and not reasons
                if not cached:
                    if args.verify:
                        raise RuntimeError(f'{stage}: not reusable -> ' + '; '.join(reasons))
                    directory.mkdir(parents=True, exist_ok=True)
                    manifest['active_stage'] = stage
                    atomic_json(run_dir / 'run.json', manifest)
                    if stage == 'geometry':
                        outputs, geometry_record = produce(pair_dir, run_dir,
                            config['framing'], config['resolution'],
                            config['keypoint_score_threshold'])
                        finish_stage(record, request, outputs,
                            {'seconds': geometry_record['seconds'],
                             'canvas': geometry_record['canvas'],
                             'persons': geometry_record['persons'],
                             'transform': geometry_record['transform']}, run_dir)
                    else:
                        run_worker(config['qwen_python'], STAGE_SCRIPT[stage], request,
                            directory / 'request.json', directory / 'run.log')
                    reasons = stage_reasons(record, request, run_dir)
                    if reasons:
                        raise RuntimeError(f'{stage}: worker finished without valid artifact '
                            f'evidence -> ' + '; '.join(reasons))
                for role, item in read_json(record)['outputs'].items():
                    resolved = str((run_dir / item['file']).resolve())
                    paths[role] = resolved
                    manifest['outputs'][role] = {'path': resolved, 'sha256': item['sha256']}
                if stage == 'prompt':
                    paths.update(select_generation_prompt(run_dir, paths, args.prompt_file,
                        args.verify, config['generator_references']))
                    for role in ('generation_prompt', 'generation_prompt_provenance'):
                        manifest['outputs'][role] = {'path': paths[role],
                            'sha256': sha256_file(paths[role])}
                checked.append((record, request))
                completed.append({'stage': stage, 'cache_hit': cached, 'record': str(record)})
                print(f'{stage.upper()} {"CACHE_HIT" if cached else "COMPLETE"}', flush=True)
                if stage == args.until:
                    break
            manifest['status'] = 'complete' if args.until == 'generate' else 'partial'
            manifest.pop('active_stage', None)
            manifest['finished_at_unix'] = time.time()
            reasons_by_stage = {}
            for record, request in checked:
                reasons = stage_reasons(record, request, run_dir)
                if reasons:
                    reasons_by_stage[record.parent.name] = reasons
            manifest['verification'] = {'status': 'mismatch' if reasons_by_stage else 'validated',
                'reasons': reasons_by_stage}
            print(f'VERIFICATION {manifest["verification"]["status"]}', flush=True)
            if not args.verify:
                atomic_json(run_dir / 'run.json', manifest)
                atomic_json(run_dir / 'outputs.json', {key: item['path']
                    for key, item in manifest['outputs'].items()})
                atomic_json(ROOT / 'latest_run.json', {'run_dir': str(run_dir),
                    'status': manifest['status'], 'manifest': str(run_dir / 'run.json'),
                    'outputs': str(run_dir / 'outputs.json')})
            print(f'{"VERIFIED" if args.verify else "WORKFLOW_FINISHED"} {run_dir}', flush=True)
            if 'final_image' in paths:
                print(f'FINAL_IMAGE {paths["final_image"]}', flush=True)
        except Exception as error:
            if not args.verify:
                manifest.update(status='failed', error=str(error), finished_at_unix=time.time())
                atomic_json(run_dir / 'run.json', manifest)
                atomic_json(ROOT / 'latest_run.json', {'run_dir': str(run_dir),
                    'status': 'failed', 'manifest': str(run_dir / 'run.json'),
                    'outputs': str(run_dir / 'outputs.json')})
            raise


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'WORKFLOW_FAILED: {error}', file=sys.stderr, flush=True)
        sys.exit(1)
