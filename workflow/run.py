#!/usr/bin/env python3
"""One-command single-person pose transfer; all artifacts stay in workflow/runs."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys
import time

from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pose_edit.common import (atomic_json, canvas_size, file_lock, finish_stage,
    letterbox, make_request, model_signature, read_json, sha256_file, stage_outputs,
    stage_reasons, stage_valid)
from pose_transfer.prompts import PE_SAMPLING, validate_generator_prompt

ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
STAGES = ('geometry', 'prompt', 'generate')
STAGE_DIRECTORY = {'geometry': 'geometry', 'prompt': 'prompt', 'generate': 'generation'}
STAGE_SCRIPT = {'geometry': 'smplx_worker.py', 'prompt': 'pe_worker.py',
    'generate': 'generate_worker.py'}
ESTIMATOR_SOURCE_FILES = ['main/inference.py', 'main/base.py', 'models/SMPLest_X.py',
    'human_models/human_models.py', 'scripts/run_smplestx_params_only.py']
INPUTS_CODE = ['src/pose_edit/common.py', 'run.py']
# Provenance files the worker should record but must never fingerprint: they hold absolute
# paths, so hashing them would make a cached run go stale when the checkout simply moves.
EXTRA_RUNTIME_ROLES = ('generation_prompt_provenance',)


def discover_image(role, explicit):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    files = sorted(path for path in (ROOT / 'pictures' / role).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if len(files) != 1:
        raise ValueError(f'pictures/{role} must have exactly one image; found {len(files)}. '
            f'Choose one explicitly with --{role.lower()} PATH or use --pair DIR')
    return files[0].resolve()


def discover_pair(directory):
    """A test pair is a directory holding A_source.png and B_target.png."""
    directory = Path(directory).expanduser().resolve()
    a, b = directory / 'A_source.png', directory / 'B_target.png'
    missing = [str(path) for path in (a, b) if not path.is_file()]
    if missing:
        found = sorted(item.name for item in directory.iterdir()) if directory.is_dir() else []
        raise FileNotFoundError(f'--pair needs A_source.png and B_target.png in {directory}; '
            f'missing {missing}; present {found}')
    return a, b


def code_identity(names):
    return {name: sha256_file((REPO_ROOT if name.startswith('src/') else ROOT) / name) for name in names}


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
        str(Path(config['smplest_root']) / 'pretrained_models' / config['smplest_checkpoint'] /
            f'{config["smplest_checkpoint"]}.pth.tar'),
        str(Path(config['smplest_root']) / 'human_models/human_model_files/smplx/SMPLX_NEUTRAL.npz'),
        str(Path(config['pe_model']) / 'system_prompt.txt'), config['qwen_model']]
    for name in required:
        if not Path(name).exists():
            raise FileNotFoundError(f'Required asset not found: {name}')
    print('ASSET_PATHS_OK', flush=True)
    # person_detection_confidence only reaches the estimator through this local hook, which sits
    # in a vendored and git-ignored checkout; without it the estimator silently keeps its own
    # default and the workflow's setting becomes decoration.
    inference = Path(config['smplest_root']) / 'main/inference.py'
    if 'SMPLESTX_DET_CONF' not in inference.read_text(encoding='utf-8', errors='replace'):
        raise RuntimeError(f'{inference} no longer reads SMPLESTX_DET_CONF, so the workflow '
            'setting person_detection_confidence would be ignored by the estimator. Re-apply the '
            'local override or drop the setting from workflow/config.json.')
    print('ESTIMATOR_CONF_HOOK_OK', flush=True)
    subprocess.run([config['geometry_python'], '-c',
        'import torch,smplx,ultralytics,numpy,PIL; print("GEOMETRY_IMPORTS_OK")'], check=True)
    subprocess.run([config['qwen_python'], '-c',
        'import torch,transformers; from diffusers import QwenImage21Pipeline; '
        'torch.cuda.init(); print("QWEN_CUDA_OK",torch.cuda.get_device_name(0),'
        '"free/total GiB",*[round(x/1024**3,2) for x in torch.cuda.mem_get_info()])'], check=True)


def stage_identity(stage, config, paths):
    """Everything a cached stage must still match; deliberately never an absolute path."""
    common_code = ['src/pose_edit/common.py']
    if stage == 'geometry':
        fields = ['smplest_root', 'smplest_checkpoint', 'person_detection_confidence',
            'pose_framing', 'subject']
        input_names = ['A_rgb', 'B_rgb', 'A_reference', 'source_preprocess']
        model = Path(config['smplest_root'])
        checkpoint = model / 'pretrained_models' / config['smplest_checkpoint']
        identities = {'checkpoint': model_signature(checkpoint),
            'detector': model_signature(model / 'pretrained_models/yolov8x.pt'),
            'human_models': model_signature(model / 'human_models/human_model_files/smplx'),
            'estimator_source': {name: sha256_file(model / name) for name in ESTIMATOR_SOURCE_FILES}}
        code = common_code + ['smplx_worker.py', 'src/pose_edit/estimator.py',
            'src/pose_edit/geometry.py']
    elif stage == 'prompt':
        fields = ['pe_model', 'pe_seed', 'prop_policy']
        input_names = ['A_reference', 'B_rgb', 'B_3d_dwpose']
        identities = model_signature(config['pe_model'])
        code = common_code + ['pe_worker.py', 'pose_transfer/prompts.py']
    else:
        fields = ['qwen_model', 'resolution', 'steps', 'seed', 'true_cfg_scale',
            'use_kv_cache', 'adapter', 'lora']
        input_names = ['A_reference', 'B_3d_dwpose', 'generation_prompt', 'source_preprocess']
        identities = model_signature(config['qwen_model'])
        code = common_code + ['generate_worker.py']
    identity = {'settings': {key: config[key] for key in fields},
        'inputs_sha256': {key: sha256_file(paths[key]) for key in input_names},
        'model_identity': identities, 'code': code_identity(code)}
    if stage == 'prompt':
        identity['sampling'], identity['enable_thinking'] = PE_SAMPLING, True
    return identity, input_names


def stage_request(stage, config, paths, run_dir):
    identity, input_names = stage_identity(stage, config, paths)
    runtime = {key: paths[key] for key in input_names}
    runtime.update({key: paths[key] for key in EXTRA_RUNTIME_ROLES
        if key in paths and key not in input_names})
    return make_request(stage, identity, run_dir, {'run_dir': str(run_dir), **runtime})


def prepare(run_dir, a, b, resolution, verify):
    """Normalise both photographs onto the generation canvas; reuse only on identical content."""
    directory = run_dir / 'inputs'
    record_path = directory / 'stage.json'
    request = make_request('inputs',
        {'A_sha256': sha256_file(a), 'B_sha256': sha256_file(b), 'resolution': resolution,
         'code': code_identity(INPUTS_CODE)},
        run_dir, {'A': a, 'B': b})
    if not stage_valid(record_path, request, run_dir):
        if verify:
            raise RuntimeError('Prepared inputs are not reusable:\n  '
                + '\n  '.join(stage_reasons(record_path, request, run_dir)))
        directory.mkdir(parents=True, exist_ok=True)
        with Image.open(a) as image:
            source = ImageOps.exif_transpose(image).convert('RGB')
        with Image.open(b) as image:
            target = ImageOps.exif_transpose(image).convert('RGB')
        source.save(directory / 'A_rgb.png')
        target.save(directory / 'B_rgb.png')
        reference, meta = letterbox(source, canvas_size(source.size, resolution))
        reference.save(directory / 'A_reference.png')
        atomic_json(directory / 'source_preprocess.json', meta)
        finish_stage(record_path, request,
            {'A_rgb': directory / 'A_rgb.png', 'B_rgb': directory / 'B_rgb.png',
             'A_reference': directory / 'A_reference.png',
             'source_preprocess': directory / 'source_preprocess.json'},
            {'source_original': str(a), 'target_original': str(b), 'preprocess': meta}, run_dir)
    return stage_outputs(record_path, run_dir), request


def select_generation_prompt(run_dir, paths, manual_prompt, verify=False):
    """Freeze the text the generator reads, without ever touching PE's own output file."""
    compiled = Path(paths['pe_compiled_prompt'])
    source = Path(manual_prompt).expanduser().resolve() if manual_prompt else compiled
    text = source.read_text(encoding='utf-8').strip()
    if not text:
        raise ValueError(f'Prompt file is empty: {source}')
    validate_generator_prompt(text)
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
    atomic_json(provenance, {'source': 'manual' if manual_prompt else 'pe_compiled',
        'source_path': str(source), 'source_sha256': sha256_file(source),
        'generation_prompt': str(target), 'sha256': sha256_file(target)})
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--pair', type=Path, help='Directory holding A_source.png and B_target.png')
    parser.add_argument('--a', help='Source person image; defaults to the sole file in pictures/A')
    parser.add_argument('--b', help='Target pose image; defaults to the sole file in pictures/B')
    parser.add_argument('--name', help='Run directory name under workflow/runs, default pair name or demo')
    parser.add_argument('--until', choices=STAGES, default='generate', help='Stop after this stage')
    parser.add_argument('--prompt-file', type=Path, help='Generate with this prompt text instead '
        'of the prompt-engineering output; PE stays cached and is never overwritten')
    parser.add_argument('--resolution', type=int, help='Qwen side-length pixel budget, default1024')
    parser.add_argument('--subject', default='auto', help='Which detected person to transfer: '
        'auto requires exactly one, largest takes the biggest box, or a 0-based index over the '
        'largest-first ordering')
    parser.add_argument('--framing', choices=('source_person', 'source_torso'),
        help='How the transferred skeleton is scaled onto the source: source_person fits B\'s '
        'detection box inside A\'s (default), source_torso matches B\'s torso to the source '
        'body\'s own scale')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--check', action='store_true', help='Check paths, imports and CUDA, without generating')
    parser.add_argument('--verify', action='store_true', help='Verify cached requests and all artifact hashes; never infer')
    parser.add_argument('--force', action='store_true', help='Recompute stages even if their cache is valid')
    args = parser.parse_args()
    if args.force and args.verify:
        parser.error('--force and --verify are mutually exclusive')
    pair_images = discover_pair(args.pair) if args.pair else (None, None)
    name = args.name or (args.pair.name if args.pair else 'demo')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', name):
        parser.error('--name must be a simple directory name')
    config = read_json(args.config)
    for key in ('resolution', 'steps', 'seed'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if not re.fullmatch(r'auto|largest|\d{1,3}', args.subject):
        parser.error("--subject must be auto, largest, or a 0-based index")
    config['subject'] = args.subject
    if args.framing:
        config['pose_framing'] = args.framing
    config.setdefault('pose_framing', 'source_person')
    if (config.get('adapter') is not False or config.get('lora') is not False
            or config.get('pose_framing') not in ('source_person', 'source_torso')
            or config.get('prop_policy') != 'source_only'):
        parser.error('This workflow is fixed to source-only objects, no adapter/LoRA, and one of '
            'the two supported pose framings (source_person, source_torso)')
    if config['resolution'] < 256 or config['steps'] < 1:
        parser.error('resolution must be >=256 and steps must be positive')
    if args.check:
        dependency_check(config)
        return
    a = discover_image('A', args.a or pair_images[0])
    b = discover_image('B', args.b or pair_images[1])
    run_dir = ROOT / 'runs' / name
    if args.verify and not run_dir.is_dir():
        raise RuntimeError(f'No existing run to verify: {run_dir}')
    with file_lock(ROOT / '.workflow.lock'):
        completed = []
        checked = []
        started = time.time()
        manifest = {'schema': 'single_person_pose_workflow_v2', 'status': 'running',
            'A_original': str(a), 'B_original': str(b), 'settings': config,
            'requested_until': args.until, 'prompt_source': str(args.prompt_file) if args.prompt_file else None,
            'stages': completed, 'outputs': {}, 'started_at_unix': started}
        try:
            manifest['active_stage'] = 'inputs'
            if not args.verify:
                atomic_json(run_dir / 'run.json', manifest)
            paths, inputs_request = prepare(run_dir, a, b, config['resolution'], args.verify)
            checked.append((run_dir / 'inputs' / 'stage.json', inputs_request))
            for stage in STAGES:
                directory = run_dir / STAGE_DIRECTORY[stage]
                record = directory / 'stage.json'
                request = stage_request(stage, config, paths, run_dir)
                reasons = stage_reasons(record, request, run_dir)
                cached = not args.force and not reasons
                if not cached:
                    if args.verify:
                        raise RuntimeError(f'{stage}: not reusable -> ' + '; '.join(reasons))
                    directory.mkdir(parents=True, exist_ok=True)
                    manifest['active_stage'] = stage
                    atomic_json(run_dir / 'run.json', manifest)
                    python = config['geometry_python'] if stage == 'geometry' else config['qwen_python']
                    run_worker(python, STAGE_SCRIPT[stage], request, directory / 'request.json',
                        directory / 'run.log')
                    reasons = stage_reasons(record, request, run_dir)
                    if reasons:
                        raise RuntimeError(f'{stage}: worker finished without valid artifact '
                            'evidence -> ' + '; '.join(reasons))
                for role, item in read_json(record)['outputs'].items():
                    resolved = str((run_dir / item['file']).resolve())
                    paths[role] = resolved
                    manifest['outputs'][role] = {'path': resolved, 'sha256': item['sha256']}
                if stage == 'prompt':
                    paths.update(select_generation_prompt(run_dir, paths, args.prompt_file,
                        args.verify))
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
            # Re-check every stage this run touched, so the recorded verdict is measured now
            # rather than asserted because the loop reached the end.
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
                atomic_json(run_dir / 'outputs.json', {key: item['path'] for key, item in manifest['outputs'].items()})
                atomic_json(ROOT / 'latest_run.json', {'run_dir': str(run_dir), 'status': manifest['status'],
                    'manifest': str(run_dir / 'run.json'), 'outputs': str(run_dir / 'outputs.json')})
            print(f'{"VERIFIED" if args.verify else "WORKFLOW_FINISHED"} {run_dir}', flush=True)
            if 'final_image' in paths:
                print(f'FINAL_IMAGE {paths["final_image"]}', flush=True)
        except Exception as error:
            if not args.verify:
                manifest.update(status='failed', error=str(error), finished_at_unix=time.time())
                atomic_json(run_dir / 'run.json', manifest)
                atomic_json(run_dir / 'outputs.json', {key: item['path'] for key, item in manifest['outputs'].items()})
                atomic_json(ROOT / 'latest_run.json', {'run_dir': str(run_dir), 'status': 'failed',
                    'manifest': str(run_dir / 'run.json'), 'outputs': str(run_dir / 'outputs.json')})
            raise


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'WORKFLOW_FAILED: {error}', file=sys.stderr, flush=True)
        sys.exit(1)
