"""Run the authorized 30/30/120/120 pilot through the shared V7 Trainer."""
from pathlib import Path
from dataclasses import replace
from datetime import datetime, timezone
import argparse
import gc
import json
import os
import re
import subprocess
import sys
import traceback

PROJECT = Path('/home/shangguanrz/project/pic-edit')
sys.path.insert(0, str(PROJECT if PROJECT.exists() else Path(__file__).resolve().parents[4]))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def normalized(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def phase_targets(configs):
    expected = [('S0', 30), ('S1', 30), ('S2', 120), ('S3', 120)]
    if [(c['stage']['phase'], c['stage']['budget_updates']) for c in configs] != expected:
        raise ValueError('Pilot phase/budget must be S0=30, S1=30, S2=120, S3=120')
    return (30, 60, 180, 300)


def check_completed_stage(summary, state, config, signature, global_update, parent):
    budget = config['stage']['budget_updates']
    checks = [summary.get('budget_completed') is True,
              summary.get('requested_updates') == summary.get('completed_stage_updates') == budget,
              summary.get('global_updates') == state.get('global_update') == global_update,
              state.get('stage_update') == budget, state.get('config_sha256') == signature,
              normalized(state.get('config')) == normalized(config), state.get('parent') == parent,
              state.get('stage_origin') == config['initialization'],
              state.get('gradient_report') == config['gradient_report']]
    if not all(checks):
        raise ValueError('Checkpoint does not prove this exact completed stage/config/parent/budget')


def new_stage_summary(segments, before, config):
    matches = []
    for folder in set(Path(segments).glob('*')) - before:
        resolved, summary = folder / 'resolved.json', folder / 'summary.json'
        if resolved.is_file() and summary.is_file() and normalized(read(resolved)['config']) == normalized(config):
            matches.append((folder, read(summary)))
    if len(matches) != 1:
        raise ValueError('Expected exactly one new completed segment for the concrete config')
    return matches[0]


def child_command(root, command, *, resource=None, vae_tiling=False, offload_hidden=False):
    policy = ['--vae-tiling'] if vae_tiling else []
    if offload_hidden:
        policy.append('--offload-hidden')
    if resource is not None:
        command = [root / 'scripts/gpu_action.py', *policy, '--resource-report', resource, '--', *command]
    else:
        command = [*command, *policy]
    return [sys.executable, '-u', *map(str, command)]


def generate(argv):
    parser = argparse.ArgumentParser(description='Isolated fixed-seed generation; never runs an optimizer')
    for name in ('config', 'output', 'report'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--vae-tiling', action='store_true')
    parser.add_argument('--offload-hidden', action='store_true')
    args = parser.parse_args(argv)
    import torch
    from src.pose_control.qwen21_full.native_v7.training import TrainingConfigV7, DataPlanV7, NativeBackendV7
    from src.pose_control.qwen21_full.native_v62.training.checkpointing import load_committed
    from src.pose_control.qwen21_full.native_v62.training.observations import generation_probe
    from src.pose_control.qwen21_full.native_v62.training.io import verify, write_json
    if args.output.exists() or args.report.exists():
        raise FileExistsError('Generation output/report must be new')
    free, total = torch.cuda.mem_get_info()
    if total * .5 > free - 2 * 1024 ** 3:
        raise RuntimeError('Generation cannot fit the fixed 50% allocator cap plus 2GiB free margin')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.5)
    cfg = TrainingConfigV7.from_dict(read(args.config))
    saved, update = None, 0
    if args.checkpoint is not None:
        if Path(cfg.initialization['path']).resolve() != args.checkpoint.resolve():
            raise ValueError('Generation checkpoint must match the SHA-bound generation config')
        saved, ref, _ = load_committed(cfg.initialization, schema=cfg.schema)
        update = saved['training_state']['global_update']
    data = DataPlanV7(cfg)
    backend = NativeBackendV7(cfg, data, saved=saved, purpose='grad')
    if args.vae_tiling:
        backend.pipe.vae.enable_tiling()
    result = generation_probe(backend, cfg, data, update, args.output, baseline=saved is None)
    if len(result['records']) != (4 if saved is None else 2):
        raise RuntimeError('Expected train/observation images for exactly two selected rows')
    result.update(memory_fraction=.5, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  checkpoint=cfg.initialization, generation_quality_claimed=False, vae_tiling=args.vae_tiling,
                  offload_hidden_requested=args.offload_hidden, hidden_offload_applied=False)
    write_json(args.report, result)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ['generate']:
        return generate(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--templates', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--existing-s0-gradient', type=Path)
    parser.add_argument('--generate-baseline', action='store_true')
    parser.add_argument('--vae-tiling', action='store_true')
    parser.add_argument('--offload-hidden', action='store_true')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.name):
        parser.error('--name must be one plain run ID')
    root = args.root.resolve()
    if not root.is_dir() or Path('/data/shangguanrz').resolve() not in root.parents:
        parser.error('--root must name an existing experiment beneath /data/shangguanrz')
    configs, reports = root / 'configs' / args.name, root / 'report' / args.name
    log = root / 'logs' / (args.name + '-events.jsonl')
    if any(p.exists() for p in (configs, reports, log, root / 'output' / args.name)):
        raise FileExistsError('Run name already exists; choose a new --name')
    configs.mkdir(parents=True); reports.mkdir(parents=True); log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('x', encoding='utf8'):
        pass
    previous = body_gate = s0_gradient = None
    completed, phase, cfg, before = 0, 'S0', None, set()

    def event(status, path=None, **values):
        row = dict(timestamp=datetime.now(timezone.utc).isoformat(), status=status, stage=phase,
                   global_update=completed, path=None if path is None else str(path), **values)
        with log.open('a', encoding='utf8') as stream:
            stream.write(normalized(row) + '\n'); stream.flush(); os.fsync(stream.fileno())
        print(normalized(row), flush=True)

    def run(command, label, *, gpu=True):
        logfile = root / 'logs' / (args.name + '-' + label + '.log')
        resource = root / 'logs' / (args.name + '-' + label + '-resources.json') if gpu else None
        command = child_command(root, command, resource=resource, vae_tiling=args.vae_tiling,
                                offload_hidden=args.offload_hidden)
        event(label + '_start', logfile, command=command)
        with logfile.open('x', encoding='utf8') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        event(label + '_passed', logfile)

    try:
        from src.pose_control.qwen21_full.native_v7.training import TrainingConfigV7
        from src.pose_control.qwen21_full.native_v62.training.io import verify, write_json, read_json_ref
        from src.pose_control.qwen21_full.native_v62.training.checkpointing import load_committed
        templates = [TrainingConfigV7.from_dict(read(args.templates / (p + '.json'))) for p in ('S0', 'S1', 'S2', 'S3')]
        targets = phase_targets([c.record() for c in templates])
        authorization = templates[0].continuous_stage_authorization
        if not authorization or any(c.continuous_stage_authorization != authorization for c in templates):
            raise ValueError('Every stage must retain the same explicit recorded user authorization')
        if any(c.output_root != str(root / 'output/training') or c.generation_every_updates != 0 for c in templates):
            raise ValueError('Templates must bind this experiment output and isolated generation')
        event('starting', reports, total_updates=300, vae_tiling=args.vae_tiling, offload_hidden=args.offload_hidden)
        segments = root / 'output/training/segments'
        for template, target in zip(templates, targets):
            phase = template.stage.phase
            parent = None if previous is None else previous['checkpoint']
            cfg = replace(template, initialization=template.initialization if parent is None else parent,
                          gradient_report=None, body_baseline_report=body_gate)
            cfg.require_ready()
            concrete = configs / (phase + '-preflight.json')
            write_json(concrete, cfg.record())
            gradient = args.existing_s0_gradient if phase == 'S0' else None
            if gradient is None:
                gradient = reports / (phase + '-gradient.json')
                run(['preflight', '--level', 'grad', '--config', concrete, '--report', gradient], phase + '-preflight')
            report, gradient_ref = read_json_ref(verify({'path': str(gradient.resolve())}))
            expected = dict(schema='native_v7_gradient_preflight_v1', level='grad', passed=True, production_passed=True,
                            config_sha256=cfg.signature(), initialization_sha256=verify(cfg.initialization)['sha256'])
            if any(report.get(k) != v for k, v in expected.items()):
                raise ValueError('Real gradient report failed or does not match the concrete stage')
            cfg = replace(cfg, gradient_report=gradient_ref)
            ready = configs / (phase + '-ready.json')
            write_json(ready, cfg.record())
            event('gradient_accepted', gradient)
            if phase == 'S0':
                s0_gradient = gradient_ref
                if args.generate_baseline:
                    run([str(Path(__file__).resolve()), 'generate', '--config', ready,
                         '--output', root / 'output' / args.name / 'baseline', '--report', reports / 'baseline-generation.json'],
                        'baseline-generation', gpu=False)
            before = set(segments.glob('*'))
            action = ['train'] if parent is None else ['advance-stage', '--checkpoint', parent['path']]
            run([*action, '--config', ready], phase + '-train')
            segment, summary = new_stage_summary(segments, before, cfg.record())
            result = summary['checkpoint']
            saved, committed_ref, _ = load_committed(result['checkpoint'], schema=cfg.schema)
            check_completed_stage(summary, saved['training_state'], cfg.record(), cfg.signature(), target, parent)
            if result['checkpoint'] != committed_ref:
                raise ValueError('Summary checkpoint is not the verified committed reference')
            if verify(result['commit']) != verify({'path': str(Path(committed_ref['path']).with_suffix('.commit.json'))}):
                raise ValueError('Summary commit reference differs from the checkpoint marker')
            previous, completed = result, target
            del saved; gc.collect()
            event('stage_completed', segment, checkpoint=previous)
            if phase == 'S0':
                body_gate = write_json(reports / 'body-technical-gate.json', dict(schema='native_v7_body_pilot_gate_v1',
                    body_accepted=True, manual_acknowledgment=False, acceptance_kind='technical_pilot_gate',
                    continuous_stage_authorization=authorization, s0_gradient_report=s0_gradient,
                    checkpoint_sha256=committed_ref['sha256'], completed_updates=completed,
                    generation_quality_claimed=False, automatic_optimizer_stage_change=False))
        generation_config = configs / 'final-generation.json'
        write_json(generation_config, replace(cfg, initialization=previous['checkpoint']).record())
        run([str(Path(__file__).resolve()), 'generate', '--config', generation_config, '--checkpoint', previous['checkpoint']['path'],
             '--output', root / 'output' / args.name / 'final', '--report', reports / 'final-generation.json'],
            'final-generation', gpu=False)
        write_json(reports / 'completed.json', dict(completed_updates=completed, checkpoint=previous,
                   scope='one_training_pair+one_same_scene_observation', generation_quality_claimed=False))
        event('completed', reports / 'completed.json', checkpoint=previous)
    except BaseException as error:
        failure = dict(error=repr(error), traceback=traceback.format_exc(), last_committed_checkpoint=previous,
                       completed_updates=completed, stage=phase, artifacts_preserved=True)
        # Preserve a partial stage's newest committed save, including optimizer and RNG.
        if cfg is not None:
            for folder in set((root / 'output/training/segments').glob('*')) - before:
                if not (folder / 'resolved.json').is_file() or normalized(read(folder / 'resolved.json')['config']) != normalized(cfg.record()):
                    continue
                failed = folder / 'failure.json'
                if failed.is_file() and (snapshot := read(failed).get('last_committed_snapshot')):
                    try:
                        saved, ref, _ = load_committed(snapshot['checkpoint'], schema=cfg.schema)
                        state = saved['training_state']
                        if normalized(state['config']) != normalized(cfg.record()) or state['config_sha256'] != cfg.signature():
                            raise ValueError('Partial checkpoint config differs')
                        failure.update(partial_stage_checkpoint=snapshot, last_committed_checkpoint=snapshot,
                                       completed_updates=state['global_update'])
                        previous, completed = snapshot, state['global_update']
                        del saved, state; gc.collect()
                    except Exception as inspection:
                        failure['partial_checkpoint_inspection_error'] = repr(inspection)
        from src.pose_control.qwen21_full.native_v62.training.io import write_json
        write_json(reports / 'failure.json', failure)
        event('failed', reports / 'failure.json', error=repr(error), last_committed_checkpoint=previous)
        raise


if __name__ == '__main__':
    main()
