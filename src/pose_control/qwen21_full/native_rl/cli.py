"""Native V6.2/V7 online RL with fixed held-out ODE evaluation."""

import argparse
from contextlib import ExitStack
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import sys
import time


def _arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check-data', 'preflight', 'train', 'resume'))
    parser.add_argument('--config', type=Path, required=True, help='Native V6.2 or V7 SFT configuration with bound data/cache/initialization')
    parser.add_argument('--output', type=Path, help='New run directory under /data/shangguanrz/')
    parser.add_argument('--checkpoint', type=Path, help='Committed RL checkpoint for resume')
    parser.add_argument('--models', type=Path, default=Path('/home/shangguanrz/project/pic-edit/models'))
    parser.add_argument('--route', help='Route name from the SFT configuration')
    parser.add_argument('--updates', '--optimizer-updates', type=int, default=1, help='Number of successful optimizer.step calls')
    parser.add_argument('--max-rollout-batches', type=int, help='Rollout cap including groups without an effective advantage (default: 5 × updates)')
    parser.add_argument('--pose-device', default='cpu', help='SDPose detector/inference device, e.g. cpu or cuda:0')
    parser.add_argument('--pose-python', type=Path, help='SDPose interpreter (default: the project SDPose environment)')
    parser.add_argument('--background-index', type=Path, help='SHA-bound Source alignment records indexed by source and pair')
    parser.add_argument('--background-weight', type=float, default=.05)
    parser.add_argument('--group-size', type=int, default=8)
    parser.add_argument('--groups-per-update', type=int, default=4)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--sde-step', type=int, default=3, help='Zero-based index of the single stochastic step')
    parser.add_argument('--noise-level', type=float, default=.7)
    parser.add_argument('--aggregation', choices=('gdpo', 'gdpo-guarded', 'sum'), default='gdpo-guarded')
    parser.add_argument('--pose-only', action='store_true')
    parser.add_argument('--std-floor', type=float, default=.001)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--clip', type=float, default=.001)
    parser.add_argument('--beta', type=float, default=.001)
    parser.add_argument('--replay-epochs', type=int, default=2)
    parser.add_argument('--parity-tolerance', type=float, default=1e-4)
    parser.add_argument('--validate-every', type=int, default=25)
    parser.add_argument('--save-every', type=int, default=5, help='Checkpoint cadence in optimizer updates')
    parser.add_argument('--stop-after-updates', type=int, help='Pause after this absolute update; keep the full --updates budget')
    parser.add_argument('--val-limit', type=int, help='Fixed sample cap for a bounded pilot')
    parser.add_argument('--pose-drop', type=float, default=.02)
    parser.add_argument('--id-drop', type=float, default=.02)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--keep-replay', action='store_true')
    args = parser.parse_args(argv)
    if (not 0 < args.sde_step < args.steps - 1 or args.group_size < 2 or
            min(args.updates, args.groups_per_update, args.replay_epochs, args.validate_every, args.save_every, args.patience) < 1 or
            (args.val_limit is not None and args.val_limit < 1) or
            (args.max_rollout_batches is not None and args.max_rollout_batches < 1)):
        parser.error('Positive budgets and one internal SDE step required')
    if (not all(math.isfinite(x) for x in (args.lr, args.clip, args.beta, args.noise_level, args.parity_tolerance,
                                          args.std_floor, args.pose_drop, args.id_drop, args.background_weight)) or
            min(args.lr, args.noise_level, args.parity_tolerance) <= 0 or not 0 < args.clip < 1 or
            min(args.beta, args.std_floor, args.pose_drop, args.id_drop, args.background_weight) < 0):
        parser.error('Invalid optimizer, SDE, variance or regression settings')
    if (args.action == 'resume') != (args.checkpoint is not None):
        parser.error('Supply --checkpoint with resume')
    if args.stop_after_updates is not None and (args.action not in ('train', 'resume') or
                                                not 0 < args.stop_after_updates < args.updates):
        parser.error('--stop-after-updates must precede the full optimizer budget')
    if args.action != 'check-data':
        if args.output is None or not args.output.resolve().is_relative_to('/data/shangguanrz'):
            parser.error('--output must be a new directory under /data/shangguanrz/')
        if args.output.exists():
            parser.error('--output already exists')
    return args


def _settings(args):
    from ..native_v62.training.config import LossSpec, TrainingConfig
    from ..native_v7.training import TrainingConfigV7
    from ..native_v7.losses import PaperLossConfig

    raw = json.loads(args.config.read_text(encoding='utf8'))
    configs = {kind.expected_schema: kind for kind in (TrainingConfig, TrainingConfigV7)}
    schema = raw.get('schema', TrainingConfig.expected_schema)
    if schema not in configs:
        raise ValueError('RL requires a Native V6.2 or V7 training configuration')
    config = configs[schema].from_dict(raw)
    routes = [r for r in config.stage.routes if args.route is None or r.name == args.route]
    if len(routes) != 1:
        raise ValueError('Select one Native route with --route')
    route = replace(routes[0], loss=LossSpec())
    rl_overrides = dict(paper_loss=PaperLossConfig(person_fm=0., contact_fm=0., identity_contrastive=0.),
                        preservation=None, preservation_index=None) if isinstance(config, TrainingConfigV7) else {}
    config = replace(config, stage=replace(config.stage, routes=(route,)), evaluators=None, **rl_overrides)
    if args.action in ('train', 'resume') and config.experiment_mode != 'standard':
        raise ValueError('Online updates require held-out validation from disjoint videos/sequences')
    if config.initialization is None or not config.initialization.get('sha256'):
        raise ValueError('Pin the complete Native SFT initialization SHA256 in --config')
    if args.action != 'check-data' and any(not getattr(config, x) for x in ('cache_root', 'cache_identity', 'model_manifest')):
        raise ValueError('Rollout requires a prepared frozen Native condition cache and model manifest')
    return config, route


def _backend_for(config):
    from ..native_v62.training.backend import NativeBackend
    from ..native_v62.training.data import DataPlan
    from ..native_v7.training import DataPlanV7, NativeBackendV7, TrainingConfigV7

    if isinstance(config, TrainingConfigV7):
        return DataPlanV7, NativeBackendV7, 'native_full_v7'
    return DataPlan, NativeBackend, 'native_full_v62'


def _background_records(path, records):
    from ..native_v62.training.io import read_json_ref

    if path is None:
        return None, {}
    index, ref = read_json_ref({'path': str(path.resolve())})
    entries = index.get('entries')
    if index.get('schema') != 'native_rl_background_index_v1' or not isinstance(entries, dict):
        raise ValueError('Invalid Native RL background index')
    keys = {(row['source_id'], row['pair_id']) for row in records}
    for source, pairs in entries.items():
        if not isinstance(pairs, dict) or any((source, pair) not in keys for pair in pairs):
            raise ValueError('Background index contains a pair outside the selected Native data')
    return ref, entries


def _targets(bank, data, records, background_entries):
    from .rewards import UnscorableTarget

    targets, skipped, provenance = {}, [], {}
    for info in records:
        key = (info['source_id'], info['index'])
        condition = data.conditions[key][1]
        sample = data.sample(*key)
        meta = sample['local_condition_v4'].metadata[0]
        provenance[key] = {name: meta[name] for name in ('driver_rgb_sha256', 'driver_record_sha256',
                                                        'target_canvas_id', 'target_person_ids', 'source_person_ids')}
        try:
            entry = background_entries.get(info['source_id'], {}).get(info['pair_id'])
            targets[key] = bank.target(sample, condition, entry)
        except UnscorableTarget as exc:
            skipped.append({'source': info['source_id'], 'pair_id': info['pair_id'], 'reason': str(exc)})
    return targets, skipped, provenance


def _generator(backend, route, key, args):
    from ..native_v62.training.data import route_options
    from .runner import NativeRollout

    sample, batch, pair = backend.pair(*key)
    options, base = route_options(route, sample['native_scene_kind'])
    rollout = NativeRollout(backend, batch, pair, route, options, base,
                            steps=args.steps, sde_step=args.sde_step, noise_level=args.noise_level)
    return rollout


def _summarize(rows):
    rewards = [row['reward'] for row in rows]
    metrics = {name: sum(item[name] for item in rewards if name in item) / sum(name in item for item in rewards)
               for name in sorted(set().union(*rewards))}
    metrics['pck_002'] = sum(r['pck_002'] for r in rows) / len(rows)
    metrics['contact_applicable_rate'] = sum('contact' in r for r in rewards) / len(rows)
    eligible_people = sum(r['id_eligible'] for r in rows)
    if eligible_people:
        metrics['id_missing_rate'] = sum(r['id_missing'] for r in rows) / eligible_people
    swaps = [r['id_swap'] for r in rows if r.get('id_swap') is not None]
    if swaps:
        metrics['id_swap_rate'] = sum(swaps) / len(swaps)
    for key in ('two_person_detected', 'person0_hand_pck_002', 'person1_hand_pck_002', 'source_background_mse'):
        values = [row[key] for row in rows if row.get(key) is not None]
        if values:
            metrics['two_person_detection_rate' if key == 'two_person_detected' else key] = sum(values) / len(values)
    metrics['background_applicable_rate'] = sum('background' in r for r in rewards) / len(rows)
    if any('occlusion_consistency' in row for row in rows):
        metrics['occlusion_consistency'] = None
    return metrics


def _evaluate(backend, data, route, bank, targets, keys, args, directory):
    directory.mkdir()
    rows = []
    for i, key in enumerate(keys):
        rollout = _generator(backend, route, key, args)
        image, _ = rollout.generate(backend.config.generation_seed + i)
        reward, detail = bank.score(*targets[key], image)
        image.save(directory / f'{i:04d}.png')
        rows.append({'source': key[0], 'pair_id': data.datasets[key[0]].rows[key[1]]['pair_id'],
                     'reward': reward, **detail})
    return _summarize(rows), rows


def _score(metrics, weights):
    return sum(w * metrics[k] for k, w in weights.items() if k in metrics) / sum(w for k, w in weights.items() if k in metrics)


def _collect_batch(backend, data, route, bank, targets, keys, args, selector, weights, directory):
    import torch
    from ..native_v62.observation import DiagnosticsV62
    from ..native_v62.training.io import write_json
    from .objective import advantages

    started = time.monotonic()
    reward_seconds = 0.
    replay_groups, reward_groups, details, native_stats = [], [], [], []
    for i, key in enumerate(keys):
        rollout = _generator(backend, route, key, args)
        replay, scores, candidates = [], [], []
        for j in range(args.group_size):
            seed = selector.randrange(2**63)
            observer = DiagnosticsV62() if backend.config.diagnostics and j == 0 else None
            report = {} if observer is not None else None
            image, row = rollout.generate(seed, stochastic=True, observer=observer, report=report)
            if observer is not None:
                native_stats.append({'group': i, 'native': observer.report(), 'sampling': report})
            image.save(directory / f'{i:02d}-{j:02d}.png')
            reward_started = time.monotonic()
            try:
                reward, detail = bank.score(*targets[key], image)
            except Exception as exc:
                write_json(directory / 'evaluation-failed.json', {'status': 'evaluator_error',
                           'source': key[0], 'index': key[1], 'seed': seed, 'message': str(exc)})
                raise
            reward_seconds += time.monotonic() - reward_started
            replay.append(row)
            scores.append(reward)
            candidates.append({'seed': seed, 'old_log_prob': float(row.old_log_prob), 'sigma': row.sigma,
                               'next_sigma': row.next_sigma, 'reward': reward, **detail})
        if any(score.keys() != scores[0].keys() for score in scores):
            raise ValueError('Reward eligibility changed within a Native Rollout group')
        replay_groups.append((replay, rollout.predict))
        reward_groups.append({name: [score[name] for score in scores] for name in scores[0]})
        details.append({'source': key[0], 'pair_id': data.datasets[key[0]].rows[key[1]]['pair_id'],
                        'condition_sha256': rollout.pair['condition_sha256'], 'candidates': candidates})
    adv, stats = advantages(reward_groups, weights, mode=args.aggregation, std_floor=args.std_floor)
    groups = [(rows, a, predict) for (rows, predict), a in zip(replay_groups, adv)]
    if args.keep_replay:
        torch.save([[{k: v.cpu() if torch.is_tensor(v) else v for k, v in vars(row).items()}
                     for row in rows] for rows, _, _ in groups], directory / 'replay.pt')
    event = {'groups': details, 'advantage_stats': stats, 'advantages': adv.tolist(),
             'reward_summary': _summarize([c for g in details for c in g['candidates']]),
             'rollout_seconds': time.monotonic() - started - reward_seconds, 'reward_seconds': reward_seconds,
             'native_diagnostics': native_stats}
    return groups, adv, event


def _pending_batch(keys, conditions, groups, means, cursor, epochs):
    return {'cursor': cursor, 'epochs': epochs, 'groups': [
        {'key': key, 'condition': condition, 'rows': [vars(row) for row in rows],
         'advantage': advantage, 'means': [means[id(row)] for row in rows]}
        for key, condition, (rows, advantage, _) in zip(keys, conditions, groups)]}


def _rebind_pending(pending, backend, route, args):
    import torch
    from .runner import Replay

    keys, conditions, groups, means = [], [], [], {}
    for saved in pending['groups']:
        key = tuple(saved['key'])
        rollout = _generator(backend, route, key, args)
        if rollout.pair['condition_sha256'] != saved['condition']:
            raise ValueError('Resumed Replay condition differs')
        rows = [Replay(**{k: v.to(backend.device) if torch.is_tensor(v) else v for k, v in row.items()})
                for row in saved['rows']]
        if len(rows) != len(saved['means']) or len(rows) != len(saved['advantage']):
            raise ValueError('Incomplete pending Native Replay')
        means.update({id(row): mean.to(backend.device) for row, mean in zip(rows, saved['means'])})
        groups.append((rows, saved['advantage'], rollout.predict))
        keys.append(key)
        conditions.append(saved['condition'])
    return keys, conditions, groups, means


def main(argv=None):
    args = _arguments(argv)
    with ExitStack() as resources:
        _run(args, resources)


def _run(args, resources):
    import numpy as np
    import torch
    from src.evaluation.sdpose import SDPOSE_PYTHON
    from ..native_v62.training.checkpointing import environment, load_committed
    from ..native_v62.training.data import route_options
    from ..native_v62.training.io import file_sha, verify, write_json
    from .checkpointing import load_checkpoint, restore_checkpoint, save_checkpoint
    from .pose_worker import SDPoseWorker
    from .rewards import DUAL_WEIGHTS, SINGLE_WEIGHTS, ImageRewardBank, evaluator_assets
    from .runner import FrozenReference, RunCounters, audit_replay, fixed_reference_means, update_policy

    config, route = _settings(args)
    training = args.action in ('train', 'resume')
    args.pose_python = (args.pose_python or Path(SDPOSE_PYTHON)).absolute()
    resumed, resume_ref = (load_checkpoint({'path': str(args.checkpoint.resolve()), 'sha256': None})
                           if args.action == 'resume' else (None, None))
    data_type, backend_type, architecture = _backend_for(config)
    data = data_type(config)
    train_rows = [r for r in data.records if r['eligible'] and r['role'] == 'train']
    val_rows = [r for r in data.records if r['eligible'] and r['role'] == 'validation']
    if not training and config.experiment_mode == 'fixed_pair_overfit':
        val_rows = [r for r in data.records if r['eligible'] and r['role'] == 'observation']
    if not train_rows or not val_rows:
        raise ValueError('Native RL requires eligible training and evaluation conditions')
    if training:
        saved, initialization, _ = load_committed(config.initialization, schema=config.schema)
    else:
        initialization = verify(config.initialization)
        saved = torch.load(initialization['path'], map_location='cpu', weights_only=True, mmap=True)
    if saved.get('architecture') != architecture or 'native_rl_state' in saved:
        raise ValueError(f'The KL reference must be an original {architecture} SFT checkpoint')
    state = saved.get('training_state', {})
    stage = state.get('config', {}).get('stage', {})
    lineage = {'stage': stage.get('stage_id'), 'stage_updates': state.get('stage_update'),
               'global_updates': state.get('global_update'), 'parent': state.get('parent'), 'stage_origin': state.get('stage_origin')}
    if training and not state.get('stage_update', 0):
        raise ValueError('The committed SFT stage must have successful updates')
    print(json.dumps({'architecture': architecture, 'initialization': initialization, 'lineage': lineage}), flush=True, file=sys.stderr)
    evaluator_hashes = {str(path): file_sha(path) for path in evaluator_assets(args.models)}
    worker = resources.enter_context(SDPoseWorker(args.pose_python, device=args.pose_device))
    bank = ImageRewardBank(args.models, pose=worker)
    background_ref, background_entries = _background_records(args.background_index, train_rows + val_rows)
    targets, skipped, provenance = _targets(bank, data, train_rows + val_rows, background_entries)
    if config.experiment_mode == 'standard':
        train_rgb, val_rgb = ({provenance[(r['source_id'], r['index'])]['driver_rgb_sha256'] for r in rows}
                              for rows in (train_rows, val_rows))
        if train_rgb & val_rgb:
            raise ValueError('Target RGB leakage between rollout and evaluation')
    train_keys = [(r['source_id'], r['index']) for r in train_rows if (r['source_id'], r['index']) in targets]
    val_keys = [(r['source_id'], r['index']) for r in val_rows if (r['source_id'], r['index']) in targets][:args.val_limit]
    if len(train_keys) < args.groups_per_update or not val_keys:
        raise ValueError(f'Insufficient scorable Native tasks ({len(train_keys)} train, {len(val_keys)} eval): {skipped[:6]}')
    scene = 'single' if config.qwen_condition_profile == 'single_2ref' else 'dual'
    weights = {'pose': 1.} if args.pose_only else dict(SINGLE_WEIGHTS if scene == 'single' else DUAL_WEIGHTS)
    if not args.pose_only and background_ref and args.background_weight:
        weights['background'] = args.background_weight
    metadata = {'architecture': architecture, 'initialization': initialization, 'sft_lineage': lineage,
                'pose_backend': 'sdpose', 'pose_worker': worker.identity,
                'data_signature': data.signature, 'evaluator_sha256': evaluator_hashes, 'background_index': background_ref,
                'scene': scene, 'weights': weights, 'train_tasks': len(train_keys), 'eval_tasks': len(val_keys),
                'evaluation_kind': config.experiment_mode, 'skipped_targets': skipped, 'route': route.name,
                'eligible': [{'source': k[0], 'index': k[1], 'id_people': sum(x is not None for x in targets[k][0].identities),
                              'contacts': len(targets[k][0].contacts), 'background': targets[k][0].background_status,
                              **provenance[k]} for k in train_keys + val_keys]}
    if args.action == 'check-data':
        print(json.dumps(metadata, ensure_ascii=False), flush=True)
        return

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    print(json.dumps({'phase': 'load_policy', 'output': str(output), 'train': len(train_keys), 'eval': len(val_keys)}), flush=True)
    backend = backend_type(config, data, saved=saved, purpose='grad' if args.action == 'preflight' else 'train')
    del saved
    backend.runtime.eval()
    backend.pipe.transformer.eval()
    options, base = route_options(route, scene)
    groups = set()
    if options.body_strength > 0:
        groups.add('native_body')
    if route.poselink_strength > 0:
        groups.add('poselink_heads')
    if scene == 'dual' and options.interaction and options.interaction_strength > 0 and base.interaction:
        groups.add('interaction_heads')
    active_names = {row.name for row in backend.catalog.rows if row.role == 'head' and row.runtime_group in groups}
    if not active_names:
        raise ValueError('Native parameter catalog has no eligible heads')
    active = []
    for name, p in backend.catalog.parameters.items():
        selected = training and name in active_names
        p.requires_grad_(selected)
        if selected:
            active.append(p)
    active_ids = {id(p) for p in active}
    parameters = dict.fromkeys([*backend.catalog.parameters.values(), *backend.pipe.transformer.parameters()])
    frozen = [p for p in parameters if id(p) not in active_ids]
    backend.catalog.assert_flags(active_names if active else set())
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    identity = {**metadata, 'config_sha256': config.signature(), 'backend': backend.identity,
                'active_names': [name for name in backend.catalog.parameters if name in active_names],
                'training_environment': environment(),
                'algorithm': {k: v for k, v in settings.items() if k not in
                              ('action', 'config', 'output', 'checkpoint', 'stop_after_updates', 'keep_replay')}}
    metadata.update({'schema': 'native_rl_run_v3', 'identity': backend.identity,
                     'active_names': sorted(active_names) if active else [], 'resume_identity': identity,
                     'resume_from': resume_ref, 'settings': settings})
    write_json(output / 'run.json', metadata)
    selector = random.Random(config.seed)
    optimizer = torch.optim.AdamW(active, lr=args.lr, weight_decay=0.) if active else None
    reference = FrozenReference(active)
    counters = RunCounters()
    pending = None
    last_checkpoint = resume_ref
    last_saved = None

    if resumed is not None:
        if resumed['architecture'] != architecture:
            raise ValueError('Native RL resume architecture differs')
        counters, pending = restore_checkpoint(resumed, backend.runtime, optimizer, selector, identity)
        selection = resumed['native_rl_state']['selection']
        baseline = selection['baseline']
        best = resume_ref if selection['best_is_current'] else selection['best']
        if best is not None:
            verify(best)
        best_score = selection['best_score']
        regressions = selection['regressions']
        last_validated = selection['last_validated']
        if counters.optimizer_updates >= (args.stop_after_updates or args.updates):
            raise ValueError('Resume must target a later optimizer update')
        del resumed
    else:
        random.seed(config.seed)
        np.random.seed(config.seed % (2**32))
        torch.manual_seed(config.seed)
        baseline, baseline_rows = _evaluate(backend, data, route, bank, targets, val_keys, args, output / 'ode-sft')
        write_json(output / 'ode-sft.json', {'metrics': baseline, 'rows': baseline_rows})
        best, best_score, regressions, last_validated = None, _score(baseline, weights), 0, 0

    if training and not args.pose_only and (
            'identity' not in baseline or not any(any(x is not None for x in targets[k][0].identities) for k in train_keys)):
        raise ValueError('Pose/ID training requires eligible faces in both training and held-out evaluation')

    def validate():
        nonlocal best_score, regressions, last_validated
        update = counters.optimizer_updates
        metrics, rows = _evaluate(backend, data, route, bank, targets, val_keys, args, output / f'ode-{update:06d}')
        rejected = any(metrics[k] < baseline[k] - tol for k, tol in
                       (('pose', args.pose_drop), ('identity', args.id_drop)) if k in baseline)
        report = {**counters.record(), 'metrics': metrics, 'regressed': rejected, 'sft_reference': initialization}
        write_json(output / f'ode-{update:06d}.json', {**report, 'rows': rows})
        score = _score(metrics, weights)
        improved = not rejected and score > best_score
        if improved:
            best_score = score
        regressions = regressions + 1 if rejected else 0
        last_validated = update
        return improved, regressions >= args.patience

    def save(pending_replay, *, improved=False, resumable=True):
        nonlocal best, last_checkpoint, last_saved
        selection = {'baseline': baseline, 'best': best, 'best_is_current': improved,
                     'best_score': best_score, 'regressions': regressions, 'last_validated': last_validated}
        payload = backend.runtime.checkpoint_payload(metadata={'native_rl': counters.record()})
        last_checkpoint = save_checkpoint(output, payload, optimizer, selector, counters, identity=identity,
                                          selection=selection, pending=pending_replay, resumable=resumable)
        last_saved = counters.record()
        if improved:
            best = last_checkpoint

    attempt_limit = (args.max_rollout_batches or args.updates * 5) if training else 1
    stop = 'rollout_budget' if training else 'preflight'
    while pending is not None or counters.rollout_batches < attempt_limit:
        continuing = pending is not None
        event = {'batch': counters.rollout_batches + (not continuing), 'resumed_replay': continuing}
        suffix = '-continued' if continuing else ''
        directory = output / f"rollout-{event['batch']:06d}{suffix}"
        directory.mkdir()
        if continuing:
            keys, conditions, groups, means = _rebind_pending(pending, backend, route, args)
            cursor, epochs = pending['cursor'], pending['epochs']
            pending = None
        else:
            keys = selector.sample(train_keys, args.groups_per_update)
            backend.pipe.vae.to(backend.device)
            groups, adv, rollout_event = _collect_batch(backend, data, route, bank, targets, keys, args, selector, weights, directory)
            counters.rollout(sum(len(rows) for rows, _, _ in groups))
            conditions = [g['condition_sha256'] for g in rollout_event['groups']]
            write_json(directory / 'rollout.json', {**event, **counters.record(), **rollout_event})
            backend.pipe.vae.to('cpu')
            if not training or not torch.count_nonzero(adv):
                event['max_ratio_error'] = audit_replay(groups, noise_level=args.noise_level, tolerance=args.parity_tolerance)
                event['status'] = 'preflight_passed' if not training else 'zero_advantage'
                write_json(directory / 'replay.json', {**event, **counters.record()})
                continue
            means = fixed_reference_means(reference, groups, noise_level=args.noise_level)
            cursor, epochs = 0, counters.epochs(args.updates, args.replay_epochs)

        event['replay'] = []
        event['status'] = 'updated'
        replay_seconds, validation_seconds = 0., 0.
        for epoch in range(cursor, epochs):
            started = time.monotonic()
            stats = update_policy(optimizer, groups, active, frozen, reference_means=means,
                                  noise_level=args.noise_level, clip=args.clip, beta=args.beta,
                                  tolerance=args.parity_tolerance, check_parity=(epoch == 0))
            replay_seconds += time.monotonic() - started
            counters.updated(sum(len(rows) for rows, _, _ in groups))
            update = counters.optimizer_updates
            event['replay'].append({'epoch': epoch, 'optimizer_update': update, **stats})
            exhausted = counters.rollout_batches == attempt_limit and epoch + 1 == epochs
            due = update % args.validate_every == 0 or update == args.updates or exhausted
            improved, regression_stop = False, False
            if due:
                started = time.monotonic()
                improved, regression_stop = validate()
                validation_seconds += time.monotonic() - started
            paused = update == args.stop_after_updates
            if due or paused or update % args.save_every == 0:
                can_resume = not regression_stop and update < args.updates and not exhausted
                remainder = _pending_batch(keys, conditions, groups, means, epoch + 1, epochs) if can_resume and epoch + 1 < epochs else None
                save(remainder, improved=improved, resumable=can_resume)
            if regression_stop or paused or update == args.updates:
                stop = 'ode_regression' if regression_stop else 'paused' if paused else 'budget'
                break
        event.update(counters.record())
        event.update(replay_seconds=replay_seconds, evaluation_and_save_seconds=validation_seconds,
                     peak_cuda_gib=torch.cuda.max_memory_allocated() / 2**30)
        write_json(directory / 'replay.json', event)
        print(json.dumps({**counters.record(), 'status': stop if stop != 'rollout_budget' else 'updated'}), flush=True)
        if stop != 'rollout_budget':
            break

    if training and not counters.optimizer_updates:
        stop = 'no_reward_signal'
    elif stop == 'rollout_budget' and counters.record() != last_saved:
        improved, regression_stop = validate() if counters.optimizer_updates > last_validated else (False, False)
        if regression_stop:
            stop = 'ode_regression'
        save(None, improved=improved, resumable=False)
    result = {'stop': stop, **counters.record(), 'requested_updates': args.updates, 'best_rl': best,
              'checkpoint': last_checkpoint, 'selected': initialization if best is None else best, 'baseline': baseline}
    write_json(output / 'summary.json', result)
    print(json.dumps(result), flush=True)
