"""Native V6.2/V7 online RL with fixed held-out ODE evaluation."""

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import random
import sys
import time


def _arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check-data', 'preflight', 'train'))
    parser.add_argument('--config', type=Path, required=True, help='Native V6.2 or V7 SFT configuration with bound data/cache/initialization')
    parser.add_argument('--output', type=Path, help='New run directory under /data/shangguanrz/')
    parser.add_argument('--models', type=Path, default=Path('/home/shangguanrz/project/pic-edit/models'))
    parser.add_argument('--route', help='Route name from the SFT configuration')
    parser.add_argument('--updates', '--optimizer-updates', type=int, default=1, help='Number of successful optimizer.step calls')
    parser.add_argument('--max-rollout-batches', type=int, help='Rollout cap including groups without an effective advantage (default: 5 × updates)')
    parser.add_argument('--pose-device', default='cpu', help='SDPose detector/inference device, e.g. cpu or cuda:0')
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
    parser.add_argument('--val-limit', type=int, help='Fixed sample cap for a bounded pilot')
    parser.add_argument('--pose-drop', type=float, default=.02)
    parser.add_argument('--id-drop', type=float, default=.02)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--keep-replay', action='store_true')
    args = parser.parse_args(argv)
    if (not 0 < args.sde_step < args.steps - 1 or args.group_size < 2 or
            min(args.updates, args.groups_per_update, args.replay_epochs, args.validate_every, args.patience) < 1 or
            (args.val_limit is not None and args.val_limit < 1) or
            (args.max_rollout_batches is not None and args.max_rollout_batches < 1)):
        parser.error('Positive budgets and one internal SDE step required')
    if (not all(math.isfinite(x) for x in (args.lr, args.clip, args.beta, args.noise_level, args.parity_tolerance,
                                          args.std_floor, args.pose_drop, args.id_drop)) or
            min(args.lr, args.noise_level, args.parity_tolerance) <= 0 or not 0 < args.clip < 1 or
            min(args.beta, args.std_floor, args.pose_drop, args.id_drop) < 0):
        parser.error('Invalid optimizer, SDE, variance or regression settings')
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
    if args.action == 'train' and config.experiment_mode != 'standard':
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


def _targets(bank, data, records):
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
            targets[key] = bank.target(sample, condition)
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
    return sum(weights.get(k, 0.) * v for k, v in metrics.items()) / sum(weights.get(k, 0.) for k in metrics)


def _save(backend, directory, update, result):
    import torch
    from ..native_v62.training.io import file_sha, write_json

    checkpoint = directory / f'update-{update:06d}.pt'
    payload = backend.runtime.checkpoint_payload(metadata={'native_rl': result})
    payload['native_rl_state'] = result
    with checkpoint.open('xb') as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    ref = {'path': str(checkpoint), 'sha256': file_sha(checkpoint)}
    write_json(checkpoint.with_suffix('.commit.json'), {'schema': 'native_rl_commit_v2', 'checkpoint': ref,
                                                        'architecture': payload['architecture'], **result})
    return ref


def main(argv=None):
    args = _arguments(argv)
    import torch
    from ..native_v62.observation import DiagnosticsV62
    from ..native_v62.training.checkpointing import load_committed
    from ..native_v62.training.data import route_options
    from ..native_v62.training.io import file_sha, verify, write_json
    from .objective import advantages
    from .rewards import DUAL_WEIGHTS, SINGLE_WEIGHTS, ImageRewardBank, evaluator_assets
    from .runner import FrozenReference, RunCounters, audit_replay, fixed_reference_means, update_policy

    config, route = _settings(args)
    data_type, backend_type, architecture = _backend_for(config)
    data = data_type(config)
    train_rows = [r for r in data.records if r['eligible'] and r['role'] == 'train']
    val_rows = [r for r in data.records if r['eligible'] and r['role'] == 'validation']
    if args.action != 'train' and config.experiment_mode == 'fixed_pair_overfit':
        val_rows = [r for r in data.records if r['eligible'] and r['role'] == 'observation']
    if not train_rows or not val_rows:
        raise ValueError('Native RL requires eligible training and evaluation conditions')
    if args.action == 'train':
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
    if args.action == 'train' and not state.get('stage_update', 0):
        raise ValueError('The committed SFT stage must have successful updates')
    print(json.dumps({'architecture': architecture, 'initialization': initialization, 'lineage': lineage}), flush=True, file=sys.stderr)
    evaluator_hashes = {str(path): file_sha(path) for path in evaluator_assets(args.models)}
    bank = ImageRewardBank(args.models, pose_device=args.pose_device)
    targets, skipped, provenance = _targets(bank, data, train_rows + val_rows)
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
    weights = {'pose': 1.} if args.pose_only else SINGLE_WEIGHTS if scene == 'single' else DUAL_WEIGHTS
    metadata = {'architecture': architecture, 'initialization': initialization, 'sft_lineage': lineage,
                'pose_backend': 'sdpose', 'data_signature': data.signature, 'evaluator_sha256': evaluator_hashes,
                'scene': scene, 'weights': weights, 'train_tasks': len(train_keys), 'eval_tasks': len(val_keys),
                'evaluation_kind': config.experiment_mode, 'skipped_targets': skipped, 'route': route.name,
                'eligible': [{'source': k[0], 'index': k[1], 'id_people': sum(x is not None for x in targets[k][0].identities),
                              'contacts': len(targets[k][0].contacts), **provenance[k]} for k in train_keys + val_keys]}
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
        selected = args.action == 'train' and name in active_names
        p.requires_grad_(selected)
        if selected:
            active.append(p)
    active_ids = {id(p) for p in active}
    parameters = dict.fromkeys([*backend.catalog.parameters.values(), *backend.pipe.transformer.parameters()])
    frozen = [p for p in parameters if id(p) not in active_ids]
    backend.catalog.assert_flags(active_names if active else set())
    metadata.update({'schema': 'native_rl_run_v2', 'identity': backend.identity, 'active_names': sorted(active_names) if active else [],
                     'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}})
    write_json(output / 'run.json', metadata)
    selector = random.Random(config.seed)
    backend.pipe.vae.to(backend.device)
    baseline, baseline_rows = _evaluate(backend, data, route, bank, targets, val_keys, args, output / 'ode-sft')
    write_json(output / 'ode-sft.json', {'metrics': baseline, 'rows': baseline_rows})
    if args.action == 'train' and not args.pose_only and (
            'identity' not in baseline or not any(any(x is not None for x in targets[k][0].identities) for k in train_keys)):
        raise ValueError('Pose/ID training requires eligible faces in both training and held-out evaluation')
    optimizer = torch.optim.AdamW(active, lr=args.lr, weight_decay=0.) if active else None
    reference = FrozenReference(active)
    best, best_score, regressions = None, _score(baseline, weights), 0
    counters = RunCounters()
    last_validated = 0

    def validate_and_save():
        nonlocal best, best_score, regressions, last_validated
        update = counters.optimizer_updates
        metrics, rows = _evaluate(backend, data, route, bank, targets, val_keys, args, output / f'ode-{update:06d}')
        rejected = any(metrics[k] < baseline[k] - tol for k, tol in
                       (('pose', args.pose_drop), ('identity', args.id_drop)) if k in baseline)
        result = {**counters.record(), 'metrics': metrics, 'regressed': rejected,
                  'sft_reference': initialization, 'run': str(output / 'run.json')}
        write_json(output / f'ode-{update:06d}.json', {**result, 'rows': rows})
        last = _save(backend, output, update, result)
        score = _score(metrics, weights)
        if not rejected and score > best_score:
            best, best_score = last, score
        regressions = regressions + 1 if rejected else 0
        last_validated = update
        return regressions >= args.patience

    attempt_limit = (args.max_rollout_batches or args.updates * 5) if optimizer else 1
    stop = 'rollout_budget' if optimizer else 'preflight'
    for batch_id in range(1, attempt_limit + 1):
        started = time.monotonic()
        keys = selector.sample(train_keys, args.groups_per_update)
        seeds = [selector.randrange(2**63) for _ in range(args.group_size * len(keys))]
        directory = output / f'rollout-{batch_id:06d}'
        directory.mkdir()
        replay_groups, reward_groups, details, native_stats = [], [], [], []
        reward_seconds = 0.
        for i, key in enumerate(keys):
            rollout = _generator(backend, route, key, args)
            replay, scores, candidates = [], [], []
            for j in range(args.group_size):
                seed = seeds[i * args.group_size + j]
                observer = DiagnosticsV62() if config.diagnostics and j == 0 else None
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
            replay_groups.append((replay, rollout.predict))
            reward_groups.append({name: [score[name] for score in scores] for name in scores[0]})
            details.append({'source': key[0], 'pair_id': data.datasets[key[0]].rows[key[1]]['pair_id'],
                            'condition_sha256': rollout.pair['condition_sha256'], 'candidates': candidates})
        adv, stats = advantages(reward_groups, weights, mode=args.aggregation, std_floor=args.std_floor)
        grouped = [(rows, a, predict) for (rows, predict), a in zip(replay_groups, adv)]
        if args.keep_replay:
            torch.save([[{**vars(row), 'state': row.state.cpu(), 'action': row.action.cpu(),
                          'old_log_prob': row.old_log_prob.cpu()} for row in rows] for rows, _, _ in grouped], directory / 'replay.pt')
        counters.rollout(len(seeds))
        event = {'batch': batch_id, **counters.record(), 'groups': details, 'advantage_stats': stats, 'advantages': adv.tolist(),
                 'reward_summary': _summarize([c for g in details for c in g['candidates']]),
                 'rollout_seconds': time.monotonic() - started - reward_seconds, 'reward_seconds': reward_seconds,
                 'native_diagnostics': native_stats}
        write_json(directory / 'rollout.json', event)
        replay_started = time.monotonic()
        evaluation_seconds = 0.
        if optimizer is None or not torch.count_nonzero(adv):
            event['max_ratio_error'] = audit_replay(grouped, noise_level=args.noise_level, tolerance=args.parity_tolerance)
            event['status'] = 'preflight_passed' if optimizer is None else 'zero_advantage'
        else:
            backend.pipe.vae.to('cpu')
            reference_means = fixed_reference_means(reference, grouped, noise_level=args.noise_level)
            event['replay'] = []
            for epoch in range(counters.epochs(args.updates, args.replay_epochs)):
                event['replay'].append(update_policy(optimizer, grouped, active, frozen, reference_means=reference_means,
                                                     noise_level=args.noise_level, clip=args.clip, beta=args.beta,
                                                     tolerance=args.parity_tolerance, check_parity=(epoch == 0)))
                counters.updated(len(seeds))
                update = counters.optimizer_updates
                if update % args.validate_every == 0 or update == args.updates:
                    eval_started = time.monotonic()
                    should_stop = validate_and_save()
                    evaluation_seconds += time.monotonic() - eval_started
                    if should_stop:
                        stop = 'ode_regression'
                        break
            event['status'] = 'updated'
            backend.pipe.vae.to(backend.device)
        event['replay_seconds'] = time.monotonic() - replay_started - evaluation_seconds
        event['evaluation_and_save_seconds'] = evaluation_seconds
        event.update(counters.record())
        event['peak_cuda_gib'] = torch.cuda.max_memory_allocated() / 2**30
        write_json(directory / 'replay.json', {k: v for k, v in event.items() if k not in ('groups', 'native_diagnostics')})
        print(json.dumps({**counters.record(), **{k: event[k] for k in ('status', 'replay_seconds', 'peak_cuda_gib')}}), flush=True)
        if stop == 'ode_regression':
            break
        if optimizer is not None and counters.optimizer_updates == args.updates:
            stop = 'budget'
            break
    if optimizer is not None and not counters.optimizer_updates:
        stop = 'no_reward_signal'
    elif counters.optimizer_updates > last_validated and validate_and_save():
        stop = 'ode_regression'
    result = {'stop': stop, **counters.record(), 'requested_updates': args.updates, 'best_rl': best,
              'selected': initialization if best is None else best, 'baseline': baseline}
    write_json(output / 'summary.json', result)
    print(json.dumps(result), flush=True)
