"""Shared options for the regional RL driver and its launchers."""

import json
import math
from pathlib import Path

from src.evaluation.native_kprpe import add_adaface_arguments, adaface_arguments
from .runner import add_sde_arguments, configure_sde
from .spatial_reward import add_spatial_arguments, configure_spatial, spatial_arguments
from .validation import add_regression_arguments, regression_limits


ROLLOUT_STEPS = 10
DEFAULT_SDE_INDICES = (1, 2, 3)


def latest_regional_commit(directory):
    markers = (json.loads(path.read_text(encoding='utf8')) for path in sorted(directory.glob('*.commit.json')))
    return max(markers, key=lambda row: (row['completed_updates'], row.get('rollout_batches', 0),
                                        bool(row.get('stop'))), default=None)


def add_arguments(parser, *, launcher=False):
    if launcher:
        parser.add_argument('--rl-python', type=Path, help='Regional RL interpreter; SFT keeps its own interpreter')
    parser.add_argument('--reward-profile', choices=('hands42', 'regional-legacy'), default='regional-legacy')
    add_sde_arguments(parser)
    add_adaface_arguments(parser)
    add_spatial_arguments(parser)
    add_regression_arguments(parser)
    parser.add_argument('--validate-every', type=int, default=10)
    parser.add_argument('--val-limit', type=int, default=8)
    parser.add_argument('--ode-pose-device', default='cpu', help='SDPose device for held-out ODE only')


def configure_arguments(parser, args):
    configure_spatial(parser, args)
    args.sde_indices = configure_sde(parser, args, steps=ROLLOUT_STEPS, default=DEFAULT_SDE_INDICES)
    if min(args.validate_every, args.val_limit, args.patience) < 1 or any(
            not math.isfinite(value) or not 0 <= value <= 1 for value in
            (*regression_limits(args).values(), args.background_drop, args.binding_drop)):
        parser.error('Invalid regional ODE validation budget or tolerance')
    return [*spatial_arguments(args), *adaface_arguments(parser, args), '--reward-profile', args.reward_profile,
            '--sde-cycle', *map(str, args.sde_indices),
            '--validate-every', str(args.validate_every), '--val-limit', str(args.val_limit),
            '--ode-pose-device', args.ode_pose_device, '--patience', str(args.patience),
            '--pose-drop', str(args.pose_drop), '--id-drop', str(args.id_drop),
            '--hands-drop', str(args.hands_drop), '--background-drop', str(args.background_drop),
            '--binding-drop', str(args.binding_drop), '--two-person-drop', str(args.two_person_drop)]
