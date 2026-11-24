"""Shared options for the regional RL driver and its launchers."""

from pathlib import Path

from src.evaluation.native_kprpe import add_adaface_arguments, adaface_arguments
from .runner import add_sde_arguments, configure_sde
from .spatial_reward import add_spatial_arguments, configure_spatial, spatial_arguments


ROLLOUT_STEPS = 10
DEFAULT_SDE_INDICES = (1, 2, 3)


def add_arguments(parser, *, launcher=False):
    if launcher:
        parser.add_argument('--rl-python', type=Path, help='Regional RL interpreter; SFT keeps its own interpreter')
    parser.add_argument('--reward-profile', choices=('hands42', 'regional-legacy'), default='regional-legacy')
    add_sde_arguments(parser)
    add_adaface_arguments(parser)
    add_spatial_arguments(parser)


def configure_arguments(parser, args):
    configure_spatial(parser, args)
    args.sde_indices = configure_sde(parser, args, steps=ROLLOUT_STEPS, default=DEFAULT_SDE_INDICES)
    return [*spatial_arguments(args), *adaface_arguments(parser, args), '--reward-profile', args.reward_profile,
            '--sde-cycle', *map(str, args.sde_indices)]
