"""Shared held-out regression criteria for Native RL."""

import math


def add_regression_arguments(parser):
    parser.add_argument('--pose-drop', type=float, default=.02)
    parser.add_argument('--id-drop', type=float, default=.02)
    parser.add_argument('--hands-drop', type=float, default=.02, help='Max held-out Hands or gesture reward drop')
    parser.add_argument('--two-person-drop', type=float, default=.02, help='Max held-out two-person detection rate drop')
    parser.add_argument('--background-drop', type=float, default=.02)
    parser.add_argument('--binding-drop', type=float, default=.02)
    parser.add_argument('--patience', type=int, default=2, help='Stop after this many consecutive ODE regressions')


def regression_limits(args):
    limits = {'pose': args.pose_drop, 'identity': args.id_drop,
              'hands': args.hands_drop, 'hands_gesture': args.hands_drop,
              'two_person_detection_rate': args.two_person_drop}
    if args.spatial_model is not None:
        limits.update(background=args.background_drop, binding=args.binding_drop)
    return limits


def regressions(metrics, baseline, limits):
    failures = []
    for key, tolerance in limits.items():
        if key not in baseline:
            continue
        measured = metrics.get(key)
        if measured is None or not math.isfinite(measured) or measured < baseline[key] - tolerance:
            failures.append(key)
    return failures
