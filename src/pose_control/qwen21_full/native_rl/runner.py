"""Grouped Native rollout, exact replay and selective policy updates."""

from contextlib import contextmanager
from dataclasses import dataclass

import torch

from .objective import clipped_loss, transition


@dataclass(frozen=True)
class Replay:
    state: torch.Tensor
    action: torch.Tensor
    old_log_prob: torch.Tensor
    timestep: float
    sigma: float
    next_sigma: float


class NativeRollout:
    def __init__(self, backend, batch, pair, route, options, base_options, *, steps, sde_step, noise_level):
        self.pipe = backend.pipe
        self.runtime = backend.runtime
        self.device = backend.device
        self.batch = {key: value for key, value in batch.items() if key not in ('tgt_image', 'training_supervision_v5')}
        self.pair = {key: value for key, value in pair.items() if key != 'target_latent'}
        self.controls = dict(options=options, base_options=base_options, poselink_strength=route.poselink_strength)
        self.steps, self.sde_step, self.noise_level = steps, sde_step, noise_level

    def generate(self, seed, *, stochastic=False, observer=None, report=None):
        from ..native_v62.generation import generate_native

        replay = []
        generator = torch.Generator(device=self.device).manual_seed(seed ^ 0x53DE)

        def sde(i, t, state, velocity, sigma, next_sigma):
            if i != self.sde_step:
                return None
            sigma, next_sigma = float(sigma), float(next_sigma)
            action, old, _, _ = transition(state, velocity, sigma, next_sigma,
                                             noise_level=self.noise_level, generator=generator)
            if not old.isfinite().all():
                raise RuntimeError('Non-finite rollout likelihood')
            replay.append(Replay(state.detach(), action.detach(), old.detach(), float(t), sigma, next_sigma))
            return action

        image = generate_native(self.pipe, self.runtime, self.batch, self.pair, seed=seed, steps=self.steps,
                                observer=observer, sampling_report=report, step_override=sde if stochastic else None,
                                **self.controls)
        if stochastic and len(replay) != 1:
            raise RuntimeError('Rollout must record exactly one Flow-SDE step')
        return image, replay[0] if replay else None

    def predict(self, row):
        from ..native_v62.generation import native_velocity

        with native_velocity(self.pipe, self.runtime, self.batch, self.pair, **self.controls) as predict:
            return predict(row.state, row.timestep)


class FrozenReference:
    """Share frozen modules. Swap only selected tensors at no-grad boundaries."""

    def __init__(self, active):
        self.parameters = tuple(active)
        self.weights = tuple(p.detach().clone() for p in self.parameters)
        self.versions = tuple(w._version for w in self.weights)

    @contextmanager
    def use(self):
        if torch.is_grad_enabled():
            raise RuntimeError('SFT reference requires a no-grad boundary')
        if tuple(w._version for w in self.weights) != self.versions:
            raise RuntimeError('Frozen SFT reference changed')
        current = [p.detach().clone() for p in self.parameters]
        try:
            for param, ref in zip(self.parameters, self.weights):
                param.copy_(ref)
            yield
        finally:
            for param, value in zip(self.parameters, current):
                param.copy_(value)


def audit_replay(groups, *, noise_level=.7, tolerance=1e-4):
    with torch.no_grad():
        return max(_parity_error(transition(row.state, predict(row), row.sigma, row.next_sigma,
                                            noise_level=noise_level, action=row.action)[1], row.old_log_prob, tolerance)
                   for rows, _, predict in groups for row in rows)


def _parity_error(new, old, tolerance):
    error = float((new.detach().float() - old.float()).expm1().abs().max())
    if not error <= tolerance:
        raise RuntimeError(f'Replay parity failed: |ratio - 1| = {error:.6g}, tolerance = {tolerance:.6g}')
    return error


def update_policy(optimizer, groups, active, frozen, *, reference_means=None, noise_level=.7,
                  clip=.001, beta=.001, max_grad=1., tolerance=1e-4, check_parity=True):
    if beta and reference_means is None:
        raise ValueError('Nonzero KL beta requires the fixed SFT reference means')
    if (not active or any(not p.requires_grad for p in active) or
            {id(p) for group in optimizer.param_groups for p in group['params']} != {id(p) for p in active} or
            any(p.requires_grad or p.grad is not None for p in frozen)):
        raise RuntimeError('Optimizer and parameter freeze mask differ')
    frozen_versions = [p._version for p in frozen]
    count = sum(len(rows) for rows, _, _ in groups)
    if not count or any(len(rows) != len(adv) for rows, adv, _ in groups):
        raise ValueError('Replay candidates and advantages must agree')
    metrics = {'max_ratio_error': 0., 'policy_loss': 0., 'kl': 0., 'clip_fraction': 0.,
               'ratio_min': float('inf'), 'ratio_max': 0.}
    optimizer.zero_grad(set_to_none=True)
    try:
        for rows, adv, predict in groups:
            for row, value in zip(rows, adv):
                _, log_prob, mean, variance = transition(row.state, predict(row), row.sigma, row.next_sigma,
                                                         noise_level=noise_level, action=row.action)
                if check_parity:
                    err = _parity_error(log_prob, row.old_log_prob, tolerance)
                    metrics['max_ratio_error'] = max(metrics['max_ratio_error'], err)
                reference = mean.detach() if reference_means is None else reference_means[id(row)]
                loss, stats = clipped_loss(log_prob, row.old_log_prob, value.reshape(1).to(log_prob), mean,
                                           reference, variance, clip=clip, beta=beta)
                (loss / count).backward()
                for key in ('policy_loss', 'kl', 'clip_fraction'):
                    metrics[key] += stats[key] / count
                metrics['ratio_min'] = min(metrics['ratio_min'], stats['ratio_min'])
                metrics['ratio_max'] = max(metrics['ratio_max'], stats['ratio_max'])
        if any(p.grad is not None or p._version != version for p, version in zip(frozen, frozen_versions)):
            raise RuntimeError('Frozen parameter mutated or received a gradient')
        norm = float(torch.nn.utils.clip_grad_norm_(active, max_grad, error_if_nonfinite=True))
        if norm == 0:
            raise RuntimeError('Native policy received zero gradient from eligible rewards')
        optimizer.step()
        return {**metrics, 'grad_norm': norm}
    finally:
        optimizer.zero_grad(set_to_none=True)
