"""Flow-SDE, GDPO and clipped GRPO on packed Native target tokens.

Flow-SDE equations and latent-mean likelihood follow Flow-Factory v0.5.0
(ced2cd9b0465a2c9d733b9b8aed4d4351f950a22, Apache-2.0).
"""

from itertools import combinations
import math

import torch


def transition(state, velocity, sigma, next_sigma, *, noise_level=.7, action=None, generator=None):
    if not (0 < sigma < 1 and 0 <= next_sigma < sigma and math.isfinite(noise_level) and noise_level > 0):
        raise ValueError('Flow-SDE requires an interior sigma and positive noise on a descending step')
    if state.shape != velocity.shape or state.ndim < 2 or not state.numel():
        raise ValueError('Velocity and packed state must have identical nonempty shapes')
    dt = next_sigma - sigma
    diffusion_sq = sigma / (1 - sigma) * noise_level**2
    mean = state.float() * (1 + diffusion_sq / (2 * sigma) * dt)
    mean = mean + velocity.float() * (1 + diffusion_sq * (1 - sigma) / (2 * sigma)) * dt
    variance = diffusion_sq * -dt
    if action is None:
        noise = torch.randn(state.shape, dtype=torch.float32, device=state.device, generator=generator)
        action = (mean.detach() + math.sqrt(variance) * noise).to(state.dtype)
    if action.shape != state.shape or action.dtype != state.dtype or action.device != state.device:
        raise ValueError('Replay action must retain the stored shape, dtype and device')
    density = -(action.detach().float() - mean).square() / (2 * variance) - .5 * math.log(2 * math.pi * variance)
    return action, density.flatten(1).mean(1), mean, variance


def _conflict(scores, contributions, fused):
    centered = {name: value - value.mean() for name, value in scores.items()}
    norms = {name: float(value.norm()) for name, value in centered.items()}
    pairs = []
    for left, right in combinations(sorted(centered), 2):
        correlation = None
        if min(norms[left], norms[right]) > 1e-12:
            cosine = float(centered[left] @ centered[right]) / (norms[left] * norms[right])
            correlation = max(-1., min(1., cosine))
        pairs.append({'left': left, 'right': right, 'pearson': correlation})
    scale = math.sqrt(fused.numel())
    components = [value - value.mean() for value in contributions]
    merged = fused - fused.mean()
    absolute_sum = sum(float(value.abs().sum()) for value in components)
    cancellation = None if absolute_sum <= 1e-12 else max(0., min(1., 1 - float(merged.abs().sum()) / absolute_sum))
    component_rms_sum = sum(float(value.norm()) for value in components) / scale
    fused_rms = float(merged.norm()) / scale
    return {'correlations': pairs, 'cancellation': cancellation, 'fused_rms': fused_rms,
            'component_rms_sum': component_rms_sum}


def advantages(groups, weights, *, mode='gdpo-guarded', std_floor=1e-3, std_floors=None, diagnostics=None):
    if mode not in ('gdpo', 'gdpo-guarded', 'sum') or not math.isfinite(std_floor) or std_floor < 0:
        raise ValueError('Invalid advantage mode or variance floor')
    if not weights or any(not math.isfinite(w) or w < 0 for w in weights.values()) or not any(weights.values()):
        raise ValueError('Finite nonnegative reward weights with positive sum required')
    std_floors = std_floors or {}
    if set(std_floors) - weights.keys() or any(not math.isfinite(v) or v < 0 for v in std_floors.values()):
        raise ValueError('Invalid per-reward variance floor')
    if not groups or not groups[0]:
        raise ValueError('One or more populated reward groups required')
    size = len(next(iter(groups[0].values())))
    if size < 2:
        raise ValueError('At least two candidates per group required')
    # Reward normalization is small and runs on CPU. Use Flow-Factory's FP64
    # statistics so subtracting nearly equal scores does not distort GDPO.
    result = torch.zeros(len(groups), size, dtype=torch.float64)
    stats, conflicts, counts = [], [], {}
    for row, group in zip(result, groups):
        scores = {name: torch.as_tensor(value, dtype=torch.float64).detach().cpu() for name, value in group.items()}
        if any(x.shape != row.shape or not x.isfinite().all() for x in scores.values()):
            raise ValueError('Every applicable reward needs finite scores for all candidates')
        active = sum(weight for name, weight in weights.items() if name in scores)
        if not active:
            raise ValueError('Each group needs a positively weighted applicable reward')
        trace, contributions = {}, []
        for name, weight in weights.items():
            if name not in scores:
                trace[name] = {'applicable': False}
                continue
            value = scores[name]
            mean, std = value.mean(), value.std(unbiased=False)
            floor = max(std_floors.get(name, std_floor), 1e-6)
            low = float(std) <= floor
            trace[name] = {'applicable': True, 'mean': float(mean), 'std': float(std), 'low_variance': low,
                           'std_floor': floor}
            if mode == 'sum':
                contribution = weight * value
            elif mode == 'gdpo':
                contribution = weight * (value - mean) / std.clamp_min(1e-6)
            elif weight and not low:
                contribution = (weight / active) * (value - mean) / std
            else:
                continue
            row += contribution
            if diagnostics is not None:
                contributions.append(contribution)
        if diagnostics is not None:
            conflicts.append(_conflict(scores, contributions, row))
            for name, value in scores.items():
                count = counts.setdefault(name, {'applicable': 0, 'below_floor': 0})
                floor = max(std_floors.get(name, std_floor), 1e-6)
                count['applicable'] += 1
                count['below_floor'] += int(float(value.std(unbiased=False)) <= floor)
        if mode == 'sum':
            row.sub_(row.mean()).div_(row.std(unbiased=False).clamp_min(1e-6))
        stats.append(trace)
    if diagnostics is not None:
        diagnostics.update(groups=conflicts, low_variance={
            name: {**count, 'rate': count['below_floor'] / count['applicable']} for name, count in counts.items()
        })
    if mode == 'gdpo':
        result = (result - result.mean()) / result.std(unbiased=False).clamp_min(1e-6)
    elif mode == 'gdpo-guarded':
        std = result.std(unbiased=False)
        if std > 1e-6:
            result = (result - result.mean()) / std
        else:
            result.zero_()
    return result.clamp(-5, 5).float(), stats


def clipped_loss(new, old, advantage, mean, reference_mean, variance, *, clip=.001, beta=.001):
    if (new.shape != old.shape or new.shape != advantage.shape or mean.shape != reference_mean.shape
            or mean.shape[0] != new.numel() or not (0 < clip < 1 and math.isfinite(beta) and beta >= 0)
            or not math.isfinite(variance) or variance <= 0):
        raise ValueError('Invalid PPO shapes or coefficients')
    log_ratio = new.float() - old.detach().float()
    ratio = log_ratio.clamp(-20, 20).exp()
    advantage = advantage.detach()
    policy = -torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage).mean()
    kl = (mean.float() - reference_mean.detach().float()).square().mean() / (2 * variance)
    loss = policy + beta * kl
    if not torch.isfinite(log_ratio).all() or not torch.isfinite(loss):
        raise RuntimeError('Non-finite GRPO likelihood or loss')
    stats = {'policy_loss': float(policy.detach()), 'kl': float(kl.detach()),
             'ratio_min': float(ratio.detach().min()), 'ratio_max': float(ratio.detach().max()),
             'clip_fraction': float(((ratio.detach() - 1).abs() > clip).float().mean())}
    return loss, stats
