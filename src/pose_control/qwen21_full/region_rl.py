"""Native Adapter Flow-SDE/GRPO primitives, independent of LoRA/model loading.

Flow-SDE transition adapted from Flow-Factory v0.5.0, commit
ced2cd9b0465a2c9d733b9b8aed4d4351f950a22, scheduler/flow_match_euler_discrete.py.
Copyright 2026 Jayce-Ping. Licensed under Apache License 2.0:
https://www.apache.org/licenses/LICENSE-2.0
The storage-dtype round trip is essential for rollout/replay parity.
"""
import math
import torch


def flow_sde_transition(latent, velocity, sigma, next_sigma, *, noise_level=.7, next_latent=None, generator=None):
    if (latent.shape != velocity.shape or latent.ndim < 2 or min(latent.shape) < 1 or
            not 0 < float(sigma) < 1 or not 0 <= float(next_sigma) < float(sigma) or
            not math.isfinite(noise_level) or noise_level <= 0):
        raise ValueError('Stochastic Flow-SDE requires a nonterminal interior sigma and descending step')
    if not torch.isfinite(latent).all() or not torch.isfinite(velocity).all():
        raise ValueError('Finite rollout/replay velocities and states required')
    x, v = latent.float(), velocity.float()
    dt = float(next_sigma) - float(sigma)
    diffusion = math.sqrt(float(sigma) / (1 - float(sigma))) * noise_level
    mean = x * (1 + diffusion**2 / (2 * sigma) * dt) + v * (1 + diffusion**2 * (1 - sigma) / (2 * sigma)) * dt
    std = diffusion * math.sqrt(-dt)
    if next_latent is None:
        noise = torch.randn(v.shape, device=v.device, dtype=v.dtype, generator=generator)
        next_latent = (mean + std * noise).to(latent.dtype)
    elif next_latent.shape != latent.shape or next_latent.device != latent.device:
        raise ValueError('Replay must use the exact stored next-state shape/device')
    log_prob = -((next_latent.detach().float() - mean).square()) / (2 * std**2) - math.log(std) - .5 * math.log(2 * math.pi)
    log_prob = log_prob.flatten(1).mean(1)
    return next_latent, log_prob, mean, std


def group_advantages(rewards, weights, applicable, *, epsilon=1e-6):
    values = list(rewards.values())
    if not values or values[0].ndim != 1 or values[0].numel() < 2:
        raise ValueError('At least two generated candidates per group required')
    advantage = torch.zeros_like(values[0], dtype=torch.float32)
    for name, value in rewards.items():
        if value.shape != advantage.shape or not torch.isfinite(value).all():
            raise ValueError('Finite, equally sized candidate reward components required')
        weight = weights.get(name, 0.)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('Finite nonnegative reward weights required')
        if not applicable.get(name, False) or weight == 0:
            continue
        value = value.detach().float()
        std = value.std(unbiased=False)
        if std > epsilon:
            advantage += weight * (value - value.mean()) / std
    std = advantage.std(unbiased=False)
    if std > epsilon:
        advantage = (advantage - advantage.mean()) / std
    return advantage.clamp(-5, 5).detach()


def grpo_loss(log_prob, old_log_prob, advantage, policy_reference_value, frozen_reference_value, *, clip=1e-4, beta=1e-3, variance=None):
    if log_prob.shape != old_log_prob.shape or log_prob.shape != advantage.shape or not 0 < clip < 1 or beta < 0:
        raise ValueError('Matched candidate likelihoods/advantages and explicit PPO settings required')
    if any(not torch.isfinite(x).all() for x in (log_prob, old_log_prob, advantage, policy_reference_value, frozen_reference_value)):
        raise ValueError('Finite policy and frozen-reference values required')
    ratio = (log_prob.float() - old_log_prob.detach().float()).clamp(-20, 20).exp()
    adv = advantage.detach().float()
    policy = torch.maximum(-adv * ratio, -adv * ratio.clamp(1 - clip, 1 + clip)).mean()
    penalty = (policy_reference_value.float() - frozen_reference_value.detach().float()).square()
    if variance is not None:
        if float(variance) <= 0:
            raise ValueError('Positive transition variance required')
        penalty = penalty / (2 * variance)
    penalty = penalty.mean()
    loss = policy + beta * penalty
    return loss, {'policy_loss': float(policy.detach()), 'reference_penalty': float(penalty.detach()),
                  'ratio_min': float(ratio.detach().min()), 'ratio_max': float(ratio.detach().max()),
                  'reference_penalty_kind': 'transition_gaussian_kl' if variance is not None else 'velocity_mse'}
