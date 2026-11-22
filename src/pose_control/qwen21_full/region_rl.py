"""Compatibility API for historical regional RL callers."""

import math

import torch

from .native_rl.objective import advantages, clipped_loss, transition


def flow_sde_transition(latent, velocity, sigma, next_sigma, *, noise_level=.7, next_latent=None, generator=None):
    action, log_prob, mean, variance = transition(
        latent, velocity, float(sigma), float(next_sigma), noise_level=noise_level,
        action=next_latent, generator=generator)
    return action, log_prob, mean, math.sqrt(variance)


def group_advantages(rewards, weights, applicable, *, epsilon=1e-6):
    available = {name: value for name, value in rewards.items() if applicable.get(name, False)}
    if not available or not any(weights.get(name, 0.) for name in available):
        example = next(iter(rewards.values()), None)
        if example is None or example.ndim != 1 or example.numel() < 2:
            raise ValueError('At least two generated candidates per group required')
        return torch.zeros_like(example, dtype=torch.float32)
    result, _ = advantages([available], weights, std_floor=epsilon)
    return result[0].to(next(iter(available.values())).device)


def grpo_loss(log_prob, old_log_prob, advantage, policy_reference_value, frozen_reference_value,
              *, clip=1e-4, beta=1e-3, variance=None):
    loss, info = clipped_loss(log_prob, old_log_prob, advantage, policy_reference_value,
                             frozen_reference_value, .5 if variance is None else float(variance),
                             clip=clip, beta=beta)
    info['reference_penalty'] = info.pop('kl')
    info['reference_penalty_kind'] = 'velocity_mse' if variance is None else 'transition_gaussian_kl'
    return loss, info
