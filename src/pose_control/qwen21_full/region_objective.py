"""Per-ROI, per-image, eligible-window FM without changing model conditions."""
import torch
from .region_config import FAMILIES, MASS_FLOORS, PRESETS


def region_eligibility(regional):
    return {name: bool(((regional['supports'][name].sum(-1) > 0) &
                        (regional['quality'][name] > 0)).any()) for name in FAMILIES}


def regional_counts(entries, routes, counts):
    """Use image eligibility, never the number of individual ROI slots."""
    if not any(getattr(routes[e['route']].loss, 'region_mode', 'legacy') != 'legacy' for e in entries):
        return counts
    counts = {**counts, **{'region_' + name: 0 for name in FAMILIES}}
    for entry in entries:
        if routes[entry['route']].loss.region_mode == 'legacy':
            continue
        eligible = entry['status'].get('region_eligible')
        if not isinstance(eligible, dict) or set(eligible) != set(FAMILIES):
            raise ValueError('Regional FM requires bound per-image ROI eligibility')
        for name in FAMILIES:
            if type(eligible[name]) is not bool:
                raise ValueError('Regional eligibility must be boolean')
            counts['region_' + name] += int(eligible[name])
    return counts


def regional_micro_loss(prediction, target, coverage, regional, mode, counts):
    if mode not in PRESETS or not isinstance(regional, dict):
        raise ValueError('Explicit regional preset and supervision required')
    if (prediction.ndim != 3 or min(prediction.shape) < 1 or prediction.shape[0] != 1 or
            target.shape != prediction.shape or coverage.shape != prediction.shape[:2] or
            target.device != prediction.device or coverage.device != prediction.device or
            not torch.isfinite(prediction).all() or not torch.isfinite(target).all() or
            not torch.isfinite(coverage).all() or ((coverage < 0) | (coverage > 1)).any() or
            coverage.sum() <= 0 or type(counts.get('velocity')) is not int or counts['velocity'] < 1):
        raise ValueError('Regional FM requires finite B1 target velocities and nonempty coverage')
    cached_coverage = regional['coverage'].to(coverage.device)
    if cached_coverage.shape != coverage.shape or not torch.allclose(cached_coverage.float(), coverage.float(), atol=1e-6, rtol=0):
        raise ValueError('Regional sidecar coverage differs from actual target')
    squared = (prediction.float() - target.float()).square()
    numerator = (squared * coverage[..., None]).sum()
    denominator = coverage.sum() * prediction.shape[-1]
    global_loss = numerator / denominator
    weights = PRESETS[mode]
    result = weights['global'] * global_loss / counts['velocity']
    parts = {'velocity': global_loss, 'velocity_numerator': numerator,
             'velocity_denominator': denominator, 'region_global': global_loss}
    error = squared.mean(-1)[0]
    for name, slots in FAMILIES.items():
        support = regional['supports'][name].to(prediction.device).detach().float()
        quality = regional['quality'][name].to(prediction.device).detach().float()
        if (support.shape != (slots, prediction.shape[1]) or quality.shape != (slots,) or
                not torch.isfinite(support).all() or not torch.isfinite(quality).all() or
                ((support < 0) | (support > coverage[0] + 1e-6)).any() or
                ((quality < 0) | (quality > 1)).any()):
            raise ValueError('Invalid regional support/quality: ' + name)
        mass = support.sum(-1)
        valid = (mass > 0) & (quality > 0)
        zero = prediction.sum() * 0
        if bool(valid.any()):
            n = (support[valid] * error).sum(-1)
            term = (quality[valid] * n / mass[valid].clamp_min(MASS_FLOORS[name])).mean()
            raw = (n / mass[valid]).mean()
            attenuation = (mass[valid] / mass[valid].clamp_min(MASS_FLOORS[name])).mean()
            effective = weights[name] * (quality[valid] * mass[valid] / mass[valid].clamp_min(MASS_FLOORS[name])).mean()
        else:
            term = raw = zero
            attenuation = effective = zero.detach()
        parts.update({f'region_{name}': term, f'region_{name}_raw': raw,
                      f'region_{name}_area': mass.sum(), f'region_{name}_count': int(valid.sum()),
                      f'region_{name}_quality_sum':quality[valid].sum(),
                      f'region_{name}_mass_attenuation':attenuation,
                      f'region_{name}_effective_coefficient':effective})
        if weights[name]:
            count = counts.get('region_' + name)
            if type(count) is not int or count < 0 or (bool(valid.any()) and count == 0):
                raise ValueError('Regional window count differs from actual ROI: ' + name)
            if count:
                result = result + weights[name] * term / count
    parts['regional_objective'] = result
    return result, parts
