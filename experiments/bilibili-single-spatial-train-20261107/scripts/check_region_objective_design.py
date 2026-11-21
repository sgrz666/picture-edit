"""CPU-only mathematical reference for Region-Normalized FM v1.

This is an experiment-design calculator, not a Trainer entry point or sidecar
reader. It never loads a model, changes training inputs, or runs an optimizer.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

FAMILIES = {'person': 2, 'joint': 2, 'hands': 4, 'background': 1}
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / 'report/region-objective-v1/loss_spec.json'


def validate_spec(spec):
    if spec['schema'] != 'region_normalized_fm_design_v1' or spec['trainer_config_compatible'] is not False:
        raise ValueError('An explicit reference-only loss specification is required')
    if spec['renormalize_missing_families'] or spec['sigma_or_route_schedule']:
        raise ValueError('The v1 definition has no implicit rescaling or schedule')
    if spec['aggregation'] != 'mean_valid_rois_per_image_then_mean_eligible_images_per_window':
        raise ValueError('The v1 averaging order must be explicit')
    if spec['hand_slot_order'] != ['A-left', 'A-right', 'B-left', 'B-right']:
        raise ValueError('Stable hand slot order required')
    if set(spec['mass_floor_tokens']) != set(FAMILIES):
        raise ValueError('Every region family needs a mass floor')
    for value in spec['mass_floor_tokens'].values():
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('Finite positive token-mass floor required')
    for weights in spec['presets'].values():
        if set(weights) != {'global', *FAMILIES}:
            raise ValueError('All five family coefficients must be explicit')
        if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in weights.values()):
            raise ValueError('Finite nonnegative family coefficients required')
        if weights['global'] <= 0:
            raise ValueError('The global anchor must be positive')
    if spec['presets']['uniform'] != {'global': 1., 'person': 0., 'joint': 0., 'hands': 0., 'background': 0.}:
        raise ValueError('Uniform must be the exact unweighted baseline')


def _validate_entry(entry):
    pred, target, coverage = entry['prediction'], entry['target'], entry['coverage']
    if pred.ndim != 3 or min(pred.shape) < 1 or pred.shape[0] != 1 or target.shape != pred.shape or coverage.shape != pred.shape[:2]:
        raise ValueError('Reference entries are B1 target tokens with matching coverage')
    if not pred.is_floating_point() or not target.is_floating_point():
        raise ValueError('Floating velocity tensors required')
    if any(x.device != pred.device or not torch.isfinite(x).all() for x in (target, coverage, pred)):
        raise ValueError('Velocity and coverage must be finite on one device')
    if ((coverage < 0) | (coverage > 1)).any() or coverage.sum() <= 0:
        raise ValueError('Nonempty valid Canvas coverage required')
    n = pred.shape[1]
    for family, size in FAMILIES.items():
        support = entry['regions'][family]
        quality = entry['quality'][family]
        if support.shape != (size, n) or quality.shape != (size,):
            raise ValueError('Incorrect slots or target token grid for ' + family)
        for value in (support, quality):
            if value.device != pred.device or not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError('Finite floating region support/quality required')
            if ((value < 0) | (value > 1)).any():
                raise ValueError('Support/quality must be in [0,1]')
        if (support > coverage[0] + 1e-6).any():
            raise ValueError('Region support exceeds actual coverage')


def reference_window(entries, spec, preset):
    """Mean each ROI, then each image, then eligible images in the window.

The list permits unequal target token counts. Real annotation/Canvas/ID binding
must be checked by the future typed sidecar reader, outside this reference.
"""
    validate_spec(spec)
    if preset not in spec['presets'] or not entries:
        raise ValueError('Known preset and a nonempty accumulation window required')
    weights = spec['presets'][preset]
    global_terms = []
    terms = {family: [] for family in FAMILIES}
    raw = {family: [] for family in FAMILIES}
    roi_counts = {family: 0 for family in FAMILIES}
    for entry in entries:
        _validate_entry(entry)
        pred, target, coverage = entry['prediction'], entry['target'], entry['coverage'].detach()
        squared = (pred.float() - target.float()).square()
        global_terms.append((squared * coverage[..., None]).sum() / (coverage.sum() * pred.shape[-1]))
        error = squared.mean(-1)[0]
        for family in FAMILIES:
            per_image, raw_per_image = [], []
            supports, qualities = entry['regions'][family].detach(), entry['quality'][family].detach()
            for support, quality in zip(supports, qualities):
                mass = support.sum()
                if mass <= 0 or quality <= 0:
                    continue
                numerator = (support * error).sum()
                per_image.append(quality * numerator / mass.clamp_min(spec['mass_floor_tokens'][family]))
                raw_per_image.append(numerator / mass)
            if per_image:
                terms[family].append(torch.stack(per_image).mean())
                raw[family].append(torch.stack(raw_per_image).mean())
                roi_counts[family] += len(per_image)
    zero = sum(x['prediction'].sum() * 0 for x in entries)
    family_losses = {'global': torch.stack(global_terms).mean()}
    family_losses.update({name: torch.stack(values).mean() if values else zero for name, values in terms.items()})
    objective = sum(weights[name] * value for name, value in family_losses.items())
    return objective, {
        'family_losses': family_losses,
        'common_raw_region_mse': {name: torch.stack(values).mean() if values else zero for name, values in raw.items()},
        'eligible_images': {'global': len(entries), **{name: len(values) for name, values in terms.items()}},
        'eligible_rois': roi_counts,
        'active_coefficient_sum': sum(weights[name] for name in family_losses if name == 'global' or terms[name]),
    }


def fixture(n=64):
    return {
        'prediction': torch.ones(1, n, 2, requires_grad=True),
        'target': torch.zeros(1, n, 2), 'coverage': torch.ones(1, n),
        'regions': {name: torch.zeros(size, n) for name, size in FAMILIES.items()},
        'quality': {name: torch.zeros(size) for name, size in FAMILIES.items()},
    }


def expect_invalid(entry, spec):
    try:
        reference_window([entry], spec, 'region_balanced_v1')
    except ValueError:
        return
    raise AssertionError('Malformed support was accepted')


def run_checks(spec):
    result = {}
    entry = fixture(2)
    entry['prediction'] = torch.tensor([[[1., 2.], [3., 4.]]], requires_grad=True)
    entry['coverage'] = torch.tensor([[1., .5]])
    actual, _ = reference_window([entry], spec, 'uniform')
    # This is the current B1 native_velocity_loss formula, including its operation order.
    sq = (entry['prediction'].float() - entry['target'].float()).square()
    legacy = (sq * entry['coverage'][..., None]).sum() / (entry['coverage'].sum() * 2)
    assert torch.equal(actual, legacy)
    assert torch.equal(torch.autograd.grad(actual, entry['prediction'], retain_graph=True)[0],
                       torch.autograd.grad(legacy, entry['prediction'])[0])
    result['uniform_value_and_gradient_exact'] = float(actual.detach())

    first, second = fixture(2), fixture(5)
    second['prediction'] = torch.full((1, 5, 2), 3., requires_grad=True)
    mixed, _ = reference_window([first, second], spec, 'uniform')
    assert mixed == 5
    result['different_token_counts_equal_image_mean'] = float(mixed.detach())

    entry = fixture()
    entry['regions']['hands'][0, :1] = 1
    entry['regions']['hands'][1, 1:17] = 1
    entry['quality']['hands'][:2] = 1
    value, parts = reference_window([entry], spec, 'region_balanced_v1')
    hand = parts['family_losses']['hands']
    gradient = torch.autograd.grad(hand, entry['prediction'], retain_graph=True)[0].abs().sum(-1)[0]
    torch.testing.assert_close(gradient[:1].sum(), gradient[1:17].sum())
    result['hands_area_ratio_1_to_16_equal_family_influence'] = True

    entry['quality']['hands'][:2] = .1
    _, low = reference_window([entry], spec, 'region_balanced_v1')
    torch.testing.assert_close(low['family_losses']['hands'], hand * .1)
    result['outside_quality_q01_scales_hand_term_by_01'] = True

    entry['quality']['hands'][:] = 0
    _, rejected_quality = reference_window([entry], spec, 'region_balanced_v1')
    assert rejected_quality['eligible_images']['hands'] == 0
    assert rejected_quality['eligible_rois']['hands'] == 0
    result['zero_quality_does_not_enter_denominator'] = True

    entry['quality']['hands'][:2] = 1
    _, window = reference_window([entry, fixture(), fixture(), fixture()], spec, 'region_balanced_v1')
    assert window['eligible_images']['hands'] == 1 and window['family_losses']['hands'] == 1
    result['one_hand_image_in_four_uses_denominator_one'] = True

    entry2 = fixture()
    entry2['prediction'] = torch.full((1, 64, 2), 2., requires_grad=True)
    entry2['regions']['hands'][0, :2] = 1
    entry2['quality']['hands'][0] = 1
    _, two_images = reference_window([entry, entry2], spec, 'region_balanced_v1')
    assert two_images['family_losses']['hands'] == 2.5
    result['per_image_then_window_mean_1_and_4'] = 2.5

    nohand, empty = reference_window([fixture(), fixture()], spec, 'region_balanced_v1')
    torch.testing.assert_close(nohand, torch.tensor(.1))
    assert empty['eligible_images']['hands'] == 0
    assert empty['active_coefficient_sum'] == .1
    assert torch.isfinite(nohand)
    result['empty_families_do_not_transfer_coefficients'] = True

    tiny = fixture()
    tiny['regions']['hands'][0, 0] = .01
    tiny['quality']['hands'][0] = 1
    tiny['regions']['hands'].requires_grad_()
    tiny['quality']['hands'].requires_grad_()
    _, tiny_parts = reference_window([tiny], spec, 'region_balanced_v1')
    tiny_loss = tiny_parts['family_losses']['hands']
    torch.testing.assert_close(tiny_loss, torch.tensor(.02))
    grad, mask_grad, quality_grad = torch.autograd.grad(tiny_loss,
        (tiny['prediction'], tiny['regions']['hands'], tiny['quality']['hands']), allow_unused=True)
    assert mask_grad is None and quality_grad is None and torch.isfinite(grad).all()
    result['tiny_mass_floor_and_detached_support_quality'] = {'loss': float(tiny_loss.detach()), 'max_gradient': float(grad.abs().max())}

    partial = fixture(2)
    partial['prediction'] = torch.tensor([[[1., 1.], [3., 3.]]], requires_grad=True)
    partial['coverage'] = torch.tensor([[1., .5]])
    partial['regions']['hands'][0] = torch.tensor([1., .5])
    partial['quality']['hands'][0] = 1
    _, padded = reference_window([partial], spec, 'region_balanced_v1')
    torch.testing.assert_close(padded['family_losses']['hands'], torch.tensor(5.5/1.5))
    result['partial_padding_applied_once'] = float(padded['family_losses']['hands'].detach())

    a = torch.tensor([[[[1., 1., 0., 0.]]]])
    b = 1 - a
    proper = F.interpolate(torch.minimum(a, b), size=(1, 1), mode='area')
    false = torch.minimum(F.interpolate(a, size=(1, 1), mode='area'), F.interpolate(b, size=(1, 1), mode='area'))
    assert proper.item() == 0 and false.item() == .5
    result['pixel_intersection_before_pooling'] = {'correct': proper.item(), 'false_token_intersection': false.item()}

    person = fixture()
    person['prediction'] = torch.cat((torch.ones(1, 4, 2), torch.full((1, 60, 2), 3.)), dim=1).requires_grad_()
    person['regions']['person'][0, :4] = 1
    person['regions']['person'][1, 4:36] = 1
    person['quality']['person'][:] = 1
    _, people = reference_window([person], spec, 'region_balanced_v1')
    assert people['family_losses']['person'] == 5
    result['two_people_equal_roi_mean'] = 5.

    rng = torch.random.get_rng_state().clone()
    reference_window([person], spec, 'region_balanced_v1')
    assert torch.equal(rng, torch.random.get_rng_state())
    result['loss_preserves_rng'] = True

    for invalid in ('negative', 'exceeds_coverage', 'nan_quality', 'wrong_hand_slots', 'zero_channels'):
        bad = fixture()
        if invalid == 'negative': bad['regions']['hands'][0, 0] = -.1
        elif invalid == 'exceeds_coverage':
            bad['coverage'][0, 0] = .1
            bad['regions']['hands'][0, 0] = .5
        elif invalid == 'nan_quality': bad['quality']['hands'][0] = float('nan')
        elif invalid == 'wrong_hand_slots': bad['regions']['hands'] = torch.zeros(2, 64)
        else:
            bad['prediction'] = torch.ones(1, 64, 0, requires_grad=True)
            bad['target'] = torch.zeros_like(bad['prediction'])
        expect_invalid(bad, spec)
    result['malformed_regions_rejected'] = ['negative', 'exceeds_coverage', 'nan_quality', 'wrong_hand_slots', 'zero_channels']

    changed = deepcopy(spec)
    changed['renormalize_missing_families'] = True
    try:
        validate_spec(changed)
    except ValueError:
        result['implicit_rescaling_rejected'] = True
    else:
        raise AssertionError('Undocumented rescaling was accepted')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, default=DEFAULT_SPEC)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing existing result: ' + str(args.output))
    spec_bytes = args.spec.read_bytes()
    spec = json.loads(spec_bytes)
    validate_spec(spec)
    checks = run_checks(spec)
    report = {'schema': 'region_objective_design_checks_v1', 'passed': True,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'CPU synthetic definition checks; no real sidecar reader, Trainer, model or optimizer',
        'loss_spec_sha256': hashlib.sha256(spec_bytes).hexdigest(), 'checks': checks,
        'unverified': ['Real GT/Canvas/person-ID binding', 'Real single/dual-person region coverage',
                       'Production Trainer integration', 'GPU gradients and generated image gains']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
