"""CPU mathematical checks for spatial-loss proposals; never constructs a model.

Synthetic checks establish properties of loss definitions, not image quality.
Every invocation requires a new output directory and records the live source SHA.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import torch

REPO = Path(__file__).resolve().parents[3]
TOPIC = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    'src/pose_control/qwen21_full/native_v6/spatial_weights.py',
    'src/pose_control/qwen21_full/native_v6/training/objective.py',
    'src/pose_control/qwen21_full/native_v62/training/config.py',
    'src/pose_control/qwen21_full/native_v7/losses.py',
    'src/pose_control/qwen21_full/native_v7/training.py',
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def coefficient_share(area_fraction, multiplier):
    return multiplier * area_fraction / (1 + (multiplier - 1) * area_fraction)


def roi_mean(error, support, *, confidence=1., mass_floor=0.):
    """One ROI; error is channel-mean squared error on the target token grid."""
    support = support.detach()
    mass = support.sum()
    if mass <= 0:
        return error.sum() * 0
    return confidence * (error * support).sum() / mass.clamp_min(mass_floor)


def check_definitions():
    checks = {}
    hand_fraction = .0045
    checks['cap_coefficient_mass_share'] = {
        str(cap): coefficient_share(hand_fraction, cap) for cap in (2.5, 5., 10.)
    }
    checks['cap_for_20_percent_mass'] = .2 * (1 - hand_fraction) / (hand_fraction * .8)
    checks['hand_area_fraction_assumed'] = hand_fraction

    # Unit prediction errors make absolute output-gradient mass equal coefficient mass.
    prediction = torch.ones(10000, dtype=torch.float64, requires_grad=True)
    hand = torch.zeros_like(prediction)
    hand[:45] = 1
    weights = 1 + 1.5 * hand
    pixel_loss = (weights * prediction.square()).sum() / weights.sum()
    gradient = torch.autograd.grad(pixel_loss, prediction)[0].abs()
    observed_share = float((gradient * hand).sum() / gradient.sum())
    assert abs(observed_share - coefficient_share(hand_fraction, 2.5)) < 1e-12
    checks['uniform_error_output_gradient_share_cap25'] = observed_share

    # A large hand must not consume the small hand's independent ROI allocation.
    small, large = torch.zeros_like(prediction), torch.zeros_like(prediction)
    small[:10], large[10:170] = 1, 1
    equal_hands = (roi_mean(prediction.square(), small) + roi_mean(prediction.square(), large)) / 2
    g_equal = torch.autograd.grad(equal_hands, prediction)[0].abs()
    checks['small_hand_share_within_hand_family'] = {
        'union_normalization': float(small.sum() / (small + large).sum()),
        'independent_per_hand': float((g_equal * small).sum() / g_equal.sum()),
    }
    assert abs(checks['small_hand_share_within_hand_family']['independent_per_hand'] - .5) < 1e-12

    error = torch.tensor([1., 9., 25.], dtype=torch.float64)
    mask = torch.tensor([1., .5, 0.], dtype=torch.float64)
    original = roi_mean(error, mask)
    cancelled = roi_mean(error, .1 * mask)
    retained = roi_mean(error, mask, confidence=.1)
    assert torch.allclose(original, cancelled) and torch.allclose(retained, .1 * original)
    checks['confidence'] = {'baseline': float(original), 'inside_mask_q01': float(cancelled),
                            'outside_ratio_q01': float(retained)}

    # Clamp the denominator using fractional token MASS, not merely nonzero cells.
    tiny = torch.tensor([.01, 0., 0.], dtype=torch.float64)
    tiny.requires_grad_(True)
    probe = torch.ones(3, dtype=torch.float64, requires_grad=True)
    stabilised = roi_mean(probe.square(), tiny, mass_floor=.5)
    gp, gm = torch.autograd.grad(stabilised, (probe, tiny), allow_unused=True)
    assert gm is None and torch.isfinite(gp).all()
    checks['tiny_support'] = {'mass': float(tiny.detach().sum()), 'mass_floor': .5,
                             'loss': float(stabilised.detach()), 'max_output_gradient': float(gp.abs().max()),
                             'mask_receives_gradient': gm is not None}
    zero = roi_mean(probe.square(), torch.zeros_like(probe))
    assert torch.isfinite(zero) and zero == 0
    checks['empty_roi_is_finite_zero'] = True

    # Soft confidence can be very small without making the support spatially concentrated.
    support = torch.full((20,), .001, dtype=torch.float64)
    ess = support.sum().square() / support.square().sum()
    checks['mass_vs_effective_support_counterexample'] = {'mass': float(support.sum()),
                                                          'effective_cells': float(ess)}
    assert ess > 19 and support.sum() < .1

    # Fractional padding is already present in support. Applying coverage twice biases it.
    e, supported = torch.tensor([1., 9.]), torch.tensor([1., .5])
    once, twice = roi_mean(e, supported), roi_mean(e, supported * supported)
    checks['partial_coverage'] = {'once': float(once), 'twice': float(twice)}
    assert not torch.allclose(once, twice)

    checks['window_eligibility'] = {'one_active_roi_loss_over_four_microsteps': {
        'all_micro_denominator': 1 / 4, 'eligible_image_denominator': 1.}}
    checks['within_image_then_window'] = {
        'image1_one_hand_error': 1., 'image2_two_hand_errors': [3., 5.],
        'equal_image_mean': (1 + (3 + 5) / 2) / 2,
        'all_hands_pooled_mean': (1 + 3 + 5) / 3,
    }

    # Same shared parameter can have opposing background/hand gradients.
    theta = torch.tensor(.5, requires_grad=True)
    bg, hp = theta.square(), (theta - 2).square()
    bg_grad = torch.autograd.grad(bg, theta, retain_graph=True)[0]
    h_grad = torch.autograd.grad(hp, theta, retain_graph=True)[0]
    checks['shared_parameter_counterexample'] = {
        'background_gradient': float(bg_grad), 'hand_gradient': float(h_grad),
        'cosine': float(bg_grad * h_grad / (bg_grad.abs() * h_grad.abs())),
        'cap25_gradient_for_99_bg_1_hand': float((99 * bg_grad + 2.5 * h_grad) / 101.5),
        'equal_region_gradient': float((bg_grad + h_grad) / 2),
    }

    # An illustrative family allocation. Overlap adds explicit effective coefficients.
    bg_fraction, body_fraction = .8281, .1719
    checks['illustrative_family_coefficients'] = {'global': .1, 'person': .3, 'joint': .15, 'hand': .35, 'background': .1}
    checks['background_coefficient_mass_if_all_rois_qualified'] = .1 + .1 * bg_fraction
    checks['hand_token_mass_mean_in_active88_from_report'] = 222.79 / 88
    checks['body_fraction_for_illustration'] = body_fraction
    checks['parameter_gradient_share_is_not_inferred_from_loss_coefficients'] = True
    return checks


def check_existing_metrics():
    input_dir = TOPIC / 'input'
    path = input_dir / 'downstream_comparison.json'
    region_path = input_dir / 'region_loss_comparison.json'
    if not path.exists() or not region_path.exists():
        raise FileNotFoundError('Copy the two documented remote JSON inputs before running.')
    data = json.loads(path.read_text(encoding='utf-8'))
    regions = json.loads(region_path.read_text(encoding='utf-8'))
    result = {'evaluated_images': data['total_images_evaluated'], 'benchmarks': {}, 'input_sha256': {}}
    for name in (path, region_path):
        result['input_sha256'][name.name] = sha(name.read_bytes())
    for bench, states in data['benchmarks'].items():
        b, a = states['s1_base_10500_joint'], states['s1_bg035_10500_joint']
        result['benchmarks'][bench] = {
            'pairs': b['count'], 'background_mse_base': b['pixel']['bg']['mse_mean'],
            'background_mse_bg035': a['pixel']['bg']['mse_mean'],
            'background_mse_change_percent': 100 * (a['pixel']['bg']['mse_mean'] / b['pixel']['bg']['mse_mean'] - 1),
            'body_pck005_base': b['pose']['body17']['pck_005_mean'],
            'body_pck005_bg035': a['pose']['body17']['pck_005_mean'],
            'hand_mse_base': b['pixel']['hand_v2']['mse_mean'],
            'hand_mse_bg035': a['pixel']['hand_v2']['mse_mean'],
        }
    assert result['evaluated_images'] == 150
    assert sum(x['pairs'] for x in result['benchmarks'].values()) == 25
    terms = regions['states']['s1_bg035_10500_joint']['subsets']['val_hand_active88']['overall']['terms']
    result['hand_active88'] = {term: {key: terms[term][key] for key in ('sum_denominator', 'token_area_vs_valid_ratio')}
                               for term in ('L_hand_v2', 'L_unweighted')}
    result['hand_active88']['mean_hand_token_mass'] = terms['L_hand_v2']['sum_denominator'] / 88
    return result


def plot(path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), layout='constrained')
    area = np.geomspace(.001, .2, 300)
    for cap, color in ((2.5, '#61758a'), (5, '#176b9b'), (10, '#d27a29')):
        axes[0].plot(area * 100, coefficient_share(area, cap) * 100, label=f'Pixel multiplier {cap:g}', color=color)
    axes[0].axhline(35, color='#31816a', linestyle='--', label='Independent hand term: coefficient 0.35')
    axes[0].axvline(.45, color='#888888', linestyle=':', linewidth=1)
    axes[0].annotate('Observed hand area ~0.45%\n(cap 2.5: ~1.12% mass)', xy=(.45, 1.117), xytext=(.7, 12),
                     arrowprops={'arrowstyle': '-', 'color': '#777777'}, fontsize=9)
    axes[0].set(xscale='log', xlabel='Hand area / valid canvas (%)', ylabel='Normalized coefficient mass (%)', ylim=(0, 75),
                title='Fixed pixel caps still depend on hand area')
    axes[0].legend(loc='upper left', fontsize=7.8, frameon=False)
    axes[0].grid(axis='y', alpha=.15)
    axes[1].bar([0, 1], [100 / 17, 50], color=['#61758a', '#31816a'], width=.55)
    axes[1].set(xticks=[0, 1], xticklabels=['Both hands pooled', 'Each hand normalized'],
                ylabel='Small hand share within hand term (%)', ylim=(0, 65),
                title='Example: large hand has 16x the area')
    for i, y in enumerate((100 / 17, 50)):
        axes[1].text(i, y + 1.5, f'{y:.1f}%', ha='center')
    axes[1].grid(axis='y', alpha=.15)
    fig.suptitle('Loss-definition analysis; not a trained-model result', fontsize=12)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f'Refusing existing output directory: {out}')
    checks, metrics = check_definitions(), check_existing_metrics()
    source_records = []
    contents = []
    for relative in SOURCE_FILES:
        data = (REPO / relative).read_bytes()
        contents.append((relative, data))
        source_records.append({'path': relative, 'sha256': sha(data), 'size': len(data)})
    out.mkdir(parents=True)
    source_root = out / 'source_snapshot'
    source_root.mkdir()
    for relative, data in contents:
        name = relative.replace('/', '__')
        with (source_root / name).open('xb') as handle:
            handle.write(data)
    plot(out / 'loss_budget_comparison.png')
    report = {'schema': 'spatial_loss_research_checks_v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
              'torch': torch.__version__, 'scope': 'CPU synthetic mathematics and existing saved metrics; no model or optimizer.',
              'checks': checks, 'existing_metrics': metrics, 'source_snapshots': source_records}
    with (out / 'checks.json').open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps({'output_dir': str(out), 'checks': checks,
                      'existing_metrics': metrics}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
