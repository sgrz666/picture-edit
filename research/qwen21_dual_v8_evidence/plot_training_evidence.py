"""Reproduce the pilot figure from the adjacent real training logs."""
from pathlib import Path
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
rows = [json.loads(line) for line in (root / 'train_metrics.jsonl').read_text().splitlines() if line.strip()]
assert [row['step'] for row in rows] == list(range(1, 301))
validation = [(int(step), float(loss)) for step, loss in re.findall(
    r'SAVE (\d+) noise_validation ([0-9.eE+-]+)', (root / 'train.log').read_text())]
assert [step for step, _ in validation] == [50, 100, 150, 200, 250, 300]
ratios = [100 * row['last_microbatch']['diagnostics']['28']['total']['cumulative_ratio'] for row in rows]
fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
axes[0].plot(*zip(*validation), marker='o', color='#126579')
axes[0].set(xlabel='Optimizer step', ylabel='Region-weighted flow loss',
            title='Same image, three held-out noise seeds')
axes[0].annotate(f'{validation[-1][1]:.5f}', validation[-1], xytext=(-8, 14),
                 textcoords='offset points', ha='right')
axes[1].plot([row['step'] for row in rows], ratios, color='#a56332', linewidth=1)
axes[1].axhline(5, color='#666666', linestyle='--', label='5% soft penalty target')
axes[1].set(xlabel='Optimizer step', ylabel='Cumulative ROI residual / initial hidden RMS (%)',
            title='Layer 28, last microbatch; sampled noise varies')
axes[1].legend(frameon=False)
for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.2)
fig.suptitle('Qwen-Image-2.1 dual V8: one-pair 300-step pilot', fontsize=13)
fig.savefig(root / 'training_evidence.png', dpi=180)
print(json.dumps(dict(validation=validation, final_cumulative_ratio_percent=ratios[-1],
                      max_cumulative_ratio_percent=max(ratios),
                      interpretation='Optimization and soft-budget evidence, not image-quality or generalization evidence')))
