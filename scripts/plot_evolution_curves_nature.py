"""Plot quantitative trends across intermediate checkpoints following Nature guidelines.

Loads stage_evolution_metrics.json (steps 500, 1000, 1500, 2000, 2500, 3000)
and plots:
- Panel a: PSNR (dB) evolution per test pair & mean
- Panel b: Structural Similarity (SSIM) progression
- Panel c: Mean Absolute Error (MAE) decay curve
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Nature styling rcParams
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.4,
    "legend.frameon": False,
})

PALETTE = {
    "blue_main":      "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_main":     "#2E9E44",
    "red_strong":     "#B64342",
    "teal":           "#2B7A78",
    "amber":          "#D97706",
    "purple":         "#785EF0",
    "neutral_mid":    "#767676",
}


def main():
    metrics_path = Path("experiments/iper_adapter_finetune_v1/stage_evolution/stage_evolution_metrics.json")
    if not metrics_path.exists():
        print(f"File not found: {metrics_path}")
        return

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    steps = [int(k) for k in data.keys()]
    steps.sort()

    mean_psnr = [data[str(s)]["avg_psnr"] for s in steps]
    mean_ssim = [data[str(s)]["avg_ssim"] for s in steps]
    mean_mae = [data[str(s)]["avg_mae"] for s in steps]

    num_pairs = len(data[str(steps[0])]["details"])
    pair_psnr = {p: [data[str(s)]["details"][p]["psnr"] for s in steps] for p in range(num_pairs)}
    pair_ssim = {p: [data[str(s)]["details"][p]["ssim"] for s in steps] for p in range(num_pairs)}
    pair_mae = {p: [data[str(s)]["details"][p]["mae"] for s in steps] for p in range(num_pairs)}

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), dpi=300)
    fig.subplots_adjust(left=0.08, right=0.96, top=0.82, bottom=0.18, wspace=0.34)

    markers = ["o", "s", "^"]
    pair_colors = [PALETTE["blue_secondary"], PALETTE["amber"], PALETTE["purple"]]

    # Panel A: PSNR
    ax_a = axes[0]
    ax_a.set_title(r"$\mathbf{a}$  PSNR evolution (dB)", loc="left", pad=8)
    line_handles = []
    line_labels = []
    for p in range(num_pairs):
        l, = ax_a.plot(steps, pair_psnr[p], marker=markers[p], markersize=4, linestyle=":",
                       color=pair_colors[p], alpha=0.7, label=f"Pair {p+1}")
        if p == 0:
            line_handles.append(l)
            line_labels.append(f"Pair {p+1}")
        else:
            line_handles.append(l)
            line_labels.append(f"Pair {p+1}")
    l_mean, = ax_a.plot(steps, mean_psnr, marker="D", markersize=5, linewidth=1.8,
                        color=PALETTE["blue_main"], label="Mean (Demo Pairs)")
    line_handles.append(l_mean)
    line_labels.append("Mean (Demo Pairs)")

    ax_a.set_xlabel("Checkpoint Step")
    ax_a.set_ylabel("PSNR (dB)")
    ax_a.set_xticks(steps)
    ax_a.set_ylim(18.5, 24.5)
    ax_a.grid(True, linestyle=":", alpha=0.5)

    # Panel B: SSIM
    ax_b = axes[1]
    ax_b.set_title(r"$\mathbf{b}$  Structural similarity (SSIM)", loc="left", pad=8)
    for p in range(num_pairs):
        ax_b.plot(steps, pair_ssim[p], marker=markers[p], markersize=4, linestyle=":",
                  color=pair_colors[p], alpha=0.7)
    ax_b.plot(steps, mean_ssim, marker="D", markersize=5, linewidth=1.8,
              color=PALETTE["green_main"])
    ax_b.set_xlabel("Checkpoint Step")
    ax_b.set_ylabel("SSIM Index")
    ax_b.set_xticks(steps)
    ax_b.set_ylim(0.945, 0.99)
    ax_b.grid(True, linestyle=":", alpha=0.5)

    # Panel C: MAE Decay
    ax_c = axes[2]
    ax_c.set_title(r"$\mathbf{c}$  Pixel error (MAE) decay", loc="left", pad=8)
    for p in range(num_pairs):
        ax_c.plot(steps, pair_mae[p], marker=markers[p], markersize=4, linestyle=":",
                  color=pair_colors[p], alpha=0.7)
    ax_c.plot(steps, mean_mae, marker="D", markersize=5, linewidth=1.8,
              color=PALETTE["red_strong"])
    ax_c.set_xlabel("Checkpoint Step")
    ax_c.set_ylabel("Mean Absolute Error")
    ax_c.set_xticks(steps)
    ax_c.set_ylim(5.5, 9.5)
    ax_c.grid(True, linestyle=":", alpha=0.5)

    # Unified Legend on top
    fig.legend(
        line_handles, line_labels,
        loc="upper center", bbox_to_anchor=(0.52, 0.99),
        ncol=4, frameon=False, fontsize=7.2,
        columnspacing=1.8, handletextpad=0.5
    )

    out_prefix = Path("experiments/iper_adapter_finetune_v1/nature_stage_evolution_trends")
    for ext, dpi in [("png", 300), ("pdf", 300), ("svg", 300)]:
        out_file = f"{out_prefix}.{ext}"
        fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
        print(f"[OK] Saved: {out_file}")
    plt.close(fig)


if __name__ == "__main__":
    main()
