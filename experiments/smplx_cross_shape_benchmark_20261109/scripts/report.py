"""Generate the figures used by report/REPORT.md."""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def read_csv(path):
    with open(path, "r", encoding="utf8") as f:
        return list(csv.DictReader(f))


def main():
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    p1 = json.loads((config.OUT_P1 / "p1_summary.json").read_text())
    p1s = read_csv(config.OUT_P1 / "p1_per_subject.csv")
    p0 = json.loads((config.OUT_P0 / "p0_summary.json").read_text())
    p0rows = read_csv(config.OUT_P0 / "p0_per_pair.csv")

    # ---- P1 method comparison ----
    arms = ["G1", "G1_mean", "G2", "G2_mean", "G2_median", "G4"]
    labels = ["SMPLest-X\nsingle", "SMPLest-X\nmean", "SHAPY\nsingle",
              "SHAPY\nmean", "SHAPY\nmedian", "GT beta\nfloor"]
    vals = [p1[a]["subject_weighted_p2p20k_mm"] for a in arms]
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    colors = ["#5b7db1", "#8fb3d9", "#b05b5b", "#d98f8f", "#e0b3b3", "#777777"]
    bars = ax.bar(range(len(arms)), vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("P2P-20K (mm), subject-weighted")
    ax.set_title("HBW val: shape error by estimator")
    ax.set_ylim(0, max(vals) * 1.18)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "fig_p1_methods.png")
    plt.close(fig)

    # ---- paired per-subject P2P ----
    by_arm = defaultdict(dict)
    for r in p1s:
        if r["arm"] in ("G1", "G2"):
            by_arm[r["arm"]][r["subject_id"]] = float(r["p2p20k_mm"])
    subjects = sorted(set(by_arm["G1"]) & set(by_arm["G2"]))
    g1 = np.array([by_arm["G1"][s] for s in subjects])
    g2 = np.array([by_arm["G2"][s] for s in subjects])
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    for i, s in enumerate(subjects):
        color = "#2e7d32" if g2[i] < g1[i] else "#c62828"
        ax.plot([i, i], [g1[i], g2[i]], color=color, lw=1.5, alpha=0.8)
        ax.scatter(i, g1[i], color="#5b7db1", s=22, zorder=3)
        ax.scatter(i, g2[i], color="#b05b5b", s=22, zorder=3)
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels(subjects, rotation=45)
    ax.set_ylabel("P2P-20K (mm)")
    ax.set_title("SHAPY vs SMPLest-X, per subject")
    pair = p1["paired"]["G2_vs_G1_p2p20k_mm"]
    ax.text(0.02, 0.98,
            f"mean diff {pair['mean_diff']:+.2f} mm\n95% CI [{pair['ci95'][0]:+.2f}, {pair['ci95'][1]:+.2f}]\n"
            f"wins {pair['wins_b']}/10, sign p={pair['sign_test_p']:.3f}",
            transform=ax.transAxes, va="top", fontsize=8)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#5b7db1", markersize=6, label="SMPLest-X"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#b05b5b", markersize=6, label="SHAPY"),
        Line2D([0], [0], color="#2e7d32", lw=1.5, label="SHAPY better"),
        Line2D([0], [0], color="#c62828", lw=1.5, label="SHAPY worse"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "fig_p1_paired.png")
    plt.close(fig)

    # ---- stability + fusion ----
    stab = json.loads((config.OUT_P1 / "p1_stability.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    iccs = [stab["G1"]["_icc_mean"], stab["G2"]["_icc_mean"]]
    ax.bar([0, 1], iccs, color=["#5b7db1", "#b05b5b"])
    for i, v in enumerate(iccs):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["SMPLest-X", "SHAPY"])
    ax.set_ylabel("ICC(1,1) across images")
    ax.set_title("Shape stability")
    ax = axes[1]
    fusion = {
        "G1": (p1["G1"]["subject_weighted_p2p20k_mm"], p1["G1_mean"]["subject_weighted_p2p20k_mm"]),
        "G2": (p1["G2"]["subject_weighted_p2p20k_mm"], p1["G2_mean"]["subject_weighted_p2p20k_mm"]),
    }
    x = np.arange(2)
    ax.bar(x - 0.18, [fusion["G1"][0], fusion["G2"][0]], width=0.34, label="single image", color="#8fb3d9")
    ax.bar(x + 0.18, [fusion["G1"][1], fusion["G2"][1]], width=0.34, label="multi-image mean", color="#5b7db1")
    for i, (a, b) in enumerate([fusion["G1"], fusion["G2"]]):
        ax.text(i - 0.18, a + 0.3, f"{a:.1f}", ha="center", fontsize=7)
        ax.text(i + 0.18, b + 0.3, f"{b:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(["SMPLest-X", "SHAPY"])
    ax.set_ylabel("P2P-20K (mm)")
    ax.set_title("Multi-image fusion")
    ax.set_ylim(0, 30)
    ax.legend(fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    fig.subplots_adjust(bottom=0.28)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "fig_p1_stability.png")
    plt.close(fig)

    # ---- P0 arms by group ----
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    groups = ["same", "cross"]
    arms0 = ["G0", "G1", "G2", "G3"]
    group_colors = {"same": "#8fb3d9", "cross": "#b05b5b"}
    for ax, metric, ylabel in [
        (axes[0], "v2v_mm", "V2V (mm)"),
        (axes[1], "iou", "Silhouette IoU"),
    ]:
        x = np.arange(len(arms0))
        for gi, grp in enumerate(groups):
            vals = []
            for arm in arms0:
                sel = [float(r[metric]) for r in p0rows if r["group"] == grp and r["arm"] == arm]
                vals.append(np.mean(sel))
            ax.bar(x + (gi - 0.5) * 0.36, vals, width=0.34, label=grp,
                   color=group_colors[grp])
        ax.set_xticks(x)
        ax.set_xticklabels(arms0)
        ax.set_ylabel(ylabel)
        ax.set_title("P0 controlled swap")
        if metric == "iou":
            ax.set_ylim(0.7, 1.0)
    axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "fig_p0_arms.png")
    plt.close(fig)

    # ---- G0 positive control ----
    g0 = [r for r in p0rows if r["arm"] == "G0"]
    by_pair = defaultdict(list)
    dshape = {}
    for r in g0:
        by_pair[r["pair_id"]].append(float(r["v2v_mm"]))
        dshape[r["pair_id"]] = float(r["d_shape_canonical_mm"])
    xs = np.array([dshape[k] for k in by_pair])
    ys = np.array([np.mean(v) for v in by_pair.values()])
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.scatter(xs, ys, s=16, color="#5b7db1")
    if len(xs) > 2:
        coef = np.polyfit(xs, ys, 1)
        xx = np.linspace(xs.min(), xs.max(), 50)
        ax.plot(xx, np.polyval(coef, xx), color="#b05b5b", lw=1)
    rho = p0["g0_vs_shape_difference_spearman_by_pair"]["rho"]
    ax.set_xlabel("true shape difference A vs B (mm)")
    ax.set_ylabel("G0 V2V error (mm)")
    ax.set_title(f"No-swap control: rho={rho:.3f}")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "fig_g0_control.png")
    plt.close(fig)

    # ---- measurements ----
    meas_keys = ["height_mm", "chest_mm", "waist_mm", "hips_mm"]
    meas_labels = ["height", "chest", "waist", "hips"]
    arms_m = ["G1", "G2", "G2_mean", "G4"]
    colors_m = ["#5b7db1", "#b05b5b", "#e0b3b3", "#777777"]
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    x = np.arange(len(meas_keys))
    for i, arm in enumerate(arms_m):
        vals = [p1[arm]["subject_weighted_" + k] for k in meas_keys]
        ax.bar(x + (i - 1.5) * 0.2, vals, width=0.19, label=arm, color=colors_m[i])
    ax.set_xticks(x)
    ax.set_xticklabels(meas_labels)
    ax.set_ylabel("MAE (mm)")
    ax.set_title("Anthropometric errors (subject-weighted)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "fig_p1_measurements.png")
    plt.close(fig)

    print("[report] figures ->", config.FIGURES)


if __name__ == "__main__":
    main()
