"""Generate publication-quality training dynamics figures following Nature guidelines.

Loads train.log and eval_metrics.json to produce high-impact, multi-panel figures:
- Panel a: Flow Matching Loss & Validation Convergence across 3000 steps
- Panel b: Step Latency and Throughput Stability
- Panel c: GPU VRAM Memory Profiling (Allocated vs Reserved vs 84GB Hardware Ceiling)
- Panel d: Test-set Quantitative Performance Distribution (PSNR & SSIM)
"""

import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Nature styling rcParams
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",     # editable text in SVG
    "pdf.fonttype": 42,         # TrueType fonts in PDF
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

# Nature palette
PALETTE = {
    "blue_main":      "#0F4D92",
    "blue_secondary": "#3775BA",
    "blue_soft":      "#7884B4",
    "blue_light":     "#DCE6F5",
    "green_1":        "#DDF3DE",
    "green_2":        "#AADCA9",
    "green_3":        "#2E9E44",
    "red_light":      "#F6CFCB",
    "red_mid":        "#E9A6A1",
    "red_strong":     "#B64342",
    "neutral_light":  "#F0F0F2",
    "neutral_line":   "#D0D0D5",
    "neutral_mid":    "#767676",
    "neutral_dark":   "#333333",
    "teal":           "#2B7A78",
    "amber":          "#D97706",
}


def parse_train_log(log_path: Path):
    text = log_path.read_text(encoding="utf-8")

    # Parse steps: Step 10/3000 | Loss: 0.3098 | Speed: 429.3ms/step | VRAM Alloc: 17.63 GiB, Resv: 18.82 GiB
    step_pattern = re.compile(
        r"Step\s+(\d+)/3000\s+\|\s+Loss:\s+([\d\.]+)\s+\|\s+Speed:\s+([\d\.]+)ms/step\s+\|\s+VRAM Alloc:\s+([\d\.]+)\s+GiB,\s+Resv:\s+([\d\.]+)\s+GiB"
    )
    steps = []
    losses = []
    speeds = []
    vram_allocs = []
    vram_resvs = []

    for m in step_pattern.finditer(text):
        steps.append(int(m.group(1)))
        losses.append(float(m.group(2)))
        speeds.append(float(m.group(3)))
        vram_allocs.append(float(m.group(4)))
        vram_resvs.append(float(m.group(5)))

    # Parse validations
    val_pattern = re.compile(
        r"--- Running validation at step (\d+) ---[\s\S]*?Validation Loss:\s+([\d\.]+)"
    )
    val_steps = []
    val_losses = []
    for m in val_pattern.finditer(text):
        val_steps.append(int(m.group(1)))
        val_losses.append(float(m.group(2)))

    return {
        "steps": np.array(steps),
        "losses": np.array(losses),
        "speeds": np.array(speeds),
        "vram_alloc": np.array(vram_allocs),
        "vram_resv": np.array(vram_resvs),
        "val_steps": np.array(val_steps),
        "val_losses": np.array(val_losses),
    }


def parse_eval_metrics(metrics_path: Path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("per_sample", [])
    psnrs = [s["psnr"] for s in samples if s.get("psnr", 0) < 100]
    ssims = [s["ssim"] for s in samples]
    maes = [s["mae"] for s in samples]
    durations = [s["duration_sec"] for s in samples]
    return {
        "psnr": np.array(psnrs),
        "ssim": np.array(ssims),
        "mae": np.array(maes),
        "duration": np.array(durations),
        "summary": data.get("summary", {}),
    }


def compute_ema(series, alpha=0.1):
    ema = np.zeros_like(series)
    ema[0] = series[0]
    for i in range(1, len(series)):
        ema[i] = alpha * series[i] + (1 - alpha) * ema[i - 1]
    return ema


def make_nature_training_dashboard(train_data, eval_data, out_prefix: Path):
    # Two-column figure layout: 180mm width (~7.1 inches), 135mm height (~5.3 inches)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), dpi=300)
    fig.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.09, hspace=0.34, wspace=0.28)

    steps = train_data["steps"]
    losses = train_data["losses"]
    val_steps = train_data["val_steps"]
    val_losses = train_data["val_losses"]

    # -------------------------------------------------------------
    # Panel A: Flow Matching Loss & Convergence
    # -------------------------------------------------------------
    ax_a = axes[0, 0]
    ax_a.set_title(r"$\mathbf{a}$  Flow matching loss & validation dynamics", loc="left", pad=6)

    # Raw train loss points
    ax_a.plot(steps, losses, color=PALETTE["blue_soft"], alpha=0.35, linewidth=0.7, label="Train loss (raw)")

    # Smoothed EMA train loss
    ema_loss = compute_ema(losses, alpha=0.12)
    ax_a.plot(steps, ema_loss, color=PALETTE["blue_main"], linewidth=1.6, label="Train loss (EMA)")

    # Validation loss markers
    ax_a.plot(
        val_steps, val_losses,
        marker="D", markersize=4.5, linestyle="--", linewidth=1.0,
        color=PALETTE["red_strong"], markerfacecolor=PALETTE["red_strong"],
        markeredgecolor="white", markeredgewidth=0.8,
        label="Val loss (50 samples)",
    )

    # Highlight minimum validation loss at step 3000
    min_idx = np.argmin(val_losses)
    min_step = val_steps[min_idx]
    min_val = val_losses[min_idx]
    ax_a.annotate(
        f"Min Val: {min_val:.4f}\n(Step {min_step})",
        xy=(min_step, min_val),
        xytext=(min_step - 780, min_val + 0.12),
        arrowprops=dict(arrowstyle="->", color=PALETTE["red_strong"], lw=0.9),
        fontsize=7,
        fontweight="bold",
        color=PALETTE["red_strong"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=PALETTE["red_mid"], alpha=0.95),
    )

    ax_a.set_xlabel("Optimization step")
    ax_a.set_ylabel(r"Flow matching objective $\mathcal{L}_{\mathrm{FM}}$")
    ax_a.set_xlim(0, 3050)
    ax_a.set_ylim(0.0, 0.95)
    ax_a.grid(True, linestyle=":", alpha=0.5)
    ax_a.legend(loc="upper right", frameon=False, fontsize=7)

    # -------------------------------------------------------------
    # Panel B: Step Execution Latency & Throughput
    # -------------------------------------------------------------
    ax_b = axes[0, 1]
    ax_b.set_title(r"$\mathbf{b}$  Training throughput & iteration latency", loc="left", pad=6)

    speeds = train_data["speeds"]
    # Drop warm-up step 1 for smooth plotting
    valid_mask = steps > 1
    v_steps = steps[valid_mask]
    v_speeds = speeds[valid_mask]

    mean_speed = np.mean(v_speeds)
    std_speed = np.std(v_speeds)

    ax_b.plot(v_steps, v_speeds, color=PALETTE["amber"], alpha=0.45, linewidth=0.7, label="Iteration latency")
    ema_speed = compute_ema(v_speeds, alpha=0.1)
    ax_b.plot(v_steps, ema_speed, color=PALETTE["amber"], linewidth=1.4, label="Smoothed latency")

    ax_b.axhline(mean_speed, color=PALETTE["neutral_dark"], linestyle="--", linewidth=0.9, alpha=0.8)
    ax_b.text(
        150, mean_speed + 14,
        f"Mean: {mean_speed:.1f} ms/step (~{1000/mean_speed:.2f} it/s)",
        fontsize=7, color=PALETTE["neutral_dark"], fontweight="bold"
    )

    ax_b.set_xlabel("Optimization step")
    ax_b.set_ylabel("Step duration (ms)")
    ax_b.set_xlim(0, 3050)
    ax_b.set_ylim(280, 520)
    ax_b.grid(True, linestyle=":", alpha=0.5)

    # Secondary text badge for total duration
    total_time_min = 1190.9 / 60.0
    ax_b.text(
        0.96, 0.12,
        f"3000 steps completed in {total_time_min:.1f} min\nEffective batch: 4 (Accum=4, bs=1)",
        transform=ax_b.transAxes,
        ha="right", va="bottom",
        fontsize=6.8,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=PALETTE["neutral_light"], edgecolor=PALETTE["neutral_line"]),
    )
    ax_b.legend(loc="upper right", frameon=False, fontsize=7)

    # -------------------------------------------------------------
    # Panel C: GPU Memory Profiling (Allocated vs Reserved vs Limits)
    # -------------------------------------------------------------
    ax_c = axes[1, 0]
    ax_c.set_title(r"$\mathbf{c}$  GPU VRAM footprint & budget headroom", loc="left", pad=6)

    vram_alloc = train_data["vram_alloc"]
    vram_resv = train_data["vram_resv"]

    ax_c.fill_between(steps, 0, vram_resv, color=PALETTE["teal"], alpha=0.15, label="VRAM Reserved (Peak 19.07 GiB)")
    ax_c.plot(steps, vram_resv, color=PALETTE["teal"], linewidth=1.2, linestyle="-.")
    ax_c.plot(steps, vram_alloc, color=PALETTE["teal"], linewidth=1.6, label="VRAM Allocated (17.63 GiB)")

    # 24GB Consumer GPU Threshold reference line
    ax_c.axhline(24.0, color=PALETTE["amber"], linestyle="--", linewidth=1.0, alpha=0.9)
    ax_c.text(
        1550, 24.7,
        "24.0 GiB Consumer GPU Ceiling (RTX 4090 / 3090)",
        fontsize=6.8, color=PALETTE["amber"], fontweight="bold", ha="center"
    )

    ax_c.set_xlabel("Optimization step")
    ax_c.set_ylabel("GPU Memory (GiB)")
    ax_c.set_xlim(0, 3050)
    ax_c.set_ylim(0, 32.0)
    ax_c.grid(True, linestyle=":", alpha=0.5)

    peak_resv = np.max(vram_resv)
    peak_alloc = np.max(vram_alloc)
    ax_c.text(
        0.96, 0.12,
        f"Operational Peak: {peak_resv:.2f} GiB / {peak_alloc:.2f} GiB\n"
        f"24GB Consumer Headroom: 4.93 GiB (20.5%)\n"
        f"RTX 6000D Headroom: 65.53 GiB (77.5%)",
        transform=ax_c.transAxes,
        ha="right", va="bottom",
        fontsize=6.8,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=PALETTE["teal"], alpha=0.92),
    )
    ax_c.legend(loc="upper left", frameon=False, fontsize=7)

    # -------------------------------------------------------------
    # Panel D: Test Set Quantitative Evaluation (PSNR & SSIM)
    # -------------------------------------------------------------
    ax_d = axes[1, 1]
    ax_d.set_title(r"$\mathbf{d}$  Test set reconstruction fidelity ($N=50$ pairs)", loc="left", pad=6)

    psnr = eval_data["psnr"]
    ssim = eval_data["ssim"]

    # Twin axis: Left for PSNR, Right for SSIM
    ax_d2 = ax_d.twinx()
    ax_d2.spines["top"].set_visible(False)
    ax_d2.spines["right"].set_visible(True)
    ax_d2.spines["right"].set_color(PALETTE["green_3"])
    ax_d2.spines["right"].set_linewidth(0.8)
    ax_d.spines["left"].set_color(PALETTE["blue_main"])

    x_psnr = np.random.normal(1.0, 0.04, size=len(psnr))
    x_ssim = np.random.normal(2.0, 0.04, size=len(ssim))

    # Boxplot for PSNR
    bp1 = ax_d.boxplot(
        psnr, positions=[1.0], widths=0.35, patch_artist=True,
        showmeans=True, meanline=True,
        boxprops=dict(facecolor=PALETTE["blue_light"], color=PALETTE["blue_main"], lw=0.9),
        medianprops=dict(color=PALETTE["blue_main"], lw=1.2),
        meanprops=dict(color=PALETTE["blue_main"], lw=1.2, linestyle=":"),
        whiskerprops=dict(color=PALETTE["blue_main"], lw=0.9),
        capprops=dict(color=PALETTE["blue_main"], lw=0.9),
        flierprops=dict(marker="o", markersize=3, markeredgecolor=PALETTE["blue_main"]),
    )
    ax_d.scatter(x_psnr, psnr, color=PALETTE["blue_main"], alpha=0.45, s=12, zorder=3)

    # Boxplot for SSIM
    bp2 = ax_d2.boxplot(
        ssim, positions=[2.0], widths=0.35, patch_artist=True,
        showmeans=True, meanline=True,
        boxprops=dict(facecolor=PALETTE["green_1"], color=PALETTE["green_3"], lw=0.9),
        medianprops=dict(color=PALETTE["green_3"], lw=1.2),
        meanprops=dict(color=PALETTE["green_3"], lw=1.2, linestyle=":"),
        whiskerprops=dict(color=PALETTE["green_3"], lw=0.9),
        capprops=dict(color=PALETTE["green_3"], lw=0.9),
        flierprops=dict(marker="o", markersize=3, markeredgecolor=PALETTE["green_3"]),
    )
    ax_d2.scatter(x_ssim, ssim, color=PALETTE["green_3"], alpha=0.45, s=12, zorder=3)

    ax_d.set_xticks([1.0, 2.0])
    ax_d.set_xticklabels(["PSNR (dB)", "SSIM"], fontweight="bold")
    ax_d.set_ylabel("PSNR (dB)", color=PALETTE["blue_main"])
    ax_d2.set_ylabel("SSIM Index", color=PALETTE["green_3"])

    ax_d.tick_params(axis="y", labelcolor=PALETTE["blue_main"])
    ax_d2.tick_params(axis="y", labelcolor=PALETTE["green_3"])

    mean_psnr = np.mean(psnr)
    mean_ssim = np.mean(ssim)

    # Perfectly aligned at bottom of plot
    ax_d.text(1.0, 0.04, f"Mean: {mean_psnr:.2f} dB", transform=ax_d.get_xaxis_transform(),
              ha="center", va="bottom", fontsize=7.2, color=PALETTE["blue_main"], fontweight="bold")
    ax_d.text(2.0, 0.04, f"Mean: {mean_ssim:.4f}", transform=ax_d.get_xaxis_transform(),
              ha="center", va="bottom", fontsize=7.2, color=PALETTE["green_3"], fontweight="bold")

    ax_d.set_xlim(0.4, 2.6)
    ax_d.set_ylim(15.0, 29.5)
    ax_d2.set_ylim(0.89, 1.0)
    ax_d.grid(True, linestyle=":", alpha=0.5, axis="y")

    # Save to SVG, PDF, PNG, TIFF per Nature submission criteria
    for ext, dpi in [("png", 300), ("pdf", 300), ("svg", 300), ("tiff", 300)]:
        out_file = f"{out_prefix}.{ext}"
        fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
        print(f"[OK] Saved Nature figure: {out_file}")

    plt.close(fig)


def main():
    exp_dir = Path("experiments/iper_adapter_finetune_v1")
    train_log = exp_dir / "train.log"
    eval_json = exp_dir / "eval_results" / "eval_metrics.json"

    print("Parsing train.log...")
    train_data = parse_train_log(train_log)
    print(f"Parsed {len(train_data['steps'])} step entries, {len(train_data['val_steps'])} validation entries.")

    print("Parsing eval_metrics.json...")
    eval_data = parse_eval_metrics(eval_json)
    print(f"Parsed {len(eval_data['psnr'])} evaluation samples.")

    out_prefix = exp_dir / "nature_training_dynamics_dashboard"
    make_nature_training_dashboard(train_data, eval_data, out_prefix)
    print("All Nature dashboard formats rendered successfully!")


if __name__ == "__main__":
    main()
