# -*- coding: utf-8 -*-
"""训练进程监控可视化模块（iPER V6.3）。

解析 train_iper_v6_native.py 的训练日志，生成三栏训练动态曲线图与一份 Markdown 监控简报，
输出到「可视化/00_总览与说明/」。不生成网页，仅输出图片与文本。

曲线内容：
  面板1  训练损失（原始 + 滑动平均）
  面板2  残差 RMS 与梯度范数
  面板3  验证损失：Adapter ON / OFF / 扰动条件

用法：
  # 自动从远端拉取最新日志后绘图
  python scripts/plot_training_dashboard.py --fetch

  # 使用本地已有日志
  python scripts/plot_training_dashboard.py --log .codex_tmp/vis_pull/train_20k_v1.log
"""
from __future__ import annotations

import argparse
import re
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RE_STEP_NEW = re.compile(
    r"Step\s+(\d+)/(\d+)\s+\(Epoch\s+(\d+)\)\s+\|\s+Loss:\s+([\d.]+)\s+"
    r"\(Human:\s+([\d.]+),\s+BG:\s+([\d.]+)\)\s+\|\s+GradNorm:\s+([\d.]+)\s+\|\s+"
    r"RMS:\s+([\d.]+)\s+\|\s+Speed:\s+([\d.]+)ms/(\w+)"
)
RE_STEP_OLD = re.compile(
    r"Step\s+(\d+)/(\d+)\s+\|\s+Loss:\s+([\d.]+).*?Speed:\s+([\d.]+)ms/step"
)
RE_VAL = re.compile(
    r"Validation Loss:\s+([\d.]+)\s+\(Weighted:\s+([\d.]+)\)\s+\|\s+OFF:\s+([\d.]+)\s+\|\s+Perturbed:\s+([\d.]+)"
)

C_LOSS, C_SMOOTH, C_RMS, C_GRAD = "#185FA5", "#0C447C", "#BA7517", "#5F5E5A"
C_ON, C_OFF, C_PERT = "#185FA5", "#888780", "#BA7517"


def fetch_log(host: str, remote: str, local: Path) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["scp", "-q", f"{host}:{remote}", str(local)], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[warn] scp 失败，沿用本地日志：{r.stderr.strip()}")
        return False
    print(f"[info] 已拉取远端日志 → {local}")
    return True


def parse_log(path: Path) -> dict:
    steps, total, epoch = [], 0, 0
    loss, human, bg, grad, rms, speed_ms, per_step = [], [], [], [], [], [], []
    val_steps, val_on, val_off, val_pert = [], [], [], []
    last_step = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = RE_STEP_NEW.search(line)
        if m:
            steps.append(int(m.group(1)))
            total, epoch = int(m.group(2)), int(m.group(3))
            loss.append(float(m.group(4)))
            human.append(float(m.group(5)))
            bg.append(float(m.group(6)))
            grad.append(float(m.group(7)))
            rms.append(float(m.group(8)))
            speed_ms.append(float(m.group(9)))
            per_step.append(m.group(10))
            last_step = steps[-1]
            continue
        m = RE_STEP_OLD.search(line)
        if m:
            steps.append(int(m.group(1)))
            total = int(m.group(2))
            loss.append(float(m.group(3)))
            speed_ms.append(float(m.group(4)))
            per_step.append("step")
            last_step = steps[-1]
            continue
        m = RE_VAL.search(line)
        if m:
            val_steps.append(last_step)
            val_on.append(float(m.group(1)))
            val_off.append(float(m.group(3)))
            val_pert.append(float(m.group(4)))
    return dict(
        steps=np.array(steps), total=total, epoch=epoch, loss=np.array(loss),
        human=np.array(human), bg=np.array(bg), grad=np.array(grad), rms=np.array(rms),
        speed_ms=np.array(speed_ms), per_step=per_step,
        val_steps=val_steps, val_on=val_on, val_off=val_off, val_pert=val_pert,
    )


def rolling(x: np.ndarray, w: int) -> np.ndarray:
    n = len(x)
    if n == 0:
        return x
    w = max(1, min(w, n))
    if w == 1:
        return x.copy()
    c = np.concatenate(([0.0], np.cumsum(x)))
    idx = np.arange(n)
    lo = np.maximum(0, idx - w // 2)
    hi = np.minimum(n, idx + w - w // 2)
    return (c[hi] - c[lo]) / (hi - lo)


def eta_text(d: dict, accum: int) -> tuple[float, str]:
    if len(d["speed_ms"]) == 0:
        return float("nan"), "未知"
    recent = d["speed_ms"][-20:]
    unit = d["per_step"][-1] if d["per_step"] else "micro-step"
    sec_per_step = np.median(recent) / 1000.0 * (accum if unit.startswith("micro") else 1)
    remain = max(0, d["total"] - d["steps"][-1])
    secs = remain * sec_per_step
    return sec_per_step, f"{secs / 3600:.1f} 小时（{sec_per_step:.2f} s/步）"


def plot(d: dict, out_png: Path, accum: int, title: str) -> tuple[float, str]:
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 12))
    ax = axes[0]
    ax.plot(d["steps"], d["loss"], color=C_LOSS, lw=0.7, alpha=0.35, label="逐步损失")
    ax.plot(d["steps"], rolling(d["loss"], 15), color=C_SMOOTH, lw=2.0, label="滑动平均 (15 点)")
    if len(d["human"]):
        ax.plot(d["steps"], rolling(d["human"], 15), color="#993C1D", lw=1.2, ls="--", alpha=0.8, label="人体区加权")
        ax.plot(d["steps"], rolling(d["bg"], 15), color="#5F5E5A", lw=1.2, ls=":", alpha=0.8, label="背景区")
    ax.set_ylabel("加权流匹配损失")
    ax.set_title(f"{title}　进度 {d['steps'][-1]}/{d['total']}（{d['steps'][-1] / max(1, d['total']) * 100:.1f}%）", fontsize=13)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_xlim(0, d["total"])

    ax = axes[1]
    ax.plot(d["steps"], d["rms"], color=C_RMS, lw=1.0, alpha=0.6)
    ax.plot(d["steps"], rolling(d["rms"], 15), color=C_RMS, lw=2.2, label="残差 RMS")
    ax.axhline(0.50, color="#A32D2D", lw=1.0, ls="--", label="计划 gate 上限 0.50")
    ax.set_ylabel("残差 RMS", color=C_RMS)
    ax.set_ylim(0, max(0.62, float(d["rms"].max()) * 1.08))
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_xlim(0, d["total"])
    if len(d["grad"]):
        ax2 = ax.twinx()
        ax2.plot(d["steps"], rolling(d["grad"], 15), color=C_GRAD, lw=1.6, label="梯度范数")
        ax2.set_ylabel("梯度范数", color=C_GRAD)
        ax2.set_ylim(0, max(0.4, float(np.percentile(d["grad"], 99)) * 1.6))
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    ax = axes[2]
    if d["val_steps"]:
        ax.plot(d["val_steps"], d["val_on"], "o-", color=C_ON, lw=2.0, ms=6, label="ON 正确姿态条件")
        ax.plot(d["val_steps"], d["val_off"], "s--", color=C_OFF, lw=1.6, ms=5, label="OFF 关闭适配器")
        ax.plot(d["val_steps"], d["val_pert"], "^:", color=C_PERT, lw=1.6, ms=5, label="Perturbed 随机扰动")
        ax.set_ylim(0, max(d["val_on"] + d["val_off"] + d["val_pert"]) * 1.15)
    else:
        ax.text(0.5, 0.5, "尚无验证记录", ha="center", va="center", transform=ax.transAxes)
    ax.set_ylabel("验证损失")
    ax.set_xlabel("优化器步")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_xlim(0, d["total"])

    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return eta_text(d, accum)


def write_monitor_md(d: dict, out_md: Path, sec_per_step: float, eta: str, accum: int, run_name: str) -> None:
    last = int(d["steps"][-1])
    tail = d["loss"][-50:]
    anomalies = []
    if len(d["grad"]):
        bad = np.where(d["grad"] > 1.0)[0]
        for i in bad[-10:]:
            anomalies.append(f"step {int(d['steps'][i])}：梯度范数 {d['grad'][i]:.3f}")
    if len(d["rms"]):
        badr = np.where(d["rms"] > 0.60)[0]
        for i in badr[-5:]:
            anomalies.append(f"step {int(d['steps'][i])}：残差 RMS {d['rms'][i]:.3f}")
    rows = ["| 验证步 | ON | OFF | 扰动 | ON 相对 OFF |", "|---:|---:|---:|---:|---:|"]
    for s, on, off, pt in zip(d["val_steps"], d["val_on"], d["val_off"], d["val_pert"]):
        rows.append(f"| {s} | {on:.4f} | {off:.4f} | {pt:.4f} | {(on - off) / off * 100:+.1f}% |")
    md = [
        f"# 训练监控简报 · {run_name}",
        "",
        f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}　自动更新：重新运行 `scripts/plot_training_dashboard.py --fetch`",
        "",
        "## 进度与吞吐",
        "",
        f"- 当前步数：**{last} / {d['total']}**（{last / max(1, d['total']) * 100:.1f}%，Epoch {d['epoch']}）",
        f"- 近期吞吐：{sec_per_step:.2f} s / 优化器步（gradient_accumulation = {accum}）",
        f"- 预计剩余：{eta}",
        f"- 训练损失（最近 50 个记录点均值）：{tail.mean():.3f}　最低 {tail.min():.3f}",
        f"- 残差 RMS（最新）：{d['rms'][-1]:.3f}　梯度范数（最近均值）：{d['grad'][-50:].mean() if len(d['grad']) else float('nan'):.3f}",
        "",
        "## 验证对比",
        "",
        *rows,
        "",
        "判读提示：ON 需持续低于 OFF 才说明适配器有效；若 Perturbed（随机扰动条件）低于 ON，"
        "说明残差对姿态信号不敏感。验证每次独立采样时间步，差异 <10% 时可能落在采样噪声内。",
        "",
    ]
    if anomalies:
        md += ["## 异常事件（梯度范数 > 1.0 或 RMS > 0.60）", ""] + [f"- {a}" for a in anomalies] + [""]
    out_md.write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=".codex_tmp/vis_pull/train_20k_v1.log")
    ap.add_argument("--out", default="可视化/00_总览与说明")
    ap.add_argument("--run-name", default="iper_v6_native_20k_v1")
    ap.add_argument("--accum", type=int, default=4, help="gradient_accumulation_steps")
    ap.add_argument("--fetch", action="store_true", help="先用 scp 从远端拉取最新日志")
    ap.add_argument("--host", default="sg")
    ap.add_argument("--remote", default="/home/shangguanrz/project/pic-edit/experiments/iper_v6_native_20k_v1_launcher.log")
    args = ap.parse_args()

    log = Path(args.log)
    if args.fetch:
        fetch_log(args.host, args.remote, log)
    if not log.exists():
        raise SystemExit(f"日志文件不存在：{log}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = parse_log(log)
    if len(d["steps"]) == 0:
        raise SystemExit(f"未在 {log} 中解析到训练步记录")

    png = out / "训练动态曲线.png"
    sec_per_step, eta = plot(d, png, args.accum, args.run_name)
    md = out / "训练监控.md"
    write_monitor_md(d, md, sec_per_step, eta, args.accum, args.run_name)
    print(f"完成：{png.resolve()}\n      {md.resolve()}\n进度 {int(d['steps'][-1])}/{d['total']}，预计剩余 {eta}")


if __name__ == "__main__":
    main()
