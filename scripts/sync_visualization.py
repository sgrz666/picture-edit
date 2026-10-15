# -*- coding: utf-8 -*-
"""一键同步远端实验产物到本地「可视化」文件夹（拉取 → 整理 → 绘图 → 简报）。

流程：
  1. scp 拉取远端 showcase / eval_results / eval_metrics.json / 训练日志 到暂存目录
  2. 调用 build_visualization_folder.py 生成六栏条件图、演化条带、四联图与指标汇总
  3. 调用 plot_training_dashboard.py 生成训练动态曲线与监控简报

用法：
  python scripts/sync_visualization.py                 # 使用默认路径与实验名
  python scripts/sync_visualization.py --hours-only    # 只刷新训练曲线，不重拉图像
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_PYTHON = r"C:\Users\sg\.workbuddy\binaries\python\envs\default\Scripts\python.exe"


def run(cmd: list[str], desc: str) -> bool:
    print(f"[run] {desc}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[warn] {desc} 失败（returncode={r.returncode}）")
        if r.stderr.strip():
            print(f"       {r.stderr.strip()[:400]}")
        return False
    if r.stdout.strip():
        print("       " + r.stdout.strip().replace("\n", "\n       "))
    return True


def scp(remote: str, local: str, recursive: bool = False) -> bool:
    cmd = ["scp", "-q"] + (["-r"] if recursive else []) + [remote, local]
    return run(cmd, f"拉取 {remote.split('/')[-1]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="sg")
    ap.add_argument("--root", default="/home/shangguanrz/project/pic-edit")
    ap.add_argument("--assets-exp", default="iper_v6_comparative_3000", help="图像资产所在实验目录")
    ap.add_argument("--train-exp", default="iper_v6_native_20k_v1", help="训练日志所在实验目录")
    ap.add_argument("--staging", default=".codex_tmp/vis_pull")
    ap.add_argument("--out", default="可视化")
    ap.add_argument("--python", default=DEFAULT_PYTHON)
    ap.add_argument("--log-only", action="store_true", help="只刷新训练曲线，不重拉图像资产")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    root = Path.cwd()
    staging = root / args.staging
    staging.mkdir(parents=True, exist_ok=True)

    base = f"{args.host}:{args.root}/experiments"
    ok = True
    if not args.log_only:
        ok &= scp(f"{base}/{args.assets_exp}/multistage_evolution_showcase", str(staging / "multistage_evolution_showcase"), True)
        ok &= scp(f"{base}/{args.assets_exp}/eval_results/images", str(staging / "eval_images"), True)
        ok &= scp(f"{base}/{args.assets_exp}/eval_results/eval_metrics.json", str(staging / "eval_metrics.json"))
    ok &= scp(f"{base}/{args.train_exp}_launcher.log", str(staging / "train_20k_v1.log"))

    out = str(root / args.out)
    if not args.log_only:
        ok &= run([args.python, str(here / "build_visualization_folder.py"), "--src", str(staging), "--out", out], "整理图像 → 可视化/")
        ok &= run([args.python, str(here / "build_edit_effect_strips.py"), "--src", str(staging), "--out", out, "--stack"], "合成三排编辑效果图 → 04_编辑效果条带/")
        ok &= run([args.python, str(here / "build_eval_template_panels.py"), "--src", str(staging), "--out", out], "四联图扩充为模板图 → 03/模板版_带条件控制/")
    ok &= run(
        [args.python, str(here / "plot_training_dashboard.py"), "--log", str(staging / "train_20k_v1.log"),
         "--out", str(root / args.out / "00_总览与说明"), "--run-name", args.train_exp],
        "绘制训练曲线与监控简报",
    )

    print("\n[done] 同步完成" if ok else "\n[done] 同步完成（含告警，见上方 [warn]）")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
