#!/usr/bin/env python
"""统一入口：把本实验的各阶段收敛成三条可复用命令。

    python run.py official --mode formal
    python run.py matrix
    python run.py report --run <matrix 运行名>

其余子命令是同一批脚本的别名，便于分步排查：freeze / audit / gate / smoke。
所有重产物写在 /data 下的 WORK/runs/<时间戳>，不覆盖既有运行。
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent


def call(script: str, extra: list[str]) -> int:
    cmd = [sys.executable, str(HERE / script), *extra]
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("official", help="官方样例复现（Pose/Depth/Inpaint）")
    p.add_argument("--mode", default="formal", choices=["smoke", "formal"])
    p.add_argument("--tasks", default="pose,depth,inpaint")
    p.add_argument("--scales", default="0.0,1.0")
    p.add_argument("--name", default=None)

    p = sub.add_parser("matrix", help="六组 × 四配置 × 两 seed 批量比较")
    p.add_argument("--name", default=None)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--only", default=None)
    p.add_argument("--configs", default="native,union_s0,union_s05,union_s10")
    p.add_argument("--seeds", default="42,43")

    p = sub.add_parser("report", help="指标与面板生成")
    p.add_argument("--run", required=True)
    p.add_argument("--panels", action="store_true", default=True)

    sub.add_parser("freeze", help="冻结六组输入并生成 SHA256 清单")
    sub.add_parser("audit", help="180 个控制权重键加载审计")
    p = sub.add_parser("gate", help="接入契约门禁")
    p.add_argument("--canvas", default="512,512")
    p.add_argument("--steps", type=int, default=1)

    args = ap.parse_args()
    if args.cmd == "official":
        return call("04_official_samples.py", ["--mode", args.mode, "--tasks", args.tasks,
                                              "--scales", args.scales] + (["--name", args.name] if args.name else []))
    if args.cmd == "matrix":
        return call("07_run_matrix.py", ["--steps", str(args.steps), "--configs", args.configs,
                                         "--seeds", args.seeds]
                    + (["--name", args.name] if args.name else [])
                    + (["--only", args.only] if args.only else []))
    if args.cmd == "report":
        rc = call("08_evaluate.py", ["--run", args.run, "--panels"])
        if rc:
            return rc
        return call("15_report.py", ["--run", args.run])
    if args.cmd == "freeze":
        return call("05_freeze_inputs.py", [])
    if args.cmd == "audit":
        return call("03_weight_audit.py", [])
    if args.cmd == "gate":
        return call("06_contract_gate.py", ["--canvas", args.canvas, "--steps", str(args.steps)])
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
