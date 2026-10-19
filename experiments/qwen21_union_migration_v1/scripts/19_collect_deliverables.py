#!/usr/bin/env python
"""交付物归位：把一次矩阵运行的 48 张结果、面板与指标从 $WORK 复制进实验目录，让实验自包含。

只复制，不移动、不删除 $WORK 里的原件；目标目录已存在的同名文件跳过并计数（不覆盖既有产物）。
复制完写 `output/deliverables_index.json`：每张图的来源路径、SHA256、样本/配置/seed/画布/耗时/峰值，
以及指标与清单文件的指纹——别人只靠实验目录就能核对与复现。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil

WORK = pathlib.Path("/data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1")
EXP = pathlib.Path(__file__).resolve().parent.parent


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def copy(src: pathlib.Path, dst: pathlib.Path, counter: dict, refresh: bool = False) -> pathlib.Path:
    """图片永不覆盖：同名不同内容就改名落地。

    refresh=True 用于 records/metrics 这类「同一次运行的派生视图」——它们本来就该反映最新一次
    评估的结果，重跑后必须替换，否则实验目录里会留下一份过期指标。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and sha256(src) != sha256(dst):
        if not refresh:
            dst = dst.with_name(dst.stem + ".conflict" + dst.suffix)
            counter["renamed_conflicts"] += 1
        else:
            counter["refreshed"] = counter.get("refreshed", 0) + 1
    elif dst.exists():
        counter["already_identical"] += 1
        return dst
    else:
        counter["copied"] += 1
    shutil.copy2(src, dst)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = WORK / "runs" / args.run
    counter = {"copied": 0, "already_identical": 0, "renamed_conflicts": 0}

    recs = [json.loads(l) for l in (run / "records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    images = []
    for r in recs:
        if not r.get("out"):
            continue
        src = pathlib.Path(r["out"])
        if not src.exists():
            continue
        dst = copy(src, EXP / "output" / "images" / src.name, counter)
        images.append({"sample": r["sample"], "kind": r["kind"], "config": r["config"], "seed": r["seed"],
                       "canvas_hw": r.get("canvas_hw"), "seconds": r.get("seconds"),
                       "peak_GiB": r.get("peak_GiB"), "path": str(dst.relative_to(EXP)),
                       "origin": str(src), "sha256": sha256(dst)})

    extras = []
    for sub in ("panels",):
        d = run / sub
        if d.is_dir():
            for p in sorted(d.glob("*.png")):
                dst = copy(p, EXP / "output" / sub / p.name, counter)
                extras.append({"path": str(dst.relative_to(EXP)), "sha256": sha256(dst), "origin": str(p)})
    for name in ("records.jsonl", "summary.json", "metrics.jsonl", "metrics_by_sample_config.csv",
                 "weight_audit.json"):
        p = run / name
        if p.exists():
            dst = copy(p, EXP / "output" / name, counter, refresh=True)
            extras.append({"path": str(dst.relative_to(EXP)), "sha256": sha256(dst), "origin": str(p)})
            stale = dst.with_name(dst.stem + ".conflict" + dst.suffix)
            if stale.exists():
                stale.unlink()      # 上一版派生文件留下的 .conflict 副本已被新版取代
                counter["stale_conflicts_removed"] = counter.get("stale_conflicts_removed", 0) + 1
    for name in ("METRICS.md",):
        p = EXP / "report" / name
        if p.exists():
            extras.append({"path": f"report/{name}", "sha256": sha256(p), "origin": str(p)})

    index = {"run": str(run), "images": len(images), "panels": len([e for e in extras if "panels/" in e["path"]]),
             "counters": counter, "images_list": images, "other": extras,
             "expected_total": 48, "complete": len(images) == 48}
    out = EXP / "output" / "deliverables_index.json"
    out.write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"run": str(run), "images": len(images), "panels": index["panels"],
                      "counters": counter, "complete": index["complete"], "index": str(out)},
                     ensure_ascii=False))
    return 0 if index["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
