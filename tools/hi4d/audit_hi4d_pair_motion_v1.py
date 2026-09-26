#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
import numpy as np

ROOT = Path(
    "/home/shangguanrz/project/pic-edit/"
    "datasets/Hi4D_pilot_v1/native_interaction_v2"
)

PAIR_FILE = ROOT / "pairs_all_with_dwpose.jsonl"


def load_pose(path):
    with np.load(path, allow_pickle=True) as d:
        xy = np.asarray(d["xy_normalized"], dtype=np.float32)
        sc = np.asarray(d["scores"], dtype=np.float32)
    return xy, sc


def person_motion(src_path, tgt_path, thr=0.30):
    sxy, ss = load_pose(src_path)
    txy, ts = load_pose(tgt_path)

    n = min(len(sxy), len(txy))
    sxy = sxy[:n]
    txy = txy[:n]
    ss = ss[:n]
    ts = ts[:n]

    valid = (
        np.isfinite(sxy).all(axis=1)
        & np.isfinite(txy).all(axis=1)
        & (ss >= thr)
        & (ts >= thr)
    )

    if valid.sum() < 4:
        return np.nan

    d = np.linalg.norm(
        txy[valid] - sxy[valid],
        axis=1,
    )

    return float(d.mean())


rows = []

with open(PAIR_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))


results = []

for i, r in enumerate(rows, 1):

    da = person_motion(
        r["source_dwpose_A"],
        r["target_dwpose_A"],
    )

    db = person_motion(
        r["source_dwpose_B"],
        r["target_dwpose_B"],
    )

    vals = [
        x for x in [da, db]
        if np.isfinite(x)
    ]

    motion = (
        float(np.mean(vals))
        if vals
        else np.nan
    )

    results.append({
        "pair_type": r["pair_type"],
        "frame_gap": int(r["frame_gap"]),
        "pose_motion": motion,
    })

    if i % 500 == 0:
        print(f"processed {i}/{len(rows)}")


print()
print("=" * 70)
print("Hi4D PAIR MOTION AUDIT")
print("=" * 70)
print("pairs:", len(results))


def report(name, subset):

    gaps = np.asarray(
        [x["frame_gap"] for x in subset],
        dtype=np.float32,
    )

    motions = np.asarray(
        [
            x["pose_motion"]
            for x in subset
            if np.isfinite(x["pose_motion"])
        ],
        dtype=np.float32,
    )

    print()
    print("----", name, "----")
    print("N =", len(subset))

    if len(gaps):
        print(
            "frame_gap p10/p25/p50/p75/p90:",
            np.percentile(
                gaps,
                [10, 25, 50, 75, 90]
            ),
        )

        for t in [1, 3, 5, 10, 20]:
            n = int((gaps <= t).sum())
            print(
                f"gap <= {t:2d}: "
                f"{n:4d} "
                f"({100*n/len(gaps):5.1f}%)"
            )

    if len(motions):
        print(
            "pose_motion p10/p25/p50/p75/p90:",
            np.round(
                np.percentile(
                    motions,
                    [10, 25, 50, 75, 90]
                ),
                4,
            ),
        )

        for t in [0.01, 0.02, 0.03, 0.05, 0.08]:
            n = int((motions <= t).sum())
            print(
                f"motion <= {t:.2f}: "
                f"{n:4d} "
                f"({100*n/len(motions):5.1f}%)"
            )


report("ALL", results)

for kind in sorted(
    set(x["pair_type"] for x in results)
):
    report(
        kind,
        [
            x for x in results
            if x["pair_type"] == kind
        ],
    )
