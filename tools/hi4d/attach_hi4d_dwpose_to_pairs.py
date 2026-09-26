#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path


ROOT = Path(
    "/home/shangguanrz/project/pic-edit/datasets/Hi4D_pilot_v1"
)

BASE = ROOT / "native_interaction_v2"

FRAMES = BASE / "frames_used_with_dwpose_v2.jsonl"


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                ) + "\n"
            )


def fkey(pair, action, camera_id, frame_id):
    return (
        str(pair),
        str(action),
        int(camera_id),
        int(frame_id),
    )


frames = read_jsonl(FRAMES)

print("DWPose frame records:", len(frames))

frame_map = {}

for r in frames:

    status = r.get(
        "dwpose_final_status",
        ""
    )

    if status == "still_incomplete":
        raise RuntimeError(
            "Found incomplete DWPose: "
            f"{r['pair']} "
            f"{r['action']} "
            f"{r['frame_id']}"
        )

    if (
        "dwpose_A" not in r
        or "dwpose_B" not in r
    ):
        raise RuntimeError(
            f"DWPose paths missing: {r}"
        )

    k = fkey(
        r["pair"],
        r["action"],
        r["camera_id"],
        r["frame_id"],
    )

    frame_map[k] = r


pair_files = [
    "pairs_all.jsonl",
    "pairs_train.jsonl",
    "pairs_val.jsonl",
    "pairs_test.jsonl",
    "pairs_train_N2C_main.jsonl",
    "pairs_train_C2C_aux.jsonl",
]


for name in pair_files:

    src = BASE / name

    if not src.exists():
        print("SKIP missing:", src)
        continue

    rows = read_jsonl(src)

    out = []

    missing = []

    for p in rows:

        ks = fkey(
            p["pair"],
            p["action"],
            p["camera_id"],
            p["source_frame_id"],
        )

        kt = fkey(
            p["pair"],
            p["action"],
            p["camera_id"],
            p["target_frame_id"],
        )

        source = frame_map.get(ks)
        target = frame_map.get(kt)

        if source is None:
            missing.append(
                ("source", ks)
            )
            continue

        if target is None:
            missing.append(
                ("target", kt)
            )
            continue

        q = dict(p)

        q.update({
            # -----------------------------
            # source DWPose
            # -----------------------------
            "source_dwpose_A":
                source["dwpose_A"],

            "source_dwpose_B":
                source["dwpose_B"],

            # -----------------------------
            # target DWPose
            # -----------------------------
            "target_dwpose_A":
                target["dwpose_A"],

            "target_dwpose_B":
                target["dwpose_B"],

            # -----------------------------
            # representation
            # -----------------------------
            "dwpose_coordinate_system":
                "native_and_normalized",

            "dwpose_native_width":
                940,

            "dwpose_native_height":
                1280,

            "pre_crop":
                False,

            "pre_resize":
                False,
        })

        out.append(q)

    if missing:
        print(
            f"{name}: missing={len(missing)}"
        )

        for x in missing[:20]:
            print(" ", x)

        raise RuntimeError(
            f"{name}: DWPose binding incomplete"
        )

    dst = BASE / name.replace(
        ".jsonl",
        "_with_dwpose.jsonl"
    )

    write_jsonl(
        dst,
        out
    )

    print(
        f"{name}: {len(rows)} -> {len(out)} OK"
    )


print()
print("ALL DWPose pair binding OK")
