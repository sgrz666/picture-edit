#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np


ROOT = Path(
    "/home/shangguanrz/project/pic-edit/datasets/Hi4D_pilot_v1"
)

SRC = ROOT / "frames.jsonl"
OUT = ROOT / "frames_1500.jsonl"
SUMMARY = ROOT / "frames_1500_summary.json"

FRAMES_PER_SEQUENCE = 50


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def evenly_pick(rows, n):
    rows = sorted(
        rows,
        key=lambda x: x["frame_id"]
    )

    if len(rows) < n:
        raise RuntimeError(
            f"Sequence only has {len(rows)} frames, "
            f"but requested {n}"
        )

    if len(rows) == n:
        return rows

    idx = np.linspace(
        0,
        len(rows) - 1,
        n
    )

    idx = np.round(idx).astype(int)

    picked = []
    used = set()

    for i in idx:
        row = rows[int(i)]
        fid = row["frame_id"]

        if fid not in used:
            used.add(fid)
            picked.append(row)

    # linspace 极少数情况下可能 round 到重复位置
    if len(picked) < n:
        for row in rows:
            fid = row["frame_id"]

            if fid not in used:
                used.add(fid)
                picked.append(row)

                if len(picked) == n:
                    break

    return sorted(
        picked,
        key=lambda x: x["frame_id"]
    )


def balanced_train_pick(rows, n):
    """
    Train:
    尽量保持 contact / non-contact 约 1:1。
    如果某一类不足，则由另一类补足。
    """

    contact = [
        x for x in rows
        if bool(x["is_contact"])
    ]

    noncontact = [
        x for x in rows
        if not bool(x["is_contact"])
    ]

    # 某 sequence 本身只有一种状态
    if not contact or not noncontact:
        return evenly_pick(rows, n)

    target_c = n // 2
    target_n = n - target_c

    nc = min(
        len(contact),
        target_c
    )

    nn = min(
        len(noncontact),
        target_n
    )

    remain = n - nc - nn

    if remain > 0:
        extra_c = min(
            len(contact) - nc,
            remain
        )

        nc += extra_c
        remain -= extra_c

    if remain > 0:
        extra_n = min(
            len(noncontact) - nn,
            remain
        )

        nn += extra_n
        remain -= extra_n

    if remain != 0:
        raise RuntimeError(
            "Not enough frames to complete balanced sampling"
        )

    picked = (
        evenly_pick(contact, nc)
        +
        evenly_pick(noncontact, nn)
    )

    return sorted(
        picked,
        key=lambda x: x["frame_id"]
    )


rows = load_jsonl(SRC)

print("Source frames:", len(rows))


# ============================================================
# GROUP BY SEQUENCE
# ============================================================

groups = defaultdict(list)

for row in rows:
    key = (
        row["split"],
        row["pair"],
        row["action"],
        int(row["camera_id"]),
    )

    groups[key].append(row)


print("Source sequences:", len(groups))

if len(groups) != 30:
    raise RuntimeError(
        f"Expected 30 sequences, found {len(groups)}"
    )


# ============================================================
# SAMPLE
# ============================================================

selected = []
sequence_stats = []

for key in sorted(groups):

    split, pair, action, camera_id = key

    seq_rows = groups[key]

    if len(seq_rows) < FRAMES_PER_SEQUENCE:
        raise RuntimeError(
            f"{pair}/{action} has only "
            f"{len(seq_rows)} frames"
        )

    if split == "train":
        picked = balanced_train_pick(
            seq_rows,
            FRAMES_PER_SEQUENCE
        )

    else:
        picked = evenly_pick(
            seq_rows,
            FRAMES_PER_SEQUENCE
        )

    selected.extend(picked)

    sequence_stats.append({
        "split": split,
        "pair": pair,
        "action": action,
        "camera_id": camera_id,

        "source_frames":
            len(seq_rows),

        "source_contact_frames":
            sum(
                bool(x["is_contact"])
                for x in seq_rows
            ),

        "selected_frames":
            len(picked),

        "selected_contact_frames":
            sum(
                bool(x["is_contact"])
                for x in picked
            ),

        "selected_noncontact_frames":
            sum(
                not bool(x["is_contact"])
                for x in picked
            ),
    })


# ============================================================
# SORT
# ============================================================

split_rank = {
    "train": 0,
    "val": 1,
    "test": 2,
}

selected = sorted(
    selected,
    key=lambda x: (
        split_rank[x["split"]],
        x["pair"],
        x["action"],
        x["frame_id"],
    )
)


# ============================================================
# VALIDATE EXACT COUNTS
# ============================================================

split_counts = Counter(
    x["split"]
    for x in selected
)

expected = {
    "train": 900,
    "val": 300,
    "test": 300,
}

if dict(split_counts) != expected:
    raise RuntimeError(
        f"Unexpected split counts: "
        f"{dict(split_counts)}, "
        f"expected={expected}"
    )

if len(selected) != 1500:
    raise RuntimeError(
        f"Expected 1500 frames, got {len(selected)}"
    )

# 防止同一帧重复
unique_keys = {
    (
        x["pair"],
        x["action"],
        x["frame_id"],
        x["camera_id"],
    )
    for x in selected
}

if len(unique_keys) != len(selected):
    raise RuntimeError(
        "Duplicate frames detected"
    )


# ============================================================
# WRITE MANIFEST
# ============================================================

with open(
    OUT,
    "w",
    encoding="utf-8"
) as f:

    for row in selected:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False
            )
            + "\n"
        )


summary = {
    "source_manifest": str(SRC),
    "output_manifest": str(OUT),

    "strategy": (
        "keep all 10 identity-disjoint pairs and all "
        "30 selected interaction sequences; "
        "sample exactly 50 frames per sequence"
    ),

    "frames_per_sequence":
        FRAMES_PER_SEQUENCE,

    "total_sequences":
        len(groups),

    "total_frames":
        len(selected),

    "split_frames":
        dict(split_counts),

    "split_contact_frames": {
        split: sum(
            x["split"] == split
            and bool(x["is_contact"])
            for x in selected
        )
        for split in [
            "train",
            "val",
            "test",
        ]
    },

    "split_noncontact_frames": {
        split: sum(
            x["split"] == split
            and not bool(x["is_contact"])
            for x in selected
        )
        for split in [
            "train",
            "val",
            "test",
        ]
    },

    "sequences":
        sequence_stats,
}


with open(
    SUMMARY,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False
    )


print()
print("======================================")
print("Hi4D 1500 MANIFEST READY")
print("======================================")

print(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False
    )
)
