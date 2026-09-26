#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from collections import defaultdict

import numpy as np


ROOT = Path(
    "/home/shangguanrz/project/pic-edit/datasets/Hi4D_pilot_v1"
)

IN_MANIFEST = (
    ROOT
    / "native_interaction_v1"
    / "targets_contact_native.jsonl"
)

OUT_MANIFEST = (
    ROOT
    / "native_interaction_v1"
    / "targets_contact_native_with_strength.jsonl"
)

OUT_SUMMARY = (
    ROOT
    / "native_interaction_v1"
    / "contact_strength_summary.json"
)


def read_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
                + "\n"
            )


def stats(x):

    x = np.asarray(
        x,
        dtype=np.float64
    )

    if len(x) == 0:
        return {}

    qs = [
        0,
        1,
        5,
        10,
        25,
        50,
        75,
        90,
        95,
        99,
        100,
    ]

    return {
        "count":
            int(len(x)),

        "mean":
            float(x.mean()),

        "percentiles": {
            f"P{q:02d}":
                float(
                    np.percentile(
                        x,
                        q
                    )
                )
            for q in qs
        },
    }


rows = read_jsonl(
    IN_MANIFEST
)

print(
    "contact targets:",
    len(rows)
)

if len(rows) != 1458:
    print(
        "WARNING: expected 1458, got",
        len(rows)
    )


out_rows = []

by_sequence = defaultdict(list)
by_action = defaultdict(list)
by_split = defaultdict(list)


for i, row in enumerate(
    rows,
    start=1
):

    smpl_path = row[
        "official_smpl"
    ]

    with np.load(
        smpl_path,
        allow_pickle=True
    ) as d:

        contact = np.asarray(
            d["contact"]
        )

    if contact.shape != (
        2,
        6890
    ):
        raise RuntimeError(
            f"bad contact shape "
            f"{contact.shape}: "
            f"{smpl_path}"
        )

    mask_a = (
        contact[0] > 0
    )

    mask_b = (
        contact[1] > 0
    )

    count_a = int(
        mask_a.sum()
    )

    count_b = int(
        mask_b.sum()
    )

    total = (
        count_a
        + count_b
    )

    min_side = min(
        count_a,
        count_b
    )

    max_side = max(
        count_a,
        count_b
    )

    # 不做任何自创阈值分类。
    # 先保存纯官方统计值。
    r = dict(row)

    r.update({
        "contact_vertex_count_A":
            count_a,

        "contact_vertex_count_B":
            count_b,

        "contact_vertex_count_total":
            total,

        "contact_vertex_count_min_side":
            min_side,

        "contact_vertex_count_max_side":
            max_side,

        "contact_vertex_ratio_A":
            float(
                count_a / 6890.0
            ),

        "contact_vertex_ratio_B":
            float(
                count_b / 6890.0
            ),

        "contact_annotation":
            "Hi4D official SMPL contact > 0",
    })

    out_rows.append(r)

    seq_key = (
        row["split"],
        row["pair"],
        row["action"],
        int(row["camera_id"]),
    )

    by_sequence[
        seq_key
    ].append(total)

    by_action[
        row["action"]
    ].append(total)

    by_split[
        row["split"]
    ].append(total)

    if (
        i == 1
        or i % 250 == 0
        or i == len(rows)
    ):
        print(
            f"{i}/{len(rows)}"
        )


write_jsonl(
    OUT_MANIFEST,
    out_rows
)


totals = [
    x[
        "contact_vertex_count_total"
    ]
    for x in out_rows
]

side_min = [
    x[
        "contact_vertex_count_min_side"
    ]
    for x in out_rows
]

summary = {
    "total_contact_targets":
        len(out_rows),

    "definition":
        (
            "Hi4D official SMPL vertex is "
            "contact iff contact[p, v] > 0"
        ),

    "total_contact_vertices":
        stats(totals),

    "min_side_contact_vertices":
        stats(side_min),

    "threshold_counts": {},

    "split_stats": {},

    "sequence_stats": [],
}


# ------------------------------------------------------------
# 只做分布观察，不据此删数据
# ------------------------------------------------------------

for th in [
    2,
    5,
    10,
    20,
    50,
    100,
    200,
    500,
    1000,
]:

    n = sum(
        x < th
        for x in totals
    )

    summary[
        "threshold_counts"
    ][f"total_lt_{th}"] = {
        "count":
            int(n),

        "ratio":
            float(
                n / len(totals)
            ),
    }


for split, vals in sorted(
    by_split.items()
):

    summary[
        "split_stats"
    ][split] = stats(vals)


for key, vals in sorted(
    by_sequence.items()
):

    split, pair, action, cam = key

    s = stats(vals)

    summary[
        "sequence_stats"
    ].append({
        "split":
            split,

        "pair":
            pair,

        "action":
            action,

        "camera_id":
            cam,

        "contact_frames":
            len(vals),

        "contact_strength":
            s,
    })


with open(
    OUT_SUMMARY,
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
print(
    "===================================="
)

print(
    "CONTACT STRENGTH SUMMARY"
)

print(
    "===================================="
)

print(
    json.dumps(
        {
            "total_contact_targets":
                summary[
                    "total_contact_targets"
                ],

            "total_contact_vertices":
                summary[
                    "total_contact_vertices"
                ],

            "min_side_contact_vertices":
                summary[
                    "min_side_contact_vertices"
                ],

            "threshold_counts":
                summary[
                    "threshold_counts"
                ],
        },
        indent=2,
        ensure_ascii=False
    )
)

print()
print(
    "manifest:",
    OUT_MANIFEST
)

print(
    "summary:",
    OUT_SUMMARY
)
