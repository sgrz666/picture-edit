#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np


ROOT = Path(
    "/home/shangguanrz/project/pic-edit/datasets/Hi4D_pilot_v1"
)

SRC = ROOT / "processed_frames_1500.jsonl"

OUT_ALL = ROOT / "pairs_1500.jsonl"
OUT_TRAIN = ROOT / "pairs_1500_train.jsonl"
OUT_VAL = ROOT / "pairs_1500_val.jsonl"
OUT_TEST = ROOT / "pairs_1500_test.jsonl"

SUMMARY = ROOT / "pairs_1500_summary.json"


# ============================================================
# CONFIG
# ============================================================

SOURCE_ANCHORS = {
    "train": 8,
    "val": 4,
    "test": 4,
}

PAIRS_PER_TARGET = {
    "train": 6,
    "val": 2,
    "test": 2,
}

# 避免直接用相邻帧
MIN_FRAME_GAP = 2


# ============================================================
# IO
# ============================================================

def load_jsonl(path):
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


# ============================================================
# SOURCE ANCHORS
# ============================================================

def evenly_select(rows, n):
    rows = sorted(
        rows,
        key=lambda x: x["frame_id"]
    )

    if len(rows) <= n:
        return rows

    ids = np.linspace(
        0,
        len(rows) - 1,
        n
    )

    ids = np.round(ids).astype(int)

    out = []
    used = set()

    for idx in ids:
        row = rows[int(idx)]

        if row["frame_id"] not in used:
            used.add(row["frame_id"])
            out.append(row)

    if len(out) < n:
        for row in rows:
            if row["frame_id"] not in used:
                used.add(row["frame_id"])
                out.append(row)

            if len(out) == n:
                break

    return out


# ============================================================
# POSE DISTANCE
# ============================================================

def load_pose(row):
    a = np.load(
        row["pose_A_npy"]
    )

    b = np.load(
        row["pose_B_npy"]
    )

    return a, b


def pose_distance(
    source_pose,
    target_pose
):
    """
    两个人的共同可见 SMPL joints 的平均 2D pixel displacement。

    保留整体位移：
    双人 interaction 中相对位置变化本身就是我们想学习的 geometry。
    """

    distances = []

    for src, tgt in zip(
        source_pose,
        target_pose
    ):

        valid = (
            (src[:, 3] > 0.5)
            &
            (tgt[:, 3] > 0.5)
        )

        if valid.sum() == 0:
            continue

        d = np.linalg.norm(
            src[valid, :2]
            -
            tgt[valid, :2],
            axis=1
        )

        distances.extend(
            d.tolist()
        )

    if not distances:
        return -1.0

    return float(
        np.mean(distances)
    )


# ============================================================
# MANIFEST RECORD
# ============================================================

def make_pair(
    split,
    seq_id,
    source,
    target,
    pose_dist
):
    return {
        "split": split,

        "sequence_id": seq_id,

        "pair": target["pair"],
        "action": target["action"],
        "camera_id": target["camera_id"],

        # ----------------------------------------
        # source identity / appearance
        # ----------------------------------------

        "source_frame_id":
            source["frame_id"],

        "source_rgb":
            source["processed_rgb"],

        "source_person_mask_A":
            source["person_mask_A_512"],

        "source_person_mask_B":
            source["person_mask_B_512"],

        "source_person_id":
            source["person_id"],

        # ----------------------------------------
        # target GT
        # ----------------------------------------

        "target_frame_id":
            target["frame_id"],

        "target_rgb":
            target["processed_rgb"],

        # ----------------------------------------
        # target Pose
        # ----------------------------------------

        "target_pose_A":
            target["pose_A"],

        "target_pose_B":
            target["pose_B"],

        "target_pose_combined":
            target["pose_combined"],

        "target_pose_A_npy":
            target["pose_A_npy"],

        "target_pose_B_npy":
            target["pose_B_npy"],

        # ----------------------------------------
        # target Depth
        # ----------------------------------------

        "target_depth_A":
            target["depth_A"],

        "target_depth_B":
            target["depth_B"],

        "target_scene_depth":
            target["scene_depth"],

        # ----------------------------------------
        # target Normal
        # ----------------------------------------

        "target_normal_A":
            target["normal_A"],

        "target_normal_B":
            target["normal_B"],

        "target_scene_normal":
            target["scene_normal"],

        # ----------------------------------------
        # identity / geometry
        # ----------------------------------------

        "target_person_mask_A":
            target["person_mask_A_512"],

        "target_person_mask_B":
            target["person_mask_B_512"],

        "target_person_id":
            target["person_id"],

        "target_scene_person_id":
            target["scene_person_id"],

        # ----------------------------------------
        # occlusion
        # ----------------------------------------

        "target_occluded_A_by_B":
            target["occluded_A_by_B"],

        "target_occluded_B_by_A":
            target["occluded_B_by_A"],

        "target_occlusion":
            target["occlusion"],

        # ----------------------------------------
        # interaction metadata
        # ----------------------------------------

        "target_is_contact":
            target["is_contact"],

        # ----------------------------------------
        # pairing QC
        # ----------------------------------------

        "frame_gap": abs(
            int(source["frame_id"])
            -
            int(target["frame_id"])
        ),

        "pose_distance_px":
            float(pose_dist),

        "resolution": 512,

        "depth_unit": "meter",

        "normal_space": "camera",
    }


# ============================================================
# MAIN
# ============================================================

rows = load_jsonl(SRC)

print("frames:", len(rows))

if len(rows) != 1500:
    raise RuntimeError(
        f"expected 1500 processed frames, got {len(rows)}"
    )


# group by exact sequence
groups = defaultdict(list)

for row in rows:
    key = (
        row["split"],
        row["pair"],
        row["action"],
        int(row["camera_id"]),
    )

    groups[key].append(row)


print("sequences:", len(groups))

if len(groups) != 30:
    raise RuntimeError(
        f"expected 30 sequences, got {len(groups)}"
    )


all_pairs = []

sequence_summary = []


for key in sorted(groups):

    split, pair_name, action, cam = key

    seq_rows = sorted(
        groups[key],
        key=lambda x: x["frame_id"]
    )

    n_anchor = SOURCE_ANCHORS[split]

    anchors = evenly_select(
        seq_rows,
        n_anchor
    )

    # cache pose arrays
    pose_cache = {}

    def get_pose(row):
        fid = row["frame_id"]

        if fid not in pose_cache:
            pose_cache[fid] = load_pose(row)

        return pose_cache[fid]

    seq_pairs = []

    seq_id = (
        f"{pair_name}/{action}/cam{cam}"
    )

    for target in seq_rows:

        target_pose = get_pose(
            target
        )

        candidates = []

        # ----------------------------------------
        # preferred candidates:
        # distinct and not immediate neighbour
        # ----------------------------------------

        for source in anchors:

            if (
                source["frame_id"]
                ==
                target["frame_id"]
            ):
                continue

            gap = abs(
                int(source["frame_id"])
                -
                int(target["frame_id"])
            )

            if gap < MIN_FRAME_GAP:
                continue

            d = pose_distance(
                get_pose(source),
                target_pose
            )

            candidates.append(
                (
                    d,
                    gap,
                    source
                )
            )

        # ----------------------------------------
        # fallback:
        # 如果因为 target 正好靠近多个 anchor 导致数量不足，
        # 只排除 self reconstruction
        # ----------------------------------------

        if (
            len(candidates)
            <
            PAIRS_PER_TARGET[split]
        ):

            existing = {
                x[2]["frame_id"]
                for x in candidates
            }

            for source in anchors:

                if (
                    source["frame_id"]
                    ==
                    target["frame_id"]
                ):
                    continue

                if (
                    source["frame_id"]
                    in existing
                ):
                    continue

                gap = abs(
                    int(source["frame_id"])
                    -
                    int(target["frame_id"])
                )

                d = pose_distance(
                    get_pose(source),
                    target_pose
                )

                candidates.append(
                    (
                        d,
                        gap,
                        source
                    )
                )

        # Pose 差越大越优先
        candidates.sort(
            key=lambda x: (
                x[0],
                x[1]
            ),
            reverse=True
        )

        k = PAIRS_PER_TARGET[
            split
        ]

        selected_sources = (
            candidates[:k]
        )

        if len(selected_sources) != k:
            raise RuntimeError(
                f"{seq_id} target "
                f"{target['frame_id']} only has "
                f"{len(selected_sources)} sources"
            )

        for d, gap, source in selected_sources:

            pair_row = make_pair(
                split,
                seq_id,
                source,
                target,
                d
            )

            seq_pairs.append(
                pair_row
            )

            all_pairs.append(
                pair_row
            )

    sequence_summary.append({
        "split": split,
        "pair": pair_name,
        "action": action,
        "camera_id": cam,

        "targets":
            len(seq_rows),

        "source_anchor_count":
            len(anchors),

        "source_anchor_frames": [
            x["frame_id"]
            for x in anchors
        ],

        "pairs":
            len(seq_pairs),

        "mean_pose_distance_px":
            float(
                np.mean(
                    [
                        x["pose_distance_px"]
                        for x in seq_pairs
                    ]
                )
            ),

        "min_pose_distance_px":
            float(
                np.min(
                    [
                        x["pose_distance_px"]
                        for x in seq_pairs
                    ]
                )
            ),

        "max_pose_distance_px":
            float(
                np.max(
                    [
                        x["pose_distance_px"]
                        for x in seq_pairs
                    ]
                )
            ),
    })


# ============================================================
# VALIDATION
# ============================================================

split_counts = Counter(
    x["split"]
    for x in all_pairs
)

expected = {
    "train": 5400,
    "val": 600,
    "test": 600,
}

if dict(split_counts) != expected:
    raise RuntimeError(
        f"bad pair counts: "
        f"{dict(split_counts)}, expected {expected}"
    )


# absolutely no self reconstruction
self_pairs = [
    x
    for x in all_pairs
    if (
        x["source_frame_id"]
        ==
        x["target_frame_id"]
    )
]

if self_pairs:
    raise RuntimeError(
        f"found {len(self_pairs)} self pairs"
    )


# ============================================================
# WRITE
# ============================================================

write_jsonl(
    OUT_ALL,
    all_pairs
)

for split, path in [
    ("train", OUT_TRAIN),
    ("val", OUT_VAL),
    ("test", OUT_TEST),
]:

    write_jsonl(
        path,
        [
            x
            for x in all_pairs
            if x["split"] == split
        ]
    )


pose_distances = np.asarray(
    [
        x["pose_distance_px"]
        for x in all_pairs
    ],
    dtype=np.float32
)

frame_gaps = np.asarray(
    [
        x["frame_gap"]
        for x in all_pairs
    ],
    dtype=np.float32
)


summary = {
    "input_frames": 1500,

    "total_pairs":
        len(all_pairs),

    "split_pairs":
        dict(split_counts),

    "train_sources_per_target":
        PAIRS_PER_TARGET["train"],

    "val_sources_per_target":
        PAIRS_PER_TARGET["val"],

    "test_sources_per_target":
        PAIRS_PER_TARGET["test"],

    "self_pairs": 0,

    "pose_distance_px": {
        "mean":
            float(
                pose_distances.mean()
            ),

        "median":
            float(
                np.median(
                    pose_distances
                )
            ),

        "min":
            float(
                pose_distances.min()
            ),

        "max":
            float(
                pose_distances.max()
            ),
    },

    "frame_gap": {
        "mean":
            float(
                frame_gaps.mean()
            ),

        "median":
            float(
                np.median(
                    frame_gaps
                )
            ),

        "min":
            int(
                frame_gaps.min()
            ),

        "max":
            int(
                frame_gaps.max()
            ),
    },

    "sequence_summary":
        sequence_summary,
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
print("Hi4D SOURCE-TARGET PAIRS READY")
print("======================================")

print(
    json.dumps(
        {
            "total_pairs":
                summary["total_pairs"],

            "split_pairs":
                summary["split_pairs"],

            "self_pairs":
                summary["self_pairs"],

            "pose_distance_px":
                summary[
                    "pose_distance_px"
                ],

            "frame_gap":
                summary[
                    "frame_gap"
                ],
        },
        indent=2,
        ensure_ascii=False
    )
)
