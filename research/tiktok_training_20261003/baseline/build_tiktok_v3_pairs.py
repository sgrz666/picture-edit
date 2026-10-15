#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np


ROOT = Path("/home/shangguanrz/project/pic-edit")

SPLIT_ROOT = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/manifests/splits"
)

OUT_ROOT = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/manifests/pairs"
)

SUMMARY_PATH = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/manifests/"
      "pair_summary.json"
)

MAX_SOURCES_PER_SEQUENCE = 4
PAIRS_PER_TARGET = 2

BODY_SCORE_THR = 0.30


def read_jsonl(path):
    return [
        json.loads(x)
        for x in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]


def load_dwpose(row):
    p = Path(
        row["dwpose_json"]
    )

    obj = json.loads(
        p.read_text(
            encoding="utf-8"
        )
    )

    kp = np.asarray(
        obj["keypoints_norm"],
        dtype=np.float32,
    )[:18]

    scores = np.asarray(
        obj["scores"],
        dtype=np.float32,
    )[:18]

    return kp, scores


def pose_descriptor(row):
    """
    Body pose representation:
    - use DWPose 18 body joints
    - use score > 0.3 joints
    - translation-normalize
    - scale-normalize

    This is only for relative pose diversity,
    not an evaluation metric.
    """

    kp, score = load_dwpose(row)

    valid = (
        np.isfinite(kp).all(axis=1)
        & (score > BODY_SCORE_THR)
    )

    desc = np.zeros(
        (18, 2),
        dtype=np.float32,
    )

    if valid.sum() < 2:
        return desc, valid

    pts = kp[valid]

    center = pts.mean(
        axis=0,
        keepdims=True,
    )

    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)

    scale = float(
        np.linalg.norm(
            maxs - mins
        )
    )

    if scale < 1e-6:
        scale = 1.0

    desc[valid] = (
        kp[valid] - center
    ) / scale

    return desc, valid


def pose_distance(a, b):
    da, va = a
    db, vb = b

    common = va & vb

    if common.sum() < 4:
        return 0.0

    d = np.linalg.norm(
        da[common] - db[common],
        axis=1,
    )

    return float(
        d.mean()
    )


def source_quality(row):
    """
    Ranking only, NOT a hard filtering threshold.

    Source mainly represents appearance / identity,
    therefore body + face + hands are used together.
    """

    body = float(
        row.get(
            "body_mean_conf",
            0.0,
        )
    )

    face = float(
        row.get(
            "face_mean_conf",
            0.0,
        )
    )

    hand = float(
        row.get(
            "hand_mean_conf",
            0.0,
        )
    )

    q = (
        0.50 * body
        + 0.30 * face
        + 0.20 * hand
    )

    if row.get(
        "pose_soft_risk",
        False,
    ):
        q -= 0.10

    return float(q)


def choose_sources(rows, pose_cache):
    """
    1. First source = highest appearance/pose quality.
    2. Remaining sources = greedily maximize pose diversity
       relative to already-selected sources.
    """

    if not rows:
        return []

    ranked = sorted(
        rows,
        key=lambda r: (
            source_quality(r),
            float(
                r.get(
                    "face_mean_conf",
                    0.0,
                )
            ),
            float(
                r.get(
                    "hand_mean_conf",
                    0.0,
                )
            ),
        ),
        reverse=True,
    )

    # Prefer frames without soft-risk.
    clean = [
        r
        for r in ranked
        if not r.get(
            "pose_soft_risk",
            False,
        )
    ]

    candidates = (
        clean
        if len(clean) >= 2
        else ranked
    )

    selected = [
        candidates[0]
    ]

    while (
        len(selected)
        < MAX_SOURCES_PER_SEQUENCE
        and len(selected) < len(candidates)
    ):

        selected_keys = {
            (
                r["sequence_id"],
                r["stem"],
            )
            for r in selected
        }

        best = None
        best_tuple = None

        for r in candidates:

            key = (
                r["sequence_id"],
                r["stem"],
            )

            if key in selected_keys:
                continue

            dists = []

            for s in selected:
                skey = (
                    s["sequence_id"],
                    s["stem"],
                )

                dists.append(
                    pose_distance(
                        pose_cache[key],
                        pose_cache[skey],
                    )
                )

            min_dist = (
                min(dists)
                if dists
                else 0.0
            )

            # Pose diversity is primary;
            # source quality breaks ties.
            value = (
                min_dist,
                source_quality(r),
            )

            if (
                best_tuple is None
                or value > best_tuple
            ):
                best_tuple = value
                best = r

        if best is None:
            break

        selected.append(best)

    # If clean set is too small, fill from all usable frames.
    if (
        len(selected)
        < min(
            MAX_SOURCES_PER_SEQUENCE,
            len(ranked),
        )
    ):
        selected_keys = {
            (
                r["sequence_id"],
                r["stem"],
            )
            for r in selected
        }

        for r in ranked:
            key = (
                r["sequence_id"],
                r["stem"],
            )

            if key in selected_keys:
                continue

            selected.append(r)

            if (
                len(selected)
                >= MAX_SOURCES_PER_SEQUENCE
            ):
                break

    return selected


def slim_frame(row):
    return {
        "sequence_id":
            row["sequence_id"],

        "stem":
            row["stem"],

        "frame_num":
            row.get("frame_num"),

        "split":
            row["split"],

        "source_video_group":
            row["source_video_group"],

        "rgb":
            row["rgb"],

        "person_mask":
            row["person_mask"],

        "densepose":
            row.get("densepose"),

        "normal":
            row["normal"],

        "depth_vis":
            row["depth_vis"],

        "smplx_mask":
            row["smplx_mask"],

        "dwpose_png":
            row["dwpose_png"],

        "dwpose_json":
            row["dwpose_json"],

        "body_mean_conf":
            row["body_mean_conf"],

        "body_valid_03":
            row["body_valid_03"],

        "hand_mean_conf":
            row["hand_mean_conf"],

        "face_mean_conf":
            row["face_mean_conf"],

        "pose_soft_risk":
            row.get(
                "pose_soft_risk",
                False,
            ),
    }


def process_split(split):
    path = (
        SPLIT_ROOT
        / f"{split}.jsonl"
    )

    rows = read_jsonl(
        path
    )

    by_seq = defaultdict(list)

    for r in rows:
        by_seq[
            r["sequence_id"]
        ].append(r)

    # --------------------------------------------------------
    # DWPose descriptor cache
    # --------------------------------------------------------

    pose_cache = {}

    for r in rows:
        key = (
            r["sequence_id"],
            r["stem"],
        )

        pose_cache[key] = (
            pose_descriptor(r)
        )

    # --------------------------------------------------------
    # Build source pool
    # --------------------------------------------------------

    source_by_seq = {}
    source_rows = []

    for seq, seq_rows in by_seq.items():

        sources = choose_sources(
            seq_rows,
            pose_cache,
        )

        source_by_seq[seq] = sources

        for rank, r in enumerate(
            sources,
            1,
        ):
            x = slim_frame(r)

            x["source_rank"] = rank
            x["source_quality"] = (
                source_quality(r)
            )

            source_rows.append(x)

    # --------------------------------------------------------
    # Pair each target with most pose-different source(s)
    # --------------------------------------------------------

    pairs = []

    target_with_pair = set()
    skipped_targets = []

    pair_id = 0

    for seq in sorted(by_seq):

        seq_rows = sorted(
            by_seq[seq],
            key=lambda r: (
                r.get(
                    "frame_num"
                )
                if r.get(
                    "frame_num"
                ) is not None
                else r["stem"]
            ),
        )

        sources = source_by_seq[
            seq
        ]

        if len(seq_rows) < 2:
            for target in seq_rows:
                skipped_targets.append({
                    "sequence_id":
                        seq,

                    "stem":
                        target["stem"],

                    "reason":
                        "SEQUENCE_HAS_LT2_USABLE_FRAMES",
                })

            continue

        for target in seq_rows:

            target_key = (
                seq,
                target["stem"],
            )

            choices = []

            for source in sources:

                source_key = (
                    seq,
                    source["stem"],
                )

                if source_key == target_key:
                    continue

                dist = pose_distance(
                    pose_cache[target_key],
                    pose_cache[source_key],
                )

                choices.append(
                    (
                        dist,
                        source_quality(source),
                        source,
                    )
                )

            choices.sort(
                key=lambda x: (
                    x[0],
                    x[1],
                ),
                reverse=True,
            )

            choices = choices[
                :PAIRS_PER_TARGET
            ]

            if not choices:
                skipped_targets.append({
                    "sequence_id":
                        seq,

                    "stem":
                        target["stem"],

                    "reason":
                        "NO_DIFFERENT_SOURCE",
                })

                continue

            target_with_pair.add(
                target_key
            )

            for source_rank, (
                dist,
                quality,
                source,
            ) in enumerate(
                choices,
                1,
            ):

                pair_id += 1

                pairs.append({
                    "pair_id":
                        f"{split}_{pair_id:07d}",

                    "split":
                        split,

                    "sequence_id":
                        seq,

                    "source_video_group":
                        target[
                            "source_video_group"
                        ],

                    "source_stem":
                        source["stem"],

                    "target_stem":
                        target["stem"],

                    "source_rgb":
                        source["rgb"],

                    "target_rgb":
                        target["rgb"],

                    "source_person_mask":
                        source["person_mask"],

                    "target_person_mask":
                        target["person_mask"],

                    "target_dwpose":
                        target["dwpose_png"],

                    "target_normal":
                        target["normal"],

                    "target_depth_vis":
                        target["depth_vis"],

                    "target_smplx_mask":
                        target["smplx_mask"],

                    "source_dwpose_json":
                        source["dwpose_json"],

                    "target_dwpose_json":
                        target["dwpose_json"],

                    "pose_distance":
                        float(dist),

                    "source_quality":
                        float(quality),

                    "source_choice_rank":
                        source_rank,

                    "target_pose_soft_risk":
                        bool(
                            target.get(
                                "pose_soft_risk",
                                False,
                            )
                        ),
                })

    # --------------------------------------------------------
    # Write outputs
    # --------------------------------------------------------

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_path = (
        OUT_ROOT
        / f"{split}_source_pool.jsonl"
    )

    pair_path = (
        OUT_ROOT
        / f"{split}_pairs.jsonl"
    )

    skip_path = (
        OUT_ROOT
        / f"{split}_unpaired_targets.jsonl"
    )

    with source_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for r in source_rows:
            f.write(
                json.dumps(
                    r,
                    ensure_ascii=False,
                )
                + "\n"
            )

    with pair_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for r in pairs:
            f.write(
                json.dumps(
                    r,
                    ensure_ascii=False,
                )
                + "\n"
            )

    with skip_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for r in skipped_targets:
            f.write(
                json.dumps(
                    r,
                    ensure_ascii=False,
                )
                + "\n"
            )

    dists = np.asarray(
        [
            x["pose_distance"]
            for x in pairs
        ],
        dtype=np.float64,
    )

    summary = {
        "frames":
            len(rows),

        "sequences":
            len(by_seq),

        "source_pool_frames":
            len(source_rows),

        "pairs":
            len(pairs),

        "targets_with_pair":
            len(target_with_pair),

        "unpaired_targets":
            len(skipped_targets),

        "pose_distance": (
            {
                "min":
                    float(dists.min()),

                "mean":
                    float(dists.mean()),

                "median":
                    float(
                        np.median(dists)
                    ),

                "p10":
                    float(
                        np.quantile(
                            dists,
                            0.10,
                        )
                    ),

                "p90":
                    float(
                        np.quantile(
                            dists,
                            0.90,
                        )
                    ),

                "max":
                    float(dists.max()),
            }
            if len(dists)
            else {}
        ),
    }

    return summary


def main():

    summaries = {}

    for split in [
        "train",
        "val",
        "test",
    ]:
        print(
            f"Processing {split}...",
            flush=True,
        )

        summaries[
            split
        ] = process_split(
            split
        )

    total_pairs = sum(
        x["pairs"]
        for x in summaries.values()
    )

    summary = {
        "version":
            "tiktok_v3_pairs_v1",

        "policy": {
            "same_sequence_only":
                True,

            "cross_split_pairing":
                False,

            "max_sources_per_sequence":
                MAX_SOURCES_PER_SEQUENCE,

            "pairs_per_target":
                PAIRS_PER_TARGET,

            "source_selection":
                (
                    "quality-first, then greedy "
                    "DWPose pose diversity"
                ),

            "target_selection":
                "all usable frames",

            "pair_selection":
                (
                    "for each target choose the "
                    "most pose-different source(s)"
                ),

            "source_equals_target":
                False,
        },

        "splits":
            summaries,

        "total_pairs":
            total_pairs,
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("TikTok V3 PAIR BUILD COMPLETE")
    print("=" * 78)

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
