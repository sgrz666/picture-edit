#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import random
import json

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


# ============================================================
# Config
# ============================================================

ROOT = Path(
    "/home/shangguanrz/project/pic-edit/"
    "datasets/3dpw_single_final_v2"
)

OUT = (
    ROOT /
    "visualization_20_pairs_diverse"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

SEED = 2026

TARGET_BIN_COUNTS = {
    "small": 7,
    "medium": 7,
    "large": 6,
}


# ============================================================
# Load all splits
# ============================================================

dfs = []

for split in [
    "train",
    "validation",
    "test",
]:

    p = ROOT / f"{split}_pairs.csv"

    df = pd.read_csv(p)

    # Ensure split is explicitly recorded
    df["split"] = split

    dfs.append(df)

df = pd.concat(
    dfs,
    ignore_index=True
)

print("=" * 90)
print("ALL FINAL PAIRS")
print("=" * 90)

print("pairs     :", len(df))
print(
    "sequences :",
    df["sequence"].nunique()
)

print()
print(
    df.groupby("split")["sequence"]
      .nunique()
)


# ============================================================
# Select 20 pairs
#
# Requirements:
# - 20 different sequences
# - 7 small
# - 7 medium
# - 6 large
# - deterministic seed
# ============================================================

rng = random.Random(SEED)

all_sequences = sorted(
    df["sequence"]
    .astype(str)
    .unique()
)

# sequence -> pose_bin -> candidate row indices
lookup = {}

for seq in all_sequences:

    seq_df = df[
        df["sequence"].astype(str) == seq
    ]

    lookup[seq] = {}

    for pose_bin in [
        "small",
        "medium",
        "large",
    ]:

        ids = seq_df[
            seq_df["pose_bin"].astype(str)
            ==
            pose_bin
        ].index.tolist()

        lookup[seq][pose_bin] = ids


# ------------------------------------------------------------
# Use randomized retries because we want both:
# 1) unique sequences
# 2) exact bin counts
# ------------------------------------------------------------

selected = None

for attempt in range(10000):

    sequences = all_sequences.copy()

    rng.shuffle(sequences)

    used_sequences = set()

    chosen = []

    # Mix requested bins rather than picking all small first
    requested_bins = (
        ["small"] * 7
        +
        ["medium"] * 7
        +
        ["large"] * 6
    )

    rng.shuffle(
        requested_bins
    )

    success = True

    for pose_bin in requested_bins:

        candidates = [
            seq
            for seq in sequences
            if (
                seq not in used_sequences
                and
                len(
                    lookup[seq][pose_bin]
                ) > 0
            )
        ]

        if not candidates:
            success = False
            break

        # Prefer sequences from less-represented split
        split_counts = {
            "train": 0,
            "validation": 0,
            "test": 0,
        }

        for x in chosen:
            split_counts[
                x["split"]
            ] += 1

        min_count = min(
            split_counts.values()
        )

        preferred_splits = {
            s
            for s, c in split_counts.items()
            if c == min_count
        }

        preferred = []

        for seq in candidates:

            rows = lookup[
                seq
            ][
                pose_bin
            ]

            row0 = df.loc[
                rows[0]
            ]

            if str(
                row0["split"]
            ) in preferred_splits:

                preferred.append(
                    seq
                )

        if preferred:
            candidates = preferred

        seq = rng.choice(
            candidates
        )

        row_idx = rng.choice(
            lookup[seq][pose_bin]
        )

        row = df.loc[
            row_idx
        ].copy()

        chosen.append(
            {
                "row_idx":
                    int(row_idx),

                "sequence":
                    str(
                        row["sequence"]
                    ),

                "split":
                    str(
                        row["split"]
                    ),

                "pose_bin":
                    str(
                        row["pose_bin"]
                    ),
            }
        )

        used_sequences.add(
            seq
        )

    if success and len(chosen) == 20:

        selected = chosen
        break


if selected is None:
    raise RuntimeError(
        "Could not find 20 unique-sequence pairs "
        "with requested pose-bin distribution."
    )


selected_indices = [
    x["row_idx"]
    for x in selected
]

selected_df = df.loc[
    selected_indices
].copy().reset_index(
    drop=True
)


# ============================================================
# Verification
# ============================================================

assert len(
    selected_df
) == 20

assert selected_df[
    "sequence"
].nunique() == 20

bin_counts = (
    selected_df[
        "pose_bin"
    ]
    .value_counts()
    .to_dict()
)

for k, v in TARGET_BIN_COUNTS.items():

    if bin_counts.get(
        k,
        0
    ) != v:

        raise RuntimeError(
            f"Wrong bin count: {bin_counts}"
        )


print()
print("=" * 90)
print("SELECTED 20 PAIRS")
print("=" * 90)

print()
print("pose bins:")
print(
    selected_df[
        "pose_bin"
    ].value_counts()
)

print()
print("splits:")
print(
    selected_df[
        "split"
    ].value_counts()
)

print()
print("sequences:")
print(
    selected_df[
        [
            "split",
            "sequence",
            "pose_bin",
            "pair_id",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# Image helpers
# ============================================================

def load_rgb(rel):

    p = ROOT / str(rel)

    if not p.exists():
        raise FileNotFoundError(
            p
        )

    return np.asarray(
        Image.open(
            p
        ).convert(
            "RGB"
        )
    )


def load_gray(rel):

    p = ROOT / str(rel)

    if not p.exists():
        raise FileNotFoundError(
            p
        )

    return np.asarray(
        Image.open(
            p
        ).convert(
            "L"
        )
    )


def load_depth(rel):

    p = ROOT / str(rel)

    if not p.exists():
        raise FileNotFoundError(
            p
        )

    return np.asarray(
        Image.open(
            p
        )
    )


def depth_vis(depth):

    depth = depth.astype(
        np.float32
    )

    valid = depth > 0

    vis = np.zeros_like(
        depth,
        dtype=np.float32
    )

    if not valid.any():
        return vis

    z = depth[
        valid
    ]

    lo = np.percentile(
        z,
        2
    )

    hi = np.percentile(
        z,
        98
    )

    if hi <= lo:
        hi = lo + 1

    # nearer = brighter
    x = (
        hi - depth
    ) / (
        hi - lo
    )

    x = np.clip(
        x,
        0,
        1
    )

    vis[
        valid
    ] = x[
        valid
    ]

    return vis


# ============================================================
# Draw one pair
# ============================================================

def draw_pair(
    row,
    display_id,
):

    src_rgb = load_rgb(
        row["source_rgb"]
    )

    tgt_rgb = load_rgb(
        row["target_rgb"]
    )

    src_pose = load_rgb(
        row["source_dwpose_render"]
    )

    tgt_pose = load_rgb(
        row["target_dwpose_render"]
    )

    src_mask = load_gray(
        row["source_person_mask"]
    )

    tgt_mask = load_gray(
        row["target_person_mask"]
    )

    src_depth = depth_vis(
        load_depth(
            row["source_depth"]
        )
    )

    tgt_depth = depth_vis(
        load_depth(
            row["target_depth"]
        )
    )

    src_normal = load_rgb(
        row["source_normal"]
    )

    tgt_normal = load_rgb(
        row["target_normal"]
    )


    fig, axes = plt.subplots(
        2,
        5,
        figsize=(
            15,
            8
        )
    )


    source_assets = [
        src_rgb,
        src_pose,
        src_mask,
        src_depth,
        src_normal,
    ]

    target_assets = [
        tgt_rgb,
        tgt_pose,
        tgt_mask,
        tgt_depth,
        tgt_normal,
    ]


    titles = [
        "RGB",
        "DWPose",
        "Person Mask",
        "SMPL Depth",
        "SMPL Normal",
    ]


    # --------------------------------------------------------
    # Source
    # --------------------------------------------------------

    for j, image in enumerate(
        source_assets
    ):

        if j in [
            2,
            3,
        ]:

            axes[
                0,
                j
            ].imshow(
                image,
                cmap="gray"
            )

        else:

            axes[
                0,
                j
            ].imshow(
                image
            )


    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    for j, image in enumerate(
        target_assets
    ):

        if j in [
            2,
            3,
        ]:

            axes[
                1,
                j
            ].imshow(
                image,
                cmap="gray"
            )

        else:

            axes[
                1,
                j
            ].imshow(
                image
            )


    # --------------------------------------------------------
    # Titles
    # --------------------------------------------------------

    for j, title in enumerate(
        titles
    ):

        axes[
            0,
            j
        ].set_title(
            title,
            fontsize=14,
            fontweight="bold"
        )


    axes[
        0,
        0
    ].set_ylabel(
        "SOURCE",
        fontsize=14,
        fontweight="bold"
    )

    axes[
        1,
        0
    ].set_ylabel(
        "TARGET",
        fontsize=14,
        fontweight="bold"
    )


    for ax in axes.flat:

        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(
                False
            )


    split = str(
        row["split"]
    )

    seq = str(
        row["sequence"]
    )

    pose_bin = str(
        row["pose_bin"]
    )

    pair_id = int(
        row["pair_id"]
    )

    sf = int(
        row[
            "source_frame_index"
        ]
    )

    tf = int(
        row[
            "target_frame_index"
        ]
    )


    fig.suptitle(
        (
            f"3DPW Example {display_id:02d}   |   "
            f"{split.upper()}   |   "
            f"Pose: {pose_bin.upper()}\n"
            f"{seq}   |   "
            f"Pair {pair_id}   |   "
            f"Frame {sf} -> {tf}"
        ),
        fontsize=16,
        fontweight="bold",
        y=0.98
    )


    fig.text(
        0.5,
        0.02,
        (
            "RGB / DWPose / Visible Person Mask / "
            "Official-SMPL Depth / Official-SMPL Normal"
        ),
        ha="center",
        fontsize=10
    )


    plt.tight_layout(
        rect=[
            0.02,
            0.05,
            0.98,
            0.92
        ]
    )


    name = (
        f"{display_id:02d}_"
        f"{split}_"
        f"{pose_bin}_"
        f"{seq}_"
        f"pair{pair_id}.png"
    )

    out = OUT / name

    fig.savefig(
        out,
        dpi=180,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(
        fig
    )

    return out


# ============================================================
# Render
# ============================================================

paths = []

for i, row in selected_df.iterrows():

    out = draw_pair(
        row,
        i + 1
    )

    paths.append(
        str(out)
    )

    print(
        f"[{i + 1:02d}/20] "
        f"{row['split']:10s} "
        f"{row['pose_bin']:6s} "
        f"{row['sequence']}"
    )


# ============================================================
# Save metadata
# ============================================================

selected_df.to_csv(
    OUT /
    "selected_20_pairs.csv",
    index=False
)


summary = {
    "num_pairs":
        20,

    "unique_sequences":
        int(
            selected_df[
                "sequence"
            ].nunique()
        ),

    "pose_bin_counts":
        {
            k: int(v)
            for k, v
            in selected_df[
                "pose_bin"
            ]
            .value_counts()
            .sort_index()
            .items()
        },

    "split_counts":
        {
            k: int(v)
            for k, v
            in selected_df[
                "split"
            ]
            .value_counts()
            .sort_index()
            .items()
        },

    "sequences":
        selected_df[
            "sequence"
        ].tolist(),

    "output_images":
        paths,
}


with open(
    OUT /
    "summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2
    )


print()
print("=" * 90)
print("DONE")
print("=" * 90)

print(
    "20 pairs       :",
    len(
        selected_df
    )
)

print(
    "unique sequence:",
    selected_df[
        "sequence"
    ].nunique()
)

print()
print(
    "saved:",
    OUT
)
