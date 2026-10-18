#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3DPW final dataset visualization

Selection:
- all 30 final sequences
- exactly 2 pairs per sequence
- total 60 pairs
- prefer large pose changes
- priority: large > medium > small
- try to avoid reusing source/target frames inside the same sequence

Each selected pair is saved as one independent montage:

             RGB | DWPose | Mask | Depth | Normal
    Source
    Target
"""

from pathlib import Path
import json
import random

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
    "visualization_60_pairs_large_pose"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

SEED = 2026

PAIRS_PER_SEQUENCE = 2

BIN_PRIORITY = {
    "large": 3,
    "medium": 2,
    "small": 1,
}


# ============================================================
# Load all final pair manifests
# ============================================================

all_df = []

for split in [
    "train",
    "validation",
    "test",
]:

    p = (
        ROOT /
        f"{split}_pairs.csv"
    )

    df = pd.read_csv(
        p
    )

    df["split"] = split

    all_df.append(
        df
    )

df = pd.concat(
    all_df,
    ignore_index=True
)


print("=" * 100)
print("FINAL DATASET")
print("=" * 100)

print(
    "pairs:",
    len(df)
)

print(
    "sequences:",
    df["sequence"].nunique()
)

if df[
    "sequence"
].nunique() != 30:

    print(
        "WARNING: expected 30 sequences, got",
        df["sequence"].nunique()
    )


# ============================================================
# Select 2 pairs / sequence
# ============================================================

rng = random.Random(
    SEED
)

selected_rows = []


def choose_two(
    seq_df,
):
    """
    Prefer:
        large > medium > small

    Also prefer two pairs that do not share any source/target
    frame with each other.
    """

    work = seq_df.copy()

    work[
        "_priority"
    ] = (
        work[
            "pose_bin"
        ]
        .astype(str)
        .map(
            BIN_PRIORITY
        )
        .fillna(0)
    )

    # Random tiebreaker, deterministic
    random_values = [
        rng.random()
        for _ in range(
            len(work)
        )
    ]

    work[
        "_random"
    ] = random_values

    work = work.sort_values(
        [
            "_priority",
            "_random",
        ],
        ascending=[
            False,
            True,
        ]
    )


    # --------------------------------------------------------
    # First pair:
    # simply highest-priority available pair.
    # --------------------------------------------------------

    first = work.iloc[
        0
    ]

    used_frames = {
        int(
            first[
                "source_frame_index"
            ]
        ),
        int(
            first[
                "target_frame_index"
            ]
        ),
    }


    # --------------------------------------------------------
    # Second pair:
    # prefer highest pose bin AND no overlap with first pair.
    # --------------------------------------------------------

    second = None

    for _, row in work.iloc[
        1:
    ].iterrows():

        frames = {
            int(
                row[
                    "source_frame_index"
                ]
            ),
            int(
                row[
                    "target_frame_index"
                ]
            ),
        }

        if used_frames.isdisjoint(
            frames
        ):

            second = row
            break


    # --------------------------------------------------------
    # Fallback:
    # if every pair shares a frame, just choose next best pair.
    # --------------------------------------------------------

    if second is None:

        if len(work) < 2:
            raise RuntimeError(
                f"Sequence has fewer than 2 pairs"
            )

        second = work.iloc[
            1
        ]


    return [
        first,
        second,
    ]


# ============================================================
# Run sequence-wise selection
# ============================================================

sequences = sorted(
    df[
        "sequence"
    ]
    .astype(str)
    .unique()
)

for seq in sequences:

    seq_df = df[
        df[
            "sequence"
        ].astype(str)
        ==
        seq
    ].copy()

    if len(seq_df) < 2:
        raise RuntimeError(
            f"{seq}: fewer than 2 final pairs"
        )

    chosen = choose_two(
        seq_df
    )

    for row in chosen:

        selected_rows.append(
            row
        )


selected = pd.DataFrame(
    selected_rows
).reset_index(
    drop=True
)


# ============================================================
# Strong verification
# ============================================================

if len(selected) != 60:

    raise RuntimeError(
        f"Expected 60 pairs, got {len(selected)}"
    )


counts = (
    selected[
        "sequence"
    ]
    .value_counts()
)

if not (
    counts == 2
).all():

    print(counts)

    raise RuntimeError(
        "Not every sequence has exactly 2 pairs"
    )


if selected[
    "sequence"
].nunique() != 30:

    raise RuntimeError(
        "Expected 30 unique sequences"
    )


# Remove internal selection columns
selected = selected.drop(
    columns=[
        "_priority",
        "_random",
    ],
    errors="ignore"
)


# ============================================================
# Selection stats
# ============================================================

print()
print("=" * 100)
print("SELECTED 60 PAIRS")
print("=" * 100)

print()
print("Pose bins:")

print(
    selected[
        "pose_bin"
    ]
    .value_counts()
)

print()
print("Split distribution:")

print(
    selected[
        "split"
    ]
    .value_counts()
)

print()
print("Per-sequence:")

print(
    selected[
        [
            "split",
            "sequence",
            "pose_bin",
            "pair_id",
            "source_frame_index",
            "target_frame_index",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# Image helpers
# ============================================================

def load_rgb(
    rel_path,
):

    p = (
        ROOT /
        str(
            rel_path
        )
    )

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


def load_gray(
    rel_path,
):

    p = (
        ROOT /
        str(
            rel_path
        )
    )

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


def load_depth(
    rel_path,
):

    p = (
        ROOT /
        str(
            rel_path
        )
    )

    if not p.exists():
        raise FileNotFoundError(
            p
        )

    return np.asarray(
        Image.open(
            p
        )
    )


def depth_vis(
    depth,
):
    """
    Metric depth is NOT modified.

    Only visualization is normalized:
    background=black
    nearer=brighter
    """

    depth = depth.astype(
        np.float32
    )

    valid = (
        depth > 0
    )

    vis = np.zeros_like(
        depth,
        dtype=np.float32
    )

    if not valid.any():

        return vis


    values = depth[
        valid
    ]

    lo = np.percentile(
        values,
        2
    )

    hi = np.percentile(
        values,
        98
    )


    if hi <= lo:
        hi = lo + 1.0


    x = (
        hi - depth
    ) / (
        hi - lo
    )

    x = np.clip(
        x,
        0.0,
        1.0
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

    # --------------------------------------------------------
    # Source assets
    # --------------------------------------------------------

    source_rgb = load_rgb(
        row[
            "source_rgb"
        ]
    )

    source_pose = load_rgb(
        row[
            "source_dwpose_render"
        ]
    )

    source_mask = load_gray(
        row[
            "source_person_mask"
        ]
    )

    source_depth = depth_vis(
        load_depth(
            row[
                "source_depth"
            ]
        )
    )

    source_normal = load_rgb(
        row[
            "source_normal"
        ]
    )


    # --------------------------------------------------------
    # Target assets
    # --------------------------------------------------------

    target_rgb = load_rgb(
        row[
            "target_rgb"
        ]
    )

    target_pose = load_rgb(
        row[
            "target_dwpose_render"
        ]
    )

    target_mask = load_gray(
        row[
            "target_person_mask"
        ]
    )

    target_depth = depth_vis(
        load_depth(
            row[
                "target_depth"
            ]
        )
    )

    target_normal = load_rgb(
        row[
            "target_normal"
        ]
    )


    source_assets = [
        source_rgb,
        source_pose,
        source_mask,
        source_depth,
        source_normal,
    ]

    target_assets = [
        target_rgb,
        target_pose,
        target_mask,
        target_depth,
        target_normal,
    ]


    titles = [
        "RGB",
        "DWPose",
        "Person Mask",
        "SMPL Depth",
        "SMPL Normal",
    ]


    # ========================================================
    # Figure
    # ========================================================

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(
            15,
            8
        )
    )


    # --------------------------------------------------------
    # Draw source / target
    # --------------------------------------------------------

    for j in range(
        5
    ):

        if j in [
            2,
            3,
        ]:

            axes[
                0,
                j
            ].imshow(
                source_assets[
                    j
                ],
                cmap="gray"
            )

            axes[
                1,
                j
            ].imshow(
                target_assets[
                    j
                ],
                cmap="gray"
            )

        else:

            axes[
                0,
                j
            ].imshow(
                source_assets[
                    j
                ]
            )

            axes[
                1,
                j
            ].imshow(
                target_assets[
                    j
                ]
            )


    # --------------------------------------------------------
    # Column titles
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


    # --------------------------------------------------------
    # Row names
    # --------------------------------------------------------

    axes[
        0,
        0
    ].set_ylabel(
        "SOURCE",
        fontsize=14,
        fontweight="bold",
        labelpad=12
    )

    axes[
        1,
        0
    ].set_ylabel(
        "TARGET",
        fontsize=14,
        fontweight="bold",
        labelpad=12
    )


    # --------------------------------------------------------
    # Remove axis decorations
    # --------------------------------------------------------

    for ax in axes.flat:

        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():

            spine.set_visible(
                False
            )


    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    split = str(
        row[
            "split"
        ]
    )

    sequence = str(
        row[
            "sequence"
        ]
    )

    pose_bin = str(
        row[
            "pose_bin"
        ]
    )

    pair_id = int(
        row[
            "pair_id"
        ]
    )

    source_frame = int(
        row[
            "source_frame_index"
        ]
    )

    target_frame = int(
        row[
            "target_frame_index"
        ]
    )


    fig.suptitle(
        (
            f"3DPW Example {display_id:02d}  |  "
            f"{split.upper()}  |  "
            f"Pose Change: {pose_bin.upper()}\n"
            f"{sequence}  |  "
            f"Pair {pair_id}  |  "
            f"Frame {source_frame} -> {target_frame}"
        ),
        fontsize=16,
        fontweight="bold",
        y=0.98
    )


    fig.text(
        0.5,
        0.018,
        (
            "RGB  |  DWPose WholeBody  |  "
            "Visible Person Mask  |  "
            "Official-SMPL Depth  |  "
            "Official-SMPL Normal"
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


    # --------------------------------------------------------
    # Filename
    # --------------------------------------------------------

    filename = (
        f"{display_id:02d}_"
        f"{split}_"
        f"{pose_bin}_"
        f"{sequence}_"
        f"pair{pair_id}.png"
    )

    out_path = (
        OUT /
        filename
    )


    fig.savefig(
        out_path,
        dpi=180,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(
        fig
    )


    return out_path


# ============================================================
# Render 60 figures
# ============================================================

outputs = []

for i, row in selected.iterrows():

    out = draw_pair(
        row,
        i + 1
    )

    outputs.append(
        str(
            out
        )
    )

    print(
        f"[{i + 1:02d}/60] "
        f"{row['split']:10s} | "
        f"{row['pose_bin']:6s} | "
        f"{row['sequence']}"
    )


# ============================================================
# Save selection manifest
# ============================================================

selected.to_csv(
    OUT /
    "selected_60_pairs.csv",
    index=False
)


summary = {
    "dataset":
        "3dpw_single_final_v2",

    "selection":
        (
            "2 pairs per sequence, "
            "prefer large pose changes"
        ),

    "seed":
        SEED,

    "num_pairs":
        int(
            len(
                selected
            )
        ),

    "unique_sequences":
        int(
            selected[
                "sequence"
            ].nunique()
        ),

    "pairs_per_sequence":
        2,

    "pose_bins": {
        k: int(v)
        for k, v
        in selected[
            "pose_bin"
        ]
        .value_counts()
        .sort_index()
        .items()
    },

    "splits": {
        k: int(v)
        for k, v
        in selected[
            "split"
        ]
        .value_counts()
        .sort_index()
        .items()
    },

    "sequence_counts": {
        k: int(v)
        for k, v
        in selected[
            "sequence"
        ]
        .value_counts()
        .sort_index()
        .items()
    },

    "outputs":
        outputs,
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
print("=" * 100)
print("DONE")
print("=" * 100)

print(
    "pairs:",
    len(
        selected
    )
)

print(
    "unique sequences:",
    selected[
        "sequence"
    ].nunique()
)

print()
print(
    "pose bins:"
)

print(
    selected[
        "pose_bin"
    ]
    .value_counts()
)

print()
print(
    "saved:",
    OUT
)

print()
print(
    "✅ 60 individual pair montages finished"
)
