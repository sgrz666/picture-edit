#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize 20 representative pairs from 3DPW single-person dataset.

Each pair is saved as one independent 2x5 montage:

             RGB | DWPose | Mask | Depth | Normal
    Source
    Target

Sampling:
- validation split
- 20 pairs
- approximately balanced across small / medium / large pose bins
- tries to spread across sequences
- fixed random seed for reproducibility

This script only visualizes assets.
It does NOT modify any dataset files.
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

DATASET_ROOT = Path(
    "/home/shangguanrz/project/pic-edit/datasets/3dpw_single_final_v2"
)

PAIR_CSV = (
    DATASET_ROOT /
    "validation_pairs.csv"
)

OUT_DIR = (
    DATASET_ROOT /
    "visualization_20_pairs"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SEED = 2026

# 20 pairs total
# Slightly balanced for visualization rather than following
# the natural validation distribution exactly.
SAMPLE_COUNTS = {
    "small": 7,
    "medium": 7,
    "large": 6,
}


# ============================================================
# Helpers
# ============================================================

def load_rgb(rel_path):
    """
    RGB / DWPose / Normal:
    read as RGB image.
    """
    path = DATASET_ROOT / str(rel_path)

    if not path.exists():
        raise FileNotFoundError(path)

    return np.asarray(
        Image.open(path).convert("RGB")
    )


def load_gray(rel_path):
    """
    Person mask.
    """
    path = DATASET_ROOT / str(rel_path)

    if not path.exists():
        raise FileNotFoundError(path)

    return np.asarray(
        Image.open(path).convert("L")
    )


def load_depth(rel_path):
    """
    Load raw uint16 metric depth PNG.

    Dataset convention:
        value = depth in millimeters
        background = 0
    """
    path = DATASET_ROOT / str(rel_path)

    if not path.exists():
        raise FileNotFoundError(path)

    depth = np.asarray(
        Image.open(path)
    )

    return depth


def depth_for_display(depth):
    """
    Convert metric depth to a grayscale image ONLY for visualization.

    Important:
    - raw depth file is untouched
    - background remains black
    - foreground is normalized per image for visibility
    """
    depth = depth.astype(np.float32)

    valid = depth > 0

    vis = np.zeros(
        depth.shape,
        dtype=np.float32
    )

    if valid.sum() == 0:
        return vis

    values = depth[valid]

    # Robust display limits.
    lo = np.percentile(values, 2)
    hi = np.percentile(values, 98)

    if hi <= lo:
        hi = lo + 1.0

    # Closer = brighter.
    normalized = (
        hi - depth
    ) / (
        hi - lo
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0
    )

    vis[valid] = normalized[valid]

    return vis


# ============================================================
# Representative sampling
# ============================================================

def select_pairs(df):
    """
    Select representative pairs while trying to spread
    samples across different sequences.
    """

    rng = random.Random(SEED)

    selected = []

    for pose_bin, n_needed in SAMPLE_COUNTS.items():

        part = df[
            df["pose_bin"].astype(str) == pose_bin
        ].copy()

        if len(part) < n_needed:
            raise RuntimeError(
                f"{pose_bin}: only {len(part)} pairs, "
                f"but need {n_needed}"
            )

        # Sequence -> candidate rows
        groups = {
            seq: group.copy()
            for seq, group
            in part.groupby("sequence")
        }

        sequences = sorted(
            groups.keys()
        )

        # Shuffle sequence order reproducibly
        rng.shuffle(sequences)

        # Shuffle rows within sequence
        group_rows = {}

        for seq in sequences:

            indices = list(
                groups[seq].index
            )

            rng.shuffle(indices)

            group_rows[seq] = indices

        chosen_indices = []

        # ----------------------------------------------------
        # Round-robin by sequence first.
        # This prevents all examples coming from one video.
        # ----------------------------------------------------

        pointers = {
            seq: 0
            for seq in sequences
        }

        while len(chosen_indices) < n_needed:

            progress = False

            current_sequences = sequences.copy()
            rng.shuffle(current_sequences)

            for seq in current_sequences:

                if len(chosen_indices) >= n_needed:
                    break

                pointer = pointers[seq]

                indices = group_rows[seq]

                if pointer >= len(indices):
                    continue

                idx = indices[pointer]

                pointers[seq] += 1

                chosen_indices.append(
                    idx
                )

                progress = True

            if not progress:
                break

        if len(chosen_indices) != n_needed:
            raise RuntimeError(
                f"Could not select enough {pose_bin} samples"
            )

        selected.extend(
            chosen_indices
        )

    selected_df = df.loc[
        selected
    ].copy()

    # Shuffle final order so bins are mixed visually.
    selected_df = selected_df.sample(
        frac=1.0,
        random_state=SEED
    ).reset_index(
        drop=True
    )

    return selected_df


# ============================================================
# Draw one pair
# ============================================================

def draw_pair(row, display_id):
    """
    Each pair -> one independent montage.
    """

    source_rgb = load_rgb(
        row["source_rgb"]
    )

    target_rgb = load_rgb(
        row["target_rgb"]
    )

    source_dwpose = load_rgb(
        row["source_dwpose_render"]
    )

    target_dwpose = load_rgb(
        row["target_dwpose_render"]
    )

    source_mask = load_gray(
        row["source_person_mask"]
    )

    target_mask = load_gray(
        row["target_person_mask"]
    )

    source_depth_raw = load_depth(
        row["source_depth"]
    )

    target_depth_raw = load_depth(
        row["target_depth"]
    )

    source_depth = depth_for_display(
        source_depth_raw
    )

    target_depth = depth_for_display(
        target_depth_raw
    )

    source_normal = load_rgb(
        row["source_normal"]
    )

    target_normal = load_rgb(
        row["target_normal"]
    )


    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(15, 8)
    )

    titles = [
        "RGB",
        "DWPose",
        "Person Mask",
        "SMPL Depth",
        "SMPL Normal",
    ]

    # --------------------------------------------------------
    # Source row
    # --------------------------------------------------------

    axes[0, 0].imshow(
        source_rgb
    )

    axes[0, 1].imshow(
        source_dwpose
    )

    axes[0, 2].imshow(
        source_mask,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[0, 3].imshow(
        source_depth,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[0, 4].imshow(
        source_normal
    )


    # --------------------------------------------------------
    # Target row
    # --------------------------------------------------------

    axes[1, 0].imshow(
        target_rgb
    )

    axes[1, 1].imshow(
        target_dwpose
    )

    axes[1, 2].imshow(
        target_mask,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[1, 3].imshow(
        target_depth,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[1, 4].imshow(
        target_normal
    )


    # --------------------------------------------------------
    # Column titles
    # --------------------------------------------------------

    for col, title in enumerate(titles):

        axes[0, col].set_title(
            title,
            fontsize=13,
            fontweight="bold",
            pad=8
        )


    # --------------------------------------------------------
    # Row labels
    # --------------------------------------------------------

    axes[0, 0].set_ylabel(
        "SOURCE",
        fontsize=14,
        fontweight="bold",
        labelpad=12
    )

    axes[1, 0].set_ylabel(
        "TARGET",
        fontsize=14,
        fontweight="bold",
        labelpad=12
    )


    # --------------------------------------------------------
    # Remove ticks
    # --------------------------------------------------------

    for ax in axes.flat:

        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)


    # --------------------------------------------------------
    # Overall title
    # --------------------------------------------------------

    pair_id = int(
        row["pair_id"]
    )

    sequence = str(
        row["sequence"]
    )

    pose_bin = str(
        row["pose_bin"]
    )

    source_frame = int(
        row["source_frame_index"]
    )

    target_frame = int(
        row["target_frame_index"]
    )

    fig.suptitle(
        (
            f"3DPW Pair {pair_id}  |  "
            f"Pose: {pose_bin.upper()}  |  "
            f"{sequence}\n"
            f"Frame {source_frame}  ->  Frame {target_frame}"
        ),
        fontsize=16,
        fontweight="bold",
        y=0.98
    )


    # --------------------------------------------------------
    # Small note
    # --------------------------------------------------------

    fig.text(
        0.5,
        0.018,
        (
            "Depth: official 3DPW SMPL + camera, metric depth "
            "(normalized only for visualization)  |  "
            "Normal: CV camera coordinates"
        ),
        ha="center",
        fontsize=9
    )


    plt.tight_layout(
        rect=[
            0.02,
            0.045,
            0.98,
            0.93
        ]
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    safe_seq = sequence.replace(
        "/",
        "_"
    )

    filename = (
        f"{display_id:02d}_"
        f"pair{pair_id:04d}_"
        f"{pose_bin}_"
        f"{safe_seq}.png"
    )

    out_path = (
        OUT_DIR /
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
# Main
# ============================================================

def main():

    print("=" * 90)
    print("3DPW 20-PAIR ASSET VISUALIZATION")
    print("=" * 90)

    if not PAIR_CSV.exists():
        raise FileNotFoundError(
            PAIR_CSV
        )

    df = pd.read_csv(
        PAIR_CSV
    )

    print(
        "validation pairs:",
        len(df)
    )

    print()
    print(
        "validation pose distribution:"
    )

    print(
        df[
            "pose_bin"
        ].value_counts()
    )


    # --------------------------------------------------------
    # Select
    # --------------------------------------------------------

    selected = select_pairs(
        df
    )

    selected[
        "display_id"
    ] = np.arange(
        1,
        len(selected) + 1
    )


    # --------------------------------------------------------
    # Save selected metadata
    # --------------------------------------------------------

    selected_csv = (
        OUT_DIR /
        "selected_20_pairs.csv"
    )

    selected.to_csv(
        selected_csv,
        index=False
    )


    # --------------------------------------------------------
    # Draw
    # --------------------------------------------------------

    print()
    print("Rendering...")

    output_paths = []

    for i, row in selected.iterrows():

        display_id = i + 1

        out_path = draw_pair(
            row,
            display_id
        )

        output_paths.append(
            str(out_path)
        )

        print(
            f"[{display_id:02d}/20] "
            f"pair_id={int(row['pair_id']):4d}  "
            f"bin={str(row['pose_bin']):6s}  "
            f"sequence={row['sequence']}"
        )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "dataset":
            "3dpw_single_final_v2",

        "source_manifest":
            str(PAIR_CSV),

        "sampling_split":
            "validation",

        "seed":
            SEED,

        "num_pairs":
            len(selected),

        "sampling_target": {
            k: int(v)
            for k, v
            in SAMPLE_COUNTS.items()
        },

        "actual_pose_bins": {
            k: int(v)
            for k, v
            in selected[
                "pose_bin"
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
            output_paths,
    }

    with open(
        OUT_DIR /
        "visualization_summary.json",
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

    print(
        "Selected pose bins:"
    )

    print(
        selected[
            "pose_bin"
        ]
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "Selected sequences:"
    )

    print(
        selected[
            "sequence"
        ]
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "Saved to:"
    )

    print(
        OUT_DIR
    )

    print()
    print(
        "✅ 20 individual pair montages finished"
    )


if __name__ == "__main__":
    main()

