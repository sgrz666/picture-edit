#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Render already-selected 3DPW 60 showcase pairs in report style.

Layout:
SOURCE RGB | TARGET RGB | TARGET DWPose | TARGET SMPL Normal | TARGET SMPL Depth

Notes
-----
- Source assets other than RGB are not shown.
- Target DWPose uses black background.
- Target Normal keeps camera-space RGB normal visualization.
- Target Depth uses a colored colormap on black background.
- One pair -> one image.
- Does NOT change dataset assets or pair selection.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt
from matplotlib import cm


# ============================================================
# Paths
# ============================================================

ROOT = Path(
    "/home/shangguanrz/project/pic-edit/"
    "datasets/3dpw_single_final_v2"
)

SELECT_DIR = (
    ROOT /
    "visualization_60_pairs_diverse_dinov2_v2"
)

SELECT_CSV = (
    SELECT_DIR /
    "selected_60_pairs.csv"
)

OUT_DIR = (
    ROOT /
    "visualization_60_pairs_report_style_v1"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Helpers
# ============================================================

def load_rgb(rel_path):

    path = ROOT / str(rel_path)

    if not path.exists():
        raise FileNotFoundError(path)

    return np.asarray(
        Image.open(path).convert("RGB")
    )


def load_depth(rel_path):

    path = ROOT / str(rel_path)

    if not path.exists():
        raise FileNotFoundError(path)

    return np.asarray(
        Image.open(path)
    )


# ============================================================
# DWPose -> black background
# ============================================================

def dwpose_black_background(image):
    """
    Existing DWPose render should normally already be black-background.

    This function additionally removes very dark near-black noise and
    guarantees a truly black background.

    It deliberately does NOT attempt aggressive scene-background removal,
    because that could damage skeleton pixels.
    """

    img = image.copy()

    # Anything extremely dark becomes exact black.
    intensity = img.max(axis=2)

    img[
        intensity < 20
    ] = 0

    return img


# ============================================================
# Normal -> exact black background
# ============================================================

def normal_black_background(image):
    """
    3DPW normal convention:
        foreground = encoded RGB normal
        background = black

    Force near-black pixels to exact black for clean visualization.
    """

    img = image.copy()

    intensity = img.max(axis=2)

    img[
        intensity < 5
    ] = 0

    return img


# ============================================================
# Depth -> colored visualization on black background
# ============================================================

def depth_color_visualization(depth):
    """
    Raw dataset:
        uint16 PNG
        unit = millimeter
        background = 0

    Visualization:
        black background
        foreground uses turbo colormap
        near/far normalized only for display

    Original depth asset is NOT modified.
    """

    depth = depth.astype(
        np.float32
    )

    valid = depth > 0

    h, w = depth.shape[:2]

    rgb = np.zeros(
        (h, w, 3),
        dtype=np.uint8
    )

    if not valid.any():
        return rgb


    values = depth[
        valid
    ]


    # Robust display range
    lo = float(
        np.percentile(
            values,
            2
        )
    )

    hi = float(
        np.percentile(
            values,
            98
        )
    )


    if hi <= lo:
        hi = lo + 1.0


    normalized = (
        depth - lo
    ) / (
        hi - lo
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0
    )


    # Near -> warmer, far -> cooler.
    # Reverse normalized depth for visual intuition.
    normalized = (
        1.0 - normalized
    )


    cmap = cm.get_cmap(
        "turbo"
    )


    colored = cmap(
        normalized
    )[
        ...,
        :3
    ]


    colored = (
        colored * 255.0
    ).astype(
        np.uint8
    )


    rgb[
        valid
    ] = colored[
        valid
    ]


    return rgb


# ============================================================
# Draw
# ============================================================

def draw_pair(
    row,
    display_id,
):

    # --------------------------------------------------------
    # Load only assets actually shown
    # --------------------------------------------------------

    source_rgb = load_rgb(
        row[
            "source_rgb"
        ]
    )

    target_rgb = load_rgb(
        row[
            "target_rgb"
        ]
    )


    target_dwpose = load_rgb(
        row[
            "target_dwpose_render"
        ]
    )

    target_dwpose = (
        dwpose_black_background(
            target_dwpose
        )
    )


    target_normal = load_rgb(
        row[
            "target_normal"
        ]
    )

    target_normal = (
        normal_black_background(
            target_normal
        )
    )


    target_depth = load_depth(
        row[
            "target_depth"
        ]
    )

    target_depth = (
        depth_color_visualization(
            target_depth
        )
    )


    # ========================================================
    # Figure
    # ========================================================

    images = [
        source_rgb,
        target_rgb,
        target_dwpose,
        target_normal,
        target_depth,
    ]


    titles = [
        "SOURCE RGB",
        "TARGET RGB",
        "TARGET DWPose",
        "TARGET SMPL Normal",
        "TARGET SMPL Depth",
    ]


    fig, axes = plt.subplots(
        1,
        5,
        figsize=(
            20,
            5.8
        ),
        gridspec_kw={
            "wspace": 0.03
        }
    )


    # --------------------------------------------------------
    # Draw
    # --------------------------------------------------------

    for ax, image, title in zip(
        axes,
        images,
        titles,
    ):

        ax.imshow(
            image
        )

        ax.set_title(
            title,
            fontsize=14,
            pad=15,
        )

        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)


    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    sequence = str(
        row[
            "sequence"
        ]
    )

    split = str(
        row[
            "split"
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


    # --------------------------------------------------------
    # Bottom metadata: similar to your HumanVid example
    # --------------------------------------------------------

    fig.text(
        0.02,
        0.085,
        (
            f"#{display_id:02d}   "
            f"split={split}   "
            f"sequence={sequence}   "
            f"pose={pose_bin.upper()}   "
            f"pair={pair_id}"
        ),
        fontsize=10,
        ha="left",
    )


    fig.text(
        0.02,
        0.045,
        (
            f"SOURCE frame={source_frame}    "
            f"TARGET frame={target_frame}"
        ),
        fontsize=10,
        ha="left",
    )


    plt.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.84,
        bottom=0.15,
        wspace=0.03,
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
        f"{split}_"
        f"{pose_bin}_"
        f"{safe_seq}_"
        f"pair{pair_id}.png"
    )


    out_path = (
        OUT_DIR /
        filename
    )


    fig.savefig(
        out_path,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )


    plt.close(
        fig
    )


    return out_path


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 100)
    print("3DPW 60-PAIR REPORT-STYLE VISUALIZATION")
    print("=" * 100)


    if not SELECT_CSV.exists():

        raise FileNotFoundError(
            SELECT_CSV
        )


    selected = pd.read_csv(
        SELECT_CSV
    )


    if len(selected) != 60:

        raise RuntimeError(
            f"Expected 60 selected pairs, got {len(selected)}"
        )


    # Clean previous report PNGs
    for p in OUT_DIR.glob(
        "*.png"
    ):
        p.unlink()


    outputs = []


    for i, row in selected.iterrows():

        out = draw_pair(
            row,
            i + 1
        )


        outputs.append(
            out
        )


        print(
            f"[{i + 1:02d}/60] "
            f"{row['split']:10s} | "
            f"{row['pose_bin']:6s} | "
            f"{row['sequence']}"
        )


    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)

    print(
        "rendered:",
        len(outputs)
    )

    print(
        "saved:",
        OUT_DIR
    )

    print()
    print(
        "✅ report-style 60 pair figures ready"
    )


if __name__ == "__main__":
    main()

