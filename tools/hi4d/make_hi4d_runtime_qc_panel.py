#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(
    "/home/shangguanrz/project/pic-edit/datasets/Hi4D_pilot_v1"
)

ROW_FILE = (
    ROOT / "_strong_contact_test.jsonl"
)

GEO_FILE = (
    ROOT
    / "_runtime_geometry_qc_v2"
    / "runtime_geometry.npz"
)

OUT = (
    ROOT
    / "_runtime_geometry_qc_v2"
    / "QC_interaction_panel.jpg"
)


# ============================================================
# LOAD
# ============================================================

row = json.loads(
    ROW_FILE
    .read_text(
        encoding="utf-8"
    )
    .splitlines()[0]
)

with np.load(
    GEO_FILE,
    allow_pickle=True
) as d:

    g = {
        k: d[k]
        for k in d.files
    }


H = int(g["height"])
W = int(g["width"])


rgb = cv2.imread(
    row["rgb"],
    cv2.IMREAD_COLOR
)

if rgb is None:
    raise RuntimeError(
        row["rgb"]
    )


# QC only:
# runtime geometry already uses this HxW.
# No training asset is modified.
rgb = cv2.resize(
    rgb,
    (W, H),
    interpolation=cv2.INTER_AREA
)


# ============================================================
# HELPERS
# ============================================================

def add_title(
    image,
    text
):
    out = image.copy()

    cv2.rectangle(
        out,
        (0, 0),
        (W, 38),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        out,
        text,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return out


def overlay_color(
    image,
    mask,
    color,
    alpha=0.55,
):
    out = image.copy()

    m = (
        np.asarray(mask)
        > 0
    )

    if not np.any(m):
        return out

    color_layer = out.copy()

    color_layer[m] = color

    out[m] = cv2.addWeighted(
        out[m],
        1.0 - alpha,
        color_layer[m],
        alpha,
        0
    )

    return out


# ============================================================
# CONTACT
# ============================================================

contact_a = np.asarray(
    g["contact_A"],
    dtype=np.float32
)

contact_b = np.asarray(
    g["contact_B"],
    dtype=np.float32
)


# visualization threshold only;
# underlying tensor is NOT thresholded.
ca = contact_a > 0.05
cb = contact_b > 0.05

contact_vis = rgb.copy()

# OpenCV BGR
# A = red
contact_vis = overlay_color(
    contact_vis,
    ca,
    (40, 40, 240),
    0.62
)

# B = blue
contact_vis = overlay_color(
    contact_vis,
    cb,
    (240, 100, 30),
    0.62
)

# overlap = yellow
both = ca & cb

contact_vis = overlay_color(
    contact_vis,
    both,
    (20, 220, 240),
    0.78
)


# ============================================================
# INTER-PERSON OCCLUSION
# ============================================================

a_by_b = (
    np.asarray(
        g["A_occluded_by_B"]
    ) > 0
)

b_by_a = (
    np.asarray(
        g["B_occluded_by_A"]
    ) > 0
)

direction_vis = rgb.copy()

# A hidden by B = red
direction_vis = overlay_color(
    direction_vis,
    a_by_b,
    (40, 40, 245),
    0.72
)

# B hidden by A = blue
direction_vis = overlay_color(
    direction_vis,
    b_by_a,
    (245, 100, 25),
    0.72
)


# ============================================================
# GENERAL OCCLUSION
# ============================================================

occ = (
    np.asarray(
        g["occlusion_mask"]
    ) > 0
)

occ_vis = overlay_color(
    rgb,
    occ,
    (20, 150, 255),
    0.58
)


# ============================================================
# DEPTH GRADIENT
# ============================================================

grad = np.asarray(
    g["occlusion_depth_gradient"],
    dtype=np.float32
)

grad_vis = rgb.copy()

gm = grad > 0

if np.any(gm):

    values = grad[gm]

    hi = float(
        np.percentile(
            values,
            99
        )
    )

    hi = max(
        hi,
        1e-8
    )

    normalized = np.clip(
        grad / hi,
        0,
        1
    )

    heat8 = (
        normalized
        * 255
    ).astype(
        np.uint8
    )

    heat = cv2.applyColorMap(
        heat8,
        cv2.COLORMAP_TURBO
    )

    strong = (
        normalized > 0.08
    )

    grad_vis[
        strong
    ] = cv2.addWeighted(
        grad_vis[
            strong
        ],
        0.35,
        heat[
            strong
        ],
        0.65,
        0
    )


# ============================================================
# SEMANTIC A / B
# ============================================================

sa = (
    np.asarray(
        g["semantic_A"]
    ) > 0
)

sb = (
    np.asarray(
        g["semantic_B"]
    ) > 0
)

semantic_vis = rgb.copy()

semantic_vis = overlay_color(
    semantic_vis,
    sa,
    (40, 40, 240),
    0.35
)

semantic_vis = overlay_color(
    semantic_vis,
    sb,
    (240, 100, 30),
    0.35
)

semantic_vis = overlay_color(
    semantic_vis,
    sa & sb,
    (20, 220, 240),
    0.58
)


# ============================================================
# INTERPERSON OVERLAP
# ============================================================

overlap = (
    np.asarray(
        g["interperson_overlap"]
    ) > 0
)

overlap_vis = overlay_color(
    rgb,
    overlap,
    (40, 220, 220),
    0.72
)


# ============================================================
# PANEL
# ============================================================

cells = [
    add_title(
        rgb,
        "RGB"
    ),

    add_title(
        contact_vis,
        "Contact: A=Red  B=Blue"
    ),

    add_title(
        semantic_vis,
        "SMPL semantic A / B"
    ),

    add_title(
        direction_vis,
        "Directional occlusion"
    ),

    add_title(
        occ_vis,
        "General occlusion"
    ),

    add_title(
        grad_vis,
        "Occlusion depth gradient"
    ),
]


top = np.hstack(
    cells[:3]
)

bottom = np.hstack(
    cells[3:]
)

panel = np.vstack(
    [
        top,
        bottom
    ]
)


cv2.imwrite(
    str(OUT),
    panel,
    [
        cv2.IMWRITE_JPEG_QUALITY,
        95
    ]
)


# ============================================================
# METRICS
# ============================================================

print(
    "frame:",
    row["pair"],
    row["action"],
    row["frame_id"]
)

print(
    "runtime:",
    f"{W}x{H}"
)

print()

print(
    "contact_A px:",
    int(ca.sum())
)

print(
    "contact_B px:",
    int(cb.sum())
)

print(
    "contact overlap px:",
    int(both.sum())
)

print()

print(
    "interperson overlap px:",
    int(overlap.sum())
)

print(
    "A occluded by B px:",
    int(a_by_b.sum())
)

print(
    "B occluded by A px:",
    int(b_by_a.sum())
)

print()

print(
    "general occlusion px:",
    int(occ.sum())
)

print(
    "depth-gradient nonzero px:",
    int(gm.sum())
)

print()

print(
    "saved:",
    OUT
)
