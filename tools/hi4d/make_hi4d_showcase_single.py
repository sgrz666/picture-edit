#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image


PROJECT = Path(
    "/home/shangguanrz/project/pic-edit"
)

ROOT = (
    PROJECT
    / "datasets/Hi4D_pilot_v1"
)

BASE = (
    ROOT
    / "native_interaction_v2"
)

PAIR_FILE = (
    BASE
    / "pairs_train_N2C_main.jsonl"
)

FRAME_FILE = (
    BASE
    / "frames_used_with_dwpose_v2.jsonl"
)

OUT_ROOT = (
    ROOT
    / "showcase_single"
)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Runtime renderer
# ============================================================

sys.path.insert(
    0,
    str(PROJECT)
)

from hi4d_runtime_geometry import (
    Hi4DRuntimeGeometry
)


# ============================================================
# DeepGen official fix_pixels
# ============================================================

DEEPGEN = (
    PROJECT
    / "deepgen"
)

sys.path.insert(
    0,
    str(DEEPGEN)
)

try:

    from configs.datasets.deepgen_512_fix_pixels.processors import (
        image_size,
        image_process,
    )

    from src.datasets.utils import (
        resize_image_fix_pixels
    )

    USE_DEEPGEN_SIZE = True

except Exception as e:

    print(
        "WARNING: DeepGen size import failed:",
        e
    )

    USE_DEEPGEN_SIZE = False


# ============================================================
# Basic IO
# ============================================================

def load_jsonl(path):

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if line.strip():

                rows.append(
                    json.loads(line)
                )

    return rows


pairs = load_jsonl(
    PAIR_FILE
)

frames = load_jsonl(
    FRAME_FILE
)


def frame_key(
    pair,
    action,
    camera,
    frame_id,
):

    return (
        str(pair),
        str(action),
        int(camera),
        int(frame_id),
    )


frame_map = {}

for r in frames:

    frame_map[
        frame_key(
            r["pair"],
            r["action"],
            r["camera_id"],
            r["frame_id"],
        )
    ] = r


# ============================================================
# Select 4 representative interactions
# ============================================================

def action_family(action):

    return re.sub(
        r"\d+$",
        "",
        action.lower()
    )


priority = [
    "hug",
    "highfive",
    "fight",
    "dance",
    "talk",
]


by_family = defaultdict(
    list
)

for p in pairs:

    by_family[
        action_family(
            p["action"]
        )
    ].append(p)


for family in by_family:

    by_family[
        family
    ].sort(
        key=lambda x:
            int(
                x.get(
                    "target_contact_vertex_count_total",
                    0
                )
            ),
        reverse=True,
    )


selected = []

used_sequence = set()


for family in priority:

    if family not in by_family:
        continue

    for p in by_family[
        family
    ]:

        seq = (
            p["pair"],
            p["action"]
        )

        if seq in used_sequence:
            continue

        selected.append(p)

        used_sequence.add(
            seq
        )

        break

    if len(selected) >= 4:
        break


if len(selected) < 4:

    remaining = sorted(
        pairs,
        key=lambda x:
            int(
                x.get(
                    "target_contact_vertex_count_total",
                    0
                )
            ),
        reverse=True,
    )

    for p in remaining:

        seq = (
            p["pair"],
            p["action"]
        )

        if seq in used_sequence:
            continue

        selected.append(p)

        used_sequence.add(
            seq
        )

        if len(selected) >= 4:
            break


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--index",
    type=int,
    required=True,
    choices=[
        1,
        2,
        3,
        4,
    ],
    help="Showcase sample number 1-4",
)

args = parser.parse_args()

pair_row = selected[
    args.index - 1
]


# ============================================================
# Resolve source / target
# ============================================================

source_key = frame_key(
    pair_row["pair"],
    pair_row["action"],
    pair_row["camera_id"],
    pair_row["source_frame_id"],
)

target_key = frame_key(
    pair_row["pair"],
    pair_row["action"],
    pair_row["camera_id"],
    pair_row["target_frame_id"],
)


source_row = frame_map[
    source_key
]

target_row = frame_map[
    target_key
]


# ============================================================
# Output size
# ============================================================

def get_output_size(
    rgb_path
):

    if USE_DEEPGEN_SIZE:

        image = (
            Image.open(
                rgb_path
            )
            .convert(
                "RGB"
            )
        )

        out = (
            resize_image_fix_pixels(
                image,
                image_size=image_size,
                unit_image_size=32,
            )
        )

        return (
            out.width,
            out.height,
        )

    # fallback only
    return (
        448,
        608,
    )


W, H = get_output_size(
    target_row["rgb"]
)


print()
print(
    "===================================="
)

print(
    "SHOWCASE SAMPLE",
    args.index
)

print(
    "===================================="
)

print(
    "pair/action:",
    pair_row["pair"],
    pair_row["action"],
)

print(
    "source -> target:",
    pair_row[
        "source_frame_id"
    ],
    "->",
    pair_row[
        "target_frame_id"
    ],
)

print(
    "contact vertices:",
    pair_row.get(
        "target_contact_vertex_count_total",
        "N/A"
    ),
)

print(
    "display/render size:",
    f"{W}x{H}"
)


# ============================================================
# RGB
# ============================================================

def load_rgb(
    path
):

    img = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR
    )

    if img is None:

        raise RuntimeError(
            path
        )

    # display resize only
    # no crop
    img = cv2.resize(
        img,
        (
            W,
            H
        ),
        interpolation=cv2.INTER_AREA,
    )

    return img


source_rgb = load_rgb(
    source_row["rgb"]
)

target_rgb = load_rgb(
    target_row["rgb"]
)


# ============================================================
# DWPose
# ============================================================

BODY_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (1, 5),
    (5, 6),
    (6, 7),

    (1, 8),

    (8, 9),
    (9, 10),

    (8, 12),
    (12, 13),
    (13, 14),

    (0, 15),
    (0, 16),
]


def load_pose(
    path
):

    with np.load(
        path,
        allow_pickle=True
    ) as d:

        xy = np.asarray(
            d[
                "xy_normalized"
            ],
            dtype=np.float32
        )

        score = np.asarray(
            d["scores"],
            dtype=np.float32
        )

    return (
        xy,
        score
    )


def draw_pose_person(
    canvas,
    pose_path,
    color,
    threshold=0.30,
):

    xy, score = load_pose(
        pose_path
    )

    xy = xy.copy()

    xy[:, 0] *= W
    xy[:, 1] *= H


    for a, b in BODY_EDGES:

        if (
            a >= len(xy)
            or
            b >= len(xy)
        ):
            continue

        if (
            score[a] < threshold
            or
            score[b] < threshold
        ):
            continue

        p1 = tuple(
            np.round(
                xy[a]
            ).astype(int)
        )

        p2 = tuple(
            np.round(
                xy[b]
            ).astype(int)
        )

        cv2.line(
            canvas,
            p1,
            p2,
            color,
            4,
            cv2.LINE_AA,
        )


    for i in range(
        min(
            18,
            len(xy)
        )
    ):

        if (
            score[i]
            < threshold
        ):
            continue

        p = tuple(
            np.round(
                xy[i]
            ).astype(int)
        )

        cv2.circle(
            canvas,
            p,
            5,
            color,
            -1,
            cv2.LINE_AA,
        )


pose_img = np.zeros(
    (
        H,
        W,
        3
    ),
    dtype=np.uint8
)

# A = red
draw_pose_person(
    pose_img,
    target_row["dwpose_A"],
    (
        40,
        40,
        245
    ),
)

# B = blue
draw_pose_person(
    pose_img,
    target_row["dwpose_B"],
    (
        245,
        110,
        35
    ),
)


# ============================================================
# Runtime geometry
# ============================================================

renderer = (
    Hi4DRuntimeGeometry()
)

g = renderer.render(
    smpl_file=
        target_row[
            "official_smpl"
        ],

    camera_file=
        target_row[
            "camera_file"
        ],

    camera_id=
        int(
            target_row[
                "camera_id"
            ]
        ),

    out_h=H,
    out_w=W,

    native_h=1280,
    native_w=940,
)


# ============================================================
# Depth
# ============================================================

def make_depth_vis(
    depth
):

    depth = np.asarray(
        depth,
        dtype=np.float32
    )

    valid = (
        depth > 0
    )

    gray = np.zeros(
        (
            H,
            W
        ),
        dtype=np.uint8
    )

    if np.any(
        valid
    ):

        values = depth[
            valid
        ]

        lo = float(
            np.percentile(
                values,
                1
            )
        )

        hi = float(
            np.percentile(
                values,
                99
            )
        )

        hi = max(
            hi,
            lo + 1e-6
        )

        norm = np.zeros_like(
            depth,
            dtype=np.float32
        )

        norm[
            valid
        ] = (
            1.0
            -
            np.clip(
                (
                    depth[valid]
                    - lo
                )
                /
                (
                    hi
                    - lo
                ),
                0,
                1
            )
        )

        gray[
            valid
        ] = (
            norm[
                valid
            ]
            * 255
        ).astype(
            np.uint8
        )


    color = cv2.applyColorMap(
        gray,
        cv2.COLORMAP_TURBO
    )

    # background strictly black
    color[
        ~valid
    ] = 0

    return color


depth_img = make_depth_vis(
    g["depth_scene"]
)


# ============================================================
# Normal
# ============================================================

normal = np.asarray(
    g["normal_scene"],
    dtype=np.float32
)

normal_mask = (
    g["depth_scene"] > 0
)

normal_rgb = (
    (
        np.clip(
            normal,
            -1,
            1
        )
        + 1.0
    )
    * 127.5
).astype(
    np.uint8
)

normal_rgb[
    ~normal_mask
] = 0

normal_img = (
    normal_rgb[
        :,
        :,
        ::-1
    ]
)


# ============================================================
# Semantic — BLACK BACKGROUND
# ============================================================

semantic_A = (
    np.asarray(
        g["semantic_A"]
    ) > 0
)

semantic_B = (
    np.asarray(
        g["semantic_B"]
    ) > 0
)

semantic_img = np.zeros(
    (
        H,
        W,
        3
    ),
    dtype=np.uint8
)

# A red
semantic_img[
    semantic_A
] = (
    40,
    40,
    240
)

# B blue
semantic_img[
    semantic_B
] = (
    240,
    110,
    35
)

# overlap yellow
semantic_img[
    semantic_A
    &
    semantic_B
] = (
    30,
    220,
    240
)


# ============================================================
# Contact — BLACK BACKGROUND
# ============================================================

contact_A = np.asarray(
    g["contact_A"],
    dtype=np.float32
)

contact_B = np.asarray(
    g["contact_B"],
    dtype=np.float32
)


# Heatmap intensity, not RGB overlay
a_strength = np.clip(
    contact_A,
    0,
    1
)

b_strength = np.clip(
    contact_B,
    0,
    1
)


contact_img = np.zeros(
    (
        H,
        W,
        3
    ),
    dtype=np.uint8
)


# A: red channel
contact_img[
    :,
    :,
    2
] = (
    a_strength
    * 255
).astype(
    np.uint8
)


# B: blue channel
contact_img[
    :,
    :,
    0
] = (
    b_strength
    * 255
).astype(
    np.uint8
)


# both strong -> yellow-ish
both = (
    (a_strength > 0.05)
    &
    (b_strength > 0.05)
)

contact_img[
    both
] = (
    20,
    220,
    245
)


# ============================================================
# Directional Occlusion — BLACK BACKGROUND
# ============================================================

A_by_B = (
    np.asarray(
        g[
            "A_occluded_by_B"
        ]
    ) > 0
)

B_by_A = (
    np.asarray(
        g[
            "B_occluded_by_A"
        ]
    ) > 0
)


occlusion_img = np.zeros(
    (
        H,
        W,
        3
    ),
    dtype=np.uint8
)


# A hidden by B -> RED
occlusion_img[
    A_by_B
] = (
    40,
    40,
    245
)


# B hidden by A -> BLUE
occlusion_img[
    B_by_A
] = (
    245,
    110,
    35
)


# ============================================================
# Save assets
# ============================================================

sample_name = (
    f"sample_{args.index:02d}_"
    f"{pair_row['pair']}_"
    f"{pair_row['action']}_"
    f"{int(pair_row['source_frame_id']):06d}_to_"
    f"{int(pair_row['target_frame_id']):06d}"
)

OUT_DIR = (
    OUT_ROOT
    / sample_name
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


assets = [
    (
        "01_source_rgb.png",
        source_rgb
    ),

    (
        "02_target_rgb.png",
        target_rgb
    ),

    (
        "03_target_dwpose.png",
        pose_img
    ),

    (
        "04_target_depth.png",
        depth_img
    ),

    (
        "05_target_normal.png",
        normal_img
    ),

    (
        "06_target_semantic.png",
        semantic_img
    ),

    (
        "07_target_contact.png",
        contact_img
    ),

    (
        "08_target_occlusion.png",
        occlusion_img
    ),
]


for name, image in assets:

    cv2.imwrite(
        str(
            OUT_DIR
            / name
        ),
        image
    )


# ============================================================
# Build ONE 2x4 panel
# ============================================================

DISPLAY_W = 300


def resize_cell(
    img
):

    h, w = img.shape[:2]

    ratio = (
        DISPLAY_W
        / float(w)
    )

    nh = int(
        round(
            h * ratio
        )
    )

    return cv2.resize(
        img,
        (
            DISPLAY_W,
            nh
        ),
        interpolation=cv2.INTER_AREA
    )


def add_title(
    img,
    title
):

    img = img.copy()

    bar_h = 52

    canvas = np.zeros(
        (
            img.shape[0]
            + bar_h,
            img.shape[1],
            3
        ),
        dtype=np.uint8
    )

    canvas[
        bar_h:
    ] = img

    cv2.putText(
        canvas,
        title,
        (
            12,
            34
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (
            255,
            255,
            255
        ),
        2,
        cv2.LINE_AA
    )

    return canvas


titles = [
    "Source RGB",
    "Target RGB",
    "Target DWPose",
    "Target Depth",

    "Target Normal",
    "Person Semantic",
    "Contact",
    "Directional Occlusion",
]


cells = []

for (
    (_, image),
    title
) in zip(
    assets,
    titles
):

    cell = resize_cell(
        image
    )

    cell = add_title(
        cell,
        title
    )

    cells.append(
        cell
    )


row1 = np.hstack(
    cells[:4]
)

row2 = np.hstack(
    cells[4:]
)


header_h = 82

header = np.zeros(
    (
        header_h,
        row1.shape[1],
        3
    ),
    dtype=np.uint8
)


header_text = (
    f"{pair_row['pair']} / "
    f"{pair_row['action']}    "
    f"Source {pair_row['source_frame_id']} "
    f"-> Target {pair_row['target_frame_id']}    "
    f"Contact vertices: "
    f"{pair_row.get('target_contact_vertex_count_total', 'N/A')}"
)


cv2.putText(
    header,
    header_text,
    (
        18,
        48
    ),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.82,
    (
        255,
        255,
        255
    ),
    2,
    cv2.LINE_AA
)


panel = np.vstack(
    [
        header,
        row1,
        row2
    ]
)


panel_path = (
    OUT_DIR
    / "SHOWCASE_PANEL.jpg"
)


cv2.imwrite(
    str(
        panel_path
    ),
    panel,
    [
        cv2.IMWRITE_JPEG_QUALITY,
        96
    ]
)


# ============================================================
# Metadata
# ============================================================

metadata = {
    "sample_index":
        args.index,

    "pair":
        pair_row[
            "pair"
        ],

    "action":
        pair_row[
            "action"
        ],

    "source_frame":
        int(
            pair_row[
                "source_frame_id"
            ]
        ),

    "target_frame":
        int(
            pair_row[
                "target_frame_id"
            ]
        ),

    "target_contact_vertices":
        int(
            pair_row.get(
                "target_contact_vertex_count_total",
                0
            )
        ),

    "render_width":
        W,

    "render_height":
        H,

    "visualization": {
        "semantic":
            "black bg; A=red, B=blue, overlap=yellow",

        "contact":
            "black bg; A=red heatmap, B=blue heatmap, overlap=yellow",

        "occlusion":
            "black bg; A_occluded_by_B=red, B_occluded_by_A=blue",
    }
}


with open(
    OUT_DIR
    / "metadata.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False
    )


print()
print(
    "===================================="
)

print(
    "DONE"
)

print(
    "===================================="
)

print(
    "folder:",
    OUT_DIR
)

print(
    "panel:",
    panel_path
)
