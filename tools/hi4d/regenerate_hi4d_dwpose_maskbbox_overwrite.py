#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Regenerate Hi4D DWPose with official Hi4D instance-mask bounding boxes.

IMPORTANT
---------
1. Uses the official DWPose / RTMPose ONNX pose estimator.
2. Does NOT rely on YOLO person association for Hi4D.
3. person A bbox comes from official person_mask_A.
4. person B bbox comes from official person_mask_B.
5. Directly overwrites existing dwpose_A / dwpose_B NPZ.
6. Rewrites existing Target DWPose QC visualizations.
7. Raw NPZ retains ALL whole-body keypoints.
8. Visualization draws body + hands only for clarity.
"""

import json
import sys
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# ============================================================
# PATHS
# ============================================================

PROJECT = Path(
    "/home/shangguanrz/project/pic-edit"
)

ROOT = (
    PROJECT
    / "datasets"
    / "Hi4D_pilot_v1"
)

FRAME_MANIFEST = (
    ROOT
    / "native_interaction_v2"
    / "frames_used_with_dwpose_v2.jsonl"
)

V42_ROOT = (
    ROOT
    / "native_interaction_v4_2"
)

FINAL_NATIVE = (
    ROOT
    / "final_assets_native_v1"
)

TOOL_ROOT = (
    PROJECT
    / "add"
    / "tools"
    / "dwpose_mimicmotion_official"
)

SCORE_THR = 0.30


# ============================================================
# IO
# ============================================================

def read_jsonl(path):
    rows = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


# ============================================================
# Locate official DWPose implementation + checkpoint
# ============================================================

def find_unique(root, filename):

    hits = list(
        root.rglob(filename)
    )

    if not hits:
        raise FileNotFoundError(
            f"Cannot find {filename} under {root}"
        )

    print(
        f"[FOUND] {filename}:",
        hits[0]
    )

    return hits[0]


POSE_MODEL = find_unique(
    TOOL_ROOT,
    "dw-ll_ucoco_384.onnx",
)

ONNXPOSE_FILE = find_unique(
    TOOL_ROOT,
    "onnxpose.py",
)


# ============================================================
# Dynamically load official onnxpose.py
# ============================================================

spec = importlib.util.spec_from_file_location(
    "hi4d_dwpose_onnxpose",
    str(ONNXPOSE_FILE),
)

onnxpose = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    onnxpose
)


# ============================================================
# ONNX Runtime session
# ============================================================

available = (
    ort.get_available_providers()
)

providers = []

if (
    "CUDAExecutionProvider"
    in available
):
    providers.append(
        "CUDAExecutionProvider"
    )

providers.append(
    "CPUExecutionProvider"
)


print()
print(
    "ONNX providers:",
    providers
)


pose_session = ort.InferenceSession(
    str(POSE_MODEL),
    providers=providers,
)


# ============================================================
# Mask -> bbox
# ============================================================

def load_mask_bbox(
    path,
    width,
    height,
):

    mask = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        raise RuntimeError(
            f"Cannot load mask: {path}"
        )

    ys, xs = np.where(
        mask > 0
    )

    if len(xs) == 0:
        raise RuntimeError(
            f"Empty mask: {path}"
        )

    x0 = int(
        xs.min()
    )

    x1 = int(
        xs.max()
    ) + 1

    y0 = int(
        ys.min()
    )

    y1 = int(
        ys.max()
    ) + 1


    # Very small safety expansion.
    #
    # DWPose's official pose preprocessing itself
    # already uses padding=1.25, so we do NOT
    # aggressively enlarge the bbox here.
    bw = x1 - x0
    bh = y1 - y0

    mx = int(
        round(
            bw * 0.03
        )
    )

    my = int(
        round(
            bh * 0.03
        )
    )


    x0 = max(
        0,
        x0 - mx,
    )

    y0 = max(
        0,
        y0 - my,
    )

    x1 = min(
        width,
        x1 + mx,
    )

    y1 = min(
        height,
        y1 + my,
    )


    return [
        x0,
        y0,
        x1,
        y1,
    ]


# ============================================================
# DWPose official OpenPose-style reordering
#
# Reproduce the official Wholebody post-processing.
# ============================================================

def dwpose_postprocess(
    keypoints,
    scores,
):

    keypoints_info = np.concatenate(
        [
            keypoints,
            scores[..., None],
        ],
        axis=-1,
    )


    # --------------------------------------------------------
    # Add neck:
    # mean of left/right shoulder.
    # --------------------------------------------------------

    neck = np.mean(
        keypoints_info[
            :,
            [5, 6],
        ],
        axis=1,
    )


    neck[
        :,
        2
    ] = np.logical_and(
        keypoints_info[
            :,
            5,
            2
        ]
        > SCORE_THR,

        keypoints_info[
            :,
            6,
            2
        ]
        > SCORE_THR,
    ).astype(
        np.float32
    )


    new_info = np.insert(
        keypoints_info,
        17,
        neck,
        axis=1,
    )


    # Official DWPose OpenPose ordering.
    mmpose_idx = [
        17,
        6,
        8,
        10,
        7,
        9,
        12,
        14,
        16,
        13,
        15,
        2,
        1,
        4,
        3,
    ]

    openpose_idx = [
        1,
        2,
        3,
        4,
        6,
        7,
        8,
        9,
        10,
        12,
        13,
        14,
        15,
        16,
        17,
    ]


    new_info[
        :,
        openpose_idx
    ] = new_info[
        :,
        mmpose_idx
    ]


    xy = new_info[
        ...,
        :2
    ].astype(
        np.float32
    )

    sc = new_info[
        ...,
        2
    ].astype(
        np.float32
    )


    return (
        xy,
        sc,
    )


# ============================================================
# Atomic NPZ overwrite
# ============================================================

def save_pose_npz(
    path,
    xy,
    scores,
    width,
    height,
):

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    xy_norm = xy.copy()

    xy_norm[
        :,
        0
    ] /= float(
        width
    )

    xy_norm[
        :,
        1
    ] /= float(
        height
    )


    tmp = path.with_suffix(
        ".tmp.npz"
    )


    with open(
        tmp,
        "wb",
    ) as f:

        np.savez_compressed(
            f,
            xy_native=xy.astype(
                np.float32
            ),
            xy_normalized=xy_norm.astype(
                np.float32
            ),
            scores=scores.astype(
                np.float32
            ),
            width=np.int32(
                width
            ),
            height=np.int32(
                height
            ),
            generator=np.array(
                "Hi4D_mask_bbox_DWPose_official"
            ),
        )


    tmp.replace(
        path
    )


# ============================================================
# Visualization
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


HAND_EDGES_LOCAL = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


# After neck insertion:
#
# body     0..17
# foot    18..23
# face    24..91
# hand L  92..112
# hand R  113..133
LEFT_HAND_START = 92
RIGHT_HAND_START = 113


COLOR_A = (
    245,
    120,
    40,
)  # BGR blue-ish

COLOR_B = (
    40,
    40,
    245,
)  # BGR red


def valid_point(
    xy,
    scores,
    idx,
):

    if idx >= len(
        scores
    ):
        return False

    if scores[
        idx
    ] < SCORE_THR:
        return False

    x, y = xy[
        idx
    ]

    if (
        x < 0
        or
        y < 0
    ):
        return False

    return True


def draw_edge(
    canvas,
    xy,
    scores,
    i,
    j,
    color,
    thickness=2,
):

    if not (
        valid_point(
            xy,
            scores,
            i,
        )
        and
        valid_point(
            xy,
            scores,
            j,
        )
    ):
        return


    p1 = tuple(
        np.round(
            xy[i]
        ).astype(
            int
        )
    )

    p2 = tuple(
        np.round(
            xy[j]
        ).astype(
            int
        )
    )


    cv2.line(
        canvas,
        p1,
        p2,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_person(
    canvas,
    xy,
    scores,
    color,
):

    # Body.
    for i, j in BODY_EDGES:

        draw_edge(
            canvas,
            xy,
            scores,
            i,
            j,
            color,
            thickness=3,
        )


    # Body points.
    for i in range(
        min(
            18,
            len(scores),
        )
    ):

        if not valid_point(
            xy,
            scores,
            i,
        ):
            continue

        p = tuple(
            np.round(
                xy[i]
            ).astype(
                int
            )
        )

        cv2.circle(
            canvas,
            p,
            3,
            color,
            -1,
            cv2.LINE_AA,
        )


    # Hands.
    for start in [
        LEFT_HAND_START,
        RIGHT_HAND_START,
    ]:

        if (
            start + 20
            >=
            len(scores)
        ):
            continue


        for li, lj in HAND_EDGES_LOCAL:

            draw_edge(
                canvas,
                xy,
                scores,
                start + li,
                start + lj,
                color,
                thickness=1,
            )


        for k in range(
            start,
            start + 21,
        ):

            if not valid_point(
                xy,
                scores,
                k,
            ):
                continue

            p = tuple(
                np.round(
                    xy[k]
                ).astype(
                    int
                )
            )

            cv2.circle(
                canvas,
                p,
                1,
                color,
                -1,
                cv2.LINE_AA,
            )


    # Intentionally do NOT draw dense face points here.
    # Raw NPZ still preserves all whole-body keypoints.


# ============================================================
# Target keys that need native DWPose visualization
# ============================================================

target_keys = set()


for manifest_name in [
    "main_targets_unique.jsonl",
    "aux_pending_c2c_dance_targets.jsonl",
]:

    path = (
        V42_ROOT
        / manifest_name
    )

    if not path.exists():
        continue


    for r in read_jsonl(
        path
    ):

        target_keys.add(
            (
                str(
                    r["pair"]
                ),
                str(
                    r["action"]
                ),
                int(
                    r["camera_id"]
                ),
                int(
                    r["frame_id"]
                ),
            )
        )


print(
    "Target visualization keys:",
    len(
        target_keys
    )
)


# ============================================================
# Frames
# ============================================================

rows = read_jsonl(
    FRAME_MANIFEST
)


# De-duplicate.
frame_rows = {}


for r in rows:

    key = (
        str(
            r["pair"]
        ),
        str(
            r["action"]
        ),
        int(
            r["camera_id"]
        ),
        int(
            r["frame_id"]
        ),
    )

    frame_rows[
        key
    ] = r


rows = [
    frame_rows[k]
    for k in sorted(
        frame_rows
    )
]


print()
print("=" * 80)
print("REGENERATE Hi4D DWPose")
print("=" * 80)

print(
    "unique frames:",
    len(rows)
)

print(
    "method:",
    "Hi4D official masks -> bbox -> official DWPose RTMPose"
)

print(
    "overwrite:",
    True
)


# ============================================================
# Main loop
# ============================================================

failed = []


for idx, r in enumerate(
    rows,
    start=1,
):

    try:

        pair = str(
            r["pair"]
        )

        action = str(
            r["action"]
        )

        cam = int(
            r["camera_id"]
        )

        fid = int(
            r["frame_id"]
        )


        rgb_path = Path(
            r["rgb"]
        )

        mask_a_path = Path(
            r[
                "person_mask_A"
            ]
        )

        mask_b_path = Path(
            r[
                "person_mask_B"
            ]
        )


        image = cv2.imread(
            str(
                rgb_path
            ),
            cv2.IMREAD_COLOR,
        )


        if image is None:

            raise RuntimeError(
                f"cannot read RGB: "
                f"{rgb_path}"
            )


        H, W = image.shape[
            :2
        ]


        bbox_a = load_mask_bbox(
            mask_a_path,
            W,
            H,
        )

        bbox_b = load_mask_bbox(
            mask_b_path,
            W,
            H,
        )


        # ----------------------------------------------------
        # IMPORTANT:
        # bbox order is [A, B].
        # inference_pose keeps bbox order.
        # Therefore identity is fixed by official Hi4D mask.
        # ----------------------------------------------------

        bboxes = np.asarray(
            [
                bbox_a,
                bbox_b,
            ],
            dtype=np.float32,
        )


        keypoints, scores = (
            onnxpose.inference_pose(
                pose_session,
                bboxes,
                image,
            )
        )


        if keypoints.shape[
            0
        ] != 2:

            raise RuntimeError(
                f"expected 2 pose outputs, "
                f"got {keypoints.shape}"
            )


        xy, sc = dwpose_postprocess(
            keypoints,
            scores,
        )


        xy_a = xy[
            0
        ]

        sc_a = sc[
            0
        ]


        xy_b = xy[
            1
        ]

        sc_b = sc[
            1
        ]


        # ----------------------------------------------------
        # Directly overwrite existing NPZ.
        # ----------------------------------------------------

        save_pose_npz(
            r[
                "dwpose_A"
            ],
            xy_a,
            sc_a,
            W,
            H,
        )

        save_pose_npz(
            r[
                "dwpose_B"
            ],
            xy_b,
            sc_b,
            W,
            H,
        )


        # ----------------------------------------------------
        # Rewrite native Target visualization.
        # ----------------------------------------------------

        target_key = (
            pair,
            action,
            cam,
            fid,
        )


        if target_key in target_keys:

            stem = str(
                r.get(
                    "frame_stem",
                    f"{fid:06d}",
                )
            )


            vis_path = (
                FINAL_NATIVE
                / pair
                / action
                / f"cam{cam}"
                / "vis"
                / "dwpose"
                / f"{stem}.png"
            )


            vis_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )


            canvas = np.zeros(
                (
                    H,
                    W,
                    3,
                ),
                dtype=np.uint8,
            )


            draw_person(
                canvas,
                xy_a,
                sc_a,
                COLOR_A,
            )

            draw_person(
                canvas,
                xy_b,
                sc_b,
                COLOR_B,
            )


            cv2.imwrite(
                str(
                    vis_path
                ),
                canvas,
            )


        if (
            idx % 50 == 0
            or
            idx == len(rows)
        ):

            print(
                f"[{idx}/{len(rows)}]"
            )


    except Exception as e:

        failed.append(
            {
                "pair":
                    r.get(
                        "pair"
                    ),

                "action":
                    r.get(
                        "action"
                    ),

                "camera_id":
                    r.get(
                        "camera_id"
                    ),

                "frame_id":
                    r.get(
                        "frame_id"
                    ),

                "error":
                    repr(e),
            }
        )

        print(
            "[FAILED]",
            failed[-1],
        )


# ============================================================
# Result
# ============================================================

log_path = (
    PROJECT
    / "logs"
    / "hi4d_dwpose_maskbbox_failures.json"
)

log_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with open(
    log_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        failed,
        f,
        ensure_ascii=False,
        indent=2,
    )


print()
print("=" * 80)
print("DWPose REGEN COMPLETE")
print("=" * 80)

print(
    "total :",
    len(rows)
)

print(
    "ok    :",
    len(rows)
    -
    len(failed)
)

print(
    "failed:",
    len(failed)
)

print(
    "failure log:",
    log_path
)


if failed:

    raise SystemExit(
        2
    )
