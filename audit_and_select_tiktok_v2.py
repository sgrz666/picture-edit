#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TikTok V2 frame audit + Source/Target selection.

Principles
----------
1. Do NOT rerun SMPLest-X.
2. Reuse existing TikTok_3d_assets_native assets.
3. Quality filtering before pair construction.
4. Source emphasizes appearance quality.
5. Target emphasizes pose diversity.
6. Full / near-full body preferred for MAIN.
7. No fixed 30fps dense sampling.
8. Existing original data and old assets are never modified.

Outputs
-------
TikTok_v2/
├── manifests/
│   ├── frames_all_qc.csv
│   ├── selected_unique.jsonl
│   ├── sources.jsonl
│   ├── targets.jsonl
│   ├── pairs_candidate.jsonl
│   └── summary.json
└── qc/
    ├── selected/
    ├── rejected/
    └── borderline/
"""

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
except Exception:
    torch = None


PROJECT = Path("/home/shangguanrz/project/pic-edit")

ROOT = (
    PROJECT
    / "datasets"
    / "TikTokDataset"
)

RAW_ROOT = (
    ROOT
    / "TikTok_dataset"
    / "TikTok_dataset"
)

ASSET_ROOT = (
    ROOT
    / "TikTok_3d_assets_native"
)

OUT_ROOT = (
    ROOT
    / "TikTok_v2"
)


# ============================================================
# Basic IO
# ============================================================

IMG_EXTS = [
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
]


def find_by_stem(folder, stem):

    folder = Path(folder)

    for ext in IMG_EXTS:

        p = folder / f"{stem}{ext}"

        if p.exists():
            return p

    return None


def numeric_frame_id(stem):

    try:
        return int(stem)
    except Exception:
        return None


def read_gray(path):

    if path is None:
        return None

    return cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )


def read_color(path):

    if path is None:
        return None

    return cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )


# ============================================================
# Mask statistics
# ============================================================

def mask_stats(mask):

    H, W = mask.shape

    fg = mask > 127

    n = int(fg.sum())

    if n == 0:

        return {
            "valid":
                False,

            "area_ratio":
                0.0,

            "bbox_h_ratio":
                0.0,

            "bbox_w_ratio":
                0.0,

            "touch_top":
                True,

            "touch_bottom":
                True,

            "touch_left":
                True,

            "touch_right":
                True,
        }


    ys, xs = np.where(
        fg
    )


    x0 = int(
        xs.min()
    )

    x1 = int(
        xs.max()
    )

    y0 = int(
        ys.min()
    )

    y1 = int(
        ys.max()
    )


    margin_y = max(
        2,
        int(
            round(
                H * 0.015
            )
        )
    )

    margin_x = max(
        2,
        int(
            round(
                W * 0.015
            )
        )
    )


    return {
        "valid":
            True,

        "area_ratio":
            float(
                n
                /
                (H * W)
            ),

        "bbox_h_ratio":
            float(
                (y1 - y0 + 1)
                /
                H
            ),

        "bbox_w_ratio":
            float(
                (x1 - x0 + 1)
                /
                W
            ),

        "touch_top":
            y0 <= margin_y,

        "touch_bottom":
            y1 >= (
                H
                - 1
                - margin_y
            ),

        "touch_left":
            x0 <= margin_x,

        "touch_right":
            x1 >= (
                W
                - 1
                - margin_x
            ),
    }


# ============================================================
# Person-region sharpness
# ============================================================

def person_sharpness(
    image,
    mask,
):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )


    fg = (
        mask > 127
    ).astype(
        np.uint8
    )


    kernel = np.ones(
        (5, 5),
        dtype=np.uint8,
    )


    interior = cv2.erode(
        fg,
        kernel,
        iterations=1,
    ) > 0


    if interior.sum() < 100:

        interior = (
            fg > 0
        )


    lap = cv2.Laplacian(
        gray,
        cv2.CV_32F,
    )


    lap_values = lap[
        interior
    ]


    if len(
        lap_values
    ) == 0:

        lap_var = 0.0

    else:

        lap_var = float(
            np.var(
                lap_values
            )
        )


    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )


    mag = np.sqrt(
        gx * gx
        +
        gy * gy
    )


    if interior.sum() > 0:

        tenengrad = float(
            np.mean(
                mag[
                    interior
                ]
            )
        )

    else:

        tenengrad = 0.0


    # Only used for ranking.
    sharpness = (
        math.log1p(
            max(
                0.0,
                tenengrad
            )
        )
        +
        0.25
        *
        math.log1p(
            max(
                0.0,
                lap_var
            )
        )
    )


    return (
        lap_var,
        tenengrad,
        sharpness,
    )


# ============================================================
# Person mask vs SMPL-X rendered mask
# ============================================================

def mask_iou(
    person_mask,
    smplx_mask,
):

    if (
        person_mask is None
        or
        smplx_mask is None
    ):

        return None


    H, W = person_mask.shape


    if smplx_mask.shape != (
        H,
        W
    ):

        smplx_mask = cv2.resize(
            smplx_mask,
            (
                W,
                H,
            ),
            interpolation=cv2.INTER_NEAREST,
        )


    a = (
        person_mask > 127
    )

    b = (
        smplx_mask > 127
    )


    union = int(
        np.logical_or(
            a,
            b
        ).sum()
    )


    if union == 0:
        return 0.0


    inter = int(
        np.logical_and(
            a,
            b
        ).sum()
    )


    return float(
        inter
        /
        union
    )


# ============================================================
# Frame map
# ============================================================

def stem_of(value):

    try:
        return Path(
            str(value)
        ).stem
    except Exception:
        return str(value)


def parse_frame_map_object(
    obj,
):

    result = {}


    if isinstance(
        obj,
        dict
    ):

        # stem -> index
        for k, v in obj.items():

            if isinstance(
                v,
                (int, np.integer)
            ):

                result[
                    stem_of(k)
                ] = int(v)


        # index -> path
        for k, v in obj.items():

            if (
                str(k).isdigit()
                and
                isinstance(
                    v,
                    str
                )
            ):

                result[
                    stem_of(v)
                ] = int(k)


        # Common nested/list structures.
        for key in [
            "frames",
            "frame_paths",
            "image_paths",
            "images",
            "stems",
            "frame_names",
        ]:

            if key not in obj:
                continue

            value = obj[
                key
            ]

            if isinstance(
                value,
                list
            ):

                for i, item in enumerate(
                    value
                ):

                    if isinstance(
                        item,
                        str
                    ):

                        result[
                            stem_of(
                                item
                            )
                        ] = i

                    elif isinstance(
                        item,
                        dict
                    ):

                        path = (
                            item.get(
                                "path"
                            )
                            or
                            item.get(
                                "image"
                            )
                            or
                            item.get(
                                "image_path"
                            )
                            or
                            item.get(
                                "frame"
                            )
                            or
                            item.get(
                                "name"
                            )
                            or
                            item.get(
                                "stem"
                            )
                        )

                        idx = item.get(
                            "index",
                            item.get(
                                "idx",
                                i
                            )
                        )

                        if path is not None:

                            result[
                                stem_of(
                                    path
                                )
                            ] = int(
                                idx
                            )


    elif isinstance(
        obj,
        list
    ):

        for i, item in enumerate(
            obj
        ):

            if isinstance(
                item,
                str
            ):

                result[
                    stem_of(
                        item
                    )
                ] = i

            elif isinstance(
                item,
                dict
            ):

                path = (
                    item.get(
                        "path"
                    )
                    or
                    item.get(
                        "image"
                    )
                    or
                    item.get(
                        "image_path"
                    )
                    or
                    item.get(
                        "frame"
                    )
                    or
                    item.get(
                        "name"
                    )
                    or
                    item.get(
                        "stem"
                    )
                )

                idx = item.get(
                    "index",
                    item.get(
                        "idx",
                        i
                    )
                )

                if path is not None:

                    result[
                        stem_of(
                            path
                        )
                    ] = int(
                        idx
                    )


    return result


# ============================================================
# SMPLest-X params -> pose features
# ============================================================

def tensor_to_numpy(x):

    if x is None:
        return None

    if hasattr(
        x,
        "detach"
    ):

        x = (
            x.detach()
            .cpu()
            .numpy()
        )

    return np.asarray(
        x
    )


def load_pose_bundle(
    seq_assets,
    frame_stems,
):

    if torch is None:

        return {
            "available":
                False,

            "reason":
                "torch unavailable",
        }


    smpl_root = (
        seq_assets
        / "smplestx"
    )

    params_path = (
        smpl_root
        / "params.pt"
    )

    fmap_path = (
        smpl_root
        / "frame_map.json"
    )


    if not params_path.exists():

        return {
            "available":
                False,

            "reason":
                "params.pt missing",
        }


    try:

        params = torch.load(
            params_path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:

        params = torch.load(
            params_path,
            map_location="cpu",
        )


    if not isinstance(
        params,
        dict
    ):

        return {
            "available":
                False,

            "reason":
                "params.pt is not dict",
        }


    joint_proj = tensor_to_numpy(
        params.get(
            "joint_proj_model"
        )
    )

    body_pose = tensor_to_numpy(
        params.get(
            "body_pose"
        )
    )


    N = None

    if (
        joint_proj is not None
        and
        joint_proj.ndim >= 3
    ):

        N = int(
            joint_proj.shape[0]
        )

    elif (
        body_pose is not None
        and
        body_pose.ndim >= 2
    ):

        N = int(
            body_pose.shape[0]
        )


    mapping = {}

    mapping_source = "none"


    if fmap_path.exists():

        try:

            with open(
                fmap_path,
                "r",
                encoding="utf-8",
            ) as f:

                fmap_obj = json.load(
                    f
                )

            mapping = parse_frame_map_object(
                fmap_obj
            )

            if mapping:
                mapping_source = (
                    "frame_map.json"
                )

        except Exception:
            pass


    if (
        not mapping
        and
        N is not None
        and
        N == len(
            frame_stems
        )
    ):

        mapping = {
            stem: i
            for i, stem
            in enumerate(
                frame_stems
            )
        }

        mapping_source = (
            "sorted_frame_fallback"
        )


    return {
        "available":
            bool(
                mapping
            )
            and
            N is not None,

        "reason":
            "",

        "joint_proj":
            joint_proj,

        "body_pose":
            body_pose,

        "mapping":
            mapping,

        "mapping_source":
            mapping_source,

        "N":
            N,

        "params_path":
            str(
                params_path
            ),

        "frame_map_path":
            str(
                fmap_path
            ),
    }


def pose_feature(
    bundle,
    stem,
):

    if not bundle.get(
        "available",
        False
    ):

        return None


    idx = bundle[
        "mapping"
    ].get(
        stem
    )


    if idx is None:
        return None


    joint_proj = bundle.get(
        "joint_proj"
    )


    if (
        joint_proj is not None
        and
        0 <= idx < joint_proj.shape[0]
    ):

        pts = np.asarray(
            joint_proj[
                idx,
                :min(
                    22,
                    joint_proj.shape[1]
                ),
                :2
            ],
            dtype=np.float32,
        )


        finite = np.isfinite(
            pts
        ).all(
            axis=1
        )


        if finite.sum() >= 8:

            valid = pts[
                finite
            ]


            center = np.median(
                valid,
                axis=0,
            )


            valid_centered = (
                valid
                -
                center
            )


            scale = float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            valid_centered
                            *
                            valid_centered,
                            axis=1,
                        )
                    )
                )
            )


            scale = max(
                scale,
                1e-6,
            )


            out = np.zeros_like(
                pts,
                dtype=np.float32,
            )


            out[
                finite
            ] = (
                pts[
                    finite
                ]
                -
                center
            ) / scale


            return out.reshape(
                -1
            )


    body_pose = bundle.get(
        "body_pose"
    )


    if (
        body_pose is not None
        and
        0 <= idx < body_pose.shape[0]
    ):

        x = np.asarray(
            body_pose[
                idx
            ],
            dtype=np.float32,
        )

        if np.isfinite(
            x
        ).all():

            return x


    return None


# ============================================================
# Full / near-full heuristic
# ============================================================

def completeness_class(
    stats,
    min_target_height,
):

    h = stats[
        "bbox_h_ratio"
    ]


    top = stats[
        "touch_top"
    ]

    bottom = stats[
        "touch_bottom"
    ]


    # This is deliberately conservative.
    # It is a screening heuristic, not a GT full-body label.

    if (
        h >= min_target_height
        and
        not top
        and
        not bottom
    ):

        return "FULL"


    if (
        h >= min_target_height
        and
        not (
            top
            and
            bottom
        )
    ):

        return "NEAR_FULL"


    return "PARTIAL"


# ============================================================
# Sequence percentile ranks
# ============================================================

def percentile_ranks(
    values,
):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    order = np.argsort(
        values
    )

    ranks = np.empty(
        len(values),
        dtype=np.float64,
    )


    if len(
        values
    ) <= 1:

        ranks[:] = 1.0
        return ranks


    ranks[
        order
    ] = np.linspace(
        0.0,
        1.0,
        len(values),
    )

    return ranks


# ============================================================
# Temporal thinning
# ============================================================

def temporal_thin(
    rows,
    min_gap,
):

    rows = sorted(
        rows,
        key=lambda r: (
            r[
                "frame_num"
            ]
            if r[
                "frame_num"
            ]
            is not None
            else
            r[
                "order_idx"
            ]
        )
    )


    keep = []

    last_value = None


    for r in rows:

        value = (
            r[
                "frame_num"
            ]
            if r[
                "frame_num"
            ]
            is not None
            else
            r[
                "order_idx"
            ]
        )


        if (
            last_value is None
            or
            value
            -
            last_value
            >=
            min_gap
        ):

            keep.append(
                r
            )

            last_value = (
                value
            )


    return keep


# ============================================================
# Farthest point sampling
# ============================================================

def fps_select(
    rows,
    max_n,
):

    if len(rows) <= max_n:
        return list(rows)


    features = []

    valid_rows = []


    for r in rows:

        feat = r.get(
            "_pose_feature"
        )

        if feat is None:
            continue

        if not np.isfinite(
            feat
        ).all():
            continue

        features.append(
            feat
        )

        valid_rows.append(
            r
        )


    if len(
        valid_rows
    ) < max(
        8,
        min(
            max_n,
            len(rows)
        )
        // 3
    ):

        # Fallback: evenly cover timeline.
        idxs = np.linspace(
            0,
            len(rows) - 1,
            max_n,
        ).round().astype(
            int
        )

        return [
            rows[i]
            for i in sorted(
                set(
                    idxs.tolist()
                )
            )
        ]


    X = np.stack(
        features,
        axis=0,
    ).astype(
        np.float32
    )


    # Normalize feature dimensions.
    std = X.std(
        axis=0,
        keepdims=True
    )

    std[
        std < 1e-6
    ] = 1.0

    X = (
        X
        -
        X.mean(
            axis=0,
            keepdims=True
        )
    ) / std


    start = int(
        np.argmax(
            [
                r[
                    "quality_score"
                ]
                for r
                in valid_rows
            ]
        )
    )


    selected = [
        start
    ]


    min_dist = np.full(
        len(valid_rows),
        np.inf,
        dtype=np.float32,
    )


    for _ in range(
        1,
        min(
            max_n,
            len(valid_rows)
        )
    ):

        last = X[
            selected[-1]
        ]


        d = np.sum(
            (
                X
                -
                last
            )
            ** 2,
            axis=1,
        )


        min_dist = np.minimum(
            min_dist,
            d,
        )


        min_dist[
            selected
        ] = -1.0


        nxt = int(
            np.argmax(
                min_dist
            )
        )


        if min_dist[
            nxt
        ] < 0:
            break


        selected.append(
            nxt
        )


    return [
        valid_rows[i]
        for i in selected
    ]


# ============================================================
# Source selection
# ============================================================

def select_sources(
    candidates,
    max_sources,
    min_gap,
    min_source_height,
):

    preferred = [
        r
        for r
        in candidates
        if
        r[
            "bbox_h_ratio"
        ]
        >=
        min_source_height
        and
        r[
            "completeness"
        ]
        ==
        "FULL"
    ]


    if len(
        preferred
    ) < max_sources:

        preferred = [
            r
            for r
            in candidates
            if
            r[
                "bbox_h_ratio"
            ]
            >=
            min_source_height
            and
            r[
                "completeness"
            ]
            in [
                "FULL",
                "NEAR_FULL",
            ]
        ]


    preferred = sorted(
        preferred,
        key=lambda r: r[
            "quality_score"
        ],
        reverse=True,
    )


    chosen = []


    for r in preferred:

        value = (
            r[
                "frame_num"
            ]
            if r[
                "frame_num"
            ]
            is not None
            else r[
                "order_idx"
            ]
        )


        good_gap = True


        for x in chosen:

            xv = (
                x[
                    "frame_num"
                ]
                if x[
                    "frame_num"
                ]
                is not None
                else x[
                    "order_idx"
                ]
            )

            if abs(
                value
                -
                xv
            ) < min_gap:

                good_gap = False
                break


        if good_gap:

            chosen.append(
                r
            )


        if len(
            chosen
        ) >= max_sources:
            break


    # Fallback if timeline spacing was too strict.
    if len(
        chosen
    ) < max_sources:

        used = {
            r["stem"]
            for r
            in chosen
        }

        for r in preferred:

            if r[
                "stem"
            ] in used:
                continue

            chosen.append(
                r
            )

            used.add(
                r["stem"]
            )

            if len(
                chosen
            ) >= max_sources:
                break


    return chosen


# ============================================================
# Preview
# ============================================================

def fit_panel(
    image,
    out_w=220,
    out_h=360,
):

    if image is None:

        return np.zeros(
            (
                out_h,
                out_w,
                3,
            ),
            dtype=np.uint8,
        )


    H, W = image.shape[
        :2
    ]


    scale = min(
        out_w / W,
        out_h / H,
    )


    nw = max(
        1,
        int(
            round(
                W * scale
            )
        )
    )

    nh = max(
        1,
        int(
            round(
                H * scale
            )
        )
    )


    resized = cv2.resize(
        image,
        (
            nw,
            nh,
        ),
        interpolation=cv2.INTER_AREA,
    )


    canvas = np.zeros(
        (
            out_h,
            out_w,
            3,
        ),
        dtype=np.uint8,
    )


    x = (
        out_w
        -
        nw
    ) // 2

    y = (
        out_h
        -
        nh
    ) // 2


    canvas[
        y:y + nh,
        x:x + nw,
    ] = resized


    return canvas


def preview_image(
    row,
    out_path,
):

    rgb = read_color(
        row[
            "rgb"
        ]
    )


    mask = read_gray(
        row[
            "mask"
        ]
    )


    smpl_mask = read_gray(
        row[
            "smplx_mask"
        ]
        if row[
            "smplx_mask"
        ]
        else None
    )


    normal = read_color(
        row[
            "normal"
        ]
        if row[
            "normal"
        ]
        else None
    )


    if mask is not None:

        mask_rgb = cv2.cvtColor(
            mask,
            cv2.COLOR_GRAY2BGR,
        )

    else:

        mask_rgb = None


    if smpl_mask is not None:

        smpl_rgb = cv2.cvtColor(
            smpl_mask,
            cv2.COLOR_GRAY2BGR,
        )

    else:

        smpl_rgb = None


    tiles = [
        fit_panel(
            rgb
        ),

        fit_panel(
            mask_rgb
        ),

        fit_panel(
            smpl_rgb
        ),

        fit_panel(
            normal
        ),
    ]


    canvas = np.concatenate(
        tiles,
        axis=1,
    )


    header_h = 82


    full = np.zeros(
        (
            canvas.shape[0]
            +
            header_h,
            canvas.shape[1],
            3,
        ),
        dtype=np.uint8,
    )


    full[
        header_h:
    ] = canvas


    line1 = (
        f"{row['sequence']} / "
        f"{row['stem']} | "
        f"{row['completeness']} | "
        f"Q={row['quality_score']:.3f}"
    )


    line2 = (
        f"blur={row['sharpness']:.3f}  "
        f"mask={row['mask_area_ratio']:.3f}  "
        f"h={row['bbox_h_ratio']:.3f}  "
        f"IoU={row['smplx_iou']:.3f}"
        if row[
            "smplx_iou"
        ]
        is not None
        else
        f"blur={row['sharpness']:.3f}  "
        f"mask={row['mask_area_ratio']:.3f}  "
        f"h={row['bbox_h_ratio']:.3f}  "
        f"IoU=N/A"
    )


    cv2.putText(
        full,
        line1,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


    cv2.putText(
        full,
        line2,
        (12, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )


    labels = [
        "RGB",
        "Person Mask",
        "SMPL-X Mask",
        "Normal",
    ]


    for i, label in enumerate(
        labels
    ):

        cv2.putText(
            full,
            label,
            (
                i * 220
                +
                8,
                header_h
                +
                24,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    cv2.imwrite(
        str(
            out_path
        ),
        full,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            88,
        ],
    )


# ============================================================
# JSON helper
# ============================================================

def public_row(
    row,
):

    return {
        k: v
        for k, v
        in row.items()
        if not k.startswith(
            "_"
        )
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--max-sources",
        type=int,
        default=6,
    )


    parser.add_argument(
        "--max-targets",
        type=int,
        default=64,
    )


    parser.add_argument(
        "--min-target-gap",
        type=int,
        default=3,
    )


    parser.add_argument(
        "--min-source-gap",
        type=int,
        default=15,
    )


    parser.add_argument(
        "--blur-quantile",
        type=float,
        default=0.10,
    )


    parser.add_argument(
        "--iou-floor",
        type=float,
        default=0.35,
    )


    parser.add_argument(
        "--min-mask-area",
        type=float,
        default=0.015,
    )


    parser.add_argument(
        "--max-mask-area",
        type=float,
        default=0.85,
    )


    parser.add_argument(
        "--min-target-height",
        type=float,
        default=0.35,
    )


    parser.add_argument(
        "--min-source-height",
        type=float,
        default=0.45,
    )


    parser.add_argument(
        "--preview-n",
        type=int,
        default=100,
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )


    args = parser.parse_args()


    if not RAW_ROOT.exists():

        raise FileNotFoundError(
            RAW_ROOT
        )


    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    manifest_root = (
        OUT_ROOT
        / "manifests"
    )

    qc_root = (
        OUT_ROOT
        / "qc"
    )


    manifest_root.mkdir(
        parents=True,
        exist_ok=True,
    )


    qc_root.mkdir(
        parents=True,
        exist_ok=True,
    )


    seq_dirs = [
        p
        for p
        in sorted(
            RAW_ROOT.iterdir()
        )
        if p.is_dir()
        and
        (
            p / "images"
        ).exists()
    ]


    print()
    print("=" * 80)
    print("TikTok V2 AUDIT + SELECTION")
    print("=" * 80)

    print(
        "raw root  :",
        RAW_ROOT
    )

    print(
        "asset root:",
        ASSET_ROOT
    )

    print(
        "sequences :",
        len(
            seq_dirs
        )
    )


    all_rows = []

    sequence_bundles = {}


    # ========================================================
    # PASS 1
    # ========================================================

    for sidx, seq_dir in enumerate(
        seq_dirs,
        1,
    ):

        seq = seq_dir.name

        image_dir = (
            seq_dir
            / "images"
        )

        mask_dir = (
            seq_dir
            / "masks"
        )


        images = []

        for ext in IMG_EXTS:

            images.extend(
                image_dir.glob(
                    f"*{ext}"
                )
            )


        images = sorted(
            set(
                images
            ),
            key=lambda p: p.name,
        )


        stems = [
            p.stem
            for p
            in images
        ]


        seq_assets = (
            ASSET_ROOT
            / seq
        )


        bundle = load_pose_bundle(
            seq_assets,
            stems,
        )


        sequence_bundles[
            seq
        ] = bundle


        for order_idx, rgb_path in enumerate(
            images
        ):

            stem = rgb_path.stem


            mask_path = find_by_stem(
                mask_dir,
                stem,
            )


            normal_path = find_by_stem(
                seq_assets
                / "normal",
                stem,
            )


            depth_vis_path = find_by_stem(
                seq_assets
                / "depth_vis",
                stem,
            )


            raw_depth_path = (
                find_by_stem(
                    seq_assets
                    / "depth",
                    stem,
                )
                if (
                    seq_assets
                    / "depth"
                ).exists()
                else None
            )


            smplx_mask_path = find_by_stem(
                seq_assets
                / "smplx_mask",
                stem,
            )


            image = read_color(
                rgb_path
            )


            mask = read_gray(
                mask_path
            )


            if (
                image is None
                or
                mask is None
            ):

                row = {
                    "sequence":
                        seq,

                    "stem":
                        stem,

                    "order_idx":
                        order_idx,

                    "frame_num":
                        numeric_frame_id(
                            stem
                        ),

                    "rgb":
                        str(
                            rgb_path
                        ),

                    "mask":
                        (
                            str(
                                mask_path
                            )
                            if mask_path
                            else ""
                        ),

                    "decode_ok":
                        False,
                }

                all_rows.append(
                    row
                )

                continue


            H, W = image.shape[
                :2
            ]


            if mask.shape != (
                H,
                W
            ):

                mask = cv2.resize(
                    mask,
                    (
                        W,
                        H,
                    ),
                    interpolation=cv2.INTER_NEAREST,
                )


            mstats = mask_stats(
                mask
            )


            lap_var, tenengrad, sharpness = (
                person_sharpness(
                    image,
                    mask,
                )
            )


            smpl_mask = read_gray(
                smplx_mask_path
            )


            iou = mask_iou(
                mask,
                smpl_mask,
            )


            feat = pose_feature(
                bundle,
                stem,
            )


            row = {
                "sequence":
                    seq,

                "stem":
                    stem,

                "order_idx":
                    order_idx,

                "frame_num":
                    numeric_frame_id(
                        stem
                    ),

                "width":
                    W,

                "height":
                    H,

                "rgb":
                    str(
                        rgb_path
                    ),

                "mask":
                    (
                        str(
                            mask_path
                        )
                        if mask_path
                        else ""
                    ),

                "normal":
                    (
                        str(
                            normal_path
                        )
                        if normal_path
                        else ""
                    ),

                "depth_vis":
                    (
                        str(
                            depth_vis_path
                        )
                        if depth_vis_path
                        else ""
                    ),

                "raw_depth":
                    (
                        str(
                            raw_depth_path
                        )
                        if raw_depth_path
                        else ""
                    ),

                "smplx_mask":
                    (
                        str(
                            smplx_mask_path
                        )
                        if smplx_mask_path
                        else ""
                    ),

                "smplestx_params":
                    bundle.get(
                        "params_path",
                        ""
                    ),

                "smplestx_mapping_source":
                    bundle.get(
                        "mapping_source",
                        ""
                    ),

                "decode_ok":
                    True,

                "mask_valid":
                    bool(
                        mstats[
                            "valid"
                        ]
                    ),

                "mask_area_ratio":
                    mstats[
                        "area_ratio"
                    ],

                "bbox_h_ratio":
                    mstats[
                        "bbox_h_ratio"
                    ],

                "bbox_w_ratio":
                    mstats[
                        "bbox_w_ratio"
                    ],

                "touch_top":
                    mstats[
                        "touch_top"
                    ],

                "touch_bottom":
                    mstats[
                        "touch_bottom"
                    ],

                "touch_left":
                    mstats[
                        "touch_left"
                    ],

                "touch_right":
                    mstats[
                        "touch_right"
                    ],

                "lap_var":
                    lap_var,

                "tenengrad":
                    tenengrad,

                "sharpness":
                    sharpness,

                "smplx_iou":
                    iou,

                "has_normal":
                    normal_path is not None,

                "has_depth_vis":
                    depth_vis_path is not None,

                "has_raw_depth":
                    raw_depth_path is not None,

                "has_smplx_mask":
                    smplx_mask_path is not None,

                "has_smplestx_pose":
                    feat is not None,

                "_pose_feature":
                    feat,
            }


            row[
                "completeness"
            ] = completeness_class(
                row,
                args.min_target_height,
            )


            all_rows.append(
                row
            )


        if (
            sidx % 20 == 0
            or
            sidx == len(
                seq_dirs
            )
        ):

            print(
                f"[PASS1] "
                f"{sidx}/"
                f"{len(seq_dirs)}"
            )


    good_rows = [
        r
        for r
        in all_rows
        if r.get(
            "decode_ok",
            False
        )
    ]


    sharp_values = np.asarray(
        [
            r[
                "sharpness"
            ]
            for r
            in good_rows
        ],
        dtype=np.float64,
    )


    iou_values = np.asarray(
        [
            r[
                "smplx_iou"
            ]
            for r
            in good_rows
            if r.get(
                "smplx_iou"
            )
            is not None
        ],
        dtype=np.float64,
    )


    global_blur_hard = float(
        np.quantile(
            sharp_values,
            0.02,
        )
    )


    if len(
        iou_values
    ):

        global_iou_q05 = float(
            np.quantile(
                iou_values,
                0.05,
            )
        )

        iou_threshold = max(
            args.iou_floor,
            global_iou_q05,
        )

    else:

        global_iou_q05 = None

        iou_threshold = (
            args.iou_floor
        )


    # ========================================================
    # PASS 2: adaptive thresholds + quality score
    # ========================================================

    by_seq = defaultdict(
        list
    )


    for r in good_rows:

        by_seq[
            r[
                "sequence"
            ]
        ].append(
            r
        )


    for seq, rows in by_seq.items():

        seq_sharp = np.asarray(
            [
                r[
                    "sharpness"
                ]
                for r
                in rows
            ],
            dtype=np.float64,
        )


        seq_blur_thr = float(
            np.quantile(
                seq_sharp,
                args.blur_quantile,
            )
        )


        sharp_rank = percentile_ranks(
            seq_sharp
        )


        for rank, r in zip(
            sharp_rank,
            rows
        ):

            r[
                "sharpness_rank"
            ] = float(
                rank
            )


            r[
                "blur_pass"
            ] = bool(
                r[
                    "sharpness"
                ]
                >=
                seq_blur_thr
                and
                r[
                    "sharpness"
                ]
                >=
                global_blur_hard
            )


            r[
                "mask_pass"
            ] = bool(
                r[
                    "mask_valid"
                ]
                and
                args.min_mask_area
                <=
                r[
                    "mask_area_ratio"
                ]
                <=
                args.max_mask_area
            )


            r[
                "size_pass"
            ] = bool(
                r[
                    "bbox_h_ratio"
                ]
                >=
                args.min_target_height
            )


            r[
                "geometry_present"
            ] = bool(
                r[
                    "has_normal"
                ]
                and
                r[
                    "has_smplx_mask"
                ]
            )


            r[
                "smplx_iou_pass"
            ] = bool(
                r[
                    "smplx_iou"
                ]
                is not None
                and
                r[
                    "smplx_iou"
                ]
                >=
                iou_threshold
            )


            completeness_score = {
                "FULL":
                    1.0,

                "NEAR_FULL":
                    0.75,

                "PARTIAL":
                    0.20,

            }[
                r[
                    "completeness"
                ]
            ]


            iou_score = (
                float(
                    r[
                        "smplx_iou"
                    ]
                )
                if r[
                    "smplx_iou"
                ]
                is not None
                else 0.0
            )


            r[
                "quality_score"
            ] = float(
                0.45
                *
                r[
                    "sharpness_rank"
                ]
                +
                0.35
                *
                iou_score
                +
                0.20
                *
                completeness_score
            )


            r[
                "main_candidate"
            ] = bool(
                r[
                    "blur_pass"
                ]
                and
                r[
                    "mask_pass"
                ]
                and
                r[
                    "size_pass"
                ]
                and
                r[
                    "geometry_present"
                ]
                and
                r[
                    "smplx_iou_pass"
                ]
                and
                r[
                    "completeness"
                ]
                in [
                    "FULL",
                    "NEAR_FULL",
                ]
            )


            reasons = []


            for name in [
                "blur_pass",
                "mask_pass",
                "size_pass",
                "geometry_present",
                "smplx_iou_pass",
            ]:

                if not r[
                    name
                ]:

                    reasons.append(
                        name
                    )


            if r[
                "completeness"
            ] == "PARTIAL":

                reasons.append(
                    "partial_body"
                )


            r[
                "reject_reasons"
            ] = ",".join(
                reasons
            )


    # ========================================================
    # Select sources / targets
    # ========================================================

    source_rows = []

    target_rows = []

    pair_rows = []


    for seq in sorted(
        by_seq.keys()
    ):

        seq_rows = by_seq[
            seq
        ]


        candidates = [
            r
            for r
            in seq_rows
            if r[
                "main_candidate"
            ]
        ]


        if not candidates:
            continue


        sources = select_sources(
            candidates,
            args.max_sources,
            args.min_source_gap,
            args.min_source_height,
        )


        source_stems = {
            r[
                "stem"
            ]
            for r
            in sources
        }


        target_pool = [
            r
            for r
            in candidates
            if r[
                "stem"
            ]
            not in source_stems
        ]


        target_pool = temporal_thin(
            target_pool,
            args.min_target_gap,
        )


        targets = fps_select(
            target_pool,
            args.max_targets,
        )


        for r in sources:

            rr = public_row(
                r
            )

            rr[
                "role"
            ] = "source"

            source_rows.append(
                rr
            )


        for r in targets:

            rr = public_row(
                r
            )

            rr[
                "role"
            ] = "target"

            target_rows.append(
                rr
            )


        # Candidate pairing only.
        # Final pairing can be frozen after QC.
        for s in sources:

            for t in targets:

                if s[
                    "stem"
                ] == t[
                    "stem"
                ]:
                    continue


                pose_distance = None


                sf = s.get(
                    "_pose_feature"
                )

                tf = t.get(
                    "_pose_feature"
                )


                if (
                    sf is not None
                    and
                    tf is not None
                    and
                    sf.shape
                    ==
                    tf.shape
                ):

                    pose_distance = float(
                        np.linalg.norm(
                            sf
                            -
                            tf
                        )
                    )


                pair_rows.append(
                    {
                        "sequence":
                            seq,

                        "source_stem":
                            s[
                                "stem"
                            ],

                        "target_stem":
                            t[
                                "stem"
                            ],

                        "source_rgb":
                            s[
                                "rgb"
                            ],

                        "target_rgb":
                            t[
                                "rgb"
                            ],

                        "source_mask":
                            s[
                                "mask"
                            ],

                        "target_mask":
                            t[
                                "mask"
                            ],

                        "target_normal":
                            t[
                                "normal"
                            ],

                        "target_depth_vis":
                            t[
                                "depth_vis"
                            ],

                        "target_raw_depth":
                            t[
                                "raw_depth"
                            ],

                        "target_smplx_mask":
                            t[
                                "smplx_mask"
                            ],

                        "smplestx_params":
                            t[
                                "smplestx_params"
                            ],

                        "pose_distance":
                            pose_distance,

                        "pair_status":
                            "CANDIDATE_NOT_FROZEN",
                    }
                )


    # ========================================================
    # Unique selected
    # ========================================================

    selected_map = {}


    for r in (
        source_rows
        +
        target_rows
    ):

        key = (
            r[
                "sequence"
            ],
            r[
                "stem"
            ],
        )

        if key not in selected_map:

            selected_map[
                key
            ] = dict(
                r
            )

            selected_map[
                key
            ][
                "roles"
            ] = [
                r[
                    "role"
                ]
            ]

        else:

            role = r[
                "role"
            ]

            if role not in (
                selected_map[
                    key
                ][
                    "roles"
                ]
            ):

                selected_map[
                    key
                ][
                    "roles"
                ].append(
                    role
                )


    selected_unique = list(
        selected_map.values()
    )


    # ========================================================
    # Save manifests
    # ========================================================

    csv_rows = [
        public_row(
            r
        )
        for r
        in all_rows
    ]


    csv_fields = sorted(
        {
            k
            for r
            in csv_rows
            for k
            in r.keys()
        }
    )


    with open(
        manifest_root
        / "frames_all_qc.csv",
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=csv_fields,
        )

        writer.writeheader()

        writer.writerows(
            csv_rows
        )


    def write_jsonl(
        name,
        rows,
    ):

        path = (
            manifest_root
            / name
        )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as f:

            for r in rows:

                f.write(
                    json.dumps(
                        r,
                        ensure_ascii=False,
                    )
                    +
                    "\n"
                )


    write_jsonl(
        "sources.jsonl",
        source_rows,
    )

    write_jsonl(
        "targets.jsonl",
        target_rows,
    )

    write_jsonl(
        "selected_unique.jsonl",
        selected_unique,
    )

    write_jsonl(
        "pairs_candidate.jsonl",
        pair_rows,
    )


    # ========================================================
    # Summary
    # ========================================================

    complete_counter = Counter(
        r.get(
            "completeness",
            "INVALID"
        )
        for r
        in good_rows
    )


    reuse = {
        "selected_total":
            len(
                selected_unique
            ),

        "with_normal":
            sum(
                bool(
                    r.get(
                        "has_normal"
                    )
                )
                for r
                in selected_unique
            ),

        "with_depth_vis":
            sum(
                bool(
                    r.get(
                        "has_depth_vis"
                    )
                )
                for r
                in selected_unique
            ),

        "with_raw_depth":
            sum(
                bool(
                    r.get(
                        "has_raw_depth"
                    )
                )
                for r
                in selected_unique
            ),

        "with_smplx_mask":
            sum(
                bool(
                    r.get(
                        "has_smplx_mask"
                    )
                )
                for r
                in selected_unique
            ),

        "with_smplestx_pose":
            sum(
                bool(
                    r.get(
                        "has_smplestx_pose"
                    )
                )
                for r
                in selected_unique
            ),
    }


    summary = {
        "version":
            "tiktok_v2_audit_select",

        "raw_root":
            str(
                RAW_ROOT
            ),

        "existing_asset_root":
            str(
                ASSET_ROOT
            ),

        "selection_policy": {
            "source":
                (
                    "appearance quality first; "
                    "prefer full-body; "
                    "temporal spacing"
                ),

            "target":
                (
                    "quality pass + "
                    "full/near-full + "
                    "temporal thinning + "
                    "SMPLest-X pose FPS"
                ),

            "smplestx_rerun":
                False,

            "pair_status":
                "candidate_not_frozen",
        },

        "counts": {
            "sequences":
                len(
                    seq_dirs
                ),

            "raw_frames":
                len(
                    all_rows
                ),

            "decoded_frames":
                len(
                    good_rows
                ),

            "main_candidates":
                sum(
                    bool(
                        r.get(
                            "main_candidate"
                        )
                    )
                    for r
                    in good_rows
                ),

            "sources":
                len(
                    source_rows
                ),

            "targets":
                len(
                    target_rows
                ),

            "selected_unique":
                len(
                    selected_unique
                ),

            "candidate_pairs":
                len(
                    pair_rows
                ),
        },

        "completeness":
            dict(
                complete_counter
            ),

        "thresholds": {
            "global_blur_hard_p02":
                global_blur_hard,

            "per_sequence_blur_quantile":
                args.blur_quantile,

            "global_smplx_iou_p05":
                global_iou_q05,

            "effective_smplx_iou_threshold":
                iou_threshold,

            "min_mask_area":
                args.min_mask_area,

            "max_mask_area":
                args.max_mask_area,

            "min_target_bbox_height":
                args.min_target_height,

            "min_source_bbox_height":
                args.min_source_height,

            "min_target_temporal_gap":
                args.min_target_gap,

            "min_source_temporal_gap":
                args.min_source_gap,
        },

        "reuse_existing_assets":
            reuse,
    }


    with open(
        manifest_root
        / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )


    # ========================================================
    # Preview
    # ========================================================

    rng = random.Random(
        args.seed
    )


    selected_lookup = {
        (
            r[
                "sequence"
            ],
            r[
                "stem"
            ],
        )
        for r
        in selected_unique
    }


    selected_preview = [
        r
        for r
        in good_rows
        if (
            r[
                "sequence"
            ],
            r[
                "stem"
            ],
        )
        in selected_lookup
    ]


    if len(
        selected_preview
    ) > args.preview_n:

        selected_preview = rng.sample(
            selected_preview,
            args.preview_n,
        )


    rejected = sorted(
        [
            r
            for r
            in good_rows
            if not r.get(
                "main_candidate",
                False
            )
        ],
        key=lambda r: r.get(
            "quality_score",
            0.0
        ),
    )[
        :args.preview_n
    ]


    borderline = sorted(
        [
            r
            for r
            in good_rows
            if r.get(
                "main_candidate",
                False
            )
        ],
        key=lambda r: abs(
            (
                r.get(
                    "smplx_iou",
                    iou_threshold
                )
                or
                0.0
            )
            -
            iou_threshold
        ),
    )[
        :args.preview_n
    ]


    for group_name, rows in [
        (
            "selected",
            selected_preview,
        ),
        (
            "rejected",
            rejected,
        ),
        (
            "borderline",
            borderline,
        ),
    ]:

        out_dir = (
            qc_root
            / group_name
        )


        for r in rows:

            preview_image(
                r,
                out_dir
                /
                (
                    f"{r['sequence']}_"
                    f"{r['stem']}.jpg"
                ),
            )


    print()
    print("=" * 80)
    print("TikTok V2 AUDIT COMPLETE")
    print("=" * 80)

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print(
        "OUTPUT:",
        OUT_ROOT
    )


if __name__ == "__main__":
    main()
