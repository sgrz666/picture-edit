#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TikTok V3 task-aware frame selection.

Stage:
    PRE-DWPose selection

Inputs:
    1. Existing 92,961 RGB frames
    2. Original person masks
    3. Existing SMPLest-X params / geometry
    4. Existing normal / SMPL-X mask assets

Design:
    - Source: appearance quality first
    - Target: pose/expression diversity first
    - Full body is NOT mandatory
    - Half-body is allowed when person is large/clear and pose is meaningful
    - Reject tiny person / severe blur / invalid masks / bad SMPL-X alignment
    - Use existing SMPLest-X; DO NOT rerun it
    - Do NOT freeze final pairs yet
    - Selected frames will later receive final DWPose QC

Outputs:
    TikTok_v3/manifests/
        frames_scored.jsonl
        sources_pre_dwpose.jsonl
        targets_pre_dwpose.jsonl
        selected_pre_dwpose.jsonl
        summary_pre_dwpose.json
"""

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


PROJECT = Path(
    "/home/shangguanrz/project/pic-edit"
)

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

V3_ROOT = (
    ROOT
    / "TikTok_v3"
)

INVENTORY = (
    V3_ROOT
    / "manifests"
    / "frame_inventory.jsonl"
)

MANIFEST_ROOT = (
    V3_ROOT
    / "manifests"
)

QC_ROOT = (
    V3_ROOT
    / "qc_pre_dwpose"
)


# ============================================================
# Basic IO
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


def write_jsonl(
    path,
    rows,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                +
                "\n"
            )


def find_frame_file(
    folder,
    stem,
):

    for ext in [
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    ]:

        p = (
            folder
            /
            f"{stem}{ext}"
        )

        if p.exists():
            return p

    return None


# ============================================================
# Mask / image statistics
# ============================================================

def load_mask(
    seq,
    stem,
):

    path = find_frame_file(
        RAW_ROOT
        / seq
        / "masks",
        stem,
    )

    if path is None:
        return None, None

    mask = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    return path, mask


def mask_geometry(
    mask,
):

    fg = (
        mask > 127
    )

    H, W = mask.shape

    if fg.sum() == 0:
        return None


    ys, xs = np.where(
        fg
    )


    x0 = int(xs.min())
    x1 = int(xs.max())

    y0 = int(ys.min())
    y1 = int(ys.max())


    bw = (
        x1 - x0 + 1
    )

    bh = (
        y1 - y0 + 1
    )


    margin_x = max(
        2,
        int(
            round(
                W * 0.01
            )
        ),
    )

    margin_y = max(
        2,
        int(
            round(
                H * 0.01
            )
        ),
    )


    return {
        "x0":
            x0,

        "x1":
            x1,

        "y0":
            y0,

        "y1":
            y1,

        "bbox_h_ratio":
            float(
                bh / H
            ),

        "bbox_w_ratio":
            float(
                bw / W
            ),

        "mask_area_ratio":
            float(
                fg.mean()
            ),

        "touch_top":
            bool(
                y0
                <=
                margin_y
            ),

        "touch_bottom":
            bool(
                y1
                >=
                H - 1 - margin_y
            ),

        "touch_left":
            bool(
                x0
                <=
                margin_x
            ),

        "touch_right":
            bool(
                x1
                >=
                W - 1 - margin_x
            ),
    }


def person_crop_sharpness(
    image,
    mask,
    geom,
):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )


    x0 = geom["x0"]
    x1 = geom["x1"] + 1

    y0 = geom["y0"]
    y1 = geom["y1"] + 1


    crop_gray = gray[
        y0:y1,
        x0:x1,
    ]


    crop_mask = (
        mask[
            y0:y1,
            x0:x1
        ]
        > 127
    )


    if (
        crop_gray.size == 0
        or
        crop_mask.sum() < 100
    ):

        return 0.0, 0.0, 0.0


    # Avoid evaluating mask boundary itself.
    eroded = cv2.erode(
        crop_mask.astype(
            np.uint8
        ),
        np.ones(
            (5, 5),
            dtype=np.uint8,
        ),
        iterations=1,
    ) > 0


    if eroded.sum() < 100:

        eroded = crop_mask


    lap = cv2.Laplacian(
        crop_gray,
        cv2.CV_32F,
    )


    lap_var = float(
        np.var(
            lap[
                eroded
            ]
        )
    )


    gx = cv2.Sobel(
        crop_gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        crop_gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )


    tenengrad = float(
        np.mean(
            np.sqrt(
                gx * gx
                +
                gy * gy
            )[
                eroded
            ]
        )
    )


    # Robust combined ranking score.
    sharpness = (
        0.7
        *
        math.log1p(
            max(
                tenengrad,
                0.0
            )
        )
        +
        0.3
        *
        math.log1p(
            max(
                lap_var,
                0.0
            )
        )
    )


    return (
        lap_var,
        tenengrad,
        sharpness,
    )


# ============================================================
# Existing SMPL-X mask IoU
# ============================================================

def smplx_mask_iou(
    seq,
    stem,
    person_mask,
):

    path = find_frame_file(
        ASSET_ROOT
        / seq
        / "smplx_mask",
        stem,
    )


    if path is None:
        return None, None


    smpl_mask = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )


    if smpl_mask is None:
        return path, None


    H, W = person_mask.shape


    if smpl_mask.shape != (
        H,
        W
    ):

        smpl_mask = cv2.resize(
            smpl_mask,
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
        smpl_mask > 127
    )


    union = np.logical_or(
        a,
        b
    ).sum()


    if union == 0:
        return path, 0.0


    iou = float(
        np.logical_and(
            a,
            b
        ).sum()
        /
        union
    )


    return path, iou


# ============================================================
# SMPLest-X loading
# ============================================================

def parse_frame_map(
    obj,
):

    mapping = {}


    if isinstance(
        obj,
        dict
    ):

        for k, v in obj.items():

            if isinstance(
                v,
                int
            ):

                mapping[
                    Path(
                        str(k)
                    ).stem
                ] = v


        for key in [
            "frames",
            "frame_paths",
            "images",
            "image_paths",
            "stems",
            "frame_names",
        ]:

            value = obj.get(
                key
            )

            if not isinstance(
                value,
                list
            ):
                continue


            for i, x in enumerate(
                value
            ):

                if isinstance(
                    x,
                    str
                ):

                    mapping[
                        Path(
                            x
                        ).stem
                    ] = i


                elif isinstance(
                    x,
                    dict
                ):

                    name = (
                        x.get(
                            "path"
                        )
                        or
                        x.get(
                            "image"
                        )
                        or
                        x.get(
                            "image_path"
                        )
                        or
                        x.get(
                            "frame"
                        )
                        or
                        x.get(
                            "name"
                        )
                        or
                        x.get(
                            "stem"
                        )
                    )


                    idx = x.get(
                        "index",
                        x.get(
                            "idx",
                            i
                        )
                    )


                    if name is not None:

                        mapping[
                            Path(
                                str(
                                    name
                                )
                            ).stem
                        ] = int(
                            idx
                        )


    elif isinstance(
        obj,
        list
    ):

        for i, x in enumerate(
            obj
        ):

            if isinstance(
                x,
                str
            ):

                mapping[
                    Path(
                        x
                    ).stem
                ] = i


    return mapping


def load_smpl_bundle(
    seq,
    sorted_stems,
):

    root = (
        ASSET_ROOT
        / seq
        / "smplestx"
    )


    params_path = (
        root
        / "params.pt"
    )

    fmap_path = (
        root
        / "frame_map.json"
    )


    if not params_path.exists():

        return {
            "available":
                False
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
                False
        }


    def npv(
        key
    ):

        value = params.get(
            key
        )

        if value is None:
            return None

        if hasattr(
            value,
            "detach"
        ):

            value = (
                value.detach()
                .cpu()
                .numpy()
            )

        return np.asarray(
            value
        )


    joints = npv(
        "joint_proj_model"
    )

    body_pose = npv(
        "body_pose"
    )

    expression = npv(
        "expression"
    )


    n = None


    for x in [
        joints,
        body_pose,
        expression,
    ]:

        if (
            x is not None
            and
            x.ndim >= 2
        ):

            n = int(
                x.shape[0]
            )

            break


    mapping = {}


    if fmap_path.exists():

        try:

            with open(
                fmap_path,
                "r",
                encoding="utf-8",
            ) as f:

                mapping = parse_frame_map(
                    json.load(
                        f
                    )
                )

        except Exception:
            pass


    # Safe fallback only if sequence length exactly matches params length.
    if (
        not mapping
        and
        n is not None
        and
        n == len(
            sorted_stems
        )
    ):

        mapping = {
            stem: i
            for i, stem
            in enumerate(
                sorted_stems
            )
        }


    return {
        "available":
            bool(
                mapping
            )
            and
            n is not None,

        "mapping":
            mapping,

        "joints":
            joints,

        "body_pose":
            body_pose,

        "expression":
            expression,
    }


# ============================================================
# Pose and expression features
# ============================================================

def normalized_pose_feature(
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


    joints = bundle.get(
        "joints"
    )


    if (
        joints is not None
        and
        idx < joints.shape[0]
    ):

        # Use main whole-body joints only.
        # We do not assume face landmark topology here.
        J = np.asarray(
            joints[
                idx,
                :min(
                    25,
                    joints.shape[1]
                ),
                :2
            ],
            dtype=np.float32,
        )


        valid = np.isfinite(
            J
        ).all(
            axis=1
        )


        if valid.sum() >= 8:

            center = np.median(
                J[
                    valid
                ],
                axis=0,
            )


            centered = (
                J[
                    valid
                ]
                -
                center
            )


            scale = float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            centered
                            *
                            centered,
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
                J,
                dtype=np.float32,
            )


            out[
                valid
            ] = (
                J[
                    valid
                ]
                -
                center
            ) / scale


            return out.reshape(
                -1
            )


    pose = bundle.get(
        "body_pose"
    )


    if (
        pose is not None
        and
        idx < pose.shape[0]
    ):

        x = np.asarray(
            pose[
                idx
            ],
            dtype=np.float32,
        )

        if np.isfinite(
            x
        ).all():

            return x


    return None


def expression_feature(
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


    expression = bundle.get(
        "expression"
    )


    if (
        expression is None
        or
        idx is None
        or
        idx
        >=
        expression.shape[0]
    ):

        return None


    x = np.asarray(
        expression[
            idx
        ],
        dtype=np.float32,
    )


    if not np.isfinite(
        x
    ).all():

        return None


    return x


# ============================================================
# Ranking utilities
# ============================================================

def percentile_rank(
    x,
):

    x = np.asarray(
        x,
        dtype=np.float64,
    )


    if len(x) <= 1:

        return np.ones_like(
            x
        )


    order = np.argsort(
        x
    )


    ranks = np.empty(
        len(x),
        dtype=np.float64,
    )


    ranks[
        order
    ] = np.linspace(
        0.0,
        1.0,
        len(x),
    )


    return ranks


def frame_number(
    stem,
    fallback,
):

    try:
        return int(
            stem
        )

    except Exception:
        return fallback


# ============================================================
# Farthest Point Sampling
# ============================================================

def fps(
    rows,
    max_n,
    expression_weight=0.15,
):

    if len(rows) <= max_n:
        return list(rows)


    valid = []

    feats = []


    pose_dims = []


    for r in rows:

        p = r.get(
            "_pose"
        )

        if p is None:
            continue

        pose_dims.append(
            len(p)
        )


    if not pose_dims:
        return []


    pose_dim = min(
        pose_dims
    )


    for r in rows:

        p = r.get(
            "_pose"
        )

        if p is None:
            continue


        p = p[
            :pose_dim
        ].astype(
            np.float32
        )


        e = r.get(
            "_expression"
        )


        if e is not None:

            e = e.astype(
                np.float32
            )

            feat = np.concatenate(
                [
                    p,
                    e
                    *
                    expression_weight,
                ]
            )

        else:

            feat = p


        if not np.isfinite(
            feat
        ).all():

            continue


        valid.append(
            r
        )

        feats.append(
            feat
        )


    if len(valid) <= max_n:

        return valid


    # Expression may be unavailable for some frames.
    min_dim = min(
        len(x)
        for x
        in feats
    )


    X = np.stack(
        [
            x[
                :min_dim
            ]
            for x
            in feats
        ],
        axis=0,
    )


    mean = X.mean(
        axis=0,
        keepdims=True,
    )

    std = X.std(
        axis=0,
        keepdims=True,
    )


    std[
        std < 1e-6
    ] = 1.0


    X = (
        X
        -
        mean
    ) / std


    quality = np.asarray(
        [
            r[
                "quality_score"
            ]
            for r
            in valid
        ],
        dtype=np.float32,
    )


    start = int(
        np.argmax(
            quality
        )
    )


    selected = [
        start
    ]


    nearest = np.full(
        len(valid),
        np.inf,
        dtype=np.float32,
    )


    while len(
        selected
    ) < max_n:

        last = X[
            selected[
                -1
            ]
        ]


        dist = np.mean(
            (
                X - last
            )
            ** 2,
            axis=1,
        )


        nearest = np.minimum(
            nearest,
            dist,
        )


        nearest[
            selected
        ] = -1.0


        nxt = int(
            np.argmax(
                nearest
            )
        )


        if nearest[
            nxt
        ] <= 0:

            break


        selected.append(
            nxt
        )


    return [
        valid[i]
        for i
        in selected
    ]


# ============================================================
# Temporal thinning
# ============================================================

def temporal_thin(
    rows,
    min_gap,
):

    ordered = sorted(
        rows,
        key=lambda r: r[
            "frame_num"
        ],
    )


    kept = []


    for r in ordered:

        if not kept:

            kept.append(
                r
            )

            continue


        if (
            r[
                "frame_num"
            ]
            -
            kept[
                -1
            ][
                "frame_num"
            ]
            >=
            min_gap
        ):

            kept.append(
                r
            )


    return kept


# ============================================================
# Source selection
# ============================================================

def choose_sources(
    rows,
    max_sources,
    min_gap,
):

    # Strong appearance frame:
    # clear + reasonably large + valid SMPL-X.
    ordered = sorted(
        rows,
        key=lambda r: (
            r[
                "source_score"
            ]
        ),
        reverse=True,
    )


    selected = []


    for r in ordered:

        if all(
            abs(
                r[
                    "frame_num"
                ]
                -
                x[
                    "frame_num"
                ]
            )
            >=
            min_gap

            for x
            in selected
        ):

            selected.append(
                r
            )


        if len(
            selected
        ) >= max_sources:

            break


    return selected


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--max-sources",
        type=int,
        default=5,
    )


    parser.add_argument(
        "--max-targets",
        type=int,
        default=30,
    )


    parser.add_argument(
        "--source-gap",
        type=int,
        default=15,
    )


    parser.add_argument(
        "--target-gap",
        type=int,
        default=5,
    )


    parser.add_argument(
        "--min-mask-area",
        type=float,
        default=0.025,
    )


    parser.add_argument(
        "--min-person-height",
        type=float,
        default=0.30,
    )


    parser.add_argument(
        "--blur-quantile",
        type=float,
        default=0.20,
    )


    parser.add_argument(
        "--smplx-iou-floor",
        type=float,
        default=0.55,
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )


    args = parser.parse_args()


    if not INVENTORY.exists():

        raise FileNotFoundError(
            INVENTORY
        )


    MANIFEST_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    inventory = read_jsonl(
        INVENTORY
    )


    grouped = defaultdict(
        list
    )


    for row in inventory:

        grouped[
            row[
                "sequence_id"
            ]
        ].append(
            row
        )


    print()
    print("=" * 80)
    print("TikTok V3 PRE-DWPOSE SELECTION")
    print("=" * 80)

    print(
        "frames    :",
        len(inventory)
    )

    print(
        "sequences :",
        len(grouped)
    )


    scored_all = []

    sources_all = []

    targets_all = []


    for seq_idx, seq in enumerate(
        sorted(
            grouped.keys()
        ),
        1,
    ):

        base_rows = grouped[
            seq
        ]


        base_rows = sorted(
            base_rows,
            key=lambda r: r[
                "frame_name"
            ],
        )


        stems = [
            Path(
                r[
                    "frame_name"
                ]
            ).stem
            for r
            in base_rows
        ]


        bundle = load_smpl_bundle(
            seq,
            stems,
        )


        seq_rows = []


        for order_idx, base in enumerate(
            base_rows
        ):

            stem = Path(
                base[
                    "frame_name"
                ]
            ).stem


            rgb_path = Path(
                base[
                    "rgb"
                ]
            )


            image = cv2.imread(
                str(
                    rgb_path
                ),
                cv2.IMREAD_COLOR,
            )


            if image is None:
                continue


            mask_path, mask = load_mask(
                seq,
                stem,
            )


            if mask is None:
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


            geom = mask_geometry(
                mask
            )


            if geom is None:
                continue


            lap, ten, sharp = (
                person_crop_sharpness(
                    image,
                    mask,
                    geom,
                )
            )


            smpl_mask_path, iou = (
                smplx_mask_iou(
                    seq,
                    stem,
                    mask,
                )
            )


            pose = normalized_pose_feature(
                bundle,
                stem,
            )


            expr = expression_feature(
                bundle,
                stem,
            )


            row = {
                "sequence_id":
                    seq,

                "stem":
                    stem,

                "frame_num":
                    frame_number(
                        stem,
                        order_idx,
                    ),

                "rgb":
                    str(
                        rgb_path
                    ),

                "person_mask":
                    str(
                        mask_path
                    ),

                "smplx_mask":
                    (
                        str(
                            smpl_mask_path
                        )
                        if
                        smpl_mask_path
                        else
                        ""
                    ),

                "width":
                    W,

                "height":
                    H,

                **geom,

                "person_laplacian":
                    lap,

                "person_tenengrad":
                    ten,

                "person_sharpness":
                    sharp,

                "smplx_iou":
                    iou,

                "has_pose_feature":
                    pose is not None,

                "has_expression":
                    expr is not None,

                "_pose":
                    pose,

                "_expression":
                    expr,
            }


            seq_rows.append(
                row
            )


        if not seq_rows:
            continue


        # ----------------------------------------------------
        # Sequence-adaptive blur ranking.
        # ----------------------------------------------------

        sharp = np.asarray(
            [
                r[
                    "person_sharpness"
                ]
                for r
                in seq_rows
            ],
            dtype=np.float64,
        )


        ranks = percentile_rank(
            sharp
        )


        blur_threshold = float(
            np.quantile(
                sharp,
                args.blur_quantile,
            )
        )


        area_values = np.asarray(
            [
                r[
                    "mask_area_ratio"
                ]
                for r
                in seq_rows
            ],
            dtype=np.float64,
        )


        area_rank = percentile_rank(
            area_values
        )


        for r, srank, arank in zip(
            seq_rows,
            ranks,
            area_rank,
        ):

            r[
                "sharpness_rank"
            ] = float(
                srank
            )


            r[
                "person_area_rank"
            ] = float(
                arank
            )


            r[
                "blur_pass"
            ] = bool(
                r[
                    "person_sharpness"
                ]
                >=
                blur_threshold
            )


            r[
                "person_valid"
            ] = bool(
                r[
                    "mask_area_ratio"
                ]
                >=
                args.min_mask_area

                and

                r[
                    "bbox_h_ratio"
                ]
                >=
                args.min_person_height
            )


            r[
                "geometry_valid"
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
                args.smplx_iou_floor

                and

                r[
                    "has_pose_feature"
                ]
            )


            # Do NOT reject half-body just because it touches an edge.
            #
            # Appearance score:
            # clear, large enough, geometry reliable.
            iou_score = (
                r[
                    "smplx_iou"
                ]
                if
                r[
                    "smplx_iou"
                ]
                is not None
                else
                0.0
            )


            r[
                "quality_score"
            ] = float(
                0.55
                *
                r[
                    "sharpness_rank"
                ]
                +
                0.25
                *
                iou_score
                +
                0.20
                *
                r[
                    "person_area_rank"
                ]
            )


            r[
                "source_score"
            ] = float(
                0.65
                *
                r[
                    "sharpness_rank"
                ]
                +
                0.20
                *
                r[
                    "person_area_rank"
                ]
                +
                0.15
                *
                iou_score
            )


            r[
                "pre_dwpose_candidate"
            ] = bool(
                r[
                    "blur_pass"
                ]
                and
                r[
                    "person_valid"
                ]
                and
                r[
                    "geometry_valid"
                ]
            )


        candidates = [
            r
            for r
            in seq_rows
            if r[
                "pre_dwpose_candidate"
            ]
        ]


        if candidates:

            sources = choose_sources(
                candidates,
                args.max_sources,
                args.source_gap,
            )


            source_ids = {
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
                not in source_ids
            ]


            # First remove dense adjacent frames.
            target_pool = temporal_thin(
                target_pool,
                args.target_gap,
            )


            # Then select structurally diverse poses.
            targets = fps(
                target_pool,
                args.max_targets,
                expression_weight=0.15,
            )


            for r in sources:

                rr = {
                    k: v
                    for k, v
                    in r.items()
                    if not k.startswith(
                        "_"
                    )
                }

                rr[
                    "role"
                ] = "source"

                rr[
                    "status"
                ] = "PRE_DWPOSE"

                sources_all.append(
                    rr
                )


            for r in targets:

                rr = {
                    k: v
                    for k, v
                    in r.items()
                    if not k.startswith(
                        "_"
                    )
                }

                rr[
                    "role"
                ] = "target"

                rr[
                    "status"
                ] = "PRE_DWPOSE"

                targets_all.append(
                    rr
                )


        for r in seq_rows:

            scored_all.append(
                {
                    k: v
                    for k, v
                    in r.items()
                    if not k.startswith(
                        "_"
                    )
                }
            )


        if (
            seq_idx % 20 == 0
            or
            seq_idx == len(
                grouped
            )
        ):

            print(
                f"[{seq_idx}/"
                f"{len(grouped)}] "
                f"sources="
                f"{len(sources_all)} "
                f"targets="
                f"{len(targets_all)}",
                flush=True,
            )


    # ========================================================
    # Unique selected
    # ========================================================

    selected = {}


    for r in (
        sources_all
        +
        targets_all
    ):

        key = (
            r[
                "sequence_id"
            ],
            r[
                "stem"
            ],
        )


        if key not in selected:

            rr = dict(
                r
            )

            rr[
                "roles"
            ] = [
                r[
                    "role"
                ]
            ]

            selected[
                key
            ] = rr


        elif (
            r[
                "role"
            ]
            not in
            selected[
                key
            ][
                "roles"
            ]
        ):

            selected[
                key
            ][
                "roles"
            ].append(
                r[
                    "role"
                ]
            )


    selected_rows = list(
        selected.values()
    )


    write_jsonl(
        MANIFEST_ROOT
        / "frames_scored.jsonl",
        scored_all,
    )


    write_jsonl(
        MANIFEST_ROOT
        / "sources_pre_dwpose.jsonl",
        sources_all,
    )


    write_jsonl(
        MANIFEST_ROOT
        / "targets_pre_dwpose.jsonl",
        targets_all,
    )


    write_jsonl(
        MANIFEST_ROOT
        / "selected_pre_dwpose.jsonl",
        selected_rows,
    )


    sequence_source_counts = defaultdict(
        int
    )

    sequence_target_counts = defaultdict(
        int
    )


    for r in sources_all:

        sequence_source_counts[
            r[
                "sequence_id"
            ]
        ] += 1


    for r in targets_all:

        sequence_target_counts[
            r[
                "sequence_id"
            ]
        ] += 1


    summary = {
        "version":
            "tiktok_v3_pre_dwpose",

        "method":
            (
                "person-region quality + "
                "existing SMPLest-X validity + "
                "temporal thinning + "
                "pose/expression FPS"
            ),

        "counts": {
            "input_frames":
                len(
                    inventory
                ),

            "scored_frames":
                len(
                    scored_all
                ),

            "sources":
                len(
                    sources_all
                ),

            "targets":
                len(
                    targets_all
                ),

            "selected_unique":
                len(
                    selected_rows
                ),

            "sequences_with_source":
                len(
                    sequence_source_counts
                ),

            "sequences_with_target":
                len(
                    sequence_target_counts
                ),
        },

        "per_sequence": {
            "source_min":
                min(
                    sequence_source_counts.values(),
                    default=0,
                ),

            "source_max":
                max(
                    sequence_source_counts.values(),
                    default=0,
                ),

            "source_mean":
                float(
                    np.mean(
                        list(
                            sequence_source_counts.values()
                        )
                    )
                )
                if
                sequence_source_counts
                else 0.0,

            "target_min":
                min(
                    sequence_target_counts.values(),
                    default=0,
                ),

            "target_max":
                max(
                    sequence_target_counts.values(),
                    default=0,
                ),

            "target_mean":
                float(
                    np.mean(
                        list(
                            sequence_target_counts.values()
                        )
                    )
                )
                if
                sequence_target_counts
                else 0.0,
        },

        "policy": {
            "max_sources":
                args.max_sources,

            "max_targets":
                args.max_targets,

            "source_gap":
                args.source_gap,

            "target_gap":
                args.target_gap,

            "min_mask_area":
                args.min_mask_area,

            "min_person_height":
                args.min_person_height,

            "per_sequence_blur_quantile":
                args.blur_quantile,

            "smplx_iou_floor":
                args.smplx_iou_floor,

            "full_body_required":
                False,

            "half_body_allowed":
                True,

            "final_dwpose_required":
                True,
        },

        "next_stage":
            (
                "Run official DWPose only on selected_pre_dwpose "
                "and perform final body/hand/face confidence QC."
            ),
    }


    summary_path = (
        MANIFEST_ROOT
        / "summary_pre_dwpose.json"
    )


    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )


    print()
    print("=" * 80)
    print("PRE-DWPOSE SELECTION COMPLETE")
    print("=" * 80)

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
