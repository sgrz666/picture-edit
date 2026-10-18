#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Project root for the categorized entrypoint.
from pathlib import Path as _EntryPath
import sys as _entry_sys
_entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))

import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("/home/shangguanrz/project/pic-edit")

MANIFEST = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/manifests/"
      "usable_frame_pool_split.jsonl"
)

OUT_ROOT = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/qc/"
      "visual_quality_v3"
)

CSV_OUT = (
    OUT_ROOT
    / "visual_quality_scores_v3.csv"
)

SUMMARY_OUT = (
    OUT_ROOT
    / "visual_quality_summary_v3.json"
)

GLOBAL_JSONL = (
    OUT_ROOT
    / "global_blur_candidates_32.jsonl"
)

FACE_JSONL = (
    OUT_ROOT
    / "face_blur_candidates_32.jsonl"
)

HAND_JSONL = (
    OUT_ROOT
    / "hand_blur_candidates_32.jsonl"
)


# ============================================================
# Audit settings
# ============================================================

N_AUDIT = 32

DWP_CONF_THR = 0.30

# These are only "is this local region large enough to evaluate?"
# engineering guardrails.
# They are NOT final blur thresholds.
MIN_FACE_NATIVE_SIDE = 32.0
MIN_HAND_NATIVE_SIDE = 24.0

MIN_FACE_VALID_POINTS = 20
MIN_HAND_VALID_POINTS = 8

FACE_TARGET_SIZE = 128
HAND_TARGET_SIZE = 96

MIN_PERSON_PIXELS = 200


# ============================================================
# IO
# ============================================================

def read_jsonl(path):
    rows = []

    with path.open(
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


def read_bgr(path):
    if path is None:
        return None

    img = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    return img


def read_binary_mask(
    path,
    shape,
):
    if path is None:
        return None

    mask = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        return None

    H, W = shape

    if mask.shape != (H, W):
        mask = cv2.resize(
            mask,
            (W, H),
            interpolation=cv2.INTER_NEAREST,
        )

    return (
        mask > 20
    ).astype(np.uint8)


# ============================================================
# Edge-conditioned sharpness
#
# Inspired by edge-based no-reference blur/focus measures:
# assess informative edges rather than averaging flat regions.
# ============================================================

def edge_sharpness(
    bgr,
    mask=None,
):
    if bgr is None:
        return float("nan")

    gray = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2GRAY,
    )

    # suppress tiny compression/noise responses
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0.8,
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

    mag = cv2.magnitude(
        gx,
        gy,
    )

    if mask is None:
        vals = mag.reshape(-1)

    else:
        valid = (
            mask > 0
        )

        if int(valid.sum()) < MIN_PERSON_PIXELS:
            return float("nan")

        vals = mag[valid]

    vals = vals[
        np.isfinite(vals)
    ]

    if vals.size < 100:
        return float("nan")

    # only use real edge-rich responses.
    # Flat shirt/background pixels do not dominate.
    threshold = float(
        np.quantile(
            vals,
            0.90,
        )
    )

    strong = vals[
        vals >= threshold
    ]

    if strong.size < 20:
        return float(
            np.median(vals)
        )

    return float(
        np.median(strong)
    )


# ============================================================
# DWPose
# ============================================================

def load_dwpose(row):
    path = Path(
        row["dwpose_json"]
    )

    obj = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    kp = np.asarray(
        obj["keypoints_norm"],
        dtype=np.float32,
    )

    score = np.asarray(
        obj["scores"],
        dtype=np.float32,
    )

    return kp, score


# ============================================================
# Landmark-local region
# ============================================================

def build_landmark_region(
    bgr,
    keypoints_norm,
    scores,
    start,
    end,
    min_valid_points,
    min_native_side,
    target_size,
):
    """
    Returns:
        normalized local crop,
        convex-hull mask,
        metadata

    Important:
    Face/hand sharpness is computed INSIDE landmark hull,
    not on the surrounding background/hair.
    """

    if bgr is None:
        return None, None, {
            "evaluable": False,
            "reason": "IMAGE_MISSING",
        }

    H, W = bgr.shape[:2]

    kp = keypoints_norm[
        start:end
    ].copy()

    sc = scores[
        start:end
    ].copy()

    valid = (
        np.isfinite(kp).all(axis=1)
        &
        np.isfinite(sc)
        &
        (sc > DWP_CONF_THR)
        &
        (kp[:, 0] > 0.0)
        &
        (kp[:, 1] > 0.0)
        &
        (kp[:, 0] < 1.0)
        &
        (kp[:, 1] < 1.0)
    )

    valid_count = int(
        valid.sum()
    )

    if valid_count < min_valid_points:
        return None, None, {
            "evaluable": False,
            "reason": "TOO_FEW_VALID_KEYPOINTS",
            "valid_points": valid_count,
        }

    pts = kp[
        valid
    ].copy()

    pts[:, 0] *= W
    pts[:, 1] *= H

    x1 = float(
        pts[:, 0].min()
    )
    x2 = float(
        pts[:, 0].max()
    )
    y1 = float(
        pts[:, 1].min()
    )
    y2 = float(
        pts[:, 1].max()
    )

    native_w = max(
        1.0,
        x2 - x1,
    )

    native_h = max(
        1.0,
        y2 - y1,
    )

    native_short = min(
        native_w,
        native_h,
    )

    meta = {
        "evaluable": False,
        "valid_points": valid_count,
        "native_width": native_w,
        "native_height": native_h,
        "native_short_side": native_short,
        "mean_keypoint_conf": float(
            sc[valid].mean()
        ),
    }

    if native_short < min_native_side:
        meta["reason"] = "TOO_SMALL_NATIVE_REGION"

        return None, None, meta

    # Small padding only for crop extraction.
    # Actual sharpness mask remains the landmark convex hull.
    pad_x = 0.12 * native_w
    pad_y = 0.12 * native_h

    cx1 = max(
        0,
        int(np.floor(x1 - pad_x)),
    )

    cx2 = min(
        W,
        int(np.ceil(x2 + pad_x)),
    )

    cy1 = max(
        0,
        int(np.floor(y1 - pad_y)),
    )

    cy2 = min(
        H,
        int(np.ceil(y2 + pad_y)),
    )

    if (
        cx2 <= cx1
        or cy2 <= cy1
    ):
        meta["reason"] = "INVALID_CROP"

        return None, None, meta

    crop = bgr[
        cy1:cy2,
        cx1:cx2,
    ].copy()

    if crop.size == 0:
        meta["reason"] = "EMPTY_CROP"

        return None, None, meta

    hull_pts = pts.copy()

    hull_pts[:, 0] -= cx1
    hull_pts[:, 1] -= cy1

    hull_pts = np.round(
        hull_pts
    ).astype(np.int32)

    hull = cv2.convexHull(
        hull_pts
    )

    local_mask = np.zeros(
        crop.shape[:2],
        dtype=np.uint8,
    )

    cv2.fillConvexPoly(
        local_mask,
        hull,
        1,
    )

    crop_r = cv2.resize(
        crop,
        (
            target_size,
            target_size,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    mask_r = cv2.resize(
        local_mask,
        (
            target_size,
            target_size,
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    meta["evaluable"] = True
    meta["reason"] = "OK"

    return (
        crop_r,
        mask_r,
        meta,
    )


def score_local_part(
    bgr,
    kp,
    scores,
    start,
    end,
    min_valid,
    min_side,
    target_size,
):
    crop, mask, meta = (
        build_landmark_region(
            bgr=bgr,
            keypoints_norm=kp,
            scores=scores,
            start=start,
            end=end,
            min_valid_points=min_valid,
            min_native_side=min_side,
            target_size=target_size,
        )
    )

    if not meta[
        "evaluable"
    ]:
        return (
            float("nan"),
            crop,
            meta,
        )

    value = edge_sharpness(
        crop,
        mask,
    )

    return (
        value,
        crop,
        meta,
    )


# ============================================================
# Sequence raw-frame index
# ============================================================

def build_dir_index(directory):
    index = {}

    directory = Path(
        directory
    )

    if not directory.is_dir():
        return index

    for p in directory.iterdir():

        if not p.is_file():
            continue

        try:
            number = int(
                p.stem
            )
        except ValueError:
            continue

        index[
            number
        ] = p

    return index


# ============================================================
# Percentile helpers
# ============================================================

def percentile_rank(
    value,
    sorted_values,
):
    if (
        not np.isfinite(value)
        or len(sorted_values) == 0
    ):
        return float("nan")

    if len(sorted_values) == 1:
        return 0.5

    left = int(
        np.searchsorted(
            sorted_values,
            value,
            side="left",
        )
    )

    right = int(
        np.searchsorted(
            sorted_values,
            value,
            side="right",
        )
    )

    rank = (
        left + right - 1
    ) / 2.0

    return float(
        rank
        /
        (
            len(sorted_values)
            - 1
        )
    )


def finite_sorted(values):
    return np.asarray(
        sorted(
            float(v)
            for v in values
            if np.isfinite(v)
        ),
        dtype=np.float64,
    )


def quantiles(arr):
    arr = np.asarray(
        [
            x
            for x in arr
            if np.isfinite(x)
        ],
        dtype=np.float64,
    )

    if len(arr) == 0:
        return {}

    result = {}

    for q in [
        0.00,
        0.01,
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
        1.00,
    ]:
        result[
            str(q)
        ] = float(
            np.quantile(
                arr,
                q,
            )
        )

    return result


# ============================================================
# Visualization
# ============================================================

def resize_keep_height(
    img,
    h,
):
    if img is None:
        return np.zeros(
            (
                h,
                h,
                3,
            ),
            dtype=np.uint8,
        )

    H, W = img.shape[:2]

    scale = (
        h / H
    )

    target_w = max(
        1,
        int(W * scale),
    )

    return cv2.resize(
        img,
        (
            target_w,
            h,
        ),
        interpolation=cv2.INTER_AREA,
    )


def fit_width(
    img,
    width,
):
    H, W = img.shape[:2]

    if W == width:
        return img

    if W > width:
        return cv2.resize(
            img,
            (
                width,
                H,
            ),
            interpolation=cv2.INTER_AREA,
        )

    canvas = np.zeros(
        (
            H,
            width,
            3,
        ),
        dtype=np.uint8,
    )

    x = (
        width - W
    ) // 2

    canvas[
        :,
        x:x + W,
    ] = img

    return canvas


def put_header(
    panel,
    lines,
):
    header_h = (
        24 * len(lines)
        + 8
    )

    canvas = np.zeros(
        (
            panel.shape[0]
            + header_h,
            panel.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    canvas[
        header_h:,
        :
    ] = panel

    for i, line in enumerate(
        lines
    ):
        cv2.putText(
            canvas,
            str(line),
            (
                5,
                20 + i * 24,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (
                255,
                255,
                255,
            ),
            1,
            cv2.LINE_AA,
        )

    return canvas


def make_global_panel(
    record,
    image_indices,
):
    seq = record[
        "sequence_id"
    ]

    n = int(
        record[
            "_frame_num_int"
        ]
    )

    idx = image_indices[
        seq
    ]

    prev_img = read_bgr(
        idx.get(
            n - 1
        )
    )

    current = read_bgr(
        record["rgb"]
    )

    next_img = read_bgr(
        idx.get(
            n + 1
        )
    )

    H = 220

    panels = [
        resize_keep_height(
            prev_img,
            H,
        ),
        resize_keep_height(
            current,
            H,
        ),
        resize_keep_height(
            next_img,
            H,
        ),
    ]

    cell_w = max(
        p.shape[1]
        for p in panels
    )

    panels = [
        fit_width(
            p,
            cell_w,
        )
        for p in panels
    ]

    strip = np.concatenate(
        panels,
        axis=1,
    )

    # mark CURRENT
    cv2.rectangle(
        strip,
        (
            cell_w,
            0,
        ),
        (
            cell_w * 2 - 1,
            H - 1,
        ),
        (
            255,
            255,
            255,
        ),
        3,
    )

    lines = [
        (
            f"{record['split']} "
            f"{seq}/{record['stem']} "
            f"edgeP={record['seq_edge_percentile']:.2f}"
        ),
        (
            f"recover={record['temporal_recovery_ratio']:.2f} "
            f"q95={record['recovery_q95_threshold']:.2f} "
            f"q99={record['recovery_q99_threshold']:.2f}"
        ),
    ]

    return put_header(
        strip,
        lines,
    )


def make_face_panel(
    record,
):
    full = read_bgr(
        record["rgb"]
    )

    kp, scores = load_dwpose(
        record
    )

    face_score, face_crop, meta = (
        score_local_part(
            bgr=full,
            kp=kp,
            scores=scores,
            start=24,
            end=92,
            min_valid=MIN_FACE_VALID_POINTS,
            min_side=MIN_FACE_NATIVE_SIDE,
            target_size=FACE_TARGET_SIZE,
        )
    )

    H = 260

    full_v = resize_keep_height(
        full,
        H,
    )

    if face_crop is None:
        face_v = np.zeros(
            (
                H,
                H,
                3,
            ),
            dtype=np.uint8,
        )
    else:
        face_v = cv2.resize(
            face_crop,
            (
                H,
                H,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

    strip = np.concatenate(
        [
            full_v,
            face_v,
        ],
        axis=1,
    )

    lines = [
        (
            f"{record['split']} "
            f"{record['sequence_id']}/{record['stem']} "
            f"face={record['face_sharpness']:.1f}"
        ),
        (
            f"faceGlobalP={record['face_global_percentile']:.2f} "
            f"faceSeqP={record['face_seq_percentile']:.2f} "
            f"size={record['face_native_short_side']:.1f}px"
        ),
    ]

    return put_header(
        strip,
        lines,
    )


def make_hand_panel(
    record,
):
    full = read_bgr(
        record["rgb"]
    )

    kp, scores = load_dwpose(
        record
    )

    l_score, l_crop, l_meta = (
        score_local_part(
            bgr=full,
            kp=kp,
            scores=scores,
            start=92,
            end=113,
            min_valid=MIN_HAND_VALID_POINTS,
            min_side=MIN_HAND_NATIVE_SIDE,
            target_size=HAND_TARGET_SIZE,
        )
    )

    r_score, r_crop, r_meta = (
        score_local_part(
            bgr=full,
            kp=kp,
            scores=scores,
            start=113,
            end=134,
            min_valid=MIN_HAND_VALID_POINTS,
            min_side=MIN_HAND_NATIVE_SIDE,
            target_size=HAND_TARGET_SIZE,
        )
    )

    H = 230

    full_v = resize_keep_height(
        full,
        H,
    )

    def local_vis(crop):
        if crop is None:
            return np.zeros(
                (
                    H,
                    H,
                    3,
                ),
                dtype=np.uint8,
            )

        return cv2.resize(
            crop,
            (
                H,
                H,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

    left_v = local_vis(
        l_crop
    )

    right_v = local_vis(
        r_crop
    )

    strip = np.concatenate(
        [
            full_v,
            left_v,
            right_v,
        ],
        axis=1,
    )

    lines = [
        (
            f"{record['split']} "
            f"{record['sequence_id']}/{record['stem']} "
            f"riskHand={record['hand_risk_side']}"
        ),
        (
            f"L={record['left_hand_sharpness']:.1f} "
            f"Lp={record['left_hand_global_percentile']:.2f} "
            f"Ls={record['left_hand_native_short_side']:.1f}px"
        ),
        (
            f"R={record['right_hand_sharpness']:.1f} "
            f"Rp={record['right_hand_global_percentile']:.2f} "
            f"Rs={record['right_hand_native_short_side']:.1f}px"
        ),
    ]

    return put_header(
        strip,
        lines,
    )


def save_contact_sheets(
    records,
    prefix,
    panel_func,
):
    if not records:
        return []

    PER_SHEET = 16
    COLS = 2

    outputs = []

    for start in range(
        0,
        len(records),
        PER_SHEET,
    ):
        chunk = records[
            start:
            start + PER_SHEET
        ]

        panels = [
            panel_func(r)
            for r in chunk
        ]

        max_h = max(
            p.shape[0]
            for p in panels
        )

        max_w = max(
            p.shape[1]
            for p in panels
        )

        fixed = []

        for p in panels:
            canvas = np.zeros(
                (
                    max_h,
                    max_w,
                    3,
                ),
                dtype=np.uint8,
            )

            canvas[
                :p.shape[0],
                :p.shape[1],
            ] = p

            fixed.append(
                canvas
            )

        rows_out = []

        for i in range(
            0,
            len(fixed),
            COLS,
        ):
            row = fixed[
                i:i + COLS
            ]

            while len(row) < COLS:
                row.append(
                    np.zeros_like(
                        fixed[0]
                    )
                )

            rows_out.append(
                np.concatenate(
                    row,
                    axis=1,
                )
            )

        sheet = np.concatenate(
            rows_out,
            axis=0,
        )

        page = (
            start // PER_SHEET
            + 1
        )

        out = (
            OUT_ROOT
            / f"{prefix}_{page:02d}.jpg"
        )

        cv2.imwrite(
            str(out),
            sheet,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                92,
            ],
        )

        outputs.append(
            str(out)
        )

    return outputs


# ============================================================
# Main
# ============================================================

def main():
    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = read_jsonl(
        MANIFEST
    )

    print("=" * 78)
    print(
        "TikTok V3 Visual Quality Audit V3"
    )
    print("=" * 78)
    print(
        "frames:",
        len(rows),
    )

    # --------------------------------------------------------
    # Sequence indices
    # --------------------------------------------------------

    by_seq = defaultdict(
        list
    )

    for row in rows:
        by_seq[
            row["sequence_id"]
        ].append(row)

    image_indices = {}
    mask_indices = {}

    for seq, seq_rows in by_seq.items():

        image_dir = Path(
            seq_rows[0]["rgb"]
        ).parent

        mask_dir = Path(
            seq_rows[0]["person_mask"]
        ).parent

        image_indices[
            seq
        ] = build_dir_index(
            image_dir
        )

        mask_indices[
            seq
        ] = build_dir_index(
            mask_dir
        )

    # --------------------------------------------------------
    # Raw-frame global edge cache
    # --------------------------------------------------------

    global_edge_cache = {}

    def raw_person_edge(
        seq,
        frame_num,
    ):
        key = (
            seq,
            int(frame_num),
        )

        if key in global_edge_cache:
            return global_edge_cache[
                key
            ]

        image_path = image_indices[
            seq
        ].get(
            int(frame_num)
        )

        mask_path = mask_indices[
            seq
        ].get(
            int(frame_num)
        )

        if image_path is None:
            global_edge_cache[
                key
            ] = float("nan")

            return float("nan")

        bgr = read_bgr(
            image_path
        )

        if bgr is None:
            global_edge_cache[
                key
            ] = float("nan")

            return float("nan")

        H, W = bgr.shape[:2]

        mask = read_binary_mask(
            mask_path,
            (
                H,
                W,
            ),
        )

        score = edge_sharpness(
            bgr,
            mask,
        )

        global_edge_cache[
            key
        ] = score

        return score

    # --------------------------------------------------------
    # First pass
    # --------------------------------------------------------

    scored = []

    for i, row in enumerate(
        rows,
        1,
    ):
        x = dict(
            row
        )

        try:
            frame_num = int(
                row["stem"]
            )
        except Exception:
            frame_num = int(
                row.get(
                    "frame_num",
                    -1,
                )
            )

        x[
            "_frame_num_int"
        ] = frame_num

        seq = row[
            "sequence_id"
        ]

        # ----------------------
        # Global / person
        # ----------------------

        current_edge = raw_person_edge(
            seq,
            frame_num,
        )

        neighbor_scores = []

        for offset in [
            -2,
            -1,
            1,
            2,
        ]:
            s = raw_person_edge(
                seq,
                frame_num + offset,
            )

            if np.isfinite(s):
                neighbor_scores.append(
                    float(s)
                )

        if (
            np.isfinite(
                current_edge
            )
            and current_edge > 1e-6
            and neighbor_scores
        ):
            # use best two temporal neighbors,
            # not one potentially noisy spike
            top = sorted(
                neighbor_scores,
                reverse=True,
            )[:2]

            neighbor_ref = float(
                np.median(
                    top
                )
            )

            recovery = (
                neighbor_ref
                /
                current_edge
            )

        else:
            neighbor_ref = float(
                "nan"
            )

            recovery = float(
                "nan"
            )

        x[
            "person_edge_sharpness"
        ] = float(
            current_edge
        )

        x[
            "temporal_neighbor_reference"
        ] = float(
            neighbor_ref
        )

        x[
            "temporal_recovery_ratio"
        ] = float(
            recovery
        )

        x[
            "temporal_neighbor_count"
        ] = len(
            neighbor_scores
        )

        # ----------------------
        # Local face / hands
        # ----------------------

        bgr = read_bgr(
            row["rgb"]
        )

        kp, scores = load_dwpose(
            row
        )

        face_score, _, face_meta = (
            score_local_part(
                bgr=bgr,
                kp=kp,
                scores=scores,
                start=24,
                end=92,
                min_valid=MIN_FACE_VALID_POINTS,
                min_side=MIN_FACE_NATIVE_SIDE,
                target_size=FACE_TARGET_SIZE,
            )
        )

        lh_score, _, lh_meta = (
            score_local_part(
                bgr=bgr,
                kp=kp,
                scores=scores,
                start=92,
                end=113,
                min_valid=MIN_HAND_VALID_POINTS,
                min_side=MIN_HAND_NATIVE_SIDE,
                target_size=HAND_TARGET_SIZE,
            )
        )

        rh_score, _, rh_meta = (
            score_local_part(
                bgr=bgr,
                kp=kp,
                scores=scores,
                start=113,
                end=134,
                min_valid=MIN_HAND_VALID_POINTS,
                min_side=MIN_HAND_NATIVE_SIDE,
                target_size=HAND_TARGET_SIZE,
            )
        )

        x[
            "face_evaluable"
        ] = bool(
            face_meta.get(
                "evaluable",
                False,
            )
        )

        x[
            "face_sharpness"
        ] = float(
            face_score
        )

        x[
            "face_native_short_side"
        ] = float(
            face_meta.get(
                "native_short_side",
                float("nan"),
            )
        )

        x[
            "face_valid_points"
        ] = int(
            face_meta.get(
                "valid_points",
                0,
            )
        )

        x[
            "left_hand_evaluable"
        ] = bool(
            lh_meta.get(
                "evaluable",
                False,
            )
        )

        x[
            "left_hand_sharpness"
        ] = float(
            lh_score
        )

        x[
            "left_hand_native_short_side"
        ] = float(
            lh_meta.get(
                "native_short_side",
                float("nan"),
            )
        )

        x[
            "left_hand_valid_points"
        ] = int(
            lh_meta.get(
                "valid_points",
                0,
            )
        )

        x[
            "right_hand_evaluable"
        ] = bool(
            rh_meta.get(
                "evaluable",
                False,
            )
        )

        x[
            "right_hand_sharpness"
        ] = float(
            rh_score
        )

        x[
            "right_hand_native_short_side"
        ] = float(
            rh_meta.get(
                "native_short_side",
                float("nan"),
            )
        )

        x[
            "right_hand_valid_points"
        ] = int(
            rh_meta.get(
                "valid_points",
                0,
            )
        )

        scored.append(
            x
        )

        if (
            i % 500 == 0
            or i == len(rows)
        ):
            print(
                f"{i}/{len(rows)}",
                flush=True,
            )

    # --------------------------------------------------------
    # Global distributions
    # --------------------------------------------------------

    global_edge_values = finite_sorted(
        [
            r[
                "person_edge_sharpness"
            ]
            for r in scored
        ]
    )

    recovery_values = finite_sorted(
        [
            r[
                "temporal_recovery_ratio"
            ]
            for r in scored
        ]
    )

    face_values = finite_sorted(
        [
            r[
                "face_sharpness"
            ]
            for r in scored
            if r[
                "face_evaluable"
            ]
        ]
    )

    left_values = finite_sorted(
        [
            r[
                "left_hand_sharpness"
            ]
            for r in scored
            if r[
                "left_hand_evaluable"
            ]
        ]
    )

    right_values = finite_sorted(
        [
            r[
                "right_hand_sharpness"
            ]
            for r in scored
            if r[
                "right_hand_evaluable"
            ]
        ]
    )

    if len(recovery_values):
        recovery_q95 = float(
            np.quantile(
                recovery_values,
                0.95,
            )
        )

        recovery_q99 = float(
            np.quantile(
                recovery_values,
                0.99,
            )
        )

    else:
        recovery_q95 = float(
            "nan"
        )

        recovery_q99 = float(
            "nan"
        )

    # --------------------------------------------------------
    # Per-sequence distributions
    # --------------------------------------------------------

    scored_by_seq = defaultdict(
        list
    )

    for r in scored:
        scored_by_seq[
            r["sequence_id"]
        ].append(r)

    for seq, seq_rows in scored_by_seq.items():

        seq_edge = finite_sorted(
            [
                r[
                    "person_edge_sharpness"
                ]
                for r in seq_rows
            ]
        )

        seq_face = finite_sorted(
            [
                r[
                    "face_sharpness"
                ]
                for r in seq_rows
                if r[
                    "face_evaluable"
                ]
            ]
        )

        seq_lh = finite_sorted(
            [
                r[
                    "left_hand_sharpness"
                ]
                for r in seq_rows
                if r[
                    "left_hand_evaluable"
                ]
            ]
        )

        seq_rh = finite_sorted(
            [
                r[
                    "right_hand_sharpness"
                ]
                for r in seq_rows
                if r[
                    "right_hand_evaluable"
                ]
            ]
        )

        for r in seq_rows:

            r[
                "sequence_usable_count"
            ] = len(
                seq_rows
            )

            r[
                "seq_edge_percentile"
            ] = percentile_rank(
                r[
                    "person_edge_sharpness"
                ],
                seq_edge,
            )

            r[
                "face_seq_percentile"
            ] = percentile_rank(
                r[
                    "face_sharpness"
                ],
                seq_face,
            )

            r[
                "left_hand_seq_percentile"
            ] = percentile_rank(
                r[
                    "left_hand_sharpness"
                ],
                seq_lh,
            )

            r[
                "right_hand_seq_percentile"
            ] = percentile_rank(
                r[
                    "right_hand_sharpness"
                ],
                seq_rh,
            )

            r[
                "face_seq_evaluable_count"
            ] = len(
                seq_face
            )

            r[
                "left_hand_seq_evaluable_count"
            ] = len(
                seq_lh
            )

            r[
                "right_hand_seq_evaluable_count"
            ] = len(
                seq_rh
            )

    # --------------------------------------------------------
    # Global percentiles for local parts
    # --------------------------------------------------------

    for r in scored:

        r[
            "person_global_percentile"
        ] = percentile_rank(
            r[
                "person_edge_sharpness"
            ],
            global_edge_values,
        )

        r[
            "face_global_percentile"
        ] = percentile_rank(
            r[
                "face_sharpness"
            ],
            face_values,
        )

        r[
            "left_hand_global_percentile"
        ] = percentile_rank(
            r[
                "left_hand_sharpness"
            ],
            left_values,
        )

        r[
            "right_hand_global_percentile"
        ] = percentile_rank(
            r[
                "right_hand_sharpness"
            ],
            right_values,
        )

        r[
            "recovery_q95_threshold"
        ] = recovery_q95

        r[
            "recovery_q99_threshold"
        ] = recovery_q99

    # --------------------------------------------------------
    # Candidate discovery
    #
    # IMPORTANT:
    # These are audit candidates, not deletion rules.
    # --------------------------------------------------------

    global_candidates = []
    face_candidates = []
    hand_candidates = []

    for r in scored:

        # ----------------------
        # GLOBAL BODY BLUR
        #
        # data-driven temporal thresholds:
        # q99 alone, or q95 + sequence low-tail
        # ----------------------

        rec = r[
            "temporal_recovery_ratio"
        ]

        ep = r[
            "seq_edge_percentile"
        ]

        global_candidate = False

        if (
            r[
                "temporal_neighbor_count"
            ]
            >= 2
            and
            np.isfinite(rec)
        ):
            if (
                np.isfinite(
                    recovery_q99
                )
                and
                rec >= recovery_q99
            ):
                global_candidate = True

            elif (
                np.isfinite(
                    recovery_q95
                )
                and
                rec >= recovery_q95
                and
                np.isfinite(ep)
                and
                ep <= 0.10
            ):
                global_candidate = True

        r[
            "global_blur_candidate"
        ] = bool(
            global_candidate
        )

        global_risk = 0.0

        if np.isfinite(rec):
            global_risk += (
                3.0
                *
                max(
                    0.0,
                    rec - 1.0,
                )
            )

        if np.isfinite(ep):
            global_risk += (
                1.0
                -
                ep
            )

        r[
            "global_blur_risk_score"
        ] = float(
            global_risk
        )

        if global_candidate:
            global_candidates.append(
                r
            )

        # ----------------------
        # FACE
        # ----------------------

        face_candidate = False

        if r[
            "face_evaluable"
        ]:

            gp = r[
                "face_global_percentile"
            ]

            sp = r[
                "face_seq_percentile"
            ]

            # Either globally poor, or clearly poor relative
            # to the same sequence.
            if (
                np.isfinite(gp)
                and
                gp <= 0.05
            ):
                face_candidate = True

            elif (
                r[
                    "face_seq_evaluable_count"
                ]
                >= 5
                and
                np.isfinite(sp)
                and
                sp <= 0.05
            ):
                face_candidate = True

        r[
            "face_blur_candidate"
        ] = bool(
            face_candidate
        )

        face_risk = 0.0

        if np.isfinite(
            r[
                "face_global_percentile"
            ]
        ):
            face_risk += (
                1.0
                -
                r[
                    "face_global_percentile"
                ]
            )

        if np.isfinite(
            r[
                "face_seq_percentile"
            ]
        ):
            face_risk += (
                0.75
                *
                (
                    1.0
                    -
                    r[
                        "face_seq_percentile"
                    ]
                )
            )

        if np.isfinite(rec):
            face_risk += (
                0.25
                *
                max(
                    0.0,
                    rec - 1.0,
                )
            )

        r[
            "face_blur_risk_score"
        ] = float(
            face_risk
        )

        if face_candidate:
            face_candidates.append(
                r
            )

        # ----------------------
        # HANDS
        # ----------------------

        side_candidates = []

        # left
        if r[
            "left_hand_evaluable"
        ]:
            gp = r[
                "left_hand_global_percentile"
            ]

            sp = r[
                "left_hand_seq_percentile"
            ]

            if (
                np.isfinite(gp)
                and
                gp <= 0.05
            ):
                side_candidates.append(
                    "LEFT"
                )

            elif (
                r[
                    "left_hand_seq_evaluable_count"
                ]
                >= 5
                and
                np.isfinite(sp)
                and
                sp <= 0.05
            ):
                side_candidates.append(
                    "LEFT"
                )

        # right
        if r[
            "right_hand_evaluable"
        ]:
            gp = r[
                "right_hand_global_percentile"
            ]

            sp = r[
                "right_hand_seq_percentile"
            ]

            if (
                np.isfinite(gp)
                and
                gp <= 0.05
            ):
                side_candidates.append(
                    "RIGHT"
                )

            elif (
                r[
                    "right_hand_seq_evaluable_count"
                ]
                >= 5
                and
                np.isfinite(sp)
                and
                sp <= 0.05
            ):
                side_candidates.append(
                    "RIGHT"
                )

        hand_candidate = bool(
            side_candidates
        )

        if len(
            side_candidates
        ) == 2:
            hand_side = "BOTH"
        elif side_candidates:
            hand_side = (
                side_candidates[0]
            )
        else:
            hand_side = "NONE"

        r[
            "hand_blur_candidate"
        ] = bool(
            hand_candidate
        )

        r[
            "hand_risk_side"
        ] = hand_side

        side_risks = []

        for prefix in [
            "left_hand",
            "right_hand",
        ]:

            gp = r[
                f"{prefix}_global_percentile"
            ]

            sp = r[
                f"{prefix}_seq_percentile"
            ]

            risk = 0.0

            if np.isfinite(gp):
                risk += (
                    1.0 - gp
                )

            if np.isfinite(sp):
                risk += (
                    0.75
                    *
                    (
                        1.0 - sp
                    )
                )

            side_risks.append(
                risk
            )

        r[
            "hand_blur_risk_score"
        ] = float(
            max(
                side_risks
            )
        )

        if hand_candidate:
            hand_candidates.append(
                r
            )

    # --------------------------------------------------------
    # Rank top candidates
    # --------------------------------------------------------

    global_ranked = sorted(
        global_candidates,
        key=lambda r:
            r[
                "global_blur_risk_score"
            ],
        reverse=True,
    )[
        :N_AUDIT
    ]

    face_ranked = sorted(
        face_candidates,
        key=lambda r:
            r[
                "face_blur_risk_score"
            ],
        reverse=True,
    )[
        :N_AUDIT
    ]

    hand_ranked = sorted(
        hand_candidates,
        key=lambda r:
            r[
                "hand_blur_risk_score"
            ],
        reverse=True,
    )[
        :N_AUDIT
    ]

    # --------------------------------------------------------
    # Write candidate JSONLs
    # --------------------------------------------------------

    def write_jsonl(
        path,
        data,
    ):
        with path.open(
            "w",
            encoding="utf-8",
        ) as f:
            for r in data:
                x = dict(
                    r
                )

                x.pop(
                    "_frame_num_int",
                    None,
                )

                f.write(
                    json.dumps(
                        x,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    write_jsonl(
        GLOBAL_JSONL,
        global_ranked,
    )

    write_jsonl(
        FACE_JSONL,
        face_ranked,
    )

    write_jsonl(
        HAND_JSONL,
        hand_ranked,
    )

    # --------------------------------------------------------
    # Full CSV
    # --------------------------------------------------------

    fields = [
        "split",
        "sequence_id",
        "stem",

        "person_edge_sharpness",
        "person_global_percentile",
        "seq_edge_percentile",

        "temporal_recovery_ratio",
        "temporal_neighbor_count",

        "face_evaluable",
        "face_sharpness",
        "face_global_percentile",
        "face_seq_percentile",
        "face_native_short_side",
        "face_valid_points",

        "left_hand_evaluable",
        "left_hand_sharpness",
        "left_hand_global_percentile",
        "left_hand_seq_percentile",
        "left_hand_native_short_side",
        "left_hand_valid_points",

        "right_hand_evaluable",
        "right_hand_sharpness",
        "right_hand_global_percentile",
        "right_hand_seq_percentile",
        "right_hand_native_short_side",
        "right_hand_valid_points",

        "global_blur_candidate",
        "face_blur_candidate",
        "hand_blur_candidate",
        "hand_risk_side",

        "global_blur_risk_score",
        "face_blur_risk_score",
        "hand_blur_risk_score",

        "rgb",
    ]

    with CSV_OUT.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for r in scored:
            writer.writerow({
                k: r.get(k)
                for k in fields
            })

    # --------------------------------------------------------
    # Contact sheets
    # --------------------------------------------------------

    global_sheets = save_contact_sheets(
        global_ranked,
        "global_blur",
        lambda r:
            make_global_panel(
                r,
                image_indices,
            ),
    )

    face_sheets = save_contact_sheets(
        face_ranked,
        "face_blur",
        make_face_panel,
    )

    hand_sheets = save_contact_sheets(
        hand_ranked,
        "hand_blur",
        make_hand_panel,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "version":
            "tiktok_v3_visual_quality_v3_audit",

        "input_frames":
            len(scored),

        "important":
            (
                "AUDIT ONLY. No frame, split, source pool "
                "or pair manifest has been modified."
            ),

        "method": {
            "global":
                (
                    "person-region edge-conditioned sharpness "
                    "+ sequence-relative percentile "
                    "+ ±1/±2 temporal recovery"
                ),

            "face":
                (
                    "DWPose face-landmark convex-hull "
                    "edge sharpness after scale normalization"
                ),

            "hands":
                (
                    "left/right DWPose hand-landmark convex-hull "
                    "edge sharpness after scale normalization"
                ),

            "flat_region_average":
                False,

            "learned_iqa_model":
                False,

            "optical_flow":
                False,
        },

        "evaluable_guardrails": {
            "dwpose_confidence":
                DWP_CONF_THR,

            "face_min_valid_points":
                MIN_FACE_VALID_POINTS,

            "face_min_native_short_side_px":
                MIN_FACE_NATIVE_SIDE,

            "hand_min_valid_points":
                MIN_HAND_VALID_POINTS,

            "hand_min_native_short_side_px":
                MIN_HAND_NATIVE_SIDE,

            "note":
                (
                    "These are engineering guards against "
                    "judging tiny/poorly-localized regions, "
                    "not literature-derived deletion thresholds."
                ),
        },

        "data_driven_temporal_thresholds": {
            "recovery_q95":
                recovery_q95,

            "recovery_q99":
                recovery_q99,

            "candidate_rule":
                (
                    "recovery>=q99 OR "
                    "(recovery>=q95 AND sequence edge percentile<=0.10)"
                ),
        },

        "counts": {
            "face_evaluable":
                sum(
                    r[
                        "face_evaluable"
                    ]
                    for r in scored
                ),

            "left_hand_evaluable":
                sum(
                    r[
                        "left_hand_evaluable"
                    ]
                    for r in scored
                ),

            "right_hand_evaluable":
                sum(
                    r[
                        "right_hand_evaluable"
                    ]
                    for r in scored
                ),

            "global_blur_candidates_all":
                len(
                    global_candidates
                ),

            "face_blur_candidates_all":
                len(
                    face_candidates
                ),

            "hand_blur_candidates_all":
                len(
                    hand_candidates
                ),
        },

        "quantiles": {
            "person_edge_sharpness":
                quantiles(
                    [
                        r[
                            "person_edge_sharpness"
                        ]
                        for r in scored
                    ]
                ),

            "temporal_recovery":
                quantiles(
                    [
                        r[
                            "temporal_recovery_ratio"
                        ]
                        for r in scored
                    ]
                ),

            "face_sharpness_evaluable":
                quantiles(
                    [
                        r[
                            "face_sharpness"
                        ]
                        for r in scored
                        if r[
                            "face_evaluable"
                        ]
                    ]
                ),

            "left_hand_sharpness_evaluable":
                quantiles(
                    [
                        r[
                            "left_hand_sharpness"
                        ]
                        for r in scored
                        if r[
                            "left_hand_evaluable"
                        ]
                    ]
                ),

            "right_hand_sharpness_evaluable":
                quantiles(
                    [
                        r[
                            "right_hand_sharpness"
                        ]
                        for r in scored
                        if r[
                            "right_hand_evaluable"
                        ]
                    ]
                ),
        },

        "audit_top_n":
            N_AUDIT,

        "global_candidate_jsonl":
            str(
                GLOBAL_JSONL
            ),

        "face_candidate_jsonl":
            str(
                FACE_JSONL
            ),

        "hand_candidate_jsonl":
            str(
                HAND_JSONL
            ),

        "global_contact_sheets":
            global_sheets,

        "face_contact_sheets":
            face_sheets,

        "hand_contact_sheets":
            hand_sheets,

        "score_csv":
            str(
                CSV_OUT
            ),
    }

    SUMMARY_OUT.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print(
        "VISUAL QUALITY V3 COMPLETE"
    )
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
