#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Project root for the categorized entrypoint.
from pathlib import Path as _EntryPath
import sys as _entry_sys
_entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))

import csv
import json
import math
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
      "visual_quality_v2"
)

CSV_OUT = (
    OUT_ROOT
    / "visual_quality_scores_v2.csv"
)

SUMMARY_OUT = (
    OUT_ROOT
    / "visual_quality_summary_v2.json"
)

HARD_JSONL = (
    OUT_ROOT
    / "hard_blur_candidates_32.jsonl"
)

SOURCE_JSONL = (
    OUT_ROOT
    / "source_blur_risk_candidates_32.jsonl"
)


# ============================================================
# Parameters
# ============================================================

N_AUDIT = 32

# only for candidate discovery, NOT final deletion threshold
SEQ_LOW_Q = 0.10

# current frame is much worse than temporal neighbors
RECOVERY_STRONG = 1.50
RECOVERY_VERY_STRONG = 1.80

FACE_SCORE_THR = 0.30

FACE_RESIZE = 128

MIN_PERSON_PIXELS = 200

SEED = 2026


# ============================================================
# Basic IO
# ============================================================

def read_jsonl(path):
    return [
        json.loads(x)
        for x in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]


def read_bgr(path):
    img = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    return img


def read_mask(path, shape):
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
# ============================================================

def edge_sharpness(
    bgr,
    mask=None,
):
    """
    Tenengrad-style strong-edge sharpness.

    Important:
    - first slightly suppress compression noise
    - calculate Sobel gradient
    - DO NOT average over all person pixels
    - only average the strongest 10% gradient responses

    Therefore a large flat/solid-color shirt does not
    automatically receive a very low score simply because
    most pixels contain no texture.
    """

    if bgr is None:
        return float("nan")

    gray = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2GRAY,
    )

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

    # ignore flat regions:
    # only strongest 10% edge responses
    q90 = float(
        np.quantile(
            vals,
            0.90,
        )
    )

    strong = vals[
        vals >= q90
    ]

    if strong.size < 20:
        return float(
            np.mean(vals)
        )

    # median of strong edges is less sensitive
    # to compression/noise spikes than max
    return float(
        np.median(strong)
    )


# ============================================================
# Face crop from saved DWPose
# ============================================================

def load_dwpose_face(row):
    p = Path(
        row["dwpose_json"]
    )

    if not p.is_file():
        return None

    obj = json.loads(
        p.read_text(
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

    # DWPose standardized face:
    # indices 24:92
    if (
        kp.shape[0] < 92
        or score.shape[0] < 92
    ):
        return None

    return (
        kp[24:92],
        score[24:92],
    )


def get_face_crop(
    bgr,
    row,
):
    if bgr is None:
        return None

    face = load_dwpose_face(
        row
    )

    if face is None:
        return None

    kp, score = face

    valid = (
        np.isfinite(kp).all(axis=1)
        &
        (score > FACE_SCORE_THR)
    )

    if int(valid.sum()) < 8:
        return None

    H, W = bgr.shape[:2]

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

    bw = max(
        1.0,
        x2 - x1,
    )

    bh = max(
        1.0,
        y2 - y1,
    )

    # include hair/head contour around face landmarks
    padx = 0.35 * bw
    pady = 0.45 * bh

    x1 = max(
        0,
        int(x1 - padx),
    )

    x2 = min(
        W,
        int(x2 + padx),
    )

    y1 = max(
        0,
        int(y1 - pady),
    )

    y2 = min(
        H,
        int(y2 + pady),
    )

    if (
        x2 - x1 < 20
        or y2 - y1 < 20
    ):
        return None

    return bgr[
        y1:y2,
        x1:x2,
    ].copy()


def face_edge_sharpness(
    bgr,
    row,
):
    crop = get_face_crop(
        bgr,
        row,
    )

    if crop is None:
        return float("nan")

    # normalize face scale before computing sharpness
    crop = cv2.resize(
        crop,
        (
            FACE_RESIZE,
            FACE_RESIZE,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    return edge_sharpness(
        crop,
        mask=None,
    )


# ============================================================
# Neighbor indexing
# ============================================================

IMAGE_EXTS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
]

MASK_EXTS = [
    ".png",
    ".jpg",
    ".jpeg",
]


def build_dir_index(directory):
    directory = Path(
        directory
    )

    index = {}

    if not directory.is_dir():
        return index

    for p in directory.iterdir():
        if not p.is_file():
            continue

        try:
            n = int(
                p.stem
            )
        except ValueError:
            continue

        index[n] = p

    return index


def get_motion_class(row):
    """
    Preserve previously computed motion class if available.
    It is NOT used as a hard rejection rule.
    """

    for key in [
        "motion_class",
        "motion_type",
        "motion_level",
        "dynamic_class",
    ]:
        if key in row:
            return row[key]

    sampling = row.get(
        "sampling",
    )

    if isinstance(
        sampling,
        dict,
    ):
        for key in [
            "motion_class",
            "motion_type",
            "motion_level",
        ]:
            if key in sampling:
                return sampling[key]

    return None


# ============================================================
# Relative percentile
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

    avg_rank = (
        left
        +
        right
        -
        1
    ) / 2.0

    return float(
        avg_rank
        /
        (
            len(sorted_values)
            - 1
        )
    )


# ============================================================
# Contact sheets
# ============================================================

def resize_keep_height(
    img,
    h,
):
    if img is None:
        return np.zeros(
            (
                h,
                int(h * 0.65),
                3,
            ),
            dtype=np.uint8,
        )

    H, W = img.shape[:2]

    scale = (
        h / H
    )

    w = max(
        1,
        int(W * scale),
    )

    return cv2.resize(
        img,
        (w, h),
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


def make_temporal_panel(
    record,
    image_indices,
):
    """
    previous frame | current | next frame

    Metric itself uses ±1/±2.
    Visual audit shows ±1 for fast inspection.
    """

    seq = record[
        "sequence_id"
    ]

    frame_num = int(
        record["_frame_num_int"]
    )

    idx = image_indices[
        seq
    ]

    prev_path = idx.get(
        frame_num - 1
    )

    next_path = idx.get(
        frame_num + 1
    )

    current = read_bgr(
        record["rgb"]
    )

    prev_img = (
        read_bgr(prev_path)
        if prev_path
        else None
    )

    next_img = (
        read_bgr(next_path)
        if next_path
        else None
    )

    H = 220

    p = resize_keep_height(
        prev_img,
        H,
    )

    c = resize_keep_height(
        current,
        H,
    )

    n = resize_keep_height(
        next_img,
        H,
    )

    cell_w = max(
        p.shape[1],
        c.shape[1],
        n.shape[1],
    )

    p = fit_width(
        p,
        cell_w,
    )

    c = fit_width(
        c,
        cell_w,
    )

    n = fit_width(
        n,
        cell_w,
    )

    strip = np.concatenate(
        [
            p,
            c,
            n,
        ],
        axis=1,
    )

    # mark current frame
    cv2.rectangle(
        strip,
        (
            cell_w,
            0,
        ),
        (
            2 * cell_w - 1,
            H - 1,
        ),
        (255, 255, 255),
        3,
    )

    title_h = 58

    canvas = np.zeros(
        (
            H + title_h,
            strip.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    canvas[
        title_h:,
        :
    ] = strip

    text1 = (
        f"{record['split']} "
        f"{seq}/{record['stem']}  "
        f"edge={record['edge_sharpness']:.1f}  "
        f"seqP={record['seq_sharpness_percentile']:.2f}"
    )

    rec = record[
        "temporal_recovery_ratio"
    ]

    text2 = (
        f"recover={rec:.2f}  "
        f"face={record['face_edge_sharpness']:.1f}  "
        f"motion={record.get('motion_class')}"
    )

    cv2.putText(
        canvas,
        text1,
        (5, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        text2,
        (5, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return canvas


def make_source_panel(
    record,
):
    bgr = read_bgr(
        record["rgb"]
    )

    face = get_face_crop(
        bgr,
        record,
    )

    H = 260

    full = resize_keep_height(
        bgr,
        H,
    )

    if face is None:
        face_vis = np.zeros(
            (
                H,
                H,
                3,
            ),
            dtype=np.uint8,
        )
    else:
        face_vis = cv2.resize(
            face,
            (
                H,
                H,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

    panel = np.concatenate(
        [
            full,
            face_vis,
        ],
        axis=1,
    )

    title_h = 58

    canvas = np.zeros(
        (
            H + title_h,
            panel.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    canvas[
        title_h:,
        :
    ] = panel

    text1 = (
        f"{record['split']} "
        f"{record['sequence_id']}/{record['stem']}  "
        f"face={record['face_edge_sharpness']:.1f}"
    )

    text2 = (
        f"faceP={record['face_seq_percentile']:.2f}  "
        f"edgeP={record['seq_sharpness_percentile']:.2f}  "
        f"recover={record['temporal_recovery_ratio']:.2f}"
    )

    cv2.putText(
        canvas,
        text1,
        (5, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        text2,
        (5, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return canvas


def save_panels(
    records,
    out_prefix,
    panel_func,
):
    """
    16 candidates per sheet.
    For 32 candidates => two sheets.
    """

    PER_SHEET = 16
    COLS = 2

    output_paths = []

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

        rows = []

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

            rows.append(
                np.concatenate(
                    row,
                    axis=1,
                )
            )

        sheet = np.concatenate(
            rows,
            axis=0,
        )

        page = (
            start // PER_SHEET
            + 1
        )

        out = OUT_ROOT / (
            f"{out_prefix}_"
            f"{page:02d}.jpg"
        )

        cv2.imwrite(
            str(out),
            sheet,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                92,
            ],
        )

        output_paths.append(
            str(out)
        )

    return output_paths


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
        "TikTok V3 Visual Quality Audit V2"
    )
    print("=" * 78)
    print(
        "input frames:",
        len(rows),
    )

    # --------------------------------------------------------
    # Build sequence image/mask directory indices
    # --------------------------------------------------------

    by_seq = defaultdict(
        list
    )

    for r in rows:
        by_seq[
            r["sequence_id"]
        ].append(r)

    image_indices = {}
    mask_indices = {}

    for seq, seq_rows in by_seq.items():

        rgb_parent = Path(
            seq_rows[0]["rgb"]
        ).parent

        mask_parent = Path(
            seq_rows[0]["person_mask"]
        ).parent

        image_indices[
            seq
        ] = build_dir_index(
            rgb_parent
        )

        mask_indices[
            seq
        ] = build_dir_index(
            mask_parent
        )

    # --------------------------------------------------------
    # Cache because neighbor frames overlap heavily
    # --------------------------------------------------------

    edge_cache = {}

    def raw_frame_edge(
        seq,
        frame_num,
    ):
        key = (
            seq,
            int(frame_num),
        )

        if key in edge_cache:
            return edge_cache[
                key
            ]

        ipath = image_indices[
            seq
        ].get(
            int(frame_num)
        )

        mpath = mask_indices[
            seq
        ].get(
            int(frame_num)
        )

        if ipath is None:
            edge_cache[
                key
            ] = float(
                "nan"
            )
            return edge_cache[
                key
            ]

        bgr = read_bgr(
            ipath
        )

        if bgr is None:
            edge_cache[
                key
            ] = float(
                "nan"
            )
            return edge_cache[
                key
            ]

        H, W = bgr.shape[:2]

        mask = (
            read_mask(
                mpath,
                (H, W),
            )
            if mpath is not None
            else None
        )

        score = edge_sharpness(
            bgr,
            mask,
        )

        edge_cache[
            key
        ] = score

        return score

    # --------------------------------------------------------
    # Score current usable frames + temporal neighbors
    # --------------------------------------------------------

    scored = []

    for i, r in enumerate(
        rows,
        1,
    ):

        x = dict(
            r
        )

        try:
            frame_num = int(
                r["stem"]
            )
        except Exception:
            frame_num = int(
                r.get(
                    "frame_num",
                    -1,
                )
            )

        x[
            "_frame_num_int"
        ] = frame_num

        current_edge = raw_frame_edge(
            r["sequence_id"],
            frame_num,
        )

        bgr = read_bgr(
            r["rgb"]
        )

        face_edge = (
            face_edge_sharpness(
                bgr,
                r,
            )
            if bgr is not None
            else float("nan")
        )

        neighbor_scores = []

        for offset in [
            -2,
            -1,
            1,
            2,
        ]:
            s = raw_frame_edge(
                r["sequence_id"],
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
            # use the median of the best two temporal neighbors
            # instead of a single maximum spike
            best = sorted(
                neighbor_scores,
                reverse=True,
            )[:2]

            neighbor_reference = float(
                np.median(best)
            )

            recovery = (
                neighbor_reference
                /
                current_edge
            )
        else:
            neighbor_reference = float(
                "nan"
            )

            recovery = float(
                "nan"
            )

        x[
            "edge_sharpness"
        ] = float(
            current_edge
        )

        x[
            "face_edge_sharpness"
        ] = float(
            face_edge
        )

        x[
            "temporal_neighbor_reference"
        ] = float(
            neighbor_reference
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

        x[
            "motion_class"
        ] = get_motion_class(
            r
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
    # Sequence-relative percentiles
    # --------------------------------------------------------

    scored_by_seq = defaultdict(
        list
    )

    for r in scored:
        scored_by_seq[
            r["sequence_id"]
        ].append(r)

    for seq, seq_rows in scored_by_seq.items():

        sharp_vals = np.asarray(
            sorted(
                [
                    r["edge_sharpness"]
                    for r in seq_rows
                    if np.isfinite(
                        r["edge_sharpness"]
                    )
                ]
            ),
            dtype=np.float64,
        )

        face_vals = np.asarray(
            sorted(
                [
                    r["face_edge_sharpness"]
                    for r in seq_rows
                    if np.isfinite(
                        r["face_edge_sharpness"]
                    )
                ]
            ),
            dtype=np.float64,
        )

        for r in seq_rows:

            r[
                "sequence_usable_count"
            ] = len(
                seq_rows
            )

            r[
                "seq_sharpness_percentile"
            ] = percentile_rank(
                r[
                    "edge_sharpness"
                ],
                sharp_vals,
            )

            r[
                "face_seq_percentile"
            ] = percentile_rank(
                r[
                    "face_edge_sharpness"
                ],
                face_vals,
            )

    # --------------------------------------------------------
    # Candidate labels
    #
    # IMPORTANT:
    # These labels are AUDIT CANDIDATES ONLY.
    # No frames are removed here.
    # --------------------------------------------------------

    hard_candidates = []

    source_candidates = []

    for r in scored:

        p = r[
            "seq_sharpness_percentile"
        ]

        rec = r[
            "temporal_recovery_ratio"
        ]

        fp = r[
            "face_seq_percentile"
        ]

        seq_n = r[
            "sequence_usable_count"
        ]

        enough_seq = (
            seq_n >= 5
        )

        enough_neighbors = (
            r[
                "temporal_neighbor_count"
            ]
            >= 2
        )

        hard = False

        if (
            enough_neighbors
            and np.isfinite(rec)
        ):

            # very clear relative collapse
            if (
                rec
                >= RECOVERY_VERY_STRONG
            ):
                hard = True

            # sequence-low-tail + strong temporal recovery
            elif (
                enough_seq
                and
                np.isfinite(p)
                and
                p <= SEQ_LOW_Q
                and
                rec >= RECOVERY_STRONG
            ):
                hard = True

        r[
            "hard_blur_candidate"
        ] = bool(
            hard
        )

        # ranking score only
        hard_score = 0.0

        if np.isfinite(rec):
            hard_score += (
                max(
                    0.0,
                    rec - 1.0,
                )
                * 2.0
            )

        if np.isfinite(p):
            hard_score += (
                max(
                    0.0,
                    0.25 - p,
                )
                * 2.0
            )

        r[
            "hard_blur_suspicion_score"
        ] = float(
            hard_score
        )

        if hard:
            hard_candidates.append(
                r
            )

        # ----------------------------------------------------
        # Source-only blur risk
        #
        # Not a target rejection rule.
        # ----------------------------------------------------

        source_risk = False

        if not hard:

            if (
                enough_seq
                and
                np.isfinite(fp)
                and
                fp <= 0.10
            ):
                source_risk = True

            elif (
                np.isfinite(p)
                and
                p <= 0.10
                and
                np.isfinite(rec)
                and
                rec >= 1.20
            ):
                source_risk = True

        r[
            "source_blur_risk_candidate"
        ] = bool(
            source_risk
        )

        source_score = 0.0

        if np.isfinite(fp):
            source_score += (
                1.0 - fp
            )

        if np.isfinite(p):
            source_score += (
                0.5
                *
                (
                    1.0 - p
                )
            )

        if np.isfinite(rec):
            source_score += (
                0.5
                *
                max(
                    0.0,
                    rec - 1.0,
                )
            )

        r[
            "source_blur_risk_score"
        ] = float(
            source_score
        )

        if source_risk:
            source_candidates.append(
                r
            )

    # --------------------------------------------------------
    # Rank candidates
    # --------------------------------------------------------

    hard_ranked = sorted(
        hard_candidates,
        key=lambda r:
            r[
                "hard_blur_suspicion_score"
            ],
        reverse=True,
    )[
        :N_AUDIT
    ]

    source_ranked = sorted(
        source_candidates,
        key=lambda r:
            r[
                "source_blur_risk_score"
            ],
        reverse=True,
    )[
        :N_AUDIT
    ]

    # --------------------------------------------------------
    # Save audit manifests
    # --------------------------------------------------------

    def write_jsonl(
        path,
        items,
    ):
        with path.open(
            "w",
            encoding="utf-8",
        ) as f:
            for r in items:
                # remove only internal helper
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
        HARD_JSONL,
        hard_ranked,
    )

    write_jsonl(
        SOURCE_JSONL,
        source_ranked,
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    fields = [
        "split",
        "sequence_id",
        "stem",
        "motion_class",

        "edge_sharpness",
        "seq_sharpness_percentile",

        "temporal_neighbor_reference",
        "temporal_recovery_ratio",
        "temporal_neighbor_count",

        "face_edge_sharpness",
        "face_seq_percentile",

        "sequence_usable_count",

        "hard_blur_candidate",
        "hard_blur_suspicion_score",

        "source_blur_risk_candidate",
        "source_blur_risk_score",

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

    hard_sheets = []

    if hard_ranked:
        hard_sheets = save_panels(
            hard_ranked,
            "hard_blur_temporal",
            lambda r:
                make_temporal_panel(
                    r,
                    image_indices,
                ),
        )

    source_sheets = []

    if source_ranked:
        source_sheets = save_panels(
            source_ranked,
            "source_blur_risk",
            make_source_panel,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    edges = np.asarray(
        [
            r[
                "edge_sharpness"
            ]
            for r in scored
            if np.isfinite(
                r[
                    "edge_sharpness"
                ]
            )
        ],
        dtype=np.float64,
    )

    recovery = np.asarray(
        [
            r[
                "temporal_recovery_ratio"
            ]
            for r in scored
            if np.isfinite(
                r[
                    "temporal_recovery_ratio"
                ]
            )
        ],
        dtype=np.float64,
    )

    faces = np.asarray(
        [
            r[
                "face_edge_sharpness"
            ]
            for r in scored
            if np.isfinite(
                r[
                    "face_edge_sharpness"
                ]
            )
        ],
        dtype=np.float64,
    )

    def qs(arr):
        if len(arr) == 0:
            return {}

        return {
            str(q):
                float(
                    np.quantile(
                        arr,
                        q,
                    )
                )
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
            ]
        }

    summary = {
        "version":
            "tiktok_v3_visual_quality_v2_audit",

        "input_frames":
            len(scored),

        "policy": {
            "delete_frames":
                False,

            "global_laplacian_hard_threshold":
                False,

            "edge_conditioned_sharpness":
                True,

            "sequence_relative_sharpness":
                True,

            "temporal_neighbor_check":
                "±1/±2 original frames",

            "face_sharpness":
                "source-oriented only",

            "motion_class_hard_rule":
                False,
        },

        "hard_blur_candidate_count":
            len(
                hard_candidates
            ),

        "source_blur_risk_candidate_count":
            len(
                source_candidates
            ),

        "audit_top_n":
            N_AUDIT,

        "edge_sharpness_quantiles":
            qs(edges),

        "temporal_recovery_quantiles":
            qs(recovery),

        "face_edge_sharpness_quantiles":
            qs(faces),

        "hard_candidate_jsonl":
            str(
                HARD_JSONL
            ),

        "source_candidate_jsonl":
            str(
                SOURCE_JSONL
            ),

        "hard_contact_sheets":
            hard_sheets,

        "source_contact_sheets":
            source_sheets,

        "score_csv":
            str(
                CSV_OUT
            ),

        "important":
            (
                "Candidate labels are for visual review only. "
                "No usable frame, split, source pool or pair "
                "manifest has been modified."
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
        "VISUAL QUALITY V2 COMPLETE"
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
