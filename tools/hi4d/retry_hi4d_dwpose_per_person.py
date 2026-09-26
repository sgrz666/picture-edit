#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
import argparse
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import onnxruntime as ort


PROJECT = Path(
    "/home/shangguanrz/project/pic-edit"
)

SCRIPT_DIR = (
    PROJECT
    / "add/tools/SMPLest-X/scripts"
)

sys.path.insert(
    0,
    str(SCRIPT_DIR)
)

import build_iper_sampled_v2 as v2


ROOT = (
    PROJECT
    / "datasets/Hi4D_pilot_v1"
)

BASE = (
    ROOT
    / "native_interaction_v2"
)

IN_MANIFEST = (
    BASE
    / "frames_used_with_dwpose.jsonl"
)

OUT_MANIFEST = (
    BASE
    / "frames_used_with_dwpose_v2.jsonl"
)

OUT_ROOT = (
    BASE
    / "dwpose_native"
)

SUMMARY = (
    BASE
    / "dwpose_summary_v2.json"
)


# ------------------------------------------------------------
# IO
# ------------------------------------------------------------

def read_jsonl(path):
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


def write_jsonl(path, rows):
    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                ) + "\n"
            )


def read_mask(path):
    m = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if m is None:
        raise RuntimeError(
            f"cannot read mask: {path}"
        )

    return m > 0


# ------------------------------------------------------------
# MASK -> ROI
# ------------------------------------------------------------

def mask_bbox(
    mask,
    margin_ratio=0.20,
):
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    H, W = mask.shape

    x1 = int(xs.min())
    x2 = int(xs.max()) + 1

    y1 = int(ys.min())
    y2 = int(ys.max()) + 1

    bw = x2 - x1
    bh = y2 - y1

    margin = int(
        round(
            max(bw, bh)
            * margin_ratio
        )
    )

    x1 = max(
        0,
        x1 - margin
    )

    y1 = max(
        0,
        y1 - margin
    )

    x2 = min(
        W,
        x2 + margin
    )

    y2 = min(
        H,
        y2 + margin
    )

    return (
        x1,
        y1,
        x2,
        y2
    )


# ------------------------------------------------------------
# DWPose output parsing
# ------------------------------------------------------------

def run_estimator(
    estimator,
    image,
):

    result = estimator(
        image
    )

    if not isinstance(
        result,
        (tuple, list)
    ):
        return None

    if len(result) < 2:
        return None

    keypoints = np.asarray(
        result[0]
    )

    scores = np.asarray(
        result[1]
    )

    if keypoints.ndim == 2:
        keypoints = keypoints[
            None
        ]

    if scores.ndim == 1:
        scores = scores[
            None
        ]

    if len(keypoints) == 0:
        return None

    return (
        keypoints,
        scores
    )


# ------------------------------------------------------------
# Candidate quality
# ------------------------------------------------------------

def choose_best_candidate(
    keypoints,
    scores,
    crop_mask,
):
    """
    ROI 内可能还是检测出多个人。
    直接选择与目标 Hi4D mask 重合最高的 candidate。
    """

    H, W = crop_mask.shape

    best = None

    for i in range(
        len(keypoints)
    ):

        xy = np.asarray(
            keypoints[i],
            dtype=np.float32
        )

        sc = np.asarray(
            scores[i],
            dtype=np.float32
        )

        valid = (
            np.isfinite(
                xy
            ).all(axis=1)
            &
            np.isfinite(sc)
            &
            (sc >= 0.30)
            &
            (xy[:, 0] >= 0)
            &
            (xy[:, 0] < W)
            &
            (xy[:, 1] >= 0)
            &
            (xy[:, 1] < H)
        )

        if valid.sum() == 0:
            continue

        ids = np.where(
            valid
        )[0]

        px = np.clip(
            np.round(
                xy[ids, 0]
            ).astype(int),
            0,
            W - 1
        )

        py = np.clip(
            np.round(
                xy[ids, 1]
            ).astype(int),
            0,
            H - 1
        )

        weight = sc[ids]

        inside = crop_mask[
            py,
            px
        ].astype(
            np.float32
        )

        all_overlap = float(
            (
                inside
                * weight
            ).sum()
            /
            (
                weight.sum()
                + 1e-8
            )
        )

        # body joints 权重大
        body_ids = ids[
            ids < min(
                18,
                len(sc)
            )
        ]

        if len(
            body_ids
        ) > 0:

            bx = np.clip(
                np.round(
                    xy[
                        body_ids,
                        0
                    ]
                ).astype(int),
                0,
                W - 1
            )

            by = np.clip(
                np.round(
                    xy[
                        body_ids,
                        1
                    ]
                ).astype(int),
                0,
                H - 1
            )

            bw = sc[
                body_ids
            ]

            body_overlap = float(
                (
                    crop_mask[
                        by,
                        bx
                    ].astype(
                        np.float32
                    )
                    * bw
                ).sum()
                /
                (
                    bw.sum()
                    + 1e-8
                )
            )

        else:
            body_overlap = (
                all_overlap
            )

        quality = (
            0.8
            * body_overlap
            +
            0.2
            * all_overlap
        )

        if (
            best is None
            or quality > best[0]
        ):
            best = (
                quality,
                i
            )

    if best is None:
        return None

    return {
        "index":
            int(best[1]),

        "quality":
            float(best[0]),
    }


# ------------------------------------------------------------
# Person-specific ROI DWPose
# ------------------------------------------------------------

def estimate_person(
    estimator,
    rgb,
    mask,
    margin_ratio,
):

    bbox = mask_bbox(
        mask,
        margin_ratio
    )

    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox

    crop = rgb[
        y1:y2,
        x1:x2
    ].copy()

    crop_mask = mask[
        y1:y2,
        x1:x2
    ]

    if (
        crop.shape[0] < 32
        or crop.shape[1] < 32
    ):
        return None

    result = run_estimator(
        estimator,
        crop
    )

    if result is None:
        return None

    keypoints, scores = result

    chosen = choose_best_candidate(
        keypoints,
        scores,
        crop_mask
    )

    if chosen is None:
        return None

    idx = chosen[
        "index"
    ]

    xy = np.asarray(
        keypoints[idx],
        dtype=np.float32
    ).copy()

    score = np.asarray(
        scores[idx],
        dtype=np.float32
    ).copy()

    # ROI coordinate -> native coordinate
    xy[:, 0] += float(x1)
    xy[:, 1] += float(y1)

    return {
        "xy":
            xy,

        "scores":
            score,

        "bbox":
            [
                int(x1),
                int(y1),
                int(x2),
                int(y2),
            ],

        "quality":
            chosen[
                "quality"
            ],

        "num_candidates":
            int(
                len(keypoints)
            ),
    }


def save_person(
    path,
    result,
    native_w,
    native_h,
):

    xy = result[
        "xy"
    ].astype(
        np.float32
    )

    scores = result[
        "scores"
    ].astype(
        np.float32
    )

    norm = xy.copy()

    norm[:, 0] /= float(
        native_w
    )

    norm[:, 1] /= float(
        native_h
    )

    np.savez_compressed(
        path,

        xy_native=
            xy,

        xy_normalized=
            norm,

        scores=
            scores,

        roi_bbox=
            np.asarray(
                result[
                    "bbox"
                ],
                dtype=np.int32
            ),

        roi_assignment_quality=
            np.float32(
                result[
                    "quality"
                ]
            ),

        width=
            np.int32(
                native_w
            ),

        height=
            np.int32(
                native_h
            ),
    )


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dwroot",
    type=Path,
    required=True
)

parser.add_argument(
    "--margin",
    type=float,
    default=0.20
)

parser.add_argument(
    "--force",
    action="store_true"
)

args = parser.parse_args()


det = (
    args.dwroot
    / "models"
    / "yolox_l.onnx"
)

pose = (
    args.dwroot
    / "models"
    / "dw-ll_ucoco_384.onnx"
)


(
    Wholebody,
    _,
    _,
    _,
) = v2.load_dwpose(
    args.dwroot
)


providers = (
    ort.get_available_providers()
)

device = (
    "cuda"
    if (
        "CUDAExecutionProvider"
        in providers
    )
    else "cpu"
)

print(
    "ONNX providers:",
    providers
)

print(
    "DWPose device:",
    device
)


estimator = Wholebody(
    str(det),
    str(pose),
    device=device,
)


rows = read_jsonl(
    IN_MANIFEST
)

print(
    "input frames:",
    len(rows)
)


status_counter = Counter()

out_rows = []

retry_total = 0
retry_success = 0
retry_failed = 0


for idx, row in enumerate(
    rows,
    start=1
):

    # 已经成功的不重跑
    if row.get(
        "dwpose_status"
    ) in (
        "complete",
        "cached",
    ):

        r = dict(row)

        r[
            "dwpose_final_status"
        ] = "original_complete"

        out_rows.append(r)

        status_counter[
            "original_complete"
        ] += 1

        continue


    retry_total += 1

    rgb = cv2.imread(
        row["rgb"],
        cv2.IMREAD_COLOR
    )

    if rgb is None:
        raise RuntimeError(
            row["rgb"]
        )

    H, W = rgb.shape[:2]

    mask_a = read_mask(
        row[
            "person_mask_A"
        ]
    )

    mask_b = read_mask(
        row[
            "person_mask_B"
        ]
    )


    result_a = estimate_person(
        estimator,
        rgb,
        mask_a,
        args.margin
    )

    result_b = estimate_person(
        estimator,
        rgb,
        mask_b,
        args.margin
    )


    out_dir = (
        OUT_ROOT
        / row["pair"]
        / row["action"]
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    stem = (
        f"{int(row['frame_id']):06d}"
    )

    out_a = (
        out_dir
        / f"{stem}_A.npz"
    )

    out_b = (
        out_dir
        / f"{stem}_B.npz"
    )


    r = dict(row)

    r[
        "dwpose_retry_method"
    ] = (
        "Hi4D official "
        "person-mask ROI"
    )


    if (
        result_a is not None
        and
        result_b is not None
    ):

        save_person(
            out_a,
            result_a,
            W,
            H
        )

        save_person(
            out_b,
            result_b,
            W,
            H
        )

        r.update({
            "dwpose_A":
                str(out_a),

            "dwpose_B":
                str(out_b),

            "dwpose_A_roi":
                result_a[
                    "bbox"
                ],

            "dwpose_B_roi":
                result_b[
                    "bbox"
                ],

            "dwpose_A_assignment_score":
                result_a[
                    "quality"
                ],

            "dwpose_B_assignment_score":
                result_b[
                    "quality"
                ],

            "dwpose_final_status":
                "roi_recovered",
        })

        retry_success += 1

        status_counter[
            "roi_recovered"
        ] += 1

    else:

        r[
            "dwpose_final_status"
        ] = "still_incomplete"

        r[
            "dwpose_A_retry_ok"
        ] = (
            result_a is not None
        )

        r[
            "dwpose_B_retry_ok"
        ] = (
            result_b is not None
        )

        retry_failed += 1

        status_counter[
            "still_incomplete"
        ] += 1


    out_rows.append(r)


    if (
        retry_total == 1
        or
        retry_total % 50 == 0
    ):

        print(
            "retry",
            retry_total,
            "success",
            retry_success,
            "failed",
            retry_failed
        )


write_jsonl(
    OUT_MANIFEST,
    out_rows
)


complete = [
    x
    for x in out_rows
    if x[
        "dwpose_final_status"
    ] != "still_incomplete"
]


summary = {
    "input_frames":
        len(rows),

    "original_complete":
        int(
            status_counter[
                "original_complete"
            ]
        ),

    "retry_total":
        int(
            retry_total
        ),

    "roi_recovered":
        int(
            retry_success
        ),

    "still_incomplete":
        int(
            retry_failed
        ),

    "final_complete":
        len(complete),

    "final_completion_ratio":
        (
            len(complete)
            / len(rows)
        ),

    "status_counts":
        dict(
            status_counter
        ),

    "method":
        (
            "full-frame DWPose first; "
            "failed frames retry per person "
            "using Hi4D official mask ROI; "
            "coordinates mapped back to "
            "native 940x1280"
        ),

    "training_crop":
        False,

    "training_resize":
        False,
}


with open(
    SUMMARY,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False
    )


print()
print(
    "================================"
)

print(
    "DWPose ROI RECOVERY COMPLETE"
)

print(
    "================================"
)

print(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False
    )
)
