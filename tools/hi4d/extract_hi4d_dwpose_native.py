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

IN_MANIFEST = (
    ROOT
    / "native_interaction_v2"
    / "frames_used_native.jsonl"
)

OUT_ROOT = (
    ROOT
    / "native_interaction_v2"
    / "dwpose_native"
)

OUT_MANIFEST = (
    ROOT
    / "native_interaction_v2"
    / "frames_used_with_dwpose.jsonl"
)

SUMMARY = (
    ROOT
    / "native_interaction_v2"
    / "dwpose_summary.json"
)


def read_jsonl(p):
    rows = []

    with open(
        p,
        "r",
        encoding="utf-8"
    ) as f:
        for line in f:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


def write_jsonl(p, rows):
    with open(
        p,
        "w",
        encoding="utf-8"
    ) as f:

        for x in rows:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False
                ) + "\n"
            )


def read_mask(path):

    x = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if x is None:
        raise RuntimeError(
            f"cannot read {path}"
        )

    return x > 0


def mask_centroid(mask):

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    return np.array(
        [
            xs.mean(),
            ys.mean()
        ],
        dtype=np.float32
    )


def candidate_score(
    xy,
    conf,
    mask,
):
    """
    用 Hi4D 官方 visible person mask
    给 DWPose detection 做 A/B ownership。

    这里只用于 identity assignment，
    不作为 target model control。
    """

    H, W = mask.shape

    xy = np.asarray(
        xy,
        dtype=np.float32
    )

    conf = np.asarray(
        conf,
        dtype=np.float32
    )

    valid = (
        np.isfinite(xy).all(axis=1)
        &
        np.isfinite(conf)
        &
        (conf >= 0.30)
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
        return -1.0

    ids = np.where(valid)[0]

    px = np.round(
        xy[ids, 0]
    ).astype(int)

    py = np.round(
        xy[ids, 1]
    ).astype(int)

    px = np.clip(
        px,
        0,
        W - 1
    )

    py = np.clip(
        py,
        0,
        H - 1
    )

    weights = conf[ids]

    inside = mask[
        py,
        px
    ].astype(
        np.float32
    )

    all_overlap = float(
        (
            inside
            * weights
        ).sum()
        /
        (
            weights.sum()
            + 1e-8
        )
    )

    # DWPose 前面一段是 body joints，
    # body overlap 权重更高，
    # 避免 face/hand大量点支配assignment。
    body_ids = ids[
        ids < min(
            18,
            len(conf)
        )
    ]

    if len(body_ids) > 0:

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

        bw = conf[
            body_ids
        ]

        body_overlap = float(
            (
                mask[
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
        body_overlap = all_overlap

    # centroid 只作为弱辅助
    mc = mask_centroid(
        mask
    )

    center_score = 0.0

    if mc is not None:

        c = np.average(
            xy[ids],
            axis=0,
            weights=weights
        )

        diag = (
            H ** 2
            + W ** 2
        ) ** 0.5

        d = (
            np.linalg.norm(
                c - mc
            )
            / diag
        )

        center_score = max(
            0.0,
            1.0 - 3.0 * d
        )

    return (
        0.75 * body_overlap
        +
        0.20 * all_overlap
        +
        0.05 * center_score
    )


def assign_ab(
    keypoints,
    scores,
    mask_a,
    mask_b,
):

    N = len(keypoints)

    if N < 2:
        return None

    score_a = [
        candidate_score(
            keypoints[i],
            scores[i],
            mask_a
        )
        for i in range(N)
    ]

    score_b = [
        candidate_score(
            keypoints[i],
            scores[i],
            mask_b
        )
        for i in range(N)
    ]

    best = None

    for ia in range(N):

        for ib in range(N):

            if ia == ib:
                continue

            total = (
                score_a[ia]
                +
                score_b[ib]
            )

            item = (
                total,
                ia,
                ib
            )

            if (
                best is None
                or total > best[0]
            ):
                best = item

    if best is None:
        return None

    _, ia, ib = best

    return {
        "A_index":
            int(ia),

        "B_index":
            int(ib),

        "A_score":
            float(
                score_a[ia]
            ),

        "B_score":
            float(
                score_b[ib]
            ),
    }


def save_person(
    prefix,
    xy,
    scores,
    W,
    H,
):

    xy = np.asarray(
        xy,
        dtype=np.float32
    )

    scores = np.asarray(
        scores,
        dtype=np.float32
    )

    norm = xy.copy()

    norm[:, 0] /= float(W)
    norm[:, 1] /= float(H)

    np.savez_compressed(
        prefix,

        xy_native=
            xy,

        xy_normalized=
            norm,

        scores=
            scores,

        width=
            np.int32(W),

        height=
            np.int32(H),
    )


parser = argparse.ArgumentParser()

parser.add_argument(
    "--dwroot",
    type=Path,
    required=True
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

if not det.exists():
    raise RuntimeError(
        f"missing {det}"
    )

if not pose.exists():
    raise RuntimeError(
        f"missing {pose}"
    )


(
    Wholebody,
    _draw_bodypose,
    _draw_handpose,
    _draw_facepose,
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
    "frames:",
    len(rows)
)


output_rows = []

detect_count_hist = Counter()

complete = 0
incomplete = 0


for idx, row in enumerate(
    rows,
    start=1
):

    rgb = cv2.imread(
        row["rgb"],
        cv2.IMREAD_COLOR
    )

    if rgb is None:
        raise RuntimeError(
            row["rgb"]
        )

    H, W = rgb.shape[:2]

    if (
        W != 940
        or H != 1280
    ):
        print(
            "WARNING native size:",
            W,
            H,
            row["rgb"]
        )

    mask_a = read_mask(
        row["person_mask_A"]
    )

    mask_b = read_mask(
        row["person_mask_B"]
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


    if (
        not args.force
        and
        out_a.exists()
        and
        out_b.exists()
    ):

        r = dict(row)

        r.update({
            "dwpose_A":
                str(out_a),

            "dwpose_B":
                str(out_b),

            "dwpose_status":
                "cached",
        })

        output_rows.append(r)

        complete += 1

        continue


    result = estimator(
        rgb
    )

    if not isinstance(
        result,
        (tuple, list)
    ):
        raise RuntimeError(
            "Unexpected Wholebody output "
            f"type: {type(result)}"
        )

    if len(result) < 2:
        raise RuntimeError(
            "Wholebody output has "
            f"{len(result)} items"
        )


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


    N = keypoints.shape[0]

    detect_count_hist[
        int(N)
    ] += 1


    assignment = assign_ab(
        keypoints,
        scores,
        mask_a,
        mask_b
    )


    r = dict(row)

    r.update({
        "dwpose_detection_count":
            int(N),

        "dwpose_native_width":
            int(W),

        "dwpose_native_height":
            int(H),

        "dwpose_pre_resize":
            False,

        "dwpose_pre_crop":
            False,
    })


    if assignment is None:

        r[
            "dwpose_status"
        ] = "incomplete"

        output_rows.append(r)

        incomplete += 1

    else:

        ia = assignment[
            "A_index"
        ]

        ib = assignment[
            "B_index"
        ]


        save_person(
            out_a,
            keypoints[ia],
            scores[ia],
            W,
            H,
        )

        save_person(
            out_b,
            keypoints[ib],
            scores[ib],
            W,
            H,
        )


        r.update({
            "dwpose_A":
                str(out_a),

            "dwpose_B":
                str(out_b),

            "dwpose_A_candidate_index":
                ia,

            "dwpose_B_candidate_index":
                ib,

            "dwpose_A_assignment_score":
                assignment[
                    "A_score"
                ],

            "dwpose_B_assignment_score":
                assignment[
                    "B_score"
                ],

            "dwpose_status":
                "complete",
        })

        output_rows.append(r)

        complete += 1


    if (
        idx == 1
        or idx % 100 == 0
        or idx == len(rows)
    ):
        print(
            f"{idx}/{len(rows)} "
            f"complete={complete} "
            f"incomplete={incomplete}"
        )


write_jsonl(
    OUT_MANIFEST,
    output_rows
)


complete_rows = [
    x
    for x in output_rows
    if x[
        "dwpose_status"
    ] in (
        "complete",
        "cached"
    )
]


summary = {
    "input_frames":
        len(rows),

    "complete":
        len(complete_rows),

    "incomplete":
        len(rows)
        - len(complete_rows),

    "completion_ratio":
        (
            len(complete_rows)
            / len(rows)
            if rows
            else 0.0
        ),

    "detection_count_histogram":
        {
            str(k): int(v)
            for k, v in sorted(
                detect_count_hist.items()
            )
        },

    "representation": {
        "storage":
            "NPZ coordinates + confidence",

        "coordinates":
            "native and normalized",

        "pre_resize":
            False,

        "pre_crop":
            False,

        "identity_assignment":
            "Hi4D official person masks",
    },
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
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False
    )
)
