#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("/home/shangguanrz/project/pic-edit")

STRICT_ROOT = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/manifests/"
      "strict_visual_v1"
)

OUT_ROOT = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/qc/"
      "pose_redundancy_v1"
)

SUMMARY_OUT = (
    OUT_ROOT
    / "pose_redundancy_summary.json"
)

DETAIL_OUT = (
    OUT_ROOT
    / "pose_redundancy_per_frame.jsonl"
)

PRIMARY_RADIUS = 2
RADII = [1, 2, 3]

BODY_SCORE_THR = 0.30

THRESHOLDS = [
    0.01,
    0.03,
    0.05,
    0.08,
    0.10,
]

TOP_N = 32


def read_jsonl(path):
    return [
        json.loads(x)
        for x in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]


def write_jsonl(path, rows):
    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for r in rows:
            f.write(
                json.dumps(
                    r,
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_pose_descriptor(row):
    obj = json.loads(
        Path(
            row["dwpose_json"]
        ).read_text(
            encoding="utf-8"
        )
    )

    kp = np.asarray(
        obj["keypoints_norm"],
        dtype=np.float32,
    )[:18]

    score = np.asarray(
        obj["scores"],
        dtype=np.float32,
    )[:18]

    valid = (
        np.isfinite(kp).all(axis=1)
        &
        np.isfinite(score)
        &
        (score > BODY_SCORE_THR)
    )

    desc = np.zeros(
        (18, 2),
        dtype=np.float32,
    )

    if int(valid.sum()) < 2:
        return desc, valid

    pts = kp[valid]

    # remove translation
    center = pts.mean(
        axis=0,
        keepdims=True,
    )

    # normalize body scale
    span = (
        pts.max(axis=0)
        -
        pts.min(axis=0)
    )

    scale = float(
        np.linalg.norm(span)
    )

    if scale < 1e-6:
        scale = 1.0

    desc[valid] = (
        kp[valid]
        -
        center
    ) / scale

    return desc, valid


def pose_distance(a, b):
    da, va = a
    db, vb = b

    common = va & vb

    if int(common.sum()) < 4:
        return float("inf")

    d = np.linalg.norm(
        da[common]
        -
        db[common],
        axis=1,
    )

    return float(
        d.mean()
    )


def nearest_for_radius(
    index,
    seq_rows,
    pose_cache,
    radius,
):
    """
    Exclude temporal neighbors in the STRICT-frame ordering.

    radius=1:
        exclude previous/next strict frame

    radius=2:
        exclude previous/next two strict frames

    radius=3:
        exclude previous/next three strict frames
    """

    row = seq_rows[index]

    key = (
        row["sequence_id"],
        row["stem"],
    )

    best = None

    for j, other in enumerate(
        seq_rows
    ):
        if j == index:
            continue

        if abs(j - index) <= radius:
            continue

        okey = (
            other["sequence_id"],
            other["stem"],
        )

        d = pose_distance(
            pose_cache[key],
            pose_cache[okey],
        )

        if not np.isfinite(d):
            continue

        if (
            best is None
            or d < best["distance"]
        ):
            try:
                frame_gap = abs(
                    int(row["stem"])
                    -
                    int(other["stem"])
                )
            except Exception:
                frame_gap = None

            best = {
                "distance": float(d),
                "stem": other["stem"],
                "rank_gap": abs(
                    j - index
                ),
                "frame_gap": frame_gap,
                "rgb": other["rgb"],
                "dwpose_png":
                    other["dwpose_png"],
            }

    return best


def summarize_values(values):
    values = np.asarray(
        [
            x
            for x in values
            if np.isfinite(x)
        ],
        dtype=np.float64,
    )

    if len(values) == 0:
        return {}

    out = {
        "count": int(
            len(values)
        ),
        "quantiles": {},
        "thresholds": {},
    }

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
        out[
            "quantiles"
        ][str(q)] = float(
            np.quantile(
                values,
                q,
            )
        )

    for thr in THRESHOLDS:
        n = int(
            (
                values < thr
            ).sum()
        )

        out[
            "thresholds"
        ][f"lt_{thr:.2f}"] = {
            "count": n,
            "ratio": float(
                n / len(values)
            ),
        }

    return out


def read_img(path):
    if not path:
        return None

    return cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )


def resize_h(
    img,
    h=260,
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

    scale = h / H

    return cv2.resize(
        img,
        (
            max(
                1,
                int(W * scale),
            ),
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


def make_pair_panel(item):
    """
    A RGB | A DWPose | B RGB | B DWPose
    """

    imgs = [
        read_img(
            item["rgb"]
        ),
        read_img(
            item["dwpose_png"]
        ),
        read_img(
            item["nearest_rgb"]
        ),
        read_img(
            item["nearest_dwpose_png"]
        ),
    ]

    imgs = [
        resize_h(x)
        for x in imgs
    ]

    cell_w = max(
        x.shape[1]
        for x in imgs
    )

    imgs = [
        fit_width(
            x,
            cell_w,
        )
        for x in imgs
    ]

    strip = np.concatenate(
        imgs,
        axis=1,
    )

    header_h = 65

    canvas = np.zeros(
        (
            strip.shape[0]
            + header_h,
            strip.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    canvas[
        header_h:,
        :
    ] = strip

    line1 = (
        f"{item['split']} "
        f"{item['sequence_id']} "
        f"{item['stem']} <-> "
        f"{item['nearest_stem']}  "
        f"poseDist={item['distance']:.4f}"
    )

    line2 = (
        f"strict-rank-gap="
        f"{item['rank_gap']}  "
        f"raw-frame-gap="
        f"{item['frame_gap']}  |  "
        f"A RGB | A Pose | B RGB | B Pose"
    )

    cv2.putText(
        canvas,
        line1,
        (5, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        line2,
        (5, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return canvas


def save_contact_sheets(
    items,
):
    if not items:
        return []

    PER_PAGE = 16
    COLS = 2

    outputs = []

    for start in range(
        0,
        len(items),
        PER_PAGE,
    ):
        chunk = items[
            start:
            start + PER_PAGE
        ]

        panels = [
            make_pair_panel(x)
            for x in chunk
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
            start // PER_PAGE
            + 1
        )

        out = (
            OUT_ROOT
            / f"train_nonlocal_pose_top32_{page:02d}.jpg"
        )

        cv2.imwrite(
            str(out),
            sheet,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                93,
            ],
        )

        outputs.append(
            str(out)
        )

    return outputs


def main():
    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_rows = []

    for split in [
        "train",
        "val",
        "test",
    ]:
        rows = read_jsonl(
            STRICT_ROOT
            / f"{split}.jsonl"
        )

        all_rows.extend(
            rows
        )

    print("=" * 80)
    print(
        "TikTok V3 STRICT POSE REDUNDANCY AUDIT"
    )
    print("=" * 80)
    print(
        "frames:",
        len(all_rows),
    )

    # --------------------------------------------------------
    # Pose cache
    # --------------------------------------------------------

    pose_cache = {}

    for i, row in enumerate(
        all_rows,
        1,
    ):
        key = (
            row["sequence_id"],
            row["stem"],
        )

        pose_cache[
            key
        ] = load_pose_descriptor(
            row
        )

        if (
            i % 500 == 0
            or i == len(all_rows)
        ):
            print(
                f"pose cache {i}/{len(all_rows)}",
                flush=True,
            )

    # --------------------------------------------------------
    # Group by split + sequence
    # --------------------------------------------------------

    by_split_seq = defaultdict(
        lambda:
            defaultdict(list)
    )

    for row in all_rows:
        by_split_seq[
            row["split"]
        ][
            row["sequence_id"]
        ].append(row)

    for split in by_split_seq:
        for seq in by_split_seq[
            split
        ]:
            by_split_seq[
                split
            ][
                seq
            ].sort(
                key=lambda r: int(
                    r["stem"]
                )
            )

    details = []

    # --------------------------------------------------------
    # Nearest pose search
    # --------------------------------------------------------

    for split in [
        "train",
        "val",
        "test",
    ]:

        print(
            f"processing {split}...",
            flush=True,
        )

        for seq, seq_rows in (
            by_split_seq[
                split
            ].items()
        ):

            for i, row in enumerate(
                seq_rows
            ):

                x = {
                    "split":
                        split,

                    "sequence_id":
                        seq,

                    "stem":
                        row["stem"],

                    "rgb":
                        row["rgb"],

                    "dwpose_png":
                        row["dwpose_png"],

                    "sequence_strict_count":
                        len(seq_rows),
                }

                for radius in RADII:

                    best = nearest_for_radius(
                        index=i,
                        seq_rows=seq_rows,
                        pose_cache=pose_cache,
                        radius=radius,
                    )

                    prefix = (
                        f"exclude_rank_radius_{radius}"
                    )

                    if best is None:
                        x[
                            f"{prefix}_distance"
                        ] = None

                        x[
                            f"{prefix}_nearest_stem"
                        ] = None

                        x[
                            f"{prefix}_rank_gap"
                        ] = None

                        x[
                            f"{prefix}_frame_gap"
                        ] = None

                    else:
                        x[
                            f"{prefix}_distance"
                        ] = best[
                            "distance"
                        ]

                        x[
                            f"{prefix}_nearest_stem"
                        ] = best[
                            "stem"
                        ]

                        x[
                            f"{prefix}_rank_gap"
                        ] = best[
                            "rank_gap"
                        ]

                        x[
                            f"{prefix}_frame_gap"
                        ] = best[
                            "frame_gap"
                        ]

                        if (
                            radius
                            ==
                            PRIMARY_RADIUS
                        ):
                            x[
                                "nearest_rgb"
                            ] = best[
                                "rgb"
                            ]

                            x[
                                "nearest_dwpose_png"
                            ] = best[
                                "dwpose_png"
                            ]

                details.append(
                    x
                )

    write_jsonl(
        DETAIL_OUT,
        details,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "version":
            "tiktok_v3_pose_redundancy_v1",

        "input_frames":
            len(all_rows),

        "definition": {
            "pose_descriptor":
                (
                    "DWPose 18 body joints; "
                    "translation removed; "
                    "body-scale normalized; "
                    "confidence > 0.3"
                ),

            "primary_nonlocal_definition":
                (
                    "exclude previous/next 2 "
                    "strict frames in the same sequence"
                ),

            "reported_exclusion_radii":
                RADII,

            "threshold_note":
                (
                    "0.01/0.03/0.05/0.08/0.10 "
                    "are descriptive audit bins, "
                    "not deletion thresholds."
                ),
        },

        "splits": {},
        "overall": {},
    }

    for split in [
        "train",
        "val",
        "test",
    ]:

        split_rows = [
            x
            for x in details
            if x[
                "split"
            ]
            ==
            split
        ]

        split_summary = {
            "frames":
                len(
                    split_rows
                ),

            "radii":
                {},
        }

        for radius in RADII:

            field = (
                f"exclude_rank_radius_{radius}_distance"
            )

            vals = [
                float(x[field])
                for x in split_rows
                if x.get(field)
                is not None
            ]

            split_summary[
                "radii"
            ][
                str(radius)
            ] = summarize_values(
                vals
            )

        primary_gap_values = [
            float(
                x[
                    f"exclude_rank_radius_{PRIMARY_RADIUS}_frame_gap"
                ]
            )
            for x in split_rows
            if x.get(
                f"exclude_rank_radius_{PRIMARY_RADIUS}_frame_gap"
            )
            is not None
        ]

        split_summary[
            "primary_nearest_raw_frame_gap"
        ] = summarize_values(
            primary_gap_values
        )

        summary[
            "splits"
        ][
            split
        ] = split_summary

    # overall
    for radius in RADII:

        field = (
            f"exclude_rank_radius_{radius}_distance"
        )

        vals = [
            float(x[field])
            for x in details
            if x.get(field)
            is not None
        ]

        summary[
            "overall"
        ][
            f"radius_{radius}"
        ] = summarize_values(
            vals
        )

    # --------------------------------------------------------
    # Top-32 most redundant NONLOCAL TRAIN pairs
    #
    # de-duplicate mutual nearest-neighbor pairs.
    # --------------------------------------------------------

    candidates = []

    seen_pairs = set()

    distance_field = (
        f"exclude_rank_radius_{PRIMARY_RADIUS}_distance"
    )

    nearest_field = (
        f"exclude_rank_radius_{PRIMARY_RADIUS}_nearest_stem"
    )

    rank_gap_field = (
        f"exclude_rank_radius_{PRIMARY_RADIUS}_rank_gap"
    )

    frame_gap_field = (
        f"exclude_rank_radius_{PRIMARY_RADIUS}_frame_gap"
    )

    for x in details:

        if x[
            "split"
        ] != "train":
            continue

        if x.get(
            distance_field
        ) is None:
            continue

        pair_key = tuple(
            sorted(
                [
                    (
                        x[
                            "sequence_id"
                        ],
                        x[
                            "stem"
                        ],
                    ),
                    (
                        x[
                            "sequence_id"
                        ],
                        x[
                            nearest_field
                        ],
                    ),
                ]
            )
        )

        if pair_key in seen_pairs:
            continue

        seen_pairs.add(
            pair_key
        )

        candidates.append({
            "split":
                x["split"],

            "sequence_id":
                x[
                    "sequence_id"
                ],

            "stem":
                x["stem"],

            "nearest_stem":
                x[
                    nearest_field
                ],

            "distance":
                float(
                    x[
                        distance_field
                    ]
                ),

            "rank_gap":
                int(
                    x[
                        rank_gap_field
                    ]
                ),

            "frame_gap":
                int(
                    x[
                        frame_gap_field
                    ]
                ),

            "rgb":
                x["rgb"],

            "dwpose_png":
                x[
                    "dwpose_png"
                ],

            "nearest_rgb":
                x[
                    "nearest_rgb"
                ],

            "nearest_dwpose_png":
                x[
                    "nearest_dwpose_png"
                ],
        })

    candidates.sort(
        key=lambda x:
            x[
                "distance"
            ]
    )

    top = candidates[
        :TOP_N
    ]

    contact_sheets = (
        save_contact_sheets(
            top
        )
    )

    summary[
        "top_nonlocal_train_pairs"
    ] = {
        "count":
            len(top),

        "contact_sheets":
            contact_sheets,
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
    print("=" * 80)
    print(
        "POSE REDUNDANCY AUDIT COMPLETE"
    )
    print("=" * 80)

    print(
        "summary:",
        SUMMARY_OUT,
    )

    print(
        "details:",
        DETAIL_OUT,
    )

    for p in contact_sheets:
        print(
            "contact sheet:",
            p,
        )


if __name__ == "__main__":
    main()
