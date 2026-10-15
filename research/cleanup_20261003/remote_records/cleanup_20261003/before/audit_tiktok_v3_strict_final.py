#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("/home/shangguanrz/project/pic-edit")

PAIR_ROOT = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/manifests/"
      "pairs_strict_v1"
)

STRICT_ROOT = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/manifests/"
      "strict_visual_v1"
)

OUT_ROOT = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/qc/"
      "strict_final_audit"
)

OUT_SHEET = (
    OUT_ROOT
    / "strict_pair_sample_24.jpg"
)

SPLITS = [
    "train",
    "val",
    "test",
]

EXPECTED_PAIRS = {
    "train": 8574,
    "val": 1120,
    "test": 1156,
}

SEED = 2026
N_SAMPLE = 24


def read_jsonl(path):
    return [
        json.loads(x)
        for x in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]


def resize_h(img, h=300):
    if img is None:
        return np.zeros(
            (h, int(h * 0.65), 3),
            dtype=np.uint8,
        )

    H, W = img.shape[:2]

    scale = h / H

    return cv2.resize(
        img,
        (
            max(1, int(W * scale)),
            h,
        ),
        interpolation=cv2.INTER_AREA,
    )


def fit_width(img, width):
    H, W = img.shape[:2]

    if W == width:
        return img

    if W > width:
        return cv2.resize(
            img,
            (width, H),
            interpolation=cv2.INTER_AREA,
        )

    out = np.zeros(
        (H, width, 3),
        dtype=np.uint8,
    )

    x = (width - W) // 2

    out[:, x:x + W] = img

    return out


def make_panel(row):
    src = cv2.imread(
        row["source_rgb"]
    )

    tgt = cv2.imread(
        row["target_rgb"]
    )

    pose = cv2.imread(
        row["target_dwpose"]
    )

    normal = cv2.imread(
        row["target_normal"]
    )

    views = [
        resize_h(src),
        resize_h(tgt),
        resize_h(pose),
        resize_h(normal),
    ]

    cell_w = max(
        x.shape[1]
        for x in views
    )

    views = [
        fit_width(
            x,
            cell_w,
        )
        for x in views
    ]

    strip = np.concatenate(
        views,
        axis=1,
    )

    header_h = 62

    panel = np.zeros(
        (
            strip.shape[0] + header_h,
            strip.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    panel[
        header_h:,
        :
    ] = strip

    line1 = (
        f"{row['split']} "
        f"{row['sequence_id']}  "
        f"{row['source_stem']} -> {row['target_stem']}  "
        f"poseDist={row['pose_distance']:.3f}"
    )

    line2 = (
        "SOURCE RGB | TARGET RGB | TARGET DWPOSE | TARGET NORMAL"
    )

    cv2.putText(
        panel,
        line1,
        (5, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        panel,
        line2,
        (5, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return panel


def main():
    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    strict_keys = {}

    for split in SPLITS:
        rows = read_jsonl(
            STRICT_ROOT
            / f"{split}.jsonl"
        )

        strict_keys[
            split
        ] = {
            (
                r["sequence_id"],
                r["stem"],
            )
            for r in rows
        }

    all_pairs = []

    group_sets = {}

    missing_paths = 0
    same_frame = 0
    outside_strict_source = 0
    outside_strict_target = 0
    wrong_sequence = 0
    bad_pair_count = 0

    print("=" * 82)
    print("TIKTOK V3 STRICT FINAL AUDIT")
    print("=" * 82)

    required = [
        "source_rgb",
        "target_rgb",
        "source_person_mask",
        "target_person_mask",
        "target_dwpose",
        "target_normal",
        "target_depth_vis",
        "target_smplx_mask",
    ]

    for split in SPLITS:
        path = (
            PAIR_ROOT
            / f"{split}_pairs.jsonl"
        )

        rows = read_jsonl(
            path
        )

        all_pairs.extend(
            rows
        )

        if len(rows) != EXPECTED_PAIRS[split]:
            bad_pair_count += 1

        groups = {
            r["source_video_group"]
            for r in rows
        }

        group_sets[
            split
        ] = groups

        low_001 = sum(
            float(r["pose_distance"]) < 0.01
            for r in rows
        )

        low_003 = sum(
            float(r["pose_distance"]) < 0.03
            for r in rows
        )

        low_005 = sum(
            float(r["pose_distance"]) < 0.05
            for r in rows
        )

        print()
        print(
            f"{split.upper():5s} "
            f"pairs={len(rows):5d}  "
            f"<.01={low_001:4d}  "
            f"<.03={low_003:4d}  "
            f"<.05={low_005:4d}"
        )

        for r in rows:
            skey = (
                r["sequence_id"],
                r["source_stem"],
            )

            tkey = (
                r["sequence_id"],
                r["target_stem"],
            )

            if (
                r["source_stem"]
                ==
                r["target_stem"]
            ):
                same_frame += 1

            if (
                skey
                not in strict_keys[split]
            ):
                outside_strict_source += 1

            if (
                tkey
                not in strict_keys[split]
            ):
                outside_strict_target += 1

            for k in required:
                if not Path(
                    r[k]
                ).is_file():

                    missing_paths += 1

                    print(
                        "MISSING:",
                        split,
                        r["pair_id"],
                        k,
                        r[k],
                    )

    leakage = 0

    print()
    print("GROUP LEAKAGE")

    for a, b in [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ]:
        overlap = (
            group_sets[a]
            &
            group_sets[b]
        )

        leakage += len(
            overlap
        )

        print(
            f"{a:5s} vs {b:5s}:",
            len(overlap),
            sorted(overlap),
        )

    # --------------------------------------------------------
    # Unpaired
    # --------------------------------------------------------

    print()
    print("UNPAIRED TARGETS")

    total_unpaired = 0

    for split in SPLITS:
        p = (
            PAIR_ROOT
            / f"{split}_unpaired_targets.jsonl"
        )

        rows = (
            read_jsonl(p)
            if p.exists()
            else []
        )

        total_unpaired += len(
            rows
        )

        print(
            split,
            len(rows),
        )

        for r in rows:
            print(
                " ",
                r,
            )

    # --------------------------------------------------------
    # Random pair sample
    # --------------------------------------------------------

    rng = random.Random(
        SEED
    )

    sample = rng.sample(
        all_pairs,
        min(
            N_SAMPLE,
            len(all_pairs),
        ),
    )

    panels = [
        make_panel(r)
        for r in sample
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

    COLS = 2

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

    cv2.imwrite(
        str(OUT_SHEET),
        sheet,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            93,
        ],
    )

    total_pairs = len(
        all_pairs
    )

    expected_total = sum(
        EXPECTED_PAIRS.values()
    )

    ok = (
        total_pairs == expected_total
        and bad_pair_count == 0
        and missing_paths == 0
        and same_frame == 0
        and outside_strict_source == 0
        and outside_strict_target == 0
        and leakage == 0
    )

    print()
    print("=" * 82)
    print("FINAL")
    print("=" * 82)

    print(
        "total_pairs              :",
        total_pairs,
    )

    print(
        "expected_total           :",
        expected_total,
    )

    print(
        "missing_paths            :",
        missing_paths,
    )

    print(
        "source==target           :",
        same_frame,
    )

    print(
        "source_outside_strict    :",
        outside_strict_source,
    )

    print(
        "target_outside_strict    :",
        outside_strict_target,
    )

    print(
        "group_leakage            :",
        leakage,
    )

    print(
        "unpaired_targets         :",
        total_unpaired,
    )

    print(
        "sample_sheet             :",
        OUT_SHEET,
    )

    print()

    print(
        "RESULT:",
        "STRICT FINAL AUDIT PASS"
        if ok
        else
        "STRICT FINAL AUDIT NEEDS REVIEW"
    )


if __name__ == "__main__":
    main()
