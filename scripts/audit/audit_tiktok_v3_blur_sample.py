#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Project root for the categorized entrypoint.
from pathlib import Path as _EntryPath
import sys as _entry_sys
_entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))

import json
import random
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
      "final_blur_audit"
)

CSV_OUT = (
    OUT_ROOT
    / "blur_scores.csv"
)

RANDOM_SHEET = (
    OUT_ROOT
    / "random_24.jpg"
)

BLURRY_SHEET = (
    OUT_ROOT
    / "blurriest_24.jpg"
)

SEED = 2026
N = 24


def read_jsonl(path):
    return [
        json.loads(x)
        for x in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]


def read_gray(path):
    img = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if img is None:
        raise RuntimeError(
            f"failed to read: {path}"
        )

    return img


def load_binary_mask(path, shape):
    mask = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        return np.ones(
            shape,
            dtype=np.uint8,
        )

    if mask.shape != shape:
        mask = cv2.resize(
            mask,
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    return (
        mask > 20
    ).astype(np.uint8)


def sharpness_scores(rgb_path, mask_path):
    """
    返回：
    1. whole-image Laplacian variance
    2. person-region Laplacian variance

    person-region 更重要，因为背景本身可能虚化。
    """

    gray = read_gray(rgb_path)

    mask = load_binary_mask(
        mask_path,
        gray.shape,
    )

    lap = cv2.Laplacian(
        gray,
        cv2.CV_64F,
    )

    whole_score = float(
        lap.var()
    )

    person_pixels = (
        lap[mask > 0]
    )

    if person_pixels.size < 100:
        person_score = whole_score
    else:
        person_score = float(
            person_pixels.var()
        )

    return (
        whole_score,
        person_score,
    )


def make_panel(row):
    rgb = cv2.imread(
        row["rgb"]
    )

    pose = cv2.imread(
        row["dwpose_png"]
    )

    if rgb is None:
        return None

    H, W = rgb.shape[:2]

    if pose is None:
        pose = np.zeros_like(
            rgb
        )
    elif pose.shape[:2] != (H, W):
        pose = cv2.resize(
            pose,
            (W, H),
            interpolation=cv2.INTER_LINEAR,
        )

    overlay = cv2.addWeighted(
        rgb,
        0.68,
        pose,
        0.85,
        0,
    )

    target_h = 320

    scale = (
        target_h / H
    )

    target_w = max(
        1,
        int(W * scale),
    )

    rgb_small = cv2.resize(
        rgb,
        (target_w, target_h),
        interpolation=cv2.INTER_AREA,
    )

    overlay_small = cv2.resize(
        overlay,
        (target_w, target_h),
        interpolation=cv2.INTER_AREA,
    )

    panel = np.concatenate(
        [
            rgb_small,
            overlay_small,
        ],
        axis=1,
    )

    cv2.rectangle(
        panel,
        (0, 0),
        (
            panel.shape[1],
            34,
        ),
        (0, 0, 0),
        -1,
    )

    text = (
        f"{row['split']} "
        f"{row['sequence_id']}/{row['stem']}  "
        f"personSharp={row['person_sharpness']:.1f}"
    )

    cv2.putText(
        panel,
        text,
        (5, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return panel


def save_sheet(rows, out_path):
    panels = []

    for row in rows:
        p = make_panel(row)

        if p is not None:
            panels.append(p)

    if not panels:
        raise RuntimeError(
            "no panels"
        )

    max_w = max(
        p.shape[1]
        for p in panels
    )

    max_h = max(
        p.shape[0]
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

    cols = 4

    rows_out = []

    for i in range(
        0,
        len(fixed),
        cols,
    ):
        chunk = fixed[
            i:i + cols
        ]

        while (
            len(chunk)
            < cols
        ):
            chunk.append(
                np.zeros_like(
                    fixed[0]
                )
            )

        rows_out.append(
            np.concatenate(
                chunk,
                axis=1,
            )
        )

    sheet = np.concatenate(
        rows_out,
        axis=0,
    )

    cv2.imwrite(
        str(out_path),
        sheet,
    )


def main():
    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = read_jsonl(
        MANIFEST
    )

    scored = []

    for i, row in enumerate(
        rows,
        1,
    ):
        whole, person = (
            sharpness_scores(
                row["rgb"],
                row["person_mask"],
            )
        )

        x = dict(row)

        x[
            "whole_sharpness"
        ] = whole

        x[
            "person_sharpness"
        ] = person

        scored.append(
            x
        )

        if i % 1000 == 0:
            print(
                f"{i}/{len(rows)}",
                flush=True,
            )

    vals = np.array(
        [
            r["person_sharpness"]
            for r in scored
        ],
        dtype=np.float64,
    )

    print()
    print("=" * 76)
    print(
        "FINAL TIKTOK PERSON SHARPNESS"
    )
    print("=" * 76)

    print(
        "frames:",
        len(vals),
    )

    for q in [
        0.00,
        0.01,
        0.02,
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
        print(
            f"q{q:4.2f}: "
            f"{np.quantile(vals, q):.3f}"
        )

    # CSV
    with CSV_OUT.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "split,sequence_id,stem,"
            "whole_sharpness,person_sharpness,rgb\n"
        )

        for r in scored:
            f.write(
                f"{r['split']},"
                f"{r['sequence_id']},"
                f"{r['stem']},"
                f"{r['whole_sharpness']:.6f},"
                f"{r['person_sharpness']:.6f},"
                f"{r['rgb']}\n"
            )

    # ---------------------------
    # Random 24
    # ---------------------------

    rng = random.Random(
        SEED
    )

    random_rows = rng.sample(
        scored,
        min(
            N,
            len(scored),
        ),
    )

    save_sheet(
        random_rows,
        RANDOM_SHEET,
    )

    # ---------------------------
    # Blurriest 24
    # ---------------------------

    blurry_rows = sorted(
        scored,
        key=lambda r:
            r["person_sharpness"],
    )[:N]

    save_sheet(
        blurry_rows,
        BLURRY_SHEET,
    )

    print()
    print(
        "random sheet :",
        RANDOM_SHEET,
    )

    print(
        "blurriest    :",
        BLURRY_SHEET,
    )

    print(
        "score csv    :",
        CSV_OUT,
    )


if __name__ == "__main__":
    main()
