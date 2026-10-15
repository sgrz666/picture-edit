#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("/home/shangguanrz/project/pic-edit")

IN_JSONL = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/qc/"
      "visual_quality_final_v1/global_hard_blur.jsonl"
)

OUT_ROOT = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/qc/"
      "visual_quality_final_v1/global_hard_review"
)

PER_PAGE = 12
COLS = 2


def read_jsonl(path):
    return [
        json.loads(x)
        for x in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]


def read_bgr(path):
    if path is None:
        return None

    return cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )


def build_index(directory):
    directory = Path(directory)

    out = {}

    if not directory.is_dir():
        return out

    for p in directory.iterdir():
        if not p.is_file():
            continue

        try:
            n = int(p.stem)
        except ValueError:
            continue

        out[n] = p

    return out


def resize_h(img, h=270):
    if img is None:
        return np.zeros(
            (h, int(h * 0.56), 3),
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

    canvas = np.zeros(
        (H, width, 3),
        dtype=np.uint8,
    )

    x = (width - W) // 2

    canvas[:, x:x+W] = img

    return canvas


def make_panel(row):
    current_path = Path(
        row["rgb"]
    )

    seq_dir = current_path.parent
    idx = build_index(seq_dir)

    try:
        frame_num = int(
            row["stem"]
        )
    except Exception:
        frame_num = int(
            current_path.stem
        )

    prev_img = read_bgr(
        idx.get(frame_num - 1)
    )

    cur_img = read_bgr(
        current_path
    )

    next_img = read_bgr(
        idx.get(frame_num + 1)
    )

    views = [
        resize_h(prev_img),
        resize_h(cur_img),
        resize_h(next_img),
    ]

    cell_w = max(
        x.shape[1]
        for x in views
    )

    views = [
        fit_width(x, cell_w)
        for x in views
    ]

    strip = np.concatenate(
        views,
        axis=1,
    )

    # current frame white border
    cv2.rectangle(
        strip,
        (cell_w, 0),
        (cell_w * 2 - 1, strip.shape[0] - 1),
        (255, 255, 255),
        4,
    )

    header_h = 72

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
        f"{row['sequence_id']}/{row['stem']}  "
        f"CURRENT = middle"
    )

    line2 = (
        f"recover={row['temporal_recovery_ratio']:.3f}  "
        f"seqEdgeP={row['seq_edge_percentile']:.3f}"
    )

    cv2.putText(
        panel,
        line1,
        (6, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        panel,
        line2,
        (6, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return panel


def save_pages(rows):
    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    panels = [
        make_panel(r)
        for r in rows
    ]

    outputs = []

    for start in range(
        0,
        len(panels),
        PER_PAGE,
    ):
        chunk = panels[
            start:start + PER_PAGE
        ]

        max_h = max(
            p.shape[0]
            for p in chunk
        )

        max_w = max(
            p.shape[1]
            for p in chunk
        )

        fixed = []

        for p in chunk:
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

            fixed.append(canvas)

        rows_out = []

        for i in range(
            0,
            len(fixed),
            COLS,
        ):
            line = fixed[
                i:i + COLS
            ]

            while len(line) < COLS:
                line.append(
                    np.zeros_like(
                        fixed[0]
                    )
                )

            rows_out.append(
                np.concatenate(
                    line,
                    axis=1,
                )
            )

        sheet = np.concatenate(
            rows_out,
            axis=0,
        )

        page = start // PER_PAGE + 1

        out = (
            OUT_ROOT
            / f"global_hard_blur_review_{page:02d}.jpg"
        )

        cv2.imwrite(
            str(out),
            sheet,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                94,
            ],
        )

        outputs.append(out)

    return outputs


def main():
    rows = read_jsonl(
        IN_JSONL
    )

    # strongest evidence first
    rows.sort(
        key=lambda x: (
            x[
                "temporal_recovery_ratio"
            ],
            -x[
                "seq_edge_percentile"
            ],
        ),
        reverse=True,
    )

    outputs = save_pages(
        rows
    )

    print("=" * 72)
    print("FINAL GLOBAL BLUR REVIEW")
    print("=" * 72)
    print("frames:", len(rows))

    for p in outputs:
        print(p)


if __name__ == "__main__":
    main()
