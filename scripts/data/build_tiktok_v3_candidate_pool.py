#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TikTok V3 formal candidate-pool construction.

Pipeline:
    official sequence
    -> sequence screening
    -> person-region motion estimation
    -> adaptive temporal sampling
    -> local sharpest-frame selection
    -> cheap frame QC
    -> candidate pool

This stage DOES NOT:
    - use SMPLest-X for selection
    - run DWPose
    - build source-target pairs
    - copy images
"""

# Project root for the categorized entrypoint.
from pathlib import Path as _EntryPath
import sys as _entry_sys
_entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))


from pathlib import Path
from collections import defaultdict
import json
import re

import cv2
import numpy as np


PROJECT = Path("/home/shangguanrz/project/pic-edit")

DATA_ROOT = PROJECT / "datasets/TikTokDataset"

FRAME_ROOT = (
    DATA_ROOT
    / "TikTok_dataset"
    / "TikTok_dataset"
)

V3_ROOT = DATA_ROOT / "TikTok_v3"

MANIFEST_ROOT = V3_ROOT / "manifests"

INVENTORY = (
    MANIFEST_ROOT
    / "frame_inventory.jsonl"
)

OUT_SEQUENCE = (
    MANIFEST_ROOT
    / "sequence_screening.jsonl"
)

OUT_DYNAMIC = (
    MANIFEST_ROOT
    / "dynamic_sampling_candidates.jsonl"
)

OUT_POOL = (
    MANIFEST_ROOT
    / "candidate_pool.jsonl"
)

OUT_SUMMARY = (
    MANIFEST_ROOT
    / "candidate_pool_summary.json"
)


# ============================================================
# Policy
# ============================================================

MIN_RESOLUTION = 256

# Very loose validity gate used for sequence screening.
BASIC_MIN_MASK_AREA = 0.015
BASIC_MIN_PERSON_HEIGHT = 0.20

# Slightly stronger candidate-frame gate.
CANDIDATE_MIN_MASK_AREA = 0.020
CANDIDATE_MIN_PERSON_HEIGHT = 0.25

SEQUENCE_MIN_FRAMES = 8
SEQUENCE_MIN_VALID = 4

# Approx. 30 FPS:
# static  ~0.60 s
# normal  ~0.33 s
# fast    ~0.17 s
STATIC_GAP = 18
NORMAL_GAP = 10
FAST_GAP = 5

LOCAL_RADIUS = 3

# Reject only the sequence-local bottom 10% sharpest-person score.
SHARPNESS_QUANTILE = 0.10


# ============================================================
# IO
# ============================================================

def read_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def frame_number(name, fallback):
    stem = Path(name).stem

    m = re.search(
        r"(\d+)$",
        stem,
    )

    if m:
        return int(
            m.group(1)
        )

    return fallback


def find_mask(seq, stem):
    folder = (
        FRAME_ROOT
        / seq
        / "masks"
    )

    for ext in [
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    ]:
        p = (
            folder
            / f"{stem}{ext}"
        )

        if p.exists():
            return p

    return None


# ============================================================
# Person measurements
# ============================================================

def mask_geometry(mask):
    fg = mask > 127

    if fg.sum() == 0:
        return None

    h, w = mask.shape

    ys, xs = np.where(fg)

    x0 = int(xs.min())
    x1 = int(xs.max())

    y0 = int(ys.min())
    y1 = int(ys.max())

    return {
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,

        "mask_area_ratio":
            float(
                fg.mean()
            ),

        "bbox_height_ratio":
            float(
                (y1 - y0 + 1)
                / h
            ),

        "bbox_width_ratio":
            float(
                (x1 - x0 + 1)
                / w
            ),

        # Record cropping information,
        # but DO NOT reject half-body simply for touching image edges.
        "touch_top":
            bool(y0 <= max(2, int(0.01 * h))),

        "touch_bottom":
            bool(y1 >= h - 1 - max(2, int(0.01 * h))),

        "touch_left":
            bool(x0 <= max(2, int(0.01 * w))),

        "touch_right":
            bool(x1 >= w - 1 - max(2, int(0.01 * w))),
    }


def person_sharpness(image, mask, g):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    crop = gray[
        g["y0"]:g["y1"] + 1,
        g["x0"]:g["x1"] + 1,
    ]

    crop_mask = mask[
        g["y0"]:g["y1"] + 1,
        g["x0"]:g["x1"] + 1,
    ] > 127

    if (
        crop.size == 0
        or crop_mask.sum() < 100
    ):
        return 0.0

    # Reduce mask-boundary contribution.
    eroded = cv2.erode(
        crop_mask.astype(np.uint8),
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    ) > 0

    if eroded.sum() < 100:
        eroded = crop_mask

    lap = cv2.Laplacian(
        crop,
        cv2.CV_32F,
    )

    lap_var = float(
        np.var(
            lap[eroded]
        )
    )

    gx = cv2.Sobel(
        crop,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        crop,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    grad = np.sqrt(
        gx * gx
        + gy * gy
    )

    tenengrad = float(
        np.mean(
            grad[eroded]
        )
    )

    # Used only for sequence-local ranking.
    return float(
        0.70 * np.log1p(max(tenengrad, 0.0))
        +
        0.30 * np.log1p(max(lap_var, 0.0))
    )


def exposure_ok(image, mask):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    fg = mask > 127

    if fg.sum() < 100:
        return False, 1.0, 0.0

    pixels = gray[fg]

    dark_ratio = float(
        np.mean(
            pixels < 15
        )
    )

    bright_ratio = float(
        np.mean(
            pixels > 240
        )
    )

    ok = bool(
        dark_ratio <= 0.85
        and
        bright_ratio <= 0.85
    )

    return (
        ok,
        dark_ratio,
        bright_ratio,
    )


# ============================================================
# Person-normalized motion proxy
# ============================================================

def make_thumbnail(image, mask, g):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    crop = gray[
        g["y0"]:g["y1"] + 1,
        g["x0"]:g["x1"] + 1,
    ]

    crop_mask = mask[
        g["y0"]:g["y1"] + 1,
        g["x0"]:g["x1"] + 1,
    ]

    if crop.size == 0:
        return None

    crop = cv2.resize(
        crop,
        (64, 64),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)

    crop_mask = cv2.resize(
        crop_mask,
        (64, 64),
        interpolation=cv2.INTER_NEAREST,
    )

    fg = crop_mask > 127

    if fg.sum() < 50:
        return None

    vals = crop[fg]

    mean = float(
        vals.mean()
    )

    std = max(
        float(
            vals.std()
        ),
        10.0,
    )

    crop = (
        crop - mean
    ) / std

    crop = np.clip(
        crop,
        -3.0,
        3.0,
    )

    crop[~fg] = 0.0

    return (
        crop,
        fg.astype(np.float32),
    )


def motion_distance(previous, current):
    if previous is None or current is None:
        return None

    prev_img, prev_mask = previous
    curr_img, curr_mask = current

    appearance = float(
        np.mean(
            np.abs(
                curr_img
                - prev_img
            )
        )
    )

    shape = float(
        np.mean(
            np.abs(
                curr_mask
                - prev_mask
            )
        )
    )

    return float(
        0.70 * appearance
        +
        0.30 * shape
    )


def smooth_motion(rows):
    raw = [
        r.get("motion_raw")
        for r in rows
    ]

    for i, row in enumerate(rows):
        values = []

        for j in range(
            max(0, i - 2),
            min(len(rows), i + 3),
        ):
            value = raw[j]

            if value is not None:
                values.append(
                    value
                )

        row["motion"] = (
            float(
                np.median(values)
            )
            if values
            else None
        )


# ============================================================
# Main
# ============================================================

def main():
    if not INVENTORY.exists():
        raise FileNotFoundError(
            INVENTORY
        )

    inventory = read_jsonl(
        INVENTORY
    )

    groups = defaultdict(list)

    for row in inventory:
        groups[
            row["sequence_id"]
        ].append(row)

    sequence_manifest = []
    dynamic_manifest = []
    candidate_pool = []

    print("=" * 72)
    print("TikTok V3 FORMAL CANDIDATE POOL")
    print("=" * 72)
    print("input frames :", len(inventory))
    print("sequences    :", len(groups))
    print(flush=True)

    for seq_idx, seq in enumerate(
        sorted(groups),
        1,
    ):
        source_rows = sorted(
            groups[seq],
            key=lambda r:
                frame_number(
                    r["frame_name"],
                    0,
                ),
        )

        rows = []
        previous_thumb = None

        # ----------------------------------------------------
        # Per-frame cheap metrics
        # ----------------------------------------------------

        for order, src in enumerate(source_rows):
            rgb_path = Path(
                src["rgb"]
            )

            stem = Path(
                src["frame_name"]
            ).stem

            row = {
                "sequence_id": seq,
                "stem": stem,
                "frame_name":
                    src["frame_name"],
                "frame_num":
                    frame_number(
                        src["frame_name"],
                        order,
                    ),
                "order_index": order,
                "rgb": str(rgb_path),
            }

            image = cv2.imread(
                str(rgb_path),
                cv2.IMREAD_COLOR,
            )

            if image is None:
                row["basic_valid"] = False
                row["reject_reason"] = [
                    "RGB_DECODE_FAILED"
                ]
                row["motion_raw"] = None
                rows.append(row)
                previous_thumb = None
                continue

            h, w = image.shape[:2]

            row["width"] = int(w)
            row["height"] = int(h)

            mask_path = find_mask(
                seq,
                stem,
            )

            if mask_path is None:
                row["basic_valid"] = False
                row["reject_reason"] = [
                    "MASK_MISSING"
                ]
                row["motion_raw"] = None
                rows.append(row)
                previous_thumb = None
                continue

            mask = cv2.imread(
                str(mask_path),
                cv2.IMREAD_GRAYSCALE,
            )

            if mask is None:
                row["basic_valid"] = False
                row["reject_reason"] = [
                    "MASK_DECODE_FAILED"
                ]
                row["motion_raw"] = None
                rows.append(row)
                previous_thumb = None
                continue

            if mask.shape != (h, w):
                mask = cv2.resize(
                    mask,
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )

            g = mask_geometry(
                mask
            )

            if g is None:
                row["basic_valid"] = False
                row["reject_reason"] = [
                    "EMPTY_MASK"
                ]
                row["motion_raw"] = None
                rows.append(row)
                previous_thumb = None
                continue

            sharpness = person_sharpness(
                image,
                mask,
                g,
            )

            exp_ok, dark_ratio, bright_ratio = (
                exposure_ok(
                    image,
                    mask,
                )
            )

            resolution_ok = bool(
                min(h, w)
                >=
                MIN_RESOLUTION
            )

            basic_valid = bool(
                resolution_ok
                and exp_ok
                and
                g["mask_area_ratio"]
                >=
                BASIC_MIN_MASK_AREA
                and
                g["bbox_height_ratio"]
                >=
                BASIC_MIN_PERSON_HEIGHT
            )

            candidate_prevalid = bool(
                resolution_ok
                and exp_ok
                and
                g["mask_area_ratio"]
                >=
                CANDIDATE_MIN_MASK_AREA
                and
                g["bbox_height_ratio"]
                >=
                CANDIDATE_MIN_PERSON_HEIGHT
            )

            row.update(g)

            row.update({
                "person_mask":
                    str(mask_path),

                "person_sharpness":
                    sharpness,

                "dark_ratio":
                    dark_ratio,

                "bright_ratio":
                    bright_ratio,

                "basic_valid":
                    basic_valid,

                "candidate_prevalid":
                    candidate_prevalid,
            })

            thumb = (
                make_thumbnail(
                    image,
                    mask,
                    g,
                )
                if basic_valid
                else None
            )

            row["motion_raw"] = (
                motion_distance(
                    previous_thumb,
                    thumb,
                )
            )

            if thumb is not None:
                previous_thumb = thumb

            rows.append(
                row
            )

        smooth_motion(
            rows
        )

        # ----------------------------------------------------
        # Sequence screening
        # ----------------------------------------------------

        valid_rows = [
            r
            for r in rows
            if r.get(
                "basic_valid",
                False,
            )
        ]

        n_total = len(rows)
        n_valid = len(valid_rows)

        valid_ratio = (
            n_valid
            /
            max(
                n_total,
                1,
            )
        )

        if (
            n_total < SEQUENCE_MIN_FRAMES
            or
            n_valid < SEQUENCE_MIN_VALID
        ):
            seq_status = "REJECT"

        elif valid_ratio < 0.15:
            seq_status = "REVIEW"

        else:
            seq_status = "PASS"

        sharp_values = [
            r["person_sharpness"]
            for r in valid_rows
        ]

        sharp_floor = (
            float(
                np.quantile(
                    sharp_values,
                    SHARPNESS_QUANTILE,
                )
            )
            if sharp_values
            else 0.0
        )

        motion_values = [
            r["motion"]
            for r in valid_rows
            if r.get("motion")
            is not None
        ]

        if len(motion_values) >= 3:
            q33 = float(
                np.quantile(
                    motion_values,
                    1 / 3,
                )
            )

            q67 = float(
                np.quantile(
                    motion_values,
                    2 / 3,
                )
            )

            motion_median = float(
                np.median(
                    motion_values
                )
            )

        elif motion_values:
            motion_median = float(
                np.median(
                    motion_values
                )
            )

            q33 = motion_median
            q67 = motion_median

        else:
            q33 = 0.0
            q67 = 0.0
            motion_median = 0.0

        sequence_manifest.append({
            "sequence_id":
                seq,

            "status":
                seq_status,

            "total_frames":
                n_total,

            "basic_valid_frames":
                n_valid,

            "basic_valid_ratio":
                valid_ratio,

            "sharpness_q10":
                sharp_floor,

            "motion_q33":
                q33,

            "motion_q67":
                q67,

            "official_sequence_boundary":
                True,

            "shot_detection_reapplied":
                False,
        })

        if seq_status == "REJECT":
            continue

        # ----------------------------------------------------
        # Dynamic temporal sampling
        # ----------------------------------------------------

        selected_stems = set()

        pos = 0

        while pos < len(rows):
            center = rows[pos]

            m = center.get(
                "motion"
            )

            if m is None:
                m = motion_median

            if m <= q33:
                motion_class = "STATIC"
                gap = STATIC_GAP

            elif m >= q67:
                motion_class = "FAST"
                gap = FAST_GAP

            else:
                motion_class = "NORMAL"
                gap = NORMAL_GAP

            left = max(
                0,
                pos - LOCAL_RADIUS,
            )

            right = min(
                len(rows),
                pos + LOCAL_RADIUS + 1,
            )

            local = [
                r
                for r in rows[left:right]
                if (
                    r.get(
                        "candidate_prevalid",
                        False,
                    )
                    and
                    r.get(
                        "person_sharpness",
                        0.0,
                    )
                    >=
                    sharp_floor
                )
            ]

            chosen = None

            if local:
                # First priority:
                # person-region sharpness.
                # Person size only used as tie-breaker.
                chosen = max(
                    local,
                    key=lambda r: (
                        r["person_sharpness"],
                        r["bbox_height_ratio"],
                    ),
                )

            event = {
                "sequence_id":
                    seq,

                "sequence_status":
                    seq_status,

                "center_stem":
                    center["stem"],

                "center_frame_num":
                    center["frame_num"],

                "motion_class":
                    motion_class,

                "motion_score":
                    float(m),

                "sampling_gap":
                    gap,

                "window_left":
                    left,

                "window_right":
                    right - 1,

                "chosen":
                    chosen is not None,
            }

            if chosen is not None:
                event["chosen_stem"] = (
                    chosen["stem"]
                )

                event["chosen_frame_num"] = (
                    chosen["frame_num"]
                )

                dynamic_manifest.append(
                    event
                )

                # Overlapping windows may choose same frame.
                if (
                    chosen["stem"]
                    not in selected_stems
                ):
                    selected_stems.add(
                        chosen["stem"]
                    )

                    candidate_pool.append({
                        **chosen,

                        "sequence_status":
                            seq_status,

                        "sampling_center_stem":
                            center["stem"],

                        "sampling_motion_class":
                            motion_class,

                        "sampling_motion_score":
                            float(m),

                        "sampling_gap":
                            gap,

                        "local_radius":
                            LOCAL_RADIUS,

                        "stage":
                            "CANDIDATE_POOL",

                        "smplx_used_for_selection":
                            False,

                        "dwpose_used_for_selection":
                            False,

                        "pose_checked":
                            False,
                    })

            else:
                dynamic_manifest.append(
                    event
                )

            pos += max(
                1,
                gap,
            )

        if (
            seq_idx % 20 == 0
            or
            seq_idx == len(groups)
        ):
            print(
                f"[{seq_idx}/{len(groups)}] "
                f"candidate_pool="
                f"{len(candidate_pool)}",
                flush=True,
            )

    # ========================================================
    # Save
    # ========================================================

    write_jsonl(
        OUT_SEQUENCE,
        sequence_manifest,
    )

    write_jsonl(
        OUT_DYNAMIC,
        dynamic_manifest,
    )

    write_jsonl(
        OUT_POOL,
        candidate_pool,
    )

    status_count = defaultdict(int)

    for row in sequence_manifest:
        status_count[
            row["status"]
        ] += 1

    motion_count = defaultdict(int)
    per_seq = defaultdict(int)

    for row in candidate_pool:
        motion_count[
            row[
                "sampling_motion_class"
            ]
        ] += 1

        per_seq[
            row["sequence_id"]
        ] += 1

    seq_counts = list(
        per_seq.values()
    )

    summary = {
        "version":
            "tiktok_v3_formal_candidate_pool_v2",

        "counts": {
            "input_sequences":
                len(groups),

            "input_frames":
                len(inventory),

            "sequence_pass":
                status_count["PASS"],

            "sequence_review":
                status_count["REVIEW"],

            "sequence_reject":
                status_count["REJECT"],

            "sampling_events":
                len(dynamic_manifest),

            "candidate_pool":
                len(candidate_pool),

            "sequences_with_candidates":
                len(per_seq),
        },

        "candidate_per_sequence": {
            "min":
                int(
                    min(
                        seq_counts,
                        default=0,
                    )
                ),

            "max":
                int(
                    max(
                        seq_counts,
                        default=0,
                    )
                ),

            "mean":
                float(
                    np.mean(
                        seq_counts
                    )
                )
                if seq_counts
                else 0.0,

            "median":
                float(
                    np.median(
                        seq_counts
                    )
                )
                if seq_counts
                else 0.0,
        },

        "motion_distribution": {
            "STATIC":
                motion_count["STATIC"],

            "NORMAL":
                motion_count["NORMAL"],

            "FAST":
                motion_count["FAST"],
        },

        "policy": {
            "official_sequence_as_clip":
                True,

            "repeat_shot_detection":
                False,

            "static_gap":
                STATIC_GAP,

            "normal_gap":
                NORMAL_GAP,

            "fast_gap":
                FAST_GAP,

            "local_radius":
                LOCAL_RADIUS,

            "sharpness_quantile":
                SHARPNESS_QUANTILE,

            "full_body_required":
                False,

            "smplx_used":
                False,

            "dwpose_used":
                False,
        },

        "next_stage":
            (
                "Run DWPose on candidate_pool.jsonl "
                "then perform structural fine screening."
            ),
    }

    with open(
        OUT_SUMMARY,
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
    print("=" * 72)
    print("CANDIDATE POOL COMPLETE")
    print("=" * 72)

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
