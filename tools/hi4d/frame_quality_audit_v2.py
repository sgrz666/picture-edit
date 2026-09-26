#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


PROJECT = Path("/home/shangguanrz/project/pic-edit")

OUT = PROJECT / "audit" / "frame_quality_v2"

EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


DATASETS = {
    "iPER":
        PROJECT
        / "datasets"
        / "iPER"
        / "iper_sampled_6src64tgt",

    "TikTok":
        PROJECT
        / "datasets"
        / "TikTokDataset"
        / "TikTok_dataset"
        / "TikTok_dataset",

    "Hi4D":
        PROJECT
        / "datasets"
        / "Hi4D_pilot_v1",
}


# ============================================================
# 基础工具
# ============================================================

def to_builtin(x):

    if isinstance(x, dict):
        return {
            str(k): to_builtin(v)
            for k, v in x.items()
        }

    if isinstance(x, (list, tuple)):
        return [
            to_builtin(v)
            for v in x
        ]

    if isinstance(x, np.generic):
        return x.item()

    if isinstance(x, Path):
        return str(x)

    return x


def read_img(path: Path, gray=False):

    try:
        buf = np.fromfile(
            str(path),
            dtype=np.uint8
        )

        flag = (
            cv2.IMREAD_GRAYSCALE
            if gray
            else cv2.IMREAD_COLOR
        )

        return cv2.imdecode(
            buf,
            flag
        )

    except Exception:
        return None


def write_jpg(
    path: Path,
    img,
    quality=90
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    ok, buf = cv2.imencode(
        ".jpg",
        img,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            quality
        ]
    )

    if ok:
        buf.tofile(
            str(path)
        )


def resize_max(
    img,
    max_side=512
):

    h, w = img.shape[:2]

    scale = min(
        1.0,
        max_side / max(h, w)
    )

    if scale >= 0.999:
        return img

    nw = max(
        1,
        round(w * scale)
    )

    nh = max(
        1,
        round(h * scale)
    )

    return cv2.resize(
        img,
        (nw, nh),
        interpolation=cv2.INTER_AREA
    )


# ============================================================
# 清晰度指标
# ============================================================

def edge_metrics(
    gray,
    valid=None
):

    lap = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    gx = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    ten = gx * gx + gy * gy

    if valid is None:

        return (
            float(np.var(lap)),
            float(np.mean(ten))
        )

    mask = valid > 0

    if int(mask.sum()) < 100:

        return (
            np.nan,
            np.nan
        )

    return (
        float(
            np.var(
                lap[mask]
            )
        ),

        float(
            np.mean(
                ten[mask]
            )
        ),
    )


def full_metrics(img):

    img = resize_max(
        img,
        512
    )

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    return edge_metrics(
        gray
    )


def person_metrics(
    img,
    mask
):

    if mask is None:

        return (
            np.nan,
            np.nan,
            np.nan
        )

    h, w = img.shape[:2]

    if mask.shape[:2] != (h, w):

        mask = cv2.resize(
            mask,
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )

    mask = (
        (mask > 0)
        .astype(np.uint8)
        * 255
    )

    ys, xs = np.where(
        mask > 0
    )

    if len(xs) < 100:

        return (
            np.nan,
            np.nan,
            np.nan
        )

    x1 = int(
        xs.min()
    )

    x2 = int(
        xs.max()
    ) + 1

    y1 = int(
        ys.min()
    )

    y2 = int(
        ys.max()
    ) + 1

    crop = img[
        y1:y2,
        x1:x2
    ]

    crop_mask = mask[
        y1:y2,
        x1:x2
    ]

    person_ratio = float(
        (mask > 0).mean()
    )

    if crop.size == 0:

        return (
            np.nan,
            np.nan,
            person_ratio
        )

    scale = min(
        1.0,
        512 / max(
            crop.shape[:2]
        )
    )

    if scale < 0.999:

        nw = max(
            1,
            round(
                crop.shape[1]
                * scale
            )
        )

        nh = max(
            1,
            round(
                crop.shape[0]
                * scale
            )
        )

        crop = cv2.resize(
            crop,
            (nw, nh),
            interpolation=cv2.INTER_AREA
        )

        crop_mask = cv2.resize(
            crop_mask,
            (nw, nh),
            interpolation=cv2.INTER_NEAREST
        )

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )

    # 避免 mask 边界本身制造假锐度
    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    crop_mask = cv2.erode(
        crop_mask,
        kernel,
        iterations=1
    )

    lap, ten = edge_metrics(
        gray,
        crop_mask
    )

    return (
        lap,
        ten,
        person_ratio
    )


# ============================================================
# 数据发现
# ============================================================

def discover_iper():

    root = DATASETS[
        "iPER"
    ]

    files = sorted({
        p.resolve()

        for p in root.glob(
            "*/rgb_1024/*/*"
        )

        if (
            p.is_file()
            and
            p.suffix.lower()
            in EXTS
        )
    })

    return (
        root,
        files
    )


def discover_tiktok():

    root = DATASETS[
        "TikTok"
    ]

    files = sorted({
        p.resolve()

        for p in root.glob(
            "*/images/*"
        )

        if (
            p.is_file()
            and
            p.suffix.lower()
            in EXTS
        )
    })

    return (
        root,
        files
    )


def discover_hi4d():

    base = DATASETS[
        "Hi4D"
    ]

    candidates = [
        base / "raw",
        base / "final_assets_native_v1",
    ]

    bad_words = (
        "mask",
        "seg",
        "densepose",
        "depth",
        "normal",
        "smpl",
        "dwpose",
        "contact",
        "occlusion",
        "overlay",
        "vis",
        "semantic",
    )

    for root in candidates:

        if not root.exists():
            continue

        files = []

        for p in root.rglob("*"):

            if not p.is_file():
                continue

            if (
                p.suffix.lower()
                not in EXTS
            ):
                continue

            text = str(
                p
            ).lower()

            if any(
                x in text
                for x in bad_words
            ):
                continue

            parts = {
                x.lower()
                for x in p.parts
            }

            if (
                "images" in parts
                or
                "image" in parts
                or
                "rgb" in parts
                or
                root.name == "raw"
            ):

                files.append(
                    p.resolve()
                )

        if files:

            return (
                root,
                sorted(
                    set(files)
                )
            )

    return (
        base,
        []
    )


# ============================================================
# frame key
# ============================================================

def frame_key(path: Path):

    match = re.search(
        r"(?:^|[_-])f?(\d{3,8})(?:\D|$)",
        path.stem
    )

    if match:
        return match.group(1)

    return path.stem.lower()


def nearest_pair_name(
    path: Path
):

    for part in reversed(
        path.parts
    ):

        if part.lower().startswith(
            "pair"
        ):
            return part

    return ""


# ============================================================
# 构建 iPER mask 索引
# ============================================================

def build_iper_mask_index(
    root: Path
):

    index = defaultdict(
        list
    )

    if not root.exists():
        return index

    for p in root.rglob("*"):

        if not p.is_file():
            continue

        if (
            p.suffix.lower()
            not in EXTS
        ):
            continue

        low = str(
            p
        ).lower()

        if (
            "mask" not in low
            and
            "/seg/" not in low
            and
            "\\seg\\" not in low
        ):
            continue

        seq = ""

        try:

            rel = p.relative_to(
                root
            )

            seq = rel.parts[
                0
            ]

        except Exception:
            pass

        key = (
            seq,
            frame_key(p)
        )

        index[
            key
        ].append(
            p
        )

    return index


# ============================================================
# 构建 Hi4D mask 索引
# ============================================================

def build_hi4d_mask_index(
    base: Path
):

    index = defaultdict(
        list
    )

    roots = [
        base / "raw",
        base / "final_assets_native_v1",
    ]

    for root in roots:

        if not root.exists():
            continue

        for p in root.rglob("*"):

            if not p.is_file():
                continue

            if (
                p.suffix.lower()
                not in EXTS
            ):
                continue

            low = str(
                p
            ).lower()

            if (
                "img_seg_mask" not in low
                and
                "mask" not in low
                and
                "/seg/" not in low
                and
                "\\seg\\" not in low
            ):
                continue

            pair = nearest_pair_name(
                p
            )

            key = (
                pair,
                frame_key(p)
            )

            index[
                key
            ].append(
                p
            )

    return index


IPER_MASK_INDEX = None

HI4D_MASK_INDEX = None


# ============================================================
# mask 匹配
# ============================================================

def get_mask(
    dataset: str,
    path: Path
):

    global IPER_MASK_INDEX
    global HI4D_MASK_INDEX

    # --------------------------------------------------------
    # TikTok
    # --------------------------------------------------------

    if dataset == "TikTok":

        mask_dir = (
            path.parent.parent
            / "masks"
        )

        candidates = [
            mask_dir
            / path.name
        ]

        candidates += [
            mask_dir
            / (
                path.stem
                + ext
            )

            for ext in EXTS
        ]

        for candidate in candidates:

            if candidate.exists():

                return read_img(
                    candidate,
                    gray=True
                )

        return None

    # --------------------------------------------------------
    # iPER
    # --------------------------------------------------------

    if dataset == "iPER":

        if IPER_MASK_INDEX is None:

            IPER_MASK_INDEX = (
                build_iper_mask_index(
                    DATASETS[
                        "iPER"
                    ]
                )
            )

        try:

            seq = (
                path
                .relative_to(
                    DATASETS[
                        "iPER"
                    ]
                )
                .parts[0]
            )

        except Exception:

            seq = ""

        key = (
            seq,
            frame_key(path)
        )

        candidates = (
            IPER_MASK_INDEX
            .get(
                key,
                []
            )
        )

        if len(
            candidates
        ) == 1:

            return read_img(
                candidates[0],
                gray=True
            )

        # 多个候选时优先 person / smpl / seg
        for p in candidates:

            low = str(
                p
            ).lower()

            if (
                "person" in low
                or
                "smpl" in low
                or
                "img_seg_mask" in low
            ):

                return read_img(
                    p,
                    gray=True
                )

        return None

    # --------------------------------------------------------
    # Hi4D
    # --------------------------------------------------------

    if dataset == "Hi4D":

        if HI4D_MASK_INDEX is None:

            HI4D_MASK_INDEX = (
                build_hi4d_mask_index(
                    DATASETS[
                        "Hi4D"
                    ]
                )
            )

        pair = nearest_pair_name(
            path
        )

        key = (
            pair,
            frame_key(path)
        )

        candidates = (
            HI4D_MASK_INDEX
            .get(
                key,
                []
            )
        )

        if len(
            candidates
        ) == 1:

            return read_img(
                candidates[0],
                gray=True
            )

        # 优先原始 segmentation
        for p in candidates:

            low = str(
                p
            ).lower()

            if (
                "img_seg_mask" in low
                or
                "person" in low
            ):

                return read_img(
                    p,
                    gray=True
                )

        return None

    return None


# ============================================================
# 统计工具
# ============================================================

def percentile(
    values,
    q
):

    arr = np.asarray(
        [
            float(v)

            for v in values

            if np.isfinite(
                float(v)
            )
        ],
        dtype=float
    )

    if len(arr) == 0:
        return np.nan

    return float(
        np.percentile(
            arr,
            q
        )
    )


def rank01(values):

    values = np.asarray(
        values,
        dtype=float
    )

    order = np.argsort(
        values
    )

    rank = np.empty(
        len(values),
        dtype=float
    )

    rank[
        order
    ] = np.arange(
        len(values),
        dtype=float
    )

    return (
        rank
        /
        max(
            1,
            len(values) - 1
        )
    )


# ============================================================
# dataset group
# ============================================================

def role_from_iper_path(
    path: str
):

    parts = Path(
        path
    ).parts

    for role in (
        "source_standard",
        "source_motion",
        "target",
    ):

        if role in parts:
            return role

    return "unknown"


def video_from_tiktok_path(
    path: str
):

    path = Path(
        path
    )

    try:

        return (
            path
            .relative_to(
                DATASETS[
                    "TikTok"
                ]
            )
            .parts[0]
        )

    except Exception:

        return "unknown"


def pair_from_hi4d_path(
    path: str
):

    return (
        nearest_pair_name(
            Path(path)
        )
        or
        "unknown"
    )


# ============================================================
# CSV
# ============================================================

def row_to_csv_safe(
    row
):

    result = {}

    for key, value in row.items():

        if (
            isinstance(
                value,
                float
            )
            and
            math.isnan(
                value
            )
        ):

            result[
                key
            ] = ""

        elif isinstance(
            value,
            np.generic
        ):

            result[
                key
            ] = value.item()

        else:

            result[
                key
            ] = value

    return result


def load_existing_report(
    dataset
):

    candidates = [

        OUT
        / dataset
        / "frame_quality_report.csv",

        PROJECT
        / "audit"
        / "frame_quality_3datasets"
        / dataset
        / "frame_quality_report.csv",
    ]

    path = next(
        (
            x

            for x in candidates

            if x.exists()
        ),
        None
    )

    if path is None:
        return None

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8-sig"
    ) as f:

        reader = csv.DictReader(
            f
        )

        for row in reader:

            r = dict(
                row
            )

            for key in (
                "width",
                "height",
                "short_side",
            ):

                if (
                    key in r
                    and
                    r[key] != ""
                ):

                    r[key] = int(
                        float(
                            r[key]
                        )
                    )

            for key in (
                "person_area_ratio",
                "full_lap",
                "full_ten",
                "person_lap",
                "person_ten",
                "primary_lap",
                "primary_ten",
                "quality_rank",
            ):

                if key in r:

                    try:

                        r[key] = float(
                            r[key]
                        )

                    except Exception:

                        r[key] = np.nan

            for key in (
                "low_res_lt512",
                "mask_found",
                "severe_suspect",
                "suspect_blur",
            ):

                if key in r:

                    r[key] = (
                        str(
                            r[key]
                        )
                        .lower()
                        in (
                            "1",
                            "true",
                            "yes",
                        )
                    )

            rows.append(
                r
            )

    return rows


def save_report(
    dataset,
    rows
):

    outdir = (
        OUT
        / dataset
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    path = (
        outdir
        / "frame_quality_report.csv"
    )

    fields = list(
        rows[0].keys()
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                row_to_csv_safe(
                    row
                )
            )

    return path


# ============================================================
# 主审计
# ============================================================

def audit_dataset(
    dataset,
    root,
    files
):

    rows = []

    for i, path in enumerate(
        files,
        1
    ):

        if (
            i == 1
            or
            i % 500 == 0
            or
            i == len(files)
        ):

            print(
                f"[{dataset}] "
                f"{i}/{len(files)} "
                f"{path.name}",
                flush=True
            )

        img = read_img(
            path
        )

        if img is None:
            continue

        h, w = img.shape[:2]

        full_lap, full_ten = (
            full_metrics(
                img
            )
        )

        mask = get_mask(
            dataset,
            path
        )

        (
            person_lap,
            person_ten,
            person_ratio
        ) = person_metrics(
            img,
            mask
        )

        use_person = (
            np.isfinite(
                person_lap
            )
            and
            np.isfinite(
                person_ten
            )
        )

        row = {

            "dataset":
                dataset,

            "path":
                str(path),

            "relative_path":
                str(
                    path.relative_to(
                        root
                    )
                ),

            "width":
                int(w),

            "height":
                int(h),

            "short_side":
                int(
                    min(
                        w,
                        h
                    )
                ),

            "low_res_lt512":
                bool(
                    min(
                        w,
                        h
                    ) < 512
                ),

            "mask_found":
                bool(
                    mask is not None
                    and
                    use_person
                ),

            "person_area_ratio":
                (
                    float(
                        person_ratio
                    )
                    if np.isfinite(
                        person_ratio
                    )
                    else np.nan
                ),

            "full_lap":
                float(
                    full_lap
                ),

            "full_ten":
                float(
                    full_ten
                ),

            "person_lap":
                (
                    float(
                        person_lap
                    )
                    if np.isfinite(
                        person_lap
                    )
                    else np.nan
                ),

            "person_ten":
                (
                    float(
                        person_ten
                    )
                    if np.isfinite(
                        person_ten
                    )
                    else np.nan
                ),

            "primary_type":
                (
                    "person"
                    if use_person
                    else "full"
                ),

            "primary_lap":
                float(
                    person_lap
                    if use_person
                    else full_lap
                ),

            "primary_ten":
                float(
                    person_ten
                    if use_person
                    else full_ten
                ),
        }

        rows.append(
            row
        )

    laps = [
        r[
            "primary_lap"
        ]
        for r in rows
    ]

    tens = [
        r[
            "primary_ten"
        ]
        for r in rows
    ]

    lap_p5 = percentile(
        laps,
        5
    )

    lap_p10 = percentile(
        laps,
        10
    )

    ten_p5 = percentile(
        tens,
        5
    )

    ten_p10 = percentile(
        tens,
        10
    )

    quality = (
        rank01(
            laps
        )
        +
        rank01(
            tens
        )
    ) / 2.0

    for row, q in zip(
        rows,
        quality
    ):

        row[
            "quality_rank"
        ] = float(
            q
        )

        row[
            "severe_suspect"
        ] = bool(
            row[
                "primary_lap"
            ] <= lap_p5

            and

            row[
                "primary_ten"
            ] <= ten_p5
        )

        row[
            "suspect_blur"
        ] = bool(
            row[
                "primary_lap"
            ] <= lap_p10

            and

            row[
                "primary_ten"
            ] <= ten_p10
        )

    rows.sort(
        key=lambda x:
        x[
            "quality_rank"
        ]
    )

    save_report(
        dataset,
        rows
    )

    return rows


# ============================================================
# Summary
# ============================================================

def summary_from_rows(
    dataset,
    rows
):

    return {

        "dataset":
            dataset,

        "total_frames":
            len(
                rows
            ),

        "mask_found_count":
            sum(
                bool(
                    r.get(
                        "mask_found"
                    )
                )

                for r in rows
            ),

        "mask_found_ratio":
            sum(
                bool(
                    r.get(
                        "mask_found"
                    )
                )

                for r in rows
            )
            /
            max(
                1,
                len(rows)
            ),

        "low_res_lt512_count":
            sum(
                bool(
                    r.get(
                        "low_res_lt512"
                    )
                )

                for r in rows
            ),

        "severe_suspect_count":
            sum(
                bool(
                    r.get(
                        "severe_suspect"
                    )
                )

                for r in rows
            ),

        "suspect_blur_count":
            sum(
                bool(
                    r.get(
                        "suspect_blur"
                    )
                )

                for r in rows
            ),

        "primary_lap_p5":
            percentile(
                [
                    r[
                        "primary_lap"
                    ]
                    for r in rows
                ],
                5
            ),

        "primary_lap_p10":
            percentile(
                [
                    r[
                        "primary_lap"
                    ]
                    for r in rows
                ],
                10
            ),

        "primary_lap_p50":
            percentile(
                [
                    r[
                        "primary_lap"
                    ]
                    for r in rows
                ],
                50
            ),

        "primary_ten_p5":
            percentile(
                [
                    r[
                        "primary_ten"
                    ]
                    for r in rows
                ],
                5
            ),

        "primary_ten_p10":
            percentile(
                [
                    r[
                        "primary_ten"
                    ]
                    for r in rows
                ],
                10
            ),

        "primary_ten_p50":
            percentile(
                [
                    r[
                        "primary_ten"
                    ]
                    for r in rows
                ],
                50
            ),
    }


# ============================================================
# 分组统计
# ============================================================

def save_group_stats(
    dataset,
    rows
):

    if dataset == "TikTok":

        keyfn = (
            video_from_tiktok_path
        )

        filename = (
            "blur_by_video.csv"
        )

        keyname = (
            "video_id"
        )

    elif dataset == "iPER":

        keyfn = (
            role_from_iper_path
        )

        filename = (
            "blur_by_role.csv"
        )

        keyname = (
            "role"
        )

    else:

        keyfn = (
            pair_from_hi4d_path
        )

        filename = (
            "blur_by_pair.csv"
        )

        keyname = (
            "pair_id"
        )

    groups = defaultdict(
        list
    )

    for row in rows:

        groups[
            keyfn(
                row[
                    "path"
                ]
            )
        ].append(
            row
        )

    output = (
        OUT
        / dataset
        / filename
    )

    with open(
        output,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        fields = [

            keyname,

            "total_frames",

            "mask_found",

            "severe_count",

            "severe_ratio",

            "suspect_count",

            "suspect_ratio",

            "median_primary_lap",

            "median_primary_ten",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        for group, rr in sorted(
            groups.items()
        ):

            total = len(
                rr
            )

            severe = sum(
                bool(
                    x.get(
                        "severe_suspect"
                    )
                )

                for x in rr
            )

            suspect = sum(
                bool(
                    x.get(
                        "suspect_blur"
                    )
                )

                for x in rr
            )

            writer.writerow({

                keyname:
                    group,

                "total_frames":
                    total,

                "mask_found":
                    sum(
                        bool(
                            x.get(
                                "mask_found"
                            )
                        )

                        for x in rr
                    ),

                "severe_count":
                    severe,

                "severe_ratio":
                    severe
                    /
                    max(
                        1,
                        total
                    ),

                "suspect_count":
                    suspect,

                "suspect_ratio":
                    suspect
                    /
                    max(
                        1,
                        total
                    ),

                "median_primary_lap":
                    percentile(
                        [
                            x[
                                "primary_lap"
                            ]
                            for x in rr
                        ],
                        50
                    ),

                "median_primary_ten":
                    percentile(
                        [
                            x[
                                "primary_ten"
                            ]
                            for x in rr
                        ],
                        50
                    ),
            })

    return output


# ============================================================
# 可视化
# ============================================================

def preview_image(
    row,
    label
):

    img = read_img(
        Path(
            row[
                "path"
            ]
        )
    )

    if img is None:
        return None

    img = resize_max(
        img,
        560
    )

    h, w = img.shape[:2]

    panel = np.zeros(
        (
            90,
            w,
            3
        ),
        dtype=np.uint8
    )

    lines = [

        (
            f"{label}  "
            f"{row['dataset']}  "
            f"{Path(row['path']).name}"
        ),

        (
            f"type={row['primary_type']} "
            f"lap={row['primary_lap']:.1f} "
            f"ten={row['primary_ten']:.1f} "
            f"mask={row['mask_found']}"
        ),

        (
            f"size={row['width']}x{row['height']} "
            f"suspect={row.get('suspect_blur', False)} "
            f"severe={row.get('severe_suspect', False)}"
        ),
    ]

    y = 22

    for text in lines:

        cv2.putText(
            panel,
            text,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        y += 26

    return np.vstack(
        [
            panel,
            img
        ]
    )


def save_contact_sheet(
    images,
    output_path,
    cols=5,
    thumb_w=260
):

    if not images:
        return

    resized = []

    for img in images:

        scale = (
            thumb_w
            /
            img.shape[1]
        )

        new_h = max(
            1,
            round(
                img.shape[0]
                * scale
            )
        )

        resized.append(
            cv2.resize(
                img,
                (
                    thumb_w,
                    new_h
                ),
                interpolation=cv2.INTER_AREA
            )
        )

    max_h = max(
        x.shape[0]
        for x in resized
    )

    blank = np.zeros(
        (
            max_h,
            thumb_w,
            3
        ),
        dtype=np.uint8
    )

    rows = []

    for i in range(
        0,
        len(resized),
        cols
    ):

        chunk = resized[
            i:i + cols
        ]

        padded = []

        for img in chunk:

            canvas = (
                blank.copy()
            )

            canvas[
                :img.shape[0],
                :img.shape[1]
            ] = img

            padded.append(
                canvas
            )

        while len(
            padded
        ) < cols:

            padded.append(
                blank.copy()
            )

        rows.append(
            np.hstack(
                padded
            )
        )

    sheet = np.vstack(
        rows
    )

    write_jpg(
        output_path,
        sheet,
        88
    )


def save_worst100(
    dataset,
    rows
):

    images = []

    for rank, row in enumerate(
        rows[:100],
        1
    ):

        preview = preview_image(
            row,
            f"rank={rank}"
        )

        if preview is not None:

            images.append(
                preview
            )

    save_contact_sheet(
        images,

        OUT
        / dataset
        / "worst_100_contact_sheet.jpg"
    )


def save_percentile_probe(
    dataset,
    rows
):

    # 边界区间：
    # P1 P3 P5 P7 P10 P15 P20
    percentiles = [
        1,
        3,
        5,
        7,
        10,
        15,
        20,
    ]

    images = []

    n = len(
        rows
    )

    rng = np.random.default_rng(
        2026
    )

    for q in percentiles:

        center = int(
            round(
                (
                    q
                    /
                    100.0
                )
                *
                (
                    n - 1
                )
            )
        )

        window = max(
            3,
            n // 500
        )

        lo = max(
            0,
            center - window
        )

        hi = min(
            n,
            center + window + 1
        )

        pool = np.arange(
            lo,
            hi
        )

        take = min(
            8,
            len(pool)
        )

        selected = np.sort(
            rng.choice(
                pool,
                size=take,
                replace=False
            )
        )

        for idx in selected:

            row = rows[
                int(idx)
            ]

            preview = preview_image(
                row,
                f"P{q}"
            )

            if preview is not None:

                images.append(
                    preview
                )

    save_contact_sheet(
        images,

        OUT
        / dataset
        / "percentile_probe_P1_P20.jpg",

        cols=4
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[
            "iPER",
            "TikTok",
            "Hi4D",
        ],
        default=[
            "iPER",
            "TikTok",
            "Hi4D",
        ]
    )

    parser.add_argument(
        "--reuse-existing",
        action="store_true"
    )

    parser.add_argument(
        "--force-recompute",
        nargs="*",
        default=[]
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    discover = {

        "iPER":
            discover_iper,

        "TikTok":
            discover_tiktok,

        "Hi4D":
            discover_hi4d,
    }

    summaries = []

    force = set(
        args.force_recompute
    )

    for dataset in args.datasets:

        print()
        print(
            "=" * 70
        )

        print(
            dataset
        )

        print(
            "=" * 70
        )

        outdir = (
            OUT
            / dataset
        )

        outdir.mkdir(
            parents=True,
            exist_ok=True
        )

        rows = None

        # ----------------------------------------------------
        # 复用旧结果
        # ----------------------------------------------------

        if (
            args.reuse_existing
            and
            dataset not in force
        ):

            rows = (
                load_existing_report(
                    dataset
                )
            )

            if rows:

                print(
                    f"[{dataset}] "
                    f"reuse existing report: "
                    f"{len(rows)} rows",
                    flush=True
                )

                # 同时复制到 v2 输出目录
                save_report(
                    dataset,
                    rows
                )

        # ----------------------------------------------------
        # 重新计算
        # ----------------------------------------------------

        if rows is None:

            root, files = (
                discover[
                    dataset
                ]()
            )

            print(
                f"[{dataset}] root = {root}",
                flush=True
            )

            print(
                f"[{dataset}] frames = {len(files)}",
                flush=True
            )

            if not files:

                print(
                    f"[{dataset}] WARNING: "
                    f"没有找到 RGB",
                    flush=True
                )

                continue

            rows = audit_dataset(
                dataset,
                root,
                files
            )

        # 确保按最差 -> 最好排序
        rows.sort(
            key=lambda x:
            float(
                x.get(
                    "quality_rank",
                    1.0
                )
            )
        )

        summary = (
            summary_from_rows(
                dataset,
                rows
            )
        )

        with open(
            outdir
            / "summary.json",
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                to_builtin(
                    summary
                ),
                f,
                ensure_ascii=False,
                indent=2
            )

        save_group_stats(
            dataset,
            rows
        )

        save_worst100(
            dataset,
            rows
        )

        save_percentile_probe(
            dataset,
            rows
        )

        summaries.append(
            summary
        )

        print(
            json.dumps(
                to_builtin(
                    summary
                ),
                ensure_ascii=False,
                indent=2
            ),
            flush=True
        )

    # ========================================================
    # combined summary
    # ========================================================

    if summaries:

        fields = [

            "dataset",

            "total_frames",

            "mask_found_count",

            "mask_found_ratio",

            "low_res_lt512_count",

            "severe_suspect_count",

            "suspect_blur_count",
        ]

        with open(
            OUT
            / "combined_summary.csv",
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fields
            )

            writer.writeheader()

            for summary in summaries:

                writer.writerow({

                    key:
                        to_builtin(
                            summary[
                                key
                            ]
                        )

                    for key in fields
                })

    print()
    print(
        "=" * 70
    )

    print(
        "ALL DONE"
    )

    print(
        "=" * 70
    )

    print(
        OUT
    )


if __name__ == "__main__":
    main()
