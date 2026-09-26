#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hi4D V4.2 Source-Target Pair QC

Purpose
-------
Visualize the *actual* V4.2 pairing relationship without recomputing geometry.

Panel:
    Row 1:
        Source RGB
        Target RGB
        Target DWPose
        Target Depth

    Row 2:
        Target Normal
        Target Semantic
        Target Contact
        Target Directional Occlusion

Important
---------
- Source must be NON-CONTACT.
- Target must be CONTACT.
- Source/Target must be same identity pair and same camera.
- Target visualizations are reused from final_assets_native_v1/vis.
- No geometry is re-rendered.
- Colored PNGs are QC visualization only, NOT training tensors.
"""

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# ============================================================
# Paths
# ============================================================

PROJECT = Path(
    "/home/shangguanrz/project/pic-edit"
)

ROOT = (
    PROJECT
    / "datasets"
    / "Hi4D_pilot_v1"
)

FINAL_ROOT = (
    ROOT
    / "final_assets_v4_2"
)

PAIR_MANIFEST = (
    FINAL_ROOT
    / "manifests"
    / "main_pairs_all.jsonl"
)

NATIVE_TARGET_ROOT = (
    ROOT
    / "final_assets_native_v1"
)

DEFAULT_OUT = (
    FINAL_ROOT
    / "pair_qc"
)


# ============================================================
# QC layout
# ============================================================

CELL_W = 282
CELL_H = 384

GRID_COLS = 4
GRID_ROWS = 2

HEADER_H = 104
FOOTER_H = 32

PANEL_W = CELL_W * GRID_COLS
PANEL_H = (
    HEADER_H
    + CELL_H * GRID_ROWS
    + FOOTER_H
)


# ============================================================
# IO
# ============================================================

def read_jsonl(path):
    rows = []

    with open(
        path,
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


def safe_name(text):
    return (
        str(text)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


# ============================================================
# Font
# ============================================================

def load_font(size=20):

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]

    for p in candidates:

        path = Path(p)

        if path.exists():

            try:
                return ImageFont.truetype(
                    str(path),
                    size=size,
                )

            except Exception:
                pass

    return ImageFont.load_default()


FONT_TITLE = load_font(22)
FONT_META = load_font(18)
FONT_CELL = load_font(18)
FONT_SMALL = load_font(15)


# ============================================================
# Image helpers
# ============================================================

def open_rgb(path):

    path = Path(path)

    if not path.exists():
        return None

    try:

        with Image.open(path) as im:
            return im.convert("RGB")

    except Exception:
        return None


def fit_image(
    image,
    width,
    height,
    background=(0, 0, 0),
):

    if image is None:

        canvas = Image.new(
            "RGB",
            (width, height),
            (35, 35, 35),
        )

        draw = ImageDraw.Draw(
            canvas
        )

        text = "MISSING"

        bbox = draw.textbbox(
            (0, 0),
            text,
            font=FONT_TITLE,
        )

        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        draw.text(
            (
                (width - tw) // 2,
                (height - th) // 2,
            ),
            text,
            fill=(230, 230, 230),
            font=FONT_TITLE,
        )

        return canvas


    image = image.copy()

    iw, ih = image.size

    scale = min(
        width / iw,
        height / ih,
    )

    nw = max(
        1,
        int(
            round(
                iw * scale
            )
        )
    )

    nh = max(
        1,
        int(
            round(
                ih * scale
            )
        )
    )

    image = image.resize(
        (nw, nh),
        Image.Resampling.LANCZOS,
    )


    canvas = Image.new(
        "RGB",
        (width, height),
        background,
    )

    x = (
        width - nw
    ) // 2

    y = (
        height - nh
    ) // 2

    canvas.paste(
        image,
        (x, y),
    )

    return canvas


def draw_cell(
    panel,
    image,
    col,
    row,
    label,
):

    x0 = (
        col
        * CELL_W
    )

    y0 = (
        HEADER_H
        +
        row
        * CELL_H
    )


    tile = fit_image(
        image,
        CELL_W,
        CELL_H,
    )


    panel.paste(
        tile,
        (x0, y0),
    )


    draw = ImageDraw.Draw(
        panel
    )


    # Label strip.
    strip_h = 30

    draw.rectangle(
        [
            x0,
            y0,
            x0 + CELL_W,
            y0 + strip_h,
        ],
        fill=(0, 0, 0),
    )


    draw.text(
        (
            x0 + 8,
            y0 + 5,
        ),
        label,
        fill=(255, 255, 255),
        font=FONT_CELL,
    )


    # Thin border.
    draw.rectangle(
        [
            x0,
            y0,
            x0 + CELL_W - 1,
            y0 + CELL_H - 1,
        ],
        outline=(95, 95, 95),
        width=1,
    )


# ============================================================
# Existing Target visualization paths
# ============================================================

def get_target_stem(row):

    # final_assets_v4_2:
    #
    # .../targets/pairXX/actionXX/camX/FRAME/rgb.jpg
    #
    # Parent directory is therefore the verified frame_stem.

    target_rgb = Path(
        row["target_rgb"]
    )

    return target_rgb.parent.name


def native_vis_paths(row):

    pair = str(
        row["pair"]
    )

    action = str(
        row["target_action"]
    )

    cam = int(
        row["camera_id"]
    )

    stem = get_target_stem(
        row
    )


    base = (
        NATIVE_TARGET_ROOT
        / pair
        / action
        / f"cam{cam}"
        / "vis"
    )


    return {
        "dwpose":
            base
            / "dwpose"
            / f"{stem}.png",

        "depth":
            base
            / "depth"
            / f"{stem}.png",

        "normal":
            base
            / "normal"
            / f"{stem}.png",

        "semantic":
            base
            / "semantic"
            / f"{stem}.png",

        "contact":
            base
            / "contact"
            / f"{stem}.png",

        "directional_occlusion":
            base
            / "directional_occlusion"
            / f"{stem}.png",
    }


# ============================================================
# Hard pairing validation
# ============================================================

def validate_pair(row):

    problems = []


    if bool(
        row.get(
            "source_contact",
            False
        )
    ):

        problems.append(
            "SOURCE_IS_CONTACT"
        )


    if not bool(
        row.get(
            "target_contact",
            True
        )
    ):

        problems.append(
            "TARGET_NOT_CONTACT"
        )


    relation = str(
        row[
            "pair_relation"
        ]
    )


    same_action = (
        str(
            row[
                "source_action"
            ]
        )
        ==
        str(
            row[
                "target_action"
            ]
        )
    )


    if (
        relation
        ==
        "CROSS_ACTION"
        and
        same_action
    ):

        problems.append(
            "CROSS_LABEL_BUT_SAME_ACTION"
        )


    if (
        relation
        ==
        "SAME_ACTION"
        and
        not same_action
    ):

        problems.append(
            "SAME_LABEL_BUT_CROSS_ACTION"
        )


    return problems


# ============================================================
# Sampling
# ============================================================

def stratified_sample(
    rows,
    per_stratum,
    seed,
):

    groups = defaultdict(
        list
    )


    for row in rows:

        key = (
            str(
                row["split"]
            ),
            str(
                row[
                    "pair_relation"
                ]
            ),
        )

        groups[
            key
        ].append(
            row
        )


    rng = random.Random(
        seed
    )


    selected = []


    for key in sorted(
        groups.keys()
    ):

        pool = list(
            groups[
                key
            ]
        )

        rng.shuffle(
            pool
        )

        take = min(
            per_stratum,
            len(pool),
        )


        selected.extend(
            pool[
                :take
            ]
        )


    return selected, groups


# ============================================================
# Panel generation
# ============================================================

def make_panel(
    row,
    output_path,
):

    source_rgb = open_rgb(
        row[
            "source_rgb"
        ]
    )

    target_rgb = open_rgb(
        row[
            "target_rgb"
        ]
    )


    vis_paths = native_vis_paths(
        row
    )


    imgs = {
        "dwpose":
            open_rgb(
                vis_paths[
                    "dwpose"
                ]
            ),

        "depth":
            open_rgb(
                vis_paths[
                    "depth"
                ]
            ),

        "normal":
            open_rgb(
                vis_paths[
                    "normal"
                ]
            ),

        "semantic":
            open_rgb(
                vis_paths[
                    "semantic"
                ]
            ),

        "contact":
            open_rgb(
                vis_paths[
                    "contact"
                ]
            ),

        "directional_occlusion":
            open_rgb(
                vis_paths[
                    "directional_occlusion"
                ]
            ),
    }


    missing = []


    if source_rgb is None:
        missing.append(
            str(
                row[
                    "source_rgb"
                ]
            )
        )


    if target_rgb is None:
        missing.append(
            str(
                row[
                    "target_rgb"
                ]
            )
        )


    for name, path in vis_paths.items():

        if imgs[
            name
        ] is None:

            missing.append(
                str(path)
            )


    panel = Image.new(
        "RGB",
        (
            PANEL_W,
            PANEL_H,
        ),
        (18, 18, 18),
    )


    draw = ImageDraw.Draw(
        panel
    )


    split = str(
        row[
            "split"
        ]
    )

    pair = str(
        row[
            "pair"
        ]
    )

    cam = int(
        row[
            "camera_id"
        ]
    )

    relation = str(
        row[
            "pair_relation"
        ]
    )


    source_action = str(
        row[
            "source_action"
        ]
    )

    source_frame = int(
        row[
            "source_frame_id"
        ]
    )


    target_action = str(
        row[
            "target_action"
        ]
    )

    target_frame = int(
        row[
            "target_frame_id"
        ]
    )


    title = (
        f"{split.upper()} | "
        f"{pair} | "
        f"cam{cam} | "
        f"{relation}"
    )


    line2 = (
        f"Source: {source_action} / "
        f"{source_frame:06d} / NON-CONTACT"
    )


    line3 = (
        f"Target: {target_action} / "
        f"{target_frame:06d} / CONTACT"
    )


    draw.text(
        (14, 8),
        title,
        fill=(255, 255, 255),
        font=FONT_TITLE,
    )


    draw.text(
        (14, 40),
        line2,
        fill=(225, 225, 225),
        font=FONT_META,
    )


    draw.text(
        (14, 66),
        line3,
        fill=(225, 225, 225),
        font=FONT_META,
    )


    # --------------------------------------------
    # 2 x 4
    # --------------------------------------------

    draw_cell(
        panel,
        source_rgb,
        0,
        0,
        "Source RGB",
    )

    draw_cell(
        panel,
        target_rgb,
        1,
        0,
        "Target RGB",
    )

    draw_cell(
        panel,
        imgs["dwpose"],
        2,
        0,
        "Target DWPose",
    )

    draw_cell(
        panel,
        imgs["depth"],
        3,
        0,
        "Target Depth",
    )

    draw_cell(
        panel,
        imgs["normal"],
        0,
        1,
        "Target Normal",
    )

    draw_cell(
        panel,
        imgs["semantic"],
        1,
        1,
        "Semantic A/B",
    )

    draw_cell(
        panel,
        imgs["contact"],
        2,
        1,
        "Contact A/B",
    )

    draw_cell(
        panel,
        imgs[
            "directional_occlusion"
        ],
        3,
        1,
        "Directional Occlusion",
    )


    footer_y = (
        HEADER_H
        +
        GRID_ROWS
        * CELL_H
        +
        6
    )


    draw.text(
        (
            10,
            footer_y,
        ),
        (
            "QC visualization only | "
            "Depth/Normal/Semantic/Contact/Occlusion "
            "are reused from verified native assets"
        ),
        fill=(180, 180, 180),
        font=FONT_SMALL,
    )


    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    panel.save(
        output_path,
        quality=94,
        subsampling=0,
    )


    return missing


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--manifest",
        type=Path,
        default=PAIR_MANIFEST,
    )


    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
    )


    parser.add_argument(
        "--per-stratum",
        type=int,
        default=25,
        help=(
            "Number sampled from each "
            "split x relation stratum. "
            "Default 25 => up to 150 QC panels."
        ),
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )


    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Generate QC for all 2740 pairs "
            "instead of stratified sampling."
        ),
    )


    args = parser.parse_args()


    rows = read_jsonl(
        args.manifest
    )


    print()
    print("=" * 80)
    print("Hi4D V4.2 PAIR QC")
    print("=" * 80)

    print(
        "manifest:",
        args.manifest
    )

    print(
        "pair rows:",
        len(rows)
    )


    # --------------------------------------------
    # Validate pairing metadata
    # --------------------------------------------

    invalid_rows = []


    for row in rows:

        problems = validate_pair(
            row
        )

        if problems:

            invalid_rows.append(
                (
                    row,
                    problems,
                )
            )


    if invalid_rows:

        print()
        print(
            "PAIR VALIDATION FAILED:"
        )

        for row, problems in invalid_rows[
            :20
        ]:

            print(
                row[
                    "pair"
                ],
                row[
                    "source_action"
                ],
                row[
                    "source_frame_id"
                ],
                "->",
                row[
                    "target_action"
                ],
                row[
                    "target_frame_id"
                ],
                problems,
            )

        raise RuntimeError(
            f"invalid pair rows: "
            f"{len(invalid_rows)}"
        )


    print(
        "pair metadata validation: OK"
    )


    # --------------------------------------------
    # Sampling
    # --------------------------------------------

    groups = defaultdict(
        list
    )


    for row in rows:

        groups[
            (
                row["split"],
                row[
                    "pair_relation"
                ],
            )
        ].append(
            row
        )


    print()
    print(
        "Available strata:"
    )


    for key in sorted(
        groups.keys()
    ):

        print(
            f"  {key[0]:5s} "
            f"{key[1]:13s} "
            f"{len(groups[key])}"
        )


    if args.all:

        selected = list(
            rows
        )

        mode = "ALL"

    else:

        selected, _ = stratified_sample(
            rows,
            args.per_stratum,
            args.seed,
        )

        mode = (
            f"STRATIFIED_"
            f"{args.per_stratum}_"
            f"PER_GROUP"
        )


    print()
    print(
        "sampling mode:",
        mode
    )

    print(
        "selected pairs:",
        len(selected)
    )


    # --------------------------------------------
    # Output root
    # --------------------------------------------

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )


    qc_rows = []

    missing_rows = []


    for i, row in enumerate(
        selected,
        start=1,
    ):

        split = str(
            row["split"]
        )

        relation = str(
            row[
                "pair_relation"
            ]
        )

        pair = str(
            row["pair"]
        )

        cam = int(
            row["camera_id"]
        )

        sa = str(
            row[
                "source_action"
            ]
        )

        sf = int(
            row[
                "source_frame_id"
            ]
        )

        ta = str(
            row[
                "target_action"
            ]
        )

        tf = int(
            row[
                "target_frame_id"
            ]
        )


        filename = (
            f"{safe_name(pair)}_"
            f"cam{cam}_"
            f"S-{safe_name(sa)}-{sf:06d}_"
            f"T-{safe_name(ta)}-{tf:06d}_"
            f"{relation}.jpg"
        )


        output_path = (
            args.out
            / split
            / relation
            / filename
        )


        missing = make_panel(
            row,
            output_path,
        )


        qc_row = {
            "split":
                split,

            "pair":
                pair,

            "camera_id":
                cam,

            "pair_relation":
                relation,

            "source_action":
                sa,

            "source_frame_id":
                sf,

            "target_action":
                ta,

            "target_frame_id":
                tf,

            "source_contact":
                False,

            "target_contact":
                True,

            "qc_panel":
                str(
                    output_path
                ),

            "missing_visual_count":
                len(
                    missing
                ),
        }


        qc_rows.append(
            qc_row
        )


        if missing:

            missing_rows.append(
                {
                    **qc_row,
                    "missing_paths":
                        missing,
                }
            )


        if (
            i % 25 == 0
            or
            i == len(selected)
        ):

            print(
                f"QC {i}/{len(selected)}"
            )


    # --------------------------------------------
    # index.csv
    # --------------------------------------------

    csv_path = (
        args.out
        / "index.csv"
    )


    with open(
        csv_path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "pair",
                "camera_id",
                "pair_relation",
                "source_action",
                "source_frame_id",
                "target_action",
                "target_frame_id",
                "source_contact",
                "target_contact",
                "qc_panel",
                "missing_visual_count",
            ],
        )

        writer.writeheader()

        writer.writerows(
            qc_rows
        )


    # --------------------------------------------
    # Missing log
    # --------------------------------------------

    missing_path = (
        args.out
        / "missing_assets.jsonl"
    )


    with open(
        missing_path,
        "w",
        encoding="utf-8",
    ) as f:

        for row in missing_rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                +
                "\n"
            )


    # --------------------------------------------
    # summary
    # --------------------------------------------

    selected_counts = Counter(
        (
            r[
                "split"
            ],
            r[
                "pair_relation"
            ],
        )
        for r in selected
    )


    summary = {
        "version":
            "hi4d_v4_2_pair_qc",

        "source_manifest":
            str(
                args.manifest
            ),

        "sampling_mode":
            mode,

        "seed":
            args.seed,

        "selected_pairs":
            len(
                selected
            ),

        "missing_asset_pairs":
            len(
                missing_rows
            ),

        "strata":
            {
                f"{k[0]}__{k[1]}":
                    int(v)

                for k, v
                in sorted(
                    selected_counts.items()
                )
            },

        "panel_layout":
            [
                [
                    "Source RGB",
                    "Target RGB",
                    "Target DWPose",
                    "Target Depth",
                ],
                [
                    "Target Normal",
                    "Semantic A/B",
                    "Contact A/B",
                    "Directional Occlusion",
                ],
            ],

        "note":
            (
                "Target visualizations are reused "
                "from final_assets_native_v1. "
                "No geometry was re-rendered. "
                "Visualization PNG/JPG files are "
                "for human QC only, not model input."
            ),
    }


    with open(
        args.out
        / "summary.json",
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
    print("=" * 80)
    print("PAIR QC COMPLETE")
    print("=" * 80)

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print(
        "OUTPUT:",
        args.out
    )

    print(
        "INDEX :",
        csv_path
    )

    print(
        "MISSING:",
        missing_path
    )


if __name__ == "__main__":
    main()
