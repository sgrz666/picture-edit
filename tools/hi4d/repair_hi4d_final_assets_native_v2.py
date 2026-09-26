#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from hi4d_runtime_geometry import Hi4DRuntimeGeometry


PROJECT = Path("/home/shangguanrz/project/pic-edit")

ROOT = (
    PROJECT
    / "datasets"
    / "Hi4D_pilot_v1"
)

V2 = (
    ROOT
    / "native_interaction_v2"
)

OUT_ROOT = (
    ROOT
    / "final_assets_native_v1"
)

FRAMES_JSONL = (
    V2
    / "frames_used_with_dwpose_v2.jsonl"
)

PAIRS_JSONL = (
    V2
    / "pairs_all_with_dwpose.jsonl"
)

ASSET_VERSION = (
    "hi4d_final_native_v2_verified_runtime_renderer"
)


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--only-key",
    type=str,
    default=None,
    help="例如 pair00/hug00/4/84",
)

parser.add_argument(
    "--limit",
    type=int,
    default=None,
)

parser.add_argument(
    "--force",
    action="store_true",
)

args = parser.parse_args()


# ============================================================
# JSONL
# ============================================================

def read_jsonl(path):
    rows = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


frames = read_jsonl(
    FRAMES_JSONL
)

pairs = read_jsonl(
    PAIRS_JSONL
)


def frame_key(
    pair,
    action,
    camera_id,
    frame_id,
):
    return (
        str(pair),
        str(action),
        int(camera_id),
        int(frame_id),
    )


frame_map = {
    frame_key(
        r["pair"],
        r["action"],
        r["camera_id"],
        r["frame_id"],
    ):
    r

    for r in frames
}


# One representative pair for each target.
target_to_pair = {}

for r in pairs:

    k = frame_key(
        r["pair"],
        r["action"],
        r["camera_id"],
        r["target_frame_id"],
    )

    if k not in target_to_pair:
        target_to_pair[k] = r


target_keys = sorted(
    target_to_pair.keys()
)


if args.only_key:

    x = args.only_key.split("/")

    if len(x) != 4:
        raise RuntimeError(
            "--only-key format must be "
            "pair00/hug00/4/84"
        )

    wanted = frame_key(
        x[0],
        x[1],
        x[2],
        x[3],
    )

    if wanted not in target_to_pair:
        raise RuntimeError(
            f"target not found: {wanted}"
        )

    target_keys = [
        wanted
    ]


if args.limit is not None:

    target_keys = (
        target_keys[
            :args.limit
        ]
    )


print()
print("=" * 72)
print("Hi4D FINAL ASSET REPAIR V2")
print("=" * 72)

print(
    "frame rows     :",
    len(frames)
)

print(
    "pair rows      :",
    len(pairs)
)

print(
    "unique targets :",
    len(target_to_pair)
)

print(
    "this run       :",
    len(target_keys)
)

print(
    "output         :",
    OUT_ROOT
)

print(
    "asset version  :",
    ASSET_VERSION
)

print()


# ============================================================
# DWPose
# Exact topology from verified showcase script.
# ============================================================

BODY_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (1, 5),
    (5, 6),
    (6, 7),

    (1, 8),

    (8, 9),
    (9, 10),

    (8, 12),
    (12, 13),
    (13, 14),

    (0, 15),
    (0, 16),
]


def load_pose(path):

    with np.load(
        path,
        allow_pickle=True,
    ) as d:

        xy = np.asarray(
            d["xy_normalized"],
            dtype=np.float32,
        )

        score = np.asarray(
            d["scores"],
            dtype=np.float32,
        )

    return xy, score


def draw_pose_person(
    canvas,
    pose_path,
    color,
    threshold=0.30,
):

    H, W = canvas.shape[:2]

    xy, score = load_pose(
        pose_path
    )

    xy = xy.copy()

    xy[:, 0] *= W
    xy[:, 1] *= H


    for a, b in BODY_EDGES:

        if (
            a >= len(xy)
            or
            b >= len(xy)
        ):
            continue

        if (
            score[a] < threshold
            or
            score[b] < threshold
        ):
            continue

        p1 = tuple(
            np.round(
                xy[a]
            ).astype(int)
        )

        p2 = tuple(
            np.round(
                xy[b]
            ).astype(int)
        )

        cv2.line(
            canvas,
            p1,
            p2,
            color,
            4,
            cv2.LINE_AA,
        )


    for i in range(
        len(xy)
    ):

        if score[i] < threshold:
            continue

        p = tuple(
            np.round(
                xy[i]
            ).astype(int)
        )

        cv2.circle(
            canvas,
            p,
            5,
            color,
            -1,
            cv2.LINE_AA,
        )


def make_pose_image(
    target_row,
    W,
    H,
):

    canvas = np.zeros(
        (
            H,
            W,
            3
        ),
        dtype=np.uint8,
    )

    # A = red
    draw_pose_person(
        canvas,
        target_row["dwpose_A"],
        (
            40,
            40,
            245
        ),
    )

    # B = blue
    draw_pose_person(
        canvas,
        target_row["dwpose_B"],
        (
            245,
            120,
            40
        ),
    )

    return canvas


# ============================================================
# Visualization
# ============================================================

def depth_vis(depth):

    depth = np.asarray(
        depth,
        dtype=np.float32,
    )

    mask = (
        depth > 0
    )

    gray = np.zeros(
        depth.shape,
        dtype=np.uint8,
    )

    if not np.any(mask):

        return np.zeros(
            (
                depth.shape[0],
                depth.shape[1],
                3
            ),
            dtype=np.uint8,
        )


    values = depth[
        mask
    ]

    lo = float(
        np.percentile(
            values,
            1
        )
    )

    hi = float(
        np.percentile(
            values,
            99
        )
    )

    hi = max(
        hi,
        lo + 1e-6
    )


    norm = np.zeros_like(
        depth,
        dtype=np.float32,
    )

    norm[mask] = np.clip(
        (
            depth[mask]
            - lo
        )
        /
        (
            hi
            - lo
        ),
        0,
        1,
    )

    # near = bright
    norm[mask] = (
        1.0
        - norm[mask]
    )

    gray[mask] = (
        norm[mask]
        * 255
    ).astype(
        np.uint8
    )


    vis = cv2.applyColorMap(
        gray,
        cv2.COLORMAP_TURBO,
    )

    # IMPORTANT:
    # keep actual background black
    vis[
        ~mask
    ] = 0

    return vis


def normal_vis(
    normal,
    mask,
):

    n = np.asarray(
        normal,
        dtype=np.float32,
    )


    rgb = (
        (
            np.clip(
                n,
                -1,
                1,
            )
            + 1.0
        )
        * 127.5
    ).astype(
        np.uint8
    )


    rgb[
        ~mask
    ] = 0


    # RGB -> OpenCV BGR
    return rgb[
        :,
        :,
        ::-1
    ]


# ============================================================
# FINAL black-background visualizations
#
# Do NOT use old RGB overlays.
# ============================================================

def semantic_vis(
    A,
    B,
):

    A = (
        np.asarray(A)
        > 0
    )

    B = (
        np.asarray(B)
        > 0
    )


    H, W = A.shape

    out = np.zeros(
        (
            H,
            W,
            3
        ),
        dtype=np.uint8,
    )


    only_A = (
        A
        &
        ~B
    )

    only_B = (
        B
        &
        ~A
    )

    both = (
        A
        &
        B
    )


    # BGR
    # A = red
    out[
        only_A
    ] = (
        40,
        40,
        245
    )

    # B = blue
    out[
        only_B
    ] = (
        245,
        120,
        40
    )

    # overlap = yellow
    out[
        both
    ] = (
        20,
        220,
        240
    )

    return out


def contact_vis(
    contact_A,
    contact_B,
):

    A = (
        np.asarray(
            contact_A
        )
        > 0.05
    )

    B = (
        np.asarray(
            contact_B
        )
        > 0.05
    )


    H, W = A.shape

    out = np.zeros(
        (
            H,
            W,
            3
        ),
        dtype=np.uint8,
    )


    only_A = (
        A
        &
        ~B
    )

    only_B = (
        B
        &
        ~A
    )

    both = (
        A
        &
        B
    )


    out[
        only_A
    ] = (
        40,
        40,
        245
    )

    out[
        only_B
    ] = (
        245,
        110,
        35
    )

    out[
        both
    ] = (
        20,
        230,
        240
    )

    return out


def directional_occ_vis(
    A_by_B,
    B_by_A,
):

    A_by_B = (
        np.asarray(
            A_by_B
        )
        > 0
    )

    B_by_A = (
        np.asarray(
            B_by_A
        )
        > 0
    )


    H, W = (
        A_by_B.shape
    )

    out = np.zeros(
        (
            H,
            W,
            3
        ),
        dtype=np.uint8,
    )


    # A hidden by B = red
    out[
        A_by_B
    ] = (
        40,
        40,
        245
    )

    # B hidden by A = blue
    out[
        B_by_A
    ] = (
        245,
        110,
        35
    )


    both = (
        A_by_B
        &
        B_by_A
    )

    out[
        both
    ] = (
        20,
        230,
        240
    )

    return out


# ============================================================
# Panel
# ============================================================

def make_cell(
    image,
    label,
    cell_w=300,
):

    h, w = (
        image.shape[:2]
    )

    scale = (
        cell_w
        / float(w)
    )

    cell_h = int(
        round(
            h
            * scale
        )
    )


    thumb = cv2.resize(
        image,
        (
            cell_w,
            cell_h
        ),
        interpolation=
            cv2.INTER_AREA,
    )


    title_h = 44

    out = np.zeros(
        (
            title_h
            + cell_h,
            cell_w,
            3
        ),
        dtype=np.uint8,
    )

    out[
        title_h:
    ] = thumb


    cv2.putText(
        out,
        label,
        (
            10,
            29
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (
            255,
            255,
            255
        ),
        2,
        cv2.LINE_AA,
    )

    return out


def make_panel(
    source_rgb,
    target_rgb,
    pose_img,
    depth_img,
    normal_img,
    semantic_img,
    contact_img,
    occ_img,
):

    items = [
        (
            source_rgb,
            "Source RGB"
        ),
        (
            target_rgb,
            "Target RGB"
        ),
        (
            pose_img,
            "Target DWPose"
        ),
        (
            depth_img,
            "Target Depth"
        ),

        (
            normal_img,
            "Target Normal"
        ),
        (
            semantic_img,
            "Person Semantic"
        ),
        (
            contact_img,
            "Contact"
        ),
        (
            occ_img,
            "Directional Occlusion"
        ),
    ]


    cells = [
        make_cell(
            img,
            label
        )
        for (
            img,
            label
        )
        in items
    ]


    max_h = max(
        x.shape[0]
        for x in cells
    )


    padded = []

    for x in cells:

        if (
            x.shape[0]
            < max_h
        ):

            x = cv2.copyMakeBorder(
                x,
                0,
                max_h
                -
                x.shape[0],
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(
                    0,
                    0,
                    0
                ),
            )

        padded.append(
            x
        )


    row1 = np.hstack(
        padded[
            0:4
        ]
    )

    row2 = np.hstack(
        padded[
            4:8
        ]
    )

    return np.vstack(
        [
            row1,
            row2
        ]
    )


# ============================================================
# Check whether already repaired
# ============================================================

def is_v2_geometry(path):

    if not path.exists():
        return False

    try:

        with np.load(
            path,
            allow_pickle=True,
        ) as d:

            if (
                "asset_version"
                not in d.files
            ):
                return False

            version = str(
                d[
                    "asset_version"
                ].item()
            )

            return (
                version
                ==
                ASSET_VERSION
            )

    except Exception:
        return False


# ============================================================
# Small metadata backup
# ============================================================

for name in [
    "final_targets_manifest.jsonl",
    "final_targets_summary.json",
]:

    p = (
        OUT_ROOT
        / name
    )

    backup = (
        OUT_ROOT
        / (
            name
            +
            ".before_v2"
        )
    )

    if (
        p.exists()
        and
        not backup.exists()
    ):

        shutil.copy2(
            p,
            backup
        )


# ============================================================
# Main
# ============================================================

renderer = (
    Hi4DRuntimeGeometry()
)


manifest_rows = []

generated = 0
skipped = 0
failed = 0


for index, key in enumerate(
    target_keys,
    start=1,
):

    pair_name, action, camera_id, target_frame = key

    pair_row = (
        target_to_pair[
            key
        ]
    )

    target_row = (
        frame_map.get(
            key
        )
    )


    source_key = frame_key(
        pair_row["pair"],
        pair_row["action"],
        pair_row["camera_id"],
        pair_row["source_frame_id"],
    )

    source_row = (
        frame_map.get(
            source_key
        )
    )


    if (
        target_row is None
        or
        source_row is None
    ):

        print(
            f"[{index}/{len(target_keys)}] "
            f"ERROR missing frame row {key}",
            flush=True,
        )

        failed += 1
        continue


    stem = str(
        target_row[
            "frame_stem"
        ]
    )


    base = (
        OUT_ROOT
        / pair_name
        / action
        / f"cam{camera_id}"
    )


    rgb_dir = (
        base
        / "rgb"
    )

    dw_A_dir = (
        base
        / "dwpose"
        / "A"
    )

    dw_B_dir = (
        base
        / "dwpose"
        / "B"
    )

    geometry_dir = (
        base
        / "geometry"
    )

    vis_root = (
        base
        / "vis"
    )


    pose_out = (
        vis_root
        / "dwpose"
        / f"{stem}.png"
    )

    depth_out = (
        vis_root
        / "depth"
        / f"{stem}.png"
    )

    normal_out = (
        vis_root
        / "normal"
        / f"{stem}.png"
    )

    semantic_out = (
        vis_root
        / "semantic"
        / f"{stem}.png"
    )

    contact_out = (
        vis_root
        / "contact"
        / f"{stem}.png"
    )

    occ_out = (
        vis_root
        / "directional_occlusion"
        / f"{stem}.png"
    )

    panel_out = (
        vis_root
        / "qc_panel"
        / f"{stem}.jpg"
    )

    geom_out = (
        geometry_dir
        / f"{stem}.npz"
    )


    expected_vis = [
        pose_out,
        depth_out,
        normal_out,
        semantic_out,
        contact_out,
        occ_out,
        panel_out,
    ]


    if (
        not args.force
        and
        is_v2_geometry(
            geom_out
        )
        and
        all(
            p.exists()
            for p in expected_vis
        )
    ):

        print(
            f"[{index}/{len(target_keys)}] "
            f"SKIP {pair_name}/{action}/"
            f"cam{camera_id}/{stem}",
            flush=True,
        )

        skipped += 1

        manifest_rows.append(
            {
                "pair": pair_name,
                "action": action,
                "camera_id": camera_id,
                "frame_id": target_frame,
                "frame_stem": stem,
                "geometry_npz":
                    str(
                        geom_out
                    ),
                "dwpose_vis":
                    str(
                        pose_out
                    ),
                "depth_vis":
                    str(
                        depth_out
                    ),
                "normal_vis":
                    str(
                        normal_out
                    ),
                "semantic_vis":
                    str(
                        semantic_out
                    ),
                "contact_vis":
                    str(
                        contact_out
                    ),
                "directional_occlusion_vis":
                    str(
                        occ_out
                    ),
                "qc_panel":
                    str(
                        panel_out
                    ),
                "asset_version":
                    ASSET_VERSION,
            }
        )

        continue


    print(
        f"[{index}/{len(target_keys)}] "
        f"RUN  {pair_name}/{action}/"
        f"cam{camera_id}/{stem}",
        flush=True,
    )


    try:

        target_rgb = cv2.imread(
            target_row[
                "rgb"
            ],
            cv2.IMREAD_COLOR,
        )

        source_rgb = cv2.imread(
            source_row[
                "rgb"
            ],
            cv2.IMREAD_COLOR,
        )


        if (
            target_rgb is None
            or
            source_rgb is None
        ):

            raise RuntimeError(
                "failed to read RGB"
            )


        H, W = (
            target_rgb.shape[:2]
        )


        if (
            source_rgb.shape[:2]
            !=
            (
                H,
                W
            )
        ):

            source_rgb = cv2.resize(
                source_rgb,
                (
                    W,
                    H
                ),
                interpolation=
                    cv2.INTER_AREA,
            )


        # --------------------------------------
        # Correct DWPose topology
        # --------------------------------------

        pose_img = make_pose_image(
            target_row,
            W,
            H,
        )


        # --------------------------------------
        # VERIFIED runtime renderer
        # No custom projection here.
        # --------------------------------------

        g = renderer.render(
            smpl_file=
                target_row[
                    "official_smpl"
                ],

            camera_file=
                target_row[
                    "camera_file"
                ],

            camera_id=
                int(
                    target_row[
                        "camera_id"
                    ]
                ),

            out_h=H,
            out_w=W,

            native_h=1280,
            native_w=940,
        )


        # --------------------------------------
        # Visualizations
        # --------------------------------------

        depth_img = depth_vis(
            g[
                "depth_scene"
            ]
        )


        normal_img = normal_vis(
            g[
                "normal_scene"
            ],
            g[
                "depth_scene"
            ]
            > 0,
        )


        semantic_img = semantic_vis(
            g[
                "semantic_A"
            ],
            g[
                "semantic_B"
            ],
        )


        contact_img = contact_vis(
            g[
                "contact_A"
            ],
            g[
                "contact_B"
            ],
        )


        occ_img = directional_occ_vis(
            g[
                "A_occluded_by_B"
            ],
            g[
                "B_occluded_by_A"
            ],
        )


        # --------------------------------------
        # Create directories
        # --------------------------------------

        for p in [
            geometry_dir,
            vis_root / "dwpose",
            vis_root / "depth",
            vis_root / "normal",
            vis_root / "semantic",
            vis_root / "contact",
            vis_root / "directional_occlusion",
            vis_root / "qc_panel",
        ]:

            p.mkdir(
                parents=True,
                exist_ok=True,
            )


        # --------------------------------------
        # OVERWRITE failed geometry
        #
        # Save the VERIFIED renderer output
        # exactly, plus version marker.
        # --------------------------------------

        save_dict = {
            k:
                np.asarray(v)

            for k, v
            in g.items()
        }

        save_dict[
            "asset_version"
        ] = np.asarray(
            ASSET_VERSION
        )


        tmp_geom = (
            geometry_dir
            / f".{stem}.repair_tmp.npz"
        )


        np.savez_compressed(
            tmp_geom,
            **save_dict,
        )


        os.replace(
            tmp_geom,
            geom_out,
        )


        # --------------------------------------
        # Overwrite visuals
        # --------------------------------------

        cv2.imwrite(
            str(
                pose_out
            ),
            pose_img,
        )

        cv2.imwrite(
            str(
                depth_out
            ),
            depth_img,
        )

        cv2.imwrite(
            str(
                normal_out
            ),
            normal_img,
        )

        cv2.imwrite(
            str(
                semantic_out
            ),
            semantic_img,
        )

        cv2.imwrite(
            str(
                contact_out
            ),
            contact_img,
        )

        cv2.imwrite(
            str(
                occ_out
            ),
            occ_img,
        )


        panel = make_panel(
            source_rgb,
            target_rgb,
            pose_img,
            depth_img,
            normal_img,
            semantic_img,
            contact_img,
            occ_img,
        )


        cv2.imwrite(
            str(
                panel_out
            ),
            panel,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                96
            ],
        )


        generated += 1


        manifest_rows.append(
            {
                "pair": pair_name,
                "action": action,
                "camera_id": camera_id,
                "frame_id": target_frame,
                "frame_stem": stem,

                "source_rgb":
                    source_row[
                        "rgb"
                    ],

                "target_rgb":
                    target_row[
                        "rgb"
                    ],

                "target_dwpose_A":
                    target_row[
                        "dwpose_A"
                    ],

                "target_dwpose_B":
                    target_row[
                        "dwpose_B"
                    ],

                "geometry_npz":
                    str(
                        geom_out
                    ),

                "dwpose_vis":
                    str(
                        pose_out
                    ),

                "depth_vis":
                    str(
                        depth_out
                    ),

                "normal_vis":
                    str(
                        normal_out
                    ),

                "semantic_vis":
                    str(
                        semantic_out
                    ),

                "contact_vis":
                    str(
                        contact_out
                    ),

                "directional_occlusion_vis":
                    str(
                        occ_out
                    ),

                "qc_panel":
                    str(
                        panel_out
                    ),

                "native_width":
                    W,

                "native_height":
                    H,

                "crop_applied":
                    False,

                "resize_applied":
                    False,

                "asset_version":
                    ASSET_VERSION,
            }
        )


    except Exception as e:

        failed += 1

        print(
            f"ERROR {pair_name}/{action}/"
            f"cam{camera_id}/{stem}: "
            f"{type(e).__name__}: {e}",
            flush=True,
        )


# ============================================================
# Manifest / summary
# Only replace canonical files after run.
# ============================================================

manifest_tmp = (
    OUT_ROOT
    / "final_targets_manifest_v2.tmp.jsonl"
)

manifest_final = (
    OUT_ROOT
    / "final_targets_manifest.jsonl"
)


with open(
    manifest_tmp,
    "w",
    encoding="utf-8",
) as f:

    for r in manifest_rows:

        f.write(
            json.dumps(
                r,
                ensure_ascii=False
            )
            +
            "\n"
        )


# Only make this canonical if complete.
complete = (
    len(manifest_rows)
    ==
    len(target_keys)
    and
    failed == 0
)


if complete:

    os.replace(
        manifest_tmp,
        manifest_final,
    )


summary = {
    "asset_version":
        ASSET_VERSION,

    "total_unique_targets_all":
        len(
            target_to_pair
        ),

    "targets_requested_this_run":
        len(
            target_keys
        ),

    "generated":
        generated,

    "skipped_already_v2":
        skipped,

    "failed":
        failed,

    "run_complete":
        complete,

    "renderer":
        "Hi4DRuntimeGeometry",

    "native_resolution":
        True,

    "crop_applied":
        False,

    "resize_applied":
        False,
}


summary_name = (
    "final_targets_summary.json"
    if complete
    else
    "repair_v2_partial_summary.json"
)


with open(
    OUT_ROOT
    / summary_name,
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
print("REPAIR V2 FINISHED")
print("=" * 72)

print(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    )
)
