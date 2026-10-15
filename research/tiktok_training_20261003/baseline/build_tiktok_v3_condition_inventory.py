#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TikTok V3 - Step 6
Existing condition alignment / reuse / completion inventory.

Input:
    TikTok_v3/manifests/candidate_pool.jsonl

Check existing:
    RGB
    person mask
    DensePose
    SMPLest-X params + frame_map mapping
    normal
    depth_vis
    SMPL-X mask
    raw depth (if any)
    DWPose (if any)

No inference.
No copying.
No deletion.
"""

from pathlib import Path
from collections import defaultdict
import json

PROJECT = Path("/home/shangguanrz/project/pic-edit")

ROOT = PROJECT / "datasets/TikTokDataset"

FRAME_ROOT = (
    ROOT
    / "TikTok_dataset"
    / "TikTok_dataset"
)

ASSET_ROOT = (
    ROOT
    / "TikTok_3d_assets_native"
)

V3_ROOT = ROOT / "TikTok_v3"

CANDIDATES = (
    V3_ROOT
    / "manifests"
    / "candidate_pool.jsonl"
)

OUT_MANIFEST = (
    V3_ROOT
    / "manifests"
    / "condition_inventory.jsonl"
)

OUT_SUMMARY = (
    V3_ROOT
    / "manifests"
    / "condition_inventory_summary.json"
)

EXTS = [
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".npy",
    ".npz",
]


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


def find_by_stem(folder, stem):
    if not folder.exists():
        return None

    for ext in EXTS:
        p = folder / f"{stem}{ext}"

        if p.exists():
            return p

    return None


def parse_frame_map_obj(obj):
    """
    Convert several common frame_map formats into:
        stem -> index
    """

    mapping = {}

    if isinstance(obj, dict):

        # form:
        # {"0001": 0, "0002": 1}
        simple = True

        for k, v in obj.items():
            if not isinstance(v, int):
                simple = False
                break

        if simple:
            for k, v in obj.items():
                mapping[
                    Path(str(k)).stem
                ] = int(v)

            return mapping

        # common list fields
        for key in [
            "frames",
            "frame_names",
            "images",
            "image_paths",
            "paths",
            "stems",
        ]:
            value = obj.get(key)

            if not isinstance(value, list):
                continue

            for i, item in enumerate(value):

                if isinstance(item, str):
                    mapping[
                        Path(item).stem
                    ] = i

                elif isinstance(item, dict):
                    name = (
                        item.get("frame")
                        or item.get("name")
                        or item.get("stem")
                        or item.get("path")
                        or item.get("image")
                        or item.get("image_path")
                    )

                    idx = item.get(
                        "index",
                        item.get(
                            "idx",
                            i,
                        ),
                    )

                    if name is not None:
                        mapping[
                            Path(str(name)).stem
                        ] = int(idx)

    elif isinstance(obj, list):

        for i, item in enumerate(obj):

            if isinstance(item, str):
                mapping[
                    Path(item).stem
                ] = i

            elif isinstance(item, dict):
                name = (
                    item.get("frame")
                    or item.get("name")
                    or item.get("stem")
                    or item.get("path")
                    or item.get("image")
                )

                idx = item.get(
                    "index",
                    item.get(
                        "idx",
                        i,
                    ),
                )

                if name is not None:
                    mapping[
                        Path(str(name)).stem
                    ] = int(idx)

    return mapping


def load_sequence_smplestx(seq):
    root = (
        ASSET_ROOT
        / seq
        / "smplestx"
    )

    params = (
        root
        / "params.pt"
    )

    frame_map = (
        root
        / "frame_map.json"
    )

    if not params.exists():
        return {
            "params": None,
            "frame_map": None,
            "mapping": {},
        }

    mapping = {}

    if frame_map.exists():
        try:
            with open(
                frame_map,
                "r",
                encoding="utf-8",
            ) as f:
                mapping = parse_frame_map_obj(
                    json.load(f)
                )
        except Exception:
            mapping = {}

    return {
        "params": params,
        "frame_map":
            frame_map
            if frame_map.exists()
            else None,

        "mapping": mapping,
    }


def find_existing_dwpose(seq, stem):
    """
    Check known / plausible existing locations only.
    This function does NOT generate anything.
    """

    roots = [
        V3_ROOT
        / "assets"
        / "dwpose_npz"
        / seq,

        ASSET_ROOT
        / seq
        / "dwpose",

        ASSET_ROOT
        / seq
        / "dwpose_npz",

        FRAME_ROOT
        / seq
        / "dwpose",
    ]

    for root in roots:
        p = find_by_stem(
            root,
            stem,
        )

        if p is not None:
            return p

    return None


def find_raw_depth(seq, stem):
    """
    depth_vis is NOT raw metric depth.
    Check whether any raw depth cache already exists.
    """

    roots = [
        ASSET_ROOT
        / seq
        / "depth",

        ASSET_ROOT
        / seq
        / "depth_raw",

        ASSET_ROOT
        / seq
        / "raw_depth",
    ]

    for root in roots:
        p = find_by_stem(
            root,
            stem,
        )

        if p is not None:
            return p

    return None


def main():
    if not CANDIDATES.exists():
        raise FileNotFoundError(
            CANDIDATES
        )

    rows = read_jsonl(
        CANDIDATES
    )

    print("=" * 72)
    print("TikTok V3 CONDITION INVENTORY")
    print("=" * 72)
    print("candidate frames:", len(rows))
    print(flush=True)

    smpl_cache = {}

    counts = defaultdict(int)

    per_sequence = defaultdict(
        lambda: defaultdict(int)
    )

    output = []

    for i, row in enumerate(
        rows,
        1,
    ):
        seq = row[
            "sequence_id"
        ]

        stem = row[
            "stem"
        ]

        rgb = Path(
            row["rgb"]
        )

        # ----------------------------------------------------
        # Existing 2D assets
        # ----------------------------------------------------

        person_mask = (
            Path(row["person_mask"])
            if row.get("person_mask")
            else
            find_by_stem(
                FRAME_ROOT
                / seq
                / "masks",
                stem,
            )
        )

        densepose = find_by_stem(
            FRAME_ROOT
            / seq
            / "densepose",
            stem,
        )

        # ----------------------------------------------------
        # Existing 3D-derived assets
        # ----------------------------------------------------

        normal = find_by_stem(
            ASSET_ROOT
            / seq
            / "normal",
            stem,
        )

        depth_vis = find_by_stem(
            ASSET_ROOT
            / seq
            / "depth_vis",
            stem,
        )

        smplx_mask = find_by_stem(
            ASSET_ROOT
            / seq
            / "smplx_mask",
            stem,
        )

        raw_depth = find_raw_depth(
            seq,
            stem,
        )

        # ----------------------------------------------------
        # Existing SMPLest-X parameter bundle
        # ----------------------------------------------------

        if seq not in smpl_cache:
            smpl_cache[
                seq
            ] = load_sequence_smplestx(
                seq
            )

        smpl = smpl_cache[
            seq
        ]

        smpl_index = (
            smpl[
                "mapping"
            ].get(stem)
        )

        has_smplestx = bool(
            smpl[
                "params"
            ] is not None
            and
            smpl_index is not None
        )

        # ----------------------------------------------------
        # Existing DWPose
        # ----------------------------------------------------

        dwpose = find_existing_dwpose(
            seq,
            stem,
        )

        flags = {
            "rgb":
                rgb.exists(),

            "person_mask":
                person_mask is not None
                and
                person_mask.exists(),

            "densepose":
                densepose is not None,

            "smplestx":
                has_smplestx,

            "normal":
                normal is not None,

            "depth_vis":
                depth_vis is not None,

            "smplx_mask":
                smplx_mask is not None,

            "raw_depth":
                raw_depth is not None,

            "dwpose":
                dwpose is not None,
        }

        for key, value in flags.items():
            if value:
                counts[key] += 1
                per_sequence[
                    seq
                ][key] += 1

        core_existing_complete = all(
            flags[k]
            for k in [
                "rgb",
                "person_mask",
                "densepose",
                "smplestx",
                "normal",
                "depth_vis",
                "smplx_mask",
            ]
        )

        if core_existing_complete:
            counts[
                "core_existing_complete"
            ] += 1

        record = {
            "sequence_id":
                seq,

            "stem":
                stem,

            "frame_num":
                row.get(
                    "frame_num"
                ),

            "sampling_motion_class":
                row.get(
                    "sampling_motion_class"
                ),

            "paths": {
                "rgb":
                    str(rgb),

                "person_mask":
                    str(person_mask)
                    if person_mask
                    else None,

                "densepose":
                    str(densepose)
                    if densepose
                    else None,

                "smplestx_params":
                    str(
                        smpl[
                            "params"
                        ]
                    )
                    if smpl[
                        "params"
                    ]
                    else None,

                "smplestx_frame_map":
                    str(
                        smpl[
                            "frame_map"
                        ]
                    )
                    if smpl[
                        "frame_map"
                    ]
                    else None,

                "normal":
                    str(normal)
                    if normal
                    else None,

                "depth_vis":
                    str(depth_vis)
                    if depth_vis
                    else None,

                "smplx_mask":
                    str(smplx_mask)
                    if smplx_mask
                    else None,

                "raw_depth":
                    str(raw_depth)
                    if raw_depth
                    else None,

                "dwpose":
                    str(dwpose)
                    if dwpose
                    else None,
            },

            "smplestx_index":
                smpl_index,

            "available":
                flags,

            "core_existing_complete":
                core_existing_complete,

            "needs": {
                "dwpose":
                    not flags[
                        "dwpose"
                    ],

                "raw_depth":
                    not flags[
                        "raw_depth"
                    ],
            },
        }

        output.append(
            record
        )

        if (
            i % 1000 == 0
            or
            i == len(rows)
        ):
            print(
                f"[{i}/{len(rows)}] "
                f"core_complete="
                f"{counts['core_existing_complete']} "
                f"dwpose="
                f"{counts['dwpose']} "
                f"raw_depth="
                f"{counts['raw_depth']}",
                flush=True,
            )

    OUT_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUT_MANIFEST,
        "w",
        encoding="utf-8",
    ) as f:
        for record in output:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                +
                "\n"
            )

    total = len(rows)

    conditions = [
        "rgb",
        "person_mask",
        "densepose",
        "smplestx",
        "normal",
        "depth_vis",
        "smplx_mask",
        "raw_depth",
        "dwpose",
    ]

    summary = {
        "version":
            "tiktok_v3_condition_inventory_v1",

        "candidate_frames":
            total,

        "coverage": {
            key: {
                "count":
                    int(
                        counts[key]
                    ),

                "ratio":
                    float(
                        counts[key]
                        /
                        max(
                            total,
                            1,
                        )
                    ),
            }
            for key in conditions
        },

        "core_existing_complete": {
            "count":
                int(
                    counts[
                        "core_existing_complete"
                    ]
                ),

            "ratio":
                float(
                    counts[
                        "core_existing_complete"
                    ]
                    /
                    max(
                        total,
                        1,
                    )
                ),
        },

        "definitions": {
            "core_existing_complete":
                (
                    "RGB + person mask + DensePose + "
                    "SMPLest-X mapping + normal + "
                    "depth_vis + SMPL-X mask"
                ),

            "depth_vis":
                (
                    "visualization only; not treated as "
                    "raw float training depth"
                ),
        },

        "policy": {
            "rerun_smplestx":
                False,

            "copy_assets":
                False,

            "generate_new_conditions":
                False,

            "reuse_existing_assets_first":
                True,
        },

        "next_action":
            (
                "Generate only missing required conditions; "
                "in particular inspect DWPose coverage first. "
                "Raw float depth can later be rendered from "
                "cached SMPL-X without rerunning SMPLest-X."
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
    print("CONDITION INVENTORY COMPLETE")
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
