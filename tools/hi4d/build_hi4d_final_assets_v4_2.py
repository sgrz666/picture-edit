#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from pathlib import Path
from collections import Counter


PROJECT = Path(
    "/home/shangguanrz/project/pic-edit"
)

ROOT = (
    PROJECT
    / "datasets"
    / "Hi4D_pilot_v1"
)

PAIR_ROOT = (
    ROOT
    / "native_interaction_v4_2"
)

NATIVE_TARGET_ROOT = (
    ROOT
    / "final_assets_native_v1"
)

OUT_ROOT = (
    ROOT
    / "final_assets_v4_2"
)

SOURCE_ROOT = (
    OUT_ROOT
    / "sources"
)

TARGET_ROOT = (
    OUT_ROOT
    / "targets"
)

AUX_ROOT = (
    OUT_ROOT
    / "aux_targets"
)

MANIFEST_ROOT = (
    OUT_ROOT
    / "manifests"
)


# ============================================================
# IO
# ============================================================

def read_jsonl(path):

    if not path.exists():
        raise FileNotFoundError(
            f"missing manifest: {path}"
        )

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


def write_jsonl(
    path,
    rows,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                +
                "\n"
            )


def make_symlink(
    src,
    dst,
):

    src = Path(src)
    dst = Path(dst)

    if not src.exists():
        raise FileNotFoundError(
            f"asset missing: {src}"
        )

    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        dst.exists()
        or
        dst.is_symlink()
    ):
        dst.unlink()

    os.symlink(
        src.resolve(),
        dst,
    )


# ============================================================
# Load V4.2 manifests
# ============================================================

pairs_all = read_jsonl(
    PAIR_ROOT
    / "main_pairs_all.jsonl"
)

pairs_train = read_jsonl(
    PAIR_ROOT
    / "main_pairs_train.jsonl"
)

pairs_val = read_jsonl(
    PAIR_ROOT
    / "main_pairs_val.jsonl"
)

pairs_test = read_jsonl(
    PAIR_ROOT
    / "main_pairs_test.jsonl"
)

sources = read_jsonl(
    PAIR_ROOT
    / "main_sources_unique.jsonl"
)

targets = read_jsonl(
    PAIR_ROOT
    / "main_targets_unique.jsonl"
)

aux_targets = read_jsonl(
    PAIR_ROOT
    / "aux_pending_c2c_dance_targets.jsonl"
)


print()
print("=" * 80)
print("Hi4D FINAL ASSETS V4.2")
print("=" * 80)

print(
    "pair rows      :",
    len(pairs_all)
)

print(
    "train pairs    :",
    len(pairs_train)
)

print(
    "val pairs      :",
    len(pairs_val)
)

print(
    "test pairs     :",
    len(pairs_test)
)

print(
    "unique sources :",
    len(sources)
)

print(
    "MAIN targets   :",
    len(targets)
)

print(
    "AUX targets    :",
    len(aux_targets)
)


# ============================================================
# Hard expected counts
# ============================================================

if len(targets) != 1378:

    raise RuntimeError(
        f"expected 1378 MAIN targets, "
        f"got {len(targets)}"
    )


if len(aux_targets) != 80:

    raise RuntimeError(
        f"expected 80 AUX targets, "
        f"got {len(aux_targets)}"
    )


if len(pairs_all) != 2740:

    raise RuntimeError(
        f"expected 2740 pair rows, "
        f"got {len(pairs_all)}"
    )


# ============================================================
# Build SOURCE assets
# ============================================================

print()
print("===== SOURCE ASSETS =====")


source_registry = {}


for i, row in enumerate(
    sources,
    start=1,
):

    split = str(
        row["split"]
    )

    pair = str(
        row["pair"]
    )

    action = str(
        row["action"]
    )

    cam = int(
        row["camera_id"]
    )

    fid = int(
        row["frame_id"]
    )

    stem = str(
        row.get(
            "frame_stem",
            f"{fid:06d}",
        )
    )


    out_dir = (
        SOURCE_ROOT
        / pair
        / f"cam{cam}"
        / action
        / stem
    )


    rgb_out = (
        out_dir
        / "rgb.jpg"
    )

    mask_a_out = (
        out_dir
        / "mask_A.png"
    )

    mask_b_out = (
        out_dir
        / "mask_B.png"
    )


    make_symlink(
        row["rgb"],
        rgb_out,
    )

    make_symlink(
        row[
            "person_mask_A"
        ],
        mask_a_out,
    )

    make_symlink(
        row[
            "person_mask_B"
        ],
        mask_b_out,
    )


    key = (
        split,
        pair,
        cam,
        action,
        fid,
    )


    source_registry[
        key
    ] = {
        "split":
            split,

        "pair":
            pair,

        "camera_id":
            cam,

        "action":
            action,

        "frame_id":
            fid,

        "frame_stem":
            stem,

        "rgb":
            str(
                rgb_out
            ),

        "person_mask_A":
            str(
                mask_a_out
            ),

        "person_mask_B":
            str(
                mask_b_out
            ),

        "is_contact":
            False,

        "asset_storage":
            "symlink",
    }


    if i % 50 == 0:

        print(
            f"source "
            f"{i}/{len(sources)}"
        )


print(
    "source assets complete:",
    len(source_registry)
)


# ============================================================
# Helper for target materialization
# ============================================================

def build_target_asset(
    row,
    output_root,
    bucket,
):

    split = str(
        row["split"]
    )

    pair = str(
        row["pair"]
    )

    action = str(
        row["action"]
    )

    cam = int(
        row["camera_id"]
    )

    fid = int(
        row["frame_id"]
    )

    stem = str(
        row[
            "frame_stem"
        ]
    )


    out_dir = (
        output_root
        / pair
        / action
        / f"cam{cam}"
        / stem
    )


    rgb_out = (
        out_dir
        / "rgb.jpg"
    )

    dw_a_out = (
        out_dir
        / "dwpose_A.npz"
    )

    dw_b_out = (
        out_dir
        / "dwpose_B.npz"
    )

    geom_out = (
        out_dir
        / "geometry.npz"
    )

    mask_a_out = (
        out_dir
        / "gt_mask_A.png"
    )

    mask_b_out = (
        out_dir
        / "gt_mask_B.png"
    )


    make_symlink(
        row["rgb"],
        rgb_out,
    )

    make_symlink(
        row[
            "dwpose_A"
        ],
        dw_a_out,
    )

    make_symlink(
        row[
            "dwpose_B"
        ],
        dw_b_out,
    )

    make_symlink(
        row[
            "geometry_npz"
        ],
        geom_out,
    )

    make_symlink(
        row[
            "person_mask_A"
        ],
        mask_a_out,
    )

    make_symlink(
        row[
            "person_mask_B"
        ],
        mask_b_out,
    )


    return {
        "dataset_bucket":
            bucket,

        "split":
            split,

        "pair":
            pair,

        "action":
            action,

        "camera_id":
            cam,

        "frame_id":
            fid,

        "frame_stem":
            stem,

        # --------------------------------------
        # Training / control assets
        # --------------------------------------

        "rgb":
            str(
                rgb_out
            ),

        "dwpose_A":
            str(
                dw_a_out
            ),

        "dwpose_B":
            str(
                dw_b_out
            ),

        "geometry_npz":
            str(
                geom_out
            ),

        # --------------------------------------
        # GT masks:
        # supervision / evaluation / QC only.
        # NOT target geometry control.
        # --------------------------------------

        "gt_person_mask_A":
            str(
                mask_a_out
            ),

        "gt_person_mask_B":
            str(
                mask_b_out
            ),

        "gt_masks_are_control":
            False,

        "is_contact":
            True,

        "asset_storage":
            "symlink",

        "native_width":
            940,

        "native_height":
            1280,

        "crop_applied":
            False,

        "resize_applied":
            False,
    }


# ============================================================
# MAIN TARGET assets
# ============================================================

print()
print("===== MAIN TARGET ASSETS =====")


target_registry = {}


for i, row in enumerate(
    targets,
    start=1,
):

    asset = build_target_asset(
        row,
        TARGET_ROOT,
        "MAIN_N2C",
    )


    key = (
        asset["split"],
        asset["pair"],
        asset["camera_id"],
        asset["action"],
        asset["frame_id"],
    )


    target_registry[
        key
    ] = asset


    if i % 200 == 0:

        print(
            f"MAIN target "
            f"{i}/{len(targets)}"
        )


print(
    "MAIN target assets complete:",
    len(target_registry)
)


# ============================================================
# AUX TARGET assets
# ============================================================

print()
print("===== AUX TARGET ASSETS =====")


aux_registry = {}


for i, row in enumerate(
    aux_targets,
    start=1,
):

    asset = build_target_asset(
        row,
        AUX_ROOT,
        "AUX_PENDING_C2C_DANCE",
    )


    asset[
        "training_status"
    ] = (
        "pending_teacher_decision"
    )

    asset[
        "suggested_future_mode"
    ] = (
        "C2C_CROSSPOSE"
    )

    asset[
        "source_assigned"
    ] = False


    key = (
        asset["split"],
        asset["pair"],
        asset["camera_id"],
        asset["action"],
        asset["frame_id"],
    )


    aux_registry[
        key
    ] = asset


print(
    "AUX target assets complete:",
    len(aux_registry)
)


# ============================================================
# Rewrite pair manifests to V4.2 final assets
# ============================================================

def convert_pair(
    row,
):

    source_key = (
        str(
            row["split"]
        ),
        str(
            row["pair"]
        ),
        int(
            row["camera_id"]
        ),
        str(
            row["source_action"]
        ),
        int(
            row["source_frame_id"]
        ),
    )


    target_key = (
        str(
            row["split"]
        ),
        str(
            row["pair"]
        ),
        int(
            row["camera_id"]
        ),
        str(
            row["target_action"]
        ),
        int(
            row["target_frame_id"]
        ),
    )


    if source_key not in source_registry:

        raise RuntimeError(
            f"missing Source asset: "
            f"{source_key}"
        )


    if target_key not in target_registry:

        raise RuntimeError(
            f"missing Target asset: "
            f"{target_key}"
        )


    source = source_registry[
        source_key
    ]

    target = target_registry[
        target_key
    ]


    out = dict(
        row
    )


    # ------------------------------------------
    # Source
    # ------------------------------------------

    out[
        "source_rgb"
    ] = source[
        "rgb"
    ]

    out[
        "source_person_mask_A"
    ] = source[
        "person_mask_A"
    ]

    out[
        "source_person_mask_B"
    ] = source[
        "person_mask_B"
    ]


    # ------------------------------------------
    # Target
    # ------------------------------------------

    out[
        "target_rgb"
    ] = target[
        "rgb"
    ]

    out[
        "target_dwpose_A"
    ] = target[
        "dwpose_A"
    ]

    out[
        "target_dwpose_B"
    ] = target[
        "dwpose_B"
    ]

    out[
        "target_geometry_npz"
    ] = target[
        "geometry_npz"
    ]


    # GT target masks:
    # explicit evaluation/supervision only.
    out[
        "target_person_mask_A"
    ] = target[
        "gt_person_mask_A"
    ]

    out[
        "target_person_mask_B"
    ] = target[
        "gt_person_mask_B"
    ]

    out[
        "target_masks_are_control"
    ] = False


    out[
        "asset_version"
    ] = (
        "hi4d_final_assets_v4_2"
    )


    return out


# ============================================================
# Convert split manifests
# ============================================================

pairs_all_final = [
    convert_pair(r)
    for r in pairs_all
]

pairs_train_final = [
    convert_pair(r)
    for r in pairs_train
]

pairs_val_final = [
    convert_pair(r)
    for r in pairs_val
]

pairs_test_final = [
    convert_pair(r)
    for r in pairs_test
]


# ============================================================
# Hard pair validation
# ============================================================

for row in pairs_all_final:

    if bool(
        row[
            "source_contact"
        ]
    ):

        raise RuntimeError(
            "contact Source in final manifest"
        )


    if not bool(
        row[
            "target_contact"
        ]
    ):

        raise RuntimeError(
            "non-contact Target in final manifest"
        )


    if not Path(
        row[
            "source_rgb"
        ]
    ).exists():

        raise RuntimeError(
            "broken Source RGB link"
        )


    if not Path(
        row[
            "target_rgb"
        ]
    ).exists():

        raise RuntimeError(
            "broken Target RGB link"
        )


    if not Path(
        row[
            "target_dwpose_A"
        ]
    ).exists():

        raise RuntimeError(
            "broken DWPose A link"
        )


    if not Path(
        row[
            "target_dwpose_B"
        ]
    ).exists():

        raise RuntimeError(
            "broken DWPose B link"
        )


    if not Path(
        row[
            "target_geometry_npz"
        ]
    ).exists():

        raise RuntimeError(
            "broken Target geometry link"
        )


# ============================================================
# Save final manifests
# ============================================================

MANIFEST_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


write_jsonl(
    MANIFEST_ROOT
    / "main_pairs_all.jsonl",
    pairs_all_final,
)

write_jsonl(
    MANIFEST_ROOT
    / "main_pairs_train.jsonl",
    pairs_train_final,
)

write_jsonl(
    MANIFEST_ROOT
    / "main_pairs_val.jsonl",
    pairs_val_final,
)

write_jsonl(
    MANIFEST_ROOT
    / "main_pairs_test.jsonl",
    pairs_test_final,
)

write_jsonl(
    MANIFEST_ROOT
    / "main_sources_unique.jsonl",

    sorted(
        source_registry.values(),
        key=lambda r: (
            r["split"],
            r["pair"],
            r["camera_id"],
            r["action"],
            r["frame_id"],
        ),
    ),
)

write_jsonl(
    MANIFEST_ROOT
    / "main_targets_unique.jsonl",

    sorted(
        target_registry.values(),
        key=lambda r: (
            r["split"],
            r["pair"],
            r["action"],
            r["camera_id"],
            r["frame_id"],
        ),
    ),
)

write_jsonl(
    MANIFEST_ROOT
    / "aux_pending_c2c_dance_targets.jsonl",

    sorted(
        aux_registry.values(),
        key=lambda r: (
            r["split"],
            r["pair"],
            r["action"],
            r["camera_id"],
            r["frame_id"],
        ),
    ),
)


# ============================================================
# Statistics
# ============================================================

relation_counts = Counter(
    r["pair_relation"]
    for r in pairs_all_final
)

split_pair_counts = Counter(
    r["split"]
    for r in pairs_all_final
)

split_source_counts = Counter(
    r["split"]
    for r in source_registry.values()
)

split_target_counts = Counter(
    r["split"]
    for r in target_registry.values()
)


summary = {
    "version":
        "hi4d_final_assets_v4_2",

    "pairing_version":
        "native_interaction_v4_2",

    "main_task":
        (
            "non-contact Source RGB + "
            "contact Target geometry -> "
            "contact Target RGB"
        ),

    "assets": {
        "storage":
            "symlink",

        "native_resolution":
            "940x1280",

        "crop_applied":
            False,

        "resize_applied":
            False,
    },

    "counts": {
        "main_pair_rows":
            len(
                pairs_all_final
            ),

        "main_pairs_train":
            len(
                pairs_train_final
            ),

        "main_pairs_val":
            len(
                pairs_val_final
            ),

        "main_pairs_test":
            len(
                pairs_test_final
            ),

        "main_unique_sources":
            len(
                source_registry
            ),

        "main_unique_targets":
            len(
                target_registry
            ),

        "aux_targets":
            len(
                aux_registry
            ),

        "cross_action_pairs":
            int(
                relation_counts[
                    "CROSS_ACTION"
                ]
            ),

        "same_action_pairs":
            int(
                relation_counts[
                    "SAME_ACTION"
                ]
            ),
    },

    "source_split": {
        k:
            int(v)
        for k, v
        in split_source_counts.items()
    },

    "target_split": {
        k:
            int(v)
        for k, v
        in split_target_counts.items()
    },

    "pair_split": {
        k:
            int(v)
        for k, v
        in split_pair_counts.items()
    },

    "target_control_policy": {
        "stage1_controls": [
            "DWPose A/B",
            "Depth scene",
            "Normal scene",
            "Semantic A/B",
        ],

        "later_ablation": [
            "Contact",
            "Directional Occlusion",
        ],

        "target_real_person_masks_are_control":
            False,
    },

    "aux_policy": {
        "enabled_for_main_training":
            False,

        "scope":
            "pair14/dance14/cam16",

        "future_candidate":
            "C2C_CROSSPOSE",
    },
}


with open(
    MANIFEST_ROOT
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


with open(
    OUT_ROOT
    / "README.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        {
            "main_training_manifest":
                str(
                    MANIFEST_ROOT
                    / "main_pairs_train.jsonl"
                ),

            "validation_manifest":
                str(
                    MANIFEST_ROOT
                    / "main_pairs_val.jsonl"
                ),

            "test_manifest":
                str(
                    MANIFEST_ROOT
                    / "main_pairs_test.jsonl"
                ),

            "note":
                (
                    "All physical assets are "
                    "symlinks to existing native "
                    "Hi4D assets. Do not treat "
                    "target gt_person_mask_A/B "
                    "as geometry control."
                ),
        },
        f,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# Finish
# ============================================================

print()
print("=" * 80)
print("FINAL ASSETS V4.2 COMPLETE")
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
    OUT_ROOT
)
