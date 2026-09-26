#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np


PROJECT = Path("/home/shangguanrz/project/pic-edit")
ROOT = PROJECT / "datasets/Hi4D_pilot_v1"
V2 = ROOT / "native_interaction_v2"


# ============================================================
# helpers
# ============================================================

def read_jsonl(path):
    rows = []

    if not path.exists():
        return rows

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def frame_key(row):
    """
    Support frame manifests.
    """
    try:
        return (
            str(row["pair"]),
            str(row["action"]),
            int(row["camera_id"]),
            int(row["frame_id"]),
        )
    except Exception:
        return None


def pair_source_key(row):
    try:
        return (
            str(row["pair"]),
            str(row["action"]),
            int(row["camera_id"]),
            int(row["source_frame_id"]),
        )
    except Exception:
        return None


def pair_target_key(row):
    try:
        return (
            str(row["pair"]),
            str(row["action"]),
            int(row["camera_id"]),
            int(row["target_frame_id"]),
        )
    except Exception:
        return None


def exists_path(v):
    if not v:
        return False

    try:
        return Path(v).exists()
    except Exception:
        return False


def pct(a, b):
    if b == 0:
        return "N/A"

    return f"{100.0 * a / b:.2f}%"


def show_missing(name, keys, limit=10):
    keys = list(keys)

    print(
        f"{name}: {len(keys)}"
    )

    for k in keys[:limit]:
        print("   ", k)

    if len(keys) > limit:
        print(
            f"    ... +{len(keys)-limit}"
        )


# ============================================================
# locate manifests
# ============================================================

raw_candidates = [
    ROOT / "frames.jsonl",
    V2 / "frames.jsonl",
]

raw_manifest = None

for p in raw_candidates:
    if p.exists():
        raw_manifest = p
        break


used_candidates = [
    V2 / "frames_used_with_dwpose_v2.jsonl",
    V2 / "frames_used_native.jsonl",
]

used_manifest = None

for p in used_candidates:
    if p.exists():
        used_manifest = p
        break


print()
print("=" * 72)
print("1. MANIFESTS")
print("=" * 72)

print(
    "raw manifest :",
    raw_manifest
)

print(
    "used manifest:",
    used_manifest
)


raw_rows = (
    read_jsonl(raw_manifest)
    if raw_manifest
    else []
)

used_rows = (
    read_jsonl(used_manifest)
    if used_manifest
    else []
)


raw_map = {
    frame_key(r): r
    for r in raw_rows
    if frame_key(r) is not None
}

used_map = {
    frame_key(r): r
    for r in used_rows
    if frame_key(r) is not None
}


print(
    "raw frames  :",
    len(raw_map)
)

print(
    "used frames :",
    len(used_map)
)

print(
    "used/raw    :",
    pct(
        len(used_map),
        len(raw_map)
    )
)


# ============================================================
# pair manifests
# ============================================================

print()
print("=" * 72)
print("2. PAIR MANIFESTS")
print("=" * 72)


pair_files = []

for p in sorted(
    V2.glob("*.jsonl")
):
    name = p.name.lower()

    if (
        "pair" in name
        and
        "frame" not in name
    ):
        pair_files.append(p)


all_pair_rows = []

for p in pair_files:
    rows = read_jsonl(p)

    # only accept actual source-target pair files
    valid = [
        r for r in rows
        if (
            pair_source_key(r) is not None
            and
            pair_target_key(r) is not None
        )
    ]

    if not valid:
        continue

    print(
        f"{p.name:45s}",
        len(valid)
    )

    all_pair_rows.extend(valid)


# Deduplicate because split/main/all manifests may overlap.
pair_sig = {}

for r in all_pair_rows:
    sig = (
        pair_source_key(r),
        pair_target_key(r),
        r.get("pair_type"),
    )

    pair_sig[sig] = r


unique_pairs = list(
    pair_sig.values()
)


source_keys = {
    pair_source_key(r)
    for r in unique_pairs
}

target_keys = {
    pair_target_key(r)
    for r in unique_pairs
}

pair_frame_keys = (
    source_keys
    |
    target_keys
)


print()
print(
    "unique pairs        :",
    len(unique_pairs)
)

print(
    "unique source frames:",
    len(source_keys)
)

print(
    "unique target frames:",
    len(target_keys)
)

print(
    "all pair-used frames:",
    len(pair_frame_keys)
)


ptype = Counter(
    str(
        r.get(
            "pair_type",
            "UNKNOWN"
        )
    )
    for r in unique_pairs
)

print(
    "pair type:",
    dict(ptype)
)


# ============================================================
# pair ↔ used manifest coverage
# ============================================================

print()
print("=" * 72)
print("3. PAIR ↔ FRAME MANIFEST COVERAGE")
print("=" * 72)


missing_pair_frames = (
    pair_frame_keys
    -
    set(used_map)
)

extra_used_frames = (
    set(used_map)
    -
    pair_frame_keys
)


print(
    "pair-used frames present in used manifest:",
    len(pair_frame_keys)
    -
    len(missing_pair_frames),
    "/",
    len(pair_frame_keys),
    pct(
        len(pair_frame_keys)
        -
        len(missing_pair_frames),
        len(pair_frame_keys)
    )
)

show_missing(
    "missing pair frames",
    sorted(missing_pair_frames)
)

print(
    "extra frames in used manifest:",
    len(extra_used_frames)
)


# ============================================================
# core native assets
# ============================================================

print()
print("=" * 72)
print("4. NATIVE FRAME ASSET COVERAGE")
print("=" * 72)


FIELDS = {
    "rgb": [
        "rgb",
        "rgb_path",
        "image",
        "image_path",
    ],

    "person_mask_A": [
        "person_mask_A",
        "mask_A",
        "person_a_mask",
    ],

    "person_mask_B": [
        "person_mask_B",
        "mask_B",
        "person_b_mask",
    ],

    "SMPL": [
        "official_smpl",
        "smpl_file",
        "smpl_path",
    ],

    "camera": [
        "camera_file",
        "camera_path",
    ],

    "DWPose_A": [
        "dwpose_A",
        "dwpose_a",
    ],

    "DWPose_B": [
        "dwpose_B",
        "dwpose_b",
    ],
}


def resolve_field(row, candidates):
    for k in candidates:
        if k in row:
            return row[k]

    return None


for display, candidates in FIELDS.items():
    count = 0
    field_present = 0

    missing = []

    for k, r in used_map.items():
        v = resolve_field(
            r,
            candidates
        )

        if v is not None:
            field_present += 1

        if exists_path(v):
            count += 1
        else:
            missing.append(k)

    print(
        f"{display:14s}:",
        f"{count}/{len(used_map)}",
        pct(
            count,
            len(used_map)
        ),
        f"(field present {field_present})"
    )

    if missing:
        print(
            "   missing examples:",
            missing[:5]
        )


# ============================================================
# DWPose file integrity
# ============================================================

print()
print("=" * 72)
print("5. DWPOSE NPZ INTEGRITY")
print("=" * 72)


dw_ok = 0
dw_bad = []


for k, r in used_map.items():

    pa = resolve_field(
        r,
        FIELDS["DWPose_A"]
    )

    pb = resolve_field(
        r,
        FIELDS["DWPose_B"]
    )

    good = True

    for p in [
        pa,
        pb
    ]:

        if (
            p is None
            or
            not Path(p).exists()
        ):
            good = False
            break

        try:
            with np.load(
                p,
                allow_pickle=True
            ) as d:

                needed = {
                    "xy_native",
                    "xy_normalized",
                    "scores",
                }

                if not needed.issubset(
                    set(d.files)
                ):
                    good = False
                    break

                xy = np.asarray(
                    d[
                        "xy_normalized"
                    ]
                )

                scores = np.asarray(
                    d[
                        "scores"
                    ]
                )

                if (
                    xy.ndim != 2
                    or
                    xy.shape[-1] != 2
                    or
                    scores.ndim != 1
                ):
                    good = False
                    break

        except Exception:
            good = False
            break

    if good:
        dw_ok += 1
    else:
        dw_bad.append(k)


print(
    "valid A+B DWPose:",
    f"{dw_ok}/{len(used_map)}",
    pct(
        dw_ok,
        len(used_map)
    )
)

show_missing(
    "invalid/missing DWPose frames",
    dw_bad
)


# ============================================================
# Target-only DWPose coverage
# ============================================================

print()
print("=" * 72)
print("6. TARGET FRAME COVERAGE")
print("=" * 72)


target_in_used = (
    target_keys
    &
    set(used_map)
)


target_dwpose_ok = 0
target_dwpose_missing = []


for k in sorted(
    target_in_used
):

    r = used_map[k]

    pa = resolve_field(
        r,
        FIELDS["DWPose_A"]
    )

    pb = resolve_field(
        r,
        FIELDS["DWPose_B"]
    )

    if (
        exists_path(pa)
        and
        exists_path(pb)
    ):
        target_dwpose_ok += 1
    else:
        target_dwpose_missing.append(k)


print(
    "targets in used manifest:",
    len(target_in_used),
    "/",
    len(target_keys)
)

print(
    "target DWPose:",
    f"{target_dwpose_ok}/{len(target_keys)}",
    pct(
        target_dwpose_ok,
        len(target_keys)
    )
)

show_missing(
    "target DWPose missing",
    target_dwpose_missing
)


# ============================================================
# Search saved geometry caches
# ============================================================

print()
print("=" * 72)
print("7. SAVED GEOMETRY CACHE")
print("=" * 72)


geometry_npzs = []


for p in ROOT.rglob("*.npz"):

    s = str(p).lower()

    # Ignore DWPose and obvious SMPL parameter files.
    if (
        "dwpose" in s
        or
        "smplest" in s
    ):
        continue

    try:
        with np.load(
            p,
            allow_pickle=True
        ) as d:

            fields = set(
                d.files
            )

            geometry_markers = {
                "depth_scene",
                "normal_scene",
                "semantic_A",
                "semantic_B",
            }

            if len(
                fields
                &
                geometry_markers
            ) >= 2:

                geometry_npzs.append(
                    (
                        p,
                        fields
                    )
                )

    except Exception:
        pass


print(
    "geometry npz files:",
    len(geometry_npzs)
)


for p, fields in geometry_npzs[:20]:
    print(
        " ",
        p.relative_to(
            ROOT
        ),
        sorted(
            fields
            &
            {
                "depth_scene",
                "normal_scene",
                "semantic_A",
                "semantic_B",
                "contact_A",
                "contact_B",
                "occlusion_mask",
                "A_occluded_by_B",
                "B_occluded_by_A",
            }
        )
    )


if len(geometry_npzs) > 20:
    print(
        f" ... +{len(geometry_npzs)-20}"
    )


# ============================================================
# Search rendered image caches
# ============================================================

print()
print("=" * 72)
print("8. RENDERED GEOMETRY IMAGES")
print("=" * 72)


keywords = {
    "depth":
        ["depth"],

    "normal":
        ["normal"],

    "semantic":
        ["semantic"],

    "contact":
        ["contact"],

    "occlusion":
        ["occlusion"],
}


for name, kws in keywords.items():

    paths = []

    for ext in [
        "*.png",
        "*.jpg",
        "*.jpeg",
    ]:

        for p in ROOT.rglob(ext):

            s = str(
                p
            ).lower()

            if any(
                kw in s
                for kw in kws
            ):
                paths.append(p)

    print(
        f"{name:10s}:",
        len(
            set(paths)
        )
    )


# ============================================================
# Raw-vs-used distinction
# ============================================================

print()
print("=" * 72)
print("9. FULL RAW DATASET VS INTERACTION SUBSET")
print("=" * 72)


if raw_map:

    raw_not_used = (
        set(raw_map)
        -
        set(used_map)
    )

    print(
        "raw total        :",
        len(raw_map)
    )

    print(
        "interaction used :",
        len(used_map)
    )

    print(
        "raw not in used  :",
        len(raw_not_used)
    )

    print(
        "IMPORTANT:"
    )

    print(
        "  100% of interaction-used frames != "
        "100% of all raw Hi4D frames."
    )


# ============================================================
# Final verdict
# ============================================================

print()
print("=" * 72)
print("10. VERDICT")
print("=" * 72)


issues = []


if missing_pair_frames:
    issues.append(
        "some source/target frames from pair manifests "
        "are absent from used-frame manifest"
    )


if dw_ok != len(used_map):
    issues.append(
        "DWPose is not complete for all interaction-used frames"
    )


if len(geometry_npzs) < len(target_keys):
    issues.append(
        "full target geometry cache is NOT confirmed/generated"
    )


if raw_map and len(used_map) < len(raw_map):
    issues.append(
        "assets cover an interaction subset, not all raw Hi4D frames"
    )


if issues:

    print(
        "STATUS: NOT FULLY MATERIALIZED"
    )

    for x in issues:
        print(
            " -",
            x
        )

else:

    print(
        "STATUS: interaction training assets appear complete"
    )


print()
print(
    "NOTE: contact vertex GT stored inside official Hi4D "
    "SMPL/contact metadata does not need a PNG per frame."
)

print(
    "NOTE: projected contact/occlusion/depth/normal caches are "
    "different from the official underlying 3D annotations."
)
