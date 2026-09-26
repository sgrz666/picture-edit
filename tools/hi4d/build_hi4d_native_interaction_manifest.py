#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from collections import defaultdict, Counter

from PIL import Image


ROOT = Path(
    "/home/shangguanrz/project/pic-edit/datasets/Hi4D_pilot_v1"
)

SRC = ROOT / "frames.jsonl"

OUT = ROOT / "native_interaction_v1"
OUT.mkdir(parents=True, exist_ok=True)

# 每个 contact target：
# train 尽量 3 个 N->C + 1 个 C->C
# val/test 固定 1 个 N->C + 1 个 C->C
N2C_PER_TARGET = {
    "train": 3,
    "val": 1,
    "test": 1,
}

C2C_PER_TARGET = {
    "train": 1,
    "val": 1,
    "test": 1,
}


# ============================================================
# IO
# ============================================================

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                ) + "\n"
            )


def frame_id(row):
    if "frame_id" in row:
        return int(row["frame_id"])

    if "frame_stem" in row:
        return int(row["frame_stem"])

    raise KeyError("frame_id/frame_stem missing")


def seq_key(row):
    return (
        row["split"],
        row["pair"],
        row["action"],
        int(row["camera_id"]),
    )


# ============================================================
# NATIVE METADATA
# ============================================================

def native_record(row):
    """
    只引用原始数据。
    不 crop。
    不 resize。
    """

    rgb_path = Path(row["rgb"])

    with Image.open(rgb_path) as im:
        W, H = im.size

    out = dict(row)

    out.update({
        "frame_id": frame_id(row),

        "native_rgb":
            row["rgb"],

        "native_person_mask_A":
            row["person_mask_A"],

        "native_person_mask_B":
            row["person_mask_B"],

        "native_width":
            int(W),

        "native_height":
            int(H),

        "native_aspect_ratio":
            float(W / H),

        "geometry_source":
            row["official_smpl"],

        "camera_source":
            row["camera_file"],

        "resize_applied":
            False,

        "crop_applied":
            False,

        "contact_source":
            "Hi4D official meta.npz/contact_ids",

        # 后续真正进入模型时才动态产生
        "target_geometry_policy":
            "render_from_SMPL_at_dataloader_resolution",
    })

    return out


# ============================================================
# SOURCE SELECTION
# ============================================================

def unique_append(out, row):
    if row is None:
        return

    fid = frame_id(row)

    if fid not in {
        frame_id(x)
        for x in out
    }:
        out.append(row)


def choose_noncontact_sources(
    target,
    noncontacts,
    k
):
    """
    优先：
    1. 最近的 target 之前 non-contact
       -> interaction formation
    2. 更远的 target 之前 non-contact
       -> larger motion
    3. 最近的 target 之后 non-contact
    4. 时间距离最大的 non-contact

    后面 DWPose 完成以后，
    会再用真实 pose distance 做第二次排序。
    """

    if k <= 0:
        return []

    tid = frame_id(target)

    before = [
        x for x in noncontacts
        if frame_id(x) < tid
    ]

    after = [
        x for x in noncontacts
        if frame_id(x) > tid
    ]

    before_near = sorted(
        before,
        key=lambda x: abs(
            frame_id(x) - tid
        )
    )

    before_far = sorted(
        before,
        key=lambda x: abs(
            frame_id(x) - tid
        ),
        reverse=True
    )

    after_near = sorted(
        after,
        key=lambda x: abs(
            frame_id(x) - tid
        )
    )

    all_far = sorted(
        noncontacts,
        key=lambda x: abs(
            frame_id(x) - tid
        ),
        reverse=True
    )

    selected = []

    # 形成交互：优先最近的 pre-contact
    if before_near:
        unique_append(
            selected,
            before_near[0]
        )

    # 较大姿态变化
    if before_far:
        unique_append(
            selected,
            before_far[0]
        )

    # 必要时加 post-contact non-contact
    if after_near:
        unique_append(
            selected,
            after_near[0]
        )

    # 再按最大时间差补齐
    for x in all_far:
        unique_append(
            selected,
            x
        )

        if len(selected) >= k:
            break

    return selected[:k]


def choose_contact_source(
    target,
    contacts
):
    """
    C->C:
    选择同 sequence 中时间距离最大的另一个 contact frame。
    """

    tid = frame_id(target)

    candidates = [
        x for x in contacts
        if frame_id(x) != tid
    ]

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda x: abs(
            frame_id(x) - tid
        )
    )


# ============================================================
# PAIR
# ============================================================

def make_pair(
    source,
    target,
    pair_type
):

    sid = frame_id(source)
    tid = frame_id(target)

    return {
        "split":
            target["split"],

        "pair":
            target["pair"],

        "action":
            target["action"],

        "camera_id":
            int(target["camera_id"]),

        "pair_type":
            pair_type,

        "source_contact":
            bool(source["is_contact"]),

        "target_contact":
            bool(target["is_contact"]),

        "source_frame_id":
            sid,

        "target_frame_id":
            tid,

        "frame_gap":
            abs(tid - sid),

        "source_temporal_relation":
            (
                "before_target"
                if sid < tid
                else "after_target"
            ),

        # -----------------------------
        # SOURCE
        # -----------------------------

        "source_rgb":
            source["rgb"],

        "source_person_mask_A":
            source["person_mask_A"],

        "source_person_mask_B":
            source["person_mask_B"],

        # -----------------------------
        # TARGET
        # -----------------------------

        "target_rgb":
            target["rgb"],

        "target_person_mask_A":
            target["person_mask_A"],

        "target_person_mask_B":
            target["person_mask_B"],

        # -----------------------------
        # GEOMETRY SOURCE
        # 不保存固定512图
        # -----------------------------

        "target_smpl":
            target["official_smpl"],

        "target_camera":
            target["camera_file"],

        # Hi4D官方 contact correspondence
        # 也在这个 npz 的 contact key 中
        "target_contact_npz":
            target["official_smpl"],

        "target_contact_key":
            "contact",

        # -----------------------------
        # ORIGINAL SIZE
        # -----------------------------

        "native_width":
            target["native_width"],

        "native_height":
            target["native_height"],

        "crop_applied":
            False,

        "resize_applied":
            False,

        "geometry_policy":
            "render_on_the_fly_at_model_target_resolution",
    }


# ============================================================
# MAIN
# ============================================================

rows_raw = load_jsonl(SRC)

print("source rows:", len(rows_raw))

rows = [
    native_record(x)
    for x in rows_raw
]

groups = defaultdict(list)

for row in rows:
    groups[seq_key(row)].append(row)

print("sequences:", len(groups))


all_contact_targets = []
all_noncontact_sources = []

all_pairs = []
sequence_summary = []

used_frame_map = {}


for key in sorted(groups):

    split, pair_name, action, camera_id = key

    seq = sorted(
        groups[key],
        key=frame_id
    )

    contacts = [
        x for x in seq
        if bool(x["is_contact"])
    ]

    noncontacts = [
        x for x in seq
        if not bool(x["is_contact"])
    ]

    all_contact_targets.extend(
        contacts
    )

    all_noncontact_sources.extend(
        noncontacts
    )

    seq_pairs = []

    n2c_count = 0
    c2c_count = 0

    for target in contacts:

        # =========================================
        # N -> C
        # =========================================

        n_sources = choose_noncontact_sources(
            target,
            noncontacts,
            N2C_PER_TARGET[split]
        )

        for source in n_sources:

            p = make_pair(
                source,
                target,
                "N2C"
            )

            seq_pairs.append(p)
            all_pairs.append(p)

            n2c_count += 1

            used_frame_map[
                (
                    source["pair"],
                    source["action"],
                    frame_id(source),
                    int(source["camera_id"])
                )
            ] = source

            used_frame_map[
                (
                    target["pair"],
                    target["action"],
                    frame_id(target),
                    int(target["camera_id"])
                )
            ] = target

        # =========================================
        # C -> C
        # =========================================

        if C2C_PER_TARGET[split] > 0:

            source_c = choose_contact_source(
                target,
                contacts
            )

            if source_c is not None:

                p = make_pair(
                    source_c,
                    target,
                    "C2C"
                )

                seq_pairs.append(p)
                all_pairs.append(p)

                c2c_count += 1

                used_frame_map[
                    (
                        source_c["pair"],
                        source_c["action"],
                        frame_id(source_c),
                        int(source_c["camera_id"])
                    )
                ] = source_c

                used_frame_map[
                    (
                        target["pair"],
                        target["action"],
                        frame_id(target),
                        int(target["camera_id"])
                    )
                ] = target

    sequence_summary.append({
        "split":
            split,

        "pair":
            pair_name,

        "action":
            action,

        "camera_id":
            camera_id,

        "total_frames":
            len(seq),

        "contact_frames":
            len(contacts),

        "noncontact_frames":
            len(noncontacts),

        "contact_ratio":
            (
                len(contacts) / len(seq)
                if seq
                else 0.0
            ),

        "N2C_pairs":
            n2c_count,

        "C2C_pairs":
            c2c_count,

        "total_pairs":
            len(seq_pairs),
    })


# ============================================================
# VALIDATION
# ============================================================

for p in all_pairs:

    assert (
        p["source_frame_id"]
        !=
        p["target_frame_id"]
    )

    assert p["target_contact"] is True

    if p["pair_type"] == "N2C":
        assert p["source_contact"] is False

    if p["pair_type"] == "C2C":
        assert p["source_contact"] is True


# ============================================================
# WRITE FRAME MANIFESTS
# ============================================================

write_jsonl(
    OUT / "frames_native_all.jsonl",
    rows
)

write_jsonl(
    OUT / "targets_contact_native.jsonl",
    all_contact_targets
)

write_jsonl(
    OUT / "sources_noncontact_native.jsonl",
    all_noncontact_sources
)

used_frames = sorted(
    used_frame_map.values(),
    key=lambda x: (
        x["split"],
        x["pair"],
        x["action"],
        frame_id(x)
    )
)

write_jsonl(
    OUT / "frames_used_by_interaction_pairs.jsonl",
    used_frames
)


# ============================================================
# WRITE PAIRS
# ============================================================

write_jsonl(
    OUT / "pairs_interaction_native.jsonl",
    all_pairs
)

for split in [
    "train",
    "val",
    "test"
]:
    write_jsonl(
        OUT /
        f"pairs_interaction_native_{split}.jsonl",

        [
            x for x in all_pairs
            if x["split"] == split
        ]
    )


# ============================================================
# SUMMARY
# ============================================================

frame_split = Counter(
    x["split"]
    for x in rows
)

contact_split = Counter(
    x["split"]
    for x in all_contact_targets
)

noncontact_split = Counter(
    x["split"]
    for x in all_noncontact_sources
)

pair_split = Counter(
    x["split"]
    for x in all_pairs
)

pair_type = Counter(
    x["pair_type"]
    for x in all_pairs
)

summary = {
    "source_frames":
        len(rows),

    "native_resolution_counts":
        dict(
            Counter(
                f"{x['native_width']}x{x['native_height']}"
                for x in rows
            )
        ),

    "frame_split":
        dict(frame_split),

    "contact_targets":
        len(all_contact_targets),

    "contact_targets_split":
        dict(contact_split),

    "noncontact_sources":
        len(all_noncontact_sources),

    "noncontact_sources_split":
        dict(noncontact_split),

    "interaction_pairs":
        len(all_pairs),

    "pair_split":
        dict(pair_split),

    "pair_type":
        dict(pair_type),

    "unique_used_frames":
        len(used_frames),

    "policy": {
        "target":
            "Hi4D official physical-contact frames only",

        "primary_pair":
            "N2C: non-contact source -> contact target",

        "secondary_pair":
            "C2C: contact source -> different contact target",

        "resize":
            "NONE at dataset construction",

        "crop":
            "NONE at dataset construction",

        "geometry":
            "render at DataLoader-selected model resolution",
    },

    "sequence_summary":
        sequence_summary,
}

with open(
    OUT / "summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False
    )


print()
print("======================================")
print("NATIVE INTERACTION DATASET READY")
print("======================================")

print(
    json.dumps(
        {
            "source_frames":
                summary["source_frames"],

            "native_resolution_counts":
                summary[
                    "native_resolution_counts"
                ],

            "contact_targets":
                summary[
                    "contact_targets"
                ],

            "contact_targets_split":
                summary[
                    "contact_targets_split"
                ],

            "noncontact_sources":
                summary[
                    "noncontact_sources"
                ],

            "interaction_pairs":
                summary[
                    "interaction_pairs"
                ],

            "pair_type":
                summary[
                    "pair_type"
                ],

            "pair_split":
                summary[
                    "pair_split"
                ],

            "unique_used_frames":
                summary[
                    "unique_used_frames"
                ],
        },
        indent=2,
        ensure_ascii=False
    )
)
