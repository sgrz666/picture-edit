#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from collections import defaultdict, Counter
from pathlib import Path


PROJECT = Path("/home/shangguanrz/project/pic-edit")

ROOT = (
    PROJECT
    / "datasets"
    / "Hi4D_pilot_v1"
)

V41_ROOT = (
    ROOT
    / "native_interaction_v4_1"
)

OUT_ROOT = (
    ROOT
    / "native_interaction_v4_2"
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


def write_jsonl(path, rows):

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


# ============================================================
# Load V4.1
# ============================================================

pairs = read_jsonl(
    V41_ROOT
    / "main_pairs_all.jsonl"
)

aux_targets = read_jsonl(
    V41_ROOT
    / "aux_pending_c2c_dance_targets.jsonl"
)

main_targets = read_jsonl(
    V41_ROOT
    / "main_targets_unique.jsonl"
)

main_sources = read_jsonl(
    V41_ROOT
    / "main_sources_unique.jsonl"
)


print()
print("=" * 76)
print("Hi4D INTERACTION V4.2")
print("=" * 76)

print(
    "V4.1 pair rows :",
    len(pairs)
)

print(
    "MAIN targets   :",
    len(main_targets)
)

print(
    "AUX targets    :",
    len(aux_targets)
)


# ============================================================
# Group by Target
# ============================================================

def target_key(r):

    return (
        r["split"],
        r["pair"],
        int(r["camera_id"]),
        r["target_action"],
        int(r["target_frame_id"]),
    )


groups = defaultdict(
    list
)


for row in pairs:

    groups[
        target_key(row)
    ].append(row)


print(
    "unique paired targets:",
    len(groups)
)


if len(groups) != 1378:

    raise RuntimeError(
        f"expected 1378 MAIN targets, "
        f"got {len(groups)}"
    )


# ============================================================
# V4.2 policy
#
# If Cross exists:
#     keep 2 Cross + 1 Same
#
# If no Cross:
#     keep only 1 Same
#
# No artificial duplication.
# ============================================================

final_pairs = []

case_counter = Counter()


for key in sorted(groups.keys()):

    rows = groups[
        key
    ]


    cross = [
        r
        for r in rows
        if r[
            "pair_relation"
        ]
        ==
        "CROSS_ACTION"
    ]


    same = [
        r
        for r in rows
        if r[
            "pair_relation"
        ]
        ==
        "SAME_ACTION"
    ]


    # deterministic
    cross = sorted(
        cross,
        key=lambda r: (
            r["source_action"],
            int(
                r["source_frame_id"]
            ),
        ),
    )


    same = sorted(
        same,
        key=lambda r: (
            r["source_action"],
            int(
                r["source_frame_id"]
            ),
        ),
    )


    selected = []


    # ========================================================
    # Case A:
    # Cross-action exists.
    # Keep 2 Cross + 1 Same.
    # ========================================================

    if cross:

        selected.extend(
            cross[
                :2
            ]
        )


        if same:

            selected.append(
                same[0]
            )


        case = (
            f"{min(2, len(cross))}C_"
            f"{1 if same else 0}S"
        )


    # ========================================================
    # Case B:
    # No Cross.
    # Keep exactly 1 Same.
    # ========================================================

    else:

        if not same:

            raise RuntimeError(
                f"Target has neither "
                f"Cross nor Same source: {key}"
            )


        selected.append(
            same[0]
        )


        case = "0C_1S"


    # Add V4.2 metadata.
    for r in selected:

        x = dict(
            r
        )

        x[
            "pairing_version"
        ] = (
            "native_interaction_v4_2"
        )

        x[
            "v4_2_target_case"
        ] = case

        x[
            "v4_2_policy"
        ] = (
            "keep_2cross_1same_if_cross_exists_"
            "else_keep_1same"
        )

        final_pairs.append(
            x
        )


    case_counter[
        case
    ] += 1


# ============================================================
# Sort
# ============================================================

final_pairs = sorted(
    final_pairs,
    key=lambda r: (
        r["split"],
        r["pair"],
        int(
            r["camera_id"]
        ),
        r["target_action"],
        int(
            r["target_frame_id"]
        ),
        r["pair_relation"],
        r["source_action"],
        int(
            r["source_frame_id"]
        ),
    ),
)


# ============================================================
# HARD validation
# ============================================================

final_target_keys = {
    target_key(r)
    for r in final_pairs
}


if len(
    final_target_keys
) != 1378:

    raise RuntimeError(
        f"lost MAIN targets: "
        f"{len(final_target_keys)}"
    )


for r in final_pairs:

    if bool(
        r.get(
            "source_contact",
            False
        )
    ):

        raise RuntimeError(
            "contact Source detected"
        )


    if not bool(
        r.get(
            "target_contact",
            True
        )
    ):

        raise RuntimeError(
            "non-contact Target detected"
        )


    same_action = (
        r[
            "source_action"
        ]
        ==
        r[
            "target_action"
        ]
    )


    if (
        r[
            "pair_relation"
        ]
        ==
        "SAME_ACTION"
        and
        not same_action
    ):

        raise RuntimeError(
            "SAME_ACTION label mismatch"
        )


    if (
        r[
            "pair_relation"
        ]
        ==
        "CROSS_ACTION"
        and
        same_action
    ):

        raise RuntimeError(
            "CROSS_ACTION label mismatch"
        )


# ============================================================
# Statistics
# ============================================================

relation_counts = Counter(
    r[
        "pair_relation"
    ]
    for r in final_pairs
)


cross = int(
    relation_counts[
        "CROSS_ACTION"
    ]
)

same = int(
    relation_counts[
        "SAME_ACTION"
    ]
)

total = len(
    final_pairs
)


split_pair_counts = Counter(
    r[
        "split"
    ]
    for r in final_pairs
)


# Unique actually used Sources.
used_sources = {
    (
        r["split"],
        r["pair"],
        int(r["camera_id"]),
        r["source_action"],
        int(r["source_frame_id"]),
    )
    for r in final_pairs
}


# ============================================================
# Output
# ============================================================

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


write_jsonl(
    OUT_ROOT
    / "main_pairs_all.jsonl",
    final_pairs,
)


for split in [
    "train",
    "val",
    "test",
]:

    write_jsonl(
        OUT_ROOT
        / f"main_pairs_{split}.jsonl",

        [
            r
            for r in final_pairs
            if r[
                "split"
            ]
            ==
            split
        ],
    )


# Target registry unchanged.
write_jsonl(
    OUT_ROOT
    / "main_targets_unique.jsonl",
    main_targets,
)


# AUX unchanged.
write_jsonl(
    OUT_ROOT
    / "aux_pending_c2c_dance_targets.jsonl",
    aux_targets,
)


# Keep only Sources actually referenced in V4.2.
source_lookup = {
    (
        r["split"],
        r["pair"],
        int(r["camera_id"]),
        r["action"],
        int(r["frame_id"]),
    ):
    r

    for r in main_sources
}


final_sources = []


for key in sorted(
    used_sources
):

    if key not in source_lookup:

        raise RuntimeError(
            f"missing source registry row: "
            f"{key}"
        )

    final_sources.append(
        source_lookup[
            key
        ]
    )


write_jsonl(
    OUT_ROOT
    / "main_sources_unique.jsonl",
    final_sources,
)


# ============================================================
# Summary
# ============================================================

summary = {
    "version":
        "native_interaction_v4_2",

    "task":
        (
            "non-contact Source RGB + "
            "contact Target geometry -> "
            "contact Target RGB"
        ),

    "policy":
        (
            "For targets with legal cross-action "
            "sources: keep 2 CROSS_ACTION + "
            "1 SAME_ACTION. "
            "For targets without cross-action "
            "sources: keep only 1 SAME_ACTION. "
            "No artificial same-action duplication."
        ),

    "counts": {
        "frozen_targets_total":
            1458,

        "main_unique_targets":
            1378,

        "aux_unique_targets":
            len(
                aux_targets
            ),

        "main_pair_rows":
            total,

        "cross_action_pairs":
            cross,

        "same_action_pairs":
            same,

        "unique_used_sources":
            len(
                final_sources
            ),

        "main_pairs_train":
            int(
                split_pair_counts[
                    "train"
                ]
            ),

        "main_pairs_val":
            int(
                split_pair_counts[
                    "val"
                ]
            ),

        "main_pairs_test":
            int(
                split_pair_counts[
                    "test"
                ]
            ),
    },

    "ratios": {
        "cross_action":
            cross / total,

        "same_action":
            same / total,

        "cross_to_same":
            cross / same,
    },

    "target_cases": {
        k:
            int(v)

        for k, v
        in sorted(
            case_counter.items()
        )
    },
}


with open(
    OUT_ROOT
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


policy = {
    "MAIN_N2C": {
        "source_must_be_noncontact":
            True,

        "target_must_be_contact":
            True,

        "same_identity_pair":
            True,

        "same_camera":
            True,

        "cross_action_rule":
            (
                "If available, retain two "
                "cross-action Sources."
            ),

        "same_action_rule":
            (
                "Retain one same-action Source "
                "per Target."
            ),

        "no_cross_fallback":
            (
                "If no cross-action Source exists, "
                "retain only one same-action Source."
            ),
    },

    "AUX_PENDING_C2C_DANCE": {
        "enabled":
            False,

        "scope":
            "pair14/dance14/cam16",

        "target_count":
            len(
                aux_targets
            ),
    },
}


with open(
    OUT_ROOT
    / "dataset_policy.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        policy,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# Display
# ============================================================

print()
print("=" * 76)
print("V4.2 BUILD COMPLETE")
print("=" * 76)

print(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    )
)

print()
print(
    "Expected approximately:"
)

print(
    "Cross = 1362"
)

print(
    "Same  = 1378"
)

print(
    "Total = 2740"
)

print()
print(
    "OUTPUT:",
    OUT_ROOT
)
