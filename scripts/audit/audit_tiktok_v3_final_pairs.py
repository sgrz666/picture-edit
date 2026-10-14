#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Project root for the categorized entrypoint.
from pathlib import Path as _EntryPath
import sys as _entry_sys
_entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))

import json
from pathlib import Path

ROOT = Path("/home/shangguanrz/project/pic-edit")

PAIR_ROOT = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/manifests/pairs"
)

SPLITS = [
    "train",
    "val",
    "test",
]

EXPECTED_TOTAL = 13812


def read_jsonl(path):
    return [
        json.loads(x)
        for x in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]


def main():
    print("=" * 78)
    print("TIKTOK V3 FINAL PAIR AUDIT")
    print("=" * 78)

    all_groups = {}

    total_pairs = 0
    missing_paths = 0
    bad_same_frame = 0

    for split in SPLITS:

        path = (
            PAIR_ROOT
            / f"{split}_pairs.jsonl"
        )

        rows = read_jsonl(path)

        total_pairs += len(rows)

        distances = [
            float(
                r["pose_distance"]
            )
            for r in rows
        ]

        print()
        print(
            f"===== {split.upper()} ====="
        )

        print(
            "pairs:",
            len(rows),
        )

        for thr in [
            0.01,
            0.03,
            0.05,
            0.08,
        ]:
            n = sum(
                d < thr
                for d in distances
            )

            ratio = (
                n / len(rows) * 100
                if rows
                else 0.0
            )

            print(
                f"pose_distance < {thr:.2f}: "
                f"{n:5d} "
                f"({ratio:6.3f}%)"
            )

        groups = {
            r["source_video_group"]
            for r in rows
        }

        all_groups[
            split
        ] = groups

        required = [
            "source_rgb",
            "target_rgb",
            "source_person_mask",
            "target_person_mask",
            "target_dwpose",
            "target_normal",
            "target_depth_vis",
            "target_smplx_mask",
        ]

        for r in rows:

            if (
                r["source_stem"]
                ==
                r["target_stem"]
            ):
                bad_same_frame += 1

            for k in required:

                p = Path(
                    r[k]
                )

                if not p.is_file():

                    missing_paths += 1

                    print(
                        "MISSING:",
                        split,
                        r["pair_id"],
                        k,
                        p,
                    )

    # --------------------------------------------------------
    # Group leakage
    # --------------------------------------------------------

    print()
    print(
        "===== GROUP LEAKAGE ====="
    )

    leakage = 0

    for a, b in [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ]:

        overlap = (
            all_groups[a]
            &
            all_groups[b]
        )

        leakage += len(
            overlap
        )

        print(
            f"{a} vs {b}:",
            len(overlap),
            sorted(overlap),
        )

    # --------------------------------------------------------
    # Unpaired targets
    # --------------------------------------------------------

    print()
    print(
        "===== UNPAIRED TARGETS ====="
    )

    total_unpaired = 0

    for split in SPLITS:

        p = (
            PAIR_ROOT
            / f"{split}_unpaired_targets.jsonl"
        )

        if not p.exists():
            print(
                split,
                0,
            )
            continue

        rows = read_jsonl(
            p
        )

        total_unpaired += len(
            rows
        )

        print(
            split,
            len(rows),
        )

        for r in rows:
            print(
                " ",
                r,
            )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("FINAL")
    print("=" * 78)

    print(
        "total_pairs          :",
        total_pairs,
    )

    print(
        "expected_total       :",
        EXPECTED_TOTAL,
    )

    print(
        "missing_paths        :",
        missing_paths,
    )

    print(
        "source==target       :",
        bad_same_frame,
    )

    print(
        "group_leakage        :",
        leakage,
    )

    print(
        "unpaired_targets     :",
        total_unpaired,
    )

    ok = (
        total_pairs
        ==
        EXPECTED_TOTAL
        and
        missing_paths
        ==
        0
        and
        bad_same_frame
        ==
        0
        and
        leakage
        ==
        0
    )

    print()

    if ok:
        print(
            "RESULT: FINAL PAIR AUDIT PASS"
        )
    else:
        print(
            "RESULT: FINAL PAIR AUDIT NEEDS REVIEW"
        )


if __name__ == "__main__":
    main()
