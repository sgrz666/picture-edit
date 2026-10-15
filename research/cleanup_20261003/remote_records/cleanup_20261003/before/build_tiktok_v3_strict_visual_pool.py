#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path("/home/shangguanrz/project/pic-edit")

USABLE_SPLIT = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/manifests/"
      "usable_frame_pool_split.jsonl"
)

V3_CSV = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/qc/"
      "visual_quality_v3/visual_quality_scores_v3.csv"
)

FINAL_DECISIONS = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/qc/"
      "visual_quality_final_v1/visual_quality_decisions.jsonl"
)

OUT_ROOT = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/manifests/"
      "strict_visual_v1"
)

OUT_ALL = (
    OUT_ROOT
    / "usable_frame_pool_strict.jsonl"
)

OUT_REJECT = (
    OUT_ROOT
    / "visual_reject_strict.jsonl"
)

SUMMARY = (
    OUT_ROOT
    / "strict_visual_summary.json"
)


def read_jsonl(path):
    return [
        json.loads(x)
        for x in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]


def as_bool(x):
    return str(x).strip().lower() in {
        "true", "1", "yes"
    }


def main():

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    usable = read_jsonl(
        USABLE_SPLIT
    )

    decisions = read_jsonl(
        FINAL_DECISIONS
    )

    decision_map = {
        (
            r["sequence_id"],
            r["stem"],
        ): r
        for r in decisions
    }

    hand_map = {}

    with V3_CSV.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        for r in csv.DictReader(f):
            hand_map[
                (
                    r["sequence_id"],
                    r["stem"],
                )
            ] = as_bool(
                r["hand_blur_candidate"]
            )

    keep = []
    reject = []

    reason_counts = Counter()
    split_keep = Counter()
    split_reject = Counter()

    overlap_counts = Counter()

    for r in usable:

        key = (
            r["sequence_id"],
            r["stem"],
        )

        d = decision_map[key]

        global_blur = bool(
            d["global_hard_blur"]
        )

        face_blur = bool(
            d["face_source_exclude"]
        )

        hand_blur = bool(
            hand_map.get(
                key,
                False,
            )
        )

        reasons = []

        if global_blur:
            reasons.append(
                "GLOBAL_HARD_BLUR"
            )

        if face_blur:
            reasons.append(
                "FACE_BLUR"
            )

        if hand_blur:
            reasons.append(
                "HAND_BLUR"
            )

        if reasons:

            x = dict(r)

            x[
                "visual_filter_status"
            ] = "REJECT"

            x[
                "visual_reject_reasons"
            ] = reasons

            reject.append(x)

            split_reject[
                r["split"]
            ] += 1

            for reason in reasons:
                reason_counts[
                    reason
                ] += 1

            overlap_counts[
                "+".join(
                    sorted(reasons)
                )
            ] += 1

        else:

            x = dict(r)

            x[
                "visual_filter_status"
            ] = "KEEP"

            x[
                "visual_reject_reasons"
            ] = []

            keep.append(x)

            split_keep[
                r["split"]
            ] += 1

    with OUT_ALL.open(
        "w",
        encoding="utf-8",
    ) as f:

        for r in keep:
            f.write(
                json.dumps(
                    r,
                    ensure_ascii=False,
                )
                + "\n"
            )

    with OUT_REJECT.open(
        "w",
        encoding="utf-8",
    ) as f:

        for r in reject:
            f.write(
                json.dumps(
                    r,
                    ensure_ascii=False,
                )
                + "\n"
            )

    for split in [
        "train",
        "val",
        "test",
    ]:

        p = (
            OUT_ROOT
            / f"{split}.jsonl"
        )

        with p.open(
            "w",
            encoding="utf-8",
        ) as f:

            for r in keep:

                if r["split"] != split:
                    continue

                f.write(
                    json.dumps(
                        r,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    summary = {
        "version":
            "tiktok_v3_strict_visual_v1",

        "input_frames":
            len(usable),

        "keep":
            len(keep),

        "reject_union":
            len(reject),

        "keep_ratio":
            len(keep) / len(usable),

        "raw_reason_counts":
            dict(reason_counts),

        "overlap_patterns":
            dict(overlap_counts),

        "splits": {
            split: {
                "keep":
                    split_keep[split],

                "reject":
                    split_reject[split],
            }
            for split in [
                "train",
                "val",
                "test",
            ]
        },

        "policy": {
            "reject_global_hard_blur":
                True,

            "reject_face_blur":
                True,

            "reject_hand_blur":
                True,

            "delete_original_files":
                False,
        },

        "important":
            (
                "Original RGB, DWPose and geometry files "
                "remain untouched. This is a manifest-only "
                "strict filtering branch."
            ),
    }

    SUMMARY.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 78)
    print("TikTok V3 STRICT VISUAL POOL")
    print("=" * 78)

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
