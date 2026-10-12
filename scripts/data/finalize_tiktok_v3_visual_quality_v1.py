#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Project root for the categorized entrypoint.
from pathlib import Path as _EntryPath
import sys as _entry_sys
_entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path("/home/shangguanrz/project/pic-edit")

CSV_IN = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/qc/"
      "visual_quality_v3/visual_quality_scores_v3.csv"
)

SUMMARY_V3 = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/qc/"
      "visual_quality_v3/visual_quality_summary_v3.json"
)

OUT_ROOT = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/qc/"
      "visual_quality_final_v1"
)

OUT_JSONL = (
    OUT_ROOT
    / "visual_quality_decisions.jsonl"
)

GLOBAL_REJECT = (
    OUT_ROOT
    / "global_hard_blur.jsonl"
)

SOURCE_EXCLUDE = (
    OUT_ROOT
    / "source_exclude_face.jsonl"
)

HAND_RISK = (
    OUT_ROOT
    / "source_hand_risk.jsonl"
)

SUMMARY_OUT = (
    OUT_ROOT
    / "visual_quality_final_summary.json"
)


def as_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def as_bool(x):
    return str(x).strip().lower() in {
        "1",
        "true",
        "yes",
    }


def write_jsonl(path, rows):
    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for r in rows:
            f.write(
                json.dumps(
                    r,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():
    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_v3 = json.loads(
        SUMMARY_V3.read_text(
            encoding="utf-8"
        )
    )

    recovery_q99 = float(
        summary_v3[
            "data_driven_temporal_thresholds"
        ][
            "recovery_q99"
        ]
    )

    rows = []

    with CSV_IN.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        for r in reader:
            rows.append(r)

    decisions = []
    hard_rows = []
    face_rows = []
    hand_rows = []

    status_counts = Counter()
    split_counts = Counter()
    source_ok_counts = Counter()
    target_ok_counts = Counter()

    for r in rows:

        rec = as_float(
            r[
                "temporal_recovery_ratio"
            ]
        )

        edge_seq_p = as_float(
            r[
                "seq_edge_percentile"
            ]
        )

        face_eval = as_bool(
            r[
                "face_evaluable"
            ]
        )

        face_gp = as_float(
            r[
                "face_global_percentile"
            ]
        )

        face_sp = as_float(
            r[
                "face_seq_percentile"
            ]
        )

        hand_candidate = as_bool(
            r[
                "hand_blur_candidate"
            ]
        )

        # ----------------------------------------------------
        # 1. GLOBAL HARD BLUR
        # Conservative target rejection:
        # temporal anomaly AND same-sequence low tail.
        # ----------------------------------------------------

        global_hard = (
            rec == rec
            and edge_seq_p == edge_seq_p
            and rec >= recovery_q99
            and edge_seq_p <= 0.10
        )

        # ----------------------------------------------------
        # 2. FACE SOURCE EXCLUDE
        #
        # Do NOT reject target only because of face blur.
        # ----------------------------------------------------

        face_source_exclude = (
            face_eval
            and (
                (
                    face_gp == face_gp
                    and face_gp <= 0.05
                )
                or
                (
                    face_sp == face_sp
                    and face_sp <= 0.05
                )
            )
        )

        # ----------------------------------------------------
        # 3. HAND SOFT RISK ONLY
        # ----------------------------------------------------

        hand_source_risk = bool(
            hand_candidate
        )

        target_ok = not global_hard

        source_ok = (
            not global_hard
            and not face_source_exclude
        )

        if global_hard:
            status = "GLOBAL_HARD_BLUR"

        elif face_source_exclude:
            status = "FACE_SOURCE_EXCLUDE"

        elif hand_source_risk:
            status = "HAND_SOURCE_RISK"

        else:
            status = "OK"

        x = {
            "split":
                r["split"],

            "sequence_id":
                r["sequence_id"],

            "stem":
                r["stem"],

            "rgb":
                r["rgb"],

            "visual_quality_status":
                status,

            "target_ok":
                target_ok,

            "source_ok":
                source_ok,

            "global_hard_blur":
                global_hard,

            "face_source_exclude":
                face_source_exclude,

            "hand_source_risk":
                hand_source_risk,

            "temporal_recovery_ratio":
                rec,

            "seq_edge_percentile":
                edge_seq_p,

            "face_global_percentile":
                face_gp,

            "face_seq_percentile":
                face_sp,

            "hand_risk_side":
                r[
                    "hand_risk_side"
                ],

            "global_blur_risk_score":
                as_float(
                    r[
                        "global_blur_risk_score"
                    ]
                ),

            "face_blur_risk_score":
                as_float(
                    r[
                        "face_blur_risk_score"
                    ]
                ),

            "hand_blur_risk_score":
                as_float(
                    r[
                        "hand_blur_risk_score"
                    ]
                ),
        }

        decisions.append(x)

        status_counts[
            status
        ] += 1

        split_counts[
            (
                r["split"],
                status,
            )
        ] += 1

        if source_ok:
            source_ok_counts[
                r["split"]
            ] += 1

        if target_ok:
            target_ok_counts[
                r["split"]
            ] += 1

        if global_hard:
            hard_rows.append(x)

        if face_source_exclude:
            face_rows.append(x)

        if hand_source_risk:
            hand_rows.append(x)

    write_jsonl(
        OUT_JSONL,
        decisions,
    )

    write_jsonl(
        GLOBAL_REJECT,
        hard_rows,
    )

    write_jsonl(
        SOURCE_EXCLUDE,
        face_rows,
    )

    write_jsonl(
        HAND_RISK,
        hand_rows,
    )

    split_summary = {}

    for split in [
        "train",
        "val",
        "test",
    ]:
        split_summary[
            split
        ] = {
            "GLOBAL_HARD_BLUR":
                split_counts[
                    (
                        split,
                        "GLOBAL_HARD_BLUR",
                    )
                ],

            "FACE_SOURCE_EXCLUDE":
                split_counts[
                    (
                        split,
                        "FACE_SOURCE_EXCLUDE",
                    )
                ],

            "HAND_SOURCE_RISK":
                split_counts[
                    (
                        split,
                        "HAND_SOURCE_RISK",
                    )
                ],

            "OK":
                split_counts[
                    (
                        split,
                        "OK",
                    )
                ],

            "target_ok":
                target_ok_counts[
                    split
                ],

            "source_ok":
                source_ok_counts[
                    split
                ],
        }

    summary = {
        "version":
            "tiktok_v3_visual_quality_final_v1",

        "input_frames":
            len(decisions),

        "policy": {
            "global_hard_blur":
                (
                    "temporal_recovery >= V3 q99 "
                    "AND sequence_edge_percentile <= 0.10"
                ),

            "global_rejects_target":
                True,

            "global_rejects_source":
                True,

            "face_blur_rejects_target":
                False,

            "face_blur_excludes_source":
                True,

            "hand_blur_rejects_target":
                False,

            "hand_blur_excludes_source":
                False,

            "hand_blur_source_role":
                "soft risk / ranking penalty only",

            "recovery_q99":
                recovery_q99,
        },

        "counts": {
            "GLOBAL_HARD_BLUR":
                len(
                    hard_rows
                ),

            "FACE_SOURCE_EXCLUDE":
                len(
                    face_rows
                ),

            "HAND_SOURCE_RISK":
                len(
                    hand_rows
                ),

            "OK":
                status_counts[
                    "OK"
                ],

            "target_ok":
                sum(
                    1
                    for x in decisions
                    if x[
                        "target_ok"
                    ]
                ),

            "source_ok":
                sum(
                    1
                    for x in decisions
                    if x[
                        "source_ok"
                    ]
                ),
        },

        "splits":
            split_summary,

        "outputs": {
            "decisions":
                str(
                    OUT_JSONL
                ),

            "global_hard":
                str(
                    GLOBAL_REJECT
                ),

            "source_face_exclude":
                str(
                    SOURCE_EXCLUDE
                ),

            "source_hand_risk":
                str(
                    HAND_RISK
                ),
        },

        "important":
            (
                "No existing usable-frame, split, source-pool "
                "or pair manifest has been modified yet."
            ),
    }

    SUMMARY_OUT.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 78)
    print(
        "TikTok V3 FINAL VISUAL QUALITY DECISION"
    )
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
