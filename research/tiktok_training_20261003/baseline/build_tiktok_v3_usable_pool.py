#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from collections import Counter


ROOT = Path("/home/shangguanrz/project/pic-edit")

GEOM = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/manifests/"
      "candidate_pool_geometry_valid.jsonl"
)

DWPOSE = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/manifests/"
      "dwpose_results.jsonl"
)

OUT_KEEP = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/manifests/"
      "usable_frame_pool.jsonl"
)

OUT_REJECT = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/manifests/"
      "frame_fine_reject.jsonl"
)

OUT_SUMMARY = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/manifests/"
      "usable_frame_pool_summary.json"
)


BODY_VALID_HARD = 10
BODY_MEAN_HARD = 0.45

BODY_VALID_SOFT = 12
BODY_MEAN_SOFT = 0.55


def read_jsonl(path):
    return [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


geom_rows = read_jsonl(GEOM)
dw_rows = read_jsonl(DWPOSE)

geom = {
    (r["sequence_id"], r["stem"]): r
    for r in geom_rows
}

dw = {
    (r["sequence_id"], r["stem"]): r
    for r in dw_rows
}


keep = []
reject = []

reason_counter = Counter()
soft_counter = Counter()


for key, g in geom.items():

    d = dw.get(key)

    reasons = []

    if d is None:
        reasons.append(
            "REJECT_DWPOSE_MISSING"
        )

    elif d.get("status") != "PASS":
        reasons.append(
            "REJECT_DWPOSE_FAILED"
        )

    else:
        detected_persons = int(
            d.get("detected_persons", 0)
        )

        body_valid = int(
            d.get("body_valid_03", 0)
        )

        body_mean = float(
            d.get("body_mean_conf", 0.0)
        )

        # ----------------------------------------------------
        # HARD REJECT
        # ----------------------------------------------------

        if detected_persons > 1:
            reasons.append(
                "REJECT_MULTI_PERSON"
            )

        if body_valid < BODY_VALID_HARD:
            reasons.append(
                "REJECT_BODY_KEYPOINTS"
            )

        if body_mean < BODY_MEAN_HARD:
            reasons.append(
                "REJECT_BODY_CONFIDENCE"
            )

    if reasons:

        x = dict(g)

        x["fine_filter_status"] = "REJECT"
        x["fine_filter_reasons"] = reasons

        if d is not None:
            x["dwpose"] = d

        reject.append(x)

        for reason in reasons:
            reason_counter[reason] += 1

        continue

    # --------------------------------------------------------
    # KEEP
    # --------------------------------------------------------

    x = dict(g)

    x["dwpose_png"] = d["dwpose_png"]
    x["dwpose_json"] = d["dwpose_json"]

    x["body_mean_conf"] = float(
        d["body_mean_conf"]
    )

    x["body_valid_03"] = int(
        d["body_valid_03"]
    )

    x["hand_mean_conf"] = float(
        d["hand_mean_conf"]
    )

    x["face_mean_conf"] = float(
        d["face_mean_conf"]
    )

    x["detected_persons"] = int(
        d["detected_persons"]
    )

    # --------------------------------------------------------
    # SOFT RISK
    # 不删除，只给后续 source / target selection 使用
    # --------------------------------------------------------

    soft_reasons = []

    if x["body_valid_03"] < BODY_VALID_SOFT:
        soft_reasons.append(
            "LOW_BODY_COMPLETENESS"
        )

    if x["body_mean_conf"] < BODY_MEAN_SOFT:
        soft_reasons.append(
            "LOW_BODY_CONFIDENCE"
        )

    x["pose_soft_risk"] = bool(
        soft_reasons
    )

    x["pose_soft_risk_reasons"] = (
        soft_reasons
    )

    x["fine_filter_status"] = "KEEP"

    for reason in soft_reasons:
        soft_counter[reason] += 1

    keep.append(x)


OUT_KEEP.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with OUT_KEEP.open(
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


summary = {
    "version":
        "tiktok_v3_usable_frame_pool_v1",

    "input_geometry_valid":
        len(geom_rows),

    "dwpose_records":
        len(dw_rows),

    "usable_keep":
        len(keep),

    "fine_reject":
        len(reject),

    "keep_ratio":
        len(keep) / len(geom_rows),

    "hard_policy": {
        "single_person_only":
            True,

        "max_detected_persons":
            1,

        "min_body_valid_03":
            BODY_VALID_HARD,

        "min_body_mean_conf":
            BODY_MEAN_HARD,

        "hand_hard_filter":
            False,

        "face_hard_filter":
            False,
    },

    "soft_policy": {
        "body_valid_03_below":
            BODY_VALID_SOFT,

        "body_mean_conf_below":
            BODY_MEAN_SOFT,
    },

    "hard_reject_reason_counts":
        dict(reason_counter),

    "soft_risk_counts":
        dict(soft_counter),

    "next_stage":
        (
            "Split usable frames, then select "
            "source/target and construct pairs."
        ),
}


OUT_SUMMARY.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


print("=" * 76)
print("TikTok V3 USABLE FRAME POOL")
print("=" * 76)

print(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    )
)
