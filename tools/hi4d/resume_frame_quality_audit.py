#!/usr/bin/env python3

import csv
import json
from pathlib import Path

import numpy as np

import frame_quality_audit_project as m


def as_bool(x):
    return str(x).strip().lower() in {
        "true", "1", "yes"
    }


def as_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def pct(values, q):
    arr = np.asarray(
        [
            v for v in values
            if np.isfinite(v)
        ],
        dtype=float
    )

    if len(arr) == 0:
        return np.nan

    return float(
        np.percentile(arr, q)
    )


def recover_existing_summary(dataset):

    csv_path = (
        m.OUT
        / dataset
        / "frame_quality_report.csv"
    )

    if not csv_path.exists():
        print(
            f"{dataset}: no existing CSV, "
            "cannot recover."
        )
        return None

    with open(
        csv_path,
        "r",
        encoding="utf-8-sig"
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    if not rows:
        return None

    shorts = [
        int(float(r["short_side"]))
        for r in rows
    ]

    laps = [
        as_float(r["primary_lap"])
        for r in rows
    ]

    tens = [
        as_float(r["primary_ten"])
        for r in rows
    ]

    summary = {
        "dataset":
            dataset,

        "root":
            str(
                Path(
                    rows[0]["path"]
                ).parents[3]
            ),

        "total_frames":
            len(rows),

        "mask_found_count":
            sum(
                as_bool(
                    r["mask_found"]
                )
                for r in rows
            ),

        "face_detected_count":
            sum(
                np.isfinite(
                    as_float(
                        r["face_lap"]
                    )
                )
                for r in rows
            ),

        "low_res_lt512_count":
            sum(
                as_bool(
                    r["low_res_lt512"]
                )
                for r in rows
            ),

        "short_side": {
            "min":
                min(shorts),

            "p50":
                pct(
                    shorts,
                    50
                ),

            "max":
                max(shorts),
        },

        "primary_lap": {
            "p5":
                pct(laps, 5),

            "p10":
                pct(laps, 10),

            "p50":
                pct(laps, 50),

            "p90":
                pct(laps, 90),
        },

        "primary_ten": {
            "p5":
                pct(tens, 5),

            "p10":
                pct(tens, 10),

            "p50":
                pct(tens, 50),

            "p90":
                pct(tens, 90),
        },

        "severe_suspect_count":
            sum(
                as_bool(
                    r["severe_suspect"]
                )
                for r in rows
            ),

        "suspect_blur_count":
            sum(
                as_bool(
                    r["suspect_blur"]
                )
                for r in rows
            ),

        "note":
            (
                "Recovered from existing "
                "frame_quality_report.csv. "
                "No image reprocessing."
            ),
    }

    summary = m.to_builtin(
        summary
    )

    summary_path = (
        m.OUT
        / dataset
        / "summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print(
        f"RECOVERED {dataset}: "
        f"{len(rows)} frames"
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2
        )
    )

    return summary


def write_combined(summaries):

    output = (
        m.OUT
        / "combined_summary.csv"
    )

    fields = [
        "dataset",
        "total_frames",
        "mask_found_count",
        "face_detected_count",
        "low_res_lt512_count",
        "severe_suspect_count",
        "suspect_blur_count",
    ]

    with open(
        output,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        for summary in summaries:

            writer.writerow({
                key:
                    summary.get(
                        key,
                        ""
                    )
                for key in fields
            })

    print(
        "\nCOMBINED:",
        output
    )


def main():

    summaries = []

    # --------------------------------
    # iPER：已经计算完，不重复计算
    # --------------------------------

    iper = recover_existing_summary(
        "iPER"
    )

    if iper:
        summaries.append(
            iper
        )

    # --------------------------------
    # TikTok：从这里继续
    # --------------------------------

    print(
        "\n\n"
        + "=" * 70
    )
    print(
        "CONTINUE: TikTok"
    )
    print(
        "=" * 70
    )

    root, files = (
        m.discover_tiktok()
    )

    tiktok = m.audit(
        "TikTok",
        root,
        files
    )

    if tiktok:
        summaries.append(
            tiktok
        )

    # --------------------------------
    # Hi4D：最后继续
    # --------------------------------

    print(
        "\n\n"
        + "=" * 70
    )
    print(
        "CONTINUE: Hi4D"
    )
    print(
        "=" * 70
    )

    root, files = (
        m.discover_hi4d()
    )

    hi4d = m.audit(
        "Hi4D",
        root,
        files
    )

    if hi4d:
        summaries.append(
            hi4d
        )

    write_combined(
        summaries
    )

    print()
    print(
        "=" * 70
    )

    print(
        "RESUME ALL DONE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
