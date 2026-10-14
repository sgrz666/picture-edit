#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Project root for the categorized entrypoint.
from pathlib import Path as _EntryPath
import sys as _entry_sys
_entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))

import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/home/shangguanrz/project/pic-edit")

USABLE = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/manifests/"
      "usable_frame_pool.jsonl"
)

RAW_ROOT = (
    ROOT
    / "datasets/TikTokDataset/TikTok_Raw_Videos/"
      "TikTok_Raw_Videos"
)

OUT_DIR = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/manifests/splits"
)

OUT_ALL = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/manifests/"
      "usable_frame_pool_split.jsonl"
)

OUT_SUMMARY = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/manifests/"
      "split_summary.json"
)

SEED = 2026
SEARCH_ITERS = 50000

TARGET = {
    "train": 0.80,
    "val": 0.10,
    "test": 0.10,
}


def read_jsonl(path):
    return [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


# ============================================================
# 1. Discover raw source-video groups
# ============================================================

pattern = re.compile(
    r"^seq_(\d{5})_(\d{5})$"
)

groups = []

for p in sorted(RAW_ROOT.iterdir()):
    if not p.is_dir():
        continue

    m = pattern.match(p.name)

    if not m:
        continue

    start = int(m.group(1))
    end = int(m.group(2))

    groups.append({
        "group": p.name,
        "start": start,
        "end": end,
        "path": str(p),
    })


if not groups:
    raise RuntimeError(
        f"No raw groups found under {RAW_ROOT}"
    )


# sequence -> raw group
seq_to_group = {}

for g in groups:
    for seq_num in range(
        g["start"],
        g["end"] + 1,
    ):
        seq = f"{seq_num:05d}"

        if seq in seq_to_group:
            raise RuntimeError(
                f"Sequence mapped twice: {seq}"
            )

        seq_to_group[seq] = g["group"]


# ============================================================
# 2. Load usable frames
# ============================================================

rows = read_jsonl(USABLE)

unmapped = sorted({
    r["sequence_id"]
    for r in rows
    if r["sequence_id"] not in seq_to_group
})

if unmapped:
    raise RuntimeError(
        "Unmapped usable sequences: "
        + ", ".join(unmapped[:20])
    )


group_frames = Counter()
group_sequences = defaultdict(set)

for r in rows:
    seq = r["sequence_id"]
    group = seq_to_group[seq]

    group_frames[group] += 1
    group_sequences[group].add(seq)


# only groups that actually contain usable frames
active_groups = sorted(
    group_frames.keys()
)

n_groups = len(active_groups)

# For the expected 26 groups this becomes 20 / 3 / 3.
n_val = max(
    1,
    round(n_groups * 0.10),
)

n_test = max(
    1,
    round(n_groups * 0.10),
)

n_train = (
    n_groups
    - n_val
    - n_test
)

if n_train <= 0:
    raise RuntimeError(
        "Too few groups for train/val/test split"
    )


total_frames = len(rows)

all_sequences = {
    r["sequence_id"]
    for r in rows
}

total_sequences = len(all_sequences)


# ============================================================
# 3. Search deterministic group-level partition
# ============================================================

rng = random.Random(SEED)


def evaluate(
    train,
    val,
    test,
):
    split_groups = {
        "train": train,
        "val": val,
        "test": test,
    }

    frame_error = 0.0
    seq_error = 0.0

    for split, gs in split_groups.items():

        f = sum(
            group_frames[g]
            for g in gs
        )

        seqs = set()

        for g in gs:
            seqs.update(
                group_sequences[g]
            )

        frame_ratio = (
            f / total_frames
        )

        seq_ratio = (
            len(seqs)
            / total_sequences
        )

        frame_error += abs(
            frame_ratio
            - TARGET[split]
        )

        seq_error += abs(
            seq_ratio
            - TARGET[split]
        )

    # frame balance is primary;
    # sequence-count balance is secondary.
    return (
        frame_error
        + 0.25 * seq_error
    )


best = None
best_score = float("inf")

work = active_groups.copy()

for _ in range(SEARCH_ITERS):

    rng.shuffle(work)

    train = tuple(
        sorted(
            work[:n_train]
        )
    )

    val = tuple(
        sorted(
            work[
                n_train:
                n_train + n_val
            ]
        )
    )

    test = tuple(
        sorted(
            work[
                n_train + n_val:
            ]
        )
    )

    score = evaluate(
        train,
        val,
        test,
    )

    if score < best_score:
        best_score = score
        best = {
            "train": train,
            "val": val,
            "test": test,
        }


# ============================================================
# 4. Freeze group -> split map
# ============================================================

group_to_split = {}

for split, gs in best.items():
    for g in gs:
        group_to_split[g] = split


# absolute leakage check
assert (
    len(group_to_split)
    ==
    len(active_groups)
)


# ============================================================
# 5. Write frame manifests
# ============================================================

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

handles = {
    split: (
        OUT_DIR
        / f"{split}.jsonl"
    ).open(
        "w",
        encoding="utf-8",
    )
    for split in [
        "train",
        "val",
        "test",
    ]
}

all_handle = OUT_ALL.open(
    "w",
    encoding="utf-8",
)

split_counts = Counter()
split_sequences = defaultdict(set)

try:
    for r in rows:

        seq = r["sequence_id"]
        group = seq_to_group[seq]
        split = group_to_split[group]

        x = dict(r)

        x["source_video_group"] = group
        x["split"] = split

        line = (
            json.dumps(
                x,
                ensure_ascii=False,
            )
            + "\n"
        )

        handles[split].write(
            line
        )

        all_handle.write(
            line
        )

        split_counts[split] += 1
        split_sequences[split].add(seq)

finally:
    for f in handles.values():
        f.close()

    all_handle.close()


# ============================================================
# 6. Sequence-level frame statistics
# ============================================================

seq_frame_counts = Counter(
    r["sequence_id"]
    for r in rows
)

vals = list(
    seq_frame_counts.values()
)

sequence_stats = {
    "min":
        min(vals),

    "median":
        statistics.median(vals),

    "mean":
        statistics.mean(vals),

    "max":
        max(vals),
}


# ============================================================
# 7. Summary
# ============================================================

summary = {
    "version":
        "tiktok_v3_groupwise_split_v1",

    "seed":
        SEED,

    "policy": {
        "split_unit":
            "raw_source_video_group",

        "frame_level_random_split":
            False,

        "sequence_level_random_split":
            False,

        "target_ratio":
            TARGET,

        "group_counts_requested": {
            "train":
                n_train,

            "val":
                n_val,

            "test":
                n_test,
        },

        "reason":
            (
                "All sequences originating from the "
                "same raw source-video group are kept "
                "in one split to avoid source-video, "
                "identity and scene leakage."
            ),
    },

    "input": {
        "usable_frames":
            total_frames,

        "usable_sequences":
            total_sequences,

        "active_raw_groups":
            n_groups,
    },

    "splits": {},

    "sequence_frame_count":
        sequence_stats,

    "group_assignment": {
        split: list(gs)
        for split, gs in best.items()
    },
}


for split in [
    "train",
    "val",
    "test",
]:
    n = split_counts[split]

    summary[
        "splits"
    ][split] = {
        "frames":
            n,

        "frame_ratio":
            n / total_frames,

        "sequences":
            len(
                split_sequences[split]
            ),

        "groups":
            len(
                best[split]
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
print("TikTok V3 GROUP-WISE SPLIT")
print("=" * 76)

print(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    )
)
