#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3DPW 60-pair diverse visualization set v2

Goal
----
Build a visually diverse 60-pair showcase without changing
the official/final training manifests.

Selection
---------
Stage 1:
    - cover all 30 final sequences
    - exactly 1 representative pair / sequence
    - use the highest available pose-change level:
          large > medium > small
    - among those candidates, use DINOv2 diversity

Stage 2:
    - add another 30 pairs
    - DINOv2 + farthest-point sampling
    - prefer large / medium pose changes
    - do not reuse any RGB frame already selected
    - max 3 showcase pairs / sequence
    - limit repeated action-family exposure

Visual output
-------------
Each pair -> one 2 x 5 montage:

              RGB | DWPose | Mask | Depth | Normal
    Source
    Target

IMPORTANT
---------
This script is ONLY for representative visualization sampling.
It does not modify train/validation/test manifests.
"""

from pathlib import Path
from collections import Counter
import json
import random
import re

import numpy as np
import pandas as pd

from PIL import Image, ImageFile

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.transforms import InterpolationMode

import matplotlib.pyplot as plt


ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# Config
# ============================================================

PROJECT_ROOT = Path(
    "/home/shangguanrz/project/pic-edit"
)

DATASET_ROOT = (
    PROJECT_ROOT /
    "datasets/3dpw_single_final_v2"
)

OUT_DIR = (
    DATASET_ROOT /
    "visualization_60_pairs_diverse_dinov2_v2"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Existing local DINOv2 repository
LOCAL_DINO_REPO = (
    PROJECT_ROOT /
    "add/tools/dinov2_e1277af"
)

FEATURE_CACHE = (
    OUT_DIR /
    "dinov2_vits14_frame_features_224.npz"
)


SEED = 2026

MODEL_NAME = "dinov2_vits14"

IMAGE_SIZE = 224

BATCH_SIZE = 64

NUM_WORKERS = 8


# ------------------------------------------------------------
# Showcase constraints
# ------------------------------------------------------------

NUM_STAGE1 = 30
NUM_STAGE2 = 30
NUM_TOTAL = 60

MAX_PAIRS_PER_SEQUENCE = 3

# Stage 1 already guarantees every sequence.
# Stage 2 may add at most 2 further examples from one
# action-family.
MAX_STAGE2_EXTRA_PER_FAMILY = 2


# ------------------------------------------------------------
# Pose priority / diversity score
# ------------------------------------------------------------

POSE_PRIORITY = {
    "large": 3,
    "medium": 2,
    "small": 1,
}

# Added to DINO cosine-distance score during Stage 2.
# Diversity remains the main criterion.
POSE_BONUS = {
    "large": 0.10,
    "medium": 0.04,
    "small": 0.00,
}

# Discourage one sequence/family from dominating Stage 2.
SEQUENCE_REPEAT_PENALTY = 0.06
FAMILY_REPEAT_PENALTY = 0.04


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Clean stale PNG outputs from previous run
# ============================================================

for p in OUT_DIR.glob("*.png"):
    p.unlink()


# ============================================================
# Load final manifests
# ============================================================

parts = []

for split in [
    "train",
    "validation",
    "test",
]:

    csv_path = (
        DATASET_ROOT /
        f"{split}_pairs.csv"
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            csv_path
        )

    part = pd.read_csv(
        csv_path
    )

    part["split"] = split

    parts.append(
        part
    )


df = pd.concat(
    parts,
    ignore_index=True
)


# ============================================================
# Basic checks
# ============================================================

print("=" * 100)
print("3DPW FINAL DATASET")
print("=" * 100)

print(
    "pairs:",
    len(df)
)

print(
    "unique sequences:",
    df["sequence"].nunique()
)

print()
print("pairs by split:")

print(
    df["split"]
    .value_counts()
    .sort_index()
)

print()
print("sequences by split:")

print(
    df.groupby(
        "split"
    )[
        "sequence"
    ].nunique()
)


if df[
    "sequence"
].nunique() != 30:

    raise RuntimeError(
        "Expected 30 final sequences, got "
        f"{df['sequence'].nunique()}"
    )


# ============================================================
# Sequence family
#
# Examples:
#
# outdoors_climbing_00
# outdoors_climbing_01
# outdoors_climbing_02
#
# -> outdoors_climbing
#
# courtyard_relaxOnBench_00 / 01
# -> courtyard_relaxOnBench
# ============================================================

def sequence_family(
    sequence,
):

    return re.sub(
        r"_\d+$",
        "",
        str(sequence)
    )


df[
    "_family"
] = (
    df[
        "sequence"
    ]
    .astype(str)
    .map(
        sequence_family
    )
)


df[
    "_pose_priority"
] = (
    df[
        "pose_bin"
    ]
    .astype(str)
    .map(
        POSE_PRIORITY
    )
    .fillna(0)
    .astype(int)
)


# ============================================================
# Collect every unique RGB frame
#
# Important:
# DINO is extracted ONCE per frame, not once per pair.
#
# Final dataset has 15,708 pairs but only ~6,663 unique frames.
# ============================================================

frame_paths = sorted(
    set(
        df[
            "source_rgb"
        ].astype(str)
    )
    |
    set(
        df[
            "target_rgb"
        ].astype(str)
    )
)


print()
print("=" * 100)
print("DINO FRAME SET")
print("=" * 100)

print(
    "unique RGB frames:",
    len(frame_paths)
)


# ============================================================
# DINOv2 model
# ============================================================

def load_dinov2():

    errors = []

    # --------------------------------------------------------
    # 1. Prefer existing local official DINOv2 repo
    # --------------------------------------------------------

    if LOCAL_DINO_REPO.exists():

        print()
        print(
            "Trying local DINOv2 repo:"
        )

        print(
            LOCAL_DINO_REPO
        )

        try:

            model = torch.hub.load(
                str(
                    LOCAL_DINO_REPO
                ),
                MODEL_NAME,
                source="local",
                pretrained=True,
            )

            print(
                "✅ DINOv2 loaded from local repo"
            )

            return model

        except Exception as e:

            errors.append(
                (
                    "local",
                    repr(e),
                )
            )

            print(
                "Local DINOv2 load failed:"
            )

            print(
                repr(e)
            )


    # --------------------------------------------------------
    # 2. Official torch hub fallback
    #
    # This may download official DINOv2 weights if they are
    # not already cached.
    # --------------------------------------------------------

    print()
    print(
        "Trying official facebookresearch/dinov2 torch hub..."
    )

    try:

        model = torch.hub.load(
            "facebookresearch/dinov2",
            MODEL_NAME,
            pretrained=True,
            trust_repo=True,
        )

        print(
            "✅ DINOv2 loaded from official torch hub"
        )

        return model

    except Exception as e:

        errors.append(
            (
                "official_hub",
                repr(e),
            )
        )


    print()
    print("=" * 100)
    print("DINO LOAD ERRORS")
    print("=" * 100)

    for where, err in errors:

        print()
        print(
            where,
            ":"
        )

        print(
            err
        )


    raise RuntimeError(
        "Could not load official DINOv2. "
        "No heuristic/random model fallback is used."
    )


# ============================================================
# DINO preprocessing
# ============================================================

transform = transforms.Compose(
    [
        transforms.Resize(
            256,
            interpolation=InterpolationMode.BICUBIC,
        ),

        transforms.CenterCrop(
            IMAGE_SIZE
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


class RGBFrameDataset(
    Dataset
):

    def __init__(
        self,
        root,
        paths,
    ):

        self.root = root
        self.paths = paths


    def __len__(
        self
    ):

        return len(
            self.paths
        )


    def __getitem__(
        self,
        idx,
    ):

        rel = self.paths[
            idx
        ]

        path = (
            self.root /
            rel
        )

        if not path.exists():

            raise FileNotFoundError(
                path
            )


        image = Image.open(
            path
        ).convert(
            "RGB"
        )


        return transform(
            image
        )


# ============================================================
# Extract / load DINO features
# ============================================================

def extract_frame_features():

    # --------------------------------------------------------
    # Cache reuse
    # --------------------------------------------------------

    if FEATURE_CACHE.exists():

        print()
        print(
            "Found DINO feature cache:"
        )

        print(
            FEATURE_CACHE
        )


        cache = np.load(
            FEATURE_CACHE,
            allow_pickle=False
        )

        cached_paths = (
            cache[
                "paths"
            ]
            .astype(str)
            .tolist()
        )

        cached_features = (
            cache[
                "features"
            ]
            .astype(
                np.float32
            )
        )


        if cached_paths == frame_paths:

            print(
                "✅ feature cache matches current dataset"
            )

            print(
                "feature shape:",
                cached_features.shape
            )

            return cached_features


        print(
            "Cache path list does not match current dataset."
        )

        print(
            "Recomputing features..."
        )


    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is required for this DINO visualization selection."
        )


    device = torch.device(
        "cuda"
    )


    print()
    print(
        "CUDA:",
        torch.cuda.get_device_name(
            0
        )
    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = load_dinov2()

    model = model.to(
        device
    )

    model.eval()


    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    dataset = RGBFrameDataset(
        DATASET_ROOT,
        frame_paths,
    )


    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )


    all_features = []


    print()
    print("=" * 100)
    print("EXTRACTING DINOv2 FEATURES")
    print("=" * 100)


    with torch.inference_mode():

        for batch_id, images in enumerate(
            loader
        ):

            images = images.to(
                device,
                non_blocking=True
            )


            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):

                # Official DINOv2 exposes forward_features.
                if hasattr(
                    model,
                    "forward_features"
                ):

                    output = model.forward_features(
                        images
                    )

                    if isinstance(
                        output,
                        dict
                    ):

                        if (
                            "x_norm_clstoken"
                            in output
                        ):

                            features = output[
                                "x_norm_clstoken"
                            ]

                        elif (
                            "x_prenorm"
                            in output
                        ):

                            features = output[
                                "x_prenorm"
                            ][
                                :,
                                0
                            ]

                        else:

                            raise RuntimeError(
                                "Unknown DINOv2 "
                                "forward_features output keys: "
                                f"{list(output.keys())}"
                            )

                    else:

                        features = output

                else:

                    features = model(
                        images
                    )


            if features.ndim != 2:

                features = features.flatten(
                    1
                )


            features = F.normalize(
                features.float(),
                dim=1
            )


            all_features.append(
                features.cpu().numpy()
            )


            done = min(
                (
                    batch_id + 1
                )
                *
                BATCH_SIZE,
                len(dataset)
            )


            if (
                batch_id == 0
                or
                done % 512 < BATCH_SIZE
                or
                done == len(dataset)
            ):

                print(
                    f"  {done:5d} / "
                    f"{len(dataset):5d}"
                )


    features = np.concatenate(
        all_features,
        axis=0
    ).astype(
        np.float32
    )


    if len(
        features
    ) != len(
        frame_paths
    ):

        raise RuntimeError(
            "DINO feature count mismatch"
        )


    # --------------------------------------------------------
    # Save reusable cache
    # --------------------------------------------------------

    np.savez_compressed(
        FEATURE_CACHE,
        paths=np.asarray(
            frame_paths
        ),
        features=features,
    )


    print()
    print(
        "saved feature cache:"
    )

    print(
        FEATURE_CACHE
    )

    print(
        "feature shape:",
        features.shape
    )


    return features


frame_features = extract_frame_features()


# ============================================================
# Frame path -> feature index
# ============================================================

frame_to_index = {
    path: i
    for i, path
    in enumerate(
        frame_paths
    )
}


source_feature_index = (
    df[
        "source_rgb"
    ]
    .astype(str)
    .map(
        frame_to_index
    )
    .to_numpy(
        dtype=np.int64
    )
)


target_feature_index = (
    df[
        "target_rgb"
    ]
    .astype(str)
    .map(
        frame_to_index
    )
    .to_numpy(
        dtype=np.int64
    )
)


# ============================================================
# Pair DINO feature
#
# Keep source/target order because this is an editing pair:
#
# [source DINO || target DINO]
#
# Then L2 normalize.
# ============================================================

source_features = frame_features[
    source_feature_index
]

target_features = frame_features[
    target_feature_index
]


pair_features = np.concatenate(
    [
        source_features,
        target_features,
    ],
    axis=1
).astype(
    np.float32
)


pair_norm = np.linalg.norm(
    pair_features,
    axis=1,
    keepdims=True
)


pair_features = (
    pair_features
    /
    np.maximum(
        pair_norm,
        1e-12
    )
)


print()
print("=" * 100)
print("PAIR FEATURES")
print("=" * 100)

print(
    "pair feature shape:",
    pair_features.shape
)


# ============================================================
# Numpy metadata arrays
# ============================================================

sequence_arr = (
    df[
        "sequence"
    ]
    .astype(str)
    .to_numpy()
)

family_arr = (
    df[
        "_family"
    ]
    .astype(str)
    .to_numpy()
)

split_arr = (
    df[
        "split"
    ]
    .astype(str)
    .to_numpy()
)

pose_arr = (
    df[
        "pose_bin"
    ]
    .astype(str)
    .to_numpy()
)

priority_arr = (
    df[
        "_pose_priority"
    ]
    .to_numpy(
        dtype=np.int32
    )
)

source_path_arr = (
    df[
        "source_rgb"
    ]
    .astype(str)
    .to_numpy()
)

target_path_arr = (
    df[
        "target_rgb"
    ]
    .astype(str)
    .to_numpy()
)


# ============================================================
# Stage 1 candidate pool:
#
# For each sequence:
# only consider its highest available pose bin.
#
# Example:
# if climbing_00 has large pairs, Stage 1 only considers large.
# If a sequence has no large but has medium, use medium.
# ============================================================

sequences = sorted(
    df[
        "sequence"
    ]
    .astype(str)
    .unique()
)


stage1_candidates_by_sequence = {}


for seq in sequences:

    idx = np.where(
        sequence_arr
        ==
        seq
    )[
        0
    ]


    if len(
        idx
    ) == 0:

        raise RuntimeError(
            f"No pair for sequence {seq}"
        )


    max_priority = int(
        priority_arr[
            idx
        ].max()
    )


    candidates = idx[
        priority_arr[
            idx
        ]
        ==
        max_priority
    ]


    stage1_candidates_by_sequence[
        seq
    ] = candidates


# ============================================================
# Stage 1:
# 30 sequences -> 30 representatives
#
# FPS under sequence-coverage constraint.
# ============================================================

print()
print("=" * 100)
print("STAGE 1: ONE REPRESENTATIVE PER SEQUENCE")
print("=" * 100)


all_stage1_candidates = np.concatenate(
    list(
        stage1_candidates_by_sequence.values()
    )
)


# ------------------------------------------------------------
# Seed:
# choose a relatively central candidate rather than a random
# visual outlier. Then FPS spreads outward.
# ------------------------------------------------------------

centroid = pair_features[
    all_stage1_candidates
].mean(
    axis=0
)


centroid = (
    centroid
    /
    max(
        float(
            np.linalg.norm(
                centroid
            )
        ),
        1e-12
    )
)


centroid_similarity = (
    pair_features[
        all_stage1_candidates
    ]
    @
    centroid
)


seed_pos = int(
    np.argmax(
        centroid_similarity
    )
)


seed_idx = int(
    all_stage1_candidates[
        seed_pos
    ]
)


selected_indices = [
    seed_idx
]

selection_stage = {
    seed_idx:
        "sequence_coverage"
}

selection_score = {
    seed_idx:
        np.nan
}


represented_sequences = {
    sequence_arr[
        seed_idx
    ]
}


# Minimum cosine distance from each pair to selected set
min_distance = (
    1.0
    -
    (
        pair_features
        @
        pair_features[
            seed_idx
        ]
    )
)


while len(
    represented_sequences
) < 30:

    eligible = []


    for seq in sequences:

        if seq in represented_sequences:
            continue

        eligible.extend(
            stage1_candidates_by_sequence[
                seq
            ].tolist()
        )


    eligible = np.asarray(
        eligible,
        dtype=np.int64
    )


    if len(
        eligible
    ) == 0:

        raise RuntimeError(
            "Stage 1 ran out of candidates"
        )


    scores = min_distance[
        eligible
    ]


    best_pos = int(
        np.argmax(
            scores
        )
    )


    best_idx = int(
        eligible[
            best_pos
        ]
    )


    selected_indices.append(
        best_idx
    )

    selection_stage[
        best_idx
    ] = "sequence_coverage"

    selection_score[
        best_idx
    ] = float(
        scores[
            best_pos
        ]
    )


    represented_sequences.add(
        sequence_arr[
            best_idx
        ]
    )


    new_distance = (
        1.0
        -
        (
            pair_features
            @
            pair_features[
                best_idx
            ]
        )
    )


    min_distance = np.minimum(
        min_distance,
        new_distance
    )


if len(
    selected_indices
) != 30:

    raise RuntimeError(
        "Stage 1 did not produce exactly 30 pairs"
    )


print(
    "covered sequences:",
    len(
        represented_sequences
    )
)


# ============================================================
# Stage 2 initialization
# ============================================================

selected_set = set(
    selected_indices
)


used_frames = set()


sequence_counts = Counter()

family_counts = Counter()

family_stage2_extra = Counter()

split_counts = Counter()


for idx in selected_indices:

    used_frames.add(
        source_path_arr[
            idx
        ]
    )

    used_frames.add(
        target_path_arr[
            idx
        ]
    )


    seq = sequence_arr[
        idx
    ]

    fam = family_arr[
        idx
    ]

    split = split_arr[
        idx
    ]


    sequence_counts[
        seq
    ] += 1

    family_counts[
        fam
    ] += 1

    split_counts[
        split
    ] += 1


# ============================================================
# Stage 2:
#
# Add 30 diverse pairs.
#
# score =
#     DINO farthest-distance
#   + pose bonus
#   - same sequence repeat penalty
#   - same family repeat penalty
#
# Hard rules:
# - no frame reuse
# - max 3 showcase pairs per sequence
# - max 2 Stage-2 extras per action family
# ============================================================

print()
print("=" * 100)
print("STAGE 2: DINOv2 FARTHEST-POINT DIVERSITY")
print("=" * 100)


for step in range(
    NUM_STAGE2
):

    eligible = []

    scores = []


    for idx in range(
        len(
            df
        )
    ):

        if idx in selected_set:
            continue


        src = source_path_arr[
            idx
        ]

        tgt = target_path_arr[
            idx
        ]


        # ----------------------------------------------------
        # No image repetition in showcase
        # ----------------------------------------------------

        if (
            src in used_frames
            or
            tgt in used_frames
        ):

            continue


        seq = sequence_arr[
            idx
        ]

        fam = family_arr[
            idx
        ]

        pose_bin = pose_arr[
            idx
        ]


        # ----------------------------------------------------
        # Prevent a single video dominating
        # ----------------------------------------------------

        if (
            sequence_counts[
                seq
            ]
            >=
            MAX_PAIRS_PER_SEQUENCE
        ):

            continue


        # ----------------------------------------------------
        # Prevent climbing/freestyle/etc. families dominating
        # ----------------------------------------------------

        if (
            family_stage2_extra[
                fam
            ]
            >=
            MAX_STAGE2_EXTRA_PER_FAMILY
        ):

            continue


        # ----------------------------------------------------
        # Diversity is primary.
        # Pose magnitude is secondary.
        # ----------------------------------------------------

        score = float(
            min_distance[
                idx
            ]
        )


        score += POSE_BONUS.get(
            pose_bin,
            0.0
        )


        score -= (
            SEQUENCE_REPEAT_PENALTY
            *
            max(
                sequence_counts[
                    seq
                ]
                -
                1,
                0
            )
        )


        score -= (
            FAMILY_REPEAT_PENALTY
            *
            family_stage2_extra[
                fam
            ]
        )


        eligible.append(
            idx
        )

        scores.append(
            score
        )


    if not eligible:

        raise RuntimeError(
            f"Stage 2 exhausted at step {step}. "
            "Constraints are too strict."
        )


    scores = np.asarray(
        scores,
        dtype=np.float32
    )


    best_pos = int(
        np.argmax(
            scores
        )
    )


    best_idx = int(
        eligible[
            best_pos
        ]
    )


    selected_indices.append(
        best_idx
    )

    selected_set.add(
        best_idx
    )


    selection_stage[
        best_idx
    ] = "dinov2_fps"

    selection_score[
        best_idx
    ] = float(
        scores[
            best_pos
        ]
    )


    src = source_path_arr[
        best_idx
    ]

    tgt = target_path_arr[
        best_idx
    ]

    seq = sequence_arr[
        best_idx
    ]

    fam = family_arr[
        best_idx
    ]

    split = split_arr[
        best_idx
    ]


    used_frames.add(
        src
    )

    used_frames.add(
        tgt
    )


    sequence_counts[
        seq
    ] += 1

    family_counts[
        fam
    ] += 1

    family_stage2_extra[
        fam
    ] += 1

    split_counts[
        split
    ] += 1


    # --------------------------------------------------------
    # FPS update
    # --------------------------------------------------------

    new_distance = (
        1.0
        -
        (
            pair_features
            @
            pair_features[
                best_idx
            ]
        )
    )


    min_distance = np.minimum(
        min_distance,
        new_distance
    )


    print(
        f"[{step + 1:02d}/30] "
        f"{split:10s} | "
        f"{pose_arr[best_idx]:6s} | "
        f"{seq}"
    )


# ============================================================
# Final selected dataframe
# ============================================================

if len(
    selected_indices
) != NUM_TOTAL:

    raise RuntimeError(
        f"Expected {NUM_TOTAL}, "
        f"got {len(selected_indices)}"
    )


selected = df.iloc[
    selected_indices
].copy().reset_index(
    drop=True
)


selected[
    "selection_stage"
] = [
    selection_stage[
        idx
    ]
    for idx in selected_indices
]


selected[
    "diversity_selection_score"
] = [
    selection_score[
        idx
    ]
    for idx in selected_indices
]


selected[
    "sequence_family"
] = (
    selected[
        "sequence"
    ]
    .astype(str)
    .map(
        sequence_family
    )
)


# ============================================================
# Strong final validation
# ============================================================

if len(
    selected
) != 60:

    raise RuntimeError(
        "Not exactly 60 pairs"
    )


if selected[
    "sequence"
].nunique() != 30:

    raise RuntimeError(
        "Not all 30 sequences are represented"
    )


# ------------------------------------------------------------
# Every sequence >= 1
# ------------------------------------------------------------

seq_final_counts = (
    selected[
        "sequence"
    ]
    .value_counts()
)


if int(
    seq_final_counts.min()
) < 1:

    raise RuntimeError(
        "Some sequence is missing"
    )


if int(
    seq_final_counts.max()
) > MAX_PAIRS_PER_SEQUENCE:

    raise RuntimeError(
        "Sequence showcase cap violated"
    )


# ------------------------------------------------------------
# 120 RGB positions must be 120 unique frame files
# ------------------------------------------------------------

showcase_frames = (
    selected[
        "source_rgb"
    ].astype(str).tolist()
    +
    selected[
        "target_rgb"
    ].astype(str).tolist()
)


if len(
    set(
        showcase_frames
    )
) != 120:

    raise RuntimeError(
        "RGB frame reuse detected in the 60 showcase pairs"
    )


# ============================================================
# Print selection distribution
# ============================================================

print()
print("=" * 100)
print("FINAL 60-PAIR SHOWCASE")
print("=" * 100)


print()
print("Pose bins:")

print(
    selected[
        "pose_bin"
    ]
    .value_counts()
)


print()
print("Splits:")

print(
    selected[
        "split"
    ]
    .value_counts()
)


print()
print("Pairs per sequence:")

print(
    selected[
        "sequence"
    ]
    .value_counts()
    .sort_index()
)


print()
print("Pairs per action family:")

print(
    selected[
        "sequence_family"
    ]
    .value_counts()
    .sort_values(
        ascending=False
    )
)


print()
print(
    "Unique RGB frames:",
    len(
        set(
            showcase_frames
        )
    )
)


# ============================================================
# Image loading helpers
# ============================================================

def load_rgb(
    rel,
):

    path = (
        DATASET_ROOT /
        str(
            rel
        )
    )

    if not path.exists():

        raise FileNotFoundError(
            path
        )


    return np.asarray(
        Image.open(
            path
        ).convert(
            "RGB"
        )
    )


def load_gray(
    rel,
):

    path = (
        DATASET_ROOT /
        str(
            rel
        )
    )

    if not path.exists():

        raise FileNotFoundError(
            path
        )


    return np.asarray(
        Image.open(
            path
        ).convert(
            "L"
        )
    )


def load_depth(
    rel,
):

    path = (
        DATASET_ROOT /
        str(
            rel
        )
    )

    if not path.exists():

        raise FileNotFoundError(
            path
        )


    return np.asarray(
        Image.open(
            path
        )
    )


def depth_to_display(
    depth,
):
    """
    Only normalize metric depth for visualization.

    Original uint16 depth file remains unchanged.
    """

    depth = depth.astype(
        np.float32
    )

    valid = (
        depth > 0
    )


    vis = np.zeros_like(
        depth,
        dtype=np.float32
    )


    if not valid.any():

        return vis


    values = depth[
        valid
    ]


    lo = float(
        np.percentile(
            values,
            2
        )
    )

    hi = float(
        np.percentile(
            values,
            98
        )
    )


    if hi <= lo:
        hi = lo + 1.0


    normalized = (
        hi
        -
        depth
    ) / (
        hi
        -
        lo
    )


    normalized = np.clip(
        normalized,
        0.0,
        1.0
    )


    vis[
        valid
    ] = normalized[
        valid
    ]


    return vis


# ============================================================
# Draw one 2x5 pair montage
# ============================================================

def draw_pair(
    row,
    display_id,
):

    source_assets = [
        load_rgb(
            row[
                "source_rgb"
            ]
        ),

        load_rgb(
            row[
                "source_dwpose_render"
            ]
        ),

        load_gray(
            row[
                "source_person_mask"
            ]
        ),

        depth_to_display(
            load_depth(
                row[
                    "source_depth"
                ]
            )
        ),

        load_rgb(
            row[
                "source_normal"
            ]
        ),
    ]


    target_assets = [
        load_rgb(
            row[
                "target_rgb"
            ]
        ),

        load_rgb(
            row[
                "target_dwpose_render"
            ]
        ),

        load_gray(
            row[
                "target_person_mask"
            ]
        ),

        depth_to_display(
            load_depth(
                row[
                    "target_depth"
                ]
            )
        ),

        load_rgb(
            row[
                "target_normal"
            ]
        ),
    ]


    titles = [
        "RGB",
        "DWPose",
        "Person Mask",
        "SMPL Depth",
        "SMPL Normal",
    ]


    fig, axes = plt.subplots(
        2,
        5,
        figsize=(
            15,
            8
        )
    )


    for col in range(
        5
    ):

        if col in [
            2,
            3,
        ]:

            axes[
                0,
                col
            ].imshow(
                source_assets[
                    col
                ],
                cmap="gray",
            )


            axes[
                1,
                col
            ].imshow(
                target_assets[
                    col
                ],
                cmap="gray",
            )

        else:

            axes[
                0,
                col
            ].imshow(
                source_assets[
                    col
                ]
            )


            axes[
                1,
                col
            ].imshow(
                target_assets[
                    col
                ]
            )


    for col, title in enumerate(
        titles
    ):

        axes[
            0,
            col
        ].set_title(
            title,
            fontsize=14,
            fontweight="bold",
            pad=8,
        )


    axes[
        0,
        0
    ].set_ylabel(
        "SOURCE",
        fontsize=14,
        fontweight="bold",
        labelpad=12,
    )


    axes[
        1,
        0
    ].set_ylabel(
        "TARGET",
        fontsize=14,
        fontweight="bold",
        labelpad=12,
    )


    for ax in axes.flat:

        ax.set_xticks(
            []
        )

        ax.set_yticks(
            []
        )


        for spine in ax.spines.values():

            spine.set_visible(
                False
            )


    sequence = str(
        row[
            "sequence"
        ]
    )

    split = str(
        row[
            "split"
        ]
    )

    pose_bin = str(
        row[
            "pose_bin"
        ]
    )

    pair_id = int(
        row[
            "pair_id"
        ]
    )

    source_frame = int(
        row[
            "source_frame_index"
        ]
    )

    target_frame = int(
        row[
            "target_frame_index"
        ]
    )

    stage = str(
        row[
            "selection_stage"
        ]
    )


    fig.suptitle(
        (
            f"3DPW Example {display_id:02d}  |  "
            f"{split.upper()}  |  "
            f"Pose: {pose_bin.upper()}\n"
            f"{sequence}  |  "
            f"Pair {pair_id}  |  "
            f"Frame {source_frame} -> {target_frame}"
        ),
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )


    fig.text(
        0.5,
        0.018,
        (
            "RGB  |  DWPose WholeBody  |  "
            "Visible Person Mask  |  "
            "Official-SMPL Depth  |  "
            "Official-SMPL Normal"
        ),
        ha="center",
        fontsize=10,
    )


    plt.tight_layout(
        rect=[
            0.02,
            0.05,
            0.98,
            0.92,
        ]
    )


    safe_sequence = sequence.replace(
        "/",
        "_"
    )


    filename = (
        f"{display_id:02d}_"
        f"{split}_"
        f"{pose_bin}_"
        f"{stage}_"
        f"{safe_sequence}_"
        f"pair{pair_id}.png"
    )


    output_path = (
        OUT_DIR /
        filename
    )


    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )


    plt.close(
        fig
    )


    return output_path


# ============================================================
# Render 60 individual montages
# ============================================================

print()
print("=" * 100)
print("RENDERING 60 MONTAGES")
print("=" * 100)


output_images = []


for i, row in selected.iterrows():

    output_path = draw_pair(
        row,
        i + 1
    )


    output_images.append(
        str(
            output_path
        )
    )


    print(
        f"[{i + 1:02d}/60] "
        f"{row['selection_stage']:18s} | "
        f"{row['split']:10s} | "
        f"{row['pose_bin']:6s} | "
        f"{row['sequence']}"
    )


# ============================================================
# Save manifest
# ============================================================

selected.to_csv(
    OUT_DIR /
    "selected_60_pairs.csv",
    index=False,
)


# ============================================================
# Summary
# ============================================================

summary = {
    "dataset":
        "3dpw_single_final_v2",

    "method":
        (
            "30-sequence coverage + "
            "DINOv2 ViT-S/14 farthest-point diversity sampling"
        ),

    "num_pairs":
        int(
            len(
                selected
            )
        ),

    "unique_rgb_frames":
        int(
            len(
                set(
                    showcase_frames
                )
            )
        ),

    "unique_sequences":
        int(
            selected[
                "sequence"
            ].nunique()
        ),

    "unique_action_families":
        int(
            selected[
                "sequence_family"
            ].nunique()
        ),

    "pose_bins": {
        k: int(v)
        for k, v
        in selected[
            "pose_bin"
        ]
        .value_counts()
        .sort_index()
        .items()
    },

    "splits": {
        k: int(v)
        for k, v
        in selected[
            "split"
        ]
        .value_counts()
        .sort_index()
        .items()
    },

    "pairs_per_sequence": {
        k: int(v)
        for k, v
        in selected[
            "sequence"
        ]
        .value_counts()
        .sort_index()
        .items()
    },

    "pairs_per_family": {
        k: int(v)
        for k, v
        in selected[
            "sequence_family"
        ]
        .value_counts()
        .sort_index()
        .items()
    },

    "constraints": {
        "all_30_sequences_covered":
            True,

        "rgb_frame_reuse":
            False,

        "max_pairs_per_sequence":
            MAX_PAIRS_PER_SEQUENCE,

        "max_stage2_extra_per_family":
            MAX_STAGE2_EXTRA_PER_FAMILY,

        "pose_preference":
            "large > medium > small",
    },

    "feature_model":
        MODEL_NAME,

    "feature_image_size":
        IMAGE_SIZE,

    "outputs":
        output_images,
}


with open(
    OUT_DIR /
    "summary.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2,
    )


print()
print("=" * 100)
print("FINISHED")
print("=" * 100)

print(
    "pairs              :",
    len(
        selected
    )
)

print(
    "unique RGB frames  :",
    len(
        set(
            showcase_frames
        )
    )
)

print(
    "unique sequences   :",
    selected[
        "sequence"
    ].nunique()
)

print(
    "action families    :",
    selected[
        "sequence_family"
    ].nunique()
)

print()
print(
    "pose distribution:"
)

print(
    selected[
        "pose_bin"
    ]
    .value_counts()
)

print()
print(
    "sequence count distribution:"
)

print(
    selected[
        "sequence"
    ]
    .value_counts()
    .value_counts()
    .sort_index()
    .rename_axis(
        "pairs_per_sequence"
    )
    .rename(
        "num_sequences"
    )
)

print()
print(
    "saved:"
)

print(
    OUT_DIR
)

print()
print(
    "✅ diverse 60-pair showcase ready"
)
