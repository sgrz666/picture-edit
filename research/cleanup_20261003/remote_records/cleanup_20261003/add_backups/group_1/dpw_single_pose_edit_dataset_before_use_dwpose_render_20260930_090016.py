from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


# ============================================================
# 13-joint pose used by our 3DPW single descriptor
#
# Original OpenPose-style joints 1..13:
#
# 0  neck
# 1  right shoulder
# 2  right elbow
# 3  right wrist
# 4  left shoulder
# 5  left elbow
# 6  left wrist
# 7  right hip
# 8  right knee
# 9  right ankle
# 10 left hip
# 11 left knee
# 12 left ankle
# ============================================================

POSE_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),

    (0, 4),
    (4, 5),
    (5, 6),

    (0, 7),
    (7, 8),
    (8, 9),

    (0, 10),
    (10, 11),
    (11, 12),

    (7, 10),
]


BIN_NAMES = [
    "small",
    "medium",
    "large",
]


class DPWSinglePoseEditDataset(Dataset):
    """
    3DPW single-person pose-edit dataset.

    Task:
        Source RGB + Target Pose -> Target RGB

    Train:
        sequence-balanced
        -> source-uniform
        -> source-supported pose-bin random
        -> degree-aware target random

    Validation / Test:
        fixed seed-2026 evaluation manifest.

    Notes
    -----
    The target pose is reconstructed from the already-built
    normalized 13-joint 2D descriptor:

        pixel_xy = normalized_xy * torso_scale + center_xy

    No pose model is re-run here.
    """

    def __init__(
        self,
        root="/home/shangguanrz/project/pic-edit/add",
        split="train",
        image_size=512,
        samples_per_epoch=None,
        seed=2026,
    ):
        super().__init__()

        self.root = Path(root)

        self.asset_root = (
            self.root /
            "outputs/3dpw_assets_all_v1"
        )

        self.image_root = (
            self.root /
            "data/3DPW/imageFiles"
        )

        self.split = split

        self.image_size = int(
            image_size
        )

        self.seed = int(
            seed
        )

        self.epoch = 0


        if split not in {
            "train",
            "validation",
            "test",
        }:
            raise ValueError(
                f"Unsupported split: {split}"
            )


        # ====================================================
        # Load metadata
        # ====================================================

        meta_csv = (
            self.asset_root /
            "3dpw_single_frame_descriptors_v2.csv"
        )

        desc_npz = (
            self.asset_root /
            "3dpw_single_frame_descriptors_v2.npz"
        )

        pool_npz = (
            self.asset_root /
            "3dpw_single_candidate_pool_v2.npz"
        )

        sampling_npz = (
            self.asset_root /
            "3dpw_single_sampling_metadata_v2.npz"
        )


        self.meta = pd.read_csv(
            meta_csv
        )

        self.desc = np.load(
            desc_npz
        )

        self.pool = np.load(
            pool_npz
        )

        self.sampling = np.load(
            sampling_npz
        )


        self.pose_normalized = self.desc[
            "pose_normalized"
        ]

        self.pose_visible = self.desc[
            "pose_visible"
        ].astype(
            bool
        )

        self.center_xy = self.desc[
            "center_xy"
        ]

        self.torso_scale = self.desc[
            "torso_scale"
        ]


        self.sequence_image_cache = {}


        # ====================================================
        # Train
        # ====================================================

        if self.split == "train":

            split_id = self.sampling[
                "split_id"
            ]

            usable_source = self.sampling[
                "usable_source"
            ].astype(
                bool
            )

            sequence_id = self.sampling[
                "sequence_id"
            ]


            self.available_bin_mask = (
                self.sampling[
                    "available_bin_mask"
                ].astype(
                    bool
                )
            )


            train_usable = (
                (split_id == 0)
                &
                usable_source
            )


            train_sequence_ids = np.unique(
                sequence_id[
                    train_usable
                ]
            )


            self.train_sequence_sources = {}


            for seq_id in train_sequence_ids:

                sources = np.flatnonzero(
                    train_usable
                    &
                    (
                        sequence_id
                        ==
                        seq_id
                    )
                ).astype(
                    np.int32
                )

                if len(sources):

                    self.train_sequence_sources[
                        int(seq_id)
                    ] = sources


            self.train_sequence_ids = np.asarray(
                sorted(
                    self.train_sequence_sources.keys()
                ),
                dtype=np.int32
            )


            if samples_per_epoch is None:

                samples_per_epoch = int(
                    train_usable.sum()
                )


            self.samples_per_epoch = int(
                samples_per_epoch
            )


        # ====================================================
        # Fixed validation / test
        # ====================================================

        else:

            if self.split == "validation":

                manifest_path = (
                    self.asset_root /
                    "3dpw_single_validation_pairs_natural_seed2026_v1.csv"
                )

            else:

                manifest_path = (
                    self.asset_root /
                    "3dpw_single_test_pairs_natural_seed2026_v1.csv"
                )


            self.eval_manifest = pd.read_csv(
                manifest_path
            )


    # ========================================================
    # Public controls
    # ========================================================

    def set_epoch(
        self,
        epoch,
    ):
        """
        Change dynamic train pairs between epochs while
        keeping the run reproducible.
        """

        self.epoch = int(
            epoch
        )


    def __len__(
        self,
    ):

        if self.split == "train":

            return self.samples_per_epoch

        return len(
            self.eval_manifest
        )


    # ========================================================
    # Deterministic per-index RNG
    # ========================================================

    def _rng_for_index(
        self,
        index,
    ):

        seed = (
            self.seed
            +
            self.epoch * 1_000_003
            +
            int(index) * 97_409
        )

        return np.random.default_rng(
            seed
        )


    # ========================================================
    # RGB path handling
    # ========================================================

    def _sequence_images(
        self,
        sequence,
    ):

        sequence = str(
            sequence
        )


        if sequence in self.sequence_image_cache:

            return self.sequence_image_cache[
                sequence
            ]


        folder = (
            self.image_root /
            sequence
        )


        paths = []


        for ext in [
            "*.jpg",
            "*.jpeg",
            "*.png",
            "*.JPG",
            "*.JPEG",
            "*.PNG",
        ]:

            paths.extend(
                folder.glob(
                    ext
                )
            )


        paths = sorted(
            set(
                paths
            )
        )


        if not paths:

            raise FileNotFoundError(
                f"No RGB frames found: {folder}"
            )


        self.sequence_image_cache[
            sequence
        ] = paths


        return paths


    def _resolve_image_path(
        self,
        descriptor_index,
    ):

        row = self.meta.iloc[
            int(
                descriptor_index
            )
        ]


        sequence = str(
            row[
                "sequence"
            ]
        )

        frame_index = int(
            row[
                "frame_index"
            ]
        )


        images = self._sequence_images(
            sequence
        )


        if not (
            0
            <=
            frame_index
            <
            len(images)
        ):

            raise IndexError(
                f"{sequence}: frame_index={frame_index}, "
                f"RGB frames={len(images)}"
            )


        return images[
            frame_index
        ]


    def _load_rgb(
        self,
        descriptor_index,
    ):

        path = self._resolve_image_path(
            descriptor_index
        )


        with Image.open(
            path
        ) as image:

            image = image.convert(
                "RGB"
            )


        return (
            image,
            path
        )


    # ========================================================
    # Target pose rendering
    # ========================================================

    def _target_pose_pixels(
        self,
        descriptor_index,
    ):

        descriptor_index = int(
            descriptor_index
        )


        pose = self.pose_normalized[
            descriptor_index
        ].astype(
            np.float32
        )

        visible = self.pose_visible[
            descriptor_index
        ]

        center = self.center_xy[
            descriptor_index
        ].astype(
            np.float32
        )

        scale = float(
            self.torso_scale[
                descriptor_index
            ]
        )


        xy = (
            pose
            *
            scale
            +
            center[
                None,
                :
            ]
        )


        return (
            xy,
            visible
        )


    def _render_target_pose(
        self,
        descriptor_index,
        original_size,
    ):
        """
        White skeleton on black background.
        """

        width, height = original_size


        canvas = Image.new(
            "RGB",
            (
                width,
                height
            ),
            "black"
        )


        draw = ImageDraw.Draw(
            canvas
        )


        xy, visible = self._target_pose_pixels(
            descriptor_index
        )


        # Scale line thickness with image resolution.
        line_width = max(
            2,
            int(
                round(
                    min(
                        width,
                        height
                    )
                    /
                    220
                )
            )
        )


        joint_radius = max(
            3,
            int(
                round(
                    min(
                        width,
                        height
                    )
                    /
                    160
                )
            )
        )


        # ----------------------------------------------------
        # Skeleton
        # ----------------------------------------------------

        for a, b in POSE_EDGES:

            if not (
                visible[a]
                and
                visible[b]
            ):
                continue


            xa, ya = xy[
                a
            ]

            xb, yb = xy[
                b
            ]


            if not np.all(
                np.isfinite(
                    [
                        xa,
                        ya,
                        xb,
                        yb,
                    ]
                )
            ):
                continue


            draw.line(
                [
                    (
                        float(xa),
                        float(ya)
                    ),
                    (
                        float(xb),
                        float(yb)
                    ),
                ],
                fill="white",
                width=line_width
            )


        # ----------------------------------------------------
        # Joints
        # ----------------------------------------------------

        for joint_id in range(
            len(xy)
        ):

            if not visible[
                joint_id
            ]:
                continue


            x, y = xy[
                joint_id
            ]


            if not np.all(
                np.isfinite(
                    [
                        x,
                        y
                    ]
                )
            ):
                continue


            draw.ellipse(
                [
                    (
                        float(x - joint_radius),
                        float(y - joint_radius)
                    ),
                    (
                        float(x + joint_radius),
                        float(y + joint_radius)
                    ),
                ],
                fill="white"
            )


        return canvas


    # ========================================================
    # Tensor conversion
    # ========================================================

    def _to_tensor(
        self,
        image,
    ):

        image = image.resize(
            (
                self.image_size,
                self.image_size
            ),
            Image.Resampling.LANCZOS
        )


        tensor = TF.pil_to_tensor(
            image
        ).float()


        tensor = (
            tensor
            /
            255.0
        )


        return tensor


    def _pose_to_tensor(
        self,
        pose_image,
    ):

        pose_image = pose_image.resize(
            (
                self.image_size,
                self.image_size
            ),
            Image.Resampling.BILINEAR
        )


        tensor = TF.pil_to_tensor(
            pose_image
        ).float()


        tensor = (
            tensor
            /
            255.0
        )


        return tensor


    # ========================================================
    # Candidate sampling
    # ========================================================

    def _candidate_targets(
        self,
        source,
        bin_name,
    ):

        indptr = self.pool[
            f"{bin_name}_indptr"
        ]

        indices = self.pool[
            f"{bin_name}_indices"
        ]


        start = int(
            indptr[
                source
            ]
        )

        end = int(
            indptr[
                source + 1
            ]
        )


        return indices[
            start:end
        ]


    def _sample_train_pair(
        self,
        index,
    ):

        rng = self._rng_for_index(
            index
        )


        # ----------------------------------------------------
        # 1. sequence-balanced
        # ----------------------------------------------------

        seq_id = int(
            rng.choice(
                self.train_sequence_ids
            )
        )


        # ----------------------------------------------------
        # 2. source-uniform
        # ----------------------------------------------------

        source_pool = (
            self.train_sequence_sources[
                seq_id
            ]
        )


        source = int(
            rng.choice(
                source_pool
            )
        )


        # ----------------------------------------------------
        # 3. Natural available-bin random
        # ----------------------------------------------------

        available_ids = np.flatnonzero(
            self.available_bin_mask[
                source
            ]
        )


        if len(
            available_ids
        ) == 0:

            raise RuntimeError(
                f"Train source {source} has no candidate bin."
            )


        bin_id = int(
            rng.choice(
                available_ids
            )
        )


        bin_name = BIN_NAMES[
            bin_id
        ]


        # ----------------------------------------------------
        # 4. Degree-aware target
        # ----------------------------------------------------

        candidates = self._candidate_targets(
            source,
            bin_name
        )


        if len(
            candidates
        ) == 0:

            raise RuntimeError(
                f"Train source {source} has no "
                f"{bin_name} target."
            )


        degree = self.sampling[
            f"{bin_name}_count"
        ][
            candidates
        ].astype(
            np.float64
        )


        weights = (
            1.0
            /
            np.sqrt(
                np.maximum(
                    degree,
                    1.0
                )
            )
        )


        weights /= (
            weights.sum()
        )


        target = int(
            rng.choice(
                candidates,
                p=weights
            )
        )


        return (
            source,
            target,
            bin_name
        )


    # ========================================================
    # Get pair
    # ========================================================

    def _get_pair(
        self,
        index,
    ):

        if self.split == "train":

            return self._sample_train_pair(
                index
            )


        row = self.eval_manifest.iloc[
            int(
                index
            )
        ]


        return (
            int(
                row[
                    "source_descriptor_index"
                ]
            ),

            int(
                row[
                    "target_descriptor_index"
                ]
            ),

            str(
                row[
                    "pose_bin"
                ]
            ),
        )


    # ========================================================
    # __getitem__
    # ========================================================

    def __getitem__(
        self,
        index,
    ):

        source_idx, target_idx, pose_bin = (
            self._get_pair(
                index
            )
        )


        source_row = self.meta.iloc[
            source_idx
        ]

        target_row = self.meta.iloc[
            target_idx
        ]


        if (
            source_row[
                "sequence"
            ]
            !=
            target_row[
                "sequence"
            ]
        ):

            raise RuntimeError(
                "Cross-sequence pair detected."
            )


        # ----------------------------------------------------
        # RGB
        # ----------------------------------------------------

        source_image, source_path = self._load_rgb(
            source_idx
        )

        target_image, target_path = self._load_rgb(
            target_idx
        )


        # ----------------------------------------------------
        # Pose condition uses TARGET geometry
        # ----------------------------------------------------

        target_pose_image = self._render_target_pose(
            target_idx,
            target_image.size
        )


        # ----------------------------------------------------
        # Convert
        # ----------------------------------------------------

        source_tensor = self._to_tensor(
            source_image
        )

        target_tensor = self._to_tensor(
            target_image
        )

        pose_tensor = self._pose_to_tensor(
            target_pose_image
        )


        return {

            "source_rgb":
                source_tensor,

            "target_pose":
                pose_tensor,

            "target_rgb":
                target_tensor,

            "split":
                self.split,

            "sequence":
                str(
                    source_row[
                        "sequence"
                    ]
                ),

            "pose_bin":
                pose_bin,

            "source_descriptor_index":
                int(
                    source_idx
                ),

            "target_descriptor_index":
                int(
                    target_idx
                ),

            "source_frame_index":
                int(
                    source_row[
                        "frame_index"
                    ]
                ),

            "target_frame_index":
                int(
                    target_row[
                        "frame_index"
                    ]
                ),

            "source_path":
                str(
                    source_path
                ),

            "target_path":
                str(
                    target_path
                ),
        }
