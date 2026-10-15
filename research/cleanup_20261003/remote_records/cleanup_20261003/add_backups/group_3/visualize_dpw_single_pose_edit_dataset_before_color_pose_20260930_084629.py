import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


ROOT = Path(
    "/home/shangguanrz/project/pic-edit/add"
)

sys.path.insert(
    0,
    str(ROOT)
)


from datasets.dpw_single_pose_edit_dataset import (
    DPWSinglePoseEditDataset,
)


def tensor_to_pil(
    tensor,
):
    array = (
        tensor
        .detach()
        .cpu()
        .clamp(
            0,
            1
        )
        .permute(
            1,
            2,
            0
        )
        .numpy()
    )

    array = (
        array
        *
        255.0
    ).round().astype(
        np.uint8
    )


    return Image.fromarray(
        array
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "validation",
            "test",
        ],
        default="train",
    )

    parser.add_argument(
        "--num",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
    )

    args = parser.parse_args()


    dataset = DPWSinglePoseEditDataset(
        root=str(ROOT),
        split=args.split,
        image_size=args.image_size,
        samples_per_epoch=max(
            args.num,
            100
        ),
        seed=args.seed,
    )


    dataset.set_epoch(
        0
    )


    # ========================================================
    # Choose samples
    # ========================================================

    if args.split == "train":

        sample_indices = np.arange(
            args.num,
            dtype=np.int32
        )

    else:

        rng = np.random.default_rng(
            args.seed
        )

        sample_indices = rng.choice(
            len(dataset),
            size=min(
                args.num,
                len(dataset)
            ),
            replace=False
        )


    samples = []


    for index in sample_indices:

        sample = dataset[
            int(index)
        ]

        samples.append(
            sample
        )


    # ========================================================
    # Save manifest
    # ========================================================

    out_dir = (
        ROOT /
        "outputs/3dpw_assets_all_v1/"
        "_sanity_single_pose_edit_dataset"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    rows = []


    for i, sample in enumerate(
        samples
    ):

        rows.append(
            {
                "sample_id":
                    i,

                "split":
                    sample[
                        "split"
                    ],

                "sequence":
                    sample[
                        "sequence"
                    ],

                "pose_bin":
                    sample[
                        "pose_bin"
                    ],

                "source_descriptor_index":
                    sample[
                        "source_descriptor_index"
                    ],

                "target_descriptor_index":
                    sample[
                        "target_descriptor_index"
                    ],

                "source_frame_index":
                    sample[
                        "source_frame_index"
                    ],

                "target_frame_index":
                    sample[
                        "target_frame_index"
                    ],

                "source_path":
                    sample[
                        "source_path"
                    ],

                "target_path":
                    sample[
                        "target_path"
                    ],
            }
        )


    manifest = pd.DataFrame(
        rows
    )


    manifest_path = (
        out_dir /
        f"{args.split}_16_pairs_seed{args.seed}.csv"
    )


    manifest.to_csv(
        manifest_path,
        index=False
    )


    # ========================================================
    # Contact sheet
    # ========================================================

    N_COLS = 4

    N_ROWS = int(
        np.ceil(
            len(samples)
            /
            N_COLS
        )
    )


    PANEL = 210

    TITLE_H = 22

    INFO_H = 48

    GAP = 8


    SAMPLE_W = (
        PANEL * 3
        +
        GAP * 4
    )

    SAMPLE_H = (
        TITLE_H
        +
        PANEL
        +
        INFO_H
        +
        GAP * 3
    )


    canvas = Image.new(
        "RGB",
        (
            SAMPLE_W * N_COLS,
            SAMPLE_H * N_ROWS
        ),
        "white"
    )


    draw = ImageDraw.Draw(
        canvas
    )


    for i, sample in enumerate(
        samples
    ):

        row = (
            i
            //
            N_COLS
        )

        col = (
            i
            %
            N_COLS
        )


        base_x = (
            col
            *
            SAMPLE_W
        )

        base_y = (
            row
            *
            SAMPLE_H
        )


        source = tensor_to_pil(
            sample[
                "source_rgb"
            ]
        ).resize(
            (
                PANEL,
                PANEL
            )
        )


        # ----------------------------------------------------
        # Target Pose
        # DWPose-style visualization:
        # pure black background + preserve colored skeleton
        # ----------------------------------------------------

        pose = tensor_to_pil(
            sample[
                "target_pose"
            ]
        ).convert(
            "RGB"
        )

        pose_np = np.array(
            pose,
            dtype=np.uint8
        )

        # DWPose skeleton is colored while the original
        # visualization background may be white / light gray.
        # Remove only near-neutral bright background pixels.
        pose_max = pose_np.max(
            axis=2
        )

        pose_min = pose_np.min(
            axis=2
        )

        pose_chroma = (
            pose_max.astype(np.int16)
            -
            pose_min.astype(np.int16)
        )

        bg_mask = (
            (pose_max >= 185)
            &
            (pose_chroma <= 35)
        )

        pose_np[
            bg_mask
        ] = 0

        pose = Image.fromarray(
            pose_np,
            mode="RGB"
        ).resize(
            (
                PANEL,
                PANEL
            ),
            Image.Resampling.NEAREST
        )


        target = tensor_to_pil(
            sample[
                "target_rgb"
            ]
        ).resize(
            (
                PANEL,
                PANEL
            )
        )


        # ----------------------------------------------------
        # Titles
        # ----------------------------------------------------

        draw.text(
            (
                base_x + GAP,
                base_y + 3
            ),
            "Source RGB",
            fill="black"
        )


        draw.text(
            (
                base_x
                +
                GAP * 2
                +
                PANEL,
                base_y + 3
            ),
            "Target Pose",
            fill="black"
        )


        draw.text(
            (
                base_x
                +
                GAP * 3
                +
                PANEL * 2,
                base_y + 3
            ),
            "Target RGB",
            fill="black"
        )


        image_y = (
            base_y
            +
            TITLE_H
            +
            GAP
        )


        x1 = (
            base_x
            +
            GAP
        )

        x2 = (
            x1
            +
            PANEL
            +
            GAP
        )

        x3 = (
            x2
            +
            PANEL
            +
            GAP
        )


        canvas.paste(
            source,
            (
                x1,
                image_y
            )
        )

        canvas.paste(
            pose,
            (
                x2,
                image_y
            )
        )

        canvas.paste(
            target,
            (
                x3,
                image_y
            )
        )


        # ----------------------------------------------------
        # Borders
        # ----------------------------------------------------

        for x in [
            x1,
            x2,
            x3,
        ]:

            draw.rectangle(
                [
                    x,
                    image_y,
                    x + PANEL,
                    image_y + PANEL,
                ],
                outline="black",
                width=1
            )


        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        info_y = (
            image_y
            +
            PANEL
            +
            GAP
        )


        line1 = (
            f"#{i:02d} "
            f"{sample['pose_bin']} "
            f"{sample['sequence']}"
        )


        line2 = (
            f"frame "
            f"{sample['source_frame_index']}"
            f" -> "
            f"{sample['target_frame_index']}"
        )


        draw.text(
            (
                base_x + GAP,
                info_y
            ),
            line1,
            fill="black"
        )


        draw.text(
            (
                base_x + GAP,
                info_y + 20
            ),
            line2,
            fill="black"
        )


    output_path = (
        out_dir /
        f"{args.split}_16_pairs_seed{args.seed}.jpg"
    )


    canvas.save(
        output_path,
        quality=95
    )


    # ========================================================
    # Print
    # ========================================================

    print("=" * 90)
    print("3DPW SINGLE POSE-EDIT DATASET SANITY CHECK")
    print("=" * 90)

    print(
        "split:",
        args.split
    )

    print(
        "samples:",
        len(samples)
    )

    print()

    print(
        manifest[
            [
                "sample_id",
                "sequence",
                "pose_bin",
                "source_frame_index",
                "target_frame_index",
            ]
        ].to_string(
            index=False
        )
    )


    print()

    print(
        "Pose bins:"
    )

    print(
        manifest[
            "pose_bin"
        ].value_counts().to_string()
    )


    print()
    print(
        "Contact sheet:",
        output_path
    )

    print(
        "Manifest:",
        manifest_path
    )

    print()
    print(
        "✅ Dataset sanity check generated"
    )


if __name__ == "__main__":

    main()
