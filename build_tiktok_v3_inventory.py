#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Build TikTok V3 frame inventory

Purpose:
    Scan existing TikTok RGB frames and existing SMPLest-X assets.

No:
    - frame extraction
    - SMPLest-X rerun
    - asset copy

Output:
    TikTok_v3/manifests/frame_inventory.jsonl

"""

from pathlib import Path
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2


ROOT = Path(
    "/home/shangguanrz/project/pic-edit"
)


RGB_ROOT = ROOT / \
    "datasets/TikTokDataset/TikTok_dataset/TikTok_dataset"


ASSET_ROOT = ROOT / \
    "datasets/TikTokDataset/TikTok_3d_assets_native"


OUT_ROOT = ROOT / \
    "datasets/TikTokDataset/TikTok_v3"


OUT_MANIFEST = (
    OUT_ROOT /
    "manifests/frame_inventory.jsonl"
)


OUT_STATS = (
    OUT_ROOT /
    "stats.json"
)


IMG_EXT = {
    ".jpg",
    ".jpeg",
    ".png"
}



def blur_score(img):

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    return float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )



def load_mask_area(mask_path):

    if not mask_path.exists():
        return None

    img = Image.open(mask_path).convert("L")

    arr = np.asarray(img)

    area = (
        arr > 127
    ).mean()

    return float(area)



def get_image_size(path):

    img = Image.open(path)

    return img.size



def find_asset(seq, frame):

    """
    Find existing SMPLest-X assets.

    Different previous scripts used:
        seq/frame
        seq/rgb/frame
        seq/normal/frame

    Here only check existence.
    """

    candidates = {}

    seq_root = (
        ASSET_ROOT /
        seq
    )


    candidates["normal"] = any(
        [
            (seq_root/"normal"/frame.name).exists(),
            (seq_root/"normal"/(frame.stem+".png")).exists()
        ]
    )


    candidates["depth"] = any(
        [
            (seq_root/"depth"/frame.name).exists(),
            (seq_root/"depth"/(frame.stem+".png")).exists()
        ]
    )


    candidates["smplx_mask"] = any(
        [
            (seq_root/"smplx_mask"/frame.name).exists(),
            (seq_root/"smplx_mask"/(frame.stem+".png")).exists()
        ]
    )


    candidates["pose"] = any(
        [
            (seq_root/"pose"/frame.name).exists(),
            (seq_root/"dwp_pose"/frame.name).exists(),
            (seq_root/"dwpose"/frame.name).exists()
        ]
    )


    return candidates



def main():


    OUT_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    frames = []


    sequences = sorted(
        [
            p for p in RGB_ROOT.iterdir()
            if p.is_dir()
        ]
    )


    print("="*80)
    print("TikTok V3 FRAME INVENTORY")
    print("="*80)

    print(
        "sequences:",
        len(sequences)
    )


    total = 0


    with open(
        OUT_MANIFEST,
        "w",
        encoding="utf-8"
    ) as fout:


        for seq_dir in tqdm(
            sequences
        ):

            seq = seq_dir.name


            image_dir = (
                seq_dir /
                "images"
            )


            if not image_dir.exists():
                continue


            images = sorted(
                [
                    p for p in image_dir.iterdir()
                    if p.suffix.lower()
                    in IMG_EXT
                ]
            )


            for frame in images:


                try:

                    img = np.asarray(
                        Image.open(frame)
                        .convert("RGB")
                    )


                    h,w = img.shape[:2]


                    blur = blur_score(
                        img
                    )


                    assets = find_asset(
                        seq,
                        frame
                    )


                    row = {

                        "sequence_id":
                            seq,

                        "frame_name":
                            frame.name,

                        "rgb":
                            str(frame),

                        "width":
                            w,

                        "height":
                            h,


                        "quality":
                        {

                            "blur":
                                blur

                        },


                        "assets":
                            assets

                    }


                    fout.write(
                        json.dumps(
                            row,
                            ensure_ascii=False
                        )
                        + "\n"
                    )


                    total += 1


                except Exception as e:

                    print(
                        "skip",
                        frame,
                        e
                    )



    stats = {

        "version":
            "tiktok_v3_frame_inventory",

        "sequences":
            len(sequences),

        "frames":
            total,

        "manifest":
            str(OUT_MANIFEST)

    }


    with open(
        OUT_STATS,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            stats,
            f,
            indent=2,
            ensure_ascii=False
        )


    print()
    print("="*80)
    print("DONE")
    print("="*80)

    print(
        json.dumps(
            stats,
            indent=2
        )
    )



if __name__ == "__main__":

    main()
