#!/usr/bin/env python3
"""Build a visual ROI sheet for the TikTok single-sample objective evaluation.

Read-only.  Produces one PNG that lets a human verify the premise of every
numeric metric: whether a face actually exists inside the target box.

Usage::

    python scripts/visualize_tiktok_objective_eval.py \
        --output experiments/tiktok_objective_eval/objective_roi_sheet.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

FEATURES = (
    "/home/shangguanrz/project/pic-edit/datasets/TikTokDataset/"
    "TikTok_3d_assets_native/00001/v65_face_features.pt"
)
FRAMES = (
    "/home/shangguanrz/project/pic-edit/datasets/TikTokDataset/"
    "TikTok_dataset/TikTok_dataset/00001/images"
)
PREVIEWS = (
    "/home/shangguanrz/project/pic-edit/experiments/"
    "tiktok_v65_face_overfit_00001/previews"
)
PURE = "/home/shangguanrz/project/pic-edit/experiments/tiktok_v65_pure_inference"

TILE = 256


def open_rgb(path: Path, size: int = 512) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return image if image.size == (size, size) else image.resize((size, size), Image.BILINEAR)


def crop_norm(image: Image.Image, box, pad: float = 0.0) -> Image.Image:
    """Crop a normalised (x0,y0,x1,y1) box, optionally padded."""
    w, h = image.size
    x0, y0, x1, y1 = box
    if pad:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = max(x1 - x0, y1 - y0) / 2 * (1 + pad)
        x0, y0, x1, y1 = cx - half, cy - half, cx + half, cy + half
    return image.crop(
        (
            max(0, int(x0 * w)),
            max(0, int(y0 * h)),
            min(w, int(x1 * w)),
            min(h, int(y1 * h)),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/tmp/objective_roi_sheet.png")
    parser.add_argument("--row-labels", action="store_true")
    args = parser.parse_args()

    features = torch.load(FEATURES, map_location="cpu", weights_only=False)
    boxes = features["face_boxes"].float()
    idx = features["stem_to_idx"]
    source_box = [float(v) for v in boxes[idx["0014"]]]
    target_box = [float(v) for v in boxes[idx["0074"]]]
    boxes_sheet = {k: [float(v) for v in boxes[idx[k]]] for k in ("0014", "0074")}

    source = open_rgb(Path(FRAMES) / "0014.png")
    target = open_rgb(Path(FRAMES) / "0074.png")
    step0_off = open_rgb(Path(PREVIEWS) / "step_000000_face_off.png")
    step0_on = open_rgb(Path(PREVIEWS) / "step_000000_face_on.png")
    step200_off = open_rgb(Path(PREVIEWS) / "step_000200_face_off.png")
    step200_on = open_rgb(Path(PREVIEWS) / "step_000200_face_on.png")
    gen_off = open_rgb(Path(PURE) / "pure_inference_face_off.png")
    gen_on = open_rgb(Path(PURE) / "pure_inference_face_on.png")

    # Row 1: full frames with the target box drawn on.
    def with_box(image: Image.Image, box) -> Image.Image:
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        w, h = canvas.size
        draw.rectangle(
            [box[0] * w, box[1] * h, box[2] * w, box[3] * h], outline=(255, 60, 60), width=3
        )
        return canvas

    row_full = [
        ("source 0014", with_box(source, boxes_sheet["0014"])),
        ("target GT 0074", with_box(target, target_box)),
        ("gen OFF (28 steps)", with_box(gen_off, target_box)),
        ("gen ON (28 steps)", with_box(gen_on, target_box)),
    ]

    # Row 2: ROI crops, padded 30% so the face boundary is visible.
    row_roi = [
        ("src ROI", crop_norm(source, boxes_sheet["0014"], pad=0.3)),
        ("tgt ROI", crop_norm(target, target_box, pad=0.3)),
        ("step0 OFF", crop_norm(step0_off, target_box, pad=0.3)),
        ("step0 ON", crop_norm(step0_on, target_box, pad=0.3)),
    ]
    row_roi2 = [
        ("step200 OFF", crop_norm(step200_off, target_box, pad=0.3)),
        ("step200 ON", crop_norm(step200_on, target_box, pad=0.3)),
        ("gen OFF ROI", crop_norm(gen_off, target_box, pad=0.3)),
        ("gen ON ROI", crop_norm(gen_on, target_box, pad=0.3)),
    ]

    label_h = 26
    cols = 4
    sheet = Image.new("RGB", (cols * TILE, 3 * (TILE + label_h)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16
        )
    except OSError:
        font = ImageFont.load_default()

    for row_index, row in enumerate((row_full, row_roi, row_roi2)):
        y = row_index * (TILE + label_h)
        for col_index, (label, image) in enumerate(row):
            tile = image.resize((TILE, TILE), Image.LANCZOS)
            sheet.paste(tile, (col_index * TILE, y + label_h))
            draw.text(
                (col_index * TILE + 6, y + 5), label, fill=(20, 30, 40), font=font
            )
        draw.line(
            [(0, y + label_h - 1), (cols * TILE, y + label_h - 1)],
            fill=(200, 210, 220),
            width=1,
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"wrote {out}  size={sheet.size}")
    print(f"source box={[round(v,4) for v in boxes_sheet['0014']]}")
    print(f"target box={[round(v,4) for v in target_box]}")


if __name__ == "__main__":
    main()
