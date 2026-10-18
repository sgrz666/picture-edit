#!/usr/bin/env python3
"""Make readable iPER comparison sheets with face crops from saved RGB outputs."""

from __future__ import annotations

# Project root for the categorized entrypoint.
from pathlib import Path as _EntryPath
import sys as _entry_sys
_entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))


import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
SAMPLED = ROOT / 'datasets/iPER/iper_sampled_6src64tgt'


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--features', type=Path, default=Path('/home/shangguanrz/nihaochen/qwen21_face/feature_cache'))
    parser.add_argument('--body-output', type=Path, required=True)
    parser.add_argument('--face-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--indices', nargs='*', type=int, default=[])
    return parser.parse_args()


def face_crop(image: Image.Image, box: list[float], normalized: bool) -> Image.Image:
    width, height = image.size
    x0, y0, x1, y1 = box
    if normalized:
        x0, x1 = x0 * width, x1 * width
        y0, y1 = y0 * height, y1 * height
    center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * 1.8
    return image.crop((center_x - side / 2, center_y - side / 2,
                       center_x + side / 2, center_y + side / 2)).resize((512, 512), Image.Resampling.LANCZOS)


def draw_sheet(paths: list[tuple[str, Path, list[float] | None, bool]], output: Path):
    side, header, gap = 512, 44, 36
    sheet = Image.new('RGB', (side * len(paths), header + side * 2 + gap), 'white')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
    for index, (label, path, box, normalized) in enumerate(paths):
        with Image.open(path) as raw:
            image = raw.convert('RGB')
        x = index * side
        draw.text((x + 8, 8), label, fill='black', font=font)
        sheet.paste(image.resize((side, side), Image.Resampling.LANCZOS), (x, header))
        draw.text((x + 8, header + side + 5), 'Face / head crop', fill='black', font=font)
        if box is not None:
            sheet.paste(face_crop(image, box, normalized), (x, header + side + gap))
        else:
            draw.text((x + 64, header + side + gap + 220), 'Face unavailable', fill='black', font=font)

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, optimize=True)


def main():
    args = arguments()
    pairs = [json.loads(line) for line in args.pairs.read_text().splitlines() if line.strip()]
    indices = args.indices or list(range(len(pairs)))
    features = {}
    for index in indices:
        row = pairs[index]
        app = row['appearance']
        if app not in features:
            features[app] = torch.load(args.features / f'{app}.pt', map_location='cpu', weights_only=False)
        source, target = row['source'], row['target']
        stem = f"{index:03d}_{app}_{source['stem']}_{target['stem']}"
        source_face = features[app]['source'][source['stem']]['detection']
        target_box = features[app]['target_geometry'][target['stem']]['box']
        source_box = source_face['bbox'] if source_face else None
        paths = [
            ('A source', SAMPLED / app / source['rgb_1024'], source_box, False),
            ('Target (evaluation only)', SAMPLED / app / target['rgb_1024'], target_box, True),
            ('Body Adapter', args.body_output / 'images' / f'{stem}_on.png', target_box, True),
            ('Body + Face Adapter', args.face_output / 'images' / f'{stem}_body_face.png', target_box, True),
            ('Body + Face + A face', args.face_output / 'images' / f'{stem}_body_face_crop.png', target_box, True),
        ]
        draw_sheet(paths, args.output / f'{stem}.png')
        print(f'{index + 1}/{len(pairs)} {stem}', flush=True)


if __name__ == '__main__':
    main()
