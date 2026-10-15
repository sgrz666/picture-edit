#!/usr/bin/env python3
"""Create paired full-image and face-detail panels for a fixed Qwen21 Face evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
FACE_ROOT = Path('/home/shangguanrz/nihaochen/qwen21_face')
SAMPLED = ROOT / 'datasets/iPER/iper_sampled_6src64tgt'


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=Path, default=FACE_ROOT / 'face_val_pairs.jsonl')
    parser.add_argument('--features', type=Path, default=FACE_ROOT / 'feature_cache')
    parser.add_argument('--image-roots', nargs='+', required=True, type=Path)
    parser.add_argument('--modes', nargs='+', default=['body', 'body_crop_role', 'body_face', 'body_face_crop', 'body_face_gated'])
    parser.add_argument('--output', type=Path, default=FACE_ROOT / 'face_panels')
    return parser.parse_args()


def image_for(roots, stem, mode):
    for root in roots:
        candidate = root / 'images' / f'{stem}_{mode}.png'
        if candidate.exists():
            return candidate
    if mode in ('body_crop_role', 'body_crop_same'):
        return image_for(roots, stem, 'body')
    if mode == 'body_face_crop':
        return image_for(roots, stem, 'body_face')
    raise FileNotFoundError(f'{stem}_{mode}.png')


def portrait(image, box):
    image = image.convert('RGB').resize((512, 512), Image.Resampling.BILINEAR)
    x0, y0, x1, y1 = box
    cx, cy = 256 * (x0 + x1), 256 * (y0 + y1)
    side = max(64, min(384, 512 * max(x1 - x0, y1 - y0) * 1.6))
    face = image.crop((cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2))
    tile = Image.new('RGB', (384, 528), 'white')
    tile.paste(image.resize((384, 384), Image.Resampling.LANCZOS), (0, 16))
    tile.paste(face.resize((128, 128), Image.Resampling.LANCZOS), (128, 400))
    return tile


def main():
    args = arguments()
    pairs = [json.loads(line) for line in args.pairs.read_text().splitlines() if line.strip()]
    args.output.mkdir(parents=True, exist_ok=True)
    cached = {}
    for index, pair in enumerate(pairs):
        app = pair['appearance']
        if app not in cached:
            cached[app] = torch.load(args.features / f'{app}.pt', map_location='cpu', weights_only=False)
        target = pair['target']
        box = cached[app]['target_geometry'][target['stem']]['box']
        source = pair['source']
        source_detection = cached[app]['source'][source['stem']]['detection']
        source_box = ([float(v) / 1024 for v in source_detection['bbox']]
                      if source_detection else box)
        stem = f"{index:03d}_{app}_{source['stem']}_{target['stem']}"
        paths = [("A source", SAMPLED / app / source['rgb_1024']),
                 ("Target (eval only)", SAMPLED / app / target['rgb_1024'])]
        paths.extend((mode, image_for(args.image_roots, stem, mode)) for mode in args.modes)
        sheet = Image.new('RGB', (384 * len(paths), 528), 'white')
        draw = ImageDraw.Draw(sheet)
        for column, (label, path) in enumerate(paths):
            with Image.open(path) as image:
                sheet.paste(portrait(image, source_box if column == 0 else box), (column * 384, 0))
            draw.text((column * 384 + 5, 2), label, fill='black')
        sheet.save(args.output / f'{stem}.jpg', quality=90)
        print(f'{index + 1}/{len(pairs)} {stem}', flush=True)
    (args.output / 'config.json').write_text(json.dumps({'pairs': str(args.pairs),
        'image_roots': [str(path) for path in args.image_roots], 'modes': args.modes}, indent=2))


if __name__ == '__main__':
    main()
