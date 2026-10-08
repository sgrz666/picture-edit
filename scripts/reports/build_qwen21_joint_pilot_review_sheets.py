#!/usr/bin/env python3
"""Render all twelve locked monitor cases in four-case baseline/candidate sheets."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--reference', required=True, help='Directory prefix, e.g. init_initial or BFSBody_hotload')
    parser.add_argument('--candidate', required=True, help='Directory prefix, e.g. J_mid')
    args = parser.parse_args()
    lock = json.loads((args.run / 'data/lock.json').read_text())
    cases = []
    for dataset in ['tiktok', 'iper']:
        folders = [args.run / 'evaluation' / f'{prefix}_{dataset}' for prefix in [args.reference, args.candidate]]
        records = [{row['pair_index']: row for row in map(json.loads, (folder / 'per_sample.jsonl').read_text().splitlines())}
                   for folder in folders]
        for index in lock[f'monitor_{dataset}_indices']:
            if any(index not in rows for rows in records):
                raise ValueError(f'Missing locked monitor case: {dataset}/{index}')
            cases.append((dataset, index, folders, [rows[index] for rows in records]))
    output = args.run / 'quality_reviews'
    output.mkdir(exist_ok=True)
    for start in range(0, len(cases), 4):
        selected = cases[start:start + 4]
        for mode in ['whole', 'face']:
            side, row_height = 256, 300
            canvas = Image.new('RGB', (4 * side, row_height * len(selected)), '#eeeeee')
            draw = ImageDraw.Draw(canvas)
            for position, (dataset, index, folders, rows) in enumerate(selected):
                top = position * row_height
                draw.text((5, top + 5), f'{dataset} {index:02d} | {args.reference} -> {args.candidate}', fill='black')
                keys = ['source', 'target', 'image', 'image'] if mode == 'whole' else ['face_source', 'face_target', 'face_output', 'face_output']
                labels = ['Source A', 'Target B (offline)', args.reference, args.candidate]
                for column, (record, folder, key, label) in enumerate(zip([rows[0], rows[0], rows[0], rows[1]],
                         [folders[0], folders[0], folders[0], folders[1]], keys, labels)):
                    left = column * side
                    draw.text((left + 5, top + 24), label, fill='black')
                    path = record.get(key)
                    if path and (folder / path).is_file():
                        with Image.open(folder / path) as image:
                            tile = ImageOps.contain(image.convert('RGB'), (side, side), Image.Resampling.LANCZOS)
                        canvas.paste(tile, (left + (side - tile.width) // 2, top + 44 + (side - tile.height) // 2))
                    else:
                        draw.text((left + 5, top + 100), record.get('failure') or 'No valid ROI/output', fill='#a02c34')
            canvas.save(output / f'{args.candidate}_{mode}_{start // 4 + 1}.jpg', quality=95)
    (output / f'{args.candidate}_review_scope.json').write_text(json.dumps({
        'reference': args.reference, 'candidate': args.candidate,
        'locked_cases': [{'dataset': dataset, 'index': index} for dataset, index, _, _ in cases],
        'count': len(cases), 'image_policy': 'unaltered saved RGBs; resizing only for comparison sheets'}, indent=2) + '\n')
    print(json.dumps({'reviewed_scope': len(cases), 'sheets': 6, 'output': str(output)}))


if __name__ == '__main__':
    main()
