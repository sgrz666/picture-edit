#!/usr/bin/env python3
"""Turn a two-person pair directory into the pose guides the generator will receive.

Pure CPU, no estimator: the interaction pose comes from Hi4D's per-person DWPose-134 files
(WP7 puts a fitted alternative behind the same interface).

Two framings exist because they answer different questions:

* ``b_canvas``   - the pair keeps B's own layout and is letterboxed onto A's canvas as a
  whole. One uniform scale and one translation act on both people, so the interaction
  geometry (distance, height difference, who leans on whom) survives untouched, and the
  generator decides where in A's frame the pair belongs.
* ``source_fit`` - the same single rigid transform, but its scale and centre are chosen so
  the pair covers A's two-person extent. This keeps A's framing.

Neither framing ever scales the two people independently. That is what broke hard_05 in the
single-person workflow, where a wide target box inside a narrow source box shrank the
requested pose to 41% of the source body.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pose_edit.common import atomic_json, canvas_size, letterbox, read_json, sha256_file
from src.pose_edit.geometry import render_poseguide, valid_points

GUIDE_FRAMINGS = ('b_canvas', 'source_fit')


def load_person(path, score_threshold):
    """Return 134 points in native pixels, with points under the score threshold dropped."""
    with np.load(path) as data:
        xy = data['xy_native'].astype(float)
        scores = data['scores'].astype(float)
    if xy.shape != (134, 2) or scores.shape != (134,):
        raise ValueError(f'{path} is not a DWPose-134 file: {xy.shape}, {scores.shape}')
    valid = np.isfinite(xy).all(axis=1) & (scores >= score_threshold)
    return np.where(valid[:, None], xy, np.nan), scores, valid


def rigid_transform(points, scale, centre_from, centre_to):
    """Scale about ``centre_from`` and move it to ``centre_to``; NaNs stay NaN.

    Applied to both people with the same arguments, this preserves their relative distance,
    which is the part of the pose that carries the interaction.
    """
    centre_from = np.asarray(centre_from, dtype=float)
    centre_to = np.asarray(centre_to, dtype=float)
    out = (points - centre_from) * scale + centre_to
    out[~np.isfinite(points).all(axis=1)] = np.nan
    return out


def measure_pair(points_list):
    """Mean body height and the centre of both people taken together, in the given coordinates."""
    heights, xs, ys = [], [], []
    for points in points_list:
        finite = np.isfinite(points).all(axis=1)
        if finite.sum() < 3:
            continue
        kept = points[finite]
        heights.append(float(kept[:, 1].max() - kept[:, 1].min()))
        xs.append(float(kept[:, 0].mean()))
        ys.append(float(kept[:, 1].mean()))
    if not heights:
        return None
    return {'extent_px': float(np.mean(heights)), 'centre_xy': [float(np.mean(xs)), float(np.mean(ys))],
        'persons_measured': len(heights)}


def measure_source(pair, canvas):
    """A's two-person extent and centre from the person masks, expressed in canvas pixels."""
    scale = min(canvas[0] / pair['A']['size'][0], canvas[1] / pair['A']['size'][1])
    offset = [(canvas[0] - round(pair['A']['size'][0] * scale)) / 2,
        (canvas[1] - round(pair['A']['size'][1] * scale)) / 2]
    heights, centres = [], []
    for person in pair['A']['persons']:
        box, centroid = person['region'].get('bbox_xyxy'), person['region'].get('centroid_xy')
        if not box or not centroid:
            continue
        heights.append(float(box[3] - box[1]) * scale)
        centres.append([centroid[0] * scale + offset[0], centroid[1] * scale + offset[1]])
    if not heights:
        return None, {'scale': scale, 'offset': offset}
    return {'extent_px': float(np.mean(heights)),
        'centre_xy': [float(np.mean([centre[i] for centre in centres])) for i in (0, 1)],
        'persons_measured': len(heights)}, {'scale': scale, 'offset': offset}


def combine_guides(images):
    """Stack per-person guides into one image without inventing links between the two bodies."""
    combined = images[0].copy()
    for image in images[1:]:
        overlay = np.asarray(image, dtype=np.uint16)
        base = np.asarray(combined, dtype=np.uint16)
        merged = np.where(overlay > base, overlay, base).astype(np.uint8)
        combined = Image.fromarray(merged)
    return combined


def build_guides(pair, pair_dir, canvas, framing, score_threshold):
    """Load both people, put them on the canvas, and apply the requested framing."""
    b_size = pair['B']['size']
    b_scale = min(canvas[0] / b_size[0], canvas[1] / b_size[1])
    b_offset = [(canvas[0] - round(b_size[0] * b_scale)) / 2,
        (canvas[1] - round(b_size[1] * b_scale)) / 2]
    persons, meta = [], []
    for slot in sorted(item['slot'] for item in pair['B']['slot_order']):
        entry = next(item for item in pair['B']['slot_order'] if item['slot'] == slot)
        keypoints = next(item for item in pair['B']['keypoints'] if item['slot'] == slot)
        native, scores, valid = load_person(pair_dir / keypoints['file'], score_threshold)
        on_canvas = rigid_transform(native, b_scale, [0.0, 0.0], b_offset)
        on_canvas = np.where(valid_points(on_canvas, *canvas)[:, None], on_canvas, np.nan)
        persons.append(on_canvas)
        meta.append({'slot': slot, 'hi4d_side': entry['hi4d_side'],
            'keypoints_file': keypoints['file'], 'keypoints_sha256': keypoints['sha256'],
            'valid_native': int(valid.sum()), 'valid_canvas': int(np.isfinite(on_canvas).all(axis=1).sum()),
            'min_score': float(scores.min()), 'max_score': float(scores.max())})
    b_pair = measure_pair(persons)
    transform = {'framing': framing, 'b_letterbox_scale': b_scale, 'b_letterbox_offset': b_offset,
        'b_pair_canvas': b_pair}
    if framing == 'source_fit':
        a_pair, a_transform = measure_source(pair, canvas)
        transform['a_pair_canvas'] = a_pair
        transform['a_letterbox'] = a_transform
        if a_pair and b_pair and a_pair['extent_px'] > 0 and b_pair['extent_px'] > 0:
            scale = a_pair['extent_px'] / b_pair['extent_px']
            persons = [rigid_transform(points, scale, b_pair['centre_xy'], a_pair['centre_xy'])
                for points in persons]
            transform.update({'applied_scale': scale, 'applied_centre_from': b_pair['centre_xy'],
                'applied_centre_to': a_pair['centre_xy'],
                'applied_as': 'one uniform scale and one translation shared by both people'})
        else:
            transform.update({'framing_applied': 'b_canvas',
                'reason': 'A exposes no usable person extent, so fitting is impossible'})
    images = [render_poseguide(points, canvas) for points in persons]
    return persons, images, combine_guides(images), transform, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair', type=Path, required=True, help='Directory holding pair.json')
    parser.add_argument('--out', type=Path, required=True, help='Run directory for this pair')
    parser.add_argument('--framing', choices=GUIDE_FRAMINGS, default='b_canvas')
    parser.add_argument('--resolution', type=int, default=768)
    parser.add_argument('--score-threshold', type=float)
    args = parser.parse_args()

    pair_dir = args.pair.resolve()
    pair = read_json(pair_dir / 'pair.json')
    threshold = args.score_threshold or pair['scoring']['score_threshold']
    started = time.perf_counter()
    out = args.out.resolve()
    (out / 'inputs').mkdir(parents=True, exist_ok=True)
    (out / 'geometry').mkdir(parents=True, exist_ok=True)

    with Image.open(pair_dir / pair['A']['image']) as image:
        source = image.convert('RGB')
    canvas = canvas_size(source.size, args.resolution)
    reference, preprocess = letterbox(source, canvas)
    reference.save(out / 'inputs' / 'A_reference.png')
    atomic_json(out / 'inputs' / 'source_preprocess.json', preprocess)

    persons, images, combined, transform, meta = build_guides(
        pair, pair_dir, canvas, args.framing, threshold)
    for entry, image in zip(meta, images):
        image.save(out / 'geometry' / f"skeleton_{entry['slot']}.png")
    combined.save(out / 'geometry' / 'skeleton_both.png')
    with Image.open(pair_dir / pair['B']['image']) as image:
        b_image = image.convert('RGB')
    # Letterbox B exactly like the points, otherwise this QC image lies about alignment.
    b_view, b_letterbox = letterbox(b_image, canvas)
    Image.blend(b_view, combined, 0.45).save(out / 'geometry' / 'B_skeleton_overlay.png')
    Image.blend(reference, combined, 0.45).save(out / 'geometry' / 'A_skeleton_overlay.png')

    record = {'schema': 'dual_geometry_v1', 'pair': pair['name'], 'kind': pair['kind'],
        'framing': args.framing, 'resolution': args.resolution, 'canvas': list(canvas),
        'score_threshold': threshold, 'transform': transform, 'persons': meta,
        'overlay_letterbox': b_letterbox,
        'inputs': {'A_sha256': sha256_file(pair_dir / pair['A']['image']),
            'B_sha256': sha256_file(pair_dir / pair['B']['image']),
            'pair_json_sha256': sha256_file(pair_dir / 'pair.json')},
        'seconds': time.perf_counter() - started,
        'limits': 'Guides carry landmark positions only; joint colours carry no meaning. '
            'Points under the score threshold are absent rather than guessed, so an occluded '
            'person can legitimately show a partial skeleton, and the combined image stacks '
            'the two bodies without drawing any link between them.'}
    atomic_json(out / 'geometry' / 'dual_geometry.json', record)
    print('DUAL_GEOMETRY', args.framing, f'{canvas[0]}x{canvas[1]}',
        ' '.join(f"slot{entry['slot']}:{entry['valid_canvas']}" for entry in meta),
        'transform=' + json_scalar(transform), flush=True)


def json_scalar(transform):
    if 'applied_scale' in transform:
        return f"scale={transform['applied_scale']:.3f}"
    return f"letterbox_scale={transform['b_letterbox_scale']:.3f}"


if __name__ == '__main__':
    main()
