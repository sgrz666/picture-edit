#!/usr/bin/env python3
"""Turn a two-person pair directory into the pose guides the generator will receive.

Pure CPU: the interaction pose comes either from Hi4D's per-person DWPose-134 files (``dwpose``,
the default) or from a multi-HMR fit measured by WP7 (``fit``). Both go through the same slot
order, letterbox and framing, so a fit-guided run differs from a ground-truth-guided run in the
landmarks only - which is what makes the two attributable.

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
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pose_edit.common import atomic_json, canvas_size, letterbox, read_json, sha256_file
from src.pose_edit.geometry import render_poseguide, valid_points

import dual_fitting

GUIDE_FRAMINGS = ('b_canvas', 'source_fit')
GUIDE_SOURCES = ('dwpose', 'fit')


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
    return {'extent_px': float(np.mean(heights)),
        'centre_xy': [float(np.mean(xs)), float(np.mean(ys))],
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
        combined = Image.fromarray(np.where(overlay > base, overlay, base).astype(np.uint8))
    return combined


def guide_persons(pair, pair_dir, score_threshold, keypoints_source='dwpose', fit_npz=None):
    """The two people of B in native pixels, plus per-person provenance, in guide slot order."""
    slots = sorted(item['slot'] for item in pair['B']['slot_order'])
    if keypoints_source == 'dwpose':
        persons, meta = [], []
        for slot in slots:
            entry = next(item for item in pair['B']['slot_order'] if item['slot'] == slot)
            keypoints = next(item for item in pair['B']['keypoints'] if item['slot'] == slot)
            native, scores, valid = load_person(Path(pair_dir) / keypoints['file'],
                score_threshold)
            persons.append(native)
            meta.append({'slot': slot, 'hi4d_side': entry['hi4d_side'], 'source': 'dwpose',
                'keypoints_file': keypoints['file'], 'keypoints_sha256': keypoints['sha256'],
                'valid_native': int(valid.sum()), 'min_score': float(scores.min()),
                'max_score': float(scores.max())})
        return persons, meta
    if keypoints_source == 'fit':
        if not fit_npz or not Path(fit_npz).is_file():
            raise FileNotFoundError(f'--keypoints-source fit needs the fit npz for this pair; '
                f'got {fit_npz}')
        people, provenance = dual_fitting.load_fit(Path(fit_npz), tuple(pair['B']['size']),
            len(slots))
        meta = []
        for slot, person in zip(slots, people):
            entry = next(item for item in pair['B']['slot_order'] if item['slot'] == slot)
            native = person['points']
            valid = np.isfinite(native).all(axis=1)
            # A fit carries no identity, so the slot is positional: hip x order, the same rule the
            # test set used to number the ground-truth people. Nothing here asserts who is who.
            meta.append({'slot': slot, 'hi4d_side': entry['hi4d_side'], 'source': 'fit',
                'side_basis': 'hip centre x order, identity not asserted',
                'fit_npz': str(fit_npz), 'fit_npz_sha256': sha256_file(fit_npz),
                'fit_index': provenance['slot_order_by_hip_x'][slots.index(slot)],
                'valid_native': int(valid.sum())})
        return [person['points'] for person in people], meta
    raise ValueError(f'unknown keypoint source {keypoints_source!r}; use dwpose or fit')


def build_guides(pair, pair_dir, canvas, framing, score_threshold, keypoints_source='dwpose', fit_npz=None):
    """Load both people, put them on the canvas, and apply the requested framing."""
    b_size = pair['B']['size']
    b_scale = min(canvas[0] / b_size[0], canvas[1] / b_size[1])
    b_offset = [(canvas[0] - round(b_size[0] * b_scale)) / 2,
        (canvas[1] - round(b_size[1] * b_scale)) / 2]
    persons, meta = guide_persons(pair, pair_dir, score_threshold, keypoints_source, fit_npz)
    for index, person in enumerate(meta):
        finite = np.isfinite(persons[index]).all(axis=1)
        person['valid_native'] = int(finite.sum())
        persons[index] = rigid_transform(persons[index], b_scale, [0.0, 0.0], b_offset)
        persons[index] = np.where(valid_points(persons[index], *canvas)[:, None],
            persons[index], np.nan)
        person['valid_canvas'] = int(np.isfinite(persons[index]).all(axis=1).sum())
    b_pair = measure_pair(persons)
    transform = {'framing': framing, 'keypoints_source': keypoints_source,
        'b_letterbox_scale': b_scale, 'b_letterbox_offset': b_offset, 'b_pair_canvas': b_pair}
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


def produce(pair_dir, out_dir, framing='b_canvas', resolution=768, score_threshold=None,
        keypoints_source='dwpose', fit_npz=None):
    """Write every geometry artifact for one run and return the artifact map plus the record."""
    pair_dir, out_dir = Path(pair_dir).resolve(), Path(out_dir).resolve()
    pair = read_json(pair_dir / 'pair.json')
    threshold = score_threshold if score_threshold is not None else pair['scoring']['score_threshold']
    started = time.perf_counter()
    (out_dir / 'inputs').mkdir(parents=True, exist_ok=True)
    (out_dir / 'geometry').mkdir(parents=True, exist_ok=True)

    with Image.open(pair_dir / pair['A']['image']) as image:
        source = ImageOps.exif_transpose(image).convert('RGB')
    canvas = canvas_size(source.size, resolution)
    reference, preprocess = letterbox(source, canvas)
    with Image.open(pair_dir / pair['B']['image']) as image:
        target = ImageOps.exif_transpose(image).convert('RGB')
    # B is letterboxed too, so every reference shares one canvas and nothing is stretched.
    b_reference, b_preprocess = letterbox(target, canvas)
    reference.save(out_dir / 'inputs' / 'A_reference.png')
    b_reference.save(out_dir / 'inputs' / 'B_reference.png')
    atomic_json(out_dir / 'inputs' / 'source_preprocess.json',
        {'A': preprocess, 'B': b_preprocess, 'canvas': list(canvas)})

    persons, images, combined, transform, meta = build_guides(
        pair, pair_dir, canvas, framing, threshold, keypoints_source, fit_npz)
    outputs = {}
    for entry, image in zip(meta, images):
        path = out_dir / 'geometry' / f"skeleton_{entry['slot']}.png"
        image.save(path)
        outputs[f'skeleton_{entry["slot"]}'] = path
    combined_path = out_dir / 'geometry' / 'skeleton_both.png'
    combined.save(combined_path)
    outputs['skeleton_both'] = combined_path
    # The evaluator needs the requested joints as numbers, not as pixels in a drawing.
    keypoints_path = out_dir / 'geometry' / 'dual_keypoints.npz'
    np.savez_compressed(keypoints_path,
        canvas_points=np.stack(persons), canvas_valid=np.isfinite(np.stack(persons)).all(axis=2),
        slots=np.array([entry['slot'] for entry in meta]), hi4d_sides=np.array(
            [entry['hi4d_side'] for entry in meta]),
        canvas=np.array(canvas), content_box=np.array(preprocess['content_box']),
        coordinate_system='canvas pixels, x-right y-down; subtract content_box for final_image')
    outputs['dual_keypoints'] = keypoints_path
    Image.blend(b_reference, combined, 0.45).save(out_dir / 'geometry' / 'B_skeleton_overlay.png')
    Image.blend(reference, combined, 0.45).save(out_dir / 'geometry' / 'A_skeleton_overlay.png')
    outputs['A_skeleton_overlay'] = out_dir / 'geometry' / 'A_skeleton_overlay.png'
    outputs['B_skeleton_overlay'] = out_dir / 'geometry' / 'B_skeleton_overlay.png'
    outputs['A_reference'] = out_dir / 'inputs' / 'A_reference.png'
    outputs['B_reference'] = out_dir / 'inputs' / 'B_reference.png'
    outputs['source_preprocess'] = out_dir / 'inputs' / 'source_preprocess.json'

    record = {'schema': 'dual_geometry_v1', 'pair': pair['name'], 'kind': pair['kind'],
        'pair_dir': str(pair_dir), 'framing': framing, 'keypoints_source': keypoints_source,
        'resolution': resolution,
        'canvas': list(canvas), 'score_threshold': threshold, 'transform': transform,
        'persons': meta,
        'inputs': {'A_sha256': sha256_file(pair_dir / pair['A']['image']),
            'B_sha256': sha256_file(pair_dir / pair['B']['image']),
            'pair_json_sha256': sha256_file(pair_dir / 'pair.json')},
        'seconds': time.perf_counter() - started,
        'limits': 'Guides carry landmark positions only; joint colours carry no meaning. '
            'Points under the score threshold are absent rather than guessed, so an occluded '
            'person can legitimately show a partial skeleton, and the combined image stacks '
            'the two bodies without drawing any link between them.'}
    atomic_json(out_dir / 'geometry' / 'dual_geometry.json', record)
    outputs['dual_geometry'] = out_dir / 'geometry' / 'dual_geometry.json'
    return outputs, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair', type=Path, required=True, help='Directory holding pair.json')
    parser.add_argument('--out', type=Path, required=True, help='Run directory for this pair')
    parser.add_argument('--framing', choices=GUIDE_FRAMINGS, default='b_canvas')
    parser.add_argument('--keypoints-source', choices=GUIDE_SOURCES, default='dwpose',
        help='draw the guides from the pair ground truth or from a multi-HMR fit')
    parser.add_argument('--fit-npz', type=Path,
        help='the fit_dual_worker npz for this pair; required with --keypoints-source fit')
    parser.add_argument('--resolution', type=int, default=768)
    parser.add_argument('--score-threshold', type=float)
    args = parser.parse_args()
    _, record = produce(args.pair, args.out, args.framing, args.resolution, args.score_threshold,
        args.keypoints_source, args.fit_npz)
    transform = record['transform']
    print('DUAL_GEOMETRY', record['framing'], f"{record['canvas'][0]}x{record['canvas'][1]}",
        ' '.join(f"slot{item['slot']}:{item['valid_canvas']}" for item in record['persons']),
        'scale=' + (f"{transform['applied_scale']:.3f}" if 'applied_scale' in transform
            else f"letterbox {transform['b_letterbox_scale']:.3f}"), flush=True)


if __name__ == '__main__':
    main()
