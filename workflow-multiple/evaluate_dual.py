#!/usr/bin/env python3
"""Score a two-person generation: did both bodies land in the requested interaction?

Run with the geometry interpreter, e.g.

    add/envs/smplestx_bw/bin/python evaluate_dual.py --run hug_pair00_084_c4_r000

The requested pose is the pair of ground-truth DWPose-134 sets the geometry stage recorded,
so this measures the generator against the same keypoint convention it was shown, not against
a second estimator's opinion. Person count is checked first: a picture that paints a third or
fourth body has not transferred the interaction, however well the bodies it did draw match.
"""
from __future__ import annotations

import argparse
from itertools import permutations
from pathlib import Path
import sys
import time

import cv2
import math
import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pose_edit.common import (atomic_json, finish_stage, make_request, read_json,
    sha256_file, stage_reasons)
from src.pose_edit.appearance import (crop_person, judge_appearance, judge_import, load_embedder,
    mask_box)
from src.pose_edit import dwpose
from src.pose_edit.geometry import (cocowholebody133_to_dwpose134, valid_points)

ROOT = Path(__file__).resolve().parent
CODE = ['evaluate_dual.py', 'src/pose_edit/common.py', 'src/pose_edit/geometry.py',
    'src/pose_edit/appearance.py', 'src/pose_edit/dwpose.py']
BODY_SLOTS = slice(0, 18)
HIP_SLOTS = (8, 11)
PCK_THRESHOLDS = (0.1, 0.2, 0.5)
AMBIGUOUS_ASSIGNMENT_PX = 40.0


def detect_people(config, image_bgr):
    """Return one entry per detected person: box, 134 points and their detector scores.

    Delegates to ``src.pose_edit.dwpose`` so the 133->134 slot convention lives in exactly one
    place; a second copy is what made the earlier 133-vs-134 bug silent.
    """
    return dwpose.detect_people(config, image_bgr)


def hip_centre(points):
    centre = np.nanmean(points[list(HIP_SLOTS)], axis=0)
    return centre if np.isfinite(centre).all() else None


def torso_length(points):
    hips = hip_centre(points)
    if hips is None or not np.isfinite(points[1]).all():
        return None
    return float(np.linalg.norm(points[1] - hips))


def cleaned(points, scores, threshold, size):
    """Keep only landmarks the detector both believed in and placed inside the frame."""
    keep = (np.isfinite(points).all(axis=1) & (scores >= threshold)
        & valid_points(points, *size))
    return np.where(keep[:, None], points, np.nan), keep


def score_person(requested, person, threshold, size):
    """Compare one requested body with one detected body, normalised by the requested torso."""
    if person is None:
        return {'comparable': False, 'reason': 'no detection assigned to this person'}
    points, keep = cleaned(person['points'], person['scores'], threshold, size)
    shared = np.isfinite(requested).all(axis=1) & keep
    body = shared[BODY_SLOTS]
    torso = torso_length(requested)
    if torso is None or body.sum() == 0:
        return {'comparable': False, 'reason': 'no shared valid body landmark'}
    distance = np.linalg.norm(requested - points, axis=1)
    body_distance = distance[BODY_SLOTS]
    normalised = body_distance[body] / torso
    detected_torso = torso_length(points)
    return {'comparable': True, 'body_joints': int(body.sum()),
        'mpjpe_px': float(body_distance[body].mean()), 'mpjpe_norm': float(normalised.mean()),
        'p95_norm': float(np.percentile(normalised, 95)),
        **{f'pck@{value}': float((normalised <= value).mean()) for value in PCK_THRESHOLDS},
        'worst_joint_px': float(body_distance[body].max()),
        'requested_torso_px': torso, 'detected_torso_px': detected_torso,
        'torso_scale_ratio': (detected_torso / torso) if detected_torso else None,
        'hip_centre_xy': [float(value) for value in hip_centre(points)]
            if hip_centre(points) is not None else None}


def assign(detected_people, requested, threshold, size):
    """Match detections to requested people by hip position, trying both permutations.

    The two people in an embrace can sit a pixel apart, so a near tie is reported as
    ambiguous instead of pretending the assignment was determined.
    """
    targets = [hip_centre(points) for points in requested]
    sources = []
    for person in detected_people:
        points, _ = cleaned(person['points'], person['scores'], threshold, size)
        sources.append(hip_centre(points))
    candidates = []
    for permutation in permutations(range(len(sources)), r=len(requested)):
        if any(sources[index] is None or targets[slot] is None
               for slot, index in enumerate(permutation)):
            continue
        cost = sum(float(np.linalg.norm(sources[index] - targets[slot]))
            for slot, index in enumerate(permutation))
        candidates.append((cost, permutation))
    if not candidates:
        return None, {'assignment': 'unavailable',
            'detail': 'not every person has a usable hip centre'}
    candidates.sort(key=lambda item: item[0])
    best_cost, best = candidates[0]
    margin = candidates[1][0] - best_cost if len(candidates) > 1 else None
    return list(best), {'permutation': [int(index) for index in best],
        'cost_px': best_cost, 'cost_margin_to_next': margin,
        'ambiguous': margin is None or margin < AMBIGUOUS_ASSIGNMENT_PX,
        'basis': 'summed distance between hip centres, both permutations tried'}


def draw_overlay(image, requested, detected_people, mapping):
    """White rings for what was asked for, filled dots for what the generator produced."""
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for points in requested:
        for index in range(18):
            if not np.isfinite(points[index]).all():
                continue
            x, y = points[index]
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(255, 255, 255), width=2)
    for slot, index in enumerate(mapping or []):
        person = detected_people[index]
        for position in person['points'][:18]:
            if not np.isfinite(position).all():
                continue
            x, y = position
            colour = (80, 220, 80) if slot == 0 else (80, 140, 255)
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=colour)
    return overlay


def source_boxes_canvas(pair, preprocess):
    """A's person mask boxes, moved from A's native pixels into the shared canvas."""
    scale_x, scale_y = preprocess['scale_xy']
    offset_x, offset_y = preprocess['offset_xy']
    boxes = []
    for person in pair['A']['persons']:
        box = person['region'].get('bbox_xyxy')
        boxes.append(None if not box else [box[0] * scale_x + offset_x, box[1] * scale_y + offset_y,
            box[2] * scale_x + offset_x, box[3] * scale_y + offset_y])
    return boxes


def target_person_boxes(pair):
    """B's two person boxes, taken from the frame's own semantic masks."""
    arrays = np.load(pair['B']['geometry_original'], allow_pickle=True)
    boxes = []
    for item in pair['B']['slot_order']:
        # The mask keys are spelled with the Hi4D side's case (semantic_A, semantic_B); matching
        # them case-insensitively would silently return no boxes and no error.
        key = next((name for name in (f"semantic_{item['hi4d_side']}",
            f"semantic_{str(item['hi4d_side']).upper()}") if name in arrays.files), None)
        boxes.append(None if key is None else mask_box(arrays[key] > 0))
    return boxes


def crops_of(image, boxes):
    """Crop every box that exists; a missing box yields None and the caller degrades."""
    return [None if box is None else crop_person(image, box) for box in boxes]


def measure_appearance(config, run_dir, geometry, assigned_people, mapping, all_people=None):
    """Did each source person stay that person, and does any body look like B's people?

    ``assigned_people`` is the set aligned with the pose assignment (one per requested person);
    ``all_people`` is every body the whole-image detector found, which is what the import test has
    to look at - an extra body has no assignment by definition.
    """
    if not mapping:
        return {'available': False, 'reason': 'no pose assignment to compare along'}
    all_people = assigned_people if all_people is None else all_people
    pair_dir = Path(geometry['pair_dir'])
    pair = read_json(pair_dir / 'pair.json')
    preprocess = read_json(run_dir / 'inputs' / 'source_preprocess.json')['A']
    with Image.open(run_dir / 'inputs' / 'A_reference.png') as image:
        reference = image.convert('RGB')
    with Image.open(run_dir / 'generation' / 'final_image.png') as image:
        generated = image.convert('RGB')
    source_crops = crops_of(reference, source_boxes_canvas(pair, preprocess))
    generated_crops = [crop_person(generated, assigned_people[slot]['box_xyxy'])
        for slot in mapping]
    if not all(source_crops) or not all(generated_crops):
        return {'available': False, 'reason': 'a person crop came out empty',
            'source_people': len(source_crops), 'generated_people': len(generated_crops)}
    embed, error = load_embedder(config['dinov2_repo'])
    if embed is None:
        return {'available': False, 'reason': error}
    similarity = np.array([[float(np.dot(embed(one), embed(other)))
        for other in generated_crops] for one in source_crops])
    verdict = judge_appearance(similarity, mapping)
    with Image.open(pair_dir / pair['B']['image']) as image:
        target = image.convert('RGB')
    target_crops = crops_of(target, target_person_boxes(pair))
    bodies = crops_of(generated, [person['box_xyxy'] for person in all_people])
    informative = pair['kind'] != 'same_identity'
    if all(target_crops) and all(bodies):
        block = judge_import(embed, source_crops, target_crops, bodies)
    else:
        block = {'available': False, 'bodies': [], 'bodies_liking_target_more': 0,
            'reason': 'a target photograph or generated body crop came out empty'}
    block['informative'] = bool(informative)
    if not informative:
        block['not_informative_reason'] = ('A and B are the same two people in the same clothes '
            'on this flavour of pair, so "looks like B" cannot be told apart from "looks like A"; '
            'the numbers are reported but only cross_pair runs can flag an import')
    verdict['target_import'] = block
    return verdict


def region_boxes(requested, size, expand: float = 0.2):
    """One crop box per requested person: their landmark extent, padded, clamped to the frame."""
    width, height = size
    boxes = []
    for points in requested:
        finite = points[np.isfinite(points).all(axis=1)]
        if not len(finite):
            boxes.append(None)
            continue
        x0, y0 = finite.min(axis=0)
        x1, y1 = finite.max(axis=0)
        pad_x, pad_y = (x1 - x0) * expand, (y1 - y0) * expand
        box = [max(0, int(x0 - pad_x)), max(0, int(y0 - pad_y)),
            min(width, int(x1 + pad_x)), min(height, int(y1 + pad_y))]
        boxes.append(box if box[2] - box[0] >= 8 and box[3] - box[1] >= 8 else None)
    return boxes


def region_count(config, image_bgr, requested, expand=0.2):
    """Ask a different question than whole-image detection: is there a body where each
    requested person should be?

    Measured, not assumed: on the real Hi4D target photograph of an embracing pair, YOLOX returns
    a single box above its own 0.3 cutoff, and loosening NMS only produces near-duplicate boxes
    over the same body (IoU ~0.9). So a whole-image count under-counts entangled pairs no matter
    what the generator did, and it must not be read as the hard gate it was assumed to be. Cropping
    to each requested person's own region lets the detector see the person it would otherwise
    merge away.
    """
    regions = []
    for box in region_boxes(requested, (image_bgr.shape[1], image_bgr.shape[0]), expand):
        if box is None:
            regions.append({'found': None, 'reason': 'no usable landmark extent for this person'})
            continue
        regions.append({'box_xyxy': box,
            'found': len(detect_people(config, image_bgr[box[1]:box[3], box[0]:box[2]]))})
    return {'regions': regions,
        'all_regions_have_a_body': all(region.get('found') for region in regions),
        'basis': 'one DWPose detection pass per requested person region, not one pass over the '
            'whole picture'}


def translate_people(people, offset):
    """Move detections made inside a crop back into full-image coordinates."""
    offset = np.asarray(offset, dtype=float)
    moved = []
    for person in people:
        points = person['points'] + offset
        moved.append({'box_xyxy': [float(value) for value in
            np.asarray(person['box_xyxy'], dtype=float) + np.concatenate([offset, offset])],
            'points': points, 'scores': person['scores']})
    return moved


def region_people(config, image_bgr, requested, threshold, size):
    """One person per requested slot, detected inside that slot's own region.

    Whole-image detection merges bodies that touch, which loses both the count and the landmarks
    for entangled pairs. Cropping to the region the guide asked for recovers a measurable body
    per person. The slot ownership here is positional by construction - the same convention the
    guide numbers its people with - so the assignment is no longer an inference.
    """
    people = []
    for box, points in zip(region_boxes(requested, size), requested):
        if box is None:
            people.append(None)
            continue
        crop = image_bgr[box[1]:box[3], box[0]:box[2]]
        found = translate_people(detect_people(config, crop), (box[0], box[1]))
        if not found:
            people.append(None)
            continue
        # Keep the body with the most believed landmarks inside this region, then only landmarks
        # the detector both believed in and placed inside the frame.
        best = max(found, key=lambda person: int((person['scores'] >= threshold).sum()))
        cleaned_points, keep = cleaned(best['points'], best['scores'], threshold, size)
        people.append({'box_xyxy': best['box_xyxy'], 'points': cleaned_points,
            'scores': np.where(keep, best['scores'], 0.0)})
    return people


def finite(value):
    """Replace NaN/inf with None: an unmeasurable metric is missing evidence, not a number."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    config = read_json(args.config)
    run_dir = (ROOT / 'runs' / args.run).resolve()
    geometry = read_json(run_dir / 'geometry' / 'dual_geometry.json')
    generation = read_json(run_dir / 'generation' / 'generation.json')
    arrays = np.load(run_dir / 'geometry' / 'dual_keypoints.npz')
    final_image = run_dir / 'generation' / 'final_image.png'
    if not final_image.is_file():
        raise FileNotFoundError(f'{final_image} is missing; run the workflow first')
    identity = {'settings': {key: config[key] for key in
            ('dwpose_det_model', 'dwpose_pose_model', 'keypoint_score_threshold',
             'dinov2_repo')},
        'inputs_sha256': {'final_image': sha256_file(final_image),
            'dual_keypoints': sha256_file(run_dir / 'geometry' / 'dual_keypoints.npz'),
            'dual_geometry': sha256_file(run_dir / 'geometry' / 'dual_geometry.json'),
            'generation': sha256_file(run_dir / 'generation' / 'generation.json')},
        'code': {name: sha256_file((REPO_ROOT if name.startswith('src/') else ROOT) / name)
            for name in CODE}}
    request = make_request('evaluate', identity, run_dir,
        {'run_dir': str(run_dir), 'final_image': final_image})
    stage_dir = run_dir / 'evaluation'
    record = stage_dir / 'stage.json'
    if not args.force and not stage_reasons(record, request, run_dir):
        print(f'EVALUATE_DUAL CACHE_HIT {args.run}', flush=True)
        return
    stage_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with Image.open(final_image) as image:
        rgb = np.asarray(image.convert('RGB'))
    size = (rgb.shape[1], rgb.shape[0])
    offset = np.asarray(arrays['content_box'][:2], dtype=float)
    requested = [arrays['canvas_points'][index] - offset
        for index in range(len(arrays['canvas_points']))]
    threshold = config['keypoint_score_threshold']
    detected_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    detected_people = detect_people(config, detected_bgr)
    if len(detected_people) == len(requested):
        pose_basis = 'whole_image'
        scoring_people = detected_people
        mapping, assignment = assign(scoring_people, requested, threshold, size)
    else:
        # The whole-image count is not a usable basis for scoring when the detector merged the
        # pair, so each requested person is detected inside their own region instead.
        pose_basis = 'regions'
        scoring_people = region_people(config, detected_bgr, requested, threshold, size)
        mapping = list(range(len(requested)))
        assignment = {'permutation': mapping, 'ambiguous': False,
            'basis': 'each person detected inside the region the guide asked for; slot ownership '
                'is positional, not inferred from hip distance'}
    persons = [score_person(requested[slot],
        scoring_people[mapping[slot]] if mapping else None, threshold, size)
        for slot in range(len(requested))]
    comparable = [person for person in persons if person.get('comparable')]
    appearance = measure_appearance(config, run_dir, geometry, scoring_people, mapping,
        detected_people)
    regions = region_count(config, detected_bgr, requested)
    summary = {
        'persons_detected': float(len(detected_people)),
        'person_count_ok': float(len(detected_people) == len(requested)),
        'person_count_regions_ok': float(regions['all_regions_have_a_body']),
        'body_pck@0.2': float(np.mean([person['pck@0.2'] for person in comparable]))
            if comparable else None,
        'body_mpjpe_norm': float(np.mean([person['mpjpe_norm'] for person in comparable]))
            if comparable else None,
        'torso_scale_ratio': float(np.mean([person['torso_scale_ratio'] for person in comparable
            if person['torso_scale_ratio']])) if comparable else None,
        'identity_similarity': appearance.get('identity_mean'),
        'identity_swap_suspected': float(appearance['identity_swap_suspected'])
            if appearance.get('available') else None,
        'target_import_suspected': float(
            (appearance.get('target_import') or {}).get('bodies_liking_target_more', 0) > 0)
            if (appearance.get('target_import') or {}).get('available')
            and (appearance.get('target_import') or {}).get('informative') else None}
    interaction = None
    if len(persons) == 2 and all(person.get('hip_centre_xy') for person in persons):
        detected_gap = float(np.linalg.norm(np.asarray(persons[0]['hip_centre_xy'])
            - np.asarray(persons[1]['hip_centre_xy'])))
        requested_gap = float(np.linalg.norm(np.nanmean(requested[0][list(HIP_SLOTS)], axis=0)
            - np.nanmean(requested[1][list(HIP_SLOTS)], axis=0)))
        interaction = {'requested_hip_gap_px': requested_gap,
            'detected_hip_gap_px': detected_gap,
            'error_px': abs(detected_gap - requested_gap),
            'error_relative': abs(detected_gap - requested_gap) / max(requested_gap, 1.0)}
    evaluation = {'schema': 'dual_pose_evaluation_v1', 'run': args.run, 'pair': geometry['pair'],
        'region_count': regions,
        'kind': geometry['kind'], 'framing': geometry['framing'],
        'generator_references': generation['generator_references'],
        'final_image': str(final_image), 'final_size': list(size),
        'requested_slots': [int(value) for value in arrays['slots']],
        'requested_hi4d_sides': [str(value) for value in arrays['hi4d_sides']],
        'detected_boxes': [{'box_xyxy': person['box_xyxy']} for person in detected_people],
        'assignment': assignment, 'pose_basis': pose_basis, 'summary': summary, 'persons': persons,
        'appearance': appearance,
        'interaction': interaction, 'seconds': time.perf_counter() - started,
        'appearance_basis': 'DINOv2 ViT-S/14 cosine similarity on each person crop. It is an '
            'appearance proxy, not biometric identity: a regenerated or occluded face lowers it '
            'without proving a swap. A swap is claimed only when pairing by appearance disagrees '
            'with the hip-based pose pairing beyond the margin.',
        'not_measured': 'Scene and object preservation beyond the two crops, and attribution of '
            'a low appearance score: a changed background inside the box moves the same number '
            'as a changed face would.',
        'limits': 'Pose fidelity is judged with the same DWPose detector that produced the '
            'Hi4D ground truth, so detector bias enters both sides. Landmarks the detector did '
            'not find in the generated image are excluded rather than counted as errors, so a '
            'result that hides a limb can score better than it looks. Person count is reported '
            'separately because extra bodies are the dominant failure when the target photograph '
            'is given to the generator.'}
    evaluation = finite(evaluation)
    summary = finite(evaluation['summary'])
    atomic_json(stage_dir / 'eval_dual.json', evaluation)
    draw_overlay(Image.fromarray(rgb), requested, scoring_people, mapping) \
        .save(stage_dir / 'eval_overlay.png')
    finish_stage(record, request, {'eval_dual': stage_dir / 'eval_dual.json',
        'overlay': stage_dir / 'eval_overlay.png'},
        {'seconds': evaluation['seconds'], 'summary': summary}, run_dir)
    print('EVALUATE_DUAL_COMPLETE', ' '.join(f'{key}={value:.3f}'
        for key, value in summary.items() if isinstance(value, (int, float))),
        'assignment=' + str(assignment.get('permutation', assignment.get('detail'))), flush=True)


if __name__ == '__main__':
    main()
