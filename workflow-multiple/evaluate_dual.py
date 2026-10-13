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
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pose_edit.common import (atomic_json, finish_stage, make_request, read_json,
    sha256_file, stage_reasons)
from src.pose_edit.geometry import (cocowholebody133_to_dwpose134, valid_points)
from src.evaluation.dwpose_onnx.onnxdet import inference_detector
from src.evaluation.dwpose_onnx.onnxpose import inference_pose

ROOT = Path(__file__).resolve().parent
CODE = ['evaluate_dual.py', 'src/pose_edit/common.py', 'src/pose_edit/geometry.py']
BODY_SLOTS = slice(0, 18)
HIP_SLOTS = (8, 11)
PCK_THRESHOLDS = (0.1, 0.2, 0.5)
AMBIGUOUS_ASSIGNMENT_PX = 40.0


def detect_people(config, image_bgr):
    """Return one entry per detected person: box, 134 points and their detector scores."""
    det_session = ort.InferenceSession(config['dwpose_det_model'],
        providers=['CPUExecutionProvider'])
    pose_session = ort.InferenceSession(config['dwpose_pose_model'],
        providers=['CPUExecutionProvider'])
    detected = inference_detector(det_session, image_bgr)
    people = []
    if len(detected):
        # This build of inference_detector returns xyxy boxes only; it has already dropped
        # candidates under its own 0.3 threshold and kept class 0.
        points, scores = inference_pose(pose_session, detected.copy(), image_bgr)
        for index in range(len(detected)):
            raw_points = np.asarray(points[index], dtype=float)
            raw_scores = np.asarray(scores[index], dtype=float)
            converted_scores = cocowholebody133_to_dwpose134(raw_scores[:, None])[:, 0]
            converted_scores[1] = float(min(raw_scores[5], raw_scores[6]))
            people.append({'box_xyxy': [float(value) for value in detected[index][:4]],
                'points': cocowholebody133_to_dwpose134(raw_points),
                'scores': converted_scores})
    return people


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
            ('dwpose_det_model', 'dwpose_pose_model', 'keypoint_score_threshold')},
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
    detected_people = detect_people(config, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    mapping, assignment = assign(detected_people[:len(requested)], requested, threshold, size)
    persons = [score_person(requested[slot],
        detected_people[mapping[slot]] if mapping else None, threshold, size)
        for slot in range(len(requested))]
    comparable = [person for person in persons if person.get('comparable')]
    summary = {
        'persons_detected': float(len(detected_people)),
        'person_count_ok': float(len(detected_people) == len(requested)),
        'body_pck@0.2': float(np.mean([person['pck@0.2'] for person in comparable]))
            if comparable else None,
        'body_mpjpe_norm': float(np.mean([person['mpjpe_norm'] for person in comparable]))
            if comparable else None,
        'torso_scale_ratio': float(np.mean([person['torso_scale_ratio'] for person in comparable
            if person['torso_scale_ratio']])) if comparable else None}
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
        'kind': geometry['kind'], 'framing': geometry['framing'],
        'generator_references': generation['generator_references'],
        'final_image': str(final_image), 'final_size': list(size),
        'requested_slots': [int(value) for value in arrays['slots']],
        'requested_hi4d_sides': [str(value) for value in arrays['hi4d_sides']],
        'detected_boxes': [{'box_xyxy': person['box_xyxy']} for person in detected_people],
        'assignment': assignment, 'summary': summary, 'persons': persons,
        'interaction': interaction, 'seconds': time.perf_counter() - started,
        'not_measured': 'identity and clothing preservation. The VGG weights the earlier dual '
            'evaluation used are absent on this host; DINOv2 weights are present, so a feature '
            'similarity metric is the candidate but is not implemented yet.',
        'limits': 'Pose fidelity only, judged with the same DWPose detector that produced the '
            'Hi4D ground truth, so detector bias enters both sides. Landmarks the detector did '
            'not find in the generated image are excluded rather than counted as errors, so a '
            'result that hides a limb can score better than it looks. Person count is reported '
            'separately because extra bodies are the dominant failure when the target photograph '
            'is given to the generator.'}
    atomic_json(stage_dir / 'eval_dual.json', evaluation)
    draw_overlay(Image.fromarray(rgb), requested, detected_people, mapping) \
        .save(stage_dir / 'eval_overlay.png')
    finish_stage(record, request, {'eval_dual': stage_dir / 'eval_dual.json',
        'overlay': stage_dir / 'eval_overlay.png'},
        {'seconds': evaluation['seconds'], 'summary': summary}, run_dir)
    print('EVALUATE_DUAL_COMPLETE', ' '.join(f'{key}={value:.3f}'
        for key, value in summary.items() if isinstance(value, (int, float))),
        'assignment=' + str(assignment.get('permutation', assignment.get('detail'))), flush=True)


if __name__ == '__main__':
    main()
