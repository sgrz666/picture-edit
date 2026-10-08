"""Target-RGB face eligibility exclusively for offline auxiliary training losses.

This cache is deliberately separate from FaceConditionStore. It supplies a
boolean loss mask, never embeddings, images, crops, or generation conditions.
SCRFD detection is a visibility proxy, not a guarantee of frontal visibility or
recognition quality. Missing detections and unmatched/undersized faces are
excluded conservatively and retain an explicit reason in the cache.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch

from .data_runtime import pair_metadata, read_pairs


ROOT = Path(__file__).resolve().parents[3]
RUN_ROOT = Path('/home/shangguanrz/nihaochen/qwen21_full')
DEFAULT_CACHE = RUN_ROOT / 'face_supervision.json'
SCHEMA = 'qwen21_full_target_face_supervision_v1'


def target_key(metadata: Mapping[str, Any]) -> str:
    row = metadata if 'target_appearance' in metadata else pair_metadata(metadata)
    return f"{row['target_appearance']}/{row['target_stem']}"


class FaceSupervisionStore:
    """Load-only gate for ID/perceptual training losses; not a generation input."""

    def __init__(self, path: str | Path = DEFAULT_CACHE):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text())
        if self.data.get('schema') != SCHEMA:
            raise ValueError('Unexpected Target face-supervision cache schema')
        self.records = self.data['records']

    def record(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        key = target_key(metadata)
        if key not in self.records:
            raise KeyError(f'Target {key} is missing from the supervision-only cache {self.path}')
        return self.records[key]

    def valid_mask(self, metadata: Sequence[Mapping[str, Any]], resolution: int, device=None, *, align_mode=None) -> torch.Tensor:
        if resolution not in (512, 1024):
            raise ValueError('Face supervision supports the explicit 512/1024 image contracts')
        if align_mode is not None and self.data.get('align_mode', 'resize') != align_mode:
            raise ValueError('Face supervision preprocessing differs from RGB/geometry preprocessing')
        values = []
        for item in metadata:
            row = item if 'source_appearance' in item and 'target_appearance' in item else pair_metadata(item)
            # Cross-person B images are pose examples, never paired identity
            # reconstruction supervision for A.
            cross_person = str(row['source_appearance']).split('_')[0] != str(row['target_appearance']).split('_')[0]
            values.append(not cross_person and bool(self.record(row)['by_resolution'][str(resolution)]['valid']))
        return torch.tensor(values, device=device, dtype=torch.bool)


def box_iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(right - left, 0.) * max(bottom - top, 0.)
    first = max(a[2] - a[0], 0.) * max(a[3] - a[1], 0.)
    second = max(b[2] - b[0], 0.) * max(b[3] - b[1], 0.)
    return intersection / max(first + second - intersection, 1e-12)


def detection_record(box, keypoints, geometry_box, image_hw, *, score_threshold, iou_threshold, align_mode='resize'):
    height, width = image_hw
    x0, y0, x1, y1, score = map(float, box)
    coordinates = [max(0., min(x0, width)), max(0., min(y0, height)),
                   max(0., min(x1, width)), max(0., min(y1, height))]
    face_w, face_h = coordinates[2] - coordinates[0], coordinates[3] - coordinates[1]
    center = ((coordinates[0] + coordinates[2]) / 2, (coordinates[1] + coordinates[3]) / 2)
    inside = geometry_box[0] <= center[0] <= geometry_box[2] and geometry_box[1] <= center[1] <= geometry_box[3]
    iou = box_iou(coordinates, geometry_box)
    reasons = []
    if not np.isfinite([x0, y0, x1, y1, score]).all() or min(face_w, face_h) <= 0:
        reasons.append('invalid_detection_box')
    if score < score_threshold:
        reasons.append('target_detection_score_below_threshold')
    if not inside:
        reasons.append('target_detection_center_outside_geometry_box')
    if iou < iou_threshold:
        reasons.append('target_detection_geometry_iou_below_threshold')
    if align_mode not in ('resize', 'letterbox'):
        raise ValueError('Unsupported Face supervision preprocessing')
    output_boxes, output_points, output_sides = {}, {}, {}
    for resolution in (512, 1024):
        if align_mode == 'letterbox':
            from src.data.letterbox import Letterbox
            transform = Letterbox.from_size((width, height), resolution)
            normalized = torch.tensor(coordinates).reshape(2, 2) / torch.tensor([width, height])
            mapped = transform.points(normalized) * resolution
            output_boxes[str(resolution)] = mapped.reshape(4).tolist()
            output_sides[str(resolution)] = float((mapped[1] - mapped[0]).min())
            if keypoints is not None:
                points = torch.as_tensor(np.asarray(keypoints), dtype=torch.float32) / torch.tensor([width, height])
                output_points[str(resolution)] = (transform.points(points) * resolution).tolist()
        else:
            output_boxes[str(resolution)] = [coordinates[0]*resolution/width, coordinates[1]*resolution/height,
                                             coordinates[2]*resolution/width, coordinates[3]*resolution/height]
            output_sides[str(resolution)] = min(face_w * resolution / width, face_h * resolution / height)
            if keypoints is not None:
                output_points[str(resolution)] = (np.asarray(keypoints) * [resolution/width, resolution/height]).tolist()
    return dict(bbox_original_px=coordinates, bbox_output_px=output_boxes, landmarks5_output_px=output_points, score=score,
                short_side_original_px=min(face_w, face_h),
                short_side_output_px=output_sides,
                geometry_iou=iou, center_in_geometry_box=inside,
                geometric_match=inside and iou >= iou_threshold,
                base_reasons=reasons,
                landmarks5_original_px=None if keypoints is None else np.asarray(keypoints).tolist())


def evaluate_detections(candidates, *, geometry_valid, min_output_short_side=32.):
    result = {}
    for resolution in (512, 1024):
        key = str(resolution)
        accepted = [i for i, detection in enumerate(candidates)
                    if not detection['base_reasons'] and detection['short_side_output_px'][key] >= min_output_short_side]
        if not geometry_valid:
            result[key] = dict(valid=False, reason='invalid_target_geometry', reasons=['invalid_target_geometry'], selected_detection=None)
        elif accepted:
            selected = max(accepted, key=lambda i: candidates[i]['score'])
            result[key] = dict(valid=True, reason='accepted', reasons=[], selected_detection=selected)
        elif not candidates:
            result[key] = dict(valid=False, reason='no_target_face_detected', reasons=['no_target_face_detected'], selected_detection=None)
        else:
            matching = [i for i, item in enumerate(candidates) if not item['base_reasons']]
            selected = max(matching or list(range(len(candidates))), key=lambda i: candidates[i]['score'])
            reasons = list(candidates[selected]['base_reasons'])
            if candidates[selected]['short_side_output_px'][key] < min_output_short_side:
                reasons.append('target_face_too_small_at_output_resolution')
            result[key] = dict(valid=False, reason=reasons[0], reasons=reasons, selected_detection=selected)
    return result


def build_cache(pair_paths, feature_root, detector_path, output, *, score_threshold=.5,
                iou_threshold=.2, min_output_short_side=32., align_mode='resize'):
    """CPU detector only: no Target ArcFace, DINO, or recognition embeddings."""
    import onnxruntime as ort
    from insightface.model_zoo import get_model

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    detector = get_model(str(detector_path), providers=['CPUExecutionProvider'], sess_options=options)
    detector.prepare(ctx_id=-1, input_size=(640, 640), det_thresh=.3)
    if detector.session.get_providers() != ['CPUExecutionProvider']:
        raise RuntimeError('Target supervision detection must run only on CPU')
    rows = []
    pair_hashes = {}
    for path in pair_paths:
        path = Path(path)
        pair_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.extend(pair_metadata(row) for row in read_pairs(path))
    geometry_caches = {}
    records = {}
    for row in rows:
        key = target_key(row)
        if key in records:
            continue
        appearance = row['target_appearance']
        if appearance not in geometry_caches:
            geometry_caches[appearance] = torch.load(Path(feature_root) / f'{appearance}.pt',
                                                     map_location='cpu', weights_only=False)
        cache = geometry_caches[appearance]
        frame = cache['stem_to_idx'][row['target_stem']]
        normalized_box = cache['face_boxes'][frame].float().tolist()
        geometry_valid = bool(cache['geometry_valid'][frame]) and bool(np.isfinite(normalized_box).all())
        with Image.open(row['target_rgb']) as image:
            rgb = np.asarray(image.convert('RGB'))
        height, width = rgb.shape[:2]
        geometry_box = [normalized_box[0] * width, normalized_box[1] * height,
                        normalized_box[2] * width, normalized_box[3] * height]
        geometry_valid = geometry_valid and geometry_box[2] > geometry_box[0] and geometry_box[3] > geometry_box[1]
        boxes, keypoints = detector.detect(np.ascontiguousarray(rgb[..., ::-1]), max_num=0)
        candidates = [detection_record(box, None if keypoints is None else keypoints[i], geometry_box,
                       (height, width), score_threshold=score_threshold, iou_threshold=iou_threshold, align_mode=align_mode)
                      for i, box in enumerate(boxes)]
        eligibility = evaluate_detections(candidates, geometry_valid=geometry_valid,
                                           min_output_short_side=min_output_short_side)
        records[key] = dict(target_appearance=appearance, target_stem=row['target_stem'],
                target_rgb=row['target_rgb'], target_rgb_sha256=hashlib.sha256(Path(row['target_rgb']).read_bytes()).hexdigest(),
                original_width=width, original_height=height,
                geometry_box_normalized=normalized_box, geometry_box_original_px=geometry_box,
                geometry_valid=geometry_valid,
                geometry_cache_fingerprint=cache.get('metadata', {}).get('cache_fingerprint'),
                detected_faces=len(candidates), detections=candidates, by_resolution=eligibility)
        print(f"{key}: " + ', '.join(f"{res}={value['reason']}" for res, value in eligibility.items()), flush=True)
    payload = dict(schema=SCHEMA, align_mode=align_mode, purpose='Target-RGB eligibility for offline ID/perceptual losses only',
            prohibited_uses=['generation conditioning', 'adapter identity or texture features', 'checkpoint selection by final test set'],
            source_validity='caller must also require existing A-source validity / prepared.face.face_valid',
            detector=dict(path=str(detector_path), sha256=hashlib.sha256(Path(detector_path).read_bytes()).hexdigest(),
                          provider='CPUExecutionProvider', input_size=[640, 640], candidate_detection_threshold=.3),
            thresholds=dict(min_score=score_threshold, min_geometry_iou=iou_threshold,
                            detection_center_must_be_inside_geometry_box=True, additional_geometry_expansion=0.,
                            geometry_box_note='Existing SMPL-X face box already includes original 0.15 expansion; no additional expansion.',
                            min_short_side_at_actual_output_px=min_output_short_side),
            visibility_limit='A confident geometrically matched SCRFD detection is a visibility proxy; no manual frontal/occlusion certification is implied.',
            pair_manifest_sha256=pair_hashes, records=records,
            summary={str(res): dict(total=len(records), valid=sum(r['by_resolution'][str(res)]['valid'] for r in records.values()),
                     reasons=dict(Counter(r['by_resolution'][str(res)]['reason'] for r in records.values()))) for res in (512, 1024)})
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    temporary.replace(destination)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', nargs='+', type=Path,
                        default=[RUN_ROOT / 'train_pairs.jsonl', RUN_ROOT / 'val_pairs.jsonl'])
    parser.add_argument('--features', type=Path, default=RUN_ROOT / 'face_cache')
    parser.add_argument('--detector', type=Path, default=ROOT / 'models/antelopev2/scrfd_10g_bnkps.onnx')
    parser.add_argument('--output', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--align-mode', choices=('resize','letterbox'), default='resize')
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = build_cache(args.pairs, args.features, args.detector, args.output, align_mode=args.align_mode)
    print(json.dumps(result['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
