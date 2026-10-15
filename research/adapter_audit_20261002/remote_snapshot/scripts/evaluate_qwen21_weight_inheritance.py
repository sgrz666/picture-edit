#!/usr/bin/env python3
"""CPU metrics for the fixed inheritance comparison; no image synthesis."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import torch
from eval_iper_four_arm import IdentityMetric, PoseMetric, image_metrics, open_rgb
from src.pose_control.qwen21_full.face_supervision import box_iou
from src.pose_control.qwen21_full.data_runtime import ASSETS
from scripts.run_qwen21_weight_inheritance import OUT, RUN_ROOT, CHECKPOINTS


def cpu_identity_metric():
    # Same models and alignment as IdentityMetric, with bounded CPU threads.
    from insightface import model_zoo
    from insightface.utils import face_align
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    metric = IdentityMetric.__new__(IdentityMetric)
    metric.face_align = face_align
    paths = ROOT / 'models/antelopev2'
    metric.det = model_zoo.get_model(str(paths / 'scrfd_10g_bnkps.onnx'), providers=['CPUExecutionProvider'], sess_options=options)
    metric.recognition = model_zoo.get_model(str(paths / 'glintr100.onnx'), providers=['CPUExecutionProvider'], sess_options=options)
    metric.det.prepare(ctx_id=-1, input_size=(640,640))
    metric.recognition.prepare(ctx_id=-1)
    return metric


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def key(row):
    return (row['set'], row['resolution'], row['candidate'], row['mode'], row['pair_index'])


def face_detection(metric, image, normalized_box):
    """Keep raw-largest and geometry-matched detections, including exclusions."""
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    boxes, landmarks = metric.det.detect(bgr)
    expected = np.asarray(normalized_box, dtype=float) * 512
    detections = []
    for box in boxes:
        coords = box[:4].astype(float)
        center = (coords[:2] + coords[2:]) / 2
        side = float(min(coords[2] - coords[0], coords[3] - coords[1]))
        inside = bool(expected[0] <= center[0] <= expected[2] and expected[1] <= center[1] <= expected[3])
        iou = box_iou(coords, expected)
        reasons = []
        if box[4] < .5: reasons.append('score_below_0.5')
        if side < 32: reasons.append('short_side_below_32_at_512')
        if not inside: reasons.append('center_outside_expected_head')
        if iou < .2: reasons.append('head_iou_below_0.2')
        detections.append({'box': coords.tolist(), 'score': float(box[4]), 'short_side_at_512': side,
                           'head_iou': iou, 'center_in_expected_head': inside,
                           'reliable_detection_proxy': not reasons, 'exclusion_reasons': reasons})
    largest = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))) if len(boxes) else None
    valid = [i for i, item in enumerate(detections) if item['reliable_detection_proxy']]
    selected = max(valid, key=lambda i: detections[i]['score']) if valid else None
    def embedding(index):
        if index is None:
            return None
        crop = metric.face_align.norm_crop(bgr, landmark=landmarks[index])
        value = metric.recognition.get_feat(crop).reshape(-1).astype(np.float32)
        return value / max(float(np.linalg.norm(value)), 1e-12)
    return {'detections': detections, 'raw_largest_index': largest, 'reliable_index': selected,
            'detected': largest is not None, 'reliable': selected is not None,
            'reason': 'accepted' if selected is not None else 'no_detection' if not detections else 'no_reliable_geometric_match'}, embedding(largest), embedding(selected)


def main():
    torch.set_num_threads(4)
    indexed = {key(row): row for row in read_jsonl(OUT / 'generations.jsonl')}
    if len(indexed) != 60 or not json.loads((OUT / 'generation_summary.json').read_text())['passed']:
        raise ValueError('Complete 60-image generation is required')
    pose, identity = PoseMetric(), cpu_identity_metric()
    conditions, features, references = {}, {}, {}
    def feature(appearance):
        if appearance not in features:
            features[appearance] = torch.load(RUN_ROOT / 'face_cache' / f'{appearance}.pt', map_location='cpu', weights_only=False)
        return features[appearance]
    def face_box(app, stem):
        cache = feature(app)
        return cache['face_boxes'][cache['stem_to_idx'][stem]].float().tolist()
    def reference(path, box):
        cache_key = str(path)
        if cache_key not in references:
            references[cache_key] = face_detection(identity, open_rgb(Path(path)), box)
        return references[cache_key]
    rows = []
    for row in sorted(indexed.values(), key=key):
        meta = row['metadata']
        source, target, generated = (open_rgb(Path(path)) for path in (meta['source_rgb'], meta['target_rgb'], row['image']))
        a_box = face_box(meta['source_appearance'], meta['source_stem'])
        b_box = face_box(meta['target_appearance'], meta['target_stem'])
        a, a_raw, a_reliable = reference(meta['source_rgb'], a_box)
        b, b_raw, b_reliable = reference(meta['target_rgb'], b_box)
        g, g_raw, g_reliable = face_detection(identity, generated, b_box)
        app = meta['target_appearance']
        if app not in conditions:
            conditions[app] = torch.load(ASSETS / app / 'v6_conditions.pt', map_location='cpu', weights_only=False)
        target_points = conditions[app]['kps_25_coords'][conditions[app]['stem_to_idx'][meta['target_stem']]].numpy()
        metrics = pose.compare(pose.detect(generated), target_points)
        metrics.update(identity_cosine_raw=identity.cosine(a_raw, g_raw),
                       identity_target_cosine_raw=identity.cosine(b_raw, g_raw),
                       identity_cosine_reliable=identity.cosine(a_reliable, g_reliable),
                       identity_target_cosine_reliable=identity.cosine(b_reliable, g_reliable),
                       source_target_identity_cosine_raw=identity.cosine(a_raw, b_raw),
                       source_target_identity_cosine_reliable=identity.cosine(a_reliable, b_reliable))
        if row['set'] == 'validation':
            from src.pose_control.qwen21_full.data_runtime import read_pairs
            pair = read_pairs(RUN_ROOT / 'val_pairs.jsonl')[row['pair_index']]
            masks = []
            for which, appearance in [('source', meta['source_appearance']), ('target', meta['target_appearance'])]:
                record = pair[which]
                path = ASSETS / appearance / 'mask_512' / record['role'] / f'{record["stem"]}.png'
                masks.append(np.asarray(Image.open(path).convert('L').resize((512, 512), Image.Resampling.NEAREST)) / 255.)
            metrics.update(image_metrics(generated, target, source, masks[1], masks[0]))
        else:
            metrics['b_minus_a_identity_cosine_raw'] = None if a_raw is None or b_raw is None or g_raw is None else float(b_raw @ g_raw - a_raw @ g_raw)
        result = {**row, 'metrics': metrics, 'face_detection': {'source': a, 'target': b, 'generated': g},
                  'face_boxes_normalized': {'source': a_box, 'target': b_box},
                  'common_reliable_identity_sample': False}
        rows.append(result)
        print(json.dumps({'evaluated': len(rows), 'case': key(row), 'pck': metrics.get('pose_pck_002'),
                          'identity_raw': metrics['identity_cosine_raw'], 'reliable_face': g['reliable']}), flush=True)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['set'], row['resolution'], row['pair_index'])].append(row)
    common = {}
    manual_path = OUT / 'visual_review.json'
    manual = json.loads(manual_path.read_text()) if manual_path.exists() else {}
    for group, variants in groups.items():
        if len(variants) != 6:
            raise ValueError('Each case must have all six candidate/mode combinations')
        decision = manual.get(f'{group[0]}_{group[1]}_{group[2]}', {})
        usable = all(row['face_detection']['source']['reliable'] and row['face_detection']['generated']['reliable']
                     and (row['set'] == 'cross_person' or row['face_detection']['target']['reliable']) for row in variants)
        # Visibility/false-positive exclusions are separate from low-quality faces.
        if decision.get('exclude_identity_due_to_visibility_or_misdetection', False):
            usable = False
        common['_'.join(map(str, group))] = {'usable': usable, 'manual_exclusion': decision.get('identity_exclusion_reason'),
            'per_variant': [{'candidate': row['candidate'], 'mode': row['mode'],
                             'source_reason': row['face_detection']['source']['reason'],
                             'target_reason': row['face_detection']['target']['reason'],
                             'generated_reason': row['face_detection']['generated']['reason']} for row in variants]}
        for row in variants:
            row['common_reliable_identity_sample'] = usable
    summary = {}
    metric_names = ('pose_pck_002', 'pose_nme_diagonal', 'psnr', 'ssim', 'human_mae', 'background_mae_source',
                    'identity_cosine_raw', 'identity_target_cosine_raw')
    for set_name, resolution in [('validation', 512), ('validation', 1024), ('cross_person', 1024)]:
        summary_key = f'{set_name}_{resolution}'
        summary[summary_key] = {}
        for candidate in CHECKPOINTS:
            for mode in ('body', 'body_face'):
                selected = [r for r in rows if (r['set'], r['resolution'], r['candidate'], r['mode']) == (set_name, resolution, candidate, mode)]
                report = {'n_images': len(selected), 'pose_detected_n': sum(r['metrics']['pose_detected'] for r in selected),
                          'face_detected_n': sum(r['face_detection']['generated']['detected'] for r in selected),
                          'reliable_face_n': sum(r['face_detection']['generated']['reliable'] for r in selected),
                          'seconds_mean': float(np.mean([r['seconds'] for r in selected])),
                          'peak_cuda_gib': max(r['peak_cuda_gib'] for r in selected)}
                for name in metric_names:
                    values = [r['metrics'].get(name) for r in selected]
                    values = [v for v in values if v is not None and np.isfinite(v)]
                    report[name] = {'mean': float(np.mean(values)) if values else None, 'n': len(values)}
                values = [r['metrics']['identity_cosine_reliable'] for r in selected if r['common_reliable_identity_sample']]
                report['identity_cosine_common'] = {'mean': float(np.mean(values)) if values else None, 'n': len(values)}
                summary[summary_key][f'{candidate}_{mode}'] = report
    (OUT / 'evaluated_samples.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    (OUT / 'metric_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (OUT / 'identity_common_subset.json').write_text(json.dumps(common, ensure_ascii=False, indent=2))
    with (OUT / 'per_sample_metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['set','resolution','candidate','mode','pair_index','image',
                               'common_reliable_identity_sample',*metric_names,'seconds','peak_cuda_gib'])
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k: row[k] for k in ['set','resolution','candidate','mode','pair_index','image',
                                        'common_reliable_identity_sample','seconds','peak_cuda_gib']},
                             **{k: row['metrics'].get(k) for k in metric_names}})
    (OUT / 'metric_definitions.json').write_text(json.dumps({
        'metric_image_resolution': 512, 'resize_filter': 'BILINEAR (existing eval_iper_four_arm.open_rgb)',
        'pose': 'existing PoseMetric / DWPose; PCK@0.02 counts missing joints as failures',
        'pixels': 'existing image_metrics; background compares A outside union of A/B body masks',
        'identity': 'existing AntelopeV2 SCRFD + glintr100; cosine against A',
        'reliable_common': 'all six variants; detector score>=0.5, short side>=32 at 512, center in B head, IoU>=0.2; same checks for A and same-person B',
        'detection_is_visibility_proxy': True, 'manual_review_applied': manual_path.exists(),
        'cross_person_target_rgb_used_as_reconstruction_target': False,
        'small_sample_limitation': '4 validation pairs and 2 cross-person cases; no generalization or statistical significance claim',
        'target_identity_used_only_offline': True}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
