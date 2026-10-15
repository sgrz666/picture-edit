"""Offline same-actor identity-B and 2D head-landmark audit; no generation."""
import copy
import json
import sys
from pathlib import Path

ROOT = Path('/home/shangguanrz/project/pic-edit')
RUN = Path('/home/shangguanrz/nihaochen/qwen21_full/bfs_compare_30_20261004')
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from PIL import Image
from eval_iper_four_arm import IdentityMetric
from scripts.eval.summarize_bfs_comparison import paired_stat

identity = IdentityMetric()
variants = ('body', 'nolora', 'bfs', 'alternative')
records = copy.deepcopy(json.loads((RUN / 'records.json').read_text()))


def detect(image):
    boxes, landmarks = identity.det.detect(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
    if len(boxes) != 1:
        return None
    points = landmarks[0].astype(np.float64)
    eye = points[1]-points[0]
    length = float(np.linalg.norm(eye))
    if length < 1e-6:
        return None
    roll = float(np.degrees(np.arctan2(eye[1], eye[0])))
    shape = (points-points.mean(axis=0))/length
    embedding = identity.embedding(image)
    return roll, shape, embedding


for record in records:
    target = detect(Image.open(record['target_path']).convert('RGB'))
    record['target_face_valid_for_head_audit'] = target is not None
    for variant in variants:
        metric = record['metrics'][variant]
        metric.update({'identity_B_same_actor': None, 'eye_line_roll_error_deg': None, 'normalized_five_point_error': None})
        path = RUN / 'images' / f'{record["stem"]}_{variant}.png'
        if not record['source_face_valid'] or target is None or not path.exists():
            continue
        generated = detect(Image.open(path).convert('RGB'))
        if generated is None:
            continue
        difference = (generated[0]-target[0]+180) % 360-180
        metric['eye_line_roll_error_deg'] = abs(difference)
        metric['normalized_five_point_error'] = float(np.sqrt(np.mean(np.sum((generated[1]-target[1])**2, axis=1))))
        metric['identity_B_same_actor'] = identity.cosine(target[2], generated[2])

common = [r for r in records if r['source_face_valid'] and all(r['metrics'][v]['identity_B_same_actor'] is not None for v in variants)]
summary = {}
for variant in variants:
    summary[variant] = {key: float(np.mean([r['metrics'][variant][key] for r in common])) for key in ('identity_B_same_actor', 'eye_line_roll_error_deg', 'normalized_five_point_error')}
paired = {v: {key: paired_stat(records, v, 'nolora', key) for key in ('identity_B_same_actor', 'eye_line_roll_error_deg', 'normalized_five_point_error')} for v in ('bfs', 'alternative')}
result = {'n_common': len(common), 'n_common_actors': len({r['appearance'].split('_')[0] for r in common}),
          'summary': summary, 'paired_vs_nolora': paired,
          'limitations': ['Target B is the same actor as A in these self-imitation pairs; identity_B is an offline auxiliary metric only',
                          'five-point shape and 2D eye-line roll are coarse proxies, not 3D head pose or expression accuracy',
                          'only unique detected target and generated faces enter this audit; missing cases remain in main records',
                          'no target RGB used for generation; this audit runs after all generation has completed'],
          'per_pair': [{'stem': r['stem'], 'target_face_valid': r['target_face_valid_for_head_audit'],
                        'metrics': {v: {k:r['metrics'][v][k] for k in ('identity_B_same_actor','eye_line_roll_error_deg','normalized_five_point_error')} for v in variants}} for r in records]}
(RUN / 'head_pose_audit.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
print(json.dumps({'n_common': len(common), 'n_actors': result['n_common_actors'], 'summary': summary}, indent=2))
