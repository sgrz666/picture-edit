#!/usr/bin/env python3
"""CPU evaluation of the fixed cross-person Qwen native/control comparison.

B supplies a pose, never an RGB reconstruction target. Face detections are
selected in each image independently; low similarity is not an exclusion rule.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
import torch

from eval_iper_four_arm import PoseMetric, COCO_TO_BODY25
from src.data.letterbox import Letterbox
from scripts.eval.qwen21_condition_combinations import (NEW_METHODS, COMBINATION_METHODS,
    COMBINATION_LABELS, factor_edges, require_protocol_outputs)

DEFAULT_ROOT = Path('/home/shangguanrz/nihaochen/qwen21_full/native_value_audit_20261008')
METHODS = ('native_ab', 'native_ab_crop', 'native_skeleton', 'native_geometry',
           'simple_body', 'full_body', 'full_body_face', 'full_body_native_face')
ALL_METHODS = (*METHODS, 'simple_body1024', *NEW_METHODS)
LABELS = {'native_ab': 'Qwen A+B', 'native_ab_crop': 'Qwen A+B+A face',
          'native_skeleton': 'Qwen A+skeleton', 'native_geometry': 'Qwen A+normal+depth',
          'simple_body': 'Simple Body 20k', 'full_body': 'Full Body 6000',
          'full_body_face': 'Full Body+Face 3000', 'full_body_native_face': 'Full Body -> Qwen face',
          'simple_body1024': 'Simple 1024 Body 3k'}
LABELS.update(COMBINATION_LABELS)
METRICS = ('pose_pck_002', 'pose_pck_005', 'pose_nme_diagonal', 'pose_joint_recall',
           'pose_torso_aligned_pck_002', 'pose_torso_aligned_nme', 'pose_joint_angle_mae_degrees',
           'background_mae_source', 'identity_a_cosine_raw', 'identity_b_cosine_raw',
           'identity_a_cosine_reliable', 'identity_b_cosine_reliable',
           'identity_b_minus_a_cosine_raw', 'source_target_identity_cosine_raw')
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rgb(path):
    with Image.open(path) as image:
        return image.convert('RGB')


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def cosine(a, b):
    return float(np.dot(a, b)) if a is not None and b is not None else None


def row_key(row):
    return f'{row["dataset"]}|{row["resolution"]}|{row["steps"]}|{row["pair_id"]}|{row["method"]}'


def group_key(row):
    return f'{row["dataset"]}_{row["resolution"]}_steps{row["steps"]}'


class FaceMetric:
    """Same SCRFD/glintr100 as existing reports, with native-resolution input."""
    def __init__(self):
        from insightface import model_zoo
        from insightface.utils import face_align
        self.face_align = face_align
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        paths = PROJECT / 'models/antelopev2'
        self.det = model_zoo.get_model(str(paths / 'scrfd_10g_bnkps.onnx'),
                                     providers=['CPUExecutionProvider'], sess_options=options)
        self.rec = model_zoo.get_model(str(paths / 'glintr100.onnx'),
                                     providers=['CPUExecutionProvider'], sess_options=options)
        self.det.prepare(ctx_id=-1, input_size=(640, 640))
        self.rec.prepare(ctx_id=-1)

    def detect(self, path):
        image = rgb(path)
        bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        boxes, points = self.det.detect(bgr)
        transform = Letterbox.from_size(image.size, 512)
        records = []
        for index, box in enumerate(boxes):
            xy = np.asarray(box[:4], dtype=np.float64)
            short_native = float(min(xy[2]-xy[0], xy[3]-xy[1]))
            normalized = xy.reshape(2, 2) / np.asarray(image.size)
            mapped = transform.points(torch.tensor(normalized)).numpy().reshape(4)*512
            side_512 = float(min(mapped[2]-mapped[0], mapped[3]-mapped[1]))
            aligned = self.face_align.norm_crop(bgr, landmark=points[index])
            embedding = self.rec.get_feat(aligned).reshape(-1).astype(np.float32)
            embedding /= max(float(np.linalg.norm(embedding)), 1e-12)
            reasons = []
            if float(box[4]) < .5:
                reasons.append('detector_score_below_0.5')
            if side_512 < 32:
                reasons.append('short_side_below_32_at_512')
            if not np.isfinite(embedding).all():
                raise ValueError(f'Non-finite face embedding: {path}')
            records.append({'index': index, 'box_native': xy.tolist(), 'box_at_512': mapped.tolist(),
                            'score': float(box[4]), 'short_side_native': short_native,
                            'short_side_at_512': side_512, 'landmarks5_native': points[index].tolist(),
                            'embedding': embedding.tolist(), 'automatic_exclusion_reasons': reasons})
        largest = max(range(len(records)), key=lambda i: (boxes[i, 2]-boxes[i, 0])*(boxes[i, 3]-boxes[i, 1])) if records else None
        valid = [r['index'] for r in records if not r['automatic_exclusion_reasons']]
        selected = valid[0] if len(valid) == 1 else None
        return {'path': str(path), 'sha256': sha(path), 'image_size': list(image.size),
                'detections': records, 'raw_largest_index': largest,
                'automatic_selected_index': selected, 'detected': bool(records),
                'automatic_reason': 'single_qualified_detection' if selected is not None else
                'no_detection' if not records else 'multiple_qualified_detections' if len(valid)>1 else 'no_qualified_detection'}


def face_selection(record, manual):
    """Manual checks may reject false detections, never low similarity alone."""
    decision = manual.get(record['path'], {})
    reviewed = decision.get('reviewed', False) is True
    index = decision.get('selected_detection_index', record['automatic_selected_index'])
    if index is not None and (not isinstance(index, int) or not 0 <= index < len(record['detections'])):
        raise ValueError(f'Invalid manual face index for {record["path"]}: {index}')
    reasons = [] if index is not None else [record['automatic_reason']]
    if index is not None:
        reasons.extend(record['detections'][index]['automatic_exclusion_reasons'])
    if reviewed:
        if decision.get('is_real_face') is not True:
            reasons.append('manual_false_or_uncertain_face_detection')
        if decision.get('visibility') in ('back', 'occluded', 'uncertain', 'not_visible'):
            reasons.append('manual_identity_not_evaluable_visibility')
        if decision.get('identity_usable') is not True:
            reasons.append(decision.get('exclusion_reason') or 'manual_identity_not_evaluable')
    elif decision:
        reasons.append('manual_review_incomplete')
    reliable = index is not None and not reasons
    return {'selected_index': index, 'reviewed': reviewed, 'reliable': reliable,
            'formal_reliable': reliable and reviewed, 'exclusion_reasons': reasons,
            'is_real_face': decision.get('is_real_face'),
            'manual_face_box_native': decision.get('manual_face_box_native'),
            'visibility': decision.get('visibility', 'pending_review'),
            'head_direction_matches_b': decision.get('head_direction_matches_b', 'pending_review'),
            'visual_notes': decision.get('notes', '')}


def embedding(record, index):
    return record['detections'][index]['embedding'] if index is not None else None


def pose_compare(detected, target, rect):
    if detected is None:
        return {'pose_detected': False, 'pose_pck_002': 0., 'pose_pck_005': 0.,
                'pose_nme_diagonal': None, 'pose_joint_recall': 0., 'pose_matched_joints': 0,
                'pose_torso_aligned_pck_002': None, 'pose_torso_aligned_nme': None,
                'pose_joint_angle_mae_degrees': None}
    points, scores = detected
    truth = target[np.asarray(COCO_TO_BODY25)]
    visible, matched = truth[:, 2] > .3, (truth[:, 2] > .3) & (scores > .3)
    total, count = int(visible.sum()), int(matched.sum())
    x0, y0, x1, y1 = rect
    distance = np.linalg.norm(points[matched]-truth[matched, :2]*512, axis=1)/np.hypot(x1-x0, y1-y0)
    result = {'pose_detected': True, 'pose_pck_002': float((distance < .02).sum()/total) if total else None,
            'pose_pck_005': float((distance < .05).sum()/total) if total else None,
            'pose_nme_diagonal': float(distance.mean()) if count else None,
            'pose_joint_recall': float(count/total) if total else None, 'pose_matched_joints': count,
            'pose_target_visible_joints': total, 'pose_generated_points_at_512': points.tolist(),
            'pose_generated_scores': scores.tolist(), 'pose_target_coco_at_512': (truth[:, :2]*512).tolist()}
    # One translation and one positive scale, using shoulders and hips; no
    # rotation or limb-specific deformation. These are secondary diagnostics.
    torso = np.asarray([5, 6, 11, 12])
    aligned_pck, aligned_nme = None, None
    if bool(matched[torso].all()) and total:
        true_xy = truth[:, :2]*512
        g_center, t_center = points[torso].mean(0), true_xy[torso].mean(0)
        g_length = float(np.linalg.norm(points[torso[:2]].mean(0)-points[torso[2:]].mean(0)))
        t_length = float(np.linalg.norm(true_xy[torso[:2]].mean(0)-true_xy[torso[2:]].mean(0)))
        if g_length > 1e-6 and t_length > 1e-6:
            mapped = (points-g_center)*(t_length/g_length)+t_center
            distances = np.linalg.norm(mapped[matched]-true_xy[matched], axis=1)/np.hypot(x1-x0,y1-y0)
            aligned_pck = float((distances<.02).sum()/total)
            aligned_nme = float(distances.mean()) if count else None
            result['pose_alignment_translation_generated'] = g_center.tolist()
            result['pose_alignment_translation_target'] = t_center.tolist()
            result['pose_alignment_uniform_scale'] = t_length/g_length
    angles = []
    for joint in ((5,7,9),(6,8,10),(11,13,15),(12,14,16)):
        indices = np.asarray(joint)
        if not matched[indices].all():
            continue
        def angle(xy):
            first, center, last = xy[indices]
            u, v = first-center, last-center
            denominator = float(np.linalg.norm(u)*np.linalg.norm(v))
            return float(np.degrees(np.arccos(np.clip(np.dot(u,v)/denominator,-1,1)))) if denominator>1e-6 else None
        a, b = angle(points), angle(truth[:,:2]*512)
        if a is not None and b is not None:
            angles.append(abs(a-b))
    result.update(pose_torso_aligned_pck_002=aligned_pck, pose_torso_aligned_nme=aligned_nme,
                  pose_joint_angle_mae_degrees=float(np.mean(angles)) if angles else None,
                  pose_joint_angle_evaluable_n=len(angles))
    return result


class Scorer:
    def __init__(self, root):
        self.root = root
        self.pose = PoseMetric()
        self.identity = FaceMetric()
        self.face_cache_path = root / 'evaluation/face_detection_cache.json'
        self.face_cache = json.loads(self.face_cache_path.read_text()) if self.face_cache_path.exists() else {}
        self.conditions = {}

    def face(self, path):
        key = str(Path(path))
        if key not in self.face_cache or self.face_cache[key]['sha256'] != sha(path):
            self.face_cache[key] = self.identity.detect(path)
            save(self.face_cache_path, self.face_cache)
        return self.face_cache[key]

    def target_points(self, pair, transform):
        if 'target_pose_points' in pair:
            points = torch.tensor(pair['target_pose_points'], dtype=torch.float32)
        else:
            asset_root = PROJECT / ('datasets/iPER/iper_assets_512_nvdiffrast' if pair['dataset']=='iper'
                                    else 'datasets/TikTokDataset/TikTok_3d_assets_native')
            path = Path(pair.get('target_condition_cache', asset_root/pair['target_appearance']/'v6_conditions.pt'))
            if str(path) not in self.conditions:
                self.conditions[str(path)] = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            bundle = self.conditions[str(path)]
            points = bundle['kps_25_coords'][bundle['stem_to_idx'][pair['target_stem']]].float()
        if points.shape != (25, 3):
            raise ValueError(f'Expected normalized BODY25: {pair["pair_id"]} {tuple(points.shape)}')
        return transform.points(points).numpy()

    def score(self, row, pair):
        a_path = row.get('source_original_rgb', pair['source_rgb'])
        b_path = row.get('target_original_rgb', pair['target_rgb'])
        a, b, g_native = rgb(a_path), rgb(b_path), rgb(row['image'])
        if g_native.size != (row['resolution'], row['resolution']):
            raise ValueError(f'Output resolution differs from metadata: {row["image"]}')
        ta, tb = Letterbox.from_size(a.size, 512), Letterbox.from_size(b.size, 512)
        a_512 = ta.image(a, fill=(128, 128, 128))
        g = g_native.resize((512, 512), Image.Resampling.BILINEAR)
        target = self.target_points(pair, tb)
        values = pose_compare(self.pose.detect(g), target, tb.rect)
        if pair.get('source_person_mask') and pair.get('target_person_mask'):
            ma = np.asarray(ta.image(Image.open(pair['source_person_mask']).convert('L'), nearest=True)) > 127
            mb = np.asarray(tb.image(Image.open(pair['target_person_mask']).convert('L'), nearest=True)) > 127
            valid = ta.valid_mask()[0].numpy().astype(bool) & tb.valid_mask()[0].numpy().astype(bool)
            background = ~ma & ~mb & valid
            error = np.abs(np.asarray(g, dtype=np.float32)-np.asarray(a_512, dtype=np.float32))
            values.update(background_mae_source=float(error[background].mean()) if background.any() else None,
                          background_pixel_count=int(background.sum()),
                          background_valid_fraction=float(background.mean()))
        else:
            values.update(background_mae_source=None, background_pixel_count=0,
                          background_failure_reason='missing_source_or_target_person_mask')
        faces = {'source': self.face(a_path), 'target': self.face(b_path), 'generated': self.face(row['image'])}
        return {**row, 'metrics': values, 'face_detection': faces,
                'metric_source_transform': ta.metadata(), 'metric_target_transform': tb.metadata(),
                'image_sha256': sha(row['image']), 'cross_person': True,
                'target_rgb_used_as_reconstruction_target': False}


def apply_face_review(rows, manual):
    for row in rows:
        faces = row['face_detection']
        selections = {k: face_selection(v, manual) for k, v in faces.items()}
        raw = {k: embedding(v, v['raw_largest_index']) for k, v in faces.items()}
        reliable = {k: embedding(v, selections[k]['selected_index']) if selections[k]['reliable'] else None for k, v in faces.items()}
        values = row['metrics']
        values.update(identity_a_cosine_raw=cosine(raw['source'], raw['generated']),
                      identity_b_cosine_raw=cosine(raw['target'], raw['generated']),
                      source_target_identity_cosine_raw=cosine(raw['source'], raw['target']),
                      identity_a_cosine_reliable=cosine(reliable['source'], reliable['generated']),
                      identity_b_cosine_reliable=cosine(reliable['target'], reliable['generated']))
        av, bv = values['identity_a_cosine_raw'], values['identity_b_cosine_raw']
        values['identity_b_minus_a_cosine_raw'] = bv-av if av is not None and bv is not None else None
        row['face_selection'] = selections
        row['individually_reliable_identity_sample'] = selections['source']['reliable'] and selections['generated']['reliable']
        row['formal_reliable_identity_sample'] = selections['source']['formal_reliable'] and selections['generated']['formal_reliable']
        row['common_reliable_identity_sample'] = False
        row['formal_common_reliable_identity_sample'] = False
    cases = defaultdict(list)
    for row in rows:
        cases[(group_key(row), row['pair_id'])].append(row)
    for variants in cases.values():
        common = all(r['individually_reliable_identity_sample'] for r in variants)
        formal = all(r['formal_reliable_identity_sample'] for r in variants)
        for row in variants:
            row['common_reliable_identity_sample'] = common
            row['formal_common_reliable_identity_sample'] = formal


def stat(values):
    values = [float(v) for v in values if finite(v)]
    return {'mean': float(np.mean(values)) if values else None, 'n': len(values)}


def aggregate(rows):
    result = {'n_images': len(rows), 'pose_detected_n': sum(r['metrics']['pose_detected'] for r in rows),
              'face_detected_n': sum(r['face_detection']['generated']['detected'] for r in rows),
              'face_auto_reliable_n': sum(r['face_detection']['generated']['automatic_selected_index'] is not None for r in rows),
              'face_reliable_n': sum(r['face_selection']['generated']['reliable'] for r in rows),
              'face_formal_reliable_n': sum(r['face_selection']['generated']['formal_reliable'] for r in rows),
              'face_manual_review_complete':all(r['face_selection'][role]['reviewed'] for r in rows for role in ('source','generated')),
              'seconds': stat([r.get('seconds') for r in rows]),
              'peak_cuda_gib': max((r.get('peak_cuda_gib', 0.) for r in rows), default=None)}
    for key in METRICS:
        result[key] = stat([r['metrics'].get(key) for r in rows])
    for key, flag in [('identity_a_cosine_common', 'common_reliable_identity_sample'),
                      ('identity_a_cosine_formal_common', 'formal_common_reliable_identity_sample')]:
        result[key] = stat([r['metrics']['identity_a_cosine_reliable'] for r in rows if r[flag]])
    result['identity_b_cosine_common'] = stat([r['metrics']['identity_b_cosine_reliable'] for r in rows if r['common_reliable_identity_sample']])
    result['identity_b_cosine_formal_common'] = stat([r['metrics']['identity_b_cosine_reliable'] for r in rows if r['formal_common_reliable_identity_sample']])
    return result


def paired_delta(left, right):
    a, b = {r['pair_id']: r for r in left}, {r['pair_id']: r for r in right}
    result = {}
    for metric in ('pose_pck_002', 'pose_torso_aligned_pck_002', 'pose_nme_diagonal',
                   'background_mae_source', 'identity_a_cosine_reliable', 'seconds'):
        changes = []
        for case in sorted(a.keys() & b.keys()):
            x, y = a[case], b[case]
            if metric == 'identity_a_cosine_reliable' and not (x['formal_reliable_identity_sample'] and y['formal_reliable_identity_sample']):
                continue
            xv, yv = (x.get(metric), y.get(metric)) if metric=='seconds' else (x['metrics'].get(metric), y['metrics'].get(metric))
            if finite(xv) and finite(yv):
                changes.append({'pair_id': case, 'first_value':xv, 'second_value':yv,
                                'difference_second_minus_first': yv-xv})
        vals = [r['difference_second_minus_first'] for r in changes]
        result[metric] = {**stat(vals), 'first_mean':stat([r['first_value'] for r in changes])['mean'],
                          'second_mean':stat([r['second_value'] for r in changes])['mean'],
                          'cases': changes, 'improved_n': sum(v>0 if metric in ('pose_pck_002', 'pose_torso_aligned_pck_002', 'identity_a_cosine_reliable') else v<0 for v in vals)}
    return result


def summarize(rows):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[group_key(row)][row['method']].append(row)
    result = {'groups': {}, 'paired_differences': {}, 'sampling_step_differences': {}}
    pairs = [('native_ab', 'native_ab_crop'), ('native_ab', 'native_skeleton'),
             ('native_ab', 'native_geometry'), ('native_ab', 'simple_body'),
             ('native_ab', 'full_body'), ('full_body', 'full_body_face'),
             ('full_body', 'full_body_native_face'), ('full_body_face', 'full_body_native_face')]
    pairs += [('native_skeleton','simple_body'),('native_skeleton','full_body')]
    pairs += [('native_skeleton',m) for m in NEW_METHODS]
    for group, methods in sorted(groups.items()):
        result['groups'][group] = {method: aggregate(records) for method, records in methods.items()}
        result['paired_differences'][group] = {f'{b}_minus_{a}': paired_delta(methods[a], methods[b])
                                              for a, b in pairs if a in methods and b in methods}
    if any(row['method'] in NEW_METHODS for row in rows):
        result['native_combination_groups'] = {}
        result['combination_factor_differences'] = {}
        result['combination_vs_adapters'] = {}
        for group, methods in sorted(groups.items()):
            if not any(m in methods for m in NEW_METHODS):
                continue
            natives=[{**r} for m in COMBINATION_METHODS for r in methods.get(m,[])]
            cases=defaultdict(list)
            for r in natives: cases[r['pair_id']].append(r)
            for variants in cases.values():
                complete={r['method'] for r in variants}==set(COMBINATION_METHODS)
                common=complete and all(r['individually_reliable_identity_sample'] for r in variants)
                formal=complete and all(r['formal_reliable_identity_sample'] for r in variants)
                for r in variants:
                    r['common_reliable_identity_sample']=common
                    r['formal_common_reliable_identity_sample']=formal
            result['native_combination_groups'][group]={m:aggregate([r for r in natives if r['method']==m])
                for m in COMBINATION_METHODS if m in methods}
            result['combination_factor_differences'][group]={f'{b}_minus_{a}':
                {'added_condition':role, 'first_method':a, 'second_method':b,
                 'metrics':paired_delta(methods[a],methods[b])}
                for role,a,b in factor_edges() if a in methods and b in methods}
            result['combination_vs_adapters'][group]={f'{b}_minus_{a}':
                {'native_method':a,'adapter_method':b,'metrics':paired_delta(methods[a],methods[b])}
                for a in COMBINATION_METHODS for b in ('simple_body','full_body','full_body_face','simple_body1024')
                if a in methods and b in methods}
    for dataset in sorted({r['dataset'] for r in rows}):
        key30,key40=f'{dataset}_1024_steps30',f'{dataset}_1024_steps40'
        if key30 not in groups or key40 not in groups:continue
        result['sampling_step_differences'][f'{dataset}_1024']={method:paired_delta(groups[key30][method],groups[key40][method])
            for method in ('native_ab','native_ab_crop') if method in groups[key30] and method in groups[key40]}
    result['formal_identity_review_complete'] = all(r['face_selection'][role]['reviewed'] for r in rows for role in ('source', 'generated'))
    return result


def review_material(root, rows):
    directory = root/'evaluation/face_review_previews'
    directory.mkdir(parents=True, exist_ok=True)
    existing = json.loads((root/'evaluation/face_review.json').read_text()) if (root/'evaluation/face_review.json').exists() else {}
    template, detection_records = {}, {}
    for row in rows:
        for role, record in row['face_detection'].items():
            path = record['path']
            if path in template:
                continue
            template[path] = existing.get(path, {'reviewed': False, 'selected_detection_index': record['automatic_selected_index'],
                'is_real_face': None, 'visibility': 'pending_review', 'identity_usable': None,
                'exclusion_reason': '', 'head_direction_matches_b': 'not_applicable' if role!='generated' else 'pending_review', 'notes': ''})
            image = rgb(path)
            whole = Letterbox.from_size(image.size, 512).image(image, fill=(128, 128, 128))
            draw = ImageDraw.Draw(whole)
            for detection in record['detections']:
                box = detection['box_at_512']
                draw.rectangle(box, outline='red', width=2)
                draw.text((box[0], max(0, box[1]-17)), f'{detection["index"]}: {detection["score"]:.2f}', fill='red')
            canvas = Image.new('RGB', (1024, 570), 'white')
            canvas.paste(whole, (0, 58))
            index = record['automatic_selected_index']
            if index is None:
                index = record['raw_largest_index']
            if index is not None:
                x0, y0, x1, y1 = record['detections'][index]['box_native']
                pad = .25*max(x1-x0, y1-y0)
                box = (max(0,int(x0-pad)), max(0,int(y0-pad)), min(image.width,int(x1+pad)), min(image.height,int(y1+pad)))
                crop = image.crop(box)
                canvas.paste(Letterbox.from_size(crop.size,512).image(crop, fill=(128,128,128)),(512,58))
            label = f'{row["pair_id"]} {role} {row["method"]} {row["resolution"]}/{row["steps"]}'
            ImageDraw.Draw(canvas).text((10,10), label, fill='black', font=ImageFont.truetype(FONT,20))
            destination = directory/f'{record["sha256"][:20]}.jpg'
            canvas.save(destination, quality=94)
            detection_records[path] = {'preview': str(destination), 'role': role, 'automatic_reason': record['automatic_reason'],
                                       'detections': [{k:v for k,v in r.items() if k!='embedding'} for r in record['detections']]}
    save(root/'evaluation/face_review_template.json', template)
    save(root/'evaluation/face_review_index.json', detection_records)
    # Each case is split into two six-column pages, preserving readable faces.
    # This is diagnostic material; unreviewed boxes remain explicitly marked.
    from scripts.reports.build_qwen21_native_value_report import crop_actual_face
    cases = defaultdict(list)
    for row in rows:
        cases[(row['pair_id'],row['resolution'],row['steps'])].append(row)
    contact_index=[]
    for (case,res,steps),variants in sorted(cases.items()):
        ordered=sorted(variants,key=lambda r: ALL_METHODS.index(r['method']))
        for batch in range((len(ordered)+3)//4):
            selected=ordered[batch*4:batch*4+4]
            anchor=selected[0]
            specs=[('A','source',anchor),('B','target',anchor)]+[(LABELS[r['method']],'generated',r) for r in selected]
            canvas=Image.new('RGB',(512*len(specs),1132),'white');draw=ImageDraw.Draw(canvas)
            for col,(label,role,row) in enumerate(specs):
                record=row['face_detection'][role];image=rgb(record['path'])
                whole=Letterbox.from_size(image.size,512).image(image,fill=(128,128,128))
                canvas.paste(whole,(col*512,52));draw.text((col*512+6,8),label,font=ImageFont.truetype(FONT,25),fill='black')
                crop,note,_=crop_actual_face(image,record,row['face_selection'][role])
                canvas.paste(crop,(col*512,616));draw.text((col*512+6,575),note,font=ImageFont.truetype(FONT,24),fill='black')
            path=root/'evaluation/case_review_previews'/f'{case}_{res}_steps{steps}_part{batch+1}.png'
            path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)
            contact_index.append({'pair_id':case,'resolution':res,'steps':steps,'methods':[r['method'] for r in selected],
                                  'path':str(path),'sha256':sha(path)})
    save(root/'evaluation/case_review_index.json',contact_index)


def write_tables(root, rows, summary):
    save(root/'evaluation/evaluated_samples.json', rows)
    save(root/'evaluation/metric_summary.json', summary)
    fields = ['pair_id','dataset','resolution','steps','method','image','image_sha256',
              'seconds','peak_cuda_gib','individually_reliable_identity_sample',
              'common_reliable_identity_sample','formal_common_reliable_identity_sample', *METRICS]
    with (root/'evaluation/per_sample_metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row['metrics'].get(k) if k in METRICS else row.get(k) for k in fields})
    save(root/'evaluation/metric_definitions.json', {
        'pose': 'DWPose COCO17 vs B BODY25; target visibility>0.3, generated confidence>0.3; undetected joints count as PCK failures',
        'pose_image_resolution': 512, 'pose_normalization': 'diagonal of letterboxed B effective image rectangle',
        'pose_limit': 'Cross-person shape/proportion differences can change absolute keypoint position. PCK alone does not establish successful retargeting.',
        'pose_torso_aligned': 'Secondary diagnostic only: translate torso center (mean of both shoulders/hips) and apply one positive uniform scale matching shoulder-midpoint to hip-midpoint distance; no rotation or per-bone deformation; requires all four torso joints detected',
        'pose_joint_angles': 'Secondary diagnostic: 2D elbow/knee angle absolute difference in degrees, only mutually valid triplets; projection/foreshortening still affect this metric',
        'background': '0-255 RGB MAE against A outside union of A/B person masks and inside both valid image rectangles; no generated-person segmentation is used',
        'identity': 'Original-resolution SCRFD / glintr100, five-point alignment from each actual detected face; independent of B head position',
        'face_automatic_rule': 'single detection score>=0.5 and short side>=32 at 512-equivalent letterboxed scale; multiple valid detections require manual selection',
        'face_manual_rule': 'real, sufficiently visible evaluable face; back/occluded/uncertain/false detections excluded with reason, never excluded because similarity is low',
        'common_subset': 'same pair_ids with qualified A and generated faces in every method of that dataset/resolution/steps group; formal common requires human review',
        'raw_faces': 'largest detected face, reported alongside reliable/common results and all failure reasons',
        'target_identity': 'B cosine is offline only, and is not a desired reconstruction score',
        'cross_person_psnr_ssim_human_reconstruction': 'not computed; B is a different person, not an ideal generated RGB target',
        'multi_stage_cost': 'runner total seconds includes first and second stages; same first-stage output reused',
        'formal_identity_review_complete': summary['formal_identity_review_complete']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--pairs', type=Path)
    parser.add_argument('--generations', type=Path)
    parser.add_argument('--protocol', type=Path, help='Supplementary expected output keys; legacy default is unchanged')
    parser.add_argument('--review-only', action='store_true', help='Reaggregate saved scores after manual face review; no model evaluation')
    parser.add_argument('--allow-partial', action='store_true', help='Useful while fixed generation jobs are in progress')
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = args.root
    manual_path = root/'evaluation/face_review.json'
    manual = json.loads(manual_path.read_text()) if manual_path.exists() else {}
    if args.review_only:
        rows = json.loads((root/'evaluation/evaluated_samples.json').read_text())
    else:
        pairs = {p['pair_id']: p for p in read_jsonl(args.pairs or root/'data/pairs.jsonl')}
        generations = read_jsonl(args.generations or root/'generations.jsonl')
        indexed = {}
        for row in generations:
            key = row_key(row)
            if key in indexed and indexed[key] != row:
                raise ValueError(f'Duplicate conflicting generation key: {key}')
            indexed[key] = row
        if not args.allow_partial and args.protocol:
            require_protocol_outputs(args.protocol,list(indexed.values()))
        elif not args.allow_partial:
            for pair in pairs.values():
                present = {r['method'] for r in indexed.values() if r['pair_id']==pair['pair_id'] and r['resolution']==512 and r['steps']==30}
                if present != set(METHODS):
                    raise ValueError(f'Incomplete main comparison: {pair["pair_id"]}, present={present}')
        scored_path = root/'evaluation/evaluated_samples.json'
        old_rows = {row_key(r): r for r in json.loads(scored_path.read_text())} if scored_path.exists() else {}
        scorer = Scorer(root)
        rows = []
        reused = 0
        for index, row in enumerate(sorted(indexed.values(), key=row_key)):
            if row['pair_id'] not in pairs:
                raise ValueError(f'Unexpected/prompt-development pair in evaluation: {row["pair_id"]}')
            pair = pairs[row['pair_id']]
            if row['dataset'] != pair['dataset']:
                raise ValueError(f'Wrong dataset for locked pair: {row_key(row)}')
            for role in ('source', 'target'):
                original = row.get(f'{role}_original_rgb')
                if original is None or Path(original).resolve() != Path(pair[f'{role}_rgb']).resolve():
                    raise ValueError(f'Wrong original {role} image for locked pair: {row_key(row)}')
            fingerprint = hashlib.sha256(json.dumps({'version': 'native_value_metric_v2',
                'pair': pair, 'image_sha256': sha(row['image']),
                'source_sha256': sha(pair['source_rgb']), 'target_sha256': sha(pair['target_rgb']),
                'source_mask_sha256': sha(pair['source_person_mask']) if pair.get('source_person_mask') else None,
                'target_mask_sha256': sha(pair['target_person_mask']) if pair.get('target_person_mask') else None},
                sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            previous = old_rows.get(row_key(row))
            if previous and previous.get('metric_input_fingerprint') == fingerprint:
                value = {**previous, **row}
                reused += 1
            else:
                value = scorer.score(row, pair)
            value['metric_input_fingerprint'] = fingerprint
            rows.append(value)
            print(json.dumps({'evaluated': index+1, 'key': row_key(row), 'pck': rows[-1]['metrics']['pose_pck_002'],
                              'reused_cached_metric': value is not None and previous is not None and previous.get('metric_input_fingerprint')==fingerprint}), flush=True)
            save(root/'evaluation/evaluated_samples_partial.json', rows)
        save(root/'evaluation/cache_reuse.json', {'rows':len(rows),'cached_metrics_reused':reused,
            'new_metric_evaluations':len(rows)-reused,'version':'native_value_metric_v2'})
    apply_face_review(rows, manual)
    if args.protocol and not args.allow_partial:
        require_protocol_outputs(args.protocol,rows)
    summary = summarize(rows)
    write_tables(root, rows, summary)
    review_material(root, rows)
    print(json.dumps({'rows': len(rows), 'groups': list(summary['groups']), 'formal_identity_review_complete': summary['formal_identity_review_complete']}), flush=True)


if __name__ == '__main__':
    main()
