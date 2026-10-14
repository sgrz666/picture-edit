#!/usr/bin/env python3
"""Freeze the CPU-audited iPER Body128 / Face64 / validation12 experiment.

This entrypoint does not load DINO or a generative model. Feature extraction is
an explicit second phase through prepare_qwen21_full_data.py.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch

from scripts.precompute_iper_face_features_v65 import compute_face_box, project_smplx_face72_to_rgb_normalized
from src.pose_control.qwen21_full.data_runtime import ASSETS, SAMPLED, pair_metadata, read_pairs
from src.pose_control.qwen21_full.face_supervision import build_cache, detection_record, evaluate_detections

RUN = Path('/home/shangguanrz/nihaochen/qwen21_full')
DEFAULT_OUT = RUN / 'face_multisample_v1/data'
PREFERRED = {
    '001': ['001_12', '001_13'], '008': ['008_1', '008_2'],
    '012': ['012_1', '012_3'], '022': ['022_1', '022_2'],
    '024': ['024_1', '024_3'], '025': ['025_2', '025_3'],
    '026': ['026_2', '026_4'], '027': ['027_1', '027_2'],
}
NEW_VAL = ['012_4', '015_1', '019_3', '024_2']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Unexpected non-JSON data type: {type(value).__name__}')


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                   default=json_scalar) + '\n')
    temporary.replace(path)


def write_rows(path, rows):
    temporary = Path(path).with_suffix('.jsonl.tmp')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    temporary.replace(path)


def pair_key(row):
    meta = pair_metadata(row)
    return (meta['source_appearance'], meta['source_stem'], meta['target_appearance'], meta['target_stem'])


def validate_sets(body, face, val, cross, dev_train, dev_val, final):
    if (len(body), len(face), len(val), len(cross)) != (128, 64, 12, 2):
        raise ValueError('Required exact pair counts are Body128, Face64, val12, cross2')
    keys = {name: {pair_key(row) for row in rows} for name, rows in
            [('body', body), ('face', face), ('val', val), ('cross', cross)]}
    if any(len(keys[name]) != len(rows) for name, rows in [('body', body), ('face', face), ('val', val), ('cross', cross)]):
        raise ValueError('Duplicate pair in a frozen manifest')
    if not keys['face'] <= keys['body']:
        raise ValueError('Face64 must be a subset of Body128')
    if not keys['body'] <= {pair_key(row) for row in dev_train}:
        raise ValueError('Training row was borrowed from outside dev_train')
    if not keys['val'] <= {pair_key(row) for row in dev_val}:
        raise ValueError('Validation row is outside dev_val')
    apps = lambda rows: {pair_metadata(row)[side + '_appearance'] for row in rows for side in ('source', 'target')}
    train_apps, val_apps, final_apps = apps(body), apps(val), apps(final)
    if not apps(cross) <= val_apps:
        raise ValueError('Cross-person inspection appearances must remain in the validation pool')
    if train_apps & val_apps or train_apps & final_apps or val_apps & final_apps:
        raise ValueError('Training, validation or final-test appearances overlap')
    if (keys['body'] | keys['val'] | keys['cross']) & {pair_key(row) for row in final}:
        raise ValueError('Frozen development pairs overlap the final test')
    if {a.split('_')[0] for a in train_apps} != set(PREFERRED) or len(train_apps) != 16:
        raise ValueError('Require the fixed eight actors with two training appearances each')
    if {'006', '009'} & {a.split('_')[0] for a in train_apps}:
        raise ValueError('Actors006/009 are reserved for validation')
    for app in train_apps:
        for rows, count, quota in [(body, 8, 4), (face, 4, 2)]:
            group = [r for r in rows if r['appearance'] == app]
            roles = Counter(r['source']['role'] for r in group)
            if len(group) != count or len({r['target']['stem'] for r in group}) != count:
                raise ValueError(f'{app}: wrong count or repeated B frame')
            if roles != {'source_standard': quota, 'source_motion': quota}:
                raise ValueError(f'{app}: source-role quotas not met')
            if len({r['source']['stem'] for r in group}) < 3:
                raise ValueError(f'{app}: fewer than three distinct A frames')
            if any(abs(r['source']['timestamp_sec'] - r['target']['timestamp_sec']) < 2 for r in group):
                raise ValueError(f'{app}: A/B temporal gap below two seconds')
    if any(pair_metadata(r)['source_appearance'] != pair_metadata(r)['target_appearance'] for r in body + val):
        raise ValueError('Reconstruction training/validation must be within one appearance')
    if any(not pair_metadata(r)['cross_person'] for r in cross):
        raise ValueError('Cross-person inspection pair must have different actors')
    return {'appearance_overlap_train_val': 0, 'appearance_overlap_train_final': 0,
            'appearance_overlap_val_final': 0, 'pair_overlap_final': 0,
            'face_subset_of_body': True, 'actors006_009_unseen_in_training': True}


class Scanner:
    def __init__(self, args):
        import onnxruntime as ort
        from insightface.model_zoo import get_model
        self.args = args
        self.detector_sha = sha(args.detector)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.detector = get_model(str(args.detector), providers=['CPUExecutionProvider'], sess_options=options)
        self.detector.prepare(ctx_id=-1, input_size=(640, 640), det_thresh=.3)
        if self.detector.session.get_providers() != ['CPUExecutionProvider']:
            raise RuntimeError('Selection detector must use CPU only')
        self.app_data = {}
        self.records = {}
        self.started = time.perf_counter()

    def load_app(self, app):
        if app not in self.app_data:
            root = self.args.sampled_root / app
            params = torch.load(root / 'smplestx/params.pt', map_location='cpu', weights_only=False)
            frames = json.loads((root / 'smplestx/frame_map.json').read_text())['frames']
            cond_path = self.args.assets_root / app / 'v6_conditions.pt'
            conditions = torch.load(cond_path, map_location='cpu', weights_only=False)
            self.app_data[app] = {'params': params, 'indices': {f['stem']: i for i, f in enumerate(frames)},
                'projected': project_smplx_face72_to_rgb_normalized(params), 'conditions': conditions,
                'hashes': {str(p): sha(p) for p in [root / 'smplestx/params.pt', root / 'smplestx/frame_map.json', cond_path]}}
        return self.app_data[app]

    def frame(self, app, entry):
        key = f"{app}/{entry['stem']}"
        if key in self.records:
            return self.records[key]
        data = self.load_app(app)
        index = data['indices'][entry['stem']]
        landmarks = data['projected'][index]
        box, geometry_valid = compute_face_box(landmarks)
        params = data['params']
        geometry_valid = bool(geometry_valid and torch.isfinite(params['jaw_pose'][index]).all()
                              and torch.isfinite(params['expression'][index]).all())
        path = self.args.sampled_root / app / entry['rgb_1024']
        with Image.open(path) as image:
            rgb = np.asarray(image.convert('RGB'))
        height, width = rgb.shape[:2]
        actual_box = (box.numpy() * np.array([width, height, width, height])).tolist()
        boxes, points = self.detector.detect(np.ascontiguousarray(rgb[..., ::-1]), max_num=0)
        detections = [detection_record(b, points[i] if points is not None else None, actual_box,
                         (height, width), score_threshold=.5, iou_threshold=.2) for i, b in enumerate(boxes)]
        eligible = evaluate_detections(detections, geometry_valid=geometry_valid, min_output_short_side=32.)
        condition_idx = data['conditions']['stem_to_idx'].get(entry['stem'])
        body_valid = condition_idx is not None
        if body_valid:
            body_valid = bool(torch.isfinite(data['conditions']['smplx_global'][condition_idx]).all()
                              and torch.isfinite(data['conditions']['kps_25_coords'][condition_idx]).all())
        assets = {}
        if entry['role'] == 'target':
            for directory in ('normal_512', 'mask_512', 'part_512', 'dwpose_512'):
                asset = self.args.assets_root / app / directory / entry['role'] / f"{entry['stem']}.png"
                if not asset.is_file():
                    body_valid = False
                else:
                    assets[str(asset)] = sha(asset)
        pose = data['conditions']['kps_25_coords'][condition_idx].float().numpy() if condition_idx is not None else np.zeros((25, 3))
        head = torch.cat([params['global_orient'][index], params['body_pose'][index, 33:36],
                          params['body_pose'][index, 42:45]]).float().tolist()
        record = {'appearance': app, 'stem': entry['stem'], 'role': entry['role'],
            'rgb': str(path), 'rgb_sha256': sha(path), 'original_width': width, 'original_height': height,
            'geometry_valid': geometry_valid, 'body_conditions_valid': body_valid,
            'geometry_box_normalized': box.tolist(), 'detections': detections,
            'by_resolution': eligible, 'pose': pose.tolist(), 'head_rotation_proxy': head,
            'geometry_source_hashes': data['hashes'], 'body_asset_sha256': assets}
        self.records[key] = record
        return record

    def scan_app(self, app, rows):
        entries = {r[side]['stem']: r[side] for r in rows for side in ('source', 'target')}
        for stem in sorted(entries):
            self.frame(app, entries[stem])
        a = sum(self.records[f'{app}/{s}']['by_resolution']['512']['valid'] for s in entries if entries[s]['role'] != 'target')
        b = sum(self.records[f'{app}/{s}']['by_resolution']['512']['valid'] for s in entries if entries[s]['role'] == 'target')
        print(f'CPU scanned {app}: {len(entries)} RGB; reliable A={a}, B={b}', flush=True)
        atomic_json(self.args.out / 'eligibility_inventory_partial.json',
                    {'detector_sha256': self.detector_sha, 'records': self.records})


def pose_distance(a, b):
    a, b = np.array(a), np.array(b)
    valid = (a[:, 2] > .2) & (b[:, 2] > .2)
    if valid.sum() < 5:
        return 0.
    def normalize(pose):
        center = (pose[9, :2] + pose[12, :2]) / 2
        scale = max(float(np.linalg.norm(np.ptp(pose[valid, :2], axis=0))), .1)
        return (pose[:, :2] - center) / scale
    return float(np.sqrt(np.mean(np.sum((normalize(a)[valid] - normalize(b)[valid]) ** 2, axis=-1))))


def annotate(row, scanner, seed):
    app = row['appearance']
    a = scanner.frame(app, row['source'])
    b = scanner.frame(app, row['target'])
    gap = abs(row['source']['timestamp_sec'] - row['target']['timestamp_sec'])
    return {**row, 'selection_metrics': {'temporal_gap_seconds': gap,
        'pose_distance': pose_distance(a['pose'], b['pose']),
        'head_rotation_proxy_distance': float(np.linalg.norm(np.array(a['head_rotation_proxy']) - b['head_rotation_proxy'])),
        'source_reliable_512': a['by_resolution']['512']['valid'],
        'target_reliable_512': b['by_resolution']['512']['valid'],
        'tie_break': hashlib.sha256(f'{seed}:{pair_key(row)}'.encode()).hexdigest()}}


def choose(rows, scanner, chosen, role, slot, *, face, allowed_sources=None):
    used_b = {r['target']['stem'] for r in chosen}
    used_a = {r['source']['stem'] for r in chosen}
    candidates = [r for r in rows if r['source']['role'] == role and r['target']['stem'] not in used_b
                  and r['selection_metrics']['temporal_gap_seconds'] >= 2
                  and r['selection_metrics']['source_reliable_512']
                  and scanner.frame(r['appearance'], r['target'])['body_conditions_valid']
                  and scanner.frame(r['appearance'], r['target'])['geometry_valid']
                  and (not face or r['selection_metrics']['target_reliable_512'])
                  and (allowed_sources is None or r['source']['stem'] in allowed_sources)]
    if role == 'source_motion' and face:
        candidates = [r for r in candidates if r['source']['stem'] not in used_a]
    if not candidates:
        raise RuntimeError(f'No candidate for role={role}, slot={slot}, face={face}')
    times = [r['target']['timestamp_sec'] for r in rows]
    first, duration = min(times), max(max(times) - min(times), 1.)
    if face:
        desired_bin = slot / 4
        time_pool = [r for r in candidates if desired_bin <= (r['target']['timestamp_sec'] - first) / duration <= desired_bin + .25]
        if time_pool:
            candidates = time_pool
    def score(row):
        b = scanner.frame(row['appearance'], row['target'])
        diversity = min((pose_distance(b['pose'], scanner.frame(r['appearance'], r['target'])['pose']) for r in chosen), default=0.)
        temporal = min((abs(row['target']['timestamp_sec'] - r['target']['timestamp_sec']) / duration for r in chosen), default=0.)
        metric = row['selection_metrics']
        change = metric['pose_distance'] + .1 * metric['head_rotation_proxy_distance']
        return (diversity + temporal + (0.15 if face else 0.5) * change,
                metric['tie_break'])
    return max(candidates, key=score)


def select_training(app, rows, scanner, seed):
    scanner.scan_app(app, rows)
    rows = [annotate(r, scanner, seed) for r in rows]
    face = []
    for slot, role in enumerate(['source_standard', 'source_motion', 'source_standard', 'source_motion']):
        face.append(choose(rows, scanner, face, role, slot, face=True))
    body = list(face)
    for slot, role in enumerate(['source_standard', 'source_motion', 'source_standard', 'source_motion']):
        body.append(choose(rows, scanner, body, role, slot, face=False,
                           allowed_sources={r['source']['stem'] for r in face}))
    if len({r['source']['stem'] for r in face}) < 3:
        raise RuntimeError(f'{app}: fewer than three distinct A frames')
    return body, face


def finalize(args):
    """Validate real Full features, then build the supervision-only CPU cache."""
    from scripts.precompute_iper_face_features_v65 import load_v65_face_cache
    out = args.out
    selection = json.loads((out / 'selection_manifest.json').read_text())
    for name, expected in selection['pair_manifest_sha256'].items():
        if sha(out / name) != expected:
            raise ValueError(f'Frozen pair manifest changed: {name}')
    inventory = json.loads((out / 'eligibility_inventory.json').read_text())['records']
    rows = [r for name in ['face64_pairs.jsonl', 'val12_pairs.jsonl', 'cross2_pairs.jsonl'] for r in read_pairs(out / name)]
    required = defaultdict(set)
    source_keys = set()
    for row in rows:
        meta = pair_metadata(row, args.sampled_root)
        for side in ('source', 'target'):
            required[meta[f'{side}_appearance']].add(meta[f'{side}_stem'])
        source_keys.add((meta['source_appearance'], meta['source_stem']))
    fingerprints = {}
    target_only_count = 0
    for app, stems in required.items():
        path = out / 'face_cache' / f'{app}.pt'
        cache = load_v65_face_cache(path)
        if set(cache['frame_stems']) != stems:
            raise ValueError(f'{app}: cache frame coverage differs from the frozen Face manifests')
        for stem in stems:
            i = cache['stem_to_idx'][stem]
            record = inventory[f'{app}/{stem}']
            if not bool(cache['geometry_valid'][i]):
                raise ValueError(f'{app}/{stem}: invalid cached face geometry')
            if not torch.allclose(cache['face_boxes'][i].float(), torch.tensor(record['geometry_box_normalized']), atol=1e-7, rtol=0):
                raise ValueError(f'{app}/{stem}: cached geometry differs from CPU selection')
            if (app, stem) in source_keys:
                if not bool(cache['valid'][i]) or not torch.isfinite(cache['arcface'][i]).all() or not torch.isfinite(cache['dino_patches'][i]).all():
                    raise ValueError(f'{app}/{stem}: missing or nonfinite real A features')
                if not .98 < float(cache['arcface'][i].float().norm()) < 1.02 or float(cache['dino_patches'][i].float().abs().sum()) == 0:
                    raise ValueError(f'{app}/{stem}: invalid ArcFace norm or empty DINO features')
                if cache['metadata']['source_rgb_sha256'][stem] != record['rgb_sha256']:
                    raise ValueError(f'{app}/{stem}: A feature RGB provenance differs')
            else:
                target_only_count += 1
                if bool(cache['valid'][i]) or torch.count_nonzero(cache['arcface'][i]) or torch.count_nonzero(cache['dino_patches'][i]):
                    raise ValueError(f'{app}/{stem}: Target-only identity/texture leakage')
        fingerprints[app] = cache['metadata']['cache_fingerprint']
    supervision = build_cache([out / 'face64_pairs.jsonl', out / 'val12_pairs.jsonl'], out / 'face_cache',
                              args.detector, out / 'face_supervision.json')
    if supervision['summary']['512']['total'] != 76 or supervision['summary']['512']['valid'] != 76:
        raise ValueError('All64 Face training plus12 validation B faces must remain reliable at512')
    for key, record in supervision['records'].items():
        selected = record['by_resolution']['512']['selected_detection']
        if record['detections'][selected]['landmarks5_original_px'] is None:
            raise ValueError(f'{key}: missing real five-point supervision landmarks')
    report = {'status': 'full_face_features_and_supervision_ready', 'source_feature_frames': len(source_keys),
              'all_face_cache_frames': sum(map(len, required.values())), 'target_only_zero_feature_frames': target_only_count,
              'cache_fingerprints': fingerprints, 'supervision': supervision['summary'],
              'face_supervision_sha256': sha(out / 'face_supervision.json'),
              'selection_manifest_sha256': sha(out / 'selection_manifest.json'),
              'target_identity_texture_exact_zero': True, 'native_qwen_cache': 'prepared separately by the root runner',
              'manual_visibility_review': selection.get('manual_visibility_review', 'pending'),
              'manual_revision_sha256': sha(out / 'manual_revision.json') if (out / 'manual_revision.json').exists() else None}
    atomic_json(out / 'readiness.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def revise_manual(args):
    """Apply explicit preparatory human review decisions, preserving all prior data.

    This CPU-only operation reuses A features only after checking their complete
    schema, fingerprints, weights, RGB and raw geometry provenance. B features
    stay zero and the independent B detection/five-point cache is rebuilt.
    """
    from scripts.precompute_iper_face_features_v65 import (
        atomic_save_cache, compute_cache_fingerprint, load_v65_face_cache, sha256_path,
    )
    out = args.out
    if args.manual_replacements is None:
        raise ValueError('--manual-replacements is required for an explicit manual revision')
    decisions = json.loads(args.manual_replacements.read_text())
    if decisions.get('status') != 'root_confirmed_before_training':
        raise ValueError('Manual decisions must explicitly record root confirmation before training')
    if (out / 'manual_revision.json').exists():
        raise FileExistsError('A manual revision is already locked; do not silently revise it again')
    selection = json.loads((out / 'selection_manifest.json').read_text())
    for filename, digest in selection['pair_manifest_sha256'].items():
        if sha(out / filename) != digest:
            raise ValueError(f'Original frozen manifest hash differs: {filename}')
    inventory_payload = json.loads((out / 'eligibility_inventory.json').read_text())
    inventory = inventory_payload['records']
    if sha(args.detector) != inventory_payload['detector_sha256']:
        raise ValueError('Original SCRFD detector hash differs')
    body, old_face, val, cross = [read_pairs(out / name) for name in
        ['body128_pairs.jsonl', 'face64_pairs.jsonl', 'val12_pairs.jsonl', 'cross2_pairs.jsonl']]
    train = read_pairs(args.sampled_root / 'splits/dev_train_pairs.jsonl')
    dev_val = read_pairs(args.sampled_root / 'splits/dev_val_pairs.jsonl')
    final = read_pairs(args.sampled_root / 'splits/baseline_50_eval_pairs.jsonl')
    validate_sets(body, old_face, val, cross, train, dev_val, final)
    original_body = {r['pair_id']: deepcopy(r) for r in body}
    body_by_id = {r['pair_id']: r for r in body}
    face_ids = [r['pair_id'] for r in old_face]
    changes = []
    class InventoryScanner:
        def frame(self, app, entry):
            return inventory[f'{app}/{entry["stem"]}']
    scanner = InventoryScanner()
    used_decisions = set()
    allowed_reasons = {'motion_blur_landmarks_unreliable', 'lower_face_hand_occlusion'}
    for decision in decisions['replacements']:
        old_id = decision['excluded_face_pair_id']
        if old_id in used_decisions or old_id not in face_ids:
            raise ValueError(f'Duplicate or absent excluded Face row: {old_id}')
        used_decisions.add(old_id)
        if decision['reason'] not in allowed_reasons:
            raise ValueError('Unexpected manual exclusion reason')
        old = original_body[old_id]
        source, target = old['source']['stem'], decision['target_stem']
        candidates = [r for r in train if r['appearance'] == old['appearance']
                      and r['source']['stem'] == source and r['target']['stem'] == target
                      and r['source']['role'] == old['source']['role']]
        if len(candidates) != 1:
            raise ValueError(f'{old_id}: replacement is not one exact same-A/source-role dev_train row')
        candidate = annotate(deepcopy(candidates[0]), scanner, selection['seed'])
        for side in ('source', 'target'):
            record = scanner.frame(candidate['appearance'], candidate[side])
            if not record['by_resolution']['512']['valid'] or not record['geometry_valid'] or not record['body_conditions_valid']:
                raise ValueError(f'{old_id}: replacement {side} fails original eligibility')
        existing_id = decision.get('existing_body_pair_id')
        if existing_id:
            replacement = body_by_id[existing_id]
            if pair_key(replacement) != pair_key(candidate) or existing_id in face_ids:
                raise ValueError(f'{old_id}: proposed existing Body replacement differs or is already Face-selected')
            body_updated = False
        else:
            replacement = {**candidate, 'pair_id': old_id}
            body[body.index(body_by_id[old_id])] = replacement
            body_by_id[old_id] = replacement
            body_updated = True
        face_ids[face_ids.index(old_id)] = replacement['pair_id']
        changes.append({'excluded_pair_id': old_id, 'excluded_pair_key': list(pair_key(old)),
            'manual_exclusion_reason': decision['reason'], 'excluded_from': 'Face identity/perceptual supervision',
            'original_body_row_retained': not body_updated, 'replacement_pair_id': replacement['pair_id'],
            'replacement_pair_key': list(pair_key(replacement)), 'same_source_rgb': True,
            'source_role': replacement['source']['role'], 'body_row_replaced': body_updated,
            'replacement_metrics': replacement['selection_metrics']})
    for row in body:
        row['multisample_face_selected'] = row['pair_id'] in face_ids
    face = [body_by_id[pair_id] for pair_id in face_ids]
    assertions = validate_sets(body, face, val, cross, train, dev_val, final)
    if val[:4] != read_pairs(RUN / 'val_pairs.jsonl') or cross != read_pairs(RUN / 'cross_person_pairs.jsonl'):
        raise ValueError('Manual revision changed preserved validation/cross rows')

    def coverage(rows):
        required, sources = defaultdict(set), set()
        for row in rows:
            meta = pair_metadata(row, args.sampled_root)
            for side in ('source', 'target'):
                required[meta[f'{side}_appearance']].add(meta[f'{side}_stem'])
            sources.add((meta['source_appearance'], meta['source_stem']))
        return required, sources
    old_required, old_sources = coverage(old_face + val + cross)
    required, sources = coverage(face + val + cross)
    if sources != old_sources:
        raise ValueError('A set changed: fresh ArcFace/DINO extraction must be explicitly scheduled; CPU revision stops')
    model_hashes = {'arcface_weight_sha256': sha256_path(args.insightface_root),
                    'dino_weight_sha256': sha256_path(args.dino_model)}
    checked = {}
    def verify_plain(path, digest):
        path = str(path)
        if path not in checked:
            checked[path] = sha(path)
        if checked[path] != digest:
            raise ValueError(f'Original data/hash provenance differs: {path}')
    # Verify all selected Body inputs too, including newly selected B assets.
    for row in body + old_face + val + cross:
        meta = pair_metadata(row, args.sampled_root)
        for side in ('source', 'target'):
            record = inventory[f'{meta[f"{side}_appearance"]}/{meta[f"{side}_stem"]}']
            verify_plain(record['rgb'], record['rgb_sha256'])
            for path, digest in {**record['geometry_source_hashes'], **record['body_asset_sha256']}.items():
                verify_plain(path, digest)
    raw_geometry = {}
    def geometry(app, stem):
        if app not in raw_geometry:
            root = args.sampled_root / app / 'smplestx'
            params = torch.load(root / 'params.pt', map_location='cpu', weights_only=False)
            frames = json.loads((root / 'frame_map.json').read_text())['frames']
            raw_geometry[app] = {'params': params, 'indices': {r['stem']: i for i, r in enumerate(frames)},
                'projected': project_smplx_face72_to_rgb_normalized(params),
                'source_hashes': {name: sha256_path(root / name) for name in ('params.pt', 'frame_map.json')}}
        app_data = raw_geometry[app]
        i = app_data['indices'][stem]
        landmarks = app_data['projected'][i].float()
        box, valid = compute_face_box(landmarks)
        jaw, expression = [app_data['params'][field][i].float() for field in ('jaw_pose', 'expression')]
        if not valid or not torch.isfinite(jaw).all() or not torch.isfinite(expression).all():
            raise ValueError(f'{app}/{stem}: invalid fresh geometry')
        record = inventory[f'{app}/{stem}']
        if not torch.allclose(box, torch.tensor(record['geometry_box_normalized']), atol=1e-7, rtol=0):
            raise ValueError(f'{app}/{stem}: original CPU inventory geometry differs')
        return {'face_landmarks': landmarks, 'face_boxes': box, 'jaw_pose': jaw, 'expression': expression}
    old_caches, new_caches, feature_proofs = {}, {}, {}
    for app, stems in old_required.items():
        cache = load_v65_face_cache(out / 'face_cache' / f'{app}.pt')
        if set(cache['frame_stems']) != stems:
            raise ValueError(f'{app}: original cache coverage differs')
        for name, digest in model_hashes.items():
            if cache['metadata'][name] != digest:
                raise ValueError(f'{app}: feature weights differ: {name}')
        for stem in stems:
            fresh = geometry(app, stem)
            i = cache['stem_to_idx'][stem]
            if cache['metadata']['source_hashes'] != raw_geometry[app]['source_hashes']:
                raise ValueError(f'{app}: raw SMPL-X hash differs from feature extraction')
            if not bool(cache['geometry_valid'][i]):
                raise ValueError(f'{app}/{stem}: old cached geometry invalid')
            for field, tensor in fresh.items():
                if not torch.equal(cache[field][i], tensor):
                    raise ValueError(f'{app}/{stem}: raw and cached {field} differ')
            if (app, stem) in sources:
                if not bool(cache['valid'][i]) or cache['metadata']['source_rgb_sha256'][stem] != inventory[f'{app}/{stem}']['rgb_sha256']:
                    raise ValueError(f'{app}/{stem}: existing A features or RGB provenance invalid')
                if not .98 < float(cache['arcface'][i].float().norm()) < 1.02 or not torch.isfinite(cache['dino_patches'][i]).all() or not torch.count_nonzero(cache['dino_patches'][i]):
                    raise ValueError(f'{app}/{stem}: invalid real A features')
                feature_proofs[f'{app}/{stem}'] = {name: hashlib.sha256(cache[name][i].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                    for name in ('arcface', 'dino_patches')}
        old_caches[app] = cache
    backup = out.parent / 'pre_manual_review'
    if backup.exists():
        raise FileExistsError(f'Preserve the existing backup; refusing overwrite: {backup}')
    backup.mkdir()
    shutil.copytree(out, backup / 'data')
    log = out.parent / 'logs/features.log'
    if log.exists():
        (backup / 'logs').mkdir()
        shutil.copy2(log, backup / 'logs/features.log')
    for name in ['prepare_qwen21_full_data.py', 'prepare_qwen21_face_multisample.py']:
        (backup / 'scripts').mkdir(exist_ok=True)
        shutil.copy2(ROOT / 'scripts/data' / name, backup / 'scripts' / name)
    backup_hashes = {str(path.relative_to(backup)): sha(path) for path in sorted(backup.rglob('*')) if path.is_file()}
    for relative, digest in backup_hashes.items():
        if relative.startswith('data/') and digest != sha(out / relative.removeprefix('data/')):
            raise ValueError(f'Pre-manual backup copy differs: {relative}')
    atomic_json(backup / 'sha256_manifest.json', {'files': backup_hashes,
        'selection_manifest_sha256': sha(out / 'selection_manifest.json'),
        'cache_fingerprints': {app: c['metadata']['cache_fingerprint'] for app, c in old_caches.items()},
        'backup_verified': True})
    for app, stems in required.items():
        old = old_caches[app]
        if stems == old_required[app]:
            new_caches[app] = old
            continue
        stems = sorted(stems)
        fresh = [geometry(app, stem) for stem in stems]
        metadata = deepcopy(old['metadata'])
        metadata['geometry_valid'] = [True] * len(stems)
        metadata['manual_revision'] = {'phase': 'pre_training_final_lock',
            'source_features_reused_after_hash_geometry_weight_checks': True,
            'original_cache_fingerprint': old['metadata']['cache_fingerprint'],
            'original_cache_file_sha256': sha(backup / 'data/face_cache' / f'{app}.pt'),
            'target_geometry_recomputed_from_raw_smplx': True}
        cache = {'metadata': metadata, 'frame_stems': stems, 'stem_to_idx': {stem: i for i, stem in enumerate(stems)},
            'arcface': torch.zeros(len(stems), 512, dtype=torch.float16),
            'dino_patches': torch.zeros(len(stems), 256, 1536, dtype=torch.float16),
            'valid': torch.zeros(len(stems), dtype=torch.bool), 'geometry_valid': torch.ones(len(stems), dtype=torch.bool)}
        for field in ('face_landmarks', 'jaw_pose', 'expression', 'face_boxes'):
            cache[field] = torch.stack([record[field] for record in fresh])
        for i, stem in enumerate(stems):
            if (app, stem) in sources:
                old_i = old['stem_to_idx'][stem]
                cache['arcface'][i] = old['arcface'][old_i]
                cache['dino_patches'][i] = old['dino_patches'][old_i]
                cache['valid'][i] = True
                for name in ('arcface', 'dino_patches'):
                    digest = hashlib.sha256(cache[name][i].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                    if digest != feature_proofs[f'{app}/{stem}'][name]:
                        raise ValueError(f'{app}/{stem}: A feature copy is not byte-exact')
        metadata['cache_fingerprint'] = compute_cache_fingerprint(cache)
        new_caches[app] = cache
    revision = {'schema': 'qwen21_full_manual_revision_v1', 'status': 'root_confirmed_before_training',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'backup': str(backup),
        'pre_manual_review_sha256_manifest': sha(backup / 'sha256_manifest.json'),
        'confirmed_decisions_sha256': sha(args.manual_replacements), 'reviewer': decisions.get('reviewer', 'root'),
        'changes': changes, 'assertions': assertions, 'source_feature_frames': len(sources),
        'source_set_unchanged': True, 'source_feature_byte_hashes': feature_proofs,
        'model_directory_hashes_verified': model_hashes, 'real_inputs_verified': len(checked),
        'cpu_only': True, 'no_new_arcface_dino_extraction': True,
        'target_identity_texture_exact_zero': True, 'final_test_used_for_quality_or_selection': False}
    # A pending marker is written first so a failed CPU rebuild cannot look ready.
    atomic_json(out / 'readiness.json', {'status': 'manual_revision_cpu_finalize_pending'})
    for name, rows in [('body128_pairs.jsonl', body), ('face64_pairs.jsonl', face),
                       ('train_pairs.jsonl', face), ('val_pairs.jsonl', val), ('cross_person_pairs.jsonl', cross)]:
        write_rows(out / name, rows)
    atomic_json(out / 'manual_revision.json', revision)
    selection['pair_manifest_sha256'] = {name: sha(out / name) for name in selection['pair_manifest_sha256']}
    selection.update(status='manual_review_final_selection_locked', assertions=assertions,
        manual_revision_sha256=sha(out / 'manual_revision.json'),
        pre_manual_review_selection_sha256=sha(backup / 'data/selection_manifest.json'),
        manual_visibility_review='root reviewed all original pages and approved four clear-face replacements; final records pending',
        full_arcface_dino_features_complete=True)
    atomic_json(out / 'selection_manifest.json', selection)
    for app, cache in new_caches.items():
        if cache is not old_caches[app]:
            atomic_save_cache(cache, out / 'face_cache' / f'{app}.pt')
    fingerprints = {app: cache['metadata']['cache_fingerprint'] for app, cache in new_caches.items()}
    data_manifest = json.loads((out / 'data_manifest.json').read_text())
    data_manifest.update(selection='explicit frozen manifests after preparatory human visibility review; exact quotas retained',
        frozen_selection_provenance={'eligibility_inventory_sha256': sha(out / 'eligibility_inventory.json'),
            'selection_manifest_sha256': sha(out / 'selection_manifest.json'), 'manual_revision_sha256': sha(out / 'manual_revision.json')},
        input_sha256={name: sha(out / filename) for name, filename in
            [('train', 'face64_pairs.jsonl'), ('val', 'val12_pairs.jsonl'), ('cross_person', 'cross2_pairs.jsonl')]},
        cache_fingerprints=fingerprints, source_feature_frames=len(sources), all_cached_frames=sum(map(len, required.values())),
        feature_reuse_provenance='unchanged A RGB/SMPL-X/weights/cache fingerprint verified; A ArcFace/DINO copied byte-exact',
        manual_revision_sha256=sha(out / 'manual_revision.json'))
    atomic_json(out / 'data_manifest.json', data_manifest)
    audit = []
    preprocessing = json.loads((out / 'preprocess_manifest.json').read_text())
    preprocessing['manual_revision_sha256'] = sha(out / 'manual_revision.json')
    preprocessing['source_features_unchanged_after_verified_reuse'] = True
    atomic_json(out / 'preprocess_manifest.json', preprocessing)
    for split, rows in [('train', face), ('val', val), ('cross_person', cross)]:
        for row in rows:
            meta = pair_metadata(row, args.sampled_root)
            cache = new_caches[meta['source_appearance']]
            audit.append({'split': split, **meta, 'selected': True,
                'reason': 'unchanged_real_A_features_verified; fresh_real_B_geometry; manual_final_lock',
                'source_detection': cache['metadata']['preprocess']['per_source'][meta['source_stem']],
                'source_features_reused_byte_exact': True, 'manual_revision_sha256': sha(out / 'manual_revision.json')})
    write_rows(out / 'feature_extraction_audit.jsonl', audit)
    # Force new independent B detections and review forms; original copies are in backup.
    (out / 'face_supervision.json').unlink()
    shutil.rmtree(out / 'visibility_review')
    finalize(args)
    make_review(args)
    print(json.dumps({'manual_revision': str(out / 'manual_revision.json'), 'backup': str(backup),
        'body': len(body), 'face': len(face), 'val': len(val), 'cross': len(cross), 'cpu_only': True}, ensure_ascii=False), flush=True)


def make_review(args):
    """Display exact original-RGB detection crops without changing the set."""
    from PIL import ImageDraw, ImageFont
    out = args.out / 'visibility_review'
    out.mkdir(parents=True, exist_ok=True)
    crops = out / 'crops'
    crops.mkdir(exist_ok=True)
    inventory = json.loads((args.out / 'eligibility_inventory.json').read_text())['records']
    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    title = ImageFont.truetype(font_path, 21)
    small = ImageFont.truetype(font_path, 16)
    sheets, review_rows = [], []
    card_w, card_h, crop_w = 520, 455, 224
    for group, manifest, per_page, columns in [('train', 'face64_pairs.jsonl', 8, 2), ('val', 'val12_pairs.jsonl', 12, 3)]:
        rows = read_pairs(args.out / manifest)
        for page, offset in enumerate(range(0, len(rows), per_page), 1):
            selected = rows[offset:offset + per_page]
            sheet = Image.new('RGB', (columns * card_w, 80 + 4 * card_h), '#f0f1f3')
            draw = ImageDraw.Draw(sheet)
            draw.text((16, 8), f'{group}  第{page}页  A源 / B目标  正式SCRFD检测框原图脸裁剪', font=title, fill='#101820')
            draw.text((16, 41), '方向、遮挡与可见性均待人工确认；放大仅方便观察，不增加真实细节。', font=small, fill='#444444')
            sheet_name = f'{group}_page_{page:02d}.png'
            for index, row in enumerate(selected):
                x, y = index % columns * card_w, 80 + index // columns * card_h
                draw.rounded_rectangle((x + 5, y + 5, x + card_w - 5, y + card_h - 5), radius=8, fill='white')
                draw.text((x + 14, y + 8), f'{row["pair_id"]}  {row["appearance"]}', font=title, fill='#101820')
                review = {'group': group, 'pair_id': row['pair_id'], 'appearance': row['appearance'],
                          'sheet': str(out / sheet_name), 'manual_usable': None, 'manual_note': '', 'faces': {}}
                for side, dx, letter in [('source', 16, 'A'), ('target', 270, 'B')]:
                    record = inventory[f'{row["appearance"]}/{row[side]["stem"]}']
                    if sha(record['rgb']) != record['rgb_sha256']:
                        raise ValueError(f'Frozen review RGB hash differs: {record["rgb"]}')
                    selected_index = record['by_resolution']['512']['selected_detection']
                    detection = record['detections'][selected_index]
                    bbox = detection['bbox_original_px']
                    with Image.open(record['rgb']) as image:
                        face = image.convert('RGB').crop((int(np.floor(bbox[0])), int(np.floor(bbox[1])),
                                                         int(np.ceil(bbox[2])), int(np.ceil(bbox[3]))))
                    face = face.resize((crop_w, round(crop_w * face.height / face.width)), Image.Resampling.LANCZOS)
                    crop_path = crops / f'{group}_{row["pair_id"]}_{letter}.png'
                    face.save(crop_path)
                    sheet.paste(face, (x + dx, y + 42))
                    stem = row[side]['stem'].replace('source_standard_', 'std_').replace('source_motion_', 'motion_').replace('target_', '')
                    draw.text((x + dx, y + 395), f'{letter} {stem}', font=small, fill='#222222')
                    draw.text((x + dx, y + 418), f'score {detection["score"]:.3f} 512短边{detection["short_side_output_px"]["512"]:.1f}px', font=small, fill='#333333')
                    # Five-point asymmetry is only an attention cue. It cannot
                    # certify a view direction or the absence of occlusion.
                    points = detection['landmarks5_original_px']
                    asymmetry = None
                    if points is not None:
                        points = np.array(points)
                        eye_width = abs(float(points[1, 0] - points[0, 0]))
                        if eye_width > 1:
                            asymmetry = abs(float(points[2, 0] - (points[0, 0] + points[1, 0]) / 2)) / eye_width
                    review['faces'][letter] = {'stem': row[side]['stem'], 'crop': str(crop_path),
                        'bbox_original_px': bbox, 'score': detection['score'],
                        'short_side_512_px': detection['short_side_output_px']['512'],
                        'nose_eye_asymmetry_attention_proxy': asymmetry,
                        'direction_visibility': 'manual_pending', 'manual_direction': None, 'manual_visible': None}
                    if asymmetry is not None and asymmetry > .25:
                        draw.text((x + dx, y + 371), '可能侧转 / 请核方向及遮挡', font=small, fill='#a24d12')
                    else:
                        draw.text((x + dx, y + 371), '方向 / 可见性：待人工确认', font=small, fill='#555555')
                review_rows.append(review)
            sheet.save(out / sheet_name)
            sheets.append({'group': group, 'page': page, 'pairs': len(selected), 'image': str(out / sheet_name)})
    atomic_json(out / 'index.json', {'sheets': sheets, 'pairs': len(review_rows), 'display_crop_width': crop_w,
        'selection_manifest_sha256': sha(args.out / 'selection_manifest.json'),
        'manual_review_pending': True, 'real_features_or_training_generated': False})
    if not (out / 'review_form.json').exists():
        atomic_json(out / 'review_form.json', {'status': 'manual_pending', 'records': review_rows})
    print(json.dumps({'review_sheets': sheets, 'pairs': len(review_rows)}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['select', 'finalize', 'review', 'revise'], default='select')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sampled-root', type=Path, default=SAMPLED)
    parser.add_argument('--assets-root', type=Path, default=ASSETS)
    parser.add_argument('--detector', type=Path, default=ROOT / 'models/antelopev2/scrfd_10g_bnkps.onnx')
    parser.add_argument('--manual-replacements', type=Path,
                        help='Explicit root-confirmed preparatory visibility-review replacements')
    parser.add_argument('--insightface-root', type=Path, default=ROOT / 'models/antelopev2')
    parser.add_argument('--dino-model', type=Path, default=ROOT / 'models/dinov2-giant')
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.phase == 'revise':
        revise_manual(args)
        return
    if args.phase == 'finalize':
        finalize(args)
        return
    if args.phase == 'review':
        make_review(args)
        return
    if (args.out / 'selection_manifest.json').exists():
        raise FileExistsError('Frozen selection already exists; use another output directory')
    split_root = args.sampled_root / 'splits'
    inputs = {'dev_train': split_root / 'dev_train_pairs.jsonl', 'dev_val': split_root / 'dev_val_pairs.jsonl',
              'final50_isolation_only': split_root / 'baseline_50_eval_pairs.jsonl',
              'preserved_val4': RUN / 'val_pairs.jsonl', 'preserved_cross2': RUN / 'cross_person_pairs.jsonl'}
    groups = {name: read_pairs(path) for name, path in inputs.items()}
    by_app = defaultdict(list)
    for row in groups['dev_train']:
        by_app[row['appearance']].append(row)
    scanner = Scanner(args)
    body, face, chosen_apps, replacements = [], [], [], []
    try:
        for actor, preferred in PREFERRED.items():
            app_order = preferred + sorted(a for a in by_app if a.split('_')[0] == actor and a not in preferred)
            actor_apps = []
            for app in app_order:
                try:
                    app_body, app_face = select_training(app, by_app[app], scanner, args.seed)
                except RuntimeError as exc:
                    replacements.append({'actor': actor, 'appearance': app, 'reason': str(exc)})
                    continue
                actor_apps.append(app)
                chosen_apps.append(app)
                body.extend(app_body)
                face.extend(app_face)
                if len(actor_apps) == 2:
                    break
            if len(actor_apps) != 2:
                raise RuntimeError(f'Actor {actor}: only {len(actor_apps)} qualified appearances; need exactly two')
        for name, rows in [('body', body), ('face', face)]:
            for i, row in enumerate(rows):
                row['pair_id'] = f'{name}_{i:03d}' if name == 'body' else row['pair_id']
                row['multisample_face_selected'] = row in face
        # Face is the exact same row objects and identifiers as its Body subset.
        val = [dict(row) for row in groups['preserved_val4']]
        for row in val:
            a = scanner.frame(row['appearance'], row['source'])
            b = scanner.frame(row['appearance'], row['target'])
            if not a['by_resolution']['512']['valid'] or not b['by_resolution']['512']['valid'] or not b['body_conditions_valid']:
                raise RuntimeError(f'Preserved validation pair {row["pair_id"]} failed fresh face eligibility')
        for app in NEW_VAL:
            rows = [row for row in groups['dev_val'] if row['appearance'] == app]
            scanner.scan_app(app, rows)
            candidates = [annotate(row, scanner, args.seed) for row in rows]
            selected = []
            for slot in range(2):
                options = [r for r in candidates if r['selection_metrics']['source_reliable_512'] and r['selection_metrics']['target_reliable_512']
                           and r['selection_metrics']['temporal_gap_seconds'] >= 2
                           and scanner.frame(app, r['target'])['body_conditions_valid']
                           and r['target']['stem'] not in {s['target']['stem'] for s in selected}]
                if not options:
                    raise RuntimeError(f'Validation {app}: fewer than two reliable distinct B pairs')
                # Prefer different B poses and distant points in the video.
                chosen = max(options, key=lambda r: (min((abs(r['target']['timestamp_sec'] - s['target']['timestamp_sec']) for s in selected), default=0.)
                              + r['selection_metrics']['pose_distance'], r['selection_metrics']['tie_break']))
                selected.append(chosen)
                val.append({**chosen, 'pair_id': f'val_{len(val):03d}'})
        cross = groups['preserved_cross2']
        for row in cross:
            meta = pair_metadata(row)
            a = scanner.frame(meta['source_appearance'], row['source'])
            b = scanner.frame(meta['target_appearance'], row['target'])
            if not a['by_resolution']['512']['valid'] or not b['geometry_valid'] or not b['body_conditions_valid']:
                raise RuntimeError(f'Preserved cross-person pair {meta["pair_id"]} failed fresh eligibility')
        assertions = validate_sets(body, face, val, cross, groups['dev_train'], groups['dev_val'], groups['final50_isolation_only'])
        if val[:4] != groups['preserved_val4'] or cross != groups['preserved_cross2']:
            raise ValueError('The original four validation and two cross-person rows must be preserved exactly')
        for filename, rows in [('body128_pairs.jsonl', body), ('face64_pairs.jsonl', face),
                               ('val12_pairs.jsonl', val), ('cross2_pairs.jsonl', cross)]:
            write_rows(args.out / filename, rows)
        atomic_json(args.out / 'eligibility_inventory.json', {'schema': 'qwen21_full_multisample_eligibility_v1',
            'detector_sha256': scanner.detector_sha, 'provider': 'CPUExecutionProvider',
            'thresholds': {'score': .5, 'short_side_512_px': 32, 'iou': .2, 'center_inside': True,
                           'geometry_expansion': .15}, 'records': scanner.records})
        manifest = {'schema': 'qwen21_full_multisample_selection_v1', 'seed': args.seed,
            'status': 'selected_real_cpu_features_pending', 'counts': {'body': 128, 'face': 64, 'val': 12, 'cross': 2},
            'actors': list(PREFERRED), 'training_appearances': chosen_apps,
            'validation_appearances': sorted({r['appearance'] for r in val}), 'replacements': replacements,
            'input_sha256': {name: sha(path) for name, path in inputs.items()},
            'pair_manifest_sha256': {name: sha(args.out / name) for name in ['body128_pairs.jsonl', 'face64_pairs.jsonl', 'val12_pairs.jsonl', 'cross2_pairs.jsonl']},
            'assertions': assertions, 'scanned_rgb_count': len(scanner.records),
            'cpu_seconds': time.perf_counter() - scanner.started,
            'final_test_used_for_quality_or_selection': False,
            'selection': 'eight actors, two appearances each; fixed source-role quotas; temporal quarters and pose/head-rotation diversity',
            'manual_visibility_review': 'pending; SCRFD matching is a proxy', 'full_arcface_dino_features_complete': False}
        atomic_json(args.out / 'selection_manifest.json', manifest)
        for stale in ['selection_failure.json', 'eligibility_inventory_failed.json', 'eligibility_inventory_partial.json']:
            (args.out / stale).unlink(missing_ok=True)
        print(json.dumps(manifest, ensure_ascii=False), flush=True)
    except Exception as exc:
        atomic_json(args.out / 'selection_failure.json', {'error': f'{type(exc).__name__}: {exc}',
            'body_pairs': len(body), 'face_pairs': len(face), 'replacements': replacements,
            'scanned_rgb_count': len(scanner.records), 'no_silent_count_reduction': True})
        atomic_json(args.out / 'eligibility_inventory_failed.json', {'records': scanner.records})
        raise


if __name__ == '__main__':
    main()
