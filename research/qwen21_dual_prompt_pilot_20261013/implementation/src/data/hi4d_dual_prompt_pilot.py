"""Audited six-pair Hi4D pilot: source identity, external geometry and text.

The prompt builder accepts a narrow typed geometry mapping. Neither target RGB
nor true target masks can enter its signature or that mapping. Fitting uses
external SMPL body geometry; each actor's shape is learned from its source only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np


DATASET_SCHEMA_VERSION = 'hi4d_dual_prompt_pilot_v1'
SELECTED_ROWS = (
    ('train', 850, 'pair14', 'talk14', 32, 'hug14', 45, 4),
    ('train', 886, 'pair14', 'talk14', 32, 'hug14', 62, 4),
    ('train', 445, 'pair12', 'highfive12', 28, 'highfive12', 33, 4),
    ('train', 456, 'pair12', 'highfive12', 28, 'highfive12', 82, 4),
    ('val', 157, 'pair22', 'game22', 52, 'hug22', 46, 4),
    ('test', 46, 'pair00', 'taichi00', 41, 'hug00', 108, 4),
)
PROMPT_CONFIDENCE_THRESHOLD = .6
GEOMETRY_PROMPT_KEYS = {'person_uids', 'joints_world', 'body_confidence',
    'hand_confidence', 'root_orient', 'contact_primitives', 'geometry_sha256'}
PART_NAMES = ('head', 'torso', 'left upper arm', 'left forearm', 'left hand',
    'right upper arm', 'right forearm', 'right hand', 'left thigh', 'left shin',
    'left foot', 'right thigh', 'right shin', 'right foot')
PART_CONFIDENCE_JOINTS = ((12, 15), (0, 3, 6, 9, 13, 14), (16, 18), (18, 20), (20,),
    (17, 19), (19, 21), (21,), (1, 4), (4, 7), (7, 10), (2, 5), (5, 8), (8, 11))
BODY_CONFIDENCE_RULE = 'uncalibrated heuristic: 0.9*exp(-individual joint error_mm/100); part=min assigned/bone endpoint body joints; hand<=0.25 prior'


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def validate_split_selection(samples, split_manifest):
    """Require actual pair-level holdout membership; duplicates never multiply data."""
    groups = {}
    for split in ('train', 'val', 'test'):
        values = split_manifest.get(f'{split}_pairs')
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise ValueError(f'authoritative {split}_pairs must be unique pair names')
        groups[split] = set(values)
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if groups[a] & groups[b]:
            raise ValueError('authoritative split pair overlap')
    seen = set()
    selected = {split: set() for split in groups}
    for row in samples:
        split, pair = row['split'], row['pair']
        if split not in groups or pair not in groups[split]:
            raise ValueError(f'split membership mismatch: {split}/{pair}')
        identity = (pair, row['target_action'], int(row['target_frame_id']), int(row['camera_id']))
        if identity in seen:
            raise ValueError(f'duplicate target identity/frame: {identity}')
        seen.add(identity)
        selected[split].add(pair)
    return dict(pair_holdout_disjoint=True,
        identity_scope='pair groups; no global cross-pair actor identity proof',
        selected_pair_groups={key: sorted(value) for key, value in selected.items()},
        selected_pair_group_counts={key: len(value) for key, value in selected.items()})


def load_selected_rows(root):
    root = Path(root).resolve()
    base = root / 'datasets/Hi4D_pilot_v1'
    records = []
    for split, line, pair, source_action, source_frame, action, frame, camera in SELECTED_ROWS:
        path = base / 'native_interaction_v4_2' / f'main_pairs_{split}.jsonl'
        text_lines = path.read_text(encoding='utf-8').splitlines()
        row = json.loads(text_lines[line - 1])
        expected = dict(split=split, pair=pair, source_action=source_action,
            source_frame_id=source_frame, target_action=action, target_frame_id=frame, camera_id=camera)
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError(f'authoritative manifest row changed at {path}:{line}')
        # Original absolute manifest paths may refer to a relocated checkout.
        for key in ('source_rgb', 'source_person_mask_A', 'source_person_mask_B',
                    'target_rgb', 'target_person_mask_A', 'target_person_mask_B',
                    'target_dwpose_A', 'target_dwpose_B', 'target_smpl', 'target_camera'):
            original = Path(row[key])
            tail = original.as_posix().split('/datasets/Hi4D_pilot_v1/', 1)[-1]
            row[key] = str(base / tail)
            if not Path(row[key]).is_file():
                raise FileNotFoundError(row[key])
        row['original_manifest'] = str(path)
        row['original_manifest_line'] = line
        row['original_manifest_sha256'] = _sha(path)
        row['original_row_sha256'] = hashlib.sha256(text_lines[line - 1].encode()).hexdigest()
        row['sample_id'] = f'{pair}_{source_action}_{source_frame:06d}_to_{action}_{frame:06d}_cam{camera}'
        records.append(row)
    split_path = base / 'split.json'
    audit = validate_split_selection(records, json.loads(split_path.read_text()))
    return records, split_path, audit


def source_person_anchors(person_uids, source_masks):
    masks = np.asarray(source_masks)
    if len(person_uids) != 2 or len(set(person_uids)) != 2 or masks.ndim != 3 or masks.shape[0] != 2:
        raise ValueError('source anchors require two ordered UIDs and masks')
    output = []
    for slot, uid in enumerate(person_uids):
        y, x = np.nonzero(masks[slot] > .5)
        if len(x) < 16:
            raise ValueError('source anchor has insufficient visible mask pixels')
        output.append(dict(uid=uid, slot='AB'[slot],
            centroid_xy=[float(x.mean() / masks.shape[2]), float(y.mean() / masks.shape[1])],
            anchor_provenance='source_mask_centroid'))
    return output


def build_action_prompts(*, person_uids, source_anchors, action, geometry):
    """Deterministic text from source slot anchors and reliable external 3D only."""
    from scipy.spatial.transform import Rotation
    forbidden = set(geometry) - GEOMETRY_PROMPT_KEYS
    if forbidden:
        raise ValueError(f'forbidden prompt geometry fields: {sorted(forbidden)}')
    if set(geometry) != GEOMETRY_PROMPT_KEYS:
        raise ValueError('prompt geometry is incomplete')
    if list(geometry['person_uids']) != list(person_uids):
        raise ValueError('prompt geometry UID order does not match source')
    if len(source_anchors) != 2 or any(anchor.get('uid') != uid or anchor.get('slot') != 'AB'[i]
            for i, (anchor, uid) in enumerate(zip(source_anchors, person_uids))):
        raise ValueError('source anchor UID binding does not match source slots')
    if any(set(anchor) != {'uid', 'slot', 'centroid_xy', 'anchor_provenance'}
           or anchor['anchor_provenance'] != 'source_mask_centroid' for anchor in source_anchors):
        raise ValueError('forbidden or unverified source anchor fields')
    joints = np.asarray(geometry['joints_world'])
    confidence = np.asarray(geometry['body_confidence'])
    roots = np.asarray(geometry['root_orient'])
    if joints.shape[0] != 2 or joints.shape[1] < 55 or joints.shape[2] != 3 or confidence.shape != (2, 22) or roots.shape != (2, 3):
        raise ValueError('prompt requires ordered SMPL-X body joints/confidence')
    if not np.isfinite(joints).all() or not np.isfinite(confidence).all() or not np.isfinite(roots).all():
        raise ValueError('nonfinite prompt geometry')
    if re.fullmatch(r'hug\d+', action):
        instruction = 'Make the two existing people hug.'
    elif re.fullmatch(r'highfive\d+', action):
        instruction = 'Make the two existing people high-five.'
    else:
        raise ValueError(f'unsupported audited action label {action}')
    preservation = 'Preserve each person\'s face, hair, clothing and body identity, the background, and the original camera view.'
    used, omitted, descriptions = [], ['finger_articulation: unobserved/model prior'], []
    x_values = [a['centroid_xy'][0] for a in source_anchors]
    if abs(x_values[0] - x_values[1]) > .03:
        locations = ['left' if x_values[i] < x_values[1-i] else 'right' for i in range(2)]
        binding = ' '.join(f'{slot} is the person on the {locations[i]} in the source image.' for i, slot in enumerate('AB'))
    else:
        binding = ' '.join(f'{slot} is the source person at normalized mask centre ({a["centroid_xy"][0]:.2f}, {a["centroid_xy"][1]:.2f}).' for slot, a in zip('AB', source_anchors))
    rotations = Rotation.from_rotvec(roots).as_matrix()
    delta = joints[1, 0] - joints[0, 0]
    horizontal = delta.copy()
    horizontal[1] = 0
    distance = float(np.linalg.norm(horizontal))
    root_confidence = float(confidence[:, 0].min())
    if root_confidence >= PROMPT_CONFIDENCE_THRESHOLD and distance > .05:
        direction = horizontal / distance
        facing = [float(np.dot(rotations[i, :, 2], direction * (1 if i == 0 else -1))) for i in range(2)]
        if min(facing) > .5:
            descriptions.append('A and B face toward one another.')
        if distance < .7:
            descriptions.append('Keep A and B close together.')
        used.append(dict(feature='root_relation', confidence=root_confidence,
            horizontal_distance_m=distance, facing_cosines=facing, threshold=PROMPT_CONFIDENCE_THRESHOLD))
    for actor, slot in enumerate('AB'):
        wrist_heights = joints[actor, [20, 21], 1]
        waist = float(joints[actor, 0, 1])
        head = float(joints[actor, 15, 1])
        shoulder = float(joints[actor, [16, 17], 1].mean())
        levels = []
        for side, wrist_id, height in zip(('left', 'right'), (20, 21), wrist_heights):
            joint_ids = [0, 15, 16, 17, wrist_id]
            joint_confidence = float(confidence[actor, joint_ids].min())
            if joint_confidence < PROMPT_CONFIDENCE_THRESHOLD:
                omitted.append(f'{slot}_{side}_wrist_height: low body confidence')
                levels.append(None)
                continue
            if height > head + .03:
                levels.append('above head level')
            elif height >= shoulder - .10:
                levels.append('at shoulder level')
            elif height <= waist + .15:
                levels.append('near waist level')
            else:
                levels.append('at chest level')
            used.append(dict(feature=f'{slot}_{side}_wrist_height', joint_indices=joint_ids,
                wrist_height_m=float(height), waist_height_m=waist, shoulder_height_m=shoulder,
                head_height_m=head, confidence=joint_confidence, threshold=PROMPT_CONFIDENCE_THRESHOLD))
        if all(level is not None for level in levels):
            phrase = ('hands ' + levels[0] if levels[0] == levels[1]
                      else f'left hand {levels[0]} and the right hand {levels[1]}')
        elif any(level is not None for level in levels):
            index = next(i for i, level in enumerate(levels) if level is not None)
            phrase = f'{("left", "right")[index]} hand {levels[index]}'
        else:
            phrase = None
        # This describes the reliable wrist/body region, never palms or fingers.
        if phrase is not None:
            descriptions.append(f'{slot} keeps the {phrase}.')
        for side, shoulder_id, elbow_id, wrist_id in (('left', 16, 18, 20), ('right', 17, 19, 21)):
            if confidence[actor, [shoulder_id, elbow_id, wrist_id]].min() < PROMPT_CONFIDENCE_THRESHOLD:
                omitted.append(f'{slot}_{side}_elbow: low body confidence')
                continue
            upper = joints[actor, shoulder_id] - joints[actor, elbow_id]
            lower = joints[actor, wrist_id] - joints[actor, elbow_id]
            denominator = np.linalg.norm(upper) * np.linalg.norm(lower)
            if denominator < 1e-6:
                continue
            angle = float(np.degrees(np.arccos(np.clip(np.dot(upper, lower) / denominator, -1, 1))))
            if angle < 80:
                descriptions.append(f'{slot} bends the {side} elbow tightly.')
            elif angle < 135:
                descriptions.append(f'{slot} keeps the {side} elbow bent.')
            else:
                descriptions.append(f'{slot} extends the {side} arm.')
            used.append(dict(feature=f'{slot}_{side}_elbow', joint_indices=[shoulder_id, elbow_id, wrist_id],
                angle_deg=angle, thresholds_deg=[80, 135], confidence=float(confidence[actor, [shoulder_id, elbow_id, wrist_id]].min())))
        if action.startswith('hug'):
            partner = 1 - actor
            chest = joints[partner, 9]
            wrist_distances = np.linalg.norm(joints[actor, [20, 21]] - chest, axis=-1)
            reliable_wrists = confidence[actor, 20:22] >= PROMPT_CONFIDENCE_THRESHOLD
            valid_contact = reliable_wrists & (wrist_distances < .45)
            if valid_contact.any() and confidence[partner, 9] >= PROMPT_CONFIDENCE_THRESHOLD:
                descriptions.append(f'{slot} brings an arm around the other person\'s upper body.')
                selected_wrist = int(np.flatnonzero(valid_contact)[0])
                used.append(dict(feature=f'{slot}_arm_to_partner_torso', wrist_joint=20 + selected_wrist,
                    wrist_to_torso_m=float(wrist_distances[selected_wrist]), threshold_m=.45,
                    confidence=float(min(confidence[actor, 20 + selected_wrist], confidence[partner, 9]))))
    contact_sentences = []
    for contact in geometry['contact_primitives']:
        if set(contact) != {'parts', 'confidence', 'mapping_error_m'}:
            raise ValueError('forbidden prompt contact fields')
        a, b = map(int, contact['parts'])
        if not 0 <= a < 14 or not 0 <= b < 14:
            raise ValueError('invalid anatomical contact parts')
        # Unobserved finger topology is always a prior, including high-five frames.
        if contact['confidence'] < PROMPT_CONFIDENCE_THRESHOLD or a in (4, 7) or b in (4, 7):
            continue
        contact_sentences.append(f'A\'s {PART_NAMES[a]} meets B\'s {PART_NAMES[b]}.')
        used.append(dict(feature='external_contact', **contact, threshold=PROMPT_CONFIDENCE_THRESHOLD))
    descriptions.extend(sorted(set(contact_sentences)))
    structured = f'{binding} {instruction} ' + ' '.join(descriptions) + f' {preservation}'
    return dict(short=f'{instruction} {preservation}', structured=structured,
        provenance=dict(template_version='source_geometry_action_v1', action_label=action,
            person_uids=list(person_uids), source_anchors=source_anchors,
            geometry_sha256=geometry['geometry_sha256'], confidence_threshold=PROMPT_CONFIDENCE_THRESHOLD,
            used=used, omitted=omitted, inputs='source mask anchors + action label + external geometry only'))


def validate_fitted_identity(source, target):
    for fit in (source, target):
        if np.asarray(fit['vertices_world']).shape != (2, 10475, 3) or np.asarray(fit['joints_world']).shape[0] != 2 or np.asarray(fit['joints_world']).shape[1] < 55 or np.asarray(fit['faces']).shape != (20908, 3):
            raise ValueError('fit must retain true SMPL-X topology and ordered anatomical joints')
        if not np.array_equal(fit['actor_ids'], [0, 1]):
            raise ValueError('SMPL-X fit actor IDs must preserve source order')
        if any(not np.isfinite(fit[key]).all() for key in ('vertices_world', 'joints_world', 'betas')):
            raise ValueError('nonfinite SMPL-X fit')
    if not np.array_equal(source['betas'], target['betas']):
        raise ValueError('target must preserve each source actor beta exactly')


def fit_smpl_frame(model, raw_smpl, *, device='cuda', steps=200, source_betas=None):
    """Fit external 3D body/surface geometry; source shape is frozen for targets.

    Raw SMPL rotations initialize homologous body joints, but SMPL betas are
    never relabelled SMPL-X betas. Hands, eyes, jaw and expression are zero priors.
    """
    import torch
    from scipy.spatial import cKDTree
    if not 1 <= steps <= 300:
        raise ValueError('fit steps must be between 1 and 300')
    started = time.monotonic()
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    params = {
        'global_orient': torch.nn.Parameter(tensor(raw_smpl['global_orient'])),
        'body_pose': torch.nn.Parameter(tensor(raw_smpl['body_pose'][:, :63])),
        'betas': (torch.nn.Parameter(tensor(np.zeros((2, 10)))) if source_betas is None else tensor(source_betas)),
        'left_hand_pose': tensor(np.zeros((2, 45))), 'right_hand_pose': tensor(np.zeros((2, 45))),
        'jaw_pose': tensor(np.zeros((2, 3))), 'expression': tensor(np.zeros((2, 10))),
        'leye_pose': tensor(np.zeros((2, 3))), 'reye_pose': tensor(np.zeros((2, 3))),
    }
    target_joints = tensor(raw_smpl['joints_3d'][:, :22])
    with torch.no_grad():
        initial = model(**params)
    params['transl'] = torch.nn.Parameter(target_joints[:, 0] - initial.joints[:, 0])
    initial_pose = params['body_pose'].detach().clone()
    trainable = [params[key] for key in ('global_orient', 'body_pose', 'transl')]
    groups = [{'params': list(trainable), 'lr': .012}]
    if source_betas is None:
        trainable.append(params['betas'])
        groups.append({'params': [params['betas']], 'lr': .025})
    optimizer = torch.optim.Adam(groups)
    weights = model.lbs_weights.detach().cpu().numpy()
    body_ids = np.flatnonzero((weights[:, :22].sum(1) > .9) & (weights[:, 15] < .3))[::4]
    body_ids_t = torch.as_tensor(body_ids, dtype=torch.long, device=device)
    trees = [cKDTree(vertices) for vertices in raw_smpl['verts']]
    before = None
    history = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        result = model(**params)
        body_error = (result.joints[:, :22] - target_joints).norm(dim=-1).mean(-1)
        if before is None:
            before = body_error.detach().cpu().numpy() * 1000
        if step % 20 == 0:
            detached = result.vertices.detach().cpu().numpy()
            correspondences = tensor(np.stack([raw_smpl['verts'][i, trees[i].query(detached[i, body_ids])[1]] for i in range(2)]))
        joint_loss = (result.joints[:, :22] - target_joints).square().mean()
        surface_loss = (result.vertices[:, body_ids_t] - correspondences).square().mean()
        prior = (params['body_pose'] - initial_pose).square().mean()
        loss = 200 * joint_loss + 25 * surface_loss + .002 * prior + .0001 * params['betas'].square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite external SMPL-X fitting loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 5.)
        optimizer.step()
        if step % 50 == 0 or step == steps - 1:
            entry = dict(step=step, loss=float(loss.detach()), body_joint_mean_mm=(body_error.detach().cpu().numpy() * 1000).tolist())
            history.append(entry)
            print(json.dumps({'fit_progress': entry, 'source_shape_optimized': source_betas is None}), flush=True)
    with torch.no_grad():
        result = model(**params)
        joint_errors = (result.joints[:, :22] - target_joints).norm(dim=-1).cpu().numpy() * 1000
        errors = joint_errors.mean(-1)
    arrays = {key: value.detach().cpu().numpy() for key, value in params.items()}
    arrays.update(vertices_world=result.vertices.cpu().numpy(), joints_world=result.joints.cpu().numpy(),
        faces=model.faces.astype(np.int32), actor_ids=np.array([0, 1]), body_joint_mean_mm=errors.astype(np.float32),
        body_joint_mm=joint_errors.astype(np.float32))
    if source_betas is not None and not np.array_equal(arrays['betas'], np.asarray(source_betas, np.float32)):
        raise ValueError('target fitting changed frozen source shape')
    meta = dict(model_type='smplx', annotation_type='fitted_geometry_proxy_not_official_ground_truth',
        source_shape_optimized=source_betas is None, shape_inputs='source SMPL 3D body only' if source_betas is None else 'fixed source actor beta10',
        body_joint_before_mm=before.tolist(), body_joint_after_mm=errors.tolist(),
        body_joint_mm=joint_errors.tolist(), confidence_rule=BODY_CONFIDENCE_RULE,
        body_quality_within_30mm=bool((errors < 30).all()),
        hands='zero model prior; no observed finger articulation', face='zero jaw/expression/eye prior',
        constraints=['external SMPL first 22 homologous body joints', 'nearest external body surface proxy'],
        initialization='external SMPL homologous axis-angle body pose; SMPL-X beta10 initialized at zero',
        fitting_scope='offline mesh parameters only; frozen SMPL-X model; no Qwen/adapter model loaded',
        target_rgb_conditioning=False, target_true_mask_conditioning=False,
        fit_steps=steps, elapsed_seconds=time.monotonic() - started, loss_history=history)
    return arrays, meta


def body_pose_points(xy_native, scores, width, height, prior_xy):
    from src.data.hi4d_smplx_dataset import letterbox_affine
    xy = np.asarray(xy_native)[:, :24].copy()
    confidence = np.asarray(scores)[:, :24].copy()
    affine = letterbox_affine(int(width), int(height))
    xy = xy * affine[0, 0] + affine[:2, 2]
    missing = (confidence < .25) | ~np.isfinite(xy).all(-1)
    xy[missing] = np.asarray(prior_xy)[missing]
    confidence[missing] = .25
    kinds = np.full((2, 24), 'camera_dwpose', dtype='<U24')
    kinds[missing] = 'fitted_smplx_prior'
    return xy.astype(np.float32), confidence.astype(np.float32), kinds.tolist()


def fitted_body24_points(joints):
    """SMPL-X anatomical body fallback, OpenPose/DWPose body and feet order."""
    joints = np.asarray(joints)
    mapping = [15, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7, 15, 15, 15, 15, 10, 10, 7, 11, 11, 8]
    return joints[:, mapping]


def body_and_part_confidence(body_joint_mm):
    errors = np.asarray(body_joint_mm, np.float32)
    if errors.shape != (2, 22) or not np.isfinite(errors).all() or (errors < 0).any():
        raise ValueError('body confidence requires finite per-joint fit errors [2,22]')
    body = (.9 * np.exp(-errors / 100)).astype(np.float32)
    parts = np.stack([body[:, list(joints)].min(-1) for joints in PART_CONFIDENCE_JOINTS], -1)
    parts[:, [4, 7]] = np.minimum(parts[:, [4, 7]], .25)
    return body, parts


def export_target_controls(*, source_fit, target_fit, raw_smpl, camera_calibration,
        pose_observations, renderer, labels, heights, output, identity,
        fit_path, source_fit_path, raw_smpl_path):
    """Rasterize mesh geometry and external pose, with no true mask/RGB access."""
    from PIL import Image, ImageDraw
    from scripts.precompute.precompute_hi4d_smplx_controls import project, contact_controls
    from src.data.hi4d_smplx_dataset import (anatomical_part_geometry, geometry_vectors,
        visible_owner, body25_heatmaps, letterbox_affine)
    from src.pose_control.qwen21_full.dual.schema import (ANATOMICAL_PART_JOINT_INDICES,
        SMPLX55_JOINT_NAMES, SMPLX14_PARTITION_SPEC, geometry_schema_metadata)
    validate_fitted_identity(source_fit, target_fit)
    width, height = pose_observations['width'], pose_observations['height']
    K = letterbox_affine(width, height) @ camera_calibration['K']
    E = np.asarray(camera_calibration['extrinsics'])
    vertices = target_fit['vertices_world']
    rendered = [renderer.render(v, E[:, :3], E[:, 3], K) for v in vertices]
    amodal = np.stack([r['mask'] for r in rendered])
    depth = np.stack([r['depth'] for r in rendered])
    owner = visible_owner(amodal, depth)
    prior_xyz = fitted_body24_points(target_fit['joints_world'])
    prior_xy = np.stack([project(points, K, E) for points in prior_xyz])
    xy, score, pose_kinds = body_pose_points(pose_observations['xy_native'], pose_observations['scores'], width, height, prior_xy)
    pose = np.stack([body25_heatmaps(xy[i], score[i]) for i in range(2)])
    body_confidence, part_confidence = body_and_part_confidence(target_fit['body_joint_mm'])
    positions = np.stack([[v[labels == part].mean(0) for part in range(14)] for v in vertices]).astype(np.float32)
    orientations, part_joints, valid = anatomical_part_geometry(target_fit['joints_world'])
    valid &= np.isfinite(positions).all(-1)
    for value in (positions, orientations, part_joints, part_confidence):
        value[~valid] = 0
    global32, relative14 = geometry_vectors(source_fit['betas'], target_fit['global_orient'],
        target_fit['joints_world'][:, 0], E[:, :3], E[:, 3], K, amodal, heights)
    normals_world = [r['vertex_normals_camera'] @ E[:, :3] for r in rendered]
    endpoints, features, contact_map, contacts = contact_controls(raw_smpl, vertices,
        labels, normals_world, owner, K, E, part_confidence)
    partition_sha = hashlib.sha256(np.asarray(labels, dtype='<i8').tobytes()).hexdigest()
    values = dict(normal=np.stack([r['normal'] for r in rendered]), depth=depth[:, None],
        part=np.stack([r['part'] for r in rendered]), pose=pose, amodal_mask=amodal[:, None].astype(np.float32),
        visible_owner=owner[:, None], global32=global32, relative14=relative14,
        part_positions=positions, part_orientations=orientations, part_joints=part_joints,
        part_valid=valid, part_confidence=part_confidence,
        contact_endpoints=endpoints, contact_features=features, contact_valid=np.ones(len(endpoints), bool),
        contact_map=contact_map, person_valid=np.ones(2, bool), person_uids=np.asarray(identity['person_uids']),
        **geometry_schema_metadata(), source_shape_shared=np.array(True),
        source_shape_fit_indices=np.array([0, 1]), anatomical_joint_indices=np.asarray(ANATOMICAL_PART_JOINT_INDICES, np.int64),
        anatomical_joint_names=np.asarray(SMPLX55_JOINT_NAMES), smplx_sha256=_sha(fit_path),
        source_shape_sha256=_sha(source_fit_path), contact_source_sha256=_sha(raw_smpl_path),
        vertex_partition_sha256=partition_sha, body_joint_mm=target_fit['body_joint_mm'],
        body_joint_confidence=body_confidence, confidence_rule=BODY_CONFIDENCE_RULE)
    path = Path(output) / f'{identity["action"]}_{identity["target_frame"]:06d}_cam{identity["camera"]}_geometry.npz'
    np.savez_compressed(path, **values)
    info = dict(**identity, **geometry_schema_metadata(), geometry=path.name, geometry_sha256=_sha(path),
        smplx_sha256=_sha(fit_path), source_shape_sha256=_sha(source_fit_path), source_shape_shared=True,
        source_shape_verification='exact source/target beta10 equality; optimized from source only',
        fitted_geometry_proxy=True, hand_status='unobserved zero model prior',
        body_joint_mean_mm=target_fit['body_joint_mean_mm'].tolist(),
        body_joint_mm=target_fit['body_joint_mm'].tolist(), body_joint_confidence=body_confidence.tolist(),
        confidence_rule=BODY_CONFIDENCE_RULE,
        part_confidence_joint_indices=[list(joints) for joints in PART_CONFIDENCE_JOINTS],
        pose_observation_paths=pose_observations['paths'], pose_observation_sha256=[_sha(p) for p in pose_observations['paths']],
        observation_width=int(width), observation_height=int(height), pose_provenance=pose_kinds,
        body_pose_xy512=xy.tolist(), body_pose_confidence=score.tolist(),
        contact_source_sha256=_sha(raw_smpl_path), contacts=contacts,
        contact_mapping='external one-based SMPL contact -> nearest SMPL-X surface proxy', signed_gap_valid=False,
        K512=K.tolist(), extrinsics=E.tolist(), source_neutral_heights=heights.tolist(),
        partition_spec=SMPLX14_PARTITION_SPEC, vertex_partition_sha256=partition_sha)
    _json(path.with_suffix('.json'), info)
    panels, names = [], []
    for i, slot in enumerate('AB'):
        panels.extend([np.moveaxis((values['normal'][i] + 1) * .5, 0, -1) * amodal[i, ..., None],
            np.repeat(owner[i, ..., None], 3, -1), np.repeat(pose[i].max(0)[..., None], 3, -1)])
        names.extend([f'{slot} fitted normals', f'{slot} mesh visible owner', f'{slot} external/prior pose'])
    panels.append(np.repeat(contact_map[0, ..., None], 3, -1))
    names.append('mapped contact proxy')
    sheet = Image.new('RGB', (512 * len(panels), 560), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 5), 'EXTERNAL FITTED CONTROL INSPECTION - not model output; hands/face are priors', fill='black')
    for n, (panel, title) in enumerate(zip(panels, names)):
        sheet.paste(Image.fromarray((panel.clip(0, 1) * 255).astype(np.uint8)), (n * 512, 48))
        draw.text((n * 512 + 8, 27), title, fill='black')
    sheet.save(path.with_name(path.stem.replace('_geometry', '_controls') + '.png'))
    prompt_geometry = dict(person_uids=identity['person_uids'], joints_world=target_fit['joints_world'],
        body_confidence=body_confidence, hand_confidence=np.full((2, 2), .25),
        root_orient=target_fit['global_orient'], contact_primitives=[{key: contact[key] for key in ('parts', 'confidence', 'mapping_error_m')} for contact in contacts],
        geometry_sha256=_sha(path))
    return path, prompt_geometry


def _materialize_assets(rgb_path, mask_paths, output, *, target):
    from PIL import Image
    from src.data.hi4d_smplx_dataset import letterbox_array, letterbox_affine
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    image = np.asarray(Image.open(rgb_path).convert('RGB'))
    width, height = image.shape[1], image.shape[0]
    Image.fromarray(letterbox_array(image)).save(output / 'rgb.png')
    masks = []
    for slot, path in zip('AB', mask_paths):
        array = np.asarray(Image.open(path).convert('L'))
        if array.shape != (height, width):
            raise ValueError('source/truth mask resolution does not match RGB')
        mask_path = output / f'{"gt_" if target else ""}mask_{slot}.png'
        Image.fromarray(letterbox_array((array > 127).astype(np.uint8) * 255, nearest=True)).save(mask_path)
        masks.append(str(mask_path))
    return dict(rgb=str(output / 'rgb.png'), masks=masks,
        preprocessing=dict(native_size=[width, height], resolution=512, resize='isotropic letterbox',
            affine=letterbox_affine(width, height).tolist(), RGB_interpolation='OpenCV linear',
            mask_interpolation='nearest', output_encoding='lossless PNG',
            raw_rgb=str(rgb_path), raw_rgb_sha256=_sha(rgb_path),
            raw_masks=list(map(str, mask_paths)), raw_mask_sha256=[_sha(p) for p in mask_paths]))


def materialize_source_assets(source_rgb, source_masks, output):
    if len(source_masks) != 2:
        raise ValueError('source requires exactly two ordered masks')
    result = _materialize_assets(source_rgb, source_masks, output, target=False)
    return dict(source_rgb=result['rgb'], source_masks=result['masks'], preprocessing=result['preprocessing'])


def _read_npz(path):
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key].copy() for key in values.files}


def _reuse_exact_source_cache(root, original, source, person_uids, output):
    """Reuse appearance tensors only when all preprocessing pixels match exactly."""
    import torch
    from PIL import Image
    from src.data.hi4d_smplx_dataset import (read_rgb, read_mask, source_person_crops,
        source_cache_metadata)
    rgb = (read_rgb(source['source_rgb']).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    masks = np.stack([read_mask(p)[0].numpy() for p in source['source_masks']])
    crops, coverage, boxes = source_person_crops(rgb, masks)
    expected_original = source_cache_metadata(original['source_rgb'],
        [original['source_person_mask_A'], original['source_person_mask_B']], person_uids)
    for candidate in (Path(root) / 'research/qwen21_dual_upgrade_20261011/pilot_controls_v2',):
        path = candidate / 'source_features.pt'
        if not path.is_file():
            continue
        payload = torch.load(path, map_location='cpu', weights_only=True)
        metadata = payload.get('metadata', {})
        if any(metadata.get(key) != value for key, value in expected_original.items()):
            continue
        if any(not (candidate / f'source_person_{i}.png').is_file()
               or not np.array_equal(np.asarray(Image.open(candidate / f'source_person_{i}.png').convert('RGB')), crops[i])
               for i in range(2)):
            continue
        new_metadata = source_cache_metadata(source['source_rgb'], source['source_masks'], person_uids)
        encoder = metadata.get('encoder_provenance')
        if not isinstance(encoder, dict):
            continue
        # Existing cache remains bound to the exact encoder files where recorded.
        encoder_files = [('dino_path', 'model.safetensors', 'dino_sha256'),
                         ('vae_path', 'config.json', 'vae_config_sha256')]
        if any(_sha(Path(encoder[folder]) / name) != encoder[key] for folder, name, key in encoder_files):
            raise ValueError('source cache encoder SHA no longer matches installed encoder')
        if encoder.get('vae_weights_sha256'):
            if _sha(Path(encoder['vae_path']) / 'diffusion_pytorch_model.safetensors') != encoder['vae_weights_sha256']:
                raise ValueError('source cache VAE weight SHA no longer matches encoder')
        new_metadata.update(encoder_provenance=encoder,
            reused_from_features=str(path), reused_from_features_sha256=_sha(path),
            reuse_verification='raw RGB/mask SHA match and exact letterboxed person crop pixels')
        payload = dict(payload, metadata=new_metadata)
        torch.save(payload, Path(output) / 'source_features.pt')
        for i, crop in enumerate(crops):
            Image.fromarray(crop).save(Path(output) / f'source_person_{i}.png')
        _json(Path(output) / 'source_features.json', dict(metadata=new_metadata,
            source_rgb=source['source_rgb'], source_masks=source['source_masks'],
            source_features_sha256=_sha(Path(output) / 'source_features.pt')))
        return True
    return False


def _validate_selection_lock(path, rows, split_path):
    locked = json.loads(Path(path).read_text())
    if locked.get('selection_locked_before_training') is not True or locked.get('split_manifest_sha256') != _sha(split_path):
        raise ValueError('selection lock must bind actual original split before training')
    audit = {(record['split'], record['one_based_line']): record for record in locked['rows']}
    if set(audit) != {(row['split'], row['original_manifest_line']) for row in rows}:
        raise ValueError('selected rows do not match fixed selection lock')
    for row in rows:
        record = audit[(row['split'], row['original_manifest_line'])]
        if record['manifest_sha256'] != row['original_manifest_sha256'] or record['row_sha256'] != row['original_row_sha256']:
            raise ValueError('selected source manifest content changed after lock')


def _truth_board(source_rgb, target_rgb, output, title):
    from PIL import Image, ImageDraw
    sheet = Image.new('RGB', (1024, 564), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 5), 'DATASET SOURCE AND GROUND TRUTH - neither panel is generated model output', fill='black')
    draw.text((8, 23), title, fill='black')
    for i, (path, label) in enumerate(((source_rgb, 'source appearance input'), (target_rgb, 'dataset target ground truth'))):
        draw.text((i * 512 + 8, 39), label, fill='black')
        sheet.paste(Image.open(path).convert('RGB'), (i * 512, 52))
    sheet.save(output)


def prepare_dataset(*, root, output, dino, vae, device='cuda', fit_steps=200, selection_lock=None):
    """Materialize the fixed audit selection into isolated strict old manifests."""
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('choose a new output directory to preserve existing data')
    if not 1 <= fit_steps <= 300:
        raise ValueError('fit steps must be between 1 and 300')
    rows, split_path, audit = load_selected_rows(root)
    if selection_lock is None:
        selection_lock = root / 'research/qwen21_dual_prompt_pilot_20261013/selection_lock.json'
    selection_lock = Path(selection_lock).resolve()
    if not selection_lock.is_file():
        raise FileNotFoundError(f'required selection lock is missing: {selection_lock}')
    _validate_selection_lock(selection_lock, rows, split_path)
    import torch
    import smplx
    from smplx.joint_names import JOINT_NAMES
    from PIL import Image
    from scripts.precompute.precompute_hi4d_smplx_controls import source_features
    from src.data.hi4d_smplx_dataset import (SMPLXRenderer, smplx_part_labels, read_mask,
        validate_source_feature_cache, validate_generation_manifest, bind_supervision_manifest,
        Hi4DSMPLXDataset)
    from src.pose_control.qwen21_full.dual.schema import generation_versions, SMPLX55_JOINT_NAMES
    if tuple(JOINT_NAMES[:55]) != SMPLX55_JOINT_NAMES:
        raise ValueError('installed SMPL-X anatomical joint ordering differs from schema')
    torch.set_num_threads(8)
    torch.manual_seed(0)
    model_base = root / 'add/tools/SMPLest-X/human_models/human_model_files'
    model = smplx.create(str(model_base), 'smplx', gender='neutral', num_betas=10,
        num_expression_coeffs=10, use_pca=False, flat_hand_mean=True, batch_size=2).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    labels = smplx_part_labels(model.lbs_weights.detach().cpu().numpy())
    renderer = SMPLXRenderer(model.faces, labels, device=device)
    output.mkdir(parents=True)
    started = time.monotonic()
    split_rows = {split: {'short': [], 'structured': [], 'truth': [], 'sample_ids': []} for split in ('train', 'val', 'test')}
    sources = {}
    records = []
    for row in rows:
        sample_dir = output / 'samples' / row['sample_id']
        sample_dir.mkdir(parents=True)
        uids = [f'{row["pair"]}_actor{i}' for i in range(2)]
        source_key = (row['pair'], row['source_action'], row['source_frame_id'], row['camera_id'])
        if source_key not in sources:
            source_dir = output / 'sources' / f'{row["pair"]}_{row["source_action"]}_{row["source_frame_id"]:06d}_cam{row["camera_id"]}'
            source_dir.mkdir(parents=True)
            source_asset_dir = output / 'assets/sources' / row['pair'] / f'cam{row["camera_id"]}' / row['source_action'] / f'{row["source_frame_id"]:06d}'
            source = materialize_source_assets(row['source_rgb'],
                [row['source_person_mask_A'], row['source_person_mask_B']], source_asset_dir)
            source_smpl_path = root / 'datasets/Hi4D_pilot_v1/raw' / row['pair'] / row['source_action'] / 'smpl' / f'{row["source_frame_id"]:06d}.npz'
            source_fit, source_meta = fit_smpl_frame(model, _read_npz(source_smpl_path), device=device, steps=fit_steps)
            source_fit_path = source_dir / 'source_smplx.npz'
            np.savez_compressed(source_fit_path, **source_fit)
            source_meta.update(source_smpl=str(source_smpl_path), source_smpl_sha256=_sha(source_smpl_path),
                source_shape_sha256=_sha(source_fit_path), source_assets=source)
            _json(source_dir / 'source_smplx.json', source_meta)
            reused = _reuse_exact_source_cache(root, row, source, uids, source_dir)
            if not reused:
                source_features(source['source_rgb'], source['source_masks'], Path(dino), Path(vae), source_dir, device, person_uids=uids)
            masks = np.stack([read_mask(p)[0].numpy() for p in source['source_masks']])
            anchors = source_person_anchors(uids, masks)
            with torch.no_grad():
                neutral = model(betas=torch.as_tensor(source_fit['betas'], dtype=torch.float32, device=device)).vertices.cpu().numpy()
            heights = np.ptp(neutral[:, :, 1], axis=1)
            sources[source_key] = dict(assets=source, fit=source_fit, fit_meta=source_meta, fit_path=source_fit_path,
                feature_path=source_dir / 'source_features.pt', anchors=anchors, heights=heights, source_cache_reused=reused)
        source = sources[source_key]
        raw_target = _read_npz(row['target_smpl'])
        target_fit, target_meta = fit_smpl_frame(model, raw_target, device=device, steps=fit_steps, source_betas=source['fit']['betas'])
        validate_fitted_identity(source['fit'], target_fit)
        fit_path = sample_dir / 'target_smplx.npz'
        np.savez_compressed(fit_path, **target_fit)
        target_meta.update(target_smpl=row['target_smpl'], target_smpl_sha256=_sha(row['target_smpl']),
            source_shape_sha256=_sha(source['fit_path']), target_fit_sha256=_sha(fit_path))
        _json(sample_dir / 'target_smplx.json', target_meta)
        camera = _read_npz(row['target_camera'])
        views = np.flatnonzero(camera['ids'] == row['camera_id'])
        if len(views) != 1 or np.max(np.abs(camera['dist_coeffs'][views[0]])) > 1e-8:
            raise ValueError('target camera must be unique and undistorted')
        view = int(views[0])
        pose_paths = [row['target_dwpose_A'], row['target_dwpose_B']]
        observed = [_read_npz(p) for p in pose_paths]
        if any(int(o['width']) != row['native_width'] or int(o['height']) != row['native_height'] for o in observed):
            raise ValueError('external pose dimensions differ from authoritative camera pixels')
        pose = dict(xy_native=np.stack([o['xy_native'] for o in observed]), scores=np.stack([o['scores'] for o in observed]),
            width=int(observed[0]['width']), height=int(observed[0]['height']), paths=pose_paths)
        pose_path = sample_dir / 'external_pose_observations.npz'
        np.savez_compressed(pose_path, **{key: pose[key] for key in ('xy_native', 'scores', 'width', 'height')})
        identity = dict(pair=row['pair'], action=row['target_action'], source_frame=int(row['source_frame_id']),
            target_frame=int(row['target_frame_id']), camera=int(row['camera_id']), person_uids=uids)
        geometry_path, prompt_geometry = export_target_controls(source_fit=source['fit'], target_fit=target_fit,
            raw_smpl=raw_target, camera_calibration=dict(K=camera['intrinsics'][view], extrinsics=camera['extrinsics'][view]),
            pose_observations=pose, renderer=renderer, labels=labels, heights=source['heights'], output=sample_dir,
            identity=identity, fit_path=fit_path, source_fit_path=source['fit_path'], raw_smpl_path=row['target_smpl'])
        geometry_info = json.loads(geometry_path.with_suffix('.json').read_text())
        geometry_info.update(external_pose_observations=str(pose_path), external_pose_observations_sha256=_sha(pose_path),
            camera_calibration_path=row['target_camera'], camera_calibration_sha256=_sha(row['target_camera']))
        _json(geometry_path.with_suffix('.json'), geometry_info)
        prompts = build_action_prompts(person_uids=uids, source_anchors=source['anchors'],
            action=row['target_action'], geometry=prompt_geometry)
        _json(sample_dir / 'prompts.json', prompts)
        base = dict(source_rgb=source['assets']['source_rgb'], source_masks=source['assets']['source_masks'],
            source_features=str(source['feature_path']), source_features_sha256=_sha(source['feature_path']),
            target_geometry=str(geometry_path), person_uids=uids, seed=42, resolution=512,
            versions=generation_versions(), target_geometry_known=True)
        generations = {variant: dict(base, prompt=prompts[variant]) for variant in ('short', 'structured')}
        for generation in generations.values():
            validate_generation_manifest(generation)
            validate_source_feature_cache(torch.load(source['feature_path'], map_location='cpu', weights_only=True), generation)
        # Read true RGB/masks only after external geometry and prompts are complete.
        truth_dir = output / 'assets/targets' / row['pair'] / row['target_action'] / f'cam{row["camera_id"]}' / f'{row["target_frame_id"]:06d}'
        truth_assets = _materialize_assets(row['target_rgb'], [row['target_person_mask_A'], row['target_person_mask_B']], truth_dir, target=True)
        truth = bind_supervision_manifest(dict(target_rgb=truth_assets['rgb'], target_masks=truth_assets['masks'],
            **{key: identity[key] for key in ('pair', 'action', 'source_frame', 'target_frame', 'camera')}), generations['short'])
        _truth_board(base['source_rgb'], truth['target_rgb'], sample_dir / 'source_GT_dataset_truth.png', row['sample_id'])
        for variant in ('short', 'structured'):
            split_rows[row['split']][variant].append(generations[variant])
        split_rows[row['split']]['truth'].append(truth)
        split_rows[row['split']]['sample_ids'].append(row['sample_id'])
        records.append(dict(sample_id=row['sample_id'], split=row['split'], **identity,
            source_action=row['source_action'], target_action=row['target_action'],
            original_manifest=row['original_manifest'], original_manifest_line=row['original_manifest_line'],
            original_manifest_sha256=row['original_manifest_sha256'], original_row_sha256=row['original_row_sha256'],
            source_assets=source['assets'], target_truth_assets=truth_assets,
            source_fit=str(source['fit_path']), source_fit_sha256=_sha(source['fit_path']), source_fit_quality=source['fit_meta'],
            target_fit=str(fit_path), target_fit_sha256=_sha(fit_path), target_fit_quality=target_meta,
            source_features=str(source['feature_path']), source_features_sha256=_sha(source['feature_path']),
            source_cache_reused=source['source_cache_reused'], target_geometry=str(geometry_path), target_geometry_sha256=_sha(geometry_path),
            target_geometry_sidecar_sha256=_sha(geometry_path.with_suffix('.json')),
            external_pose_observations=str(pose_path), external_pose_observations_sha256=_sha(pose_path),
            pose_observation_paths=pose_paths, pose_observation_sha256=[_sha(p) for p in pose_paths],
            prompts=prompts, source_GT_board=str(sample_dir / 'source_GT_dataset_truth.png')))
        print(json.dumps(dict(sample_prepared=row['sample_id'], split=row['split'],
            body_joint_mean_mm=target_meta['body_joint_after_mm'], source_cache_reused=source['source_cache_reused'])), flush=True)
    split_manifests = {}
    for split, values in split_rows.items():
        directory = output / split
        directory.mkdir()
        binding = {'sample_ids': values['sample_ids']}
        for variant in ('short', 'structured'):
            path = directory / f'{variant}_generation.json'
            _json(path, values[variant])
            binding[f'{variant}_generation'] = str(path)
            binding[f'{variant}_generation_sha256'] = _sha(path)
        truth_path = directory / 'supervision.json'
        _json(truth_path, values['truth'])
        binding.update(supervision=str(truth_path), supervision_sha256=_sha(truth_path))
        # Exercise the strict typed contract and truth binding on actual files.
        dataset = Hi4DSMPLXDataset(binding['structured_generation'], truth_path)
        for index in range(len(dataset)):
            dataset.generation(index)
            dataset.supervision(index)
        split_manifests[split] = binding
    manifest = dict(schema_version=DATASET_SCHEMA_VERSION, split_manifest=str(split_path),
        split_manifest_sha256=_sha(split_path), splits=split_manifests, selected_samples=records,
        **audit, resolution=512, fit_steps=fit_steps, body_model=str(model_base / 'smplx/SMPLX_NEUTRAL.npz'),
        body_model_sha256=_sha(model_base / 'smplx/SMPLX_NEUTRAL.npz'),
        selection_lock=str(selection_lock), selection_lock_sha256=_sha(selection_lock),
        offline_geometry_preparation=dict(optimized='mesh pose/root/translation and source-only shape',
            SMPLX_model_frozen=True, Qwen_or_adapter_loaded=False, target_RGB_or_true_masks_condition_geometry=False,
            test_model_backpropagation=False, fit_parameters_do_not_update_generation_model=True),
        elapsed_seconds=time.monotonic() - started)
    _json(output / 'dataset.json', manifest)
    return output / 'dataset.json'
