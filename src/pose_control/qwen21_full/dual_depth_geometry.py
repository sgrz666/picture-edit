"""Deterministic source-grid dual geometry, physical ownership and V2 feature reuse."""
from dataclasses import asdict, replace
import torch
from torch.nn import functional as F

from .depth_condition_v2 import DepthConditionV2, DepthFusionContext, CONDITION_SCHEMA as PERSON_SCHEMA, CONTEXT_SCHEMA, FEATURE_SCHEMA, DETAIL_CHANNELS
from .depth_geometry import source_features, onto_canvas, context_to_canvas, camera_rays
from .depth_condition import canvas_identity
from .dual_depth_condition import DualDepthConditionV3, DualContactRelations, CONDITION_SCHEMA


def relative_depth(z, valid):
    out = torch.zeros_like(z)
    for i in range(z.shape[0]):
        values = z[i][valid[i]]
        if values.numel():
            lo, hi = values.min(), values.max()
            out[i] = torch.where(valid[i], (hi-z[i])/(hi-lo) if hi-lo > 1e-8 else .5, 0.)
    return out


def ownership(z, valid, heights, tolerance):
    eps = heights.mean(1)[:, None, None, None] * tolerance
    a, b = z[:, 0], z[:, 1]
    va, vb = valid[:, 0], valid[:, 1]
    overlap = va & vb
    tie = overlap & ((b-a).abs() <= eps)
    front_a = va & (~vb | (a < b-eps))
    front_b = vb & (~va | (b < a-eps))
    visible = torch.stack((front_a | tie, front_b | tie), 1)
    weights = visible.float() / visible.sum(1, keepdim=True).clamp_min(1)
    scene = torch.minimum(torch.where(va, a, torch.inf), torch.where(vb, b, torch.inf))
    return torch.where(va | vb, scene, 0.), va | vb, overlap, tie, torch.stack((front_a, front_b), 1), visible, weights


def person_condition(cache, metadata, spec, person, config, *, physical, relative_override=None, relative_valid_override=None):
    z = torch.from_numpy(cache['z'][person].copy()).float()[None, None]
    valid = torch.from_numpy(cache['valid'][person].copy()).bool()[None, None]
    projected = valid.clone()
    if relative_valid_override is not None:
        valid = valid & relative_valid_override
    normal = torch.from_numpy(cache['normal'][person].copy()).float().permute(2, 0, 1)[None]
    labels = torch.from_numpy(cache['part'][person].copy()).long()[None, None]
    if ((labels < 0) | (labels > 14)).any():
        raise ValueError('Independent Part requires 0..14 labels')
    part = F.one_hot(labels[:, 0], 15)[..., 1:].permute(0, 3, 1, 2).float()
    original_k = torch.from_numpy(cache['K'].copy()).float()[None]
    asset_hw = z.shape[-2:]
    sy, sx = spec.original_hw[0]/asset_hw[0], spec.original_hw[1]/asset_hw[1]
    affine = torch.tensor([[sx, 0., 0.], [0., sy, 0.], [0., 0., 1.]])
    person_id = metadata['person_ids'][person]
    context_meta = {'schema': CONTEXT_SCHEMA, 'canvas_id': canvas_identity(spec.metadata()),
                    'pose_record_id': metadata['record_id']+f':{person}', 'normal_basis': 'camera_cv',
                    'units': metadata['units'], 'scale_status': metadata['scale_status']}
    ctx = DepthFusionContext(original_k, torch.tensor([cache['joints_cam'][person, 0, 2]], dtype=torch.float32),
        torch.tensor([cache['body_height'][person]], dtype=torch.float32), normal, projected.clone(), part,
        (labels > 0) & projected, projected.clone(), torch.tensor([physical]),
        torch.tensor([physical and metadata['units'] == 'metre' and metadata['scale_status'] == 'calibrated']), (context_meta,))
    relative = relative_depth(z, valid) if relative_override is None else torch.where(valid, relative_override, 0.)
    confidence = valid.float()
    ctx_asset = replace(ctx, intrinsics=torch.linalg.inv(affine)[None] @ original_k)
    detail, gv = source_features(relative, valid, z if physical else None, ctx_asset, confidence,
                                 torch.tensor([False]), config, asset_to_original=affine, original_hw=spec.original_hw)
    detail = torch.cat([onto_canvas(detail[:, i:i+1], spec, edge=i in (3, 4, 9)) for i in range(23)], 1)
    ctx_canvas = context_to_canvas(ctx, spec)
    support = onto_canvas(valid.float(), spec, valid=valid) > 0
    support &= ctx_canvas.smpl_mask
    zvalid = (onto_canvas(valid.float(), spec) > .5) & support if physical else torch.zeros_like(support)
    geometry_valid = (onto_canvas(gv.float(), spec) > .5) & zvalid
    rel_canvas = torch.where(support, onto_canvas(relative, spec, valid=valid), 0.).clamp(0, 1)
    for i in (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22):
        detail[:, i:i+1] *= geometry_valid
    detail[:, 10:13] = torch.where(geometry_valid.expand(-1, 3, -1, -1), camera_rays(ctx_canvas.intrinsics, spec.canvas_hw), 0.)
    detail[:, 15:16], detail[:, 16:17] = support.float(), geometry_valid.float()
    detail[:, 17:18], detail[:, 18:19] = support.float(), 0.
    pmeta = {'schema': PERSON_SCHEMA, 'feature_schema': FEATURE_SCHEMA, 'detail_channels': list(DETAIL_CHANNELS),
        'geometry_config': asdict(config), 'kind': 'camera_z' if physical else 'relative_png', 'polarity': 'near_one',
        'source_id': metadata['record_id']+f':{person}', 'canvas_id': canvas_identity(spec.metadata()),
        'pose_canvas_id': spec.fingerprint(), 'pose_record_id': metadata['record_id']+f':{person}',
        'pose_record_sha256': metadata['record_sha256'], 'person_id': person_id,
        'scale_group_id': metadata['scale_group_id'], 'camera_id': metadata['camera_id']}
    condition = DepthConditionV2(rel_canvas, support, spec.valid_mask()[None].bool(), support.flatten(1).any(1),
        torch.where(zvalid, onto_canvas(z, spec), 0.) if physical else None, zvalid, geometry_valid,
        support.float(), torch.tensor([False]), ctx_canvas, detail, (pmeta,))
    return condition.validate(target_token_hw=spec.token_hw)


def build_dual_condition(cache, metadata, spec, config, *, mode='full_geometry', scene_png=None, scene_png_valid=None):
    physical = mode == 'full_geometry'
    if mode not in ('full_geometry', 'scene_only'):
        raise ValueError('Explicit data modality required')
    if not physical and (scene_png is None or scene_png_valid is None):
        raise ValueError('scene_only requires explicit PNG AND validity; no asset guessing')
    people = [person_condition(cache, metadata, spec, p, config.depth_config().geometry_config(), physical=physical,
                               relative_override=scene_png if not physical else None,
                               relative_valid_override=scene_png_valid if not physical else None) for p in range(2)]
    z_native = torch.from_numpy(cache['z'].copy()).float()[None, :, None]
    v_native = torch.from_numpy(cache['valid'].copy()).bool()[None, :, None]
    heights = torch.from_numpy(cache['body_height'].copy()).float()[None]
    roots = torch.from_numpy(cache['joints_cam'][:, 0, 2].copy()).float()[None]
    scene, valid, overlap, tie, front, visible, weights = ownership(z_native, v_native, heights, config.owner_tolerance)
    center, scale = roots.mean(1)[:, None, None, None], heights.mean(1)[:, None, None, None]
    raster = torch.cat((torch.where(v_native[:, 0], torch.asinh((z_native[:, 0]-center)/scale), 0.),
                        torch.where(v_native[:, 1], torch.asinh((z_native[:, 1]-center)/scale), 0.),
                        v_native[:, 0].float(), v_native[:, 1].float(), overlap.float(),
                        torch.where(overlap, torch.asinh((z_native[:, 1]-z_native[:, 0])/scale), 0.),
                        front[:, 0].float(), front[:, 1].float()), 1)
    project = torch.stack([p.context.smpl_mask for p in people], 1)
    def warp_person(value):
        return torch.stack([onto_canvas(value[:, p].float(), spec) for p in range(2)], 1)
    visible_c = warp_person(visible) > .5
    front_c = warp_person(front) > .5
    tie_c = onto_canvas(tie.float(), spec) > .5
    weights_c = warp_person(weights)
    scene_valid = onto_canvas(valid.float(), spec) > .5
    if physical:
        rel = relative_depth(scene, valid)
        scene_depth = torch.where(scene_valid, onto_canvas(rel, spec, valid=valid), 0.)
    else:
        scene_valid = (onto_canvas(scene_png_valid.float(), spec) > .5) & project.any(1)
        scene_depth = torch.where(scene_valid, onto_canvas(scene_png, spec, valid=scene_png_valid), 0.).clamp(0, 1)
        visible_c &= scene_valid[:, None]
        front_c &= scene_valid[:, None]
        tie_c &= scene_valid
        # Preserve geometric source/target ownership, but do not expose physical Z channels.
        weights_c *= scene_valid[:, None]
    distance = torch.from_numpy(cache['part_distance'].copy()).float()[None]
    dv = torch.from_numpy(cache['part_distance_valid'].copy()).bool()[None]
    witness = torch.from_numpy(cache['closest_points'].copy()).float()[None]
    wv = torch.from_numpy(cache['closest_points_valid'].copy()).bool()[None]
    contact = torch.from_numpy(cache['part_contact'].copy()).float()[None]
    indices = torch.from_numpy(cache['contact_indices'].copy()).long()[None]
    cd = torch.from_numpy(cache['contact_distance'].copy()).float()[None]
    cv = torch.ones(cd.shape, dtype=torch.bool)
    if not physical:
        raster.zero_(); distance.zero_(); dv.zero_(); witness.zero_(); wv.zero_(); contact.zero_(); cv.zero_(); cd.zero_()
    contacts = DualContactRelations(indices, torch.zeros(cd.shape, dtype=torch.long), cd, cv,
        ({'source': 'official_hi4d_correspondence', 'index_base': 1, 'zero': 'no_record'},))
    meta = {**metadata, 'schema': CONDITION_SCHEMA, 'mode': mode, 'canvas_id': spec.fingerprint(),
            'geometry_config': asdict(config), 'normal_basis': 'camera_cv'}
    return DualDepthConditionV3(*people, scene_depth, scene_valid, project, visible_c, weights_c,
        onto_canvas(raster, spec), onto_canvas(overlap.float(), spec) > .5, front_c,
        tie_c, distance, dv, witness, wv, contact, contacts,
        torch.tensor([metadata['model_family_indices']], dtype=torch.long), (meta,)).validate(target_token_hw=spec.token_hw)


def relative11(joints, root_rotations, heights, boxes):
    scale = heights.mean()
    delta = (joints[1, 0]-joints[0, 0])/scale
    center = (boxes[:, :2]+boxes[:, 2:])/2
    size = (boxes[:, 2:]-boxes[:, :2]).clamp_min(1e-6)
    r = root_rotations[0].T @ root_rotations[1]
    direction = torch.stack((r[0, 2], r[2, 2]))
    valid = direction.norm() > 1e-6
    sc = direction/direction.norm().clamp_min(1e-6) if valid else torch.zeros_like(direction)
    return torch.cat((delta, center[1]-center[0], (size[1]/size[0]).log(), delta.norm()[None], sc, delta[2].sign()[None])), bool(valid)
