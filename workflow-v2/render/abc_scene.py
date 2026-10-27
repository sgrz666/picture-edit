"""Render retargeted SMPL-X meshes through one CV camera, without RGB surrogates.

CUDA dependencies load only inside ``render_scene``. The CPU helpers retain the
same evidence rules in tests and export: occluded masks remain available, contact
requires explicit vertex pairs, and no person rectangle is used as geometry.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import ImageDraw


_PART_COLORS = np.asarray([[0, 0, 0], [255, 110, 110], [255, 184, 80],
    [236, 225, 75], [151, 220, 80], [77, 203, 114], [55, 200, 190],
    [55, 149, 225], [106, 107, 225], [163, 87, 217], [214, 77, 184],
    [229, 108, 150], [183, 160, 105], [115, 168, 160], [170, 170, 170]], np.uint8)


def stick_contact_image(people,size,*,endpoints=None,contact_pairs=(),person_slots=None):
    """Anatomical body22 sticks and explicit contact marks; no inferred fingers."""
    from src.pose_control.qwen21_full.body_pose import PARENTS_22
    image=Image.new('RGB',tuple(size),'black');draw=ImageDraw.Draw(image)
    width=max(2,round(min(size)*.008));radius=max(2,width)
    colors=((55,200,255),(255,170,65));slots=person_slots or list(range(len(people)))
    for pair in contact_pairs:
        pair=np.asarray(pair)
        if pair.shape==(2,2) and np.isfinite(pair).all():draw.line([tuple(p) for p in pair],fill=(120,255,90),width=max(2,width//2))
    for index,joints in enumerate(people):
        joints=np.asarray(joints)
        if joints.shape!=(22,2):raise ValueError('Stick-contact guide requires body22 projected joints')
        color=colors[slots[index]%2]
        for joint,parent in enumerate(PARENTS_22):
            if parent>=0 and np.isfinite(joints[[parent,joint]]).all():
                draw.line([tuple(joints[parent]),tuple(joints[joint])],fill=color,width=width)
        for x,y in joints:
            if np.isfinite([x,y]).all():draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill=color)
        if endpoints is not None:
            for x,y in np.asarray(endpoints[index]).reshape(-1,2):
                if np.isfinite([x,y]).all():draw.ellipse((x-radius-2,y-radius-2,x+radius+2,y+radius+2),fill=(120,255,90))
    return image


def project_points(points, k):
    """Exact shared pinhole projection; invalid/behind-camera points become NaN."""
    points, k = np.asarray(points, np.float64), np.asarray(k, np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or k.shape != (3, 3):
        raise ValueError('Require points [J,3] and K [3,3]')
    if not np.isfinite(k).all() or not np.allclose(k[2], [0., 0., 1.]):
        raise ValueError('K must be a finite CV pinhole matrix')
    if k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError('K focal lengths must be positive')
    valid = np.isfinite(points).all(1) & (points[:, 2] > 0)
    uv = np.full((len(points), 2), np.nan)
    homogeneous = points[valid] @ k.T
    uv[valid] = homogeneous[:, :2] / homogeneous[:, 2, None]
    return uv


def visible_ownership(depth, mask):
    """One nearest visible owner per pixel; equal z has deterministic slot-0 priority."""
    depth, mask = np.asarray(depth), np.asarray(mask, bool)
    if depth.ndim != 3 or depth.shape[0] != 2 or depth.shape != mask.shape:
        raise ValueError('depth and mask must be matching [2,H,W]')
    if not np.isfinite(depth[mask]).all() or (depth[mask] <= 0).any():
        raise ValueError('Covered mesh depths must be finite and positive')
    nearest = np.where(mask, depth, np.inf).argmin(0)
    covered = mask.any(0)
    return np.stack([covered & (nearest == p) for p in range(2)])


def face_labels(faces, vertex_labels):
    """Majority label over each actual triangle; ties use the lowest canonical ID."""
    faces, labels = np.asarray(faces), np.asarray(vertex_labels)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in 'iu':
        raise ValueError('Require integral triangle faces [F,3]')
    if labels.ndim != 1 or labels.dtype.kind not in 'iu' or not len(labels):
        raise ValueError('Require integral vertex part labels 0..13')
    if (labels < 0).any() or (labels > 13).any():
        raise ValueError('Vertex part labels must be 0..13, before adding background')
    if not len(faces) or faces.min() < 0 or faces.max() >= len(labels):
        raise ValueError('Face vertex index outside mesh')
    counts = np.eye(14, dtype=np.int32)[labels[faces]].sum(1)
    return counts.argmax(1).astype(np.int64)


def contact_evidence(people, correspondences=None):
    """Measure known vertex correspondences; absent evidence stays unavailable."""
    empty = {'status': 'unavailable', 'indices': np.empty((0, 4), np.int64),
        'distance_m': np.empty((0,), np.float32), 'points_cam': np.empty((0, 2, 3), np.float32)}
    if correspondences is None:
        return empty
    raw = np.asarray(correspondences)
    if raw.size == 0:
        return empty
    if raw.ndim != 2 or raw.shape[1] != 4 or not np.isfinite(raw).all() or not np.array_equal(raw, np.round(raw)):
        raise ValueError('Contact requires integral [N,4] vertex correspondences')
    indices = raw.astype(np.int64)
    if np.any(indices[:, [0, 2]] < 0) or np.any(indices[:, [0, 2]] > 1):
        raise ValueError('Contact person indices must be 0 or 1')
    if np.any(indices[:, 0] == indices[:, 2]):
        raise ValueError('Contact must link different people')
    points = []
    for p, a, q, b in indices:
        va = np.asarray(people[p].get('vertices_cam', people[p].get('mesh_cam')))
        vb = np.asarray(people[q].get('vertices_cam', people[q].get('mesh_cam')))
        if min(a, b) < 0 or a >= len(va) or b >= len(vb):
            raise ValueError('Contact vertex index outside mesh')
        points.append([va[a], vb[b]])
    points = np.asarray(points, np.float32)
    if not np.isfinite(points).all():
        raise ValueError('Contact vertices must be finite')
    return {'status': 'explicit_vertex_correspondences', 'indices': indices,
        'distance_m': np.linalg.norm(points[:, 0] - points[:, 1], axis=-1).astype(np.float32),
        'points_cam': points}


def _person_for_raster(person):
    vertices = np.asarray(person.get('vertices_cam', person.get('mesh_cam')), np.float32)
    faces = np.asarray(person['faces'])
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all() or np.any(vertices[:, 2] <= 0):
        raise ValueError('Finite camera-space mesh [V,3] with positive z required')
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in 'iu' or not len(faces):
        raise ValueError('Integral triangle mesh [F,3] required')
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('Face vertex index outside mesh')
    labels = person.get('part_face', person.get('face_part'))
    if labels is None:
        labels = face_labels(faces, person.get('part_vertex', person.get('vertex_part')))
    labels = np.asarray(labels)
    if labels.shape != (len(faces),) or labels.dtype.kind not in 'iu' or (labels < 0).any() or (labels > 13).any():
        raise ValueError('Require one integral 0..13 part label per real triangle')
    return {'mesh_cam': vertices, 'faces': faces.astype(np.int32), 'face_part': labels.astype(np.int64)}


def _contact_raster(evidence, k, hw, sigma=4.):
    h, w = hw
    result = np.zeros((2, h, w), np.float32)
    for index, points in zip(evidence['indices'], evidence['points_cam']):
        for slot, point in zip((index[0], index[2]), points):
            uv = project_points(point[None], k)[0]
            if not np.isfinite(uv).all():
                continue
            u, v = uv
            x0, x1 = max(0, int(np.floor(u - 3 * sigma))), min(w, int(np.ceil(u + 3 * sigma)) + 1)
            y0, y1 = max(0, int(np.floor(v - 3 * sigma))), min(h, int(np.ceil(v + 3 * sigma)) + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            patch = np.exp(-((xx + .5 - u) ** 2 + (yy + .5 - v) ** 2) / (2 * sigma ** 2))
            result[slot, y0:y1, x0:x1] = np.maximum(result[slot, y0:y1, x0:x1], patch)
    return result


def _depth_image(depth, mask, limits):
    lo, hi = limits
    value = np.zeros(depth.shape, np.uint8)
    value[mask] = np.round(255 * (1 - (depth[mask] - lo) / max(hi - lo, 1e-6))).clip(1, 255).astype(np.uint8)
    return value


def render_scene(people, K, canvas_hw, out_dir, *, device='cuda', rasterizer=None, contact_indices=None,guide_style='dwpose'):
    """Export two dense independent meshes and their shared z composition.

    ``rasterizer`` is injectable for CPU export tests; normal execution always
    instantiates the repository's nvdiffrast triangle rasterizer. All part labels
    are zero-based V6 IDs before rasterization; saved maps use 0 background/1..14.
    ``joints_cam`` is retained verbatim. Guides require the estimator's explicit
    137-landmark order, either in that field or in ``joints_137_cam``.
    """
    if len(people) != 2:
        raise ValueError('ABC scene requires exactly two logical A/B targets')
    hw = tuple(int(v) for v in canvas_hw)
    if len(hw) != 2 or min(hw) <= 0 or any(v % 32 for v in hw):
        raise ValueError('Canvas must be positive H,W multiples of 32')
    k = np.asarray(K, np.float64)
    project_points(np.array([[0., 0., 1.]]), k)
    prepared = [_person_for_raster(p) for p in people]
    packed = {}
    for key, tail in (('body_pose', 63), ('global_orient', 3), ('transl', 3), ('betas', 10)):
        values = [np.asarray(p[key], np.float32).reshape(-1) for p in people]
        if any(v.shape != (tail,) or not np.isfinite(v).all() for v in values):
            raise ValueError(f'Each target requires finite {key}[{tail}]')
        packed[key] = np.stack(values)
    joints = [np.asarray(p['joints_cam'], np.float32) for p in people]
    if any(j.ndim != 2 or j.shape[1] != 3 or not np.isfinite(j).all() for j in joints) or joints[0].shape != joints[1].shape:
        raise ValueError('Targets must have matching finite joints_cam[J,3]')
    guide_joints = [np.asarray(p.get('joints_137_cam', p['joints_cam']), np.float32) for p in people]
    if any(j.shape != (137, 3) or not np.isfinite(j).all() for j in guide_joints):
        raise ValueError('Guide requires actual SMPLest-X joints_137_cam[137,3], never raw joint IDs')
    contact = contact_evidence(people, contact_indices)
    if guide_style not in ('dwpose','hi4d_stick_contact'):raise ValueError('Unknown pose guide style')
    output = Path(out_dir).resolve()
    expected = [output / 'geometry_arrays.npz', output / 'render_manifest.json']
    expected += [output / f'{slot}_{kind}.png' for slot in ('person_0', 'person_1', 'combined')
        for kind in ('guide', 'depth', 'normal', 'part', 'mask', 'owner', 'contact')]
    if any(p.exists() for p in expected):
        raise FileExistsError(f'Render outputs already exist under {output}; use a fresh run directory')
    output.mkdir(parents=True, exist_ok=True)
    if rasterizer is None:
        from src.pose_control.qwen21_full.dual_hi4d_geometry import IndependentMeshRasterizer
        rasterizer = IndependentMeshRasterizer(device=device)
    raw = [rasterizer.render(p, k, hw) for p in prepared]
    depth = np.stack([r['z'] for r in raw]).astype(np.float32)
    mask = np.stack([r['valid'] for r in raw]).astype(bool)
    normal = np.stack([r['normal'] for r in raw]).astype(np.float32)
    part = np.stack([r['part'] for r in raw]).astype(np.uint8)
    if depth.shape != (2, *hw) or normal.shape != (2, *hw, 3) or part.shape != mask.shape or mask.shape != depth.shape:
        raise ValueError('Triangle raster returned incompatible dense array shapes')
    if not np.isfinite(normal).all() or (part > 14).any() or not np.array_equal(part > 0, mask):
        raise ValueError('Triangle raster returned invalid normals/part IDs/masks')
    if not mask.reshape(2, -1).any(1).all():
        raise ValueError('Both target meshes must cover pixels on the shared canvas')
    owner = visible_ownership(depth, mask)
    contact_map = _contact_raster(contact, k, hw)
    scene_mask = mask.any(0)
    scene_depth = np.sum(depth * owner, axis=0)
    scene_normal = np.sum(normal * owner[..., None], axis=0)
    scene_part = np.sum(part * owner, axis=0).astype(np.uint8)
    packed.update(normal_cam=normal, depth=depth, part=part, mask=mask,
        visible_owner=owner, joints_cam=np.stack(joints), joints_137_cam=np.stack(guide_joints),
        K=k.astype(np.float32), contact_indices=contact['indices'], contact_distance_m=contact['distance_m'],
        contact_points_cam=contact['points_cam'], contact_raster=contact_map,
        combined_depth=scene_depth, combined_normal_cam=scene_normal, combined_part=scene_part)
    packed['contact_parts']=np.asarray([[people[int(p)]['part_vertex'][int(a)],people[int(q)]['part_vertex'][int(b)]]
        if 'part_vertex' in people[int(p)] and 'part_vertex' in people[int(q)] else [-1,-1]
        for p,a,q,b in contact['indices']],np.int64).reshape(-1,2)
    arrays_path = output / 'geometry_arrays.npz'
    np.savez_compressed(arrays_path, **packed)
    from src.pose_edit.geometry import render_poseguide, smplx137_to_dwpose134
    guides, guide_images, records = {}, [], {}
    all_depth = depth[mask]
    limits = (float(all_depth.min()), float(all_depth.max()))
    for p in range(2):
        slot = f'person_{p}'
        uv = project_points(guide_joints[p], k)
        if guide_style=='hi4d_stick_contact':
            endpoints=contact['points_cam'][:,p] if len(contact['indices']) else np.empty((0,3))
            guide=stick_contact_image([project_points(joints[p][:22],k)],(hw[1],hw[0]),
                endpoints=[project_points(endpoints,k)],person_slots=[p])
        else:guide = render_poseguide(smplx137_to_dwpose134(uv), (hw[1], hw[0]))
        guide_images.append(np.asarray(guide))
        path = output / f'{slot}_guide.png'; guide.save(path); guides[slot] = str(path)
        records[slot] = _write_renders(output, slot, depth[p], normal[p], part[p], mask[p], owner[p], contact_map[p], limits)
    combined_guide = output / 'combined_guide.png'
    if guide_style=='hi4d_stick_contact':
        endpoints=[project_points(contact['points_cam'][:,p],k) if len(contact['indices']) else np.empty((0,2)) for p in range(2)]
        stick_contact_image([project_points(j[:22],k) for j in joints],(hw[1],hw[0]),endpoints=endpoints,
            contact_pairs=[project_points(points,k) for points in contact['points_cam']]).save(combined_guide)
    else:Image.fromarray(np.maximum(guide_images[0], guide_images[1])).save(combined_guide)
    guides['combined'] = str(combined_guide)
    owner_colors = np.zeros((*hw, 3), np.uint8)
    owner_colors[owner[0]] = [255, 105, 90]; owner_colors[owner[1]] = [55, 155, 255]
    records['combined'] = _write_renders(output, 'combined', scene_depth, scene_normal, scene_part,
        scene_mask, scene_mask, contact_map.max(0), limits, owner_colors=owner_colors)
    diagnostics = {'renderer': 'nvdiffrast_independent_triangle_shared_cv_camera',
        'normal_basis': 'camera_cv', 'unit': 'm', 'part_encoding': '0_background_1_to_14_V6',
        'depth_visualization_range_m': list(limits), 'contact_status': contact['status'],
        'contact_pair_count': int(len(contact['indices'])),
        'contact_distance_min_m': float(contact['distance_m'].min()) if len(contact['indices']) else None,
        'contact_distance_max_m': float(contact['distance_m'].max()) if len(contact['indices']) else None,
        'independent_mask_pixels': [int(m.sum()) for m in mask],
        'visible_pixels': [int(m.sum()) for m in owner], 'overlap_pixels': int(np.all(mask, axis=0).sum()),
        'joint_order_guide': 'SMPLest-X joint_idx137 -> DWPose134',
        'part_labels_source': 'explicit_SMPLX_triangle_partition',
        'owner_depth_tie_policy': 'logical_A_slot_0'}
    diagnostics['guide_style']=guide_style
    diagnostics['guide_body_source']='retargeted_official_Hi4D_body22' if guide_style=='hi4d_stick_contact' else 'SMPLest-X137_to_DWPose134'
    if device == 'cpu' or rasterizer.__class__.__name__ != 'IndependentMeshRasterizer':
        diagnostics['renderer'] = f'injected:{rasterizer.__class__.__name__}'
    result = {'schema': 'abc_shared_camera_render_v1', 'geometry_arrays': str(arrays_path),
        'guides': guides, 'renders': records, 'diagnostics': diagnostics}
    (output / 'render_manifest.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return result


def _write_renders(output, slot, depth, normal, part, mask, owner, contact, limits, owner_colors=None):
    normal_rgb = np.where(mask[..., None], (normal * .5 + .5) * 255, 0).clip(0, 255).astype(np.uint8)
    images = {'depth': _depth_image(depth, mask, limits), 'normal': normal_rgb,
        'part': _PART_COLORS[part], 'mask': mask.astype(np.uint8) * 255,
        'owner': owner.astype(np.uint8) * 255 if owner_colors is None else owner_colors,
        'contact': (contact * 255).clip(0, 255).astype(np.uint8)}
    paths = {}
    for name, pixels in images.items():
        path = output / f'{slot}_{name}.png'
        Image.fromarray(pixels).save(path)
        paths[name] = str(path)
    return paths
