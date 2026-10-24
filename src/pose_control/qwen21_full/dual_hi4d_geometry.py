"""Offline Hi4D SMPL/FCL/raster tools. Heavy dependencies are lazy and export-only."""
from pathlib import Path
import numpy as np
import torch

from .body_pose import JOINT_NAMES, PARENTS_22, axis_angle_to_matrix, matrix_to_rotation_6d

SMPL_JOINTS = ('pelvis', *JOINT_NAMES, 'left_hand', 'right_hand')
SMPL_TO_PART = np.array([1, 8, 11, 1, 9, 12, 1, 10, 13, 1, 10, 13,
                         0, 1, 1, 0, 2, 5, 3, 6, 4, 7, 4, 7], np.int64)
CACHE_SCHEMA = 'native_dual_geometry_cache_v3'


def decode_contact(contact, verts, scene_height):
    """Hi4D: 0=no correspondence; k>0 -> opponent vertex k-1. No NN replacement."""
    if contact.shape != (2, verts.shape[1]) or not np.isfinite(contact).all() or not np.array_equal(contact, np.round(contact)):
        raise ValueError('Official contact must be integral [2,V] correspondences')
    if np.any(contact < 0) or np.any(contact > verts.shape[1]):
        raise ValueError('Official one-based contact index outside mesh')
    records, distances = [], []
    for p in range(2):
        src = np.flatnonzero(contact[p] > 0)
        dst = contact[p, src].astype(np.int64)-1
        records.extend((p, int(a), 1-p, int(b)) for a, b in zip(src, dst))
        distances.extend((np.linalg.norm(verts[p, src]-verts[1-p, dst], axis=1)/scene_height).tolist())
    return np.asarray(records, np.int64).reshape(-1, 4), np.asarray(distances, np.float32)


def face_partition(lbs_weights, faces):
    """Disjoint face assignment, identical for raster and surface distance meshes."""
    if lbs_weights.shape[1] != 24:
        raise ValueError('SMPL24 skinning topology required, not SMPL-X vertex IDs')
    weights = np.zeros((lbs_weights.shape[0], 14), np.float64)
    for joint, part in enumerate(SMPL_TO_PART):
        weights[:, part] += lbs_weights[:, joint]
    vertex_part = weights.argmax(1)
    return weights[faces].sum(1).argmax(1).astype(np.int64), vertex_part


def body21_from_smpl(params, meta, person, model_path, extr, k, original_hw):
    import smplx
    path = Path(model_path)
    gender = str(meta['genders'][person])
    if gender not in ('male', 'female', 'neutral') or path.name != f'SMPL_{gender.upper()}.pkl':
        raise ValueError('Require declared gender-specific SMPL model; no neutral fallback')
    model = smplx.SMPL(str(path), gender=gender, batch_size=1).float().eval()
    if tuple(model.parents[:22].tolist()) != PARENTS_22 or params['body_pose'].shape != (2, 69):
        raise ValueError('Unexpected SMPL joint tree/body69 schema')
    aa = torch.as_tensor(params['body_pose'][person], dtype=torch.float32).reshape(1, 23, 3)
    kwargs = {name: torch.as_tensor(params[name][person], dtype=torch.float32).reshape(1, -1)
              for name in ('body_pose', 'global_orient', 'transl', 'betas')}
    with torch.no_grad():
        posed = model(**kwargs)
        rest = model(betas=kwargs['betas'], body_pose=torch.zeros_like(kwargs['body_pose']),
                     global_orient=torch.zeros(1, 3), transl=torch.zeros(1, 3))
    mesh = posed.vertices[0].cpu().numpy()
    error = float(np.max(np.abs(mesh-params['verts'][person])))
    joint_error = float(np.max(np.abs(posed.joints[0, :24].cpu().numpy()-params['joints_3d'][person])))
    if not np.isfinite(error) or max(error, joint_error) > 1e-4:
        raise ValueError(f'SMPL reconstruction differs: mesh={error}, joints={joint_error}; do not substitute a model')
    r = torch.as_tensor(extr[:, :3], dtype=torch.float32)
    t = torch.as_tensor(extr[:, 3], dtype=torch.float32)
    root = axis_angle_to_matrix(kwargs['global_orient'][0])
    pelvis_rest = rest.joints[0, 0]
    translation = r @ kwargs['transl'][0] + t + (r-torch.eye(3)) @ pelvis_rest
    root_camera = r @ root
    joints = posed.joints[0, :22] @ r.T + t
    mesh_cam = mesh @ extr[:, :3].T + extr[:, 3]
    # Full local rotations are needed when composing a matrix root.
    with torch.no_grad():
        transformed = model(betas=kwargs['betas'], body_pose=axis_angle_to_matrix(aa),
                            global_orient=root_camera[None, None], transl=translation[None], pose2rot=False)
    camera_error = float(np.max(np.abs(transformed.vertices[0].cpu().numpy()-mesh_cam)))
    if camera_error > 1e-4:
        raise ValueError('Pelvis-centered camera transform is inconsistent with the mesh')
    h, w = original_hw
    camera4 = torch.tensor([k[0, 0]/w, k[1, 1]/h, k[0, 2]/w, k[1, 2]/h], dtype=torch.float32)
    global26 = torch.cat((kwargs['betas'][0], matrix_to_rotation_6d(root_camera), translation, camera4, translation))
    height = float(torch.ptp(rest.vertices[0, :, 1])) if hasattr(torch, 'ptp') else float(rest.vertices[0, :, 1].max()-rest.vertices[0, :, 1].min())
    if height <= 0 or not torch.isfinite(global26).all():
        raise ValueError('Invalid canonical height/Global26')
    indices = [SMPL_JOINTS.index(name)-1 for name in JOINT_NAMES]
    faces = model.faces.astype(np.int32)
    face_part, vertex_part = face_partition(model.lbs_weights.cpu().numpy(), faces)
    return {'body_pose': aa[0, indices].reshape(63).numpy(), 'joints_cam': joints.numpy(),
            'global26': global26.numpy(), 'root_rotation': root_camera.numpy(), 'body_height': height,
            'mesh_cam': mesh_cam.astype(np.float32), 'faces': faces, 'face_part': face_part,
            'vertex_part': vertex_part, 'mesh_error': error, 'joint_error': joint_error,
            'camera_mesh_error': camera_error, 'gender': gender, 'model_family': 'smpl'}


def part_distances(people):
    """Exact triangle-surface distance via FCL; never silently substitute vertex NN."""
    import fcl
    models = []
    for p in people:
        row = []
        for part in range(14):
            faces = p['faces'][p['face_part'] == part]
            if not len(faces):
                row.append(None)
                continue
            mesh = fcl.BVHModel()
            mesh.beginModel(len(p['mesh_cam']), len(faces))
            mesh.addSubModel(p['mesh_cam'].astype(np.float64), faces)
            mesh.endModel()
            row.append(fcl.CollisionObject(mesh, fcl.Transform()))
        models.append(row)
    distance = np.zeros((14, 14), np.float32)
    points = np.zeros((14, 14, 2, 3), np.float32)
    valid = np.zeros((14, 14), bool)
    witness_valid = np.zeros_like(valid)
    intersect = np.zeros_like(valid)
    scale = np.mean([p['body_height'] for p in people])
    for a in range(14):
        for b in range(14):
            if models[0][a] is None or models[1][b] is None:
                continue
            request = fcl.DistanceRequest(enable_nearest_points=True)
            result = fcl.DistanceResult()
            value = fcl.distance(models[0][a], models[1][b], request, result)
            if value < 0:
                collision = fcl.CollisionResult()
                fcl.collide(models[0][a], models[1][b], fcl.CollisionRequest(), collision)
                if not collision.is_collision:
                    raise ValueError('FCL negative distance without a confirmed surface intersection')
                intersect[a, b] = True
                # FCL may omit distance witnesses for intersecting meshes. They are
                # diagnostic only; do not fabricate a contact witness or direction.
                valid[a, b] = True
                continue
            witness = np.asarray(result.nearest_points, np.float64)
            if witness.shape != (2, 3) or not np.isfinite(witness).all() or not np.isfinite(value):
                raise ValueError('FCL did not return finite surface witnesses')
            distance[a, b], points[a, b], valid[a, b] = value/scale, witness, True
            witness_valid[a, b] = True
    return distance, points, valid, witness_valid, intersect


class IndependentMeshRasterizer:
    def __init__(self, device='cuda'):
        import nvdiffrast.torch as dr
        self.dr, self.device = dr, torch.device(device)
        self.context = dr.RasterizeCudaContext(device=self.device)

    def render(self, person, k, hw):
        h, w = hw
        vertices = person['mesh_cam']
        if not np.isfinite(vertices).all() or (vertices[:, 2] <= 0).any():
            raise ValueError('Mesh must be finite and in front of the shared CV camera')
        v = torch.as_tensor(vertices, dtype=torch.float32, device=self.device)
        faces = torch.as_tensor(person['faces'], dtype=torch.int32, device=self.device)
        uv = v @ torch.as_tensor(k, device=self.device, dtype=torch.float32).T
        u, y = uv[:, 0]/uv[:, 2], uv[:, 1]/uv[:, 2]
        z = v[:, 2]
        near, far = max(float(z.min())*.5, 1e-5), float(z.max())*2
        clip_z = (far+near)/(far-near)*z - 2*far*near/(far-near)
        clip = torch.stack(((2*u/w-1)*z, (1-2*y/h)*z, clip_z, z), -1)[None].contiguous()
        raster, _ = self.dr.rasterize(self.context, clip, faces, resolution=(h, w))
        position, _ = self.dr.interpolate(v[None].contiguous(), raster, faces)
        # Smooth CV normals; a single, explicit bottom-up -> top-down transform.
        f = faces.long()
        fn = torch.cross(v[f[:, 1]]-v[f[:, 0]], v[f[:, 2]]-v[f[:, 0]], dim=-1)
        vn = torch.zeros_like(v)
        for corner in range(3):
            vn.index_add_(0, f[:, corner], fn)
        vn = torch.nn.functional.normalize(vn, dim=-1, eps=1e-12)
        normal, _ = self.dr.interpolate(vn[None].contiguous(), raster, faces)
        normal = torch.nn.functional.normalize(normal[0], dim=-1, eps=1e-12).flip(0)
        triangle = raster[0, :, :, 3].long().flip(0)-1
        valid = triangle >= 0
        labels = torch.as_tensor(person['face_part'], device=self.device)[triangle.clamp_min(0)] + 1
        depth = position[0, :, :, 2].flip(0)
        return {'z': torch.where(valid, depth, 0.).cpu().numpy(), 'valid': valid.cpu().numpy(),
                'normal': torch.where(valid[..., None], normal, 0.).cpu().numpy(),
                'part': torch.where(valid, labels, 0).cpu().numpy().astype(np.uint8)}
