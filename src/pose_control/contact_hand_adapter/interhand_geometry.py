"""Optional CPU reconstruction and joint z-buffer of official MANO NeuralAnnot fits."""

from dataclasses import replace
import pickle

import numpy as np
import torch

from src.pose_edit.common import sha256_file
from src.pose_control.qwen21_pro.geometry import LiteralChumpyArray, _LiteralSMPLUnpickler

from .conditions import HandMesh, HandPair, UNKNOWN
from .interhand import HANDS, JOINT_ORDER


# MANO articulation: wrist, index, middle, pinky, ring, thumb. In the
# Adapter's wrist/thumb/index/middle/ring/pinky order, fingertips have no rotation.
ARTICULATED = np.array([0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19])
MANO_JOINTS = np.array([0, 13, 14, 15, 1, 2, 3, 4, 5, 6, 10, 11, 12, 7, 8, 9])
MANO_TO_PART = np.argsort(MANO_JOINTS)
SURFACE_MAX_M = .025
SURFACE_SINGLE_VIEW_PRIOR = .35
SURFACE_GAP_TOLERANCE_M = .001
SURFACE_GAP_SCALE_M = .004


class _SelectedShapeDirections:
    """Read the MANO release's indexed Chumpy shapedirs without executing Chumpy."""

    def __setstate__(self, state):
        allowed = {"a", "preferred_shape", "idxs", "_dirty_vars", "_itr", "_depends_on_deps"}
        indexes = state.get("idxs") if isinstance(state, dict) else None
        if (not isinstance(state, dict) or set(state) - allowed
                or type(state.get("a")) is not LiteralChumpyArray
                or state.get("preferred_shape") != (778, 3, 10)
                or not isinstance(indexes, np.ndarray) or indexes.shape != (778 * 3 * 10,)
                or not np.issubdtype(indexes.dtype, np.integer)
                or state.get("_dirty_vars", set()) not in (set(), {"a", "idxs", "preferred_shape"})
                or state.get("_itr") is not None or state.get("_depends_on_deps", {}) != {}):
            raise pickle.UnpicklingError("Unexpected MANO shape direction selector")
        source = state["a"].array()
        if source.shape != (778, 3, 20):
            raise pickle.UnpicklingError("Unexpected MANO shape direction source")
        official_indexes = np.arange(source.size).reshape(source.shape)[:, :, :10].ravel()
        if not np.array_equal(indexes, official_indexes):
            raise pickle.UnpicklingError("Unexpected MANO shape direction indices")
        self._value = source[:, :, :10].copy()

    def array(self):
        return self._value


class _LiteralManoUnpickler(_LiteralSMPLUnpickler):
    def find_class(self, module, name):
        if (module, name) == ("chumpy.reordering", "Select"):
            return _SelectedShapeDirections
        return super().find_class(module, name)


def _load_mano_model(path):
    from scipy.sparse import csc_matrix

    with path.open("rb") as handle:
        model = _LiteralManoUnpickler(handle, encoding="latin1").load()
    required = {"f": (1538, 3), "J_regressor": (16, 778), "kintree_table": (2, 16), "weights": (778, 16),
                "posedirs": (778, 3, 135), "shapedirs": (778, 3, 10), "v_template": (778, 3),
                "hands_mean": (45,), "hands_components": (45, 45)}
    if not isinstance(model, dict):
        raise ValueError("MANO data must be a dictionary")
    data = {}
    for name, shape in required.items():
        value = model[name]
        if name == "shapedirs" and isinstance(value, (LiteralChumpyArray, _SelectedShapeDirections)):
            value = value.array()
        if name == "J_regressor" and isinstance(value, csc_matrix):
            value = value.toarray()
        if (not isinstance(value, np.ndarray) or value.shape != shape or not np.issubdtype(value.dtype, np.number)
                or not np.isfinite(value).all()):
            raise ValueError(f"MANO {name} must be finite numeric data with shape {shape}")
        data[name] = value
    faces = data["f"]
    if not np.issubdtype(faces.dtype, np.integer) or faces.min() < 0 or faces.max() >= 778:
        raise ValueError("Invalid MANO face indices")
    weights = data["weights"]
    if (weights < 0).any() or not np.allclose(weights.sum(1), 1, atol=1e-4):
        raise ValueError("Invalid MANO skinning weights")
    return data


def rasterize_hand(canvas, vertices, pixels, faces, vertex_parts):
    """Rasterize nearest hand triangle, perspective-correct depth, face normal and skinning part."""
    h, w = canvas.canvas_hw
    rx0, ry0, rx1, ry1 = canvas.valid_rect
    depth = np.full((h, w), np.inf, dtype=np.float32)
    normals = np.zeros((3, h, w), dtype=np.float32)
    parts = np.zeros((h, w), dtype=np.int8)
    for face in faces:
        positions = vertices[face]
        z = positions[:, 2]
        if not np.isfinite(positions).all() or (z <= 0).any():
            continue
        xy = pixels[face]
        x0, y0 = np.maximum(np.ceil(xy.min(axis=0)), (rx0, ry0)).astype(int)
        x1, y1 = np.minimum(np.floor(xy.max(axis=0)), (rx1 - 1, ry1 - 1)).astype(int)
        if x1 < x0 or y1 < y0:
            continue
        edge_1, edge_2 = xy[1:] - xy[0]
        area = edge_1[0] * edge_2[1] - edge_1[1] * edge_2[0]
        if abs(area) < 1e-8:
            continue
        y, x = np.ogrid[y0:y1 + 1, x0:x1 + 1]
        dx, dy = x - xy[0, 0], y - xy[0, 1]
        b1 = (dx * edge_2[1] - dy * edge_2[0]) / area
        b2 = (edge_1[0] * dy - edge_1[1] * dx) / area
        barycentric = np.stack((1 - b1 - b2, b1, b2))
        inside = (barycentric >= -1e-7).all(axis=0)
        corrected = barycentric / z[:, None, None]
        reciprocal_depth = corrected.sum(axis=0)
        tile_depth = np.divide(1, reciprocal_depth, out=np.full_like(reciprocal_depth, np.inf),
                               where=inside & (reciprocal_depth > 0))
        tile = np.s_[y0:y1 + 1, x0:x1 + 1]
        nearest = tile_depth < depth[tile]
        if not nearest.any():
            continue
        normal = np.cross(positions[1] - positions[0], positions[2] - positions[0])
        norm = np.linalg.norm(normal)
        if norm <= 1e-12:
            continue
        normal /= norm
        depth[tile][nearest] = tile_depth[nearest]
        normals[:, tile[0], tile[1]][:, nearest] = normal[:, None]
        parts[tile][nearest] = vertex_parts[face][corrected[:, nearest].argmax(axis=0)]
    return depth, normals, parts


def _anchors(template, vertex_parts):
    """Four same-index farthest-spaced MANO vertices per canonical skinning part."""
    anchors = []
    for part in range(16):
        candidates = np.flatnonzero(vertex_parts == part)
        if len(candidates) < 4:
            raise ValueError(f"MANO template part {part} has fewer than four vertices")
        xyz = template[candidates]
        first = np.argmin(((xyz - xyz.mean(0)) ** 2).sum(1))
        chosen = [first]
        for _ in range(3):
            distance = ((xyz[:, None] - xyz[chosen]) ** 2).sum(-1).min(1)
            distance[chosen] = -1
            chosen.append(distance.argmax())
        anchors.extend(candidates[chosen])
    return np.asarray(anchors)[:, None]


def _rotation_6d(matrix):
    return torch.as_tensor(np.asarray(matrix[..., :2, :], dtype=np.float32)).flatten(-2)


def _outward_faces(template, faces):
    triangles = template[faces] - template.mean(axis=0)
    volume_sign = np.einsum("ij,ij->", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2]))
    return faces if volume_sign >= 0 else faces[:, [0, 2, 1]]


def _vertex_normals(vertices, faces):
    face_xyz = vertices[faces]
    face_normals = np.cross(face_xyz[:, 1] - face_xyz[:, 0], face_xyz[:, 2] - face_xyz[:, 0])
    normals = np.zeros_like(vertices)
    for column in faces.T:
        np.add.at(normals, column, face_normals)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    return np.divide(normals, norm, out=np.zeros_like(normals), where=norm > 1e-12)


def _surface_candidates(positions, normals, mean_fit_mm, *, max_fit_mm=15.):
    """Bidirectional nearest anchors are soft geometry with unknown contact labels."""
    left, right = positions
    normal_l, normal_r = normals
    offset = right[None, :, :] - left[:, None, :]
    distance = np.linalg.norm(offset, axis=-1)
    valid_l = np.linalg.norm(normal_l, axis=1) > .5
    valid_r = np.linalg.norm(normal_r, axis=1) > .5
    distance[~(valid_l[:, None] & valid_r[None, :])] = np.inf
    forward = np.stack((np.arange(len(left)), distance.argmin(axis=1)), axis=1)
    reverse = np.stack((distance.argmin(axis=0), np.arange(len(right))), axis=1)
    candidates = np.unique(np.concatenate((forward, reverse)), axis=0)
    indexes = candidates[distance[tuple(candidates.T)] <= SURFACE_MAX_M]
    if len(indexes) == 0:
        return {}
    a, b = indexes.T
    opposing = np.clip(np.einsum("ij,ij->i", normal_l[a], normal_r[b]), -1, 1)
    gap_l = (offset[a, b] * normal_l[a]).sum(-1)
    gap_r = (-offset[a, b] * normal_r[b]).sum(-1)
    overlap = np.maximum(0, -np.minimum(gap_l, gap_r) - SURFACE_GAP_TOLERANCE_M)
    penetration_quality = np.exp(-.5 * (overlap / SURFACE_GAP_SCALE_M) ** 2)
    fit_quality = np.exp(-.5 * (max(mean_fit_mm) / max_fit_mm) ** 2)
    factors = np.column_stack((np.full(len(indexes), fit_quality), np.maximum(0, -opposing),
                               np.full(len(indexes), SURFACE_SINGLE_VIEW_PRIOR), penetration_quality))
    features = np.column_stack((distance[a, b], opposing, np.full(len(indexes), UNKNOWN), np.ones(len(indexes))))
    return {
        "surface_indices": torch.from_numpy(indexes.astype(np.int64)),
        "surface_features": torch.from_numpy(features.astype(np.float32)),
        "surface_quality": torch.from_numpy(factors.astype(np.float32)),
    }


class ManoGeometry:
    def __init__(self, model_dir, *, joint_regressor=None, max_fit_mm=15.0, occlusion_gap_mm=10.0):
        # Optional dependency: the projection-only converter uses no SMPL-X package.
        import smplx
        from smplx.lbs import batch_rodrigues
        from smplx.utils import Struct

        model_paths = {side: model_dir / f"MANO_{side.upper()}.pkl" for side in HANDS}
        self.layers = {side: smplx.MANO(str(path), data_struct=Struct(**_load_mano_model(path)),
                                        is_rhand=side == "right", use_pca=False).eval()
                       for side, path in model_paths.items()}
        left, right = (self.layers[side] for side in HANDS)
        # Official InterHand rendering fixes the mirrored MANO shape-direction bug.
        self.left_shape_fix = bool((left.shapedirs[:, 0] - right.shapedirs[:, 0]).abs().sum() < 1)
        if self.left_shape_fix:
            left.shapedirs[:, 0] *= -1
        self.batch_rodrigues = batch_rodrigues
        self.max_fit_m = max_fit_mm / 1000
        self.occlusion_gap_m = occlusion_gap_mm / 1000
        self.faces = {side: _outward_faces(layer.v_template.detach().numpy(), np.asarray(layer.faces, dtype=np.int64))
                      for side, layer in self.layers.items()}
        self.parts = {side: MANO_TO_PART[layer.lbs_weights.detach().numpy().argmax(1)]
                      for side, layer in self.layers.items()}
        self.anchors = _anchors(right.v_template.detach().numpy(), self.parts["right"])
        triangles = np.concatenate([faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
                                    for faces in self.faces.values()])
        self.edges = np.unique(np.sort(triangles, axis=-1), axis=0).T
        self.regressor = None if joint_regressor is None else np.load(joint_regressor, allow_pickle=False)
        if (self.regressor is not None and
                (self.regressor.shape != (21, 778) or not np.isfinite(self.regressor).all()
                 or not np.allclose(self.regressor.sum(1), 1, atol=.05))):
            raise ValueError("Expected the official 21x778 InterHand MANO joint regressor")
        self.provenance = {
            "models": {side: sha256_file(path) for side, path in model_paths.items()},
            "joint_regressor": sha256_file(joint_regressor) if joint_regressor is not None else None,
            "fit_reference": "official_21_joint_regressor" if joint_regressor is not None else "smplx_16_articulated_joints",
            "max_fit_mm": max_fit_mm,
            "occlusion_gap_mm": occlusion_gap_mm,
            "left_shape_fix": self.left_shape_fix,
            "loader": "restricted_literal_mano",
            "surface_candidates": {
                "selection": "unique_bidirectional_nearest_anchor",
                "max_edges": 128,
                "max_distance_mm": SURFACE_MAX_M * 1000,
                "cross_view": "single_view_conservative_prior",
                "cross_view_prior": SURFACE_SINGLE_VIEW_PRIOR,
                "signed_gap_tolerance_mm": SURFACE_GAP_TOLERANCE_M * 1000,
                "signed_gap_penalty_scale_mm": SURFACE_GAP_SCALE_M * 1000,
                "contact_label": "unknown",
            },
        }

    def _fit(self, side, slot, params, condition, camera):
        pose = torch.as_tensor(params["pose"], dtype=torch.float32).reshape(1, 48)
        shape = torch.as_tensor(params["shape"], dtype=torch.float32).reshape(1, 10)
        trans = torch.as_tensor(params["trans"], dtype=torch.float32).reshape(1, 3)
        if not all(torch.isfinite(value).all() for value in (pose, shape, trans)):
            return None, {"status": "nonfinite_parameters"}
        layer = self.layers[side]
        output = layer(global_orient=pose[:, :3], hand_pose=pose[:, 3:], betas=shape, transl=trans)
        world = output.vertices[0].numpy()
        vertices = camera.world_to_camera(world)
        if self.regressor is not None:
            start = 21 if side == "left" else 0
            fitted = (self.regressor @ vertices)[JOINT_ORDER[slot] - start]
            indexes = np.arange(21)
        else:
            fitted = camera.world_to_camera(output.joints[0, MANO_JOINTS].numpy())
            indexes = ARTICULATED
        valid = condition.joint_3d_valid[slot, indexes].numpy()
        if valid.sum() < 8:
            return None, {"status": "insufficient_calibration_joints", "compared_joints": int(valid.sum())}
        reference = condition.joints_3d[slot, indexes].numpy()[valid]
        errors = np.linalg.norm(fitted[valid] - reference, axis=1)
        if not np.isfinite(errors).all() or not np.isfinite(vertices).all():
            return None, {"status": "nonfinite_reconstruction"}
        mean, max_error = errors.mean(), errors.max()
        record = {"mean_mm": float(mean * 1000), "max_mm": float(max_error * 1000),
                  "compared_joints": int(valid.sum())}
        if mean > self.max_fit_m or max_error > 2 * self.max_fit_m:
            return None, {**record, "status": "fit_above_threshold"}
        pixels, positive = camera.project(vertices, condition.canvas)
        if not positive.all():
            return None, {**record, "status": "mesh_behind_camera"}
        effective_pose = pose + layer.pose_mean
        rotation = self.batch_rodrigues(effective_pose.view(16, 3)).numpy()
        rotation[0] = camera.rotation @ rotation[0]
        return (vertices, pixels, shape[0], rotation), {**record, "status": "verified"}

    @torch.no_grad()
    def enrich(self, condition, frame, camera, params):
        h, w = condition.canvas.canvas_hw
        fits = {}
        mesh_valid = torch.zeros(4, dtype=torch.bool)
        shape = torch.zeros(4, 10)
        wrist_rotation = torch.zeros(4, 6)
        joint_rotation = torch.zeros(4, 21, 6)
        rotation_valid = torch.zeros(4, 21, dtype=torch.bool)
        palm = torch.zeros(4)
        camera_ids = [None] * 4
        vertices = torch.zeros(4, 778, 3)
        anchor_xy = torch.zeros(4, 64, 2)
        anchor_camera = []
        anchor_normal = []
        depth = np.full((4, h, w), np.inf, dtype=np.float32)
        normals = np.zeros((4, 3, h, w), dtype=np.float32)
        parts = np.zeros((4, h, w), dtype=np.int8)
        for slot, side in enumerate(HANDS):
            if not condition.hand_present[slot]:
                continue
            if not params.get(side):
                fits[side] = {"status": "missing_fit"}
                continue
            fit, fits[side] = self._fit(side, slot, params[side], condition, camera)
            if fit is None:
                continue
            xyz, xy, coefficients, rotation = fit
            shape[slot] = coefficients
            wrist = condition.joints_3d[slot, 0]
            anchors = xy[self.anchors[:, 0]]
            vertices[slot] = torch.from_numpy(xyz) - wrist
            anchor_xy[slot] = torch.from_numpy(anchors)
            anchor_camera.append(xyz[self.anchors[:, 0]])
            anchor_normal.append(_vertex_normals(xyz, self.faces[side])[self.anchors[:, 0]])
            mesh_valid[slot] = True
            camera_ids[slot] = f"InterHand2.6M:{frame.split}:{frame.capture}:{frame.camera}"
            wrist_rotation[slot] = _rotation_6d(rotation[0])
            joint_rotation[slot, ARTICULATED] = _rotation_6d(rotation[MANO_JOINTS])
            rotation_valid[slot, ARTICULATED] = condition.joint_3d_valid[slot, ARTICULATED]
            if condition.joint_3d_valid[slot, 9]:
                palm[slot] = torch.linalg.vector_norm(condition.joints_3d[slot, 9] - wrist)
            # The fitted template is a fallback only when the labeled MCP is unavailable.
            if palm[slot] <= 0:
                palm[slot] = (vertices[slot, self.anchors[28, 0]]).norm()
            depth[slot], normals[slot], parts[slot] = rasterize_hand(
                condition.canvas, xyz, xy, self.faces[side], self.parts[side],
            )
        mano_fields = dict(mano_shape=shape, mano_valid=mesh_valid, mano_wrist_rotation_6d=wrist_rotation,
                           mano_palm_scale=palm, joint_rotation_6d=joint_rotation,
                           joint_rotation_valid=rotation_valid)
        if not mesh_valid.any():
            return replace(condition, **mano_fields).validate(), fits
        mesh = HandMesh(vertices, torch.from_numpy(self.edges), torch.from_numpy(self.anchors), anchor_xy,
                        mesh_valid, tuple(camera_ids))
        nearest = depth.min(axis=0)
        owned = np.isfinite(depth) & (depth == nearest) & (depth.argmin(axis=0)[None] == np.arange(4)[:, None, None])
        maps = np.concatenate((np.where(owned, depth, 0)[:, None], normals,
                               parts[:, None].astype(np.float32) / 15), axis=1)
        maps *= owned[:, None]
        map_valid = np.broadcast_to(owned[:, None], maps.shape).copy()
        occluded = _joint_occlusion(condition, depth, self.occlusion_gap_m) if mesh_valid[:2].all() else condition.joint_occluded
        if mesh_valid[:2].all():
            order = _part_occlusion(depth, parts, self.occlusion_gap_m)
            surface = _surface_candidates(anchor_camera, anchor_normal,
                                          [fits[side]["mean_mm"] for side in HANDS],
                                          max_fit_mm=self.max_fit_m * 1000)
            hand_pairs = (HandPair((0, 1), torch.full((16, 16), UNKNOWN, dtype=torch.int8),
                                   occlusion=torch.from_numpy(order), **surface),)
        else:
            hand_pairs = ()
        return replace(condition, mesh=mesh, geometry_maps=torch.from_numpy(maps),
                       map_valid=torch.from_numpy(map_valid), joint_occluded=occluded,
                       pairs=hand_pairs, **mano_fields).validate(), fits


def _joint_occlusion(condition, depth, gap):
    """Set only mesh-supported cross-hand occlusion, leaving all other visibility unknown."""
    result = torch.zeros_like(condition.visible)
    x0, y0, x1, y1 = condition.canvas.valid_rect
    for slot in (0, 1):
        xy = condition.joints[slot, :, :2].round().long().numpy()
        x, y = xy.T
        in_view = (x >= x0) & (x < x1) & (y >= y0) & (y < y1) & condition.projected_joints()[slot].numpy()
        keypoints = np.flatnonzero(in_view)
        own = depth[slot, y[keypoints], x[keypoints]]
        other = depth[1 - slot, y[keypoints], x[keypoints]]
        result[slot, keypoints] = torch.from_numpy(np.isfinite(own) & (other + gap < own))
    return result


def _part_occlusion(depth, parts, gap):
    left, right = depth[:2]
    overlap = np.isfinite(left) & np.isfinite(right)
    front_left = overlap & (left + gap < right)
    front_right = overlap & (right + gap < left)
    relation = np.zeros((16, 16), dtype=np.int8)
    left_counts = np.zeros((16, 16), dtype=np.int64)
    right_counts = np.zeros_like(left_counts)
    for counts, mask in ((left_counts, front_left), (right_counts, front_right)):
        np.add.at(counts, (parts[0][mask], parts[1][mask]), 1)
    relation[(left_counts >= 4) & (right_counts == 0)] = 1
    relation[(right_counts >= 4) & (left_counts == 0)] = -1
    return relation
