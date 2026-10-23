"""SMPL-X body joint contract. Rotation6D stores COLUMN 0 then COLUMN 1.

Joint IDs are canonical SMPL-X IDs, never Body25 or SMPLest-X 137 landmark IDs.
No estimator, FK model or learned tokens are cached by this module.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch.nn import functional as F

POSE_SCHEMA = 'smplx_body21_camera_v2'
ROTATION_CONVENTION = 'matrix_columns_0_then_1'
JOINT_NAMES = ('left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2',
               'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck',
               'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
               'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist')
PARENTS_22 = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
# Anatomical name mapping only; spine/collar have no Body25 point. "head" is
# not the Body25 nose. left/right_foot are not interchangeable with toes/heel.
BODY25_BY_NAME = {'left_hip': 12, 'right_hip': 9, 'left_knee': 13, 'right_knee': 10,
                  'left_ankle': 14, 'right_ankle': 11, 'neck': 1,
                  'left_shoulder': 5, 'right_shoulder': 2, 'left_elbow': 6,
                  'right_elbow': 3, 'left_wrist': 7, 'right_wrist': 4}
POSE_INPUT_FIELDS = ('body_pose', 'joints_cam', 'joint_uv', 'fk_valid', 'projection_valid',
                     'body_height', 'smpl_bbox_height')


@dataclass(frozen=True)
class BodyPoseConfig:
    rotation: str = 'rotation6d_columns'
    spatial_bias: bool = True
    token_dim: int = 256
    layers: int = 2
    heads: int = 8
    ffn_ratio: int = 4
    dropout: float = 0.
    min_sigma_px: float = 16.
    bbox_sigma_ratio: float = .15

    def __post_init__(self):
        if self.rotation not in ('rotation6d_columns', 'axis_angle'):
            raise ValueError('Unknown body rotation representation')
        if (self.token_dim, self.layers, self.heads, self.ffn_ratio, self.dropout) != (256, 2, 8, 4, 0.):
            raise ValueError('Body21 contract requires 256 width, 2 layers, 8 heads, FFN4, dropout0')
        if any(not math.isfinite(x) or x <= 0 for x in (self.min_sigma_px, self.bbox_sigma_ratio)):
            raise ValueError('Spatial sigma settings must be finite positive')

    def to_dict(self):
        return asdict(self)


def axis_angle_to_matrix(axis_angle):
    if axis_angle.shape[-1] != 3 or not axis_angle.is_floating_point() or not torch.isfinite(axis_angle).all():
        raise ValueError('axis_angle must be finite floating [...,3]')
    x, y, z = axis_angle.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*x.shape, 3, 3)
    theta = torch.linalg.vector_norm(axis_angle, dim=-1)[..., None, None]
    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    return eye + torch.sinc(theta / math.pi) * skew + .5 * torch.sinc(theta / (2 * math.pi)).square() * (skew @ skew)


def matrix_to_rotation_6d(matrix):
    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)


def rotation_6d_to_matrix(rotation):
    if rotation.shape[-1] != 6:
        raise ValueError('rotation6D must have six columns components')
    a, b = rotation[..., :3], rotation[..., 3:]
    u = F.normalize(a, dim=-1)
    v = F.normalize(b - (u * b).sum(-1, keepdim=True) * u, dim=-1)
    return torch.stack((u, v, torch.cross(u, v, dim=-1)), dim=-1)


def joint_tree_distances():
    # Floyd-Warshall on all 22 nodes BEFORE excluding pelvis.
    d = torch.full((22, 22), 100., dtype=torch.float32)
    d.fill_diagonal_(0)
    for i, parent in enumerate(PARENTS_22[1:], 1):
        d[i, parent] = d[parent, i] = 1
    for k in range(22):
        d = torch.minimum(d, d[:, k, None] + d[None, k, :])
    return d[1:, 1:]


def pose_kwargs_from_batch(batch, *, device, dtype=torch.float32, person='a', suffix=''):
    missing = [key + suffix for key in POSE_INPUT_FIELDS if key + suffix not in batch]
    if missing:
        raise ValueError(f'Body Pose v2 requires canonical cache fields: {missing}')
    return {f'{key}_{person}': batch[key + suffix].to(device=device,
             dtype=torch.bool if key in ('fk_valid', 'projection_valid') else dtype)
            for key in POSE_INPUT_FIELDS}
