"""Canonical dual_v8 data contracts shared by caches and checkpoints."""
from __future__ import annotations

import hashlib
import json


GEOMETRY_SCHEMA_VERSION = 'smplx_dual_v1'
SOURCE_SCHEMA_VERSION = 'source_v1'
GEOMETRY_ENCODER_VERSION = 'dual_v8_geometry_v1'
APPEARANCE_ENCODER_VERSION = 'dual_v8_vae_dino_v1'
SOURCE_CROP_VERSION = 'exclusive_square_letterbox512_margin1p2_gray128_v1'
SOURCE_ENCODER_VERSION = 'qwen21_rgba_vae_dinov2_g_whole224_v1'
GEOMETRY_COORDINATE_CONVENTION = (
    'Hi4D world metres Y-up; X_cam=R_cw@X_world+t_cw; '
    'normal=camera; depth=camera_z; part_centres/joints/orientations=world')
ROTATION6D_CONVENTION = 'columns_concatenated_R[:,0]_then_R[:,1]'
SOURCE_SHAPE_SEMANTICS = 'global32[:10]=source_actor_beta10_fixed_across_target_pose'

SMPLX14_PART_NAMES = ('head', 'torso', 'left_upper_arm', 'left_forearm', 'left_hand',
    'right_upper_arm', 'right_forearm', 'right_hand', 'left_thigh', 'left_shin',
    'left_foot', 'right_thigh', 'right_shin', 'right_foot')
SMPLX_JOINT_TO_PART = (1,8,11,1,9,12,1,10,13,1,10,13,0,1,1,0,2,5,3,6,4,7,
    0,0,0) + (4,) * 15 + (7,) * 15
# Canonical hash covers semantic order, topology, and the dominant LBS rule.
# The concrete model's vertex-label hash is recorded separately in provenance.
SMPLX14_PARTITION_SPEC = dict(topology='smplx_10475_vertices_55_skinning_joints',
    rule='joint_to_part[argmax(lbs_weights,axis=-1)]',
    part_names=list(SMPLX14_PART_NAMES), joint_to_part=list(SMPLX_JOINT_TO_PART))
SMPLX14_PARTITION_SHA256 = hashlib.sha256(json.dumps(SMPLX14_PARTITION_SPEC,
    sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()

SMPLX55_JOINT_NAMES = ('pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee',
    'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot',
    'right_foot', 'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'jaw', 'left_eye_smplhf', 'right_eye_smplhf', 'left_index1', 'left_index2',
    'left_index3', 'left_middle1', 'left_middle2', 'left_middle3', 'left_pinky1',
    'left_pinky2', 'left_pinky3', 'left_ring1', 'left_ring2', 'left_ring3',
    'left_thumb1', 'left_thumb2', 'left_thumb3', 'right_index1', 'right_index2',
    'right_index3', 'right_middle1', 'right_middle2', 'right_middle3', 'right_pinky1',
    'right_pinky2', 'right_pinky3', 'right_ring1', 'right_ring2', 'right_ring3',
    'right_thumb1', 'right_thumb2', 'right_thumb3')
ANATOMICAL_PART_JOINT_INDICES = (15,9,16,18,20,17,19,21,1,4,7,2,5,8)


def generation_versions():
    return dict(source=SOURCE_SCHEMA_VERSION, geometry=GEOMETRY_SCHEMA_VERSION,
        crop=SOURCE_CROP_VERSION, encoder=SOURCE_ENCODER_VERSION,
        coordinate_convention=GEOMETRY_COORDINATE_CONVENTION,
        rotation6d_convention=ROTATION6D_CONVENTION,
        partition_sha256=SMPLX14_PARTITION_SHA256)


def geometry_schema_metadata():
    return dict(schema_version=GEOMETRY_SCHEMA_VERSION,
        coordinate_convention=GEOMETRY_COORDINATE_CONVENTION,
        rotation6d_convention=ROTATION6D_CONVENTION,
        partition_sha256=SMPLX14_PARTITION_SHA256,
        source_shape_semantics=SOURCE_SHAPE_SEMANTICS)
