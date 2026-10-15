import torch

from scripts.prepare_tiktok_qwen21_face import assemble_cache
from scripts.precompute_iper_face_features_v65 import validate_v65_face_cache
from src.pose_control.qwen21_full.data_runtime import FaceConditionStore
from src.pose_control.qwen21_full.face_supervision import FaceSupervisionStore, SCHEMA
import json
import pytest


def test_target_only_frames_keep_zero_identity_and_texture():
    stems = ["source", "target"]
    geometry = []
    for _ in stems:
        geometry.append(dict(face_landmarks=torch.zeros(72, 3), jaw_pose=torch.zeros(3),
                             expression=torch.zeros(10), face_boxes=torch.tensor([.2, .2, .4, .4]),
                             geometry_valid=True))
    source_features = {"source": (torch.ones(512), torch.ones(256, 1536))}
    cache = assemble_cache(stems, geometry, source_features, {"schema_version": 1, "appearance": "00001"})
    validate_v65_face_cache(cache)
    assert cache["valid"].tolist() == [True, False]
    assert cache["geometry_valid"].tolist() == [True, True]
    assert cache["arcface"][1].count_nonzero() == 0
    assert cache["dino_patches"][1].count_nonzero() == 0
    assert cache["arcface"][0].count_nonzero() == 512


def test_full_face_store_accepts_raw_tiktok_provenance(tmp_path):
    stems = ['0001', '0002']
    geom = [dict(face_landmarks=torch.full((72, 3), .3), jaw_pose=torch.zeros(3),
                 expression=torch.zeros(10), face_boxes=torch.tensor([.2, .2, .4, .4]),
                 geometry_valid=True) for _ in stems]
    cache = assemble_cache(stems, geom, {'0001': (torch.ones(512), torch.ones(256, 1536))},
                           {'schema_version': 1, 'appearance': '00001'})
    torch.save(cache, tmp_path / '00001.pt')
    row = dict(sequence_id='00001', source_stem='0001', target_stem='0002',
               source_rgb='source.png', target_rgb='target.png')
    condition, features = FaceConditionStore(tmp_path).for_pairs([row], device='cpu', dtype=torch.float32)
    assert condition.face_valid[0, 0]
    assert features.arcface[0, 0, 0].count_nonzero() == 512


def test_target_auxiliary_gate_accepts_raw_tiktok_provenance(tmp_path):
    row = dict(sequence_id='00001', source_stem='0001', target_stem='0002',
               source_rgb='source.png', target_rgb='target.png')
    path = tmp_path / 'supervision.json'
    path.write_text(json.dumps(dict(schema=SCHEMA, records={
        '00001/0002': {'by_resolution': {'512': {'valid': True}}}})))
    store = FaceSupervisionStore(path)
    assert store.valid_mask([row], 512).tolist() == [True]


def test_full_face_store_rejects_modified_cache_content(tmp_path):
    geometry = [dict(face_landmarks=torch.zeros(72, 3), jaw_pose=torch.zeros(3),
                    expression=torch.zeros(10), face_boxes=torch.tensor([.2, .2, .4, .4]), geometry_valid=True)]
    cache = assemble_cache(['0001'], geometry, {'0001': (torch.ones(512), torch.ones(256, 1536))},
                           {'schema_version': 1, 'appearance': '00001'})
    cache['arcface'][0, 0] = 10
    torch.save(cache, tmp_path / '00001.pt')
    with pytest.raises(ValueError, match='fingerprint'):
        FaceConditionStore(tmp_path)._load('00001')
