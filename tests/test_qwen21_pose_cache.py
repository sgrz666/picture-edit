import importlib
import json

import pytest
import torch


def api():
    assert importlib.util.find_spec('src.pose_control.qwen21_full.pose_cache') is not None, 'versioned FK cache reader missing'
    return importlib.import_module('src.pose_control.qwen21_full.pose_cache')


def test_canonical_global_raw_camera_translation_both_slots():
    m = api()
    params = {'body_pose': torch.zeros(1, 63), 'betas': torch.ones(1, 10),
              'global_orient': torch.zeros(1, 3), 'transl': torch.tensor([[1., 2., 3.]]),
              'transl_raw': torch.tensor([[4., 5., 6.]]), 'focal_length_xy': torch.ones(1, 2),
              'princpt': torch.zeros(1, 2), 'focal_length_xy_raw': torch.tensor([[100., 200.]]),
              'princpt_raw': torch.tensor([[10., 20.]])}
    value, camera = m.canonical_global(params, 0, (200, 100), dataset='iper')
    torch.testing.assert_close(value[16:19], torch.tensor([4., 5., 6.]))
    torch.testing.assert_close(value[23:26], value[16:19])
    torch.testing.assert_close(value[19:23], torch.tensor([1., 1., .1, .1]))
    assert camera['selection'] == 'raw'
    del params['transl_raw']
    with pytest.raises(ValueError, match='raw'):
        m.canonical_global(params, 0, (200, 100), dataset='iper')


def test_reader_refuses_legacy_schema_and_missing_frame(tmp_path):
    m = api()
    path = tmp_path / 'old.pt'
    torch.save({'smplx_global': torch.zeros(8, 26)}, path)
    manifest = tmp_path / 'records.jsonl'
    manifest.write_text(json.dumps({'record_id': 'tiktok:00001:target:0001', 'path': str(path),
                                    'sha256': m.sha256(path)}) + '\n')
    reader = m.BodyPoseRecords(manifest)
    with pytest.raises(ValueError, match='schema'):
        reader.read('tiktok:00001:target:0001', {'original_hw': [32, 32], 'resized_hw': [32, 32],
                                             'canvas_hw': [32, 32]})
    with pytest.raises(ValueError, match='missing'):
        reader.read('unknown', {})


def test_projection_preserves_out_of_frame_and_separates_validity():
    m = api()
    joints = torch.tensor([[0., 0., 1.], [2., 0., 1.], [0., 0., -1.], [0., 0., 0.]])
    uv, valid = m.project_camera(joints, torch.tensor([100., 100.]), torch.tensor([16., 16.]), (32, 32))
    assert uv[1, 0] == 216 and valid.tolist() == [True, False, False, False]
    assert torch.isfinite(uv).all()


def test_reader_applies_canvas_once_and_rejects_source_drift(tmp_path):
    m = api()
    from src.pose_control.qwen21_full.body_pose import JOINT_NAMES, PARENTS_22, ROTATION_CONVENTION, POSE_SCHEMA
    original = tmp_path / 'source.txt'
    original.write_text('source')
    meta = {'schema': POSE_SCHEMA, 'record_id': 'tiktok:1:target:1', 'joint_names': list(JOINT_NAMES),
            'parents_22': list(PARENTS_22), 'rotation_convention': ROTATION_CONVENTION,
            'projection_space': 'original_camera_pixels', 'mesh_max_abs_error': 1e-6,
            'sources': {key: {'path': str(original), 'sha256': m.sha256(original)} for key in
                        ('params','frame_map','rgb','legacy_conditions','normal','smpl_mask','part','depth_vis','model','fk_implementation')},
            'original_hw': [64, 32], 'smpl_mask_bbox_original_xyxy': [0, 0, 32, 64], 'camera': {'selection': 'standard'},
            'condition_to_original': {'normal': {'source_hw': [32, 16], 'scale_xy': [2, 2]}}}
    data = {'metadata': meta, 'body_pose': torch.zeros(63), 'joints_cam': torch.ones(22, 3),
            'joint_uv': torch.ones(21, 2) * 16, 'fk_valid': torch.ones(22, dtype=torch.bool),
            'projection_valid': torch.ones(21, dtype=torch.bool), 'body_height': torch.tensor(1.7),
            'smplx_global': torch.zeros(26)}
    cache = tmp_path / 'pose.pt'
    torch.save(data, cache)
    manifest = tmp_path / 'records.jsonl'
    manifest.write_text(json.dumps({'record_id': meta['record_id'], 'path': str(cache), 'sha256': m.sha256(cache)}) + '\n')
    reader = m.BodyPoseRecords(manifest)
    out = reader.read(meta['record_id'], {'original_hw':[64,32], 'resized_hw':[32,16], 'canvas_hw':[32,32], 'offset_xy':[8,0]})
    torch.testing.assert_close(out['joint_uv'][0], torch.tensor([16.,8.]))
    assert out['smpl_bbox_height'] == 32
    original.write_text('changed')
    with pytest.raises(ValueError, match='fingerprint'):
        reader.read(meta['record_id'], {'original_hw':[64,32], 'resized_hw':[32,16], 'canvas_hw':[32,32]})


def test_pose_dataset_is_available_as_v2_canonical_entry():
    m = api()
    module = importlib.import_module('src.pose_control.qwen21_full.pose_data_runtime')
    assert hasattr(module, 'NativePoseDataset')
