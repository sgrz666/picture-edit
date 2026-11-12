"""Optional source-repair sidecars are raw conditions, never supervision masks."""
from dataclasses import replace
import json

import pytest
import torch

from tests.test_native_v6_data import _manifest
from src.pose_control.qwen21_full.native_full_assets import write_record


def repair_for(sample, bank):
    meta = sample['local_condition_v4'].metadata[0]
    h, w = sample['target_canvas']['canvas_hw']
    return {'schema': 'native_pose_repair_v61', 'role': 'source_repair',
        'mask': torch.ones(1, 2, h, w), 'metadata': ({
            'target_canvas_id': meta['target_canvas_id'],
            'source_rgb_sha256s': [source.rgb_sha256 for source in bank.sources],
            'target_person_ids': meta['target_person_ids'],
            'driver_record_sha256': meta['driver_record_sha256']},)}


def test_v61_reader_and_collator_keep_optional_typed_repair(tmp_path):
    from src.pose_control.qwen21_full.native_v61.data import NativeFullDatasetV61, collate_native_full_v61
    from src.pose_control.qwen21_full.native_v6.data import NativeFullDatasetV6
    manifest, _, _ = _manifest(tmp_path)
    old = NativeFullDatasetV6(manifest)[0]
    repair = repair_for(old, old['source_bank_v5'])
    row = json.loads(manifest.read_text())
    row['pose_repair_v61'] = write_record(tmp_path/'repair.npz', repair, schema='native_pose_repair_v61')
    manifest.write_text(json.dumps(row)+'\n')
    sample = NativeFullDatasetV61(manifest)[0]
    assert torch.equal(sample['pose_repair_v61']['mask'], repair['mask'])
    assert 'training_supervision_v5' not in sample and 'tgt_image' not in sample
    batch = collate_native_full_v61([sample])
    assert batch['pose_repair_v61']['mask'].shape == (1, 2, 64, 64)
    assert batch['pose_repair_v61']['metadata'] == repair['metadata']
    assert batch['source_bank_v5'] is sample['source_bank_v5']


def test_v61_reader_does_not_require_new_sidecars(tmp_path):
    from src.pose_control.qwen21_full.native_v61.data import NativeFullDatasetV61, collate_native_full_v61
    manifest, _, _ = _manifest(tmp_path)
    sample = NativeFullDatasetV61(manifest)[0]
    assert 'pose_repair_v61' not in sample
    assert 'pose_repair_v61' not in collate_native_full_v61([sample])


def test_wrong_repair_provenance_is_rejected_before_runtime(tmp_path):
    from src.pose_control.qwen21_full.native_v61.data import NativeFullDatasetV61
    from src.pose_control.qwen21_full.native_v6.data import NativeFullDatasetV6
    manifest, _, _ = _manifest(tmp_path)
    sample = NativeFullDatasetV6(manifest)[0]
    repair = repair_for(sample, sample['source_bank_v5'])
    repair['metadata'][0]['source_rgb_sha256s'].reverse()
    row = json.loads(manifest.read_text())
    row['pose_repair_v61'] = write_record(tmp_path/'repair.npz', repair, schema='native_pose_repair_v61')
    manifest.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError): NativeFullDatasetV61(manifest)[0]
