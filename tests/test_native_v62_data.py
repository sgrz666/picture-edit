"""V6.2 requires no new masks, and keeps the explicit V6.1 repair bindings."""
import json
import pytest

from tests.test_native_v6_data import _manifest
from tests.test_native_v61_data import repair_for
from src.pose_control.qwen21_full.native_full_assets import write_record


def test_existing_five_ref_input_stays_valid_without_repair(tmp_path):
    from src.pose_control.qwen21_full.native_v62.data import NativeFullDatasetV62,collate_native_full_v62
    manifest,_,_=_manifest(tmp_path)
    sample=NativeFullDatasetV62(manifest)[0]
    batch=collate_native_full_v62([sample])
    assert 'pose_repair_v61' not in batch
    assert batch['source_bank_v5'] is sample['source_bank_v5']
    assert 'tgt_image' not in sample and 'training_supervision_v5' not in sample


def test_optional_repair_keeps_source_sha_validation(tmp_path):
    from src.pose_control.qwen21_full.native_v62.data import NativeFullDatasetV62,collate_native_full_v62
    manifest,_,_=_manifest(tmp_path)
    sample=NativeFullDatasetV62(manifest)[0]
    repair=repair_for(sample,sample['source_bank_v5'])
    row=json.loads(manifest.read_text());row['pose_repair_v61']=write_record(tmp_path/'repair.npz',repair,schema='native_pose_repair_v61')
    manifest.write_text(json.dumps(row)+'\n')
    batch=collate_native_full_v62([NativeFullDatasetV62(manifest)[0]])
    assert batch['pose_repair_v61']['metadata']==repair['metadata']
    repair['metadata'][0]['source_rgb_sha256s'].reverse()
    row['pose_repair_v61']=write_record(tmp_path/'wrong_repair.npz',repair,schema='native_pose_repair_v61')
    manifest.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError):NativeFullDatasetV62(manifest)[0]
